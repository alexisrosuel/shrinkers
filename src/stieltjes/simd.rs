//! Hardware-accelerated helpers shared by the hot kernels.
//!
//! # Why this module exists
//!
//! AArch64 NEON has **no FP64 vector divide**: an `f64` division lowers to a
//! scalar `fdiv`, so 4-wide loops built around `1/(d²+η²)` serialize on the
//! divide unit (~13–20 cycles latency, poor pipelining). The M1 fix is the
//! classic Newton–Raphson refined reciprocal (`vrecpeq_f64` + three
//! `vrecpsq_f64` steps: 8 → 17 → 34 → ≥53 significant bits), which keeps
//! every lane on fully-pipelined multiply/add units.
//!
//! # Unsafe policy
//!
//! The `std::arch` AArch64 intrinsics are `unsafe` functions. All `unsafe`
//! blocks of the entire crate live HERE, behind thin safe `F64x2` / `F32x4`
//! abstractions whose contracts are enforced by construction:
//!
//! - every load reads lanes `i..i+width` of a live slice whose length the
//!   caller checks (`debug_assert` documents the invariant);
//! - FP64/FP32 NEON is architectural on AArch64 and the crate builds with
//!   `-C target-cpu=native` (`RUSTFLAGS`), so the feature set these
//!   intrinsics require is a compile-time constant on supported targets —
//!   no runtime detection can be missed, hence no UB from mis-dispatch.
//!
//! `F32x4` is the four-lane f32 companion used by the optional (`FastMode::F32`)
//! far-field path: f32 has 4 lanes on AArch64 and a cheap refined reciprocal,
//! so a term-by-term kernel that only needs ~1e-5 accuracy can run wider.
//!
//! On non-AArch64 targets `F64x2`/`F32x4` degrade to `[f64; 2]`/`[f32; 4]`
//! with identical semantics (true division instead of the refined
//! reciprocal), so hot kernels stay a single portable code path. Outside
//! this module the crate contains no `unsafe`.

/// Two-lane double vector used by hot kernels.
///
/// One shared implementation drives both backends: AArch64 NEON registers,
/// or a `[f64; 2]` pair elsewhere (LLVM auto-vectorizes or splits it).
#[derive(Copy, Clone, Debug)]
pub(crate) struct F64x2(F64x2Repr);

#[cfg(target_arch = "aarch64")]
type F64x2Repr = std::arch::aarch64::float64x2_t;
#[cfg(not(target_arch = "aarch64"))]
type F64x2Repr = [f64; 2];

impl F64x2 {
    /// Both lanes set to `x`.
    #[inline(always)]
    pub(crate) fn splat(x: f64) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self(unsafe { std::arch::aarch64::vdupq_n_f64(x) })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self([x, x])
        }
    }

    /// Both lanes zero.
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Lanes `[i, i+1]` of `s`.
    ///
    /// Contract: `i + 2 <= s.len()` (checked in debug builds; enforced by
    /// construction in the kernels' `j + 2 <= n` loop guards).
    #[inline(always)]
    pub(crate) fn load(s: &[f64], i: usize) -> Self {
        debug_assert!(i + 2 <= s.len());
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: `s` is a live borrow of length ≥ i+2; the read stays
            // within its 16 bytes (see module docs for the feature-set
            // argument).
            Self(unsafe { std::arch::aarch64::vld1q_f64(s.as_ptr().add(i)) })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self([s[i], s[i + 1]])
        }
    }

    /// Lanes `[a, b]`.
    #[inline(always)]
    pub(crate) fn from_array(v: [f64; 2]) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: pure register shuffle from a stack array; feature set
            // fixed at compile time (see module docs).
            Self(unsafe { std::arch::aarch64::vld1q_f64(v.as_ptr()) })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self(v)
        }
    }

    /// Value of lane `i` (must be 0 or 1).
    #[inline(always)]
    pub(crate) fn lane(self, i: usize) -> f64 {
        let a = [self.lane0(), self.lane1()];
        a[i]
    }

    #[inline(always)]
    fn lane0(self) -> f64 {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: register-to-register extract; feature fixed at compile time.
            unsafe { std::arch::aarch64::vgetq_lane_f64(self.0, 0) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.0[0]
        }
    }

    #[inline(always)]
    fn lane1(self) -> f64 {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: register-to-register extract; feature fixed at compile time.
            unsafe { std::arch::aarch64::vgetq_lane_f64(self.0, 1) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.0[1]
        }
    }

    /// Lane-wise `self − rhs`.
    #[inline(always)]
    pub(crate) fn sub(self, rhs: Self) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: pure register arithmetic, feature fixed at compile time.
            Self(unsafe { std::arch::aarch64::vsubq_f64(self.0, rhs.0) })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
        }
    }

    /// Lane-wise `self * rhs`.
    #[inline(always)]
    pub(crate) fn mul(self, rhs: Self) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self(unsafe { std::arch::aarch64::vmulq_f64(self.0, rhs.0) })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self([self.0[0] * rhs.0[0], self.0[1] * rhs.0[1]])
        }
    }

    /// Lane-wise `self + a*b` (fused on NEON via VFMA).
    #[inline(always)]
    pub(crate) fn fma(self, a: Self, b: Self) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self(unsafe { std::arch::aarch64::vfmaq_f64(self.0, a.0, b.0) })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self([self.0[0] + a.0[0] * b.0[0], self.0[1] + a.0[1] * b.0[1]])
        }
    }

    /// Lane-wise refined reciprocal `1/self`.
    ///
    /// NEON: initial FRECPE estimate plus three FRECPS refinement steps
    /// (8 → 17 → 34 → ≥53 significant bits) — accurate to ≤1 ulp of the
    /// correctly-rounded quotient while staying entirely on the multiply /
    /// add pipelines. Elsewhere: plain division.
    #[inline(always)]
    pub(crate) fn recip(self) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self(unsafe {
                use std::arch::aarch64::*;
                let mut e = vrecpeq_f64(self.0);
                e = vmulq_f64(e, vrecpsq_f64(self.0, e));
                e = vmulq_f64(e, vrecpsq_f64(self.0, e));
                e = vmulq_f64(e, vrecpsq_f64(self.0, e));
                e
            })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self([1.0 / self.0[0], 1.0 / self.0[1]])
        }
    }

    /// One-refinement-step reciprocal: `FRECPE` + a single `FRECPS`/mul
    /// Newton step, i.e. ~17 significant bits (~7.6e-6 relative).
    ///
    /// Measured +1.19-1.24x on the f64 ChebCode far field (error floor
    /// ~1.4e-6). Kept as a measured reference: the crate's speed presets now
    /// use the 4-lane [`F32x4`] far field instead, which is both faster
    /// (≈1.3x) and more accurate (≈6e-7).
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn recip_fast(self) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self(unsafe {
                use std::arch::aarch64::*;
                let e = vrecpeq_f64(self.0);
                vmulq_f64(e, vrecpsq_f64(self.0, e))
            })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self([1.0 / self.0[0], 1.0 / self.0[1]])
        }
    }

    /// Sum of both lanes.
    #[inline(always)]
    pub(crate) fn hsum(self) -> f64 {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { std::arch::aarch64::vaddvq_f64(self.0) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.0[0] + self.0[1]
        }
    }
}

impl std::ops::Sub for F64x2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        F64x2::sub(self, rhs)
    }
}

impl std::ops::Mul for F64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        F64x2::mul(self, rhs)
    }
}

/// Four-lane single vector used by the optional `f32` far-field path.
///
/// Same design as [`F64x2`]: NEON registers on AArch64, `[f32; 4]` elsewhere.
/// The far-field `Σ_j w_j/(z−t_j)` is already an approximation of the exact
/// leaf sums (see `chebcode`), so it can afford f32 arithmetic; four f32
/// lanes plus a fully pipelined f32 reciprocal beat two f64 lanes plus the
/// multi-step f64 Newton refinement on AArch64.
#[derive(Copy, Clone, Debug)]
pub(crate) struct F32x4(F32x4Repr);

#[cfg(target_arch = "aarch64")]
type F32x4Repr = std::arch::aarch64::float32x4_t;
#[cfg(not(target_arch = "aarch64"))]
type F32x4Repr = [f32; 4];

impl F32x4 {
    /// All four lanes set to `x`.
    #[inline(always)]
    pub(crate) fn splat(x: f32) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self(unsafe { std::arch::aarch64::vdupq_n_f32(x) })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self([x, x, x, x])
        }
    }

    /// All four lanes zero.
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Lanes `[i, i+1, i+2, i+3]` of `s`.
    ///
    /// Contract: `i + 4 <= s.len()` (checked in debug builds; enforced by
    /// construction in the kernels' `j + 4 <= n` loop guards).
    #[inline(always)]
    pub(crate) fn load(s: &[f32], i: usize) -> Self {
        debug_assert!(i + 4 <= s.len());
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: `s` is a live borrow of length ≥ i+4; the read stays
            // within its 16 bytes (see module docs for the feature-set
            // argument).
            Self(unsafe { std::arch::aarch64::vld1q_f32(s.as_ptr().add(i)) })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self([s[i], s[i + 1], s[i + 2], s[i + 3]])
        }
    }

    /// Lanes `[a, b, c, d]`.
    #[inline(always)]
    pub(crate) fn from_array(v: [f32; 4]) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: pure register load from a stack array; feature set
            // fixed at compile time (see module docs).
            Self(unsafe { std::arch::aarch64::vld1q_f32(v.as_ptr()) })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self(v)
        }
    }

    /// Value of lane `i` (must be 0..=3).
    #[inline(always)]
    #[allow(dead_code)] // parity API with F64x2; exercised by tests
    pub(crate) fn lane(self, i: usize) -> f32 {
        let a = [self.lane0(), self.lane1(), self.lane2(), self.lane3()];
        a[i]
    }

    #[inline(always)]
    #[allow(dead_code)] // parity API with F64x2; exercised by tests
    fn lane0(self) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: register-to-register extract; feature fixed at compile time.
            unsafe { std::arch::aarch64::vgetq_lane_f32(self.0, 0) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.0[0]
        }
    }

    #[inline(always)]
    #[allow(dead_code)] // parity API with F64x2; exercised by tests
    fn lane1(self) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: register-to-register extract; feature fixed at compile time.
            unsafe { std::arch::aarch64::vgetq_lane_f32(self.0, 1) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.0[1]
        }
    }

    #[inline(always)]
    #[allow(dead_code)] // parity API with F64x2; exercised by tests
    fn lane2(self) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: register-to-register extract; feature fixed at compile time.
            unsafe { std::arch::aarch64::vgetq_lane_f32(self.0, 2) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.0[2]
        }
    }

    #[inline(always)]
    #[allow(dead_code)] // parity API with F64x2; exercised by tests
    fn lane3(self) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: register-to-register extract; feature fixed at compile time.
            unsafe { std::arch::aarch64::vgetq_lane_f32(self.0, 3) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.0[3]
        }
    }

    /// Lane-wise `self − rhs`.
    #[inline(always)]
    pub(crate) fn sub(self, rhs: Self) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: pure register arithmetic, feature fixed at compile time.
            Self(unsafe { std::arch::aarch64::vsubq_f32(self.0, rhs.0) })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self([
                self.0[0] - rhs.0[0],
                self.0[1] - rhs.0[1],
                self.0[2] - rhs.0[2],
                self.0[3] - rhs.0[3],
            ])
        }
    }

    /// Lane-wise `self * rhs`.
    #[inline(always)]
    pub(crate) fn mul(self, rhs: Self) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self(unsafe { std::arch::aarch64::vmulq_f32(self.0, rhs.0) })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self([
                self.0[0] * rhs.0[0],
                self.0[1] * rhs.0[1],
                self.0[2] * rhs.0[2],
                self.0[3] * rhs.0[3],
            ])
        }
    }

    /// Lane-wise `self + a*b` (fused on NEON via VFMA).
    #[inline(always)]
    pub(crate) fn fma(self, a: Self, b: Self) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self(unsafe { std::arch::aarch64::vfmaq_f32(self.0, a.0, b.0) })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self([
                self.0[0] + a.0[0] * b.0[0],
                self.0[1] + a.0[1] * b.0[1],
                self.0[2] + a.0[2] * b.0[2],
                self.0[3] + a.0[3] * b.0[3],
            ])
        }
    }

    /// Lane-wise refined reciprocal `1/self`.
    ///
    /// NEON: initial FRECPE estimate (~8 bits) plus two FRECPS refinement
    /// steps (8 → 16 → ≥24 bits, i.e. near-correctly-rounded f32), staying on
    /// the multiply/add pipelines. Elsewhere: plain division.
    #[inline(always)]
    pub(crate) fn recip(self) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self(unsafe {
                use std::arch::aarch64::*;
                let e = vrecpeq_f32(self.0);
                let e = vmulq_f32(e, vrecpsq_f32(self.0, e));
                vmulq_f32(e, vrecpsq_f32(self.0, e))
            })
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self([
                1.0 / self.0[0],
                1.0 / self.0[1],
                1.0 / self.0[2],
                1.0 / self.0[3],
            ])
        }
    }

    /// Sum of all four lanes (left-to-right pair tree, matching FA add order).
    #[inline(always)]
    pub(crate) fn hsum(self) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { std::arch::aarch64::vaddvq_f32(self.0) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            (self.0[0] + self.0[1]) + (self.0[2] + self.0[3])
        }
    }
}

impl std::ops::Sub for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        F32x4::sub(self, rhs)
    }
}

impl std::ops::Mul for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        F32x4::mul(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    trait F64EpsilonExt {
        fn eps_scale(self) -> f64;
    }
    impl F64EpsilonExt for f64 {
        fn eps_scale(self) -> f64 {
            self.abs().max(1.0) * f64::EPSILON
        }
    }

    #[test]
    fn refined_reciprocal_matches_division() {
        // Values spanning magnitudes and including awkward bit patterns.
        let vals: Vec<f64> = vec![
            1.0,
            -1.0,
            std::f64::consts::PI * 0.5,
            1e-300,
            -1e300,
            0.0316227766,
            7.0 / 3.0,
            f64::MIN_POSITIVE * 8.0,
        ];
        for &x in &vals {
            let v = F64x2::load(&[x, 2.0], 0);
            let r = v.recip();
            assert!((r.hsum() - (1.0 / x + 0.5)).abs() <= (1.0 / x).abs().eps_scale() * 4.0);
        }
    }

    #[test]
    fn f64x2_ops_match_scalar() {
        let a = F64x2::load(&[3.5, -2.25], 0);
        let b = F64x2::load(&[1.25, 4.0], 0);
        let s = (a - b).hsum();
        assert!((s - ((3.5 - 1.25) + (-2.25 - 4.0))).abs() < 1e-15);
        let m = (a * b).hsum();
        assert!((m - (3.5 * 1.25 + -2.25 * 4.0)).abs() < 1e-14);
        let f = F64x2::splat(1.0).fma(a, b).hsum();
        assert!((f - (1.0 + 3.5 * 1.25 + 1.0 + -2.25 * 4.0)).abs() < 1e-14);
    }

    #[test]
    fn f32x4_refined_reciprocal_matches_division() {
        // f32 has ~24 bits; the 2-step FRECPS refinement targets ≤1 ulp.
        let vals: Vec<f32> = vec![
            1.0,
            -1.0,
            std::f32::consts::PI * 0.5,
            1e-30,
            -1e30,
            0.0316227766,
            7.0 / 3.0,
            f32::MIN_POSITIVE * 8.0,
        ];
        for &x in &vals {
            let v = F32x4::load(&[x, 2.0, -3.0, 0.5], 0);
            let r = v.recip();
            let got = r.lane(0);
            let want = 1.0f32 / x;
            assert!(
                (got - want).abs() <= want.abs() * 1e-6 + f32::MIN_POSITIVE,
                "x={x}: recip={got} want={want}"
            );
        }
    }

    #[test]
    fn f32x4_ops_match_scalar() {
        let a = F32x4::load(&[3.5, -2.25, 1.0, -8.0], 0);
        let b = F32x4::load(&[1.25, 4.0, -0.5, 2.0], 0);
        let s = (a - b).hsum();
        let ws = (3.5 - 1.25) + (-2.25 - 4.0) + (1.0 + 0.5) + (-8.0 - 2.0);
        assert!((s - ws).abs() < 1e-5);
        let m = (a * b).hsum();
        let wm = 3.5 * 1.25 + -2.25 * 4.0 + 1.0 * -0.5 + -8.0 * 2.0;
        assert!((m - wm).abs() < 1e-4);
        // `lane` round-trips each lane.
        assert_eq!(a.lane(2), 1.0);
        assert_eq!(a.lane(3), -8.0);
        let f = F32x4::splat(1.0).fma(a, b).hsum();
        let wf: f32 = 1.0 + 3.5 * 1.25 + 1.0 + -2.25 * 4.0 + 1.0 + 1.0 * -0.5 + 1.0 + -8.0 * 2.0;
        assert!((f - wf).abs() < 1e-4);
    }
}
