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
//! blocks of the entire crate live HERE, behind a thin safe `F64x2`
//! abstraction whose contracts are enforced by construction:
//!
//! - every load reads lanes `i..i+2` of a live `&[f64]` whose length the
//!   caller checks (`debug_assert` documents the invariant);
//! - FP64 NEON is architectural on AArch64 and the crate builds with
//!   `-C target-cpu=native` (`RUSTFLAGS`), so the feature set these
//!   intrinsics require is a compile-time constant on supported targets —
//!   no runtime detection can be missed, hence no UB from mis-dispatch.
//!
//! On non-AArch64 targets `F64x2` degrades to `[f64; 2]` with identical
//! semantics (true division instead of the refined reciprocal), so hot
//! kernels stay a single portable code path. Outside this module the crate
//! contains no `unsafe`.

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
}
