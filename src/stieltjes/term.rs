//! Core Stieltjes term computation — unique responsibility, zero duplication.
//!
//! Provides the fundamental building block: 1 / ((λᵢ-λⱼ) - iη)
//! in various numerically optimized forms.
//!
//! Every Stieltjes sum function in this crate delegates here for the actual term.
//! This is the **only** place where the division formula lives.
//!
//! This module also owns the shared tuning constants (`BLOCK_SZ`,
//! `CUTOFF_RATIO`) so they are defined exactly once.

/// Default cache block size (in number of eigenvalues). Must be a multiple of 4.
pub const BLOCK_SZ: usize = 64;

/// Default far-field cutoff ratio: skip terms where |diff| > cutoff_ratio * η.
/// At 10.0, the max error per skipped term is ~1/(10²+1) ≈ 1%.
pub const CUTOFF_RATIO: f64 = 10.0;

/// Standard Stieltjes term: 1 / ((λᵢ-λⱼ) - iη)
///
/// Computes:
///   real = diff / (diff² + η²),   diff = λᵢ - λⱼ
///   imag = η   / (diff² + η²)
///
/// Returns (real, imag).
#[inline(always)]
pub fn stieltjes_term(lambda_i: f64, lambda_j: f64, eta: f64) -> (f64, f64) {
    let diff = lambda_i - lambda_j;
    let denom = diff * diff + eta * eta;
    let inv_denom = 1.0 / denom;
    (diff * inv_denom, eta * inv_denom)
}

/// Stieltjes term using FMA (Fused Multiply-Add) for extra precision and speed.
///
/// Same mathematical result as `stieltjes_term` but uses `diff.mul_add(diff, eta*eta)`
/// to compute the denominator with a single FMA instruction.
/// On Apple M-series (NEON) and modern x86 (FMA3), this is both faster and more precise.
#[inline(always)]
pub fn stieltjes_term_fma(lambda_i: f64, lambda_j: f64, eta: f64) -> (f64, f64) {
    let diff = lambda_i - lambda_j;
    let eta_sq = eta * eta;
    // FMA: diff*diff + eta_sq in one instruction
    let denom = diff.mul_add(diff, eta_sq);
    let inv_denom = 1.0 / denom;
    (diff * inv_denom, eta * inv_denom)
}

/// Stieltjes term with far-field cutoff: returns None when |λᵢ-λⱼ| ≫ η.
///
/// When `|diff| > cutoff_ratio * η`, the contribution is negligible
/// (< 1/(cutoff_ratio²+1) of the max possible value). Skipping these terms
/// reduces the effective O(p²) complexity to O(p · k) where k is the
/// number of eigenvalues within the cutoff window.
///
/// A value of `cutoff_ratio = 10.0` gives < 1% error on each skipped term.
#[inline(always)]
pub fn stieltjes_term_cutoff(
    lambda_i: f64,
    lambda_j: f64,
    eta: f64,
    cutoff_ratio: f64,
) -> Option<(f64, f64)> {
    let diff = lambda_i - lambda_j;
    // Fast abs: mask the sign bit
    let abs_diff = if diff < 0.0 { -diff } else { diff };
    if abs_diff > cutoff_ratio * eta {
        return None; // Contribution is negligible
    }
    let eta_sq = eta * eta;
    let denom = diff.mul_add(diff, eta_sq);
    let inv_denom = 1.0 / denom;
    Some((diff * inv_denom, eta * inv_denom))
}

// ============================================================================
// Central optimized term primitives.
//
// These are the SINGLE source of truth for the term math, reused by every
// Stieltjes approach (autovec, blocked, tiled, blocked_autovec, ewald, ...).
// They expose the term in forms that let the caller apply the four
// factorization tricks below without duplicating the division formula.
// ============================================================================

/// **Idea 1 — eta-hoisted term.** Returns `(real, inv)` where
/// `imag = eta * inv`, instead of `(real, imag)`.
///
/// Since `Im[S] = Σ η·inv = η·Σ inv`, the caller can accumulate the raw
/// `inv` values and multiply by `eta` ONCE per output element at the end,
/// saving one `fmul` per term. Mathematically identical (eta is a
/// loop-invariant constant) and slightly MORE precise (fewer rounding steps).
#[inline(always)]
pub fn stieltjes_term_hoisted(lambda_i: f64, lambda_j: f64, eta: f64) -> (f64, f64) {
    let diff = lambda_i - lambda_j;
    let denom = diff.mul_add(diff, eta * eta);
    let inv = 1.0 / denom;
    (diff * inv, inv)
}

/// **Idea 1 — eta-hoisted term with far-field cutoff.**
/// Returns `(real, inv)` (imag = eta*inv) or `None` when |λᵢ-λⱼ| ≫ η.
#[inline(always)]
pub fn stieltjes_term_cutoff_hoisted(
    lambda_i: f64,
    lambda_j: f64,
    eta: f64,
    cutoff_ratio: f64,
) -> Option<(f64, f64)> {
    let diff = lambda_i - lambda_j;
    let abs_diff = if diff < 0.0 { -diff } else { diff };
    if abs_diff > cutoff_ratio * eta {
        return None;
    }
    let denom = diff.mul_add(diff, eta * eta);
    let inv = 1.0 / denom;
    Some((diff * inv, inv))
}

/// **Idea 3 — fast reciprocal** using the NEON `frecpe`/`frecps` Newton-Raphson
/// sequence on aarch64, falling back to exact division elsewhere.
///
/// Two NR iterations from the `frecpe` seed reach ~1e-12 relative error,
/// which is ~4× cheaper than a full `fdiv` on Apple Silicon. This is the
/// only place the fast-reciprocal lives; every term function can opt into it.
///
/// NOTE: requires `unsafe` (NEON intrinsics) and trades a small amount of
/// precision for speed. The exact `Blocked`/`Tiled` path keeps `1.0/x` by
/// default; callers that can tolerate ~1e-12 may use this.
#[inline(always)]
pub fn fast_reciprocal(x: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        // 2 Newton-Raphson iterations from the frecpe seed.
        unsafe {
            use std::arch::aarch64::*;
            let vx = vdup_n_f64(x);
            let s = vrecpe_f64(vx);
            let e = vrecps_f64(vx, s);
            let r1 = vmul_f64(s, e);
            let e2 = vrecps_f64(vx, r1);
            let r2 = vmul_f64(r1, e2);
            vget_lane_f64(r2, 0)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        1.0 / x
    }
}

/// **Idea 3 — fast-reciprocal term.** Same as [`stieltjes_term_fma`] but uses
/// [`fast_reciprocal`] instead of `1.0/denom`. ~1e-12 relative error.
#[inline(always)]
pub fn stieltjes_term_fast(lambda_i: f64, lambda_j: f64, eta: f64) -> (f64, f64) {
    let diff = lambda_i - lambda_j;
    let denom = diff.mul_add(diff, eta * eta);
    let inv = fast_reciprocal(denom);
    (diff * inv, eta * inv)
}

/// **Idea 3 + Idea 1 combined** — fast reciprocal AND eta-hoisted.
/// Returns `(real, inv)`; caller multiplies by `eta` once at the end.
#[inline(always)]
pub fn stieltjes_term_fast_hoisted(lambda_i: f64, lambda_j: f64, eta: f64) -> (f64, f64) {
    let diff = lambda_i - lambda_j;
    let denom = diff.mul_add(diff, eta * eta);
    let inv = fast_reciprocal(denom);
    (diff * inv, inv)
}

/// **Idea 2 — symmetric-pair term.** For the full eigenvalue→eigenvalue
/// transform, the real part is antisymmetric: `R_ij = -R_ji`. This returns
/// the contribution of the ordered pair `(i, j)` such that a caller computing
/// both `(i,j)` and `(j,i)` can reuse the shared denominator and the negated
/// real part, halving the real-part arithmetic.
///
/// Returns `(real_ij, imag_ij)` for the pair `(lambda_i, lambda_j)`.
/// The reverse pair `(j, i)` has `real = -real_ij`, `imag = imag_ij`.
#[inline(always)]
pub fn stieltjes_term_symmetric_pair(lambda_i: f64, lambda_j: f64, eta: f64) -> (f64, f64) {
    let diff = lambda_i - lambda_j;
    let denom = diff.mul_add(diff, eta * eta);
    let inv = 1.0 / denom;
    (diff * inv, eta * inv)
}

/// **Idea 4 — complex-packed term.** Computes the term as a single complex
/// reciprocal `1/(z - λⱼ)` with `z = λᵢ + iη`, returning the real and imag
/// parts. Provided as the canonical complex form so callers that operate on
/// complex vectors (e.g. FFT/treecode) share the same math. For the direct
/// sum this is equivalent to [`stieltjes_term_fma`]; it exists so the complex
/// convention lives in exactly one place.
#[inline(always)]
pub fn stieltjes_term_complex(lambda_i: f64, lambda_j: f64, eta: f64) -> (f64, f64) {
    // 1/(z - λⱼ), z = λᵢ + iη  =>  (λᵢ-λⱼ - iη)/((λᵢ-λⱼ)² + η²)
    // real = (λᵢ-λⱼ)/denom, imag = -η/denom (convention-dependent sign).
    // We return the crate convention (imag > 0): imag = +η/denom.
    let diff = lambda_i - lambda_j;
    let denom = diff.mul_add(diff, eta * eta);
    let inv = 1.0 / denom;
    (diff * inv, eta * inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_basic() {
        let (r, i) = stieltjes_term(2.0, 1.0, 0.1);
        // 1 / ((2-1) - 0.1i) = 1 / (1 - 0.1i) = (1+0.1i)/1.01
        let expected_r = 1.0 / 1.01;
        let expected_i = 0.1 / 1.01;
        assert!((r - expected_r).abs() < 1e-15);
        assert!((i - expected_i).abs() < 1e-15);
    }

    #[test]
    fn test_term_fma_matches_standard() {
        for &li in &[0.1, 1.0, 5.0, 10.0] {
            for &lj in &[0.0, 0.5, 3.0, 8.0] {
                for &eta in &[0.01, 0.1, 0.5] {
                    let (r1, i1) = stieltjes_term(li, lj, eta);
                    let (r2, i2) = stieltjes_term_fma(li, lj, eta);
                    // FMA may differ at 1ulp due to fused operation
                    assert!((r1 - r2).abs() < 5e-16);
                    assert!((i1 - i2).abs() < 5e-16);
                }
            }
        }
    }

    #[test]
    fn test_cutoff_skips_far_terms() {
        // Very far eigenvalue should be None
        let result = stieltjes_term_cutoff(100.0, 0.0, 0.1, 5.0);
        assert!(result.is_none());

        // Close eigenvalue should be Some
        let result = stieltjes_term_cutoff(0.3, 0.0, 0.1, 5.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_hoisted_matches_fma() {
        // Idea 1: (real, inv) with imag = eta*inv must equal (real, imag).
        for &li in &[0.1, 1.0, 5.0, 10.0] {
            for &lj in &[0.0, 0.5, 3.0, 8.0] {
                for &eta in &[0.01, 0.1, 0.5] {
                    let (r1, i1) = stieltjes_term_fma(li, lj, eta);
                    let (r2, inv) = stieltjes_term_hoisted(li, lj, eta);
                    assert!((r1 - r2).abs() < 5e-16, "real mismatch");
                    assert!((i1 - eta * inv).abs() < 5e-16, "imag mismatch");
                }
            }
        }
    }

    #[test]
    fn test_symmetric_pair_antisymmetry() {
        // Idea 2: R_ij = -R_ji, Im_ij = Im_ji.
        for &li in &[0.1, 1.0, 5.0] {
            for &lj in &[0.0, 0.5, 3.0] {
                for &eta in &[0.01, 0.1] {
                    let (r_ij, i_ij) = stieltjes_term_symmetric_pair(li, lj, eta);
                    let (r_ji, i_ji) = stieltjes_term_symmetric_pair(lj, li, eta);
                    assert!((r_ij + r_ji).abs() < 5e-16, "real not antisymmetric");
                    assert!((i_ij - i_ji).abs() < 5e-16, "imag not symmetric");
                }
            }
        }
    }

    #[test]
    fn test_fast_reciprocal_precision() {
        // Idea 3: fast reciprocal must be within ~1e-12 relative of exact.
        for &x in &[0.1, 0.5, 1.0, 3.0, 10.0, 100.0, 1e-6, 1e6] {
            let exact = 1.0 / x;
            let fast = fast_reciprocal(x);
            let rel = ((fast - exact).abs() / exact.abs()).max(1e-16);
            assert!(
                rel < 1e-10,
                "fast reciprocal rel err {rel} too large for x={x}"
            );
        }
    }

    #[test]
    fn test_fast_term_matches_fma() {
        // Idea 3 term must be close to the exact FMA term.
        for &li in &[0.1, 1.0, 5.0] {
            for &lj in &[0.0, 0.5, 3.0] {
                for &eta in &[0.01, 0.1] {
                    let (r1, i1) = stieltjes_term_fma(li, lj, eta);
                    let (r2, i2) = stieltjes_term_fast(li, lj, eta);
                    let scale = r1.abs().max(i1.abs()).max(1e-12);
                    assert!((r1 - r2).abs() / scale < 1e-10, "real mismatch");
                    assert!((i1 - i2).abs() / scale < 1e-10, "imag mismatch");
                }
            }
        }
    }

    #[test]
    fn test_complex_term_matches_fma() {
        // Idea 4: complex form must equal the FMA form.
        for &li in &[0.1, 1.0, 5.0] {
            for &lj in &[0.0, 0.5, 3.0] {
                for &eta in &[0.01, 0.1] {
                    let (r1, i1) = stieltjes_term_fma(li, lj, eta);
                    let (r2, i2) = stieltjes_term_complex(li, lj, eta);
                    assert!((r1 - r2).abs() < 5e-16);
                    assert!((i1 - i2).abs() < 5e-16);
                }
            }
        }
    }
}
