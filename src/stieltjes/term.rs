//! Core Stieltjes term — formula ownership.
//!
//! The kernel is `1 / ((λᵢ-λⱼ) - iη)`:
//!
//! ```text
//! real = diff / (diff² + η²)    imag = η / (diff² + η²)    diff = λᵢ-λⱼ
//! ```
//!
//! [`stieltjes_term`] and [`stieltjes_term_hoisted`] are the canonical
//! forms every high-level path goes through when it does NOT need manual
//! control of its hot loop. The direct-sum kernels (cacheblock,
//! blocked_autovec, the symmetric-pair sweep, hodlr's near field) inline
//! this same arithmetic BY HAND so they can unroll, fuse and vectorize
//! it — that duplication is deliberate, measured and documented at each
//! site; a change to the formula must be mirrored there.
//!
//! This module also owns [`BLOCK_SZ`], the shared default block size of
//! the blocked family. (The far-field cutoff ratio lives with its only
//! consumer as `adaptive::DEFAULT_CUTOFF_RATIO`.)

/// Default cache block size (in number of eigenvalues). Must be a multiple of 4.
pub const BLOCK_SZ: usize = 64;

/// Standard Stieltjes term: `1 / ((λᵢ-λⱼ) - iη)` → `(real, imag)`.
#[inline(always)]
pub fn stieltjes_term(lambda_i: f64, lambda_j: f64, eta: f64) -> (f64, f64) {
    let diff = lambda_i - lambda_j;
    let denom = diff * diff + eta * eta;
    let inv_denom = 1.0 / denom;
    (diff * inv_denom, eta * inv_denom)
}

/// **Eta-hoisted term.** Returns `(real, inv)` where `imag = eta · inv`.
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
    fn test_hoisted_matches_plain_form() {
        // (real, inv) with imag = eta·inv must equal (real, imag).
        for &li in &[0.1, 1.0, 5.0, 10.0] {
            for &lj in &[0.0, 0.5, 3.0, 8.0] {
                for &eta in &[0.01, 0.1, 0.5] {
                    let (r1, i1) = stieltjes_term(li, lj, eta);
                    let (r2, inv) = stieltjes_term_hoisted(li, lj, eta);
                    assert!((r1 - r2).abs() < 5e-16, "real mismatch");
                    assert!((i1 - eta * inv).abs() < 5e-16, "imag mismatch");
                }
            }
        }
    }
}
