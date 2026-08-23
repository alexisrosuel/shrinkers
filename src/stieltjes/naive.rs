//! Naive O(p²) scalar Stieltjes transform.
//!
//! Pure scalar loop — no auto-vectorization hints, no SIMD.
//! Serves as the baseline for benchmarking.

use crate::stieltjes::stieltjes_term;

/// Compute the raw sum S(λᵢ) = Σⱼ 1/((λᵢ-λⱼ) - iη)
/// using a fully naive scalar loop.
///
/// The compiler *may* still autovec but we keep it deliberately simple
/// so it matches the performance of a straightforward C/Python loop.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // deliberate: matches naive C/Python baseline
pub fn naive_stieltjes_sum(lambda_i: f64, eigenvalues: &[f64], eta: f64) -> (f64, f64) {
    let mut sum_real = 0.0;
    let mut sum_imag = 0.0;
    // Deliberately uses indexing (not iter) to match naive C/Python style
    for j in 0..eigenvalues.len() {
        let (r, i) = stieltjes_term(lambda_i, eigenvalues[j], eta);
        sum_real += r;
        sum_imag += i;
    }
    (sum_real, sum_imag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_self_consistency() {
        let evals = vec![0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
        let eta = 0.1;
        let (r, i) = naive_stieltjes_sum(1.2, &evals, eta);

        let mut expected_r = 0.0;
        let mut expected_i = 0.0;
        for &lj in &evals {
            let diff = 1.2 - lj;
            let denom = diff * diff + eta * eta;
            let inv = 1.0 / denom;
            expected_r += diff * inv;
            expected_i += eta * inv;
        }
        assert!((r - expected_r).abs() < 1e-15);
        assert!((i - expected_i).abs() < 1e-15);
    }
}
