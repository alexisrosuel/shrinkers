//! 2-FFT O(p log p) Stieltjes transform — delegates to fft5.
//!
//! Semantically identical to the dual-convolution approach in `fft5`.
//! Exists for API completeness and benchmark comparison.

use crate::stieltjes::fft5::compute_all_stieltjes_fft5;

/// Compute all Stieltjes transforms via 2-FFT grid convolution.
///
/// Semantically identical to `fft5::compute_all_stieltjes_fft5`.
pub fn compute_all_stieltjes_fft2(
    eigenvalues: &[f64],
    eta: f64,
    grid_size_opt: Option<usize>,
) -> Vec<(f64, f64)> {
    compute_all_stieltjes_fft5(eigenvalues, eta, grid_size_opt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stieltjes::fft5::compute_all_stieltjes_fft5;

    #[test]
    fn test_2fft_equals_original() {
        for p in [257, 512, 1024, 2048] {
            let evals: Vec<f64> = (0..p).map(|i| ((i as f64 + 1.0) / 50.0).ln_1p()).collect();
            let eta = 0.05;
            let ref_result = compute_all_stieltjes_fft5(&evals, eta, None);
            let result = compute_all_stieltjes_fft2(&evals, eta, None);
            assert_eq!(result.len(), ref_result.len());
            for (i, ((r1, i1), (r2, i2))) in result.iter().zip(ref_result.iter()).enumerate() {
                let eps = 1e-14;
                assert!((r1 - r2).abs() < eps, "Real mismatch at {i}: {r1} vs {r2}");
                assert!((i1 - i2).abs() < eps, "Imag mismatch at {i}: {i1} vs {i2}");
            }
        }
    }
}
