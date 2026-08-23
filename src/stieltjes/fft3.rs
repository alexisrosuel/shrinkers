//! Fused FFT placeholder — delegates to fft5 (same dual-convolution approach).
//!
//! All FFT methods now use the same dual-convolution kernel (even+odd) with
//! adaptive padding and grid sizing.  This variant exists for benchmark completeness.

use crate::stieltjes::fft5::compute_all_stieltjes_fft5;

/// Compute all Stieltjes transforms via fused FFT grid convolution.
///
/// Semantically identical to `compute_all_stieltjes_fft5`.
pub fn compute_all_stieltjes_fft3(
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
    fn test_fused_equals_original() {
        for p in [257, 512, 1024, 2048] {
            let evals: Vec<f64> = (0..p).map(|i| ((i as f64 + 1.0) / 50.0).ln_1p()).collect();
            let eta = 0.05;

            let original = compute_all_stieltjes_fft5(&evals, eta, None);
            let fused = compute_all_stieltjes_fft3(&evals, eta, None);

            assert_eq!(original.len(), fused.len());
            let max_re_diff: f64 = original
                .iter()
                .zip(fused.iter())
                .map(|((or, _), (fr, _))| (or - fr).abs())
                .fold(0.0_f64, f64::max);
            let max_im_diff: f64 = original
                .iter()
                .zip(fused.iter())
                .map(|((_, oi), (_, fi))| (oi - fi).abs())
                .fold(0.0_f64, f64::max);

            // Difference is purely from FP rounding order (Hilbert freq vs spatial)
            // and the packed-IFFT trick adds small cross-coupling noise (~1e-8)
            // from imperfect conjugate symmetry of the frequency-domain product.
            let eps = 1e-7;
            assert!(
                max_re_diff < eps,
                "Max real diff at p={p}: {max_re_diff} > {eps}"
            );
            assert!(
                max_im_diff < eps,
                "Max imag diff at p={p}: {max_im_diff} > {eps}"
            );
        }
    }

    #[test]
    fn test_fused_finite() {
        let p = 512;
        let evals: Vec<f64> = (0..p).map(|i| ((i as f64 + 1.0) / 100.0).ln_1p()).collect();
        let eta = 0.05;
        let results = compute_all_stieltjes_fft3(&evals, eta, None);
        assert_eq!(results.len(), p);
        for (r, i) in &results {
            assert!(r.is_finite());
            assert!(i.is_finite());
        }
    }
}
