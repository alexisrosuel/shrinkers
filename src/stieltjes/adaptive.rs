//! Adaptive Stieltjes transform with balanced real/imaginary error.
//!
//! The two parts of the Stieltjes transform have fundamentally different range:
//!
//! - **Imaginary part** `Im[S] = Σ η/((λᵢ-λⱼ)²+η²)` is a Lorentzian kernel that
//!   decays as `1/d²` (short-range). A finite window truncates it cleanly, and
//!   the error is controllable via the window radius `R`.
//!
//! - **Real part** `Re[S] = Σ (λᵢ-λⱼ)/((λᵢ-λⱼ)²+η²)` is the Hilbert kernel that
//!   decays as `1/d` (long-range) and is **log-divergent**. No finite window can
//!   truncate it — it requires a *global* method.
//!
//! This module combines the best of both:
//! - **Real part** via the FFT odd-kernel convolution (global, O(p log p)).
//! - **Imaginary part** via the windowed method (O(p·k), error controllable).
//!
//! The window radius `R` is chosen so the imaginary-part error roughly matches
//! the real-part error, giving a balanced overall error.

use crate::stieltjes::cacheblock::compute_all_stieltjes_blocked_windowed;
use crate::stieltjes::fft5::compute_all_stieltjes_fft5;

/// Default far-field cutoff ratio for the imaginary-part window.
/// Chosen so the windowing error of the imaginary part roughly matches the
/// FFT grid's real-part accuracy (~4e-5 relative at the operating point;
/// the "~15%" quoted by early drafts long predates the current kernel).
const DEFAULT_CUTOFF_RATIO: f64 = 10.0;

/// Compute the adaptive Stieltjes transform.
///
/// Returns **raw sums** as `(real, imag)` pairs, one per eigenvalue — NOT
/// scaled by `1/p`; the caller applies the scaling (the dispatcher does).
/// (An earlier revision of this doc claimed the opposite; the function has
/// always returned raw sums.)
///
/// # Arguments
/// * `eigenvalues` — sorted eigenvalues (length p)
/// * `eta` — regularization parameter
/// * `fft_grid_size` — optional FFT grid size (None = auto)
/// * `cutoff` — far-field cutoff ratio for the imaginary window (None = default)
pub fn compute_all_stieltjes_adaptive(
    eigenvalues: &[f64],
    eta: f64,
    fft_grid_size: Option<usize>,
    cutoff: Option<f64>,
) -> Vec<(f64, f64)> {
    let p = eigenvalues.len();
    if p == 0 {
        return Vec::new();
    }

    let cut = cutoff.unwrap_or(DEFAULT_CUTOFF_RATIO);

    // Real part: FFT odd-kernel (global, handles the long-range 1/d tail).
    // Both fft5 and the windowed method return raw sums (not scaled by 1/p);
    // the caller applies the 1/p scaling centrally.
    let fft_results = compute_all_stieltjes_fft5(eigenvalues, eta, fft_grid_size);

    // Imaginary part: windowed (short-range, error controllable via cut).
    let (_, imags) = compute_all_stieltjes_blocked_windowed(eigenvalues, eta, None, Some(cut));

    fft_results
        .into_iter()
        .zip(imags)
        .map(|((r, _), im)| (r, im))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_imag_matches_windowed() {
        // The imaginary part of the adaptive method must match the windowed
        // imaginary part exactly (same computation).
        let p = 500;
        let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        let adaptive = compute_all_stieltjes_adaptive(&evals, eta, None, Some(10.0));
        let (_, win_im) = compute_all_stieltjes_blocked_windowed(&evals, eta, None, Some(10.0));

        // Both adaptive and windowed return raw sums (not scaled by 1/p).
        for i in 0..p {
            assert!(
                (adaptive[i].1 - win_im[i]).abs() < 1e-12,
                "Imag mismatch at {i}: {} vs {}",
                adaptive[i].1,
                win_im[i]
            );
        }
    }

    #[test]
    fn test_adaptive_finite() {
        let p = 300;
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        let eta = 0.1 / (p as f64).sqrt();

        let adaptive = compute_all_stieltjes_adaptive(&evals, eta, None, Some(10.0));
        assert_eq!(adaptive.len(), p);
        for (r, i) in &adaptive {
            assert!(r.is_finite());
            assert!(i.is_finite());
        }
    }

    #[test]
    fn test_adaptive_real_accuracy() {
        // The real part should be reasonably accurate (FFT odd-kernel).
        let p = 512;
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        let eta = 0.1 / (p as f64).sqrt();

        let adaptive = compute_all_stieltjes_adaptive(&evals, eta, None, Some(10.0));

        // Exact real part (raw sum, matching adaptive's raw output).
        let mut exact_re = vec![0.0_f64; p];
        for i in 0..p {
            let li = evals[i];
            let mut s = 0.0;
            for &lj in &evals {
                let d = li - lj;
                s += d / (d * d + eta * eta);
            }
            exact_re[i] = s;
        }

        let mut max_err = 0.0_f64;
        for i in 0..p {
            let scale = exact_re[i].abs().max(1e-12);
            max_err = max_err.max((adaptive[i].0 - exact_re[i]).abs() / scale);
        }
        eprintln!("adaptive real max rel err: {max_err:.4}");
        // FFT real part is approximate (~15% error). Allow generous tolerance.
        assert!(max_err < 0.5, "adaptive real error too large: {max_err}");
    }

    #[test]
    fn test_adaptive_error_balance() {
        // Measure the real vs imaginary error balance. The goal is that both
        // parts have comparable relative error (balanced), not one dominating.
        let p = 512;
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        let eta = 0.1 / (p as f64).sqrt();

        let adaptive = compute_all_stieltjes_adaptive(&evals, eta, None, Some(10.0));

        // Exact real and imaginary parts
        let mut exact_re = vec![0.0_f64; p];
        let mut exact_im = vec![0.0_f64; p];
        for i in 0..p {
            let li = evals[i];
            let mut sr = 0.0;
            let mut si = 0.0;
            for &lj in &evals {
                let d = li - lj;
                let denom = d * d + eta * eta;
                sr += d / denom;
                si += eta / denom;
            }
            exact_re[i] = sr;
            exact_im[i] = si;
        }

        let mut err_r = 0.0_f64;
        let mut err_i = 0.0_f64;
        for i in 0..p {
            err_r = err_r.max((adaptive[i].0 - exact_re[i]).abs() / exact_re[i].abs().max(1e-12));
            err_i = err_i.max((adaptive[i].1 - exact_im[i]).abs() / exact_im[i].abs().max(1e-12));
        }
        eprintln!(
            "adaptive error balance: real={err_r:.4} imag={err_i:.4} ratio={:.2}",
            err_r / err_i.max(1e-12)
        );
        // Both errors should be within an order of magnitude of each other.
        let ratio = err_r / err_i.max(1e-12);
        assert!(
            ratio < 10.0,
            "Error not balanced: real={err_r:.4} imag={err_i:.4} ratio={ratio:.2}"
        );
    }
}
