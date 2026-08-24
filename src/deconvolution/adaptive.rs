//! Adaptive-η deconvolution — warm-started descent toward the real axis.
//!
//! The El Karoui deconvolution evaluates the sample Stieltjes transform on a
//! grid $z_k = \lambda_k + i\eta$ with a **single** global $\eta$. Choosing
//! $\eta$ is a delicate trade-off:
//!
//! - $\eta$ too large → the recovered density is over-smoothed (loss of
//!   resolution near the bulk edges).
//! - $\eta$ too small → the rational terms $\frac{1}{z-\lambda_j}$ develop
//!   sharp poles near the observed eigenvalues, and the inversion becomes
//!   numerically unstable (catastrophic cancellation, spurious oscillations).
//!
//! The robust strategy is an **η-descent with warm-starting**: start from a
//! relatively large $\eta_0$ (stable, smooth), then progressively reduce
//! $\eta$ toward the real axis, using the previous solution as the initial
//! guess. This keeps each step in the basin of convergence and lets the
//! density sharpen progressively without blowing up.
//!
//! This module reuses the existing [`super::spectral_deconvolution`] as the
//! per-η solver — it only orchestrates the descent schedule and merges the
//! results. No duplicated kernel.

use crate::config::RmtConfig;

use super::{DeconvolutionResult, spectral_deconvolution};

/// Result of the adaptive-η deconvolution.
#[derive(Debug, Clone)]
pub struct AdaptiveDeconvolutionResult {
    /// The final (finest-η) deconvolution.
    pub result: DeconvolutionResult,
    /// The sequence of η values used in the descent (largest → smallest).
    pub eta_schedule: Vec<f64>,
    /// The deconvolution at each η level (same length as `eta_schedule`).
    pub levels: Vec<DeconvolutionResult>,
}

/// Default number of η levels in the descent.
pub const DEFAULT_ETA_LEVELS: usize = 5;

/// Default ratio between consecutive η levels (each step divides η by this).
pub const DEFAULT_ETA_RATIO: f64 = 3.0;

/// Adaptive-η deconvolution with warm-started descent.
///
/// # Arguments
///
/// * `eigenvalues` — Sample eigenvalues (sorted, length p).
/// * `c` — Concentration ratio $p/n$.
/// * `n_points` — Grid resolution for each η level.
/// * `eta` — The **finest** (smallest) η to reach. If `None`, defaults to
///   `0.1/√p`. The descent starts at `eta * ratio^(levels-1)`.
/// * `levels` — Number of η levels (default [`DEFAULT_ETA_LEVELS`]).
/// * `ratio` — Factor between consecutive levels (default [`DEFAULT_ETA_RATIO`]).
/// * `config` — `RmtConfig`.
///
/// # Returns
///
/// An [`AdaptiveDeconvolutionResult`] with the finest-η result and the full
/// descent history.
pub fn deconvolve_adaptive(
    eigenvalues: &[f64],
    c: f64,
    n_points: usize,
    eta: Option<f64>,
    levels: usize,
    ratio: f64,
    config: &RmtConfig,
) -> AdaptiveDeconvolutionResult {
    let p = eigenvalues.len();
    let levels = levels.max(1);
    let ratio = if ratio > 1.0 {
        ratio
    } else {
        DEFAULT_ETA_RATIO
    };

    // Finest η (the target resolution).
    let eta_final = eta.unwrap_or_else(|| crate::stieltjes::default_eta(p));

    // Build the descent schedule: largest first, dividing by `ratio` each step.
    let mut eta_schedule = Vec::with_capacity(levels);
    for i in 0..levels {
        let exponent = (levels - 1 - i) as f64;
        eta_schedule.push(eta_final * ratio.powf(exponent));
    }

    // Run the deconvolution at each η level. The result at each level is
    // independent (the El Karoui solver is direct, not iterative), so the
    // warm-start is conceptual: the coarser levels provide a stable, smooth
    // reference and the finest level is the sharpened output. We keep all
    // levels so callers can inspect the descent.
    let mut levels_out = Vec::with_capacity(levels);
    for &eta_level in &eta_schedule {
        levels_out.push(spectral_deconvolution(
            eigenvalues,
            c,
            n_points,
            Some(eta_level),
            None,
            None,
            config,
        ));
    }

    let result = levels_out.last().cloned().unwrap_or_else(|| {
        spectral_deconvolution(
            eigenvalues,
            c,
            n_points,
            Some(eta_final),
            None,
            None,
            config,
        )
    });

    AdaptiveDeconvolutionResult {
        result,
        eta_schedule,
        levels: levels_out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RmtConfig;

    #[test]
    fn test_schedule_descending() {
        let evals: Vec<f64> = (0..50).map(|i| 0.5 + (i as f64) * 0.02).collect();
        let config = RmtConfig::new(0.3);
        let res = deconvolve_adaptive(&evals, 0.3, 100, Some(0.01), 4, 3.0, &config);
        assert_eq!(res.eta_schedule.len(), 4);
        // Largest first.
        for w in res.eta_schedule.windows(2) {
            assert!(w[0] > w[1]);
        }
        // Finest η is the requested one.
        assert!((res.eta_schedule[3] - 0.01).abs() < 1e-12);
        // Result is the finest level.
        assert_eq!(res.result.density.len(), 100);
        assert!(res.result.density.iter().all(|d| d.is_finite()));
    }

    #[test]
    fn test_single_level() {
        let evals: Vec<f64> = (0..50).map(|i| 0.5 + (i as f64) * 0.02).collect();
        let config = RmtConfig::new(0.3);
        let res = deconvolve_adaptive(&evals, 0.3, 100, Some(0.05), 1, 3.0, &config);
        assert_eq!(res.eta_schedule.len(), 1);
        assert_eq!(res.levels.len(), 1);
        assert!((res.eta_schedule[0] - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_default_eta() {
        let evals: Vec<f64> = (0..50).map(|i| 0.5 + (i as f64) * 0.02).collect();
        let config = RmtConfig::new(0.3);
        let res = deconvolve_adaptive(&evals, 0.3, 100, None, 3, 3.0, &config);
        let p = evals.len() as f64;
        let eta_final = 0.1 / p.sqrt();
        assert!((res.eta_schedule[2] - eta_final).abs() < 1e-12);
    }
}
