//! Hybrid spiked deconvolution — separate spikes (BBP) + deconvolve the bulk.
//!
//! The classical free-probability deconvolution ([`super::spectral_deconvolution`])
//! assumes a **continuous** population spectral density. When the true spectrum
//! contains isolated *spikes* (strong signal eigenvalues, e.g. market factors),
//! applying it directly distorts both the spike locations and the adjacent bulk
//! (Gibbs oscillations, BBP bias). The correct approach is the two-stage hybrid:
//!
//! 1. **Spike separation** — detect the sample eigenvalues that escape the bulk
//!    edge (BEMA) and *debias* each one back to its population value via the
//!    inverse BBP / DGJ formula. This logic lives in
//!    [`crate::spiked::separate_spikes`] (the spiked toolkit), not here.
//! 2. **Bulk deconvolution** — remove the spike eigenvalues and run the
//!    free-probability deconvolution on the remaining bulk spectrum, where the
//!    continuous-density assumption holds.
//!
//! This module only **orchestrates** the two stages — it reuses the spiked
//! toolkit ([`crate::spiked`]) and the El Karoui bulk deconvolution
//! ([`super::spectral_deconvolution`]); it adds no duplicated kernel.

use crate::config::RmtConfig;
use crate::spiked::separate_spikes;

use super::{DeconvolutionResult, spectral_deconvolution};

/// Result of the hybrid spiked deconvolution.
#[derive(Debug, Clone)]
pub struct SpikedDeconvolutionResult {
    /// Estimated number of spikes $K$.
    pub k: usize,
    /// Estimated **population** spike eigenvalues $\ell_i$ (descending),
    /// recovered from the sample spikes via inverse BBP.
    pub spikes: Vec<f64>,
    /// The sample eigenvalues that were classified as spikes (descending).
    pub spike_sample: Vec<f64>,
    /// Estimated bulk edge $\lambda_+ = \sigma^2(1+\sqrt\gamma)^2$.
    pub bulk_edge: f64,
    /// Estimated noise variance $\sigma^2$.
    pub sigma2: f64,
    /// Deconvolution of the **bulk** spectrum (spikes removed).
    pub bulk: DeconvolutionResult,
}

/// **Bulk deconvolution.**
///
/// Runs the free-probability (El Karoui) deconvolution on a **bulk-only**
/// spectrum (spikes already removed, e.g. via
/// [`crate::spiked::separate_spikes`]). This is where the continuous-density
/// assumption of the MP inversion holds.
///
/// # Arguments
///
/// * `bulk_evals` — Bulk eigenvalues (spikes removed; any order).
/// * `c` — Concentration ratio $p/n$.
/// * `n_points` — Grid resolution for the bulk deconvolution.
/// * `eta` — Regularization for the bulk deconvolution (default `0.1/√p`).
/// * `config` — `RmtConfig` (used for eta consistency with the rest of the crate).
///
/// # Returns
///
/// A [`DeconvolutionResult`] for the bulk spectrum.
pub fn deconvolve_bulk(
    bulk_evals: &[f64],
    c: f64,
    n_points: usize,
    eta: Option<f64>,
    config: &RmtConfig,
) -> DeconvolutionResult {
    spectral_deconvolution(bulk_evals, c, n_points, eta, None, None, config)
}

/// Hybrid spiked deconvolution.
///
/// Convenience wrapper that chains the two stages:
/// [`crate::spiked::separate_spikes`] (detect + debias + remove spikes) then
/// [`deconvolve_bulk`] (El Karoui deconvolution of the remaining bulk).
///
/// # Arguments
///
/// * `eigenvalues` — Sample eigenvalues (any order; sorted internally).
/// * `c` — Concentration ratio $p/n$.
/// * `n_points` — Grid resolution for the bulk deconvolution.
/// * `eta` — Regularization for the bulk deconvolution (default `0.1/√p`).
/// * `margin` — Multiplicative margin above the fitted bulk edge for spike
///   detection (default 1.0; slightly above 1.0 adds robustness).
/// * `config` — `RmtConfig` (used for eta consistency with the rest of the crate).
///
/// # Returns
///
/// A [`SpikedDeconvolutionResult`] with the debiased spikes and the bulk
/// deconvolution.
pub fn deconvolve_spiked(
    eigenvalues: &[f64],
    c: f64,
    n_points: usize,
    eta: Option<f64>,
    margin: f64,
    config: &RmtConfig,
) -> SpikedDeconvolutionResult {
    let p = eigenvalues.len();
    if p == 0 {
        return SpikedDeconvolutionResult {
            k: 0,
            spikes: Vec::new(),
            spike_sample: Vec::new(),
            bulk_edge: 0.0,
            sigma2: 0.0,
            bulk: DeconvolutionResult {
                lambda_grid: Vec::new(),
                density: Vec::new(),
                w_re: Vec::new(),
                sample_stieltjes_real: Vec::new(),
                sample_stieltjes_imag: Vec::new(),
                population_stieltjes_real: Vec::new(),
                population_stieltjes_imag: Vec::new(),
            },
        };
    }

    // Stage 1: separate the spikes from the bulk (lives in crate::spiked).
    let sep = separate_spikes(eigenvalues, c, margin);

    // Stage 2: deconvolve the bulk (spikes removed).
    let bulk = deconvolve_bulk(&sep.bulk_evals, c, n_points, eta, config);

    SpikedDeconvolutionResult {
        k: sep.k,
        spikes: sep.spikes,
        spike_sample: sep.spike_sample,
        bulk_edge: sep.bulk_edge,
        sigma2: sep.sigma2,
        bulk,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RmtConfig;

    /// A pure-bulk spectrum (no spikes) should yield k = 0 and a bulk
    /// deconvolution identical to the plain `spectral_deconvolution`.
    #[test]
    fn test_no_spikes_delegates_to_bulk() {
        let evals: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.01).collect();
        let config = RmtConfig::new(0.25);
        let res = deconvolve_spiked(&evals, 0.25, 100, Some(0.1), 1.0, &config);
        assert_eq!(res.k, 0);
        assert!(res.spikes.is_empty());
        assert!(res.spike_sample.is_empty());
        // Bulk deconvolution should match the plain one on the same input.
        let plain = spectral_deconvolution(&evals, 0.25, 100, Some(0.1), None, None, &config);
        assert_eq!(res.bulk.density.len(), plain.density.len());
        for (a, b) in res.bulk.density.iter().zip(plain.density.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    /// A spectrum with clear spikes should detect them, debias them, and
    /// deconvolve the remaining bulk.
    #[test]
    fn test_spiked_hybrid() {
        let mut evals: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.015).collect();
        evals.push(5.0);
        evals.push(7.0);
        evals.push(10.0);
        let config = RmtConfig::new(0.25);
        let res = deconvolve_spiked(&evals, 0.25, 100, Some(0.1), 1.0, &config);
        assert_eq!(res.k, 3);
        assert_eq!(res.spikes.len(), 3);
        assert_eq!(res.spike_sample.len(), 3);
        // Spikes descending.
        assert!(res.spikes[0] > res.spikes[1]);
        assert!(res.spikes[1] > res.spikes[2]);
        // Population spikes should be below their sample counterparts (BBP bias).
        for (pop, samp) in res.spikes.iter().zip(res.spike_sample.iter()) {
            assert!(
                pop < samp,
                "population spike {pop} should be < sample {samp}"
            );
        }
        // Bulk deconvolution should be finite and non-empty.
        assert_eq!(res.bulk.density.len(), 100);
        assert!(res.bulk.density.iter().all(|d| d.is_finite()));
    }

    #[test]
    fn test_empty() {
        let config = RmtConfig::new(0.5);
        let res = deconvolve_spiked(&[], 0.5, 100, None, 1.0, &config);
        assert_eq!(res.k, 0);
        assert!(res.bulk.density.is_empty());
    }

    /// The two stages must compose: `separate_spikes` + `deconvolve_bulk`
    /// must reproduce `deconvolve_spiked` exactly.
    #[test]
    fn test_stages_compose() {
        let mut evals: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.015).collect();
        evals.push(5.0);
        evals.push(7.0);
        evals.push(10.0);
        let config = RmtConfig::new(0.25);

        let sep = crate::spiked::separate_spikes(&evals, 0.25, 1.0);
        assert_eq!(sep.k, 3);
        assert_eq!(sep.spikes.len(), 3);
        assert_eq!(sep.spike_sample.len(), 3);
        // Bulk has all eigenvalues minus the spikes.
        assert_eq!(sep.bulk_evals.len(), evals.len() - 3);
        // Spikes descending.
        assert!(sep.spikes[0] > sep.spikes[1]);
        assert!(sep.spikes[1] > sep.spikes[2]);
        // Population spikes below their sample counterparts (BBP bias).
        for (pop, samp) in sep.spikes.iter().zip(sep.spike_sample.iter()) {
            assert!(
                pop < samp,
                "population spike {pop} should be < sample {samp}"
            );
        }

        // Stage 2 on the separated bulk.
        let bulk = deconvolve_bulk(&sep.bulk_evals, 0.25, 100, Some(0.1), &config);
        assert_eq!(bulk.density.len(), 100);
        assert!(bulk.density.iter().all(|d| d.is_finite()));

        // Composition equals the one-shot wrapper.
        let combined = deconvolve_spiked(&evals, 0.25, 100, Some(0.1), 1.0, &config);
        assert_eq!(combined.k, sep.k);
        assert_eq!(combined.spikes, sep.spikes);
        assert_eq!(combined.spike_sample, sep.spike_sample);
        assert_eq!(combined.bulk_edge, sep.bulk_edge);
        assert_eq!(combined.sigma2, sep.sigma2);
        for (a, b) in combined.bulk.density.iter().zip(bulk.density.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    /// A pure-bulk spectrum (no spikes) should yield k = 0 and a bulk
    /// identical to the input.
    #[test]
    fn test_separate_spikes_no_spikes() {
        let evals: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.01).collect();
        let sep = crate::spiked::separate_spikes(&evals, 0.25, 1.0);
        assert_eq!(sep.k, 0);
        assert!(sep.spikes.is_empty());
        assert!(sep.spike_sample.is_empty());
        // Bulk equals the full (sorted) input.
        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sep.bulk_evals, sorted);
    }
}
