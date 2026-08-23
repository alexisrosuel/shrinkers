//! Population eigenvalue estimation — the crate's primary entry point.
//!
//! This is the **core** of the crate: estimate the *population* eigenvalues
//! from sample eigenvalues under a spiked covariance model. It is a single
//! call that orchestrates the full pipeline:
//!
//! 1. **Spike detection** (BEMA) — find the sample eigenvalues that escape
//!    the bulk edge.
//! 2. **Spike debiasing** (inverse BBP / DGJ) — recover the population spike
//!    eigenvalues $\ell_i$ from the biased sample spikes.
//! 3. **Bulk deconvolution** (Ledoit–Wolf / RIE) — map every remaining sample
//!    eigenvalue to its population estimate via the Stieltjes transform.
//!
//! The RIE / Ledoit–Wolf shrinkage is the *pointwise* special case of the
//! free-probability deconvolution: it recovers the population counterpart of
//! each individual eigenvalue, whereas the full deconvolution
//! ([`super::spectral_deconvolution`]) recovers the whole population density.
//! This module reuses the existing spiked toolkit and the existing
//! Ledoit–Wolf shrinkage — no duplicated kernel.

use crate::config::RmtConfig;
use crate::spiked::{detect_debias_split, ledoit_wolf_shrinkage};

/// Result of the population eigenvalue estimation.
#[derive(Debug, Clone)]
pub struct PopulationEigenvalues {
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
    /// Estimated **population** eigenvalues for the **bulk** (Ledoit–Wolf /
    /// RIE pointwise deconvolution), ascending, one per bulk sample eigenvalue.
    pub bulk_population: Vec<f64>,
    /// The bulk sample eigenvalues (spikes removed), ascending.
    pub bulk_sample: Vec<f64>,
}

/// Estimate the population eigenvalues from sample eigenvalues.
///
/// This is the crate's primary entry point. It detects spikes (BEMA),
/// debiases them (inverse BBP), and deconvolves the remaining bulk
/// (Ledoit–Wolf / RIE pointwise deconvolution).
///
/// # Arguments
///
/// * `eigenvalues` — Sample eigenvalues (any order; sorted internally).
/// * `c` — Concentration ratio $p/n$.
/// * `margin` — Multiplicative margin above the fitted bulk edge for spike
///   detection (default 1.0; slightly above 1.0 adds robustness).
/// * `config` — `RmtConfig` controlling the Stieltjes method used for the
///   Ledoit–Wolf bulk deconvolution.
///
/// # Returns
///
/// A [`PopulationEigenvalues`] with the debiased spikes and the deconvolved
/// bulk population eigenvalues.
pub fn estimate_population_eigenvalues(
    eigenvalues: &[f64],
    c: f64,
    margin: f64,
    config: &RmtConfig,
) -> PopulationEigenvalues {
    let p = eigenvalues.len();
    if p == 0 {
        return PopulationEigenvalues {
            k: 0,
            spikes: Vec::new(),
            spike_sample: Vec::new(),
            bulk_edge: 0.0,
            sigma2: 0.0,
            bulk_population: Vec::new(),
            bulk_sample: Vec::new(),
        };
    }

    // Shared stage-1 pipeline (sort + detect + debias + split).
    let split = detect_debias_split(eigenvalues, c, margin);
    let det = split.det;
    let sigma2 = det.sigma2;
    let bulk_edge = det.bulk_edge;

    // Population spikes and their sample counterparts, descending.
    let mut spikes = split.spikes;
    spikes.reverse();
    let mut spike_sample = split.spike_sample;
    spike_sample.reverse();

    // 4. Deconvolve the bulk via Ledoit–Wolf (RIE pointwise deconvolution).
    //    This reuses the fast Stieltjes library through the provided config.
    let bulk_population = ledoit_wolf_shrinkage(&split.bulk_evals, config);

    PopulationEigenvalues {
        k: det.k,
        spikes,
        spike_sample,
        bulk_edge,
        sigma2,
        bulk_population,
        bulk_sample: split.bulk_evals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RmtConfig;

    /// A pure-bulk spectrum (no spikes) should yield k = 0 and a bulk
    /// population estimate identical to plain Ledoit–Wolf on the same input.
    #[test]
    fn test_no_spikes_delegates_to_lw() {
        let evals: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.01).collect();
        let config = RmtConfig::new(0.25);
        let res = estimate_population_eigenvalues(&evals, 0.25, 1.0, &config);
        assert_eq!(res.k, 0);
        assert!(res.spikes.is_empty());
        assert!(res.spike_sample.is_empty());
        // Bulk population should match plain Ledoit–Wolf on the same input.
        let lw = ledoit_wolf_shrinkage(&evals, &config);
        assert_eq!(res.bulk_population.len(), lw.len());
        for (a, b) in res.bulk_population.iter().zip(lw.iter()) {
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
        let res = estimate_population_eigenvalues(&evals, 0.25, 1.0, &config);
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
        // Bulk population should be finite and non-empty.
        assert_eq!(res.bulk_population.len(), 100);
        assert!(res.bulk_population.iter().all(|d| d.is_finite()));
        assert_eq!(res.bulk_sample.len(), 100);
    }

    #[test]
    fn test_empty() {
        let config = RmtConfig::new(0.5);
        let res = estimate_population_eigenvalues(&[], 0.5, 1.0, &config);
        assert_eq!(res.k, 0);
        assert!(res.bulk_population.is_empty());
    }
}
