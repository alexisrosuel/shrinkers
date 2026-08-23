//! Spiked covariance model: spike eigenvalue & eigenvector estimation.
//!
//! This module implements the SOTA methods for estimating population spike
//! eigenvalues $\ell_i$ and spike eigenvectors $v_i$ from sample counterparts
//! $(\hat\lambda_i, \hat u_i)$ under the high-dimensional regime
//! $\gamma = p/n \to \gamma > 0$.
//!
//! # Methods
//!
//! | Goal | Method | Module |
//! |---|---|---|
//! | Estimate population eigenvalues $\ell_i$ | Inverse BBP / DGJ, Ledoit–Peché non-linear shrinkage | [`estimation`] |
//! | Estimate population eigenvectors $v_i$ | BBP angle formula (Benaych-Georges & Nadakuditi), debiased projection | [`eigenvector`] |
//! | Determine number of spikes $K$ | BEMA (Bulk Eigenvalue Matching), Tracy–Widom edge thresholding | [`detection`] |
//!
//! # Reuse
//!
//! The Ledoit–Peché shrinkage reuses the crate's fast Stieltjes library
//! ([`crate::stieltjes::compute_all_stieltjes`]) and the
//! [`crate::deconvolution::shrinkage_factor`] formula — no duplicated kernel.

pub mod detection;
pub mod eigenvector;
pub mod estimation;

pub use detection::*;
pub use eigenvector::*;
pub use estimation::*;

use crate::config::RmtConfig;

/// Shared stage-1 spike pipeline used by [`analyze_spikes`],
/// [`separate_spikes`], and
/// [`crate::deconvolution::estimate_population_eigenvalues`].
///
/// Sorts ascending, detects spikes (BEMA), debiases each sample spike to its
/// population value via inverse BBP, and splits the spectrum into spikes and
/// bulk. All vectors are returned in **ascending** order; callers reverse the
/// spike vectors when they want descending output.
pub(crate) struct SpikeSplit {
    /// Raw BEMA detection result (indices ascending into `sorted`).
    pub det: SpikeDetection,
    /// The full spectrum, sorted ascending.
    pub sorted: Vec<f64>,
    /// Population spike estimates ℓ_i, ascending (smallest first).
    pub spikes: Vec<f64>,
    /// Sample eigenvalues classified as spikes, ascending.
    pub spike_sample: Vec<f64>,
    /// Bulk eigenvalues (spikes removed), ascending.
    pub bulk_evals: Vec<f64>,
}

/// Sort ascending, detect spikes (BEMA), debias via inverse BBP, split bulk.
pub(crate) fn detect_debias_split(eigenvalues: &[f64], gamma: f64, margin: f64) -> SpikeSplit {
    // Sort ascending (BEMA expects ascending).
    let mut sorted = eigenvalues.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // 1. Detect spikes (BEMA). spike_indices are ascending indices into `sorted`.
    let det = detection::detect_spikes_bema(&sorted, gamma, margin);
    let sigma2 = det.sigma2;

    // 2. De-bias each sample spike to its population value via inverse BBP
    //    (in units of σ²; pass sigma2 to get absolute ℓ).
    let mut spikes = Vec::with_capacity(det.k);
    let mut spike_sample = Vec::with_capacity(det.k);
    for &idx in &det.spike_indices {
        let lambda_hat = sorted[idx];
        spike_sample.push(lambda_hat);
        spikes.push(estimation::inverse_bbp(lambda_hat, gamma, sigma2));
    }

    // 3. Bulk = all eigenvalues except the spikes (indices are ascending).
    let mut bulk_evals = Vec::with_capacity(sorted.len() - det.k);
    let mut spike_iter = det.spike_indices.iter().peekable();
    for (i, &lam) in sorted.iter().enumerate() {
        if spike_iter.peek() == Some(&&i) {
            spike_iter.next();
        } else {
            bulk_evals.push(lam);
        }
    }

    SpikeSplit {
        det,
        sorted,
        spikes,
        spike_sample,
        bulk_evals,
    }
}

/// High-level result of spiked-model analysis.
#[derive(Debug, Clone)]
pub struct SpikedResult {
    /// Estimated number of spikes $K$.
    pub k: usize,
    /// Estimated population spike eigenvalues $\ell_i$ (length K), descending.
    pub spikes: Vec<f64>,
    /// Estimated squared angular overlaps $\alpha_i^2$ for the spikes.
    pub overlaps: Vec<f64>,
    /// Estimated bulk edge $\lambda_+ = \sigma^2(1+\sqrt\gamma)^2$.
    pub bulk_edge: f64,
    /// Estimated noise variance $\sigma^2$.
    pub sigma2: f64,
    /// Ledoit–Wolf non-linear shrinkage estimates for **all** eigenvalues
    /// (population estimates), computed via the fast Stieltjes library.
    pub ledoit_wolf: Vec<f64>,
}

/// Run the full spiked-model analysis on a sample eigenvalue spectrum.
///
/// # Arguments
///
/// * `eigenvalues` — Sample eigenvalues (any order; sorted internally).
/// * `gamma` — Concentration ratio $p/n$.
/// * `config` — `RmtConfig` controlling the Stieltjes method used for
///   Ledoit–Peché shrinkage.
/// * `margin` — Multiplicative margin above the fitted bulk edge for spike
///   detection (default 1.0).
///
/// # Returns
///
/// A [`SpikedResult`] with the detected spikes, their population estimates,
/// their angular overlaps, and the full Ledoit–Wolf shrinkage vector.
pub fn analyze_spikes(
    eigenvalues: &[f64],
    gamma: f64,
    config: &RmtConfig,
    margin: f64,
) -> SpikedResult {
    let n = eigenvalues.len();
    if n == 0 {
        return SpikedResult {
            k: 0,
            spikes: Vec::new(),
            overlaps: Vec::new(),
            bulk_edge: 0.0,
            sigma2: 0.0,
            ledoit_wolf: Vec::new(),
        };
    }

    let split = detect_debias_split(eigenvalues, gamma, margin);
    let det = split.det;
    let sigma2 = det.sigma2;

    // 2. Population spikes, descending (largest first).
    let mut spikes = split.spikes;
    spikes.reverse();

    // 3. Angular overlaps (in units of σ²=1, so divide ℓ by σ²).
    let overlaps = eigenvector::bbp_angle_overlaps(
        &spikes.iter().map(|&l| l / sigma2).collect::<Vec<_>>(),
        gamma,
    );

    // 4. Ledoit–Wolf non-linear shrinkage for all eigenvalues (reuses the
    //    fast Stieltjes library via the provided config).
    let ledoit_wolf = estimation::ledoit_wolf_shrinkage(&split.sorted, config);

    SpikedResult {
        k: det.k,
        spikes,
        overlaps,
        bulk_edge: det.bulk_edge,
        sigma2,
        ledoit_wolf,
    }
}

/// Result of **spike separation**: the sample spectrum split into its
/// isolated spikes and the remaining bulk (spikes removed).
///
/// This is the output of [`separate_spikes`] — the first stage of the hybrid
/// spiked deconvolution. The bulk eigenvalues are ready to be fed to the
/// free-probability deconvolution ([`crate::deconvolution::deconvolve_bulk`]).
#[derive(Debug, Clone)]
pub struct SpikeSeparation {
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
    /// The **bulk** eigenvalues (ascending) — the sample spectrum with the
    /// spikes removed.
    pub bulk_evals: Vec<f64>,
}

/// **Spike separation** — detect the sample eigenvalues that escape the bulk
/// edge (BEMA), debias each one back to its population value via the inverse
/// BBP / DGJ formula, and return the remaining bulk spectrum (spikes removed).
///
/// This is the first stage of the hybrid spiked deconvolution: treat the
/// spikes first, then deconvolve the matrix stripped of its spikes.
///
/// # Arguments
///
/// * `eigenvalues` — Sample eigenvalues (any order; sorted internally).
/// * `gamma` — Concentration ratio $p/n$.
/// * `margin` — Multiplicative margin above the fitted bulk edge for spike
///   detection (default 1.0; slightly above 1.0 adds robustness).
///
/// # Returns
///
/// A [`SpikeSeparation`] with the debiased spikes and the bulk eigenvalues.
pub fn separate_spikes(eigenvalues: &[f64], gamma: f64, margin: f64) -> SpikeSeparation {
    let p = eigenvalues.len();
    if p == 0 {
        return SpikeSeparation {
            k: 0,
            spikes: Vec::new(),
            spike_sample: Vec::new(),
            bulk_edge: 0.0,
            sigma2: 0.0,
            bulk_evals: Vec::new(),
        };
    }

    // Shared stage-1 pipeline (sort + detect + debias + split).
    let split = detect_debias_split(eigenvalues, gamma, margin);
    let det = split.det;

    // Population spikes and their sample counterparts, descending.
    let mut spikes = split.spikes;
    spikes.reverse();
    let mut spike_sample = split.spike_sample;
    spike_sample.reverse();

    SpikeSeparation {
        k: det.k,
        spikes,
        spike_sample,
        bulk_edge: det.bulk_edge,
        sigma2: det.sigma2,
        bulk_evals: split.bulk_evals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RmtConfig;

    #[test]
    fn test_analyze_spikes_known() {
        // Bulk around 1.0 (σ²≈1, γ=0.25 → edge 2.25) + 3 clear spikes.
        let mut evals: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.015).collect();
        evals.push(5.0);
        evals.push(7.0);
        evals.push(10.0);

        let config = RmtConfig::new(0.25);
        let result = analyze_spikes(&evals, 0.25, &config, 1.0);
        assert_eq!(result.k, 3);
        assert_eq!(result.spikes.len(), 3);
        assert_eq!(result.overlaps.len(), 3);
        // Spikes descending.
        assert!(result.spikes[0] > result.spikes[1]);
        assert!(result.spikes[1] > result.spikes[2]);
        // All overlaps in [0,1].
        for &o in &result.overlaps {
            assert!((0.0..=1.0).contains(&o));
        }
    }

    #[test]
    fn test_analyze_spikes_no_spikes() {
        let evals: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.01).collect();
        let config = RmtConfig::new(0.25);
        let result = analyze_spikes(&evals, 0.25, &config, 1.0);
        assert_eq!(result.k, 0);
        assert!(result.spikes.is_empty());
    }

    /// `separate_spikes` detects spikes, debiases them, and returns the bulk
    /// (spikes removed).
    #[test]
    fn test_separate_spikes_known() {
        let mut evals: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.015).collect();
        evals.push(5.0);
        evals.push(7.0);
        evals.push(10.0);

        let sep = separate_spikes(&evals, 0.25, 1.0);
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
    }

    /// A pure-bulk spectrum (no spikes) should yield k = 0 and a bulk
    /// identical to the input.
    #[test]
    fn test_separate_spikes_no_spikes() {
        let evals: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.01).collect();
        let sep = separate_spikes(&evals, 0.25, 1.0);
        assert_eq!(sep.k, 0);
        assert!(sep.spikes.is_empty());
        assert!(sep.spike_sample.is_empty());
        // Bulk equals the full (sorted) input.
        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sep.bulk_evals, sorted);
    }
}
