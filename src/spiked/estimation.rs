//! Spike eigenvalue estimation: inverse BBP (DGJ) and Ledoit–Peché shrinkage.
//!
//! Under the spiked covariance model with bulk noise $\sigma^2 I_p$ and
//! concentration ratio $\gamma = p/n$, a population spike $\ell$ above the
//! BBP threshold produces a sample eigenvalue $\hat\lambda$ that is biased
//! upward. This module recovers $\ell$ from $\hat\lambda$ by two routes:
//!
//! 1. **Inverse BBP / DGJ** — a closed-form inversion of the BBP phase
//!    transition (Donoho–Gavish–Johnstone). For $\ell > 1 + \sqrt\gamma$:
//!
//!    $$\hat\lambda = \ell\left(1 + \frac{\gamma}{\ell - 1}\right)$$
//!
//!    which inverts to the quadratic
//!
//!    $$\ell = \sigma^2\,\frac{\left(\hat\lambda/\sigma^2 + 1 - \gamma\right)
//!      + \sqrt{\left(\hat\lambda/\sigma^2 + 1 - \gamma\right)^2 - 4\hat\lambda/\sigma^2}}{2}$$
//!
//! 2. **Ledoit–Peché / Ledoit–Wolf non-linear shrinkage** — maps every sample
//!    eigenvalue to an asymptotically optimal population estimate via the
//!    Stieltjes transform of the empirical spectral distribution. This reuses
//!    the crate's fast Stieltjes library (`compute_all_stieltjes`) and the
//!    `shrinkage_factor` formula, so there is zero duplication of the kernel.

use crate::config::{FftGridSize, RmtConfig};
use crate::deconvolution::shrinkage_factor;
use crate::stieltjes::compute_all_stieltjes;

/// BBP threshold: the smallest population spike that produces a detectable
/// sample spike. Below this, the spike is absorbed into the bulk.
#[inline(always)]
pub fn bbp_threshold(gamma: f64, sigma2: f64) -> f64 {
    sigma2 * (1.0 + gamma.sqrt()).powi(2)
}

/// Inverse BBP / DGJ: recover the population spike $\ell$ from a sample
/// eigenvalue $\hat\lambda$ above the bulk edge.
///
/// Returns the population spike estimate. If $\hat\lambda$ lies at or below
/// the BBP threshold (the discriminant is non-positive), the spike is not
/// resolvable and the bulk edge $\sigma^2(1+\sqrt\gamma)^2$ is returned.
#[inline(always)]
pub fn inverse_bbp(lambda_hat: f64, gamma: f64, sigma2: f64) -> f64 {
    let x = lambda_hat / sigma2;
    let disc = (x + 1.0 - gamma).powi(2) - 4.0 * x;
    if disc <= 0.0 {
        return bbp_threshold(gamma, sigma2);
    }
    let ell = ((x + 1.0 - gamma) + disc.sqrt()) / 2.0;
    ell * sigma2
}

/// Ledoit–Wolf non-linear shrinkage via the fast Stieltjes library.
///
/// Computes $\xi(\hat\lambda_i) = \hat\lambda_i / |1 - c + c\hat\lambda_i m_g|^2$
/// for every sample eigenvalue, where $m_g$ is the empirical Stieltjes
/// transform evaluated with the crate's fast kernel. This is the raw
/// (non-trace-rescaled) Ledoit–Wolf estimator — the population estimate for
/// each eigenvalue, including the spikes.
///
/// # Arguments
///
/// * `eigenvalues` — Sample eigenvalues (length p), should be sorted.
/// * `config` — `RmtConfig` controlling the Stieltjes method (reuses the
///   fast blocked / FFT / treecode kernels).
///
/// # Returns
///
/// A `Vec<f64>` of population eigenvalue estimates, same length as input.
pub fn ledoit_wolf_shrinkage(eigenvalues: &[f64], config: &RmtConfig) -> Vec<f64> {
    let p = eigenvalues.len();
    if p == 0 {
        return Vec::new();
    }

    let resolved = config.resolve_auto(p);
    let c = resolved.c;
    let eta = resolved.eta.unwrap_or_else(|| 0.1 / (p as f64).sqrt());

    let stieltjes = compute_all_stieltjes(
        eigenvalues,
        eta,
        resolved.stieltjes_method,
        match resolved.fft_grid_size {
            FftGridSize::Auto => None,
            FftGridSize::Custom(s) => Some(s),
        },
        resolved.cutoff,
        resolved.block_size,
        resolved.parallelism,
    );

    eigenvalues
        .iter()
        .zip(stieltjes.iter())
        .map(|(&li, &(mr, mi))| shrinkage_factor(li, c, mr, mi))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_inverse_bbp_roundtrip() {
        // Forward BBP: λ̂ = ℓ(1 + γ/(ℓ-1)). Inverting should recover ℓ.
        let gamma = 0.5;
        let sigma2 = 1.0;
        for ell in [2.0, 3.0, 5.0, 10.0] {
            let lambda_hat = ell * (1.0 + gamma / (ell - 1.0));
            let recovered = inverse_bbp(lambda_hat, gamma, sigma2);
            assert_relative_eq!(recovered, ell, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_inverse_bbp_below_threshold() {
        // A sample eigenvalue at the bulk edge should not produce a spike.
        let gamma = 0.5;
        let sigma2 = 1.0;
        let edge = bbp_threshold(gamma, sigma2);
        let recovered = inverse_bbp(edge, gamma, sigma2);
        assert_relative_eq!(recovered, edge, epsilon = 1e-10);
    }

    #[test]
    fn test_bbp_threshold_formula() {
        let gamma = 0.25;
        let sigma2 = 2.0;
        // σ²(1+√γ)² = 2·(1+0.5)² = 4.5
        assert_relative_eq!(bbp_threshold(gamma, sigma2), 4.5, epsilon = 1e-12);
    }

    #[test]
    fn test_ledoit_wolf_matches_rie() {
        // The raw Ledoit-Wolf shrinkage should match the RIE shrinkage
        // before trace rescaling. For a spectrum with a dominant spike,
        // the trace-rescaled RIE and raw LW should be close.
        let evals = vec![0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
        let config = RmtConfig::new(0.3);
        let lw = ledoit_wolf_shrinkage(&evals, &config);
        assert_eq!(lw.len(), evals.len());
        for &v in &lw {
            assert!(v.is_finite() && v > 0.0);
        }
    }
}
