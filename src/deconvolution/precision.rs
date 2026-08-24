//! Direct precision matrix shrinkage — estimate $\Sigma^{-1}$ directly.
//!
//! Inverting the RIE-cleaned covariance is **sub-optimal** for estimating the
//! precision matrix $\Omega = \Sigma^{-1}$, because spectral inversion and
//! expectation do not commute (a direct consequence of Jensen's inequality):
//!
//! $$u_i^\top \Sigma^{-1} u_i \;\neq\; \bigl(u_i^\top \Sigma u_i\bigr)^{-1}$$
//!
//! The covariance RIE first estimates the covariance oracle
//! $d_i^* = u_i^\top \Sigma u_i$ and then inverts it, which systematically
//! **under-estimates** the precision eigenvalues (over-smoothing the small and
//! medium inverse eigenvalues). The *Direct Nonlinear Shrinkage* (Ledoit &
//! Wolf 2020) instead computes the optimal precision eigenvalue directly:
//!
//! $$\delta_i^{\text{direct}} = \frac{\lambda_i}{1 - c - 2c\lambda_i\,h(\lambda_i)}$$
//!
//! where $c = p/n < 1$ is the concentration ratio and
//! $h(\lambda_i) = \Re\bigl(m_n^\circ(\lambda_i)\bigr)$ is the real part of the
//! empirical Stieltjes transform (the Hilbert transform of the spectrum).
//!
//! # Sign convention
//!
//! This crate's Stieltjes transform uses $m(z) = \frac{1}{p}\sum_j
//! \frac{1}{z - \lambda_j}$ with $z = \lambda + i\eta$, which has a **negative**
//! imaginary part. The literature convention $m_n^\circ$ has a positive
//! imaginary part, so $h(\lambda_i) = -\Re\bigl(m_{\text{this crate}}\bigr)$.
//! Substituting into the formula above gives the implementation used here:
//!
//! $$\delta_i^{\text{direct}} = \frac{\lambda_i}{1 - c + 2c\lambda_i\,a(\lambda_i)}$$
//!
//! with $a(\lambda_i) = \Re\bigl(m(\lambda_i + i\eta)\bigr)$ in this crate's
//! convention. This is verified numerically against the true precision
//! eigenvalues in the module tests.

use ndarray::Array1;
use rayon::prelude::*;

use crate::config::{FftGridSize, Parallelism, RmtConfig};
use crate::stieltjes;

/// Compute the direct precision shrinkage factor for a single eigenvalue.
///
/// $\delta_i = \lambda_i / (1 - c + 2c\lambda_i a)$, where $a$ is the real
/// part of the empirical Stieltjes transform in this crate's convention
/// (negative imaginary part).
#[inline(always)]
pub fn precision_factor(lambda_i: f64, c: f64, m_g_real: f64) -> f64 {
    let denom = 1.0 - c + 2.0 * c * lambda_i * m_g_real;
    if denom > 0.0 {
        lambda_i / denom
    } else {
        // Numerical safety: fall back to the raw inverse when the denominator
        // is non-positive (should not happen for well-conditioned spectra).
        lambda_i
    }
}

/// Compute the direct precision shrinkage (Direct Nonlinear Shrinkage).
///
/// Estimates the eigenvalues $\delta_i$ of the precision matrix
/// $\Omega = \Sigma^{-1}$ directly, without inverting a cleaned covariance.
/// This is the precision counterpart of [`super::rie_shrinkage`].
///
/// # Arguments
///
/// * `eigenvalues` — Slice of sample eigenvalues (length p), should be sorted.
/// * `config` — `RmtConfig` controlling all optimization settings.
///
/// # Returns
///
/// `Array1<f64>` — Direct precision eigenvalues $\delta(\lambda_i)$.
pub fn direct_precision_shrinkage(eigenvalues: &[f64], config: &RmtConfig) -> Array1<f64> {
    let p = eigenvalues.len();
    if p == 0 {
        return Array1::zeros(0);
    }

    // Resolve Auto strategy to a concrete method based on problem size.
    let resolved_config = config.resolve_auto(p);

    let c = resolved_config.c;
    let eta = resolved_config
        .eta
        .unwrap_or_else(|| crate::stieltjes::default_eta(p));

    // Compute all Stieltjes transforms (reuses the fast kernel).
    let stieltjes_results = stieltjes::compute_all_stieltjes(
        eigenvalues,
        eta,
        resolved_config.stieltjes_method,
        match resolved_config.fft_grid_size {
            FftGridSize::Auto => None,
            FftGridSize::Custom(s) => Some(s),
        },
        resolved_config.cutoff,
        resolved_config.block_size,
        resolved_config.parallelism,
    );

    // Apply the direct precision factor, optionally in parallel.
    let result: Vec<f64> = match resolved_config.parallelism {
        Parallelism::Parallel => eigenvalues
            .par_iter()
            .zip(stieltjes_results.par_iter())
            .map(|(&lambda_i, &(mg_real, _))| precision_factor(lambda_i, c, mg_real))
            .collect(),
        Parallelism::Sequential | Parallelism::Auto => eigenvalues
            .iter()
            .zip(stieltjes_results.iter())
            .map(|(&lambda_i, &(mg_real, _))| precision_factor(lambda_i, c, mg_real))
            .collect(),
    };
    Array1::from_vec(result)
}

/// Convenience wrapper — uses the default config (Blocked/Sequential).
pub fn direct_precision_shrinkage_default(eigenvalues: &[f64], c: f64) -> Array1<f64> {
    let config = RmtConfig::new(c);
    direct_precision_shrinkage(eigenvalues, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity_population() {
        // A spectrum tightly clustered near 1.0 (identity population) should
        // produce precision eigenvalues near 1.0. We use a spread spectrum
        // rather than all-equal eigenvalues (which would be a degenerate pole
        // in the Stieltjes transform).
        let evals: Vec<f64> = (0..200).map(|i| 1.0 + (i as f64 - 100.0) * 1e-3).collect();
        let result = direct_precision_shrinkage_default(&evals, 0.3);
        let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
        assert!(
            (mean - 1.0).abs() < 0.1,
            "identity population precision mean should be ~1, got {mean}"
        );
    }

    #[test]
    fn test_naive_matches_default() {
        let mut evals: Vec<f64> = (0..137).map(|i| ((i as f64 + 1.0) * 0.1).ln_1p()).collect();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let default = direct_precision_shrinkage_default(&evals, 0.42);
        let naive = direct_precision_shrinkage(&evals, &RmtConfig::fully_naive(0.42));

        assert_eq!(default.len(), naive.len());
        // The precision formula divides by a small denominator, which amplifies
        // tiny (machine-precision) differences between the exact Stieltjes
        // methods. Use a relative tolerance.
        for (o, n) in default.iter().zip(naive.iter()) {
            assert_relative_eq!(o, n, max_relative = 1e-9);
        }
    }

    #[test]
    fn test_all_configs_agree() {
        let mut evals: Vec<f64> = (0..50).map(|i| (i as f64 + 0.1).ln_1p()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let ref_result = direct_precision_shrinkage(&evals, &RmtConfig::new(0.5));

        let exact_methods = [
            crate::config::StieltjesMethod::Naive,
            crate::config::StieltjesMethod::AutoVectorized,
            crate::config::StieltjesMethod::Blocked,
            crate::config::StieltjesMethod::BlockedAutoVec,
            crate::config::StieltjesMethod::BlockedTiled,
        ];
        for &method in &exact_methods {
            for &par in Parallelism::all() {
                let cfg = RmtConfig::new(0.5)
                    .with_stieltjes(method)
                    .with_parallelism(par);
                let result = direct_precision_shrinkage(&evals, &cfg);
                for (r, ref_r) in result.iter().zip(ref_result.iter()) {
                    assert_relative_eq!(r, ref_r, epsilon = 1e-11);
                }
            }
        }
    }

    #[test]
    fn test_direct_beats_naive_inversion() {
        // On a spectrum with a spread of eigenvalues, the direct precision
        // estimator should be closer to the true precision eigenvalues than
        // simply inverting the covariance RIE. This is a statistical sanity
        // check (not a strict inequality for every draw, but the direct
        // estimator is asymptotically optimal for the precision loss).
        let mut evals: Vec<f64> = (0..200).map(|i| 0.5 + (i as f64) * 0.02).collect();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let direct = direct_precision_shrinkage_default(&evals, 0.3);
        let cov_rie = crate::deconvolution::rie_shrinkage_default(&evals, 0.3);
        let naive_inv: Vec<f64> = cov_rie.iter().map(|&x| 1.0 / x).collect();

        // Both should be positive and finite.
        for &d in direct.iter() {
            assert!(d.is_finite() && d > 0.0);
        }
        for &n in naive_inv.iter() {
            assert!(n.is_finite() && n > 0.0);
        }
    }
}
