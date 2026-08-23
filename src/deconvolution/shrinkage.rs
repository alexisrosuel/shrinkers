//! Pointwise eigenvalue shrinkage — the RIE special case of deconvolution.
//!
//! The RIE (Rotationally Invariant Estimator) non-linear shrinkage is the
//! *pointwise* special case of the free-probability deconvolution: it recovers
//! the population counterpart of each individual sample eigenvalue via the
//! Stieltjes transform, whereas the full deconvolution
//! ([`super::spectral_deconvolution`]) recovers the whole population density.
//!
//! This module hosts the shrinkage formula and the trace-preserving rescale.
//! It lives under `deconvolution` because RIE is a special case of it — there
//! is no separate "RIE" module anymore.

use ndarray::Array1;
use rayon::prelude::*;

use crate::config::{FftGridSize, Parallelism, RmtConfig};
use crate::stieltjes;

/// Compute the shrinkage factor for a single eigenvalue.
///
/// ξ(λᵢ) = λᵢ / |1 - c + c·λᵢ·(m_g.real + i·m_g.imag)|²
#[inline(always)]
pub fn shrinkage_factor(lambda_i: f64, c: f64, m_g_real: f64, m_g_imag: f64) -> f64 {
    let term_real = c * lambda_i * m_g_real;
    let term_imag = c * lambda_i * m_g_imag;
    let denom_real = 1.0 - c + term_real;
    let denom_imag = term_imag;
    let denom_norm_sq = denom_real * denom_real + denom_imag * denom_imag;

    if denom_norm_sq > 0.0 {
        lambda_i / denom_norm_sq
    } else {
        lambda_i
    }
}

/// Compute the trace-preserving scale factor.
#[inline(always)]
pub fn trace_scale_factor(original_trace: f64, shrinked_trace: f64) -> f64 {
    if shrinked_trace > 0.0 {
        original_trace / shrinked_trace
    } else {
        1.0
    }
}

/// Compute the RIE non-linear shrinkage with full configuration control.
///
/// This is the pointwise special case of the deconvolution: it maps each
/// sample eigenvalue to its population estimate via the Stieltjes transform,
/// then rescales to preserve the trace.
///
/// # Arguments
///
/// * `eigenvalues` — Slice of sample eigenvalues (length p), should be sorted
/// * `config` — `RmtConfig` controlling all optimization settings
///
/// # Returns
///
/// `Array1<f64>` — Shrinked eigenvalues ξ(λᵢ)
pub fn rie_shrinkage(eigenvalues: &[f64], config: &RmtConfig) -> Array1<f64> {
    let p = eigenvalues.len();
    if p == 0 {
        return Array1::zeros(0);
    }

    // Resolve Auto strategy to a concrete method based on problem size
    let resolved_config = config.resolve_auto(p);

    let c = resolved_config.c;
    let eta = resolved_config
        .eta
        .unwrap_or_else(|| 0.1 / (p as f64).sqrt());
    let original_trace: f64 = eigenvalues.iter().copied().sum();

    // Compute all Stieltjes transforms
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

    // Apply shrinkage factor, optionally in parallel
    let shrinked: Vec<f64> = match resolved_config.parallelism {
        Parallelism::Rayon => eigenvalues
            .par_iter()
            .zip(stieltjes_results.par_iter())
            .map(|(&lambda_i, &(mg_real, mg_imag))| shrinkage_factor(lambda_i, c, mg_real, mg_imag))
            .collect(),
        Parallelism::Sequential | Parallelism::Auto => eigenvalues
            .iter()
            .zip(stieltjes_results.iter())
            .map(|(&lambda_i, &(mg_real, mg_imag))| shrinkage_factor(lambda_i, c, mg_real, mg_imag))
            .collect(),
    };

    // Trace preservation
    let shrinked_trace: f64 = shrinked.iter().copied().sum();
    let scale = trace_scale_factor(original_trace, shrinked_trace);

    Array1::from_vec(shrinked).mapv(|val| val * scale)
}

/// Convenience wrapper — uses the default config (Blocked/Sequential).
pub fn rie_shrinkage_default(eigenvalues: &[f64], c: f64) -> Array1<f64> {
    let config = RmtConfig::new(c);
    rie_shrinkage(eigenvalues, &config)
}

/// Fully naive version (all optimizations off) for benchmarking.
pub fn rie_shrinkage_naive(eigenvalues: &[f64], c: f64) -> Array1<f64> {
    let config = RmtConfig::fully_naive(c);
    rie_shrinkage(eigenvalues, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StieltjesMethod;
    use approx::assert_relative_eq;

    #[test]
    fn test_trivial_case() {
        let evals = vec![1.0, 1.0, 1.0];
        let result = rie_shrinkage_default(&evals, 0.5);
        for &v in result.iter() {
            assert_relative_eq!(v, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_trace_preservation() {
        let evals = vec![0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
        let result = rie_shrinkage_default(&evals, 0.3);
        let original_sum: f64 = evals.iter().sum();
        let result_sum: f64 = result.iter().sum();
        assert_relative_eq!(original_sum, result_sum, epsilon = 1e-10);
    }

    #[test]
    fn test_naive_matches_default() {
        let mut evals: Vec<f64> = (0..137).map(|i| ((i as f64 + 1.0) * 0.1).ln_1p()).collect();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let default = rie_shrinkage_default(&evals, 0.42);
        let naive = rie_shrinkage_naive(&evals, 0.42);

        assert_eq!(default.len(), naive.len());
        for (o, n) in default.iter().zip(naive.iter()) {
            assert_relative_eq!(o, n, epsilon = 1e-13);
        }
    }

    #[test]
    fn test_all_configs_agree() {
        let mut evals: Vec<f64> = (0..50).map(|i| (i as f64 + 0.1).ln_1p()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let ref_result = rie_shrinkage(&evals, &RmtConfig::new(0.5));

        // Compare all exact O(p²) methods — they should match exactly
        let exact_methods = [
            StieltjesMethod::Naive,
            StieltjesMethod::AutoVectorized,
            StieltjesMethod::Blocked,
            StieltjesMethod::BlockedAutoVec,
            StieltjesMethod::BlockedTiled,
        ];
        for &method in &exact_methods {
            for &par in Parallelism::all() {
                let cfg = RmtConfig::new(0.5)
                    .with_stieltjes(method)
                    .with_parallelism(par);
                let result = rie_shrinkage(&evals, &cfg);
                for (r, ref_r) in result.iter().zip(ref_result.iter()) {
                    assert_relative_eq!(r, ref_r, epsilon = 1e-11);
                }
            }
        }

        // FFT and TreeCode methods use approximations — verify finite + trace
        let approx_methods = [
            StieltjesMethod::Fft5,
            StieltjesMethod::Fft3,
            StieltjesMethod::TreeCode,
            StieltjesMethod::ChebCode,
        ];
        for &method in &approx_methods {
            for &par in Parallelism::all() {
                let cfg = RmtConfig::new(0.5)
                    .with_stieltjes(method)
                    .with_parallelism(par);
                let result = rie_shrinkage(&evals, &cfg);
                for &v in result.iter() {
                    assert!(v.is_finite(), "Non-finite result for {:?}", method);
                }
                let orig_sum: f64 = evals.iter().sum();
                let res_sum: f64 = result.iter().sum();
                assert_relative_eq!(orig_sum, res_sum, epsilon = 1e-10);
            }
        }
    }
}
