//! Fast RMT (Random Matrix Theory) Shrinkage Kernel — public API.
//!
//! This module re-exports items from submodules and provides crate-level
//! functionality that doesn't belong to a single submodule.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |---|---|
//! | [`deconvolution`](crate::deconvolution) | **Primary**: population eigenvalue estimation (spiked + bulk deconvolution) + pointwise RIE shrinkage |
//! | [`stieltjes`](crate::stieltjes) | Stieltjes transform (multiple algorithms) |
//! | [`eigenvector_overlaps`](crate::eigenvector_overlaps) | Theoretical eigenvector angular overlap (α²) |
//! | [`spiked`](crate::spiked) | Spiked covariance model: spike eigenvalue/eigenvector estimation |
//! | [`pipeline`](crate::pipeline) | Full covariance cleaning pipeline |
//! | [`math`](crate::math) | Manual complex number arithmetic |
//! | [`config`](crate::config) | Configuration (`RmtConfig`, strategies, enums) |
//! | [`python`](crate::python) | PyO3 bindings (feature-gated) |
//!
//! # Primary entry point
//!
//! [`deconvolve_spiked`](crate::deconvolution::deconvolve_spiked) is the
//! primary high-level entry point: given sample eigenvalues it detects spikes
//! (BEMA), debiases them (inverse BBP), removes them, and deconvolves the
//! remaining bulk (El Karoui) into a population spectral density.
//!
//! [`estimate_population_eigenvalues`](crate::deconvolution::estimate_population_eigenvalues)
//! is its eigenvalue-domain counterpart: instead of a density on a grid, it
//! returns a per-bulk-eigenvalue population estimate via Ledoit–Wolf / RIE
//! pointwise deconvolution, alongside the debiased spikes.
//!
//! # Theory
//!
//! For sample eigenvalues λ₁ … λₚ and concentration ratio c = p/n:
//!
//! ξ(λᵢ) = λᵢ / |1 - c + c·λᵢ·m_g(λᵢ - iη)|²
//!
//! m_g(z) = (1/p) Σⱼ 1/(z - λⱼ),   z = λᵢ - iη

/// Reconstruct a covariance matrix from eigenvectors and shrunk eigenvalues:
/// Σ_clean = U · diag(ξ(Λ)) · Uᵀ
///
/// This is the basic reconstruction without eigenvector angular overlap
/// correction. For the overlap-corrected version used by the cleaning
/// pipeline, use [`crate::pipeline::reconstruct_covariance`].
pub fn reconstruct_covariance_basic(
    eigenvectors: &ndarray::Array2<f64>,
    shrinked_eigenvalues: &ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    let scaled = eigenvectors * &shrinked_eigenvalues.view().insert_axis(ndarray::Axis(0));
    scaled.dot(&eigenvectors.t())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_reconstruct_covariance() {
        use ndarray::Array1;
        let diag = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let eigenvectors = ndarray::Array2::from_diag(&diag);
        let evals = vec![1.0, 2.0, 3.0];
        let shrunk = crate::deconvolution::rie_shrinkage_default(&evals, 0.5);
        let reconstructed = reconstruct_covariance_basic(&eigenvectors, &shrunk);
        for i in 0..3 {
            assert_relative_eq!(reconstructed[[i, i]], shrunk[i], epsilon = 1e-10);
        }
    }
}
