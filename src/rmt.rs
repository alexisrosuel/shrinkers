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
