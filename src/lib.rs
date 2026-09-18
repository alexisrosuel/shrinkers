//! Fast RMT (Random Matrix Theory) Shrinkage Kernel — public API.
//!
//! Module declarations live here because Rust resolves sibling modules from
//! the crate root; the implementations live in the modules below.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |---|---|
//! | [`deconvolution`] | **Primary**: population eigenvalue estimation (spiked + bulk deconvolution) + pointwise RIE shrinkage |
//! | [`stieltjes`] | Stieltjes transform (multiple algorithms) |
//! | [`eigenvector_overlaps`] | Theoretical eigenvector angular overlap (α²) |
//! | [`spiked`] | Spiked covariance model: spike eigenvalue/eigenvector estimation |
//! | [`pipeline`] | Full covariance cleaning pipeline |
//! | [`config`] | Configuration (`RmtConfig`, strategies, enums) |
//! | `python` | PyO3 bindings (only with the `python` feature, so unlinked here) |
//!
//! # Primary entry point
//!
//! [`deconvolution::deconvolve_spiked`] is the primary high-level entry point:
//! given sample eigenvalues it detects spikes (BEMA), debiases them (inverse
//! BBP), removes them, and deconvolves the remaining bulk (El Karoui) into a
//! population spectral density.
//!
//! [`deconvolution::estimate_population_eigenvalues`] is its eigenvalue-domain
//! counterpart: instead of a density on a grid, it returns a per-bulk-
//! eigenvalue population estimate via Ledoit–Wolf / RIE pointwise
//! deconvolution, alongside the debiased spikes.
//!
//! # Theory
//!
//! For sample eigenvalues λ₁ … λₚ and concentration ratio c = p/n:
//!
//! ξ(λᵢ) = λᵢ / |1 - c + c·λᵢ·m_g(λᵢ - iη)|²
//!
//! m_g(z) = (1/p) Σⱼ 1/(z - λⱼ),   z = λᵢ - iη
//!
//! # Input contracts
//!
//! The numerical kernels in this crate make the following assumptions about
//! their inputs. They are enforced (with `ValueError`s) at the Python
//! boundary; Rust callers must uphold them themselves:
//!
//! - All eigenvalues must be **finite** (`NaN`/`±inf` are unsupported and may
//!   panic in sorts or propagate silently).
//! - Eigenvalues of covariance/correlation spectra are expected to be
//!   **non-negative**. Tiny negative round-off (≥ −1e-10·scale) is clamped
//!   to zero at the Python boundary; meaningfully negative values raise.
//! - The concentration ratio `c = p/n` must lie in **(0, 1]**; the
//!   Marchenko–Pastur-based estimators clamp or misbehave outside this range.

pub mod config;
pub mod deconvolution;
pub mod eigenvector_overlaps;
pub mod pipeline;
pub mod spiked;
pub mod stieltjes;

// Re-exports so that downstream code can use `shrinkers::RmtConfig` etc.
pub use config::*;
pub use deconvolution::*;
pub use spiked::*;
pub use stieltjes::stieltjes_term;

#[cfg(feature = "python")]
pub mod python;
