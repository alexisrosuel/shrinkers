//! Crate root — required by Rust/Cargo.
//!
//! Module declarations live here because Rust resolves sibling modules
//! from the crate root. The actual implementation is in each module;
//! the crate-level public API documentation and top-level functions
//! are in [`rmt`].
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

// Private modules used internally
mod math;

// High-level items (reconstruct_covariance_basic, top-level docs) live in rmt.rs
mod rmt;

// Re-exports so that downstream code can use `shrinkers::RmtConfig` etc.
pub use config::*;
pub use deconvolution::*;
pub use rmt::*;
pub use spiked::*;
pub use stieltjes::stieltjes_term;

#[cfg(feature = "python")]
pub mod python;
