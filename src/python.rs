//! PyO3 bindings for the RMT Shrinkage Kernel.
//!
//! The package's public entry points are:
//! - `deconvolve_spiked`: given the sample eigenvalues, applies the spiked +
//!   bulk cleaning via free-probability deconvolution. Spikes are detected
//!   (BEMA) and debiased (inverse BBP), then the remaining bulk is deconvolved
//!   with the El Karoui method to recover the population spectral density.
//! - `direct_precision_shrinkage`: direct precision-matrix eigenvalue
//!   shrinkage (Ledoit & Wolf 2020), without inverting a cleaned covariance.
//! - `clean_correlation_matrix`: given a sample correlation matrix, applies RIE
//!   eigenvalue shrinkage + eigenvector angular overlap correction and returns
//!   the cleaned covariance matrix together with the eigenvectors and their
//!   theoretical alignment with the population eigenvectors.
//! - `stieltjes_transform`: raw empirical Stieltjes transform with a choice of
//!   kernel, precision, far-field cutoff, and a parallel switch.
//!
//! # Threading & error behaviour
//!
//! - All heavy computation runs inside `py.detach`, so the GIL is
//!   released while the kernels run (Python threads stay responsive).
//! - Inputs are validated at the boundary: non-finite eigenvalues,
//!   non-positive spectra where positivity is required, and out-of-range
//!   concentration ratios raise `ValueError` instead of panicking.
//!
//! NOTE: numpy 0.29 bundles ndarray 0.16 internally while the project uses
//! ndarray 0.17. We bridge via `Vec<f64>` to avoid version mismatch.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::config::{CutoffConfig, Parallelism, RmtConfig, StieltjesMethod};
use crate::deconvolution::{
    deconvolve_spiked, direct_precision_shrinkage, estimate_population_eigenvalues, rie_shrinkage,
};
use crate::pipeline::clean_correlation_matrix;
use crate::spiked;

/// A parameter accepted as either a scalar float or a 1-D array of floats.
#[derive(Clone)]
enum FloatOrVec {
    Scalar(f64),
    Vector(Vec<f64>),
}

impl FloatOrVec {
    /// Map the elementwise function over the scalar/vector, returning a
    /// Python `float` or `np.ndarray` respectively.
    fn map(self, f: impl Fn(f64) -> f64 + Sync) -> Py<PyAny> {
        Python::attach(|py| match self {
            FloatOrVec::Scalar(x) => f(x)
                .into_pyobject(py)
                .expect("float conversion")
                .unbind()
                .into_any(),
            FloatOrVec::Vector(xs) => xs
                .iter()
                .map(|&x| f(x))
                .collect::<Vec<f64>>()
                .into_pyarray(py)
                .unbind()
                .into_any(),
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for FloatOrVec {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<f64>() {
            return Ok(FloatOrVec::Scalar(v));
        }
        let vec: Vec<f64> = ob
            .extract::<Vec<f64>>()
            .map_err(|_| PyValueError::new_err("expected a float or a 1-D array of floats"))?;
        require_finite(&vec, "lambda_hat")?;
        if vec.iter().any(|&v| v <= 0.0) {
            return Err(PyValueError::new_err(
                "lambda_hat must be positive (sample spikes cannot be ≤ 0)",
            ));
        }
        Ok(FloatOrVec::Vector(vec))
    }
}

// ──────────────────────────────────────────────
//  Helpers: validation + config parsing
// ──────────────────────────────────────────────

/// A parameter that is either an explicit value or the sentinel `"inferred"`,
/// meaning the default is computed at runtime (e.g. `eta = 0.1/sqrt(p)`).
///
/// The `Inferred` variant is a unit variant so it can be used directly as the
/// default expression in a `#[pyo3(signature = ...)]` annotation.
#[derive(Clone, Copy)]
enum InferredF64 {
    Value(f64),
    Inferred,
}

impl InferredF64 {
    fn value(self) -> Option<f64> {
        match self {
            InferredF64::Value(v) => Some(v),
            InferredF64::Inferred => None,
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for InferredF64 {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<f64>() {
            return Ok(InferredF64::Value(v));
        }
        // None is accepted as a synonym of "inferred": every kwarg of this
        // type documents "float or None/'inferred'", and None is what
        // callers naturally pass to mean "no explicit value".
        if ob.is_none() {
            return Ok(InferredF64::Inferred);
        }
        if let Ok(s) = ob.extract::<String>() {
            if s == "inferred" {
                return Ok(InferredF64::Inferred);
            }
        }
        Err(PyValueError::new_err(
            "expected a float, None, or the string 'inferred'",
        ))
    }
}

/// Reject NaN/±inf eigenvalues up front (they would panic in sorts or
/// propagate silently through the kernels).
fn require_finite(eigenvalues: &[f64], what: &str) -> PyResult<()> {
    for &v in eigenvalues {
        if !v.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{what} must be finite (found NaN or infinity)"
            )));
        }
    }
    Ok(())
}

/// Copy a 1-D read-only array into an owned `Vec<f64>`, rejecting
/// non-contiguous inputs (the kernels operate on flat slices).
fn owned_f64_vec(array: PyReadonlyArray1<'_, f64>, what: &str) -> PyResult<Vec<f64>> {
    array
        .as_slice()
        .map(|s| s.to_vec())
        .map_err(|_| PyValueError::new_err(format!("{what} must be contiguous")))
}

/// Resolve the `eta` argument: explicit float, or the crate-wide default
/// η = 0.1/√p for the sentinel `"inferred"`. Validates positivity.
fn validated_eta(eta: InferredF64, p: usize) -> PyResult<f64> {
    let v = eta
        .value()
        .unwrap_or_else(|| crate::stieltjes::default_eta(p));
    if !(v.is_finite() && v > 0.0) {
        return Err(PyValueError::new_err(format!(
            "eta must be a positive finite number, got {v}"
        )));
    }
    Ok(v)
}

/// Build the cutoff configuration from the Python-facing optional ratio,
/// validating positivity when enabled.
fn validated_cutoff(cutoff: InferredF64) -> PyResult<CutoffConfig> {
    Ok(match cutoff.value() {
        Some(ratio) => {
            if !(ratio.is_finite() && ratio > 0.0) {
                return Err(PyValueError::new_err(format!(
                    "cutoff ratio must be a positive finite number, got {ratio}"
                )));
            }
            CutoffConfig::Enabled { ratio }
        }
        None => CutoffConfig::Disabled,
    })
}

/// Require a positive spectrum, tolerating floating-point round-off.
///
/// Eigendecompositions of centered sample covariances routinely produce tiny
/// negative eigenvalues (~−1e-15·scale). Those are numerical dust, not data:
/// they are clamped to `0.0` in place. Anything meaningfully negative
/// (< −1e-10·scale) is rejected.
fn sanitize_positive_spectrum(eigenvalues: &mut [f64], what: &str) -> PyResult<()> {
    require_finite(eigenvalues, what)?;
    let scale = eigenvalues.iter().fold(1.0_f64, |acc, &v| acc.max(v.abs()));
    let tol = 1e-10 * scale;
    for v in eigenvalues.iter_mut() {
        if *v < 0.0 {
            if *v < -tol {
                return Err(PyValueError::new_err(format!(
                    "{what} must be non-negative (found {v}, tolerance {tol:.3e}); \
                     covariance eigenvalues cannot be significantly negative"
                )));
            }
            *v = 0.0; // round-off dust
        }
    }
    Ok(())
}

/// The Marchenko–Pastur-based estimators assume 0 < c ≤ 1.
fn require_concentration(c: f64) -> PyResult<()> {
    if !c.is_finite() || c <= 0.0 || c > 1.0 {
        return Err(PyValueError::new_err(format!(
            "c (concentration ratio p/n) must satisfy 0 < c <= 1, got {c}"
        )));
    }
    Ok(())
}

/// Parse a method string into a `StieltjesMethod`.
fn parse_method(method: &str) -> PyResult<StieltjesMethod> {
    Ok(match method {
        "naive" => StieltjesMethod::Naive,
        "autovec" => StieltjesMethod::AutoVectorized,
        "blocked" => StieltjesMethod::Blocked,
        "blocked_autovec" => StieltjesMethod::BlockedAutoVec,
        "blocked_tiled" => StieltjesMethod::BlockedTiled,
        "blocked_windowed" => StieltjesMethod::BlockedWindowed,
        "blocked_hybrid" => StieltjesMethod::BlockedHybrid,
        "adaptive" => StieltjesMethod::Adaptive,
        "fft5" => StieltjesMethod::Fft5,
        "fft3" => StieltjesMethod::Fft3,
        "fft2" => StieltjesMethod::Fft2,
        "fmm" | "treecode" => StieltjesMethod::TreeCode,
        "chebcode" | "chebyshev" => StieltjesMethod::ChebCode,
        "chebcode_fast" | "chebf" => StieltjesMethod::ChebCodeFast,
        "chebcode_xtreme" | "chebx" => StieltjesMethod::ChebCodeXtreme,
        "ewald" => StieltjesMethod::Ewald,
        "dst" => StieltjesMethod::Dst,
        "auto" => StieltjesMethod::Auto,
        "hodlr" => StieltjesMethod::Hodlr,
        "speed_auto" | "speed" => StieltjesMethod::SpeedAuto,
        "accuracy_auto" | "accuracy" => StieltjesMethod::AccuracyAuto,
        other => {
            return Err(PyValueError::new_err(format!("unknown method '{other}'")));
        }
    })
}

/// Map the user-facing `parallel` switch onto an execution mode.
///
/// `None` lets the library decide from the problem size, `Some(false)`
/// forces single-threaded execution, `Some(true)` forces multi-threaded
/// execution. The threading backend is an implementation detail and is
/// deliberately not part of the API.
fn parse_parallel(parallel: Option<bool>) -> Parallelism {
    match parallel {
        None => Parallelism::Auto,
        Some(false) => Parallelism::Sequential,
        Some(true) => Parallelism::Parallel,
    }
}

/// Build an `RmtConfig` from the parsed Python kwargs.
fn config_from_kwargs(
    c: f64,
    method: &str,
    parallel: Option<bool>,
    cutoff: InferredF64,
) -> PyResult<RmtConfig> {
    Ok(RmtConfig::new(c)
        .with_stieltjes(parse_method(method)?)
        .with_parallelism(parse_parallel(parallel))
        .with_cutoff(validated_cutoff(cutoff)?))
}

// ──────────────────────────────────────────────
//  deconvolve_spiked (hybrid spike + bulk deconvolution)
// ──────────────────────────────────────────────

/// Hybrid spiked deconvolution: detect spikes (BEMA), debias them (inverse
/// BBP), remove them, and deconvolve the remaining bulk with the El Karoui
/// method.
///
/// Args:
///   eigenvalues: sample eigenvalues (p,), any order; finite, non-negative
///     (tiny negative round-off is clamped to 0 at the boundary).
///   c: concentration ratio p/n, in (0, 1].
///   n_points: grid resolution for the bulk deconvolution (default 200).
///   eta: regularization for the bulk deconvolution; float or "inferred"
///     (default: 0.1 / sqrt(p)).
///   margin: multiplicative margin above the fitted bulk edge for spike
///     detection (default 1.0; slightly above 1.0 adds robustness).
///   method: Stieltjes kernel used for the bulk deconvolution —
///     exact family: "blocked", "blocked_tiled" (zero error);
///     treecode presets: "chebcode_fast"/"chebf" (~1e-8, fastest),
///       "chebcode" (~5e-10, default), "chebcode_xtreme"/"chebx" (~6e-13);
///     FFT family: "fft2", "fft3", "fft5" (~4e-5); also "fmm", "ewald",
///       "dst", "hodlr", "auto", "speed_auto", "accuracy_auto".
///     The three chebcode presets share one measured parameter set
///     (ChebPreset in the Rust API).
///   parallel: None lets the library pick (multi-core only where its
///     measured size thresholds say it pays off), False forces
///     single-threaded (default), True forces multi-core.
///   cutoff: far-field cutoff ratio (float) or None/"inferred" to disable.
///
/// Returns a dict with keys:
///   - "k": number of detected spikes
///   - "spikes": estimated population spike eigenvalues ℓ_i (descending)
///   - "spike_sample": the sample eigenvalues classified as spikes (descending)
///   - "bulk_edge": estimated bulk edge λ₊ = σ²(1+√γ)²
///   - "sigma2": estimated noise variance σ²
///   - "bulk": dict with the bulk deconvolution (same keys as
///     `spectral_deconvolution`)
#[pyfunction]
#[pyo3(
    name = "deconvolve_spiked",
    signature = (eigenvalues, c, n_points = 200, eta = InferredF64::Inferred, margin = 1.0, *, method = "auto", parallel = false, cutoff = InferredF64::Inferred)
)]
#[allow(clippy::too_many_arguments)]
fn deconvolve_spiked_py<'py>(
    py: Python<'py>,
    eigenvalues: PyReadonlyArray1<'py, f64>,
    c: f64,
    n_points: usize,
    eta: InferredF64,
    margin: f64,
    method: &str,
    parallel: Option<bool>,
    cutoff: InferredF64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let mut ev_vec = owned_f64_vec(eigenvalues, "eigenvalues")?;
    sanitize_positive_spectrum(&mut ev_vec, "eigenvalues")?;
    require_concentration(c)?;
    if n_points == 0 {
        return Err(PyValueError::new_err("n_points must be >= 1"));
    }

    // Validate margin early (BEMA multiplies the fitted edge by max(1, m)).
    if !margin.is_finite() || margin <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "margin must be a positive finite number, got {margin}"
        )));
    }

    let config = config_from_kwargs(c, method, parallel, cutoff)?;

    // Heavy computation runs without the GIL.
    let result =
        py.detach(|| deconvolve_spiked(&ev_vec, c, n_points, eta.value(), margin, &config));

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("k", result.k)?;
    dict.set_item("spikes", result.spikes.into_pyarray(py))?;
    dict.set_item("spike_sample", result.spike_sample.into_pyarray(py))?;
    dict.set_item("bulk_edge", result.bulk_edge)?;
    dict.set_item("sigma2", result.sigma2)?;
    dict.set_item("bulk", deconvolution_result_to_dict(py, &result.bulk)?)?;
    Ok(dict)
}

/// Helper: convert a `DeconvolutionResult` to a Python dict.
fn deconvolution_result_to_dict<'py>(
    py: Python<'py>,
    r: &crate::deconvolution::DeconvolutionResult,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("lambda_grid", r.lambda_grid.clone().into_pyarray(py))?;
    dict.set_item("density", r.density.clone().into_pyarray(py))?;
    dict.set_item("w_re", r.w_re.clone().into_pyarray(py))?;
    dict.set_item(
        "sample_stieltjes_real",
        r.sample_stieltjes_real.clone().into_pyarray(py),
    )?;
    dict.set_item(
        "sample_stieltjes_imag",
        r.sample_stieltjes_imag.clone().into_pyarray(py),
    )?;
    dict.set_item(
        "population_stieltjes_real",
        r.population_stieltjes_real.clone().into_pyarray(py),
    )?;
    dict.set_item(
        "population_stieltjes_imag",
        r.population_stieltjes_imag.clone().into_pyarray(py),
    )?;
    Ok(dict)
}

// ──────────────────────────────────────────────
//  direct_precision_shrinkage
// ──────────────────────────────────────────────

/// Direct precision matrix shrinkage (Direct Nonlinear Shrinkage).
///
/// Estimates the eigenvalues of the precision matrix $\Omega = \Sigma^{-1}$
/// directly, without inverting a cleaned covariance. This is the precision
/// counterpart of the RIE covariance shrinkage and is asymptotically optimal
/// for the precision loss.
///
/// Args:
///   eigenvalues: sample eigenvalues (p,), finite, non-negative.
///   c: concentration ratio p/n, in (0, 1].
///
/// Returns a dict with keys:
///   - "precision_eigenvalues": direct precision eigenvalues $\delta_i$ (p,)
#[pyfunction]
#[pyo3(name = "direct_precision_shrinkage", signature = (eigenvalues, c))]
fn direct_precision_shrinkage_py<'py>(
    py: Python<'py>,
    eigenvalues: PyReadonlyArray1<'py, f64>,
    c: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let mut ev_vec = owned_f64_vec(eigenvalues, "eigenvalues")?;
    sanitize_positive_spectrum(&mut ev_vec, "eigenvalues")?;
    require_concentration(c)?;

    let config = RmtConfig::new(c);
    let result = py.detach(|| direct_precision_shrinkage(&ev_vec, &config));

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("precision_eigenvalues", result.into_pyarray(py))?;
    Ok(dict)
}

// ──────────────────────────────────────────────
//  clean_correlation_matrix
// ──────────────────────────────────────────────

/// Clean a sample correlation matrix via RIE eigenvalue shrinkage + eigenvector
/// angular overlap correction.
///
/// Args:
///   correlation: sample correlation matrix (p, p), symmetric, finite.
///   c: concentration ratio p/n, in (0, 1].
///
/// Returns a dict with keys:
///   - "covariance": cleaned covariance matrix (p, p)
///   - "eigenvectors": sample eigenvectors (p, p), columns sorted descending
///   - "eigenvalues": RIE-cleaned eigenvalues (p,), descending
///   - "overlaps": squared angular overlaps α_i² = cos²(θ_i) (p,), parallel to
///     `eigenvalues` — the theoretical alignment of each sample eigenvector
///     with its (unknown) population counterpart
///   - "sigma2": estimated noise variance σ² (MP-median-corrected)
#[pyfunction]
#[pyo3(name = "clean_correlation_matrix", signature = (correlation, c))]
fn clean_correlation_matrix_py<'py>(
    py: Python<'py>,
    correlation: PyReadonlyArray2<'py, f64>,
    c: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let corr = correlation.as_array().to_owned();

    let (rows, cols) = corr.dim();
    if rows != cols {
        return Err(PyValueError::new_err("correlation must be a square matrix"));
    }
    for &v in corr.iter() {
        if !v.is_finite() {
            return Err(PyValueError::new_err(
                "correlation matrix must contain only finite values",
            ));
        }
    }
    require_concentration(c)?;

    let config = RmtConfig::new(c);
    // Eigen-decomposition + shrinkage run without the GIL.
    let result = py.detach(|| clean_correlation_matrix(&corr, c, &config));

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("covariance", result.covariance.into_pyarray(py))?;
    dict.set_item("eigenvectors", result.eigenvectors.into_pyarray(py))?;
    dict.set_item("eigenvalues", result.eigenvalues.into_pyarray(py))?;
    dict.set_item("overlaps", result.overlaps.into_pyarray(py))?;
    dict.set_item("sigma2", result.sigma2)?;
    Ok(dict)
}

// ──────────────────────────────────────────────
//  stieltjes_transform_with_deriv (values + analytic derivative)
// ──────────────────────────────────────────────

/// Compute the empirical Stieltjes transform S(λᵢ) = (1/p) Σⱼ 1/(λᵢ-λⱼ-iη)
/// for all eigenvalues, together with its analytic derivative
/// S'(λᵢ) = −(1/p) Σⱼ 1/(λᵢ-λⱼ-iη)² (derivative w.r.t. the real query
/// point) — exact auto-vectorized kernel, O(p²), sequential.
///
/// Args:
///   eigenvalues: sample eigenvalues (p,), finite.
///   eta: regularization parameter; float or "inferred"
///     (default 0.1/sqrt(p)).
///
/// Returns a dict with "real", "imag", "deriv_real" and "deriv_imag"
/// arrays (p,).
#[pyfunction]
#[pyo3(
    name = "stieltjes_transform_with_deriv",
    signature = (eigenvalues, eta = InferredF64::Inferred)
)]
fn stieltjes_transform_with_deriv_py<'py>(
    py: Python<'py>,
    eigenvalues: PyReadonlyArray1<'py, f64>,
    eta: InferredF64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let ev_vec = owned_f64_vec(eigenvalues, "eigenvalues")?;
    require_finite(&ev_vec, "eigenvalues")?;
    let p = ev_vec.len();
    if p == 0 {
        return Err(PyValueError::new_err("eigenvalues must be non-empty"));
    }
    let eta_val = validated_eta(eta, p)?;
    let (vals, derivs) =
        py.detach(|| crate::stieltjes::compute_all_stieltjes_with_deriv(&ev_vec, eta_val));
    let dict = pyo3::types::PyDict::new(py);
    for (key, v) in [
        ("real", vals.iter().map(|t| t.0).collect::<Vec<_>>()),
        ("imag", vals.iter().map(|t| t.1).collect::<Vec<_>>()),
        ("deriv_real", derivs.iter().map(|t| t.0).collect::<Vec<_>>()),
        ("deriv_imag", derivs.iter().map(|t| t.1).collect::<Vec<_>>()),
    ] {
        dict.set_item(key, v.into_pyarray(py))?;
    }
    Ok(dict)
}

#[pyfunction]
#[pyo3(
    name = "stieltjes_transform",
    signature = (eigenvalues, eta = InferredF64::Inferred, method = "blocked", precision = "f64", cutoff = InferredF64::Inferred, parallel = false)
)]
#[allow(clippy::too_many_arguments)]
fn stieltjes_transform_py<'py>(
    py: Python<'py>,
    eigenvalues: PyReadonlyArray1<'py, f64>,
    eta: InferredF64,
    method: &str,
    precision: &str,
    cutoff: InferredF64,
    parallel: Option<bool>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let ev_vec = owned_f64_vec(eigenvalues, "eigenvalues")?;
    require_finite(&ev_vec, "eigenvalues")?;
    let p = ev_vec.len();
    if p == 0 {
        return Err(PyValueError::new_err("eigenvalues must be non-empty"));
    }
    let eta_val = validated_eta(eta, p)?;

    let st_method = parse_method(method)?;
    let par = parse_parallel(parallel);
    let cutoff_cfg = validated_cutoff(cutoff)?;

    let dict = pyo3::types::PyDict::new(py);
    match precision {
        "f64" => {
            // Heavy computation runs without the GIL.
            let results = py.detach(|| {
                crate::stieltjes::compute_all_stieltjes(
                    &ev_vec, eta_val, st_method, None, cutoff_cfg, 64, par,
                )
            });
            let reals: Vec<f64> = results.iter().map(|(r, _)| *r).collect();
            let imags: Vec<f64> = results.iter().map(|(_, i)| *i).collect();
            dict.set_item("real", reals.into_pyarray(py))?;
            dict.set_item("imag", imags.into_pyarray(py))?;
        }
        "f32" => {
            // Dedicated single-precision kernel (~2× faster, ~1e-2 error).
            let ev32: Vec<f32> = ev_vec.iter().map(|&x| x as f32).collect();
            let eta32 = eta_val as f32;
            let results = py.detach(|| {
                crate::stieltjes::compute_all_stieltjes_f32(&ev32, eta32, st_method, cutoff_cfg)
            });
            let reals: Vec<f32> = results.iter().map(|(r, _)| *r).collect();
            let imags: Vec<f32> = results.iter().map(|(_, i)| *i).collect();
            dict.set_item("real", reals.into_pyarray(py))?;
            dict.set_item("imag", imags.into_pyarray(py))?;
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown precision '{other}' (expected 'f64' or 'f32')"
            )));
        }
    }
    Ok(dict)
}

// ──────────────────────────────────────────────
//  Spiked-model toolkit (detection / debiasing / analysis)
// ──────────────────────────────────────────────

/// Detect spikes via BEMA (Bulk Eigenvalue Matching Analysis).
///
/// Args:
///   eigenvalues: sample eigenvalues (p,), finite, non-negative. Any
///     order; sorted internally.
///   c: concentration ratio p/n, in (0, 1].
///   margin: multiplicative margin above the fitted bulk edge
///     (default 1.0; slightly above 1.0 adds robustness).
///
/// Returns a dict with keys "k", "spike_indices" (indices into the
/// ascending-sorted eigenvalue array), "bulk_edge", and "sigma2".
#[pyfunction]
#[pyo3(name = "detect_spikes_bema", signature = (eigenvalues, c, margin = 1.0))]
fn detect_spikes_bema_py<'py>(
    py: Python<'py>,
    eigenvalues: PyReadonlyArray1<'py, f64>,
    c: f64,
    margin: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let mut ev = owned_f64_vec(eigenvalues, "eigenvalues")?;
    sanitize_positive_spectrum(&mut ev, "eigenvalues")?;
    require_concentration(c)?;
    ev.sort_by(|a, b| a.partial_cmp(b).expect("validated finite"));

    let det = py.detach(|| spiked::detect_spikes_bema(&ev, c, margin));

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("k", det.k)?;
    dict.set_item("spike_indices", det.spike_indices.into_pyarray(py))?;
    dict.set_item("bulk_edge", det.bulk_edge)?;
    dict.set_item("sigma2", det.sigma2)?;
    Ok(dict)
}

/// Detect spikes by Tracy–Widom edge thresholding.
///
/// Args:
///   eigenvalues: sample eigenvalues (p,), finite, non-negative; any order.
///   c: concentration ratio p/n, in (0, 1].
///   sigma2: noise variance; float, or None/"inferred" to estimate it from
///     the data (MP-median-corrected median).
///   significance: upper-tail probability for the Tracy–Widom quantile
///     (default 0.05).
///
/// Returns a dict with keys "k", "spike_indices" (indices into the
/// ascending-sorted eigenvalue array), "bulk_edge", and "sigma2".
#[pyfunction]
#[pyo3(
    name = "detect_spikes_tracy_widom",
    signature = (eigenvalues, c, sigma2 = InferredF64::Inferred, significance = 0.05)
)]
fn detect_spikes_tracy_widom_py<'py>(
    py: Python<'py>,
    eigenvalues: PyReadonlyArray1<'py, f64>,
    c: f64,
    sigma2: InferredF64,
    significance: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let mut ev = owned_f64_vec(eigenvalues, "eigenvalues")?;
    sanitize_positive_spectrum(&mut ev, "eigenvalues")?;
    require_concentration(c)?;
    let sigma2 = sigma2.value();
    if !(significance.is_finite() && significance > 0.0 && significance < 1.0) {
        return Err(PyValueError::new_err(format!(
            "significance must be in (0, 1), got {significance}"
        )));
    }
    if let Some(s2) = sigma2 {
        if !(s2.is_finite() && s2 > 0.0) {
            return Err(PyValueError::new_err(format!(
                "sigma2 must be a positive finite number, got {s2}"
            )));
        }
    }
    ev.sort_by(|a, b| a.partial_cmp(b).expect("validated finite"));

    let det = py.detach(|| spiked::detect_spikes_tracy_widom(&ev, c, sigma2, significance));

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("k", det.k)?;
    dict.set_item("spike_indices", det.spike_indices.into_pyarray(py))?;
    dict.set_item("bulk_edge", det.bulk_edge)?;
    dict.set_item("sigma2", det.sigma2)?;
    Ok(dict)
}

/// Inverse BBP / DGJ: recover the population spike ℓ from a sample spike λ̂.
///
/// Accepts a scalar or a 1-D array of sample spikes (must be > 0) and returns
/// the same shape. Values at or below the BBP threshold are returned as the
/// bulk edge (the spike is not resolvable).
#[pyfunction]
#[pyo3(name = "inverse_bbp", signature = (lambda_hat, c, sigma2 = 1.0))]
fn inverse_bbp_py(lambda_hat: FloatOrVec, c: f64, sigma2: f64) -> PyResult<Py<PyAny>> {
    require_concentration(c)?;
    if !(sigma2.is_finite() && sigma2 > 0.0) {
        return Err(PyValueError::new_err(format!(
            "sigma2 must be a positive finite number, got {sigma2}"
        )));
    }
    Ok(lambda_hat.map(|x| spiked::inverse_bbp(x, c, sigma2)))
}

/// Full spiked-model analysis: BEMA detection + inverse-BBP debiasing +
/// BBP angle overlaps + Ledoit–Wolf shrinkage of the full spectrum.
///
/// Returns a dict with keys "k", "spikes" (population, descending),
/// "overlaps" (squared angular overlaps α_i² for the spikes),
/// "bulk_edge", "sigma2", and "ledoit_wolf" (population estimates for all
/// p eigenvalues, input order).
#[pyfunction]
#[pyo3(name = "analyze_spikes", signature = (eigenvalues, c, margin = 1.0))]
fn analyze_spikes_py<'py>(
    py: Python<'py>,
    eigenvalues: PyReadonlyArray1<'py, f64>,
    c: f64,
    margin: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let mut ev = owned_f64_vec(eigenvalues, "eigenvalues")?;
    sanitize_positive_spectrum(&mut ev, "eigenvalues")?;
    require_concentration(c)?;

    let config = RmtConfig::new(c);
    let res = py.detach(|| spiked::analyze_spikes(&ev, c, &config, margin));

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("k", res.k)?;
    dict.set_item("spikes", res.spikes.into_pyarray(py))?;
    dict.set_item("overlaps", res.overlaps.into_pyarray(py))?;
    dict.set_item("bulk_edge", res.bulk_edge)?;
    dict.set_item("sigma2", res.sigma2)?;
    dict.set_item("ledoit_wolf", res.ledoit_wolf.into_pyarray(py))?;
    Ok(dict)
}

/// Estimate the population eigenvalues from sample eigenvalues.
///
/// Detects spikes (BEMA), debiases them (inverse BBP), and maps every
/// remaining bulk eigenvalue to its population estimate via Ledoit–Wolf /
/// RIE pointwise deconvolution.
///
/// Returns a dict with keys "k", "spikes" (population, descending),
/// "spike_sample" (descending), "bulk_edge", "sigma2",
/// "bulk_population" (per-bulk-eigenvalue population estimates, ascending),
/// and "bulk_sample" (ascending).
#[pyfunction]
#[pyo3(
    name = "estimate_population_eigenvalues",
    signature = (eigenvalues, c, margin = 1.0)
)]
fn estimate_population_eigenvalues_py<'py>(
    py: Python<'py>,
    eigenvalues: PyReadonlyArray1<'py, f64>,
    c: f64,
    margin: f64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let mut ev = owned_f64_vec(eigenvalues, "eigenvalues")?;
    sanitize_positive_spectrum(&mut ev, "eigenvalues")?;
    require_concentration(c)?;

    let config = RmtConfig::new(c);
    let res = py.detach(|| estimate_population_eigenvalues(&ev, c, margin, &config));

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("k", res.k)?;
    dict.set_item("spikes", res.spikes.into_pyarray(py))?;
    dict.set_item("spike_sample", res.spike_sample.into_pyarray(py))?;
    dict.set_item("bulk_edge", res.bulk_edge)?;
    dict.set_item("sigma2", res.sigma2)?;
    dict.set_item("bulk_population", res.bulk_population.into_pyarray(py))?;
    dict.set_item("bulk_sample", res.bulk_sample.into_pyarray(py))?;
    Ok(dict)
}

/// Raw Ledoit–Wolf non-linear shrinkage ξ(λᵢ) for every eigenvalue
/// (NOT trace-rescaled — see `shrink_eigenvalues` for the trace-preserving
/// variant).
#[pyfunction]
#[pyo3(name = "ledoit_wolf_shrinkage", signature = (eigenvalues, c))]
fn ledoit_wolf_shrinkage_py<'py>(
    py: Python<'py>,
    eigenvalues: PyReadonlyArray1<'py, f64>,
    c: f64,
) -> PyResult<Py<PyAny>> {
    let mut ev = owned_f64_vec(eigenvalues, "eigenvalues")?;
    sanitize_positive_spectrum(&mut ev, "eigenvalues")?;
    require_concentration(c)?;

    let config = RmtConfig::new(c);
    let result = py.detach(|| spiked::ledoit_wolf_shrinkage(&ev, &config));
    Ok(result.into_pyarray(py).unbind().into_any())
}

/// Trace-preserving RIE non-linear shrinkage: ξ(λᵢ) with the trace rescaled
/// to match the original spectrum.
#[pyfunction]
#[pyo3(
    name = "shrink_eigenvalues",
    signature = (eigenvalues, c, *, method = "auto", parallel = false)
)]
fn shrink_eigenvalues_py<'py>(
    py: Python<'py>,
    eigenvalues: PyReadonlyArray1<'py, f64>,
    c: f64,
    method: &str,
    parallel: Option<bool>,
) -> PyResult<Py<PyAny>> {
    let ev = owned_f64_vec(eigenvalues, "eigenvalues")?;
    require_finite(&ev, "eigenvalues")?;
    // A covariance spectrum may legitimately contain zeros after centering;
    // only finiteness is required here.
    require_concentration(c)?;

    let config = config_from_kwargs(c, method, parallel, InferredF64::Inferred)?;
    let result = py.detach(|| rie_shrinkage(&ev, &config));
    Ok(result.into_pyarray(py).unbind().into_any())
}

// ──────────────────────────────────────────────
//  Python module definition
// ──────────────────────────────────────────────

#[pymodule]
fn shrinkers(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(deconvolve_spiked_py, m)?)?;
    m.add_function(wrap_pyfunction!(direct_precision_shrinkage_py, m)?)?;
    m.add_function(wrap_pyfunction!(clean_correlation_matrix_py, m)?)?;
    m.add_function(wrap_pyfunction!(stieltjes_transform_py, m)?)?;
    m.add_function(wrap_pyfunction!(stieltjes_transform_with_deriv_py, m)?)?;
    m.add_function(wrap_pyfunction!(detect_spikes_bema_py, m)?)?;
    m.add_function(wrap_pyfunction!(detect_spikes_tracy_widom_py, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_bbp_py, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_spikes_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_population_eigenvalues_py, m)?)?;
    m.add_function(wrap_pyfunction!(ledoit_wolf_shrinkage_py, m)?)?;
    m.add_function(wrap_pyfunction!(shrink_eigenvalues_py, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__doc__",
        "Fast RMT shrinkage kernel. Spiked + bulk eigenvalue cleaning via free-probability deconvolution.",
    )?;
    Ok(())
}
