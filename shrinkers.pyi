"""Type stubs for the shrinkers PyO3 extension module.

Most functions return plain dicts documented by the TypedDicts below;
`inverse_bbp`, `ledoit_wolf_shrinkage` and `shrink_eigenvalues` return
`np.ndarray` / `float` directly.
"""

from typing import Literal, TypedDict

import numpy as np
from typing_extensions import TypeAlias

__version__: str

Method: TypeAlias = Literal[
    "naive",
    "autovec",
    "blocked",
    "blocked_autovec",
    "blocked_tiled",
    "blocked_windowed",
    "blocked_hybrid",
    "adaptive",
    "fft5",
    "fft3",
    "fft2",
    "fmm",
    "treecode",
    "chebcode",
    "chebyshev",
    "chebcode_fast",
    "chebf",
    "chebcode_xtreme",
    "chebx",
    "chebcode_balanced",
    "chebb",
    "hodlr",
    "ewald",
    "dst",
    "auto",
    "speed_auto",
    "speed",
    "accuracy_auto",
    "accuracy",
]
Eta: TypeAlias = float | Literal["inferred"] | None
Precision: TypeAlias = Literal["f64", "f32"]
Cutoff: TypeAlias = float | Literal["inferred"] | None


class BulkDeconvolution(TypedDict):
    """Result of the El Karoui bulk deconvolution."""

    lambda_grid: np.ndarray  # float64, shape (n_points,)
    density: np.ndarray  # population spectral density rho(lambda)
    w_re: np.ndarray  # Re[w] where w = z / a(z)
    sample_stieltjes_real: np.ndarray
    sample_stieltjes_imag: np.ndarray
    population_stieltjes_real: np.ndarray
    population_stieltjes_imag: np.ndarray


class DeconvolveSpikedResult(TypedDict):
    k: int
    spikes: np.ndarray  # population spikes l_i, descending
    spike_sample: np.ndarray  # sample spikes, descending
    bulk_edge: float  # lambda_+ = sigma^2 (1 + sqrt(gamma))^2
    sigma2: float  # estimated noise variance
    bulk: BulkDeconvolution


class CleanCorrelationMatrixResult(TypedDict):
    covariance: np.ndarray  # cleaned covariance matrix (p, p)
    eigenvectors: np.ndarray  # sample eigenvectors (p, p), descending columns
    eigenvalues: np.ndarray  # RIE-cleaned eigenvalues (p,), descending
    overlaps: np.ndarray  # squared angular overlaps alpha_i^2 (p,)
    sigma2: float


class DirectPrecisionShrinkageResult(TypedDict):
    precision_eigenvalues: np.ndarray  # direct precision delta_i (p,)


InverseMethod: TypeAlias = Literal["qis", "lis", "gis"]


class InverseNonlinearShrinkageResult(TypedDict):
    """Result of inverse nonlinear shrinkage (Ledoit & Wolf 2022)."""

    precision_eigenvalues: np.ndarray  # eigenvalues of Omega_hat (p,)
    covariance_eigenvalues: np.ndarray  # matching Sigma_hat eigenvalues (p,)
    smoothing: float  # Ledoit-Wolf bandwidth h


class PrecisionMatrixResult(TypedDict):
    """Result of `estimate_precision_matrix`."""

    precision: np.ndarray  # precision matrix estimate (p, p)
    eigenvectors: np.ndarray  # sample eigenvectors (p, p), descending columns
    precision_eigenvalues: np.ndarray  # Omega_hat eigenvalues, paired with columns
    covariance_eigenvalues: np.ndarray  # Sigma_hat eigenvalues, paired with columns
    smoothing: float  # Ledoit-Wolf bandwidth h


class StieltjesTransformResult(TypedDict):
    real: np.ndarray  # Re[S(lambda_i)] (p,)
    imag: np.ndarray  # Im[S(lambda_i)] (p,)


class SpikeDetection(TypedDict):
    k: int
    spike_indices: np.ndarray  # int64 indices into the ascending-sorted array
    bulk_edge: float
    sigma2: float


class AnalyzeSpikesResult(TypedDict):
    k: int
    spikes: np.ndarray  # population spikes l_i, descending
    overlaps: np.ndarray  # squared angular overlaps alpha_i^2 (len K)
    bulk_edge: float
    sigma2: float
    ledoit_wolf: np.ndarray  # raw LW estimates for all p eigenvalues


class EstimatePopulationEigenvaluesResult(TypedDict):
    k: int
    spikes: np.ndarray  # population spikes, descending
    spike_sample: np.ndarray  # sample spikes, descending
    bulk_edge: float
    sigma2: float
    bulk_population: np.ndarray  # per-bulk-eigenvalue estimates, ascending
    bulk_sample: np.ndarray  # ascending


def deconvolve_spiked(
    eigenvalues: np.ndarray,
    c: float,
    n_points: int = 200,
    eta: Eta = ...,
    margin: float = 1.0,
    *,
    method: Method = "auto",
    parallel: bool | None = False,
    cutoff: Cutoff = ...,
) -> DeconvolveSpikedResult: ...


def clean_correlation_matrix(
    correlation: np.ndarray, c: float
) -> CleanCorrelationMatrixResult: ...

def clean_correlation_matrix_complex(
    correlation: np.ndarray, c: float
) -> CleanCorrelationMatrixResult:
    """Clean a complex Hermitian correlation matrix (e.g. a spectral coherence
    matrix). Same estimator as ``clean_correlation_matrix``; the conjugate
    transpose replaces the transpose. ``covariance`` and ``eigenvectors`` come
    back complex."""
    ...


def direct_precision_shrinkage(
    eigenvalues: np.ndarray, c: float
) -> DirectPrecisionShrinkageResult: ...


def inverse_nonlinear_shrinkage(
    eigenvalues: np.ndarray,
    c: float,
    *,
    method: InverseMethod = "qis",
    parallel: bool | None = False,
) -> InverseNonlinearShrinkageResult: ...


def estimate_precision_matrix(
    covariance: np.ndarray,
    c: float,
    *,
    method: InverseMethod = "qis",
    parallel: bool | None = False,
) -> PrecisionMatrixResult: ...


def stieltjes_transform(
    eigenvalues: np.ndarray,
    eta: Eta = ...,
    method: Method = "blocked",
    precision: Precision = "f64",
    cutoff: Cutoff = ...,
    parallel: bool | None = False,
) -> StieltjesTransformResult: ...


class StieltjesWithDerivResult(TypedDict):
    """S and its analytic derivative dS/dx at every sample eigenvalue."""
    real: np.ndarray
    imag: np.ndarray
    deriv_real: np.ndarray
    deriv_imag: np.ndarray


def stieltjes_transform_with_deriv(
    eigenvalues: np.ndarray, eta: Eta = ...
) -> StieltjesWithDerivResult: ...


def detect_spikes_bema(
    eigenvalues: np.ndarray, c: float, margin: float = ...
) -> SpikeDetection: ...


def detect_spikes_tracy_widom(
    eigenvalues: np.ndarray,
    c: float,
    sigma2: Eta = ...,
    significance: float = 0.05,
) -> SpikeDetection: ...


def inverse_bbp(
    lambda_hat: float | np.ndarray, c: float, sigma2: float = ...
) -> float | np.ndarray: ...


def analyze_spikes(
    eigenvalues: np.ndarray, c: float, margin: float = ...
) -> AnalyzeSpikesResult: ...


def estimate_population_eigenvalues(
    eigenvalues: np.ndarray, c: float, margin: float = ...
) -> EstimatePopulationEigenvaluesResult: ...


def ledoit_wolf_shrinkage(eigenvalues: np.ndarray, c: float) -> np.ndarray: ...


def shrink_eigenvalues(
    eigenvalues: np.ndarray,
    c: float,
    *,
    method: Method = "auto",
    parallel: bool | None = False,
) -> np.ndarray: ...
