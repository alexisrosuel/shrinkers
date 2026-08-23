"""Type stubs for the shrinkers PyO3 extension module.

All functions return plain dicts; these TypedDicts document their exact
shapes for type checkers and IDEs.
"""

from typing import Literal, Optional, TypedDict, Union

import numpy as np

__version__: str

Method = Literal[
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
    "ewald",
    "dst",
    "auto",
]
Parallelism = Literal["seq", "sequential", "rayon", "parallel", "auto"]
Eta = Union[float, Literal["inferred"]]
Precision = Literal["f64", "f32"]
Cutoff = Union[float, None, Literal["inferred"]]


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
    method: Method = ...,
    parallelism: Parallelism = ...,
    cutoff: Cutoff = ...,
) -> DeconvolveSpikedResult: ...


def clean_correlation_matrix(
    correlation: np.ndarray, c: float
) -> CleanCorrelationMatrixResult: ...


def direct_precision_shrinkage(
    eigenvalues: np.ndarray, c: float
) -> DirectPrecisionShrinkageResult: ...


def stieltjes_transform(
    eigenvalues: np.ndarray,
    eta: Eta = ...,
    method: Method = ...,
    precision: Precision = ...,
    cutoff: Cutoff = ...,
    parallelism: Parallelism = ...,
) -> StieltjesTransformResult: ...


def detect_spikes_bema(
    eigenvalues: np.ndarray, c: float, margin: float = ...
) -> SpikeDetection: ...


def detect_spikes_tracy_widom(
    eigenvalues: np.ndarray,
    c: float,
    sigma2: float | None = ...,
    significance: float = ...,
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
    method: Method = ...,
    parallel: Parallelism = ...,
) -> np.ndarray: ...
