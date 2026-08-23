"""
rie_core.py — Core RIE (Rotationally Invariant Estimator) eigenvalue shrinkage.

This module is the standalone RIE engine. It only performs eigenvalue shrinkage:
given a sorted array of sample eigenvalues and a concentration ratio c = N/T,
it returns shrunk eigenvalues preserving the total trace.

No dependency on eigenvectors, data matrices, or covariance reconstruction.
This is the "pure RIE" core that can be called independently.
"""

import numpy as np


def rie_shrinkage(evals: np.ndarray, c: float, eta: float | None = None) -> np.ndarray:
    """Apply RIE non-linear shrinkage to a sorted array of eigenvalues.

    Implements the Rotationally Invariant Estimator (RIE) from
    Bun, Bouchaud & Potters, "Cleaning large Correlation Matrices" (2017).

    The Stieltjes transform is computed via O(p²) broadcasting (exact).

    Parameters
    ----------
    evals : np.ndarray, shape (p,)
        Sorted sample eigenvalues (descending or ascending — internally re-sorted).
    c : float
        Concentration ratio N / T (number of features / number of observations).
    eta : float, optional
        Regularization (imaginary shift) for the Stieltjes transform.
        Default: 0.1 / sqrt(p).

    Returns
    -------
    lambda_rie : np.ndarray, shape (p,)
        Shrunk eigenvalues, sorted descending, with total trace preserved.
    """
    p = len(evals)
    eta = 0.1 / np.sqrt(p) if eta is None else eta
    original_trace = np.sum(evals)

    # Ensure descending sort
    if evals[0] < evals[-1]:
        evals = evals[::-1]

    # ── Stieltjes transform via broadcasting ──
    diff = evals[:, np.newaxis] - evals[np.newaxis, :]  # (p, p)
    denom = diff * diff + eta * eta                      # (p, p)
    mg_real = np.mean(diff / denom, axis=1)               # (p,)
    mg_imag = np.mean(eta / denom, axis=1)                # (p,)

    # ── Shrinkage factor ──
    term_real = c * evals * mg_real
    term_imag = c * evals * mg_imag
    denom_real = 1.0 - c + term_real
    denom_imag = term_imag
    denom_norm_sq = denom_real * denom_real + denom_imag * denom_imag

    lambda_rie = np.where(denom_norm_sq > 0.0, evals / denom_norm_sq, evals)

    # ── Trace-preserving rescaling ──
    shrunk_trace = np.sum(lambda_rie)
    if shrunk_trace > 0.0:
        lambda_rie *= original_trace / shrunk_trace

    return lambda_rie


def rie_shrinkage_with_rust(
    evals: np.ndarray,
    c: float,
    method: str = "blocked",
    parallel: bool = False,
    **kwargs,
) -> np.ndarray:
    """Apply RIE shrinkage using the Rust shrinkers (PyO3) backend.

    Falls back to pure NumPy if shrinkers is not installed.

    Parameters
    ----------
    evals : np.ndarray, shape (p,)
        Sorted sample eigenvalues.
    c : float
        Concentration ratio N / T.
    method : str
        Stieltjes method: "blocked", "autovec", "fft5", "fft3", "fft2", "fmm".
    parallel : bool
        Whether to use Rayon parallelism.
    **kwargs : dict
        Additional kwargs forwarded to shrinkers.shrink_eigenvalues.

    Returns
    -------
    lambda_rie : np.ndarray, shape (p,)
        Shrunk eigenvalues.
    """
    try:
        import shrinkers
        return shrinkers.shrink_eigenvalues(
            evals, c, method=method, parallel=parallel, **kwargs,
        )
    except ImportError:
        import warnings
        warnings.warn("shrinkers not installed, falling back to pure NumPy RIE")
        return rie_shrinkage(evals, c)