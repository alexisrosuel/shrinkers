"""Shared helpers for the analysis, plotting and measurement scripts.

Deliberately small: only helpers that were duplicated across three or more
scripts live here, and each keeps the exact arithmetic of the copies it
replaces (same operations in the same order — see the docstrings) so the
measured numbers in `docs/pareto/` stay comparable. Import as
``from _common import ...`` — the scripts are run from the repository root, so
``scripts/`` is on ``sys.path``.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS_IMG = REPO_ROOT / "docs" / "img"
DOCS_PARETO = REPO_ROOT / "docs" / "pareto"
FIGURES = REPO_ROOT / "figures"


def mp_spectrum(p, c=0.5, seed=42):
    """Deterministic Marchenko-Pastur-like spectrum (ascending, length p).

    Uniform bulk on `[max(1-sqrt(c), 0.01)^2, (1+sqrt(c))^2]` plus a small
    uniform jitter — the recipe the benchmark scripts have always used, kept
    verbatim so recorded numbers stay comparable.
    """
    rng = np.random.default_rng(seed)
    lambda_min = max(1.0 - np.sqrt(c), 0.01) ** 2
    lambda_max = (1.0 + np.sqrt(c)) ** 2
    u = rng.uniform(0, 1, p)
    t = lambda_min + u * (lambda_max - lambda_min)
    return np.sort(t + rng.uniform(0, 0.1, p))


def numpy_stieltjes(evals, eta, block=None):
    """Reference NumPy Stieltjes transform `m_g(z)`, `z = lambda - i*eta`.

    Returns `(real, imag)`, each already scaled by `1/p`. `block=None` is the
    plain p x p broadcast used by the small-p scripts; `block=k` is the
    chunked form that never materialises the p x p intermediate (identical
    arithmetic and identical rounding — the chunked path was already written
    this way in `make_readme_figures.py`).
    """
    ev = np.asarray(evals, dtype=np.float64)
    if block is None:
        diff = ev[:, None] - ev[None, :]
        denom = diff * diff + eta * eta
        return np.mean(diff / denom, axis=1), np.mean(eta / denom, axis=1)

    p = ev.shape[0]
    re = np.empty(p)
    im = np.empty(p)
    eta2 = eta * eta
    inv_p = 1.0 / p
    for a in range(0, p, block):
        d = ev[a:a + block, None] - ev[None, :]
        inv = 1.0 / (d * d + eta2)
        re[a:a + block] = (d * inv).sum(axis=1) * inv_p
        im[a:a + block] = (eta * inv).sum(axis=1) * inv_p
    return re, im


def simulate_spiked(p, n, spikes, seed=0):
    """Sample `X` (n x p) from a spiked covariance model.

    `Sigma = I_p + sum_i (ell_i - 1) v_i v_i^T` with unit-norm spike
    directions. Returns `(X, Sigma, true_spikes)`; callers that only need the
    data unpack `X, Sigma, _`.
    """
    rng = np.random.default_rng(seed)
    sigma = np.eye(p)
    for ell in spikes:
        v = rng.standard_normal(p)
        v /= np.linalg.norm(v)
        sigma += (ell - 1.0) * np.outer(v, v)
    x = rng.standard_normal((n, p)) @ np.linalg.cholesky(sigma).T
    return x, sigma, np.array(spikes)


def bench_us(fn, *args, n_runs):
    """Mean wall time of one call, in microseconds (one warm-up run)."""
    fn(*args)  # warm-up
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(*args)
    return (time.perf_counter() - t0) / n_runs * 1e6


def setup_mpl():
    """Select the headless Agg backend and return ``matplotlib.pyplot``.

    pyplot must be imported *after* ``use("Agg")`` for the scripts to work on
    a headless machine; centralising the sequence removes the repeated
    import-after-use dance (and the ``# noqa: E402`` it required).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def savefig(fig, path, dpi=150):
    """Write ``fig`` to ``path`` (creating parent directories) and close it."""
    import matplotlib.pyplot as plt

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {path}")
