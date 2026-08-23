#!/usr/bin/env python
"""Example: population spectral reconstruction beats naive covariance.

Simulates high-dimensional covariance models (bulk + spikes) and shows that
`deconvolve_spiked` reconstructs the TRUE population spectral density far
better than the naive sample eigenvalue histogram.

This version:
  * sweeps several TRUE population-distribution cases (identity bulk, weak /
    strong / many spikes, structured bulk, pure bulk),
  * plays with the concentration ratio c = p/n across multiple values,
  * reports wall-clock timings for the `deconvolve_spiked` call.

Produces two figures:
  1. `figures/population_reconstruction.png` — true population density vs
     sample density vs reconstructed population density (one panel per case).
  2. `figures/reconstruction_error.png` — reconstruction error (density RMSE)
     of the naive sample density vs the estimated population density, plus a
     timing panel.

Run:  python scripts/example_population.py
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import shrinkers as rk

# ──────────────────────────────────────────────
#  Simulation helpers
# ──────────────────────────────────────────────

def build_pop_evals(p: int, spikes: list[float], bulk_type: str) -> np.ndarray:
    """Build the true population eigenvalues (descending) for a case.

    `bulk_type` controls the non-spike (bulk) part of the spectrum:
      - "identity"  : all bulk eigenvalues = 1
      - "two_block" : half at 0.5, half at 1.5
      - "uniform"   : spread uniformly over [0.5, 1.5]
    """
    n_spikes = len(spikes)
    n_bulk = p - n_spikes
    if bulk_type == "identity":
        bulk = np.ones(n_bulk)
    elif bulk_type == "two_block":
        n_lo = n_bulk // 2
        bulk = np.concatenate([np.full(n_lo, 0.5),
                               np.full(n_bulk - n_lo, 1.5)])
    elif bulk_type == "uniform":
        bulk = np.linspace(0.5, 1.5, n_bulk)
    else:
        raise ValueError(f"unknown bulk_type {bulk_type}")
    return np.concatenate([np.array(spikes, dtype=float), bulk])


def simulate(pop_evals: np.ndarray, n: int, seed: int = 0):
    """Simulate X (n x p) with population covariance having eigenvalues pop_evals.

    Returns (X, Sigma). Sigma = Q diag(pop_evals) Q^T with Q a random
    orthogonal matrix.
    """
    p = len(pop_evals)
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    Sigma = (Q * pop_evals) @ Q.T
    X = rng.standard_normal((n, p)) @ np.linalg.cholesky(Sigma).T
    return X, Sigma


def rmse(a, b):
    """Root mean squared error between two aligned vectors."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def grid_centers(grid: np.ndarray) -> np.ndarray:
    """Midpoints between consecutive grid points (length len(grid)-1)."""
    return 0.5 * (grid[:-1] + grid[1:])


def true_density(true_evals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Normalized histogram density of the true eigenvalues on `grid` centers."""
    hist, _ = np.histogram(true_evals, bins=grid, density=True)
    return hist


def reconstructed_density(res: dict, grid: np.ndarray) -> np.ndarray:
    """Reconstructed population density on `grid` centers from a deconvolve_spiked result.

    Bulk density is interpolated onto the grid centers; detected spikes are
    added as unit point masses at their reconstructed locations. The result
    is normalized to integrate to 1.
    """
    centers = grid_centers(grid)
    api_grid = np.asarray(res["bulk"]["lambda_grid"])
    api_density = np.asarray(res["bulk"]["density"])
    rec = np.interp(centers, api_grid, api_density, left=0.0, right=0.0)
    for s in np.asarray(res["spikes"]):
        idx = int(np.argmin(np.abs(centers - s)))
        rec[idx] += 1.0
    area = np.trapezoid(rec, centers)
    if area > 0:
        rec = rec / area
    return rec


def sample_density(sample_evals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Normalized histogram density of the sample eigenvalues on `grid` centers."""
    hist, _ = np.histogram(sample_evals, bins=grid, density=True)
    return hist


# ──────────────────────────────────────────────
#  Cases & parameters
# ──────────────────────────────────────────────

# (name, spikes, bulk_type)
CASES = [
    ("Identity bulk, 3 spikes",      [6.0, 4.0, 2.5],   "identity"),
    ("Identity bulk, weak spikes",   [3.0, 2.2, 1.8],   "identity"),
    ("Identity bulk, strong spikes", [10.0, 7.0, 5.0, 3.0], "identity"),
    ("Identity bulk, many spikes",   [8.0, 6.0, 5.0, 4.0, 3.0, 2.5], "identity"),
    ("Two-block bulk + spikes",      [6.0, 3.5],        "two_block"),
    ("Uniform bulk, no spikes",      [],                "uniform"),
]

C_VALUES = [0.25, 0.5, 0.75]   # concentration ratio p/n
P = 200                        # ambient dimension
REF_C = 0.5                    # c used for the reconstruction figure
N_POINTS = 300                 # deconvolution grid resolution
N_REPEAT = 3                   # timing repetitions


def main():
    os.makedirs("figures", exist_ok=True)

    # ── Run every case × c, collect results ──
    results = []  # list of dicts
    for ci, (name, spikes, bulk_type) in enumerate(CASES):
        for c in C_VALUES:
            n = int(round(P / c))
            seed = 100 * ci + 1
            pop_evals = build_pop_evals(P, spikes, bulk_type)
            X, Sigma = simulate(pop_evals, n, seed=seed)
            S = X.T @ X / n
            sample_evals = np.sort(np.linalg.eigvalsh(S))[::-1].copy()

            # Common grid covering true + sample range.
            lo = 0.0
            hi = max(pop_evals.max(), sample_evals.max()) * 1.05
            grid = np.linspace(lo, hi, N_POINTS)

            # Time the primary API call.
            times = []
            for _ in range(N_REPEAT):
                t0 = time.perf_counter()
                res = rk.deconvolve_spiked(sample_evals, c=c, n_points=N_POINTS)
                times.append(time.perf_counter() - t0)
            t_ms = 1000.0 * float(np.mean(times))

            # Reconstruction errors (density RMSE vs true density).
            naive_err = rmse(sample_density(sample_evals, grid),
                             true_density(pop_evals, grid))
            est_err = rmse(reconstructed_density(res, grid),
                           true_density(pop_evals, grid))

            results.append({
                "name": name, "spikes": spikes, "bulk_type": bulk_type,
                "c": c, "n": n, "k": res["k"],
                "sigma2": res["sigma2"], "bulk_edge": res["bulk_edge"],
                "est_spikes": np.asarray(res["spikes"]),
                "naive_err": naive_err, "est_err": est_err,
                "t_ms": t_ms, "grid": grid,
                "true_density": true_density(pop_evals, grid),
                "sample_density": sample_density(sample_evals, grid),
                "rec_density": reconstructed_density(res, grid),
            })

    # ── Console summary ──
    print(f"p={P}, deconvolution grid={N_POINTS} pts, "
          f"timing averaged over {N_REPEAT} runs\n")
    hdr = (f"{'case':<28} {'c':>5} {'n':>5} {'K':>3} "
           f"{'naiveRMSE':>10} {'estRMSE':>9} {'impr':>6} {'time(ms)':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        impr = r["naive_err"] / r["est_err"] if r["est_err"] > 0 else float("inf")
        print(f"{r['name']:<28} {r['c']:>5.2f} {r['n']:>5d} {r['k']:>3d} "
              f"{r['naive_err']:>10.4f} {r['est_err']:>9.4f} {impr:>6.2f} "
              f"{r['t_ms']:>9.2f}")
    print()

    # ── Figure 1: spectrum reconstruction (one panel per case, at REF_C) ──
    n_cases = len(CASES)
    ncols = 2
    nrows = int(np.ceil(n_cases / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, spikes, bulk_type) in zip(axes, CASES):
        r = next(x for x in results
                 if x["name"] == name and abs(x["c"] - REF_C) < 1e-9)
        g = grid_centers(r["grid"])
        ax.plot(g, r["true_density"], "k-", lw=2, label="True population")
        ax.plot(g, r["sample_density"], "r--", lw=1.5, alpha=0.7,
                label="Naive sample")
        ax.plot(g, r["rec_density"], "b-", lw=1.5, alpha=0.9,
                label="Estimated population")
        ax.set_title(f"{name}  (c={REF_C:.2f}, K={r['k']})", fontsize=10)
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[n_cases:]:
        ax.axis("off")
    fig.suptitle("Population spectral reconstruction via deconvolve_spiked",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("figures/population_reconstruction.png", dpi=150)
    plt.close(fig)
    print("Saved figures/population_reconstruction.png")

    # ── Figure 2: reconstruction error + timings ──
    fig, (ax_err, ax_time) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: reconstruction error vs c, per case (solid = estimated,
    # dashed = naive).
    for name, _, _ in CASES:
        rs = [x for x in results if x["name"] == name]
        cs = [x["c"] for x in rs]
        est = [x["est_err"] for x in rs]
        nav = [x["naive_err"] for x in rs]
        ax_err.plot(cs, est, "o-", lw=1.8, label=f"{name} (est)")
        ax_err.plot(cs, nav, "s--", lw=1.2, alpha=0.5)
    ax_err.set_xlabel("Concentration ratio c = p/n")
    ax_err.set_ylabel("Density RMSE vs true population")
    ax_err.set_title("Reconstruction error (solid=estimated, dashed=naive)")
    ax_err.grid(alpha=0.3)
    ax_err.legend(fontsize=7, ncol=2)

    # Right: timing vs c, per case.
    for name, _, _ in CASES:
        rs = [x for x in results if x["name"] == name]
        cs = [x["c"] for x in rs]
        ts = [x["t_ms"] for x in rs]
        ax_time.plot(cs, ts, "o-", lw=1.8, label=name)
    ax_time.set_xlabel("Concentration ratio c = p/n")
    ax_time.set_ylabel("Wall-clock time (ms)")
    ax_time.set_title(f"deconvolve_spiked timing (p={P}, avg of {N_REPEAT})")
    ax_time.grid(alpha=0.3)
    ax_time.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig("figures/reconstruction_error.png", dpi=150)
    plt.close(fig)
    print("Saved figures/reconstruction_error.png")


if __name__ == "__main__":
    main()
