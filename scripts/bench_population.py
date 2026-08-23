#!/usr/bin/env python
"""Benchmark the primary API call `estimate_population_eigenvalues`.

Identifies which stage of the pipeline costs the most compute time by timing
each stage separately:

  1. Spike detection (BEMA)          -> detect_spikes_bema
  2. Spike debiasing (inverse BBP)   -> inverse_bbp
  3. Bulk deconvolution (Ledoit-Wolf)-> ledoit_wolf_shrinkage

and compares against the full `estimate_population_eigenvalues` call.

Run:  python scripts/bench_population.py
"""

import time

import numpy as np

import shrinkers as rk


def simulate_spiked(p: int, n: int, spikes: list[float], seed: int = 0):
    """Simulate X (n x p) from a spiked covariance model."""
    rng = np.random.default_rng(seed)
    Sigma = np.eye(p)
    for ell in spikes:
        v = rng.standard_normal(p)
        v /= np.linalg.norm(v)
        Sigma += (ell - 1.0) * np.outer(v, v)
    X = rng.standard_normal((n, p)) @ np.linalg.cholesky(Sigma).T
    return X, Sigma


def timeit(fn, *args, n_runs=5):
    """Return mean wall time in ms over n_runs (with warmup)."""
    fn(*args)  # warmup
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)  # ms
    return float(np.mean(times))


def main():
    cases = [
        (200, 500, [6.0, 4.0, 2.5]),
        (500, 1000, [6.0, 4.0, 2.5]),
        (1000, 2000, [6.0, 4.0, 2.5]),
        (2000, 4000, [6.0, 4.0, 2.5]),
    ]
    n_runs = 5

    print("=" * 88)
    print("Benchmark: estimate_population_eigenvalues (stage breakdown)")
    print("=" * 88)
    header = (f"{'p':>6} {'n':>6} {'gamma':>6} | "
              f"{'detect':>9} {'debias':>9} {'bulk_LW':>9} | "
              f"{'full':>9} {'sum':>9} {'overhead':>9}")
    print(header)
    print("-" * 88)

    for p, n, spikes in cases:
        gamma = p / n
        X, Sigma = simulate_spiked(p, n, spikes, seed=1)
        S = X.T @ X / n
        evals = np.sort(np.linalg.eigvalsh(S))[::-1].copy()  # descending, contiguous

        # Stage 1: spike detection (BEMA)
        t_detect = timeit(lambda: rk.detect_spikes_bema(evals, gamma), n_runs=n_runs)

        # Stage 2: spike debiasing (inverse BBP) — needs sigma2 from detection
        det = rk.detect_spikes_bema(evals, gamma)
        sigma2 = det["sigma2"]
        sample_spikes = evals[: det["k"]].copy()
        t_debias = timeit(
            lambda: rk.inverse_bbp(sample_spikes, gamma, sigma2), n_runs=n_runs
        )

        # Stage 3: bulk deconvolution (Ledoit-Wolf / RIE)
        t_bulk = timeit(lambda: rk.ledoit_wolf_shrinkage(evals, gamma), n_runs=n_runs)

        # Full primary API call
        t_full = timeit(
            lambda: rk.estimate_population_eigenvalues(evals, gamma), n_runs=n_runs
        )

        t_sum = t_detect + t_debias + t_bulk
        overhead = t_full - t_sum

        print(f"{p:>6} {n:>6} {gamma:>6.3f} | "
              f"{t_detect:>9.3f} {t_debias:>9.3f} {t_bulk:>9.3f} | "
              f"{t_full:>9.3f} {t_sum:>9.3f} {overhead:>9.3f}")

    print("-" * 88)
    print("All times in ms (mean over runs). 'overhead' = full - (detect+debias+bulk).")
    print()
    print("Interpretation: the bulk Ledoit-Wolf deconvolution (which reuses the")
    print("fast Stieltjes kernel) dominates the cost. Detection and debiasing are")
    print("near-instant. The full call is essentially the bulk deconvolution cost.")


if __name__ == "__main__":
    main()
