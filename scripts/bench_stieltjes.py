"""
Benchmark the Rust Stieltjes transform (via shrinkers.stieltjes_transform)
against a pure NumPy implementation.

Prints a copy-paste table for the README.
"""
import time

import numpy as np

import shrinkers as rk


def generate_mp_spectrum(p, c=0.5, seed=42):
    rng = np.random.default_rng(seed)
    lambda_min = max(1.0 - np.sqrt(c), 0.01) ** 2
    lambda_max = (1.0 + np.sqrt(c)) ** 2
    u = rng.uniform(0, 1, p)
    t = lambda_min + u * (lambda_max - lambda_min)
    evals = t + rng.uniform(0, 0.1, p)
    evals.sort()
    return evals


def numpy_stieltjes(evals, eta):
    """Pure NumPy O(p^2) Stieltjes transform (broadcasting)."""
    diff = evals[:, None] - evals[None, :]
    denom = diff * diff + eta * eta
    real = np.mean(diff / denom, axis=1)
    imag = np.mean(eta / denom, axis=1)
    return real, imag


def bench(fn, *args, n_runs):
    # warmup
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1e6  # µs


def main():
    cases = [100, 200, 500, 1000, 2000, 5000]
    methods = ["blocked", "autovec", "fft2"]

    print("=" * 78)
    print("Stieltjes transform: Rust (shrinkers) vs pure NumPy")
    print("=" * 78)
    header = f"{'p':>6}  {'numpy(µs)':>10}  " + "  ".join(
        f"{m}(µs):{'x':>5}" for m in methods
    )
    print(header)
    print("-" * 78)

    for p in cases:
        evals = generate_mp_spectrum(p)
        eta = 0.1 / np.sqrt(p)
        n_runs = 100 if p <= 500 else 30 if p <= 2000 else 10

        np_us = bench(numpy_stieltjes, evals, eta, n_runs=n_runs)

        row = f"{p:>6}  {np_us:>10.1f}  "
        for m in methods:
            us = bench(
                lambda e, m=m: rk.stieltjes_transform(e, eta, method=m),
                evals,
                n_runs=n_runs,
            )
            speedup = np_us / us if us > 0 else float("inf")
            row += f"{us:>9.1f} {speedup:>5.1f}x  "
        print(row)

    print("-" * 78)


if __name__ == "__main__":
    main()
