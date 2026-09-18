"""
Benchmark the Rust Stieltjes transform (via shrinkers.stieltjes_transform)
against a pure NumPy implementation.

Prints a copy-paste table for the README.
"""
import numpy as np
from _common import bench_us, mp_spectrum, numpy_stieltjes

import shrinkers as rk


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
        evals = mp_spectrum(p)
        eta = 0.1 / np.sqrt(p)
        n_runs = 100 if p <= 500 else 30 if p <= 2000 else 10

        np_us = bench_us(numpy_stieltjes, evals, eta, n_runs=n_runs)

        row = f"{p:>6}  {np_us:>10.1f}  "
        for m in methods:
            us = bench_us(
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
