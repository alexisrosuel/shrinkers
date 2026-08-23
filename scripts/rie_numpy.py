"""
Pure NumPy reference implementation of RMT RIE non-linear shrinkage.

Uses NumPy broadcasting for the O(p²) Stieltjes transform — same algorithm
as the Rust `Naive` method, but in pure Python/NumPy.

  m_g(z) = (1/p) Σⱼ 1/(z - λⱼ),   z = λᵢ - iη  
  ξ(λᵢ) = λᵢ / |1 - c + c·λᵢ·m_g(λᵢ - iη)|²

The timings it produces are a baseline reference for the Rust/Python
benchmarks in `scripts/bench_*.py`.
"""

import time

import numpy as np


def generate_mp_spectrum(p, c, seed=42):
    """Generate a Marchenko-Pastur-like eigenvalue spectrum."""
    rng = np.random.default_rng(seed)
    lambda_min = max(1.0 - np.sqrt(c), 0.01) ** 2
    lambda_max = (1.0 + np.sqrt(c)) ** 2

    u = rng.uniform(0, 1, p)
    t = lambda_min + u * (lambda_max - lambda_min)
    evals = t + rng.uniform(0, 0.1, p)
    evals.sort()
    return evals


def rie_shrinkage(evals: np.ndarray, c: float) -> np.ndarray:
    """Pure NumPy RIE shrinkage via broadcasting (O(p²)).

    Parameters
    ----------
    evals : ndarray, shape (p,)
        Sorted sample eigenvalues.
    c : float
        Concentration ratio p / n.

    Returns
    -------
    shrinked : ndarray, shape (p,)
        Shrinked eigenvalues preserving total trace.
    """
    p = len(evals)
    eta = 0.1 / np.sqrt(p)
    original_trace = np.sum(evals)

    # Step 1: Stieltjes transform — broadcast diff into a p×p matrix
    diff = evals[:, np.newaxis] - evals[np.newaxis, :]  # (p, p)
    denom = diff * diff + eta * eta                      # (p, p)
    mg_real = np.mean(diff / denom, axis=1)               # (p,)
    mg_imag = np.mean(eta / denom, axis=1)                # (p,)

    # Step 2: Shrinkage factor for each eigenvalue
    term_real = c * evals * mg_real
    term_imag = c * evals * mg_imag
    denom_real = 1.0 - c + term_real
    denom_imag = term_imag
    denom_norm_sq = denom_real * denom_real + denom_imag * denom_imag

    shrinked = np.where(denom_norm_sq > 0.0, evals / denom_norm_sq, evals)

    # Step 3: Trace-preserving scale
    shrinked_trace = np.sum(shrinked)
    if shrinked_trace > 0.0:
        shrinked *= original_trace / shrinked_trace

    return shrinked


def benchmark():
    """Benchmark across (p, c) combos and print copy-paste table."""

    cases = [
        (100, 0.1),
        (100, 0.5),
        (100, 0.9),
        (500, 0.1),
        (500, 0.5),
        (500, 0.9),
        (1000, 0.1),
        (1000, 0.5),
        (1000, 0.9),
    ]

    print("=" * 72)
    print("Pure NumPy RIE Shrinkage — Reference Benchmark")
    print("=" * 72)
    print(f"{'p':>4}  {'c':>4}  {'time(µs)':>10}")
    print("-" * 72)

    results = {}
    for p, c in cases:
        evals = generate_mp_spectrum(p, c)

        # Warmup
        _ = rie_shrinkage(evals, c)

        # Timed run
        n_runs = 50 if p <= 100 else 20 if p <= 500 else 10
        t0 = time.perf_counter()
        for _ in range(n_runs):
            rie_shrinkage(evals, c)
        t1 = time.perf_counter()

        avg_us = (t1 - t0) / n_runs * 1e6
        results[(p, c)] = avg_us
        print(f"{p:4d}  {c:4.1f}  {avg_us:10.1f}")

    print("-" * 72)
    print()
    print(">>> Frozen NumPy reference timings (NUMPY_REF_US):")
    print("NUMPY_REF_US = {")
    for (p, c), t in results.items():
        print(f"    ({p:>4}, {c:>3.1f}): {t:>8.1f},")
    print("}")


def verify_equal():
    """Sanity check: compare with Rust naive and accuracy methods."""
    try:
        import shrinkers
    except ImportError:
        print("Skipping verification (shrinkers not installed)")
        return

    print("Verifying against shrinkers...")
    for p, c in [(100, 0.5), (500, 0.5)]:
        evals = generate_mp_spectrum(p, c)
        np_out = rie_shrinkage(evals, c)

        # Rust with method="autovec" (no cutoff, exact) should match NumPy
        rust_auto = shrinkers.shrink_eigenvalues(evals, c, method="autovec")
        max_diff_auto = np.max(np.abs(rust_auto - np_out))
        print(f"  p={p}, c={c}: vs autovec = {max_diff_auto:.2e}")


if __name__ == "__main__":
    benchmark()
    print()
    verify_equal()