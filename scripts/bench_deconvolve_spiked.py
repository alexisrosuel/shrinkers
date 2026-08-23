"""
Benchmark `deconvolve_spiked` (bulk deconvolution Stieltjes bottleneck).

Measures wall-clock time over:
  - n_points in [50, 100, 200, 400, 800] at p=1000
  - p in [100, 500, 1000, 2000]

Also verifies numerical correctness: compares the fast-kernel output against a
pure-NumPy reference implementation of the El Karoui deconvolution (naive
O(p·n_points) Stieltjes loop) and reports the max relative error.
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


def reference_deconvolve_spiked(evals, c, n_points, eta, margin=1.0):
    """Pure-NumPy reference: naive O(p·n_points) Stieltjes loop per grid point.

    Mirrors the ORIGINAL Rust `spectral_deconvolution` (convention A then
    negate to convention B) so we can confirm the fast-kernel path is
    numerically identical.
    """
    p = len(evals)
    sorted_ev = np.sort(evals)
    # BEMA spike detection is not reimplemented here; for a pure-bulk MP
    # spectrum with no spikes, k=0 and bulk == full spectrum. We just run the
    # bulk deconvolution on the full spectrum (matches deconvolve_spiked when
    # no spikes are detected).
    ev = sorted_ev

    eta_val = eta if eta is not None else 0.1 / np.sqrt(p)
    min_ev = ev.min()
    max_ev = ev.max()
    rng = max_ev - min_ev if max_ev - min_ev > 0 else 1.0
    margin_g = 0.2 * rng
    lo = max(min_ev - margin_g, 0.0)
    hi = max_ev + margin_g
    lam = np.linspace(lo, hi, n_points)

    density = np.zeros(n_points)
    w_re = np.zeros(n_points)
    s_real = np.zeros(n_points)
    s_imag = np.zeros(n_points)
    m_real = np.zeros(n_points)
    m_imag = np.zeros(n_points)

    for k in range(n_points):
        z = lam[k] + 1j * eta_val
        # convention A: 1/(z - λⱼ)
        diff = z - ev
        ga = np.sum(1.0 / diff) / p
        # convention B: g = -g_A
        g = -ga
        zg = z * g
        a = 1.0 - c - c * zg
        w = z / a
        m = a * g
        density[k] = m.imag / np.pi
        w_re[k] = w.real
        s_real[k] = g.real
        s_imag[k] = g.imag
        m_real[k] = m.real
        m_imag[k] = m.imag

    return {
        "lambda_grid": lam,
        "density": density,
        "w_re": w_re,
        "sample_stieltjes_real": s_real,
        "sample_stieltjes_imag": s_imag,
        "population_stieltjes_real": m_real,
        "population_stieltjes_imag": m_imag,
    }


def max_rel_err(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom == 0, 1.0, denom)
    return np.max(np.abs(a - b) / denom)


def bench(fn, *args, n_runs):
    fn(*args)  # warmup
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1e6  # µs


def main():
    print("=" * 78)
    print("deconvolve_spiked: bulk deconvolution timing (fast Stieltjes kernel)")
    print("=" * 78)

    # --- Numerical correctness check (p=1000, n_points=200) ---
    p = 1000
    evals = generate_mp_spectrum(p)
    eta = 0.1 / np.sqrt(p)
    ref = reference_deconvolve_spiked(evals, 0.5, 200, eta)
    got = rk.deconvolve_spiked(evals.copy(), 0.5, 200, eta, 1.0)
    bulk = got["bulk"]
    print("\nNumerical correctness (p=1000, n_points=200, pure-bulk MP spectrum):")
    for key in ["density", "w_re", "sample_stieltjes_real", "sample_stieltjes_imag",
                "population_stieltjes_real", "population_stieltjes_imag"]:
        err = max_rel_err(ref[key], bulk[key])
        print(f"  {key:>28}: max rel err = {err:.3e}")

    # --- Timing over n_points at p=1000 ---
    print("\n" + "=" * 78)
    print("Timing over n_points (p=1000)")
    print("=" * 78)
    print(f"{'n_points':>8}  {'time(µs)':>10}")
    for n_points in [50, 100, 200, 400, 800]:
        n_runs = 200 if n_points <= 200 else 100
        us = bench(lambda: rk.deconvolve_spiked(evals.copy(), 0.5, n_points, eta, 1.0),
                   n_runs=n_runs)
        print(f"{n_points:>8}  {us:>10.1f}")

    # --- Timing over p at n_points=200 ---
    print("\n" + "=" * 78)
    print("Timing over p (n_points=200)")
    print("=" * 78)
    print(f"{'p':>6}  {'time(µs)':>10}")
    for p in [100, 500, 1000, 2000]:
        ev = generate_mp_spectrum(p)
        eta_p = 0.1 / np.sqrt(p)
        n_runs = 200 if p <= 500 else 100 if p <= 1000 else 50
        us = bench(lambda: rk.deconvolve_spiked(ev.copy(), 0.5, 200, eta_p, 1.0),
                   n_runs=n_runs)
        print(f"{p:>6}  {us:>10.1f}")


if __name__ == "__main__":
    main()
