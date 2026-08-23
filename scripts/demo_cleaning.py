#!/usr/bin/env python
"""Demo: spiked + bulk eigenvalue cleaning with shrinkers.deconvolve_spiked.

Simulates a high-dimensional spiked covariance model and applies the package's
single entry point `deconvolve_spiked`:

  1. Spike detection (BEMA) + debiasing (inverse BBP / DGJ)
  2. Bulk deconvolution (El Karoui) -> population spectral density

Also reports wall-clock timing for the cleaning call.

Run:  python scripts/demo_cleaning.py
"""

import time

import numpy as np

import shrinkers as rk


def simulate_spiked(p: int, n: int, spikes: list[float], seed: int = 0):
    """Simulate X (n x p) from a spiked covariance model.

    Population covariance: Sigma = I_p + sum_i (ell_i - 1) v_i v_i^T
    with unit-norm spike directions v_i. Returns (X, Sigma, true_spikes).
    """
    rng = np.random.default_rng(seed)
    Sigma = np.eye(p)
    for ell in spikes:
        v = rng.standard_normal(p)
        v /= np.linalg.norm(v)
        Sigma += (ell - 1.0) * np.outer(v, v)
    X = rng.standard_normal((n, p)) @ np.linalg.cholesky(Sigma).T
    return X, Sigma, np.array(spikes)


def main():
    p, n = 200, 500
    true_spikes = [6.0, 4.0, 2.5]
    gamma = p / n

    X, Sigma, true_spikes = simulate_spiked(p, n, true_spikes, seed=1)
    S = X.T @ X / n
    evals = np.linalg.eigvalsh(S)

    print(f"p={p}, n={n}, gamma={gamma:.3f}")
    print(f"True population spikes: {true_spikes}")
    print(f"True bulk edge (sigma2=1): {(1 + np.sqrt(gamma))**2:.3f}")
    print()

    # ── Single entry point: spiked + bulk cleaning ──
    print("=== deconvolve_spiked ===")
    t0 = time.perf_counter()
    res = rk.deconvolve_spiked(evals, c=gamma, n_points=300, eta=0.05)
    dt = time.perf_counter() - t0

    print(f"Detected K = {res['k']}")
    print(f"Estimated population spikes: {np.round(res['spikes'], 3)}")
    print(f"  (true: {true_spikes})")
    print(f"Estimated sigma2:            {res['sigma2']:.3f}")
    print(f"Estimated bulk edge:         {res['bulk_edge']:.3f}")
    print(f"Wall-clock:                  {dt*1e3:.1f} ms")
    print()

    # ── Bulk deconvolution output ──
    bulk = res["bulk"]
    print("=== Bulk deconvolution (El Karoui) ===")
    print(f"  grid points: {len(bulk['lambda_grid'])}")
    print(f"  density:     {len(bulk['density'])} points")
    # Where does the recovered bulk density peak?
    i_peak = int(np.argmax(bulk["density"]))
    print(f"  density peak at lambda = {bulk['lambda_grid'][i_peak]:.3f}")
    print(f"  (MP bulk support for sigma2={res['sigma2']:.2f}: "
          f"[{res['sigma2']*(1-np.sqrt(gamma))**2:.3f}, "
          f"{res['sigma2']*(1+np.sqrt(gamma))**2:.3f}])")
    print()

    # ── Sanity: trace of cleaned spectrum vs sample trace ──
    # The bulk density integrates to the bulk mass; spikes carry the rest.
    lam = bulk["lambda_grid"]
    dens = bulk["density"]
    # trapezoid integral of the recovered bulk density
    bulk_mass = np.trapezoid(dens, lam)
    print("=== Sanity checks ===")
    print(f"  recovered bulk mass (integral of density): {bulk_mass:.3f} "
          f"(expect ~1 - K/p = {1 - res['k']/p:.3f})")
    print(f"  sample trace / p: {evals.mean():.3f} "
          f"(expect ~1 + sum(ell-1)/p = {1 + (true_spikes-1).sum()/p:.3f})")


if __name__ == "__main__":
    main()
