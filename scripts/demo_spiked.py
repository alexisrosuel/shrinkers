#!/usr/bin/env python
"""Demo: spiked covariance model analysis with shrinkers.

Simulates a high-dimensional spiked covariance model and demonstrates the
SOTA spike-estimation methods implemented in the `spiked` module:

1. Spike detection (BEMA + Tracy-Widom edge thresholding)
2. Population spike eigenvalue estimation (inverse BBP / DGJ)
3. Ledoit-Wolf non-linear shrinkage (via the fast Stieltjes library)
4. Eigenvector angular overlap (BBP angle formula / Benaych-Georges & Nadakuditi)

Run:  python scripts/demo_spiked.py
"""

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

    # ── 1. Spike detection ──
    print("=== Spike detection ===")
    bema = rk.detect_spikes_bema(evals, gamma)
    tw = rk.detect_spikes_tracy_widom(evals, gamma)
    print(f"BEMA:          K={bema['k']}, sigma2={bema['sigma2']:.3f}, "
          f"edge={bema['bulk_edge']:.3f}")
    print(f"Tracy-Widom:   K={tw['k']}, sigma2={tw['sigma2']:.3f}, "
          f"edge={tw['bulk_edge']:.3f}")
    print()

    # ── 2. Full analysis ──
    print("=== Full spiked analysis ===")
    res = rk.analyze_spikes(evals, gamma)
    print(f"Detected K = {res['k']}")
    print(f"Estimated population spikes: {np.round(res['spikes'], 3)}")
    print(f"Angular overlaps alpha^2:    {np.round(res['overlaps'], 3)}")
    print(f"Estimated sigma2:            {res['sigma2']:.3f}")
    print(f"Estimated bulk edge:         {res['bulk_edge']:.3f}")
    print()

    # ── 3. Ledoit-Wolf shrinkage (all eigenvalues) ──
    print("=== Ledoit-Wolf non-linear shrinkage ===")
    lw = rk.ledoit_wolf_shrinkage(evals, gamma)
    top = np.argsort(lw)[::-1][:5]
    print("Top-5 Ledoit-Wolf population estimates:")
    for i in top:
        print(f"  sample={evals[i]:.3f}  ->  LW={lw[i]:.3f}")
    print()

    # ── 4. Inverse BBP on the detected spikes ──
    print("=== Inverse BBP (DGJ) on sample spikes ===")
    sample_spikes = np.sort(evals)[::-1][: res["k"]].copy()  # contiguous copy
    recovered = rk.inverse_bbp(sample_spikes, gamma, res["sigma2"])
    print("sample lambda_hat -> recovered population ell")
    for sh, ell in zip(sample_spikes, recovered):
        print(f"  {sh:.3f}  ->  {ell:.3f}")
    print()

    # ── 5. Eigenvector debiasing (S-POET style) ──
    print("=== Eigenvector debiasing ===")
    # Recover the sample eigenvectors of S.
    _, U = np.linalg.eigh(S)
    # Debiased direction for the top spike using its estimated overlap.
    alpha2 = res["overlaps"][0]
    u_top = U[:, -1]
    u_debiased = u_top / np.sqrt(alpha2)
    print(f"Top spike overlap alpha^2 = {alpha2:.3f}")
    print(f"Sample eigenvector norm:   {np.linalg.norm(u_top):.3f}")
    print(f"Debiased vector norm:      {np.linalg.norm(u_debiased):.3f}")


if __name__ == "__main__":
    main()
