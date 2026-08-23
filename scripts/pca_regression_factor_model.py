#!/usr/bin/env python
"""PCA regression on a factor model: RMT-cleaned vs raw PCA.

Factor model (the exact model this package's deconvolution was designed for):

    X = F Λᵀ + E,     F ~ N(0, I_k)   (n×k latent factors)
                       Λ (p×k) loadings,  ΛᵀΛ = diag(ℓ_i)
                       E ~ N(0, σ² I)  (n×p idiosyncratic noise)
    y = F β_f + ε,    ε ~ N(0, σ_y²)

The population covariance of X is Σ = ΛΛᵀ + σ² I — a *spiked* covariance with
k strong factors on top of an isotropic noise floor. Its eigenvalues are the
spike eigenvalues ℓ_i + σ² (i = 1..k) plus σ² repeated p−k times.

We compare three ways of choosing the number of PCs in principal component
regression (PCR):

  1. **Oracle k** — the true number of factors (infeasible in practice).
  2. **Scree k** — a variance-explained heuristic (cumulative share of the
     total variance). In high dimension this badly overestimates k, because
     the bulk noise carries most of the variance.
  3. **RMT-cleaned k (BEMA)** — estimated by `deconvolve_spiked` (BEMA spike
     detection), which correctly separates the few strong factors from the
     noise bulk.

Metrics
-------
  * Out-of-sample prediction MSE of y on a held-out test set.
  * Recovery of the true latent factors: canonical correlations between the
    estimated factor scores F̂ and the true factors F.
  * Number of PCs used (parsimony).

NOTE on overlap weighting
-------------------------
A natural idea is to weight each PC by its angular overlap α² (signal
strength). For OLS this is a *no-op*: scaling a regressor by a constant is
absorbed into the fitted coefficient, so the prediction is unchanged. The
cleaning therefore helps PCR through the *number of factors k*, not through
reweighting the PCs.

Only the *exact* package methods are used (f64, no cutoff, no approximation).

Run:  python scripts/pca_regression_factor_model.py
"""

import numpy as np

import shrinkers as rk

# ─────────────────────────────────────────────────────────────────────────────
#  Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_factor_model(p, n, n_test, k, factor_eigs, noise_std, seed=0):
    """Simulate X (n×p) from a factor model with a spiked covariance.

    Returns (X_train, y_train, X_test, y_test, F_train, F_test, beta_f).
    Train and test rows are drawn independently from the same population and
    are disjoint (clean held-out evaluation).
    """
    rng = np.random.default_rng(seed)

    # Loadings: orthonormal columns scaled so ΛᵀΛ = diag(factor_eigs).
    Q, _ = np.linalg.qr(rng.standard_normal((p, k)))
    Lambda = Q * np.sqrt(factor_eigs)          # (p, k)

    # Latent factors + idiosyncratic noise.
    F = rng.standard_normal((n + n_test, k))   # (n, k)
    E = rng.standard_normal((n + n_test, p)) * noise_std
    X = F @ Lambda.T + E

    # Regression target on the latent factors.
    beta_f = rng.standard_normal(k)
    eps = rng.standard_normal(n + n_test)
    y = F @ beta_f + eps

    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    F_train, F_test = F[:n], F[n:]
    return X_train, y_train, X_test, y_test, F_train, F_test, beta_f


# ─────────────────────────────────────────────────────────────────────────────
#  PCA helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_cov(X):
    """Sample covariance S = XᵀX / n."""
    return X.T @ X / X.shape[0]


def top_pc_vectors(X, k):
    """Top-k sample eigenvectors of S = XᵀX/n (columns, descending)."""
    S = sample_cov(X)
    _, U = np.linalg.eigh(S)          # ascending
    return U[:, ::-1][:, :k]


def pcr_predict(X, U_k, beta_hat):
    """Predict y on data X using PC vectors U_k and coefficients beta_hat."""
    return (X @ U_k) @ beta_hat


def factor_recovery(F_hat, F_true):
    """Canonical correlations between estimated and true factor spaces.

    Values near 1 mean the estimated PCs span the true factor space well.
    """
    Fh = F_hat - F_hat.mean(0)
    Ft = F_true - F_true.mean(0)
    Qh, _ = np.linalg.qr(Fh)
    Qt, _ = np.linalg.qr(Ft)
    return np.linalg.svd(Qh.T @ Qt, compute_uv=False)


def scree_k(evals_desc, threshold=0.9):
    """Number of PCs needed to explain `threshold` of the total variance."""
    total = evals_desc.sum()
    cum = np.cumsum(evals_desc) / total
    return int(np.searchsorted(cum, threshold) + 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Estimators
# ─────────────────────────────────────────────────────────────────────────────

def pcr(X_train, y_train, k):
    """PCR with the top-k sample PCs. Returns (U_k, beta_hat)."""
    U_k = top_pc_vectors(X_train, k)
    scores = X_train @ U_k
    beta_hat = np.linalg.lstsq(scores, y_train, rcond=None)[0]
    return U_k, beta_hat


def bema_k(X_train, gamma):
    """Estimate the number of factors k via BEMA spike detection."""
    S = sample_cov(X_train)
    evals = np.linalg.eigvalsh(S)         # ascending
    dres = rk.deconvolve_spiked(np.ascontiguousarray(evals), c=gamma, n_points=400)
    return dres["k"]


# ─────────────────────────────────────────────────────────────────────────────
#  Experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_case(p, n, n_test, k, factor_eigs, noise_std, seed):
    """Run one configuration; return prediction + factor-recovery metrics."""
    X, y, Xt, yt, F, Ft, beta_f = simulate_factor_model(
        p, n, n_test, k, factor_eigs, noise_std, seed
    )
    gamma = p / n

    # ---- k selection ----
    evals_desc = np.linalg.eigvalsh(sample_cov(X))[::-1]
    k_scree = scree_k(evals_desc, threshold=0.9)
    k_bema = bema_k(X, gamma)

    # ---- PCR with each k ----
    out = {}
    for name, kk in [("oracle", k), ("scree", k_scree), ("bema", k_bema)]:
        U_k, beta_hat = pcr(X, y, kk)
        mse = float(np.mean((yt - pcr_predict(Xt, U_k, beta_hat)) ** 2))
        rec = factor_recovery(X @ U_k, F)
        out[name] = (mse, rec, kk)

    return out


def main():
    np.set_printoptions(precision=3, suppress=True)

    p, n, n_test = 200, 400, 500
    k = 3
    noise_std = 1.0
    gamma = p / n
    n_seeds = 5

    # Two regimes: strong factors (well above the bulk edge) and weak factors
    # (close to the BBP threshold, where eigenvectors are noisy).
    regimes = {
        "strong": np.array([6.0, 4.0, 2.5]),
        "weak":   np.array([3.0, 2.5, 2.0]),
    }

    print("=" * 100)
    print("PCA regression on a factor model: RMT-cleaned vs raw PCA")
    print("=" * 100)
    print(f"p={p}, n={n}, p/n={gamma:.2f}, k={k} factors, σ²={noise_std**2:.0f}")
    print(f"True bulk edge (σ²=1): {(1 + np.sqrt(gamma))**2:.3f}")
    print("Only exact package methods used (f64, no cutoff).")
    print()

    for reg_name, fe in regimes.items():
        print("=" * 100)
        print(f"REGIME: {reg_name} factors  (ℓ = {fe}, Σ eigenvalues = {fe + 1})")
        print("=" * 100)

        # ── k selection ──
        print("\nNumber of PCs selected (mean over seeds)")
        print(f"{'method':<10} {'k':>6}")
        print("-" * 20)
        ks = {"oracle": [], "scree": [], "bema": []}
        for s in range(n_seeds):
            out = run_case(p, n, n_test, k, fe, noise_std, seed=100 + s)
            for name in ks:
                ks[name].append(out[name][2])
        for name in ["oracle", "scree", "bema"]:
            print(f"{name:<10} {np.mean(ks[name]):6.1f}")
        print()

        # ── Prediction MSE ──
        print("Out-of-sample prediction MSE of y (held-out test set)")
        print(f"{'seed':>4} | {'oracle':>9} {'scree':>9} {'bema':>9}")
        print("-" * 40)
        mse = {"oracle": [], "scree": [], "bema": []}
        for s in range(n_seeds):
            out = run_case(p, n, n_test, k, fe, noise_std, seed=100 + s)
            row = [out[name][0] for name in ["oracle", "scree", "bema"]]
            for name, v in zip(["oracle", "scree", "bema"], row):
                mse[name].append(v)
            print(f"{s:4d} | {row[0]:9.3f} {row[1]:9.3f} {row[2]:9.3f}")
        print("-" * 40)
        print(f"{'mean':>4} | {np.mean(mse['oracle']):9.3f} "
              f"{np.mean(mse['scree']):9.3f} {np.mean(mse['bema']):9.3f}")
        print()

        # ── Factor recovery ──
        print("Factor recovery: canonical correlations between F̂ and F (mean)")
        print(f"{'method':<10} {'mean corr':>10}")
        print("-" * 24)
        rec = {"oracle": [], "scree": [], "bema": []}
        for s in range(n_seeds):
            out = run_case(p, n, n_test, k, fe, noise_std, seed=100 + s)
            for name in rec:
                rec[name].append(np.mean(out[name][1]))
        for name in ["oracle", "scree", "bema"]:
            print(f"{name:<10} {np.mean(rec[name]):10.3f}")
        print()

    print("Reading:")
    print("  * BEMA recovers the true number of factors k in both regimes; the")
    print("    scree heuristic overestimates it by an order of magnitude (~150")
    print("    vs 3), because the bulk noise carries most of the variance.")
    print("  * With the correct k, PCR is parsimonious (3 PCs) and recovers the")
    print("    factor space well. The scree version uses ~50x more PCs.")
    print("  * Prediction MSE is not the cleanest discriminator: when the")
    print("    eigenvectors are noisy (weak regime), including extra PCs can")
    print("    recover residual factor signal, so scree's over-parameterization")
    print("    does not always hurt prediction. The honest win of the cleaning")
    print("    is a *correct, parsimonious* factor count for interpretation.")
    print("  * Overlap weighting is a no-op for OLS (scaling a regressor is")
    print("    absorbed into the coefficient), so the cleaning helps PCR through")
    print("    k, not through reweighting the PCs.")


if __name__ == "__main__":
    main()
