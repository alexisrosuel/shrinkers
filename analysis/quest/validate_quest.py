"""Validate the QuEST reference port against Monte Carlo and closed forms."""

import numpy as np
from quest_reference import quest


def sample_spectrum(rng, tau, n):
    p = tau.size
    X = rng.standard_normal((p, n)) * np.sqrt(tau)[:, None]
    S = X @ X.T / n
    return np.linalg.eigvalsh(S)


def main():
    rng = np.random.default_rng(0)
    p, c = 1000, 0.25
    n = round(p / c)
    tau = np.concatenate([[12.0, 7.0, 4.0], np.ones(p - 3)])

    res = quest(tau, n)
    print("== QuEST forward, spiked tau = [12,7,4,1,...], p=1000 c=0.25 ==")
    lam_q = res["lambda"]
    print("top-6 QuEST sample eigenvalues:", np.round(lam_q[::-1][:6], 4))
    # BBP closed forms for isolated spikes
    for ell in (12.0, 7.0, 4.0):
        bbp = ell * (1 + c / (ell - 1))
        print(f"   BBP forward for ell={ell}: {bbp:.4f}")
    bulk_min = lam_q[0]
    print("leftmost sample eigenvalue:", bulk_min, " MP edge:", (1 - np.sqrt(c)) ** 2)

    # Monte Carlo pooled empirical CDF
    reps = 120
    pooled = np.concatenate([sample_spectrum(rng, tau, n) for _ in range(reps)])
    # compare empirical CDF of pooled sample eigenvalues to QuEST F(x)
    xs = np.quantile(pooled, np.linspace(0.001, 0.999, 400))
    emp = np.searchsorted(np.sort(pooled), xs) / pooled.size
    theo = np.interp(xs, res["x"], res["F"])
    err = np.max(np.abs(emp - theo))
    print(f"MC pooled CDF vs QuEST F: max abs error = {err:.4e}  (reps={reps})")

    # top spike order statistics
    top = np.sort(np.array([sample_spectrum(rng, tau, n)[-3:] for _ in range(200)]), axis=1)
    print("MC mean top-3 sample eigenvalues:", np.round(top.mean(axis=0)[::-1], 4))
    print("QuEST top-3 quantized:        ", np.round(lam_q[-3:][::-1], 4))

    # smooth (no-separation) population: MP quantiles of a Gamma-like law
    print()
    print("== QuEST forward, smooth population (no separated spikes) ==")
    q = (np.arange(1, p + 1) - 0.5) / p
    tau2 = 0.5 + 1.5 * q  # linear population spectrum on [0.5, 2]
    res2 = quest(tau2, n)
    print("support intervals:", [(a, b) for a, b in res2["support"]])
    lam2 = res2["lambda"]
    reps2 = 60
    pooled2 = np.concatenate([sample_spectrum(rng, tau2, n) for _ in range(reps2)])
    xs2 = np.quantile(pooled2, np.linspace(0.001, 0.999, 300))
    emp2 = np.searchsorted(np.sort(pooled2), xs2) / pooled2.size
    theo2 = np.interp(xs2, res2["x"], res2["F"])
    print(f"MC pooled CDF vs QuEST F: max abs error = {np.max(np.abs(emp2 - theo2)):.4e}")
    print("mean |QuEST lambda - MC mean order stats|:",
          np.mean(np.abs(lam2 - np.sort(np.mean([sample_spectrum(rng, tau2, n) for _ in range(40)], axis=0)))))


if __name__ == "__main__":
    main()
