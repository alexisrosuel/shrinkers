"""Head-to-head: Ledoit-Wolf QuEST (forward map) vs shrinkers (its inverse).

Run from this directory:  ../../.pixi/envs/default/bin/python compare.py
"""

from __future__ import annotations

import json
import time

import numpy as np
from quest_reference import quest

import shrinkers as sh


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def spiked_tau(p, spikes, bulk=1.0):
    return np.concatenate([np.asarray(spikes, float), np.full(p - len(spikes), bulk)])


def sample_spectrum(rng, tau, n):
    p = tau.size
    X = rng.standard_normal((p, n)) * np.sqrt(tau)[:, None]
    return np.linalg.eigvalsh(X @ X.T / n)


def empirical_shrinkage(lam, c, eta, parallel=True):
    """LW nonlinear shrinkage from the EMPIRICAL Stieltjes transform of lam
    (exactly shrinkers' formula, with an explicit eta)."""
    st = sh.stieltjes_transform(lam, eta=float(eta), parallel=parallel)
    g = st["real"] + 1j * st["imag"]
    return lam / np.abs(1.0 - c + c * lam * g) ** 2


def relerr(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))


def timeit(fn, repeats=7, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


# --------------------------------------------------------------------------
# A. the two shrinkage formulas on the same (asymptotic) sample spectrum
# --------------------------------------------------------------------------
def experiment_A(report):
    print("\n" + "=" * 78)
    print("A. QuEST oracle shrinkage d  vs  shrinkers empirical xi")
    print("=" * 78)

    # --- A1: the population bulk (continuous support) ---
    c = 0.25
    print("A1. Two-block population [3 x p/2, 1 x p/2] (no isolated atoms):")
    print(f"    {'p':>6} {'median rel err':>15} {'90th pct':>10} {'max rel err':>12}")
    rows = []
    for p in (2000, 5000, 10000, 20000):
        n = round(p / c)
        tau = np.concatenate([np.full(p // 2, 3.0), np.full(p - p // 2, 1.0)])
        res = quest(tau, n)
        lam, d = res["lambda"], res["d"]
        eta = 0.1 / np.sqrt(p)
        xi = empirical_shrinkage(lam, c, eta)
        m = slice(20, p - 20)                       # drop the support edges
        rel = np.abs(xi[m] - d[m]) / np.abs(d[m])
        med, p90, mx = float(np.median(rel)), float(np.quantile(rel, 0.9)), float(rel.max())
        rows.append({"p": p, "eta": eta, "median": med, "p90": p90, "max": mx})
        print(f"    {p:>6} {med:>15.3e} {p90:>10.3e} {mx:>12.3e}")

    # --- A2: spiked population: bulk vs atomic spikes ---
    print("A2. Spiked population [12, 7, 4, 1 x (p-3)]:")
    p = 2000
    n = round(p / c)
    tau = spiked_tau(p, [12.0, 7.0, 4.0])
    res = quest(tau, n)
    lam, d = res["lambda"], res["d"]
    eta = 0.1 / np.sqrt(p)
    xi = empirical_shrinkage(lam, c, eta)
    print(f"    p={p}")
    print(f"    QuEST lambda (top 3)      : {np.round(lam[-3:][::-1], 5)}")
    print(f"    QuEST d      (top 3)      : {np.round(d[-3:][::-1], 5)}  <- atom bins: d is not defined by the continuous formula")
    print(f"    shrinkers xi (top 3)      : {np.round(xi[-3:][::-1], 5)}")
    # bulk = everything except the atomic bins.  The atoms carry mass ~1/p each,
    # so the top 4 bins are contaminated; compare the rest.
    mb = slice(20, p - 4)
    rel_bulk = np.abs(xi[mb] - d[mb]) / np.abs(d[mb])
    print(f"    bulk median rel err       : {np.median(rel_bulk):.3e}   "
          f"max : {rel_bulk.max():.3e}")
    # shrinkers' hybrid recovers the atoms via inverse BBP instead
    est = sh.estimate_population_eigenvalues(lam, c)
    print(f"    shrinkers hybrid spikes   : {np.round(est['spikes'][::-1], 5)}  (true [12 7 4])")
    print(f"    shrinkers hybrid bulk mean: {np.mean(est['bulk_population']):.5f}  (true 1.0)")
    report["A"] = {"blocks": rows,
                   "spiked_p": p,
                   "spiked_bulk_median": float(np.median(rel_bulk)),
                   "spiked_bulk_max": float(rel_bulk.max()),
                   "shrinkers_spikes": est["spikes"].tolist(),
                   "shrinkers_bulk_mean": float(np.mean(est["bulk_population"]))}


# --------------------------------------------------------------------------
# B. is shrinkers the numerical inverse of QuEST?
# --------------------------------------------------------------------------
def experiment_B(report):
    print("\n" + "=" * 78)
    print("B. Inverse round-trip: tau -> QuEST -> lambda -> shrinkers -> tau_hat")
    print("=" * 78)
    c = 0.25
    rows = []
    for p in (500, 1000, 2000, 4000):
        n = round(p / c)
        tau = spiked_tau(p, [12.0, 7.0, 4.0])
        lam = quest(tau, n)["lambda"]
        est = sh.estimate_population_eigenvalues(lam, c)
        sp_hat = np.asarray(est["spikes"])
        bulk_hat = np.asarray(est["bulk_population"])
        tau_hat = np.sort(np.concatenate([sp_hat, bulk_hat]))
        refit = float(np.max(np.abs(np.sort(quest(tau_hat, n)["lambda"]) - np.sort(lam))))
        rows.append({"p": p, "k": int(est["k"]), "spikes": sp_hat.tolist(),
                     "spike_rel_err": float(relerr(sp_hat, np.array([12.0, 7.0, 4.0]))),
                     "bulk_mean": float(bulk_hat.mean()),
                     "quest_refit_inf": refit})
        print(f"  p={p:>5}: k={est['k']}  spikes={np.round(sp_hat, 4)}  "
              f"spike rel err={rows[-1]['spike_rel_err']:.3e}  "
              f"bulk mean={bulk_hat.mean():.6f}  ||QuEST(tau_hat)-lambda||inf={refit:.3e}")
    report["B"] = rows


# --------------------------------------------------------------------------
# C. finite-sample data: QuEST fixed point on a real sample spectrum
# --------------------------------------------------------------------------
def experiment_C(report):
    print("\n" + "=" * 78)
    print("C. Finite-sample Monte Carlo at p=1000, c=0.25")
    print("=" * 78)
    rng = np.random.default_rng(3)
    p, c = 1000, 0.25
    n = round(p / c)
    tau = spiked_tau(p, [12.0, 7.0, 4.0])
    lam_obs = sample_spectrum(rng, tau, n)
    est = sh.estimate_population_eigenvalues(lam_obs, c)
    bulk = np.asarray(est["bulk_population"])
    tau_hat = np.sort(np.concatenate([np.asarray(est["spikes"]), bulk]))
    refit = float(np.max(np.abs(np.sort(quest(tau_hat, n)["lambda"]) - np.sort(lam_obs))))
    rel2 = float(np.linalg.norm(np.sort(quest(tau_hat, n)["lambda"]) - np.sort(lam_obs))
                 / np.linalg.norm(lam_obs))
    print("  observed top-3     :", np.round(lam_obs[-3:][::-1], 4))
    print("  shrinkers k        :", est["k"], " spikes:", np.round(est["spikes"][::-1], 4))
    print("  shrinkers bulk mean:", round(float(bulk.mean()), 5), " (true 1.0)")
    print("  mean |lambda - tau| (raw)      :", round(float(np.mean(np.abs(np.sort(lam_obs) - np.sort(tau)))), 4))
    print("  mean |tau_hat - tau| (cleaned) :", round(float(np.mean(np.abs(tau_hat - np.sort(tau)))), 4))
    print(f"  QuEST(tau_hat) vs lambda: inf={refit:.4e}  relL2={rel2:.4e}")
    report["C"] = {"k": int(est["k"]), "spikes": est["spikes"].tolist(),
                   "bulk_mean": float(bulk.mean()), "refit_inf": refit, "refit_relL2": rel2}


# --------------------------------------------------------------------------
# D. runtime
# --------------------------------------------------------------------------
def experiment_D(report):
    print("\n" + "=" * 78)
    print("D. Runtime: QuEST forward (NumPy port) vs shrinkers (Rust)")
    print("=" * 78)
    c = 0.25
    rows = []
    print(f"  {'p':>6} {'QuEST fwd':>12} {'stieltjes':>12} {'LW shrink':>12} "
          f"{'deconvolve':>12} {'pop_est':>12}")
    for p in (500, 1000, 2000, 4000, 8000):
        n = round(p / c)
        tau = np.sort(spiked_tau(p, [12.0, 7.0, 4.0]))
        lam = np.sort(quest(tau, n)["lambda"])
        row = {
            "p": p,
            "quest_forward_ms": timeit(lambda tau=tau, n=n: quest(tau, n), repeats=5, warmup=1),
            "stieltjes_ms": timeit(
                lambda lam=lam: sh.stieltjes_transform(lam, method="blocked"), repeats=7
            ),
            "lw_shrink_ms": timeit(lambda lam=lam: sh.ledoit_wolf_shrinkage(lam, c), repeats=7),
            "deconvolve_ms": timeit(lambda lam=lam: sh.deconvolve_spiked(lam, c=c), repeats=7),
            "pop_est_ms": timeit(
                lambda lam=lam: sh.estimate_population_eigenvalues(lam, c), repeats=7
            ),
        }
        rows.append(row)
        print(f"  {p:>6} {row['quest_forward_ms']:>11.3f}ms {row['stieltjes_ms']:>11.3f}ms "
              f"{row['lw_shrink_ms']:>11.3f}ms {row['deconvolve_ms']:>11.3f}ms "
              f"{row['pop_est_ms']:>11.3f}ms")

    p = 8000
    n = round(p / c)
    tau = np.sort(spiked_tau(p, [12.0, 7.0, 4.0]))
    lam = np.sort(quest(tau, n)["lambda"])
    seq = timeit(lambda: sh.stieltjes_transform(lam, method="blocked", parallel=False), repeats=5)
    par = timeit(lambda: sh.stieltjes_transform(lam, method="blocked", parallel=True), repeats=5)
    print(f"  p={p}: stieltjes seq {seq:.3f}ms  parallel {par:.3f}ms  speedup {seq/par:.2f}x")
    report["D"] = {"rows": rows, "p8000_parallel": {"seq_ms": seq, "par_ms": par}}


# --------------------------------------------------------------------------
# E. the actual LW estimator: invert QuEST with a nonlinear optimizer
# --------------------------------------------------------------------------
def experiment_E(report):
    print("\n" + "=" * 78)
    print("E. LW's estimator inverts QuEST with an optimizer -- measured cost")
    print("=" * 78)
    from scipy.optimize import minimize

    c = 0.25
    p = 100
    n = round(p / c)
    tau = spiked_tau(p, [12.0, 7.0, 4.0])
    # deterministic QuEST spectrum: isolates the cost of the inversion itself
    # from finite-sample noise in the target.
    lam = np.sort(quest(tau, n)["lambda"])

    calls = {"n": 0}

    def objective(theta):
        calls["n"] += 1
        s2 = theta[3]
        t = np.concatenate([theta[:3], np.full(p - 3, s2)])
        q = quest(np.sort(t), n)["lambda"]
        return float(np.sum((q - lam) ** 2))

    x0 = np.array([lam[-1], lam[-2], lam[-3], np.median(lam[:-3])])
    t0 = time.perf_counter()
    opt = minimize(objective, x0, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 4000})
    t_quest_opt = (time.perf_counter() - t0) * 1e3

    t_shrink = timeit(lambda: sh.estimate_population_eigenvalues(lam, c), repeats=7)
    est = sh.estimate_population_eigenvalues(lam, c)

    print(f"  p={p}, n={n}, one QuEST eval = "
          f"{timeit(lambda: quest(np.sort(tau), n), repeats=7):.3f}ms")
    print("  target lambda = QuEST(tau), tau = [12,7,4,1...]")
    print(f"  QuEST inversion (Nelder-Mead, {calls['n']} evals): {t_quest_opt:.1f}ms"
          f"  -> theta = {np.round(opt.x, 4)}  (true [12 7 4 1])")
    print(f"  shrinkers estimate_population_eigenvalues          : {t_shrink:.4f}ms"
          f"  -> spikes {np.round(est['spikes'], 4)}, "
          f"bulk {np.mean(est['bulk_population']):.4f}")
    print(f"  runtime ratio (QuEST-inversion / shrinkers)        : {t_quest_opt/t_shrink:.0f}x")
    report["E"] = {"p": p, "quest_evals": int(calls["n"]),
                   "quest_inversion_ms": t_quest_opt,
                   "shrinkers_ms": t_shrink,
                   "ratio": t_quest_opt / t_shrink,
                   "theta": opt.x.tolist(),
                   "shrinkers_spikes": est["spikes"].tolist(),
                   "shrinkers_bulk_mean": float(np.mean(est["bulk_population"]))}


def main():
    np.set_printoptions(suppress=True, precision=6)
    report = {"shrinkers_version": sh.__version__}
    experiment_A(report)
    experiment_B(report)
    experiment_C(report)
    experiment_D(report)
    experiment_E(report)
    with open("results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
