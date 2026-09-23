"""Figure for the QuEST vs shrinkers comparison."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from quest_reference import quest

import shrinkers as sh

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))

# --- (a) the two directions on the spiked model -------------------------
c, p = 0.25, 2000
n = round(p / c)
tau = np.concatenate([[12.0, 7.0, 4.0], np.ones(p - 3)])
res = quest(tau, n)
lam = res["lambda"]
est = sh.estimate_population_eigenvalues(lam, c)
tau_hat = np.sort(np.concatenate([est["spikes"], np.asarray(est["bulk_population"])]))
q = np.arange(1, p + 1) / p
a = ax[0, 0]
a.plot(q, np.sort(tau), color="C0", lw=1.6, label=r"population $\tau$")
a.plot(q, np.sort(lam), color="C1", lw=1.2, label=r"QuEST forward $Q(\tau)$")
a.plot(q, tau_hat, color="C2", lw=1.2, ls="--", label=r"shrinkers inverse $\hat\tau$")
a.set_yscale("log")
a.set_xlabel("quantile $i/p$")
a.set_ylabel("eigenvalue")
a.set_title(f"(a) forward vs inverse, spiked model (p={p}, c={c})")
a.legend(fontsize=8)

# --- (b) shrinkage functions on a continuous spectrum --------------------
c2, p2 = 0.25, 20000
n2 = round(p2 / c2)
tau2 = np.concatenate([np.full(p2 // 2, 3.0), np.full(p2 - p2 // 2, 1.0)])
r2 = quest(tau2, n2)
lam2, d2 = r2["lambda"], r2["d"]
st = sh.stieltjes_transform(lam2, eta=0.1 / np.sqrt(p2), parallel=True)
g = st["real"] + 1j * st["imag"]
xi2 = lam2 / np.abs(1 - c2 + c2 * lam2 * g) ** 2
b = ax[0, 1]
order = np.argsort(lam2)
b.plot(lam2[order], d2[order], color="C1", lw=2.5, label="QuEST oracle $d$")
b.plot(lam2[order], xi2[order], color="C3", lw=1.0, ls="--",
       label=r"shrinkers empirical $\xi$")
b.set_xlabel(r"sample eigenvalue $\lambda$")
b.set_ylabel("optimal shrinkage")
b.set_title(f"(b) same formula, two ways (p={p2})")
b.legend(fontsize=8)

# --- (c) round-trip accuracy vs p ---------------------------------------
ps = np.array([500, 1000, 2000, 4000, 8000])
spike_err, refit = [], []
for pp in ps:
    nn = round(pp / c)
    tt = np.concatenate([[12.0, 7.0, 4.0], np.ones(pp - 3)])
    ll = quest(tt, nn)["lambda"]
    e = sh.estimate_population_eigenvalues(ll, c)
    th = np.sort(np.concatenate([np.asarray(e["spikes"]), np.asarray(e["bulk_population"])]))
    spike_err.append(np.max(np.abs(np.asarray(e["spikes"]) - [12, 7, 4]) / [12, 7, 4]))
    refit.append(np.max(np.abs(np.sort(quest(th, nn)["lambda"]) - np.sort(ll))))
cc = ax[1, 0]
cc.loglog(ps, spike_err, "o-", color="C2", label="spike relative error")
cc.loglog(ps, refit, "s-", color="C4", label=r"$\|Q(\hat\tau)-\lambda\|_\infty$")
cc.loglog(ps, 5.0 / ps, "k:", lw=1, label=r"$O(1/p)$")
cc.set_xlabel("p"); cc.set_ylabel("error")
cc.set_title("(c) shrinkers recovers the QuEST pre-image")
cc.legend(fontsize=8)

# --- (d) runtime ---------------------------------------------------------
rt = [
    (500, 7.845, 0.068, 0.028),
    (1000, 8.788, 0.258, 0.040),
    (2000, 12.915, 1.012, 0.059),
    (4000, 15.395, 4.028, 0.101),
    (8000, 23.595, 16.270, 0.256),
]
rt = np.array(rt)
dd = ax[1, 1]
dd.loglog(rt[:, 0], rt[:, 1], "o-", color="C1", label="QuEST forward (NumPy ref.)")
dd.loglog(rt[:, 0], rt[:, 2], "s-", color="C2", label="shrinkers $O(p^2)$ Stieltjes")
dd.loglog(rt[:, 0], rt[:, 3], "^-", color="C0", label="shrinkers deconvolve (treecode)")
dd.set_xlabel("p"); dd.set_ylabel("time [ms]")
dd.set_title("(d) one-call runtime (Apple M-series, 1 thread)")
dd.legend(fontsize=8)

fig.tight_layout()
fig.savefig("quest_vs_shrinkers.png", dpi=150)
print("wrote quest_vs_shrinkers.png")
