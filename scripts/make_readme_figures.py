"""Generate the two README front-page figures with measured data.

Figure 1 — what the cleaning does: sample vs cleaned vs true population
eigenvalues under a spiked Marchenko-Pastur model (diagonal population,
Gaussian samples).

Figure 2 — how fast: shrinkers vs a naive pure-Python double loop vs a
vectorized NumPy baseline (chunked broadcasting; NO scipy, NO FFT — the
comparison isolates "same arithmetic, better engine").

Outputs:
  docs/img/cleaning_quality.png
  docs/img/performance.png
  docs/img/readme_figures.json   (measured numbers behind both figures)

Run: .pixi/envs/default/bin/python scripts/make_readme_figures.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import shrinkers as rk

OUT_DIR = Path("docs/img")
OUT_DIR.mkdir(parents=True, exist_ok=True)

C = 0.25  # concentration ratio p/n


# ──────────────────────────────────────────────
# Figure 1: cleaning quality
# ──────────────────────────────────────────────

def simulate_spiked(p: int, spikes: list[float], seed: int):
    """Diagonal-population spiked model with Gaussian samples."""
    rng = np.random.default_rng(seed)
    pop = np.concatenate([np.asarray(spikes), np.ones(p - len(spikes))])
    n = round(p / C)
    y = rng.standard_normal((p, n)) * np.sqrt(pop)[:, None]
    sample = np.linalg.eigvalsh((y @ y.T) / n)[::-1]
    truth_desc = np.sort(pop)[::-1].copy()  # largest eigenvalue first
    return truth_desc, sample


def fig_cleaning() -> dict:
    p = 1000
    spikes = [12.0, 7.0, 4.0]
    truth_desc, sample_desc = simulate_spiked(p, spikes, seed=42)

    res = rk.estimate_population_eigenvalues(np.sort(sample_desc), c=C)
    cleaned_desc = np.sort(
        np.concatenate([res["spikes"], res["bulk_population"]])
    )[::-1]
    # Sanity: every series must be genuinely descending so rank i aligns
    # across panels (a previous revision plotted the truth ascending).
    assert truth_desc[0] > truth_desc[1] > truth_desc[2], truth_desc[:4]
    assert sample_desc[0] > sample_desc[1], sample_desc[:3]
    assert cleaned_desc[0] > cleaned_desc[1], cleaned_desc[:3]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    idx = np.arange(1, p + 1)
    # Focus the axis on the bulk; spikes beyond the cap are rendered as a
    # clipped marker with their value annotated, so three tall points at
    # ranks 1-3 don't squash the bulk into an unreadable band.
    y_top = float(res["bulk_edge"]) * 4.0
    ax.plot(idx, truth_desc, "-", color="black", lw=1.5, label="true population")
    ax.plot(idx, sample_desc, ".", color="#9aa5b1", ms=3.5,
            label=f"sample (p={p}, c={C})")
    ax.plot(idx, cleaned_desc, ".", color="#d62728", ms=3.5,
            label="cleaned by shrinkers")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.2, top=y_top)
    for r in range(len(spikes)):
        if truth_desc[r] > y_top:
            ax.plot(r + 1, y_top * 0.93, marker="^", color="black", ms=5)
            ax.annotate(f"{truth_desc[r]:.1f}", (r + 1, y_top * 0.60),
                        ha="center", fontsize=7)
    ax.axhline(res["bulk_edge"], color="#2b6cb0", lw=0.8, ls="--",
               label=f"estimated bulk edge ({res['bulk_edge']:.2f})")
    ax.set_xlabel("rank (descending order)")
    ax.set_ylabel("eigenvalue")
    ax.set_title(f"{len(spikes)} spikes injected, noise σ² = "
                 f"{res['sigma2']:.2f} (true 1.0)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    ax = axes[1]
    eps = 1e-12
    err_sample = np.abs(sample_desc - truth_desc) / np.maximum(truth_desc, eps)
    err_clean = np.abs(cleaned_desc - truth_desc) / np.maximum(truth_desc, eps)
    ax.plot(idx, err_sample, ".", color="#9aa5b1", ms=3.5, label="raw sample")
    ax.plot(idx, err_clean, ".", color="#d62728", ms=3.5, label="cleaned")
    ax.set_yscale("log")
    ax.set_xlabel("rank (descending order)")
    ax.set_ylabel("|error| / true value")
    med_s = float(np.median(err_sample))
    med_c = float(np.median(err_clean))
    ax.set_title(f"median error: {med_s:.1%} → {med_c:.1%}")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "RMT cleaning: recovering the population eigenvalues",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cleaning_quality.png", dpi=150)
    plt.close(fig)

    return {
        "p": p, "c": C, "spikes": spikes,
        "k_detected": int(res["k"]),
        "sigma2_est": float(res["sigma2"]),
        "bulk_edge_est": float(res["bulk_edge"]),
        "median_rel_err_sample": med_s,
        "median_rel_err_cleaned": med_c,
        "spike_estimates": res["spikes"].tolist(),
    }


# ──────────────────────────────────────────────
# Figure 2: runtime vs naive Python / NumPy
# ──────────────────────────────────────────────

def stieltjes_python_naive(lam: np.ndarray, eta: float):
    """Textbook double loop — the 'obvious' Python implementation."""
    p = lam.shape[0]
    out_r = [0.0] * p
    out_i = [0.0] * p
    inv_p = 1.0 / p
    for i in range(p):
        li = lam[i]
        sr = 0.0
        si = 0.0
        for lj in lam:
            d = li - lj
            inv = 1.0 / (d * d + eta * eta)
            sr += d * inv
            si += eta * inv
        out_r[i] = sr * inv_p
        out_i[i] = si * inv_p
    return out_r, out_i


def stieltjes_numpy_chunked(lam: np.ndarray, eta: float, block: int = 128):
    """Same arithmetic, vectorized with plain NumPy broadcasting (no scipy,
    no FFT), chunked so the p×p intermediate never materializes."""
    p = lam.shape[0]
    re = np.empty(p)
    im = np.empty(p)
    eta2 = eta * eta
    inv_p = 1.0 / p
    for a in range(0, p, block):
        d = lam[a:a + block, None] - lam[None, :]
        inv = 1.0 / (d * d + eta2)
        re[a:a + block] = (d * inv).sum(axis=1) * inv_p
        im[a:a + block] = (eta * inv).sum(axis=1) * inv_p
    return re, im


def bench(fn, *args, repeats: int = 3):
    fn(*args)  # warmup
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def fig_runtime(reuse: bool = False) -> dict:
    cached = OUT_DIR / "readme_figures.json"
    if reuse and cached.exists():
        # Re-render only: keep the previously measured numbers so README
        # tables stay in sync with the figure.
        return {"rows": json.loads(cached.read_text())["runtime"]["rows"]}

    # Full 10^0..10^5 range, log-spaced; 50000 caps the sweep.
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
             8192, 16384, 32768, 50000]
    rows = []
    for p in sizes:
        rng = np.random.default_rng(p)
        lam = np.sort(rng.uniform(0.25, 2.25, p)).astype(np.float64)
        eta = 1.0 / np.sqrt(p)
        row = {"p": p}

        if p <= 4096:  # pure Python is O(p^2) interpreted — stop while sane
            row["python_naive"] = bench(stieltjes_python_naive, lam, eta,
                                        repeats=1 if p > 2048 else 3)
        reps = 9 if p <= 256 else 3
        row["numpy"] = bench(stieltjes_numpy_chunked, lam, eta, repeats=reps)
        row["shrinkers_exact_parallel"] = bench(
            lambda l=lam, e=eta: rk.stieltjes_transform(
                l, eta=e, method="blocked_tiled", parallel=True),
            repeats=reps)
        row["shrinkers_chebcode_parallel"] = bench(
            lambda l=lam, e=eta: rk.stieltjes_transform(
                l, eta=e, method="chebcode_fast", parallel=True),
            repeats=reps)
        rows.append(row)
        print(row)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    series = [
        ("python_naive", "#9aa5b1", "o", "Naive Python (double loop)"),
        ("numpy", "#2b6cb0", "s", "Vectorized NumPy"),
        ("shrinkers_exact_parallel", "#d62728", "^", "shrinkers — exact, all cores"),
        ("shrinkers_chebcode_parallel", "#e05252", "v", "shrinkers — treecode, all cores"),
    ]
    for key, color, marker, label in series:
        pts = [(r["p"], r[key]) for r in rows if key in r]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, color=color, ms=5, lw=1.6, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e5)
    ax.set_xlabel("p (number of eigenvalues)")
    ax.set_ylabel("runtime (s)")
    ax.set_title("Full Stieltjes transform — same arithmetic,\ndifferent engines",
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    # annotate headline speedups at the largest shared p
    last_full = next(r for r in reversed(rows) if "python_naive" in r)
    if "python_naive" in last_full:
        speedup_np = last_full["numpy"] / last_full["shrinkers_exact_parallel"]
        speedup_py = last_full["python_naive"] / last_full["shrinkers_exact_parallel"]
        ax.annotate(
            f"at p={last_full['p']}:\n{speedup_py:.0f}× vs naive Python\n{speedup_np:.1f}× vs NumPy",
            xy=(last_full["p"], last_full["shrinkers_exact_parallel"]),
            xytext=(-120, 30), textcoords="offset points",
            fontsize=8.5, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "performance.png", dpi=150)
    plt.close(fig)

    return {"rows": rows}


if __name__ == "__main__":
    reuse = "--reuse" in sys.argv
    # Figure 1's simulation is seeded and cheap -> always re-render it, so
    # styling changes reach the PNG without re-measuring speed.
    cleaning = fig_cleaning()
    runtime = fig_runtime(reuse)
    meta = {
        "machine": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "eta_convention": "eta = 1/sqrt(p)",
        "timing": "median of 3 (naive: 1 rep above p=2048)",
        "date_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }
    payload = {"meta": meta, "runtime": runtime, "cleaning": cleaning}
    (OUT_DIR / "readme_figures.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(meta, indent=2))
    print("figures written to", OUT_DIR)
