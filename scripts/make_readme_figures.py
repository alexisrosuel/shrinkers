"""Generate the three README front-page figures with measured data.

Figure 1 — what the cleaning does: sample vs cleaned vs true population
eigenvalues under a spiked Marchenko-Pastur model (diagonal population,
Gaussian samples).

Figure 2 — how fast: shrinkers vs a naive pure-Python double loop vs a
vectorized NumPy baseline (chunked broadcasting; NO scipy, NO FFT — the
comparison isolates "same arithmetic, better engine").

Figure 3 — cleaning the matrix itself, not just its spectrum: the real
symmetric entry point on a spiked correlation matrix and the complex Hermitian
entry point on a spectral coherence matrix.

Outputs:
  docs/img/cleaning_quality.png
  docs/img/performance.png
  docs/img/correlation_cleaning.png
  docs/img/readme_figures.json   (measured numbers behind the three figures)

Run: .pixi/envs/default/bin/python scripts/make_readme_figures.py
"""

from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np
from _common import DOCS_IMG, numpy_stieltjes, savefig, setup_mpl

import shrinkers as rk

plt = setup_mpl()
OUT_DIR = DOCS_IMG

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

    # The cleaned estimates must stay in the SAMPLE-eigenvalue order, not be
    # sorted by value: `spikes` is descending and `bulk_population` is parallel
    # to the *ascending* `bulk_sample`, so reversing the latter lines every
    # estimate up with the sample eigenvalue it came from. Sorting the cleaned
    # values (what this figure used to do) instead turns the red curve into an
    # order statistic whose features sit at different ranks from the sample's,
    # which reads as a horizontal shift against the grey and black curves. The
    # multiset of cleaned values — hence the median error — is identical either
    # way; only the rank alignment changes.
    k = len(spikes)
    assert res["k"] == k, f"expected {k} detected spikes, got {res['k']}"
    assert np.allclose(res["spike_sample"], sample_desc[:k])
    cleaned_desc = np.concatenate(
        [res["spikes"], np.asarray(res["bulk_population"])[::-1]]
    )
    assert cleaned_desc.shape == sample_desc.shape
    # Truth and sample are genuinely descending; the cleaned series follows the
    # sample's ranks, so it is only descending up to the estimator's noise.
    assert truth_desc[0] > truth_desc[1] > truth_desc[2], truth_desc[:4]
    assert sample_desc[0] > sample_desc[1], sample_desc[:3]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    idx = np.arange(1, p + 1)
    # Bulk as a line starting AFTER the spike ranks (no vertical jump in the
    # trace); the spikes themselves are isolated scatter markers.
    ax.plot(idx[k:], truth_desc[k:], "-", color="black", lw=1.5,
            label="true population")
    ax.plot(idx[:k], truth_desc[:k], "D", color="black", ms=5)
    ax.plot(idx, sample_desc, ".", color="#9aa5b1", ms=3.5,
            label=f"sample (p={p}, c={C})")
    ax.plot(idx, cleaned_desc, ".", color="#d62728", ms=3.5,
            label="cleaned by shrinkers")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.2)
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
    savefig(fig, OUT_DIR / "cleaning_quality.png", dpi=150)

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
# Figure 3: cleaning a correlation matrix (real and complex Hermitian)
# ──────────────────────────────────────────────

def simulate_factor_correlation(p: int, n: int, k: int, noise: float, seed: int):
    """A spiked *correlation* matrix with a known population.

    Three factor loadings plus a diagonal residual build a population
    correlation matrix with unit diagonal; ``n`` Gaussian draws give its sample
    correlation. Returns ``(sample, population)``.
    """
    rng = np.random.default_rng(seed)
    loadings = rng.standard_normal((p, k)) / np.sqrt(p)
    sigma = loadings @ loadings.T + noise * np.eye(p)
    d = np.sqrt(np.diag(sigma))
    pop = sigma / np.outer(d, d)
    z = rng.standard_normal((n, p)) @ np.linalg.cholesky(pop).T
    return np.ascontiguousarray(np.corrcoef(z, rowvar=False)), pop


def simulate_complex_correlation(
    m: int, b: int, modes: list[float], seed: int
):
    """A complex Hermitian correlation matrix with a known population.

    A random unitary basis carries ``modes`` coherent eigenvalues on top of a
    unit bulk; ``b`` complex Gaussian vectors drawn from that population are
    averaged into the sample coherence matrix (the smoothed-periodogram
    construction in the API docs, with ``c = m / b``). Returns
    ``(sample, population)``.
    """
    rng = np.random.default_rng(seed)
    z0 = (
        rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    ) / np.sqrt(2.0)
    q, _ = np.linalg.qr(z0)
    evals = np.concatenate([np.asarray(modes), np.ones(m - len(modes))])
    sigma = (q * evals) @ q.conj().T
    d = np.sqrt(np.real(np.diag(sigma)))
    pop = sigma / np.outer(d, d)
    w = (
        rng.standard_normal((b, m)) + 1j * rng.standard_normal((b, m))
    ) / np.sqrt(2.0)
    z = w @ np.linalg.cholesky(pop).T
    s = z.conj().T @ z / b
    d = np.sqrt(np.real(np.diag(s)))
    return np.ascontiguousarray(s / np.outer(d, d)), pop


def fig_correlation_cleaning() -> dict:
    # ── Panel A: real symmetric correlation matrix (3-factor population) ──
    p, c_real = 400, C
    corr, pop_real = simulate_factor_correlation(p, round(p / c_real), 3, 0.5, seed=7)
    real = rk.clean_correlation_matrix(corr, c=c_real)

    # ── Panel B: complex Hermitian correlation matrix (2 coherent modes) ──
    m, b = 200, 1000
    c_cplx = m / b
    coh, pop_cplx = simulate_complex_correlation(m, b, [6.0, 3.0], seed=11)
    cplx = rk.clean_correlation_matrix_complex(coh, c=c_cplx)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    stats: dict[str, dict[str, object]] = {}
    panels = [
        (axes[0], corr, pop_real, real, c_real, real["sigma2"],
         f"real symmetric (p={p}, c={c_real})", "real"),
        (axes[1], coh, pop_cplx, cplx, c_cplx, cplx["sigma2"],
         f"complex Hermitian coherence (M={m}, B={b}, c={c_cplx:.2f})", "complex"),
    ]
    for ax, sample_mat, pop_mat, result, c, sigma2, title, key in panels:
        truth = np.sort(np.linalg.eigvalsh(pop_mat))[::-1]
        sample = np.sort(np.linalg.eigvalsh(sample_mat))[::-1]
        cleaned = np.sort(np.asarray(result["eigenvalues"]))[::-1]
        idx = np.arange(1, sample.size + 1)

        edge = (1.0 + np.sqrt(c)) ** 2 * sigma2
        ax.plot(idx, truth, "-", color="black", lw=1.5, label="true population")
        ax.plot(idx, sample, ".", color="#9aa5b1", ms=3.5, label="sample")
        ax.plot(idx, cleaned, ".", color="#d62728", ms=3.5,
                label="cleaned by shrinkers")
        ax.axhline(edge, color="#2b6cb0", lw=0.9, ls="--",
                   label=f"MP bulk edge ({edge:.2f})")
        ax.set_yscale("log")
        ax.set_xlabel("rank (descending order)")
        ax.set_ylabel("eigenvalue")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        err_sample = float(
            np.linalg.norm(sample_mat - pop_mat) / np.linalg.norm(pop_mat)
        )
        err_clean = float(
            np.linalg.norm(result["covariance"] - pop_mat) / np.linalg.norm(pop_mat)
        )
        ax.text(0.02, 0.03,
                f"rel. Frobenius error: {err_sample:.2f} → {err_clean:.2f}",
                transform=ax.transAxes, fontsize=8.5, color="#333333")
        stats[key] = {
            "c": float(c),
            "sigma2_est": float(sigma2),
            "mp_edge": float(edge),
            "rel_frobenius_sample": err_sample,
            "rel_frobenius_cleaned": err_clean,
            "max_sample_eig": float(sample[0]),
            "max_clean_eig": float(cleaned[0]),
            "max_true_eig": float(truth[0]),
        }

    fig.suptitle(
        "Cleaning a correlation matrix: RIE + eigenvector-overlap correction",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, OUT_DIR / "correlation_cleaning.png", dpi=150)

    stats["real"].update({"p": p, "factors": 3, "noise": 0.5})
    stats["complex"].update({"channels": m, "bins": b, "modes": [6.0, 3.0]})
    return stats


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
    """Same arithmetic as `_common.numpy_stieltjes`, in the chunked form that
    never materializes the p×p intermediate (no scipy, no FFT)."""
    return numpy_stieltjes(lam, eta, block=block)


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
            arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 0.8})

    fig.tight_layout()
    savefig(fig, OUT_DIR / "performance.png", dpi=150)

    return {"rows": rows}


if __name__ == "__main__":
    reuse = "--reuse" in sys.argv
    # Figures 1 and 3's simulations are seeded and cheap -> always re-render
    # them, so styling changes reach the PNG without re-measuring speed.
    cleaning = fig_cleaning()
    correlation = fig_correlation_cleaning()
    runtime = fig_runtime(reuse)
    meta = {
        "machine": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "eta_convention": "eta = 1/sqrt(p)",
        "timing": "median of 3 (naive: 1 rep above p=2048)",
        "date_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }
    payload = {
        "meta": meta,
        "runtime": runtime,
        "cleaning": cleaning,
        "correlation_cleaning": correlation,
    }
    (OUT_DIR / "readme_figures.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(meta, indent=2))
    print("figures written to", OUT_DIR)
