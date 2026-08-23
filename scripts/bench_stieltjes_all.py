"""
Benchmark every Stieltjes transform implementation across a range of p, in both
sequential and parallel (Rayon) modes.

For each (p, method, parallelism) it measures:
  - runtime (µs)  — mean over n_runs
  - relative error — max |m_method - m_ref| / max|m_ref| vs the pure-NumPy
    O(p²) reference (exact), computed on the real part.

Outputs:
  - figures/stieltjes_all.png   : runtime vs p, one line per method (sequential)
  - figures/stieltjes_parallel.png : runtime vs p, one line per method (Rayon)
  - figures/stieltjes_error.png : relative error vs p, one line per method
  - figures/stieltjes_best.png  : fastest method per p (runtime) + its error,
    with both the sequential and parallel optimal frontiers
  - figures/stieltjes_data.csv  : all raw (p, method, parallelism, runtime,
    error) rows

Methods (from src/python.rs parse_method):
  naive, autovec, blocked, blocked_autovec, blocked_tiled, blocked_windowed,
  blocked_hybrid, adaptive, fft5, fft3, fft2, treecode, chebcode, ewald, dst,
  auto

Usage:
  .pixi/envs/default/bin/python scripts/bench_stieltjes_all.py [--out-dir figures]
"""
import argparse
import csv
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import shrinkers as rk

# All implementations exposed via the Python API.
METHODS = [
    "reference",  # pure NumPy O(p²) naive — plotted as a solid reference line
    "naive",
    "autovec",
    "blocked",
    "blocked_autovec",
    "blocked_tiled",
    "blocked_windowed",
    "blocked_hybrid",
    "adaptive",
    "fft5",
    "fft3",
    "fft2",
    "treecode",
    "chebcode",
    "ewald",
    "dst",
    "auto",
]

# Methods that have a parallel (Rayon) implementation worth benchmarking.
PARALLEL_METHODS = [
    "blocked",
    "blocked_autovec",
    "blocked_tiled",
    "blocked_windowed",
    "blocked_hybrid",
    "treecode",
    "chebcode",
]

# p values spanning small (fits L1) to large (exceeds L2).
P_VALUES = [100, 200, 400, 800, 1000, 2000, 4000, 8000, 10000, 20000]


# ──────────────────────────────────────────────
#  Pure NumPy reference (exact O(p²))
# ──────────────────────────────────────────────
def _ref_stieltjes(ev, eta):
    """Reference Stieltjes transform m_g(z); returns the real part (the one used
    by the Rust pipeline, z = λ - iη). Broadcasting into a p×p matrix."""
    diff = ev[:, np.newaxis] - ev[np.newaxis, :]  # (p, p)
    denom = diff * diff + eta * eta
    return np.mean(diff / denom, axis=1)


def _bench_ref(ev, eta, n_runs):
    _ref_stieltjes(ev, eta)  # warmup
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ref_stieltjes(ev, eta)
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1e6


def generate_mp_spectrum(p, c=0.5, seed=42):
    rng = np.random.default_rng(seed)
    lo = max(1.0 - np.sqrt(c), 0.01) ** 2
    hi = (1.0 + np.sqrt(c)) ** 2
    ev = lo + rng.uniform(0, 1, p) * (hi - lo) + rng.uniform(0, 0.1, p)
    return np.sort(ev)


def bench_one(ev, eta, method, n_runs, parallelism="seq", precision="f64"):
    """Return mean runtime in µs for a single method at a given p."""
    rk.stieltjes_transform(ev, eta, method, precision, parallelism=parallelism)  # warmup
    t0 = time.perf_counter()
    for _ in range(n_runs):
        rk.stieltjes_transform(ev, eta, method, precision, parallelism=parallelism)
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1e6


def rel_error(ev, eta, method, ref_real, parallelism="seq", precision="f64"):
    """Max relative error of a method's real part vs the exact reference."""
    out = rk.stieltjes_transform(ev, eta, method, precision, parallelism=parallelism)
    real = np.asarray(out["real"])
    denom = np.max(np.abs(ref_real))
    if denom <= 0:
        return float("nan")
    return float(np.max(np.abs(real - ref_real)) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--ps", default=",".join(map(str, P_VALUES)))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ps = [int(x) for x in args.ps.split(",") if x.strip()]

    # runtime_us[m][i], err[m][i] for sequential; runtime_par[m][i] for Rayon.
    runtime = {m: [] for m in METHODS}
    err = {m: [] for m in METHODS}
    runtime_par = {m: [] for m in PARALLEL_METHODS}
    # f32 variants (sequential only for the per-method plots; parallel f32 is
    # not exposed by the Python API, so we only add f32 to the sequential plots).
    runtime_f32 = {m: [] for m in METHODS}
    err_f32 = {m: [] for m in METHODS}

    print(f"{'p':>7}" + "".join(f"{m:>16}" for m in METHODS))
    for p in ps:
        ev = generate_mp_spectrum(p)
        eta = 0.1 / np.sqrt(p)
        # Fewer samples per algo to keep the bench fast; the reference (NumPy)
        # is the slowest, so cap its runs separately.
        n_runs = max(3, int(10_000 / p))
        n_ref = max(1, int(1_000 / p))

        # Exact reference (NumPy) — ground truth for error.
        ref_real = _ref_stieltjes(ev, eta)
        t_ref = _bench_ref(ev, eta, n_ref)
        runtime["reference"].append(t_ref)
        err["reference"].append(0.0)
        runtime_f32["reference"].append(t_ref)
        err_f32["reference"].append(0.0)

        row = [t_ref]
        for m in METHODS[1:]:
            try:
                t = bench_one(ev, eta, m, n_runs)
                e = rel_error(ev, eta, m, ref_real)
            except Exception as ex:  # some methods may fail at certain p
                t, e = float("nan"), float("nan")
                print(f"  [{m} @ p={p}] failed: {ex}")
            runtime[m].append(t)
            err[m].append(e)
            row.append(t)

            # f32 variant (sequential).
            try:
                t32 = bench_one(ev, eta, m, n_runs, precision="f32")
                e32 = rel_error(ev, eta, m, ref_real, precision="f32")
            except Exception as ex:
                t32, e32 = float("nan"), float("nan")
                print(f"  [f32 {m} @ p={p}] failed: {ex}")
            runtime_f32[m].append(t32)
            err_f32[m].append(e32)
        print(f"{p:>7}" + "".join(f"{t:>16.2f}" if t == t else f"{'nan':>16}" for t in row))

        # Parallel (Rayon) versions of the methods that support it.
        for m in PARALLEL_METHODS:
            try:
                t = bench_one(ev, eta, m, n_runs, parallelism="rayon")
            except Exception as ex:
                t = float("nan")
                print(f"  [par {m} @ p={p}] failed: {ex}")
            runtime_par[m].append(t)

    # ── Dump CSV ──────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "stieltjes_data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "method", "parallelism", "precision", "runtime_us", "rel_error"])
        for i, p in enumerate(ps):
            for m in METHODS:
                w.writerow([p, m, "seq", "f64", runtime[m][i], err[m][i]])
                w.writerow([p, m, "seq", "f32", runtime_f32[m][i], err_f32[m][i]])
            for m in PARALLEL_METHODS:
                w.writerow([p, m, "rayon", "f64", runtime_par[m][i], err[m][i]])
    print(f"\nSaved data -> {csv_path}")

    # ── Figure 1: runtime vs p (sequential) ───────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "H", "d", "p", "8", "x", "+", "o", "s"]
    ys_ref = np.array(runtime["reference"])
    ax.plot(ps, ys_ref, "--", color="black", linewidth=1.4, label="reference", markersize=0)
    for m, mk in zip(METHODS[1:], markers[1:]):
        ys = np.array(runtime[m])
        ax.plot(ps, ys, marker=mk, label=m, linewidth=1.2, markersize=5)
    # f32 variants (dashed, lighter) for the same methods.
    for m, mk in zip(METHODS[1:], markers[1:]):
        ys = np.array(runtime_f32[m])
        ax.plot(ps, ys, marker=mk, linestyle="--", color="gray", linewidth=1.0,
                markersize=4, label=f"{m} (f32)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p (number of eigenvalues)")
    ax.set_ylabel("runtime (µs)")
    ax.set_title("Stieltjes transform: runtime vs p (sequential, f64 + f32)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout()
    out1 = os.path.join(args.out_dir, "stieltjes_all.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {out1}")

    # ── Figure 1b: runtime vs p (parallel / Rayon) ────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(ps, ys_ref, "--", color="black", linewidth=1.4, label="reference", markersize=0)
    for m, mk in zip(PARALLEL_METHODS, markers[1:]):
        ys = np.array(runtime_par[m])
        ax.plot(ps, ys, marker=mk, label=m, linewidth=1.2, markersize=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p (number of eigenvalues)")
    ax.set_ylabel("runtime (µs)")
    ax.set_title("Stieltjes transform: runtime vs p (parallel / Rayon, f64)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout()
    out1b = os.path.join(args.out_dir, "stieltjes_parallel.png")
    fig.savefig(out1b, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {out1b}")

    # ── Figure 2: relative error vs p ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    for m, mk in zip(METHODS[1:], markers[1:]):
        ys = np.array(err[m])
        ax.plot(ps, ys, marker=mk, label=m, linewidth=1.2, markersize=5)
    # f32 variants (dashed, lighter).
    for m, mk in zip(METHODS[1:], markers[1:]):
        ys = np.array(err_f32[m])
        ax.plot(ps, ys, marker=mk, linestyle="--", color="gray", linewidth=1.0,
                markersize=4, label=f"{m} (f32)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p (number of eigenvalues)")
    ax.set_ylabel("max relative error (vs NumPy reference)")
    ax.set_title("Stieltjes transform: relative error vs p (f64 + f32)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout()
    out2 = os.path.join(args.out_dir, "stieltjes_error.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {out2}")

    # ── Figure 3: best method per p (runtime) + its error ─────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    best_runtime = []
    best_err = []
    best_names = []
    best_par_runtime = []
    best_par_names = []
    best_par_err = []
    best_f32_runtime = []
    best_f32_names = []
    best_f32_err = []
    for i, p in enumerate(ps):
        # Sequential: fastest among the Rust methods (exclude reference/auto).
        cand = [m for m in METHODS[1:] if m != "auto" and runtime[m][i] == runtime[m][i]]
        if not cand:
            best_names.append("none")
            best_runtime.append(float("nan"))
            best_err.append(float("nan"))
        else:
            bm = min(cand, key=lambda m: runtime[m][i])
            best_names.append(bm)
            best_runtime.append(runtime[bm][i])
            best_err.append(err[bm][i])

        # Parallel: fastest among the Rayon-capable methods. Error is the same
        # as the sequential version of that method (same algorithm, parallelized).
        pcand = [m for m in PARALLEL_METHODS if runtime_par[m][i] == runtime_par[m][i]]
        if not pcand:
            best_par_names.append("none")
            best_par_runtime.append(float("nan"))
            best_par_err.append(float("nan"))
        else:
            pm = min(pcand, key=lambda m: runtime_par[m][i])
            best_par_names.append(pm)
            best_par_runtime.append(runtime_par[pm][i])
            best_par_err.append(err[pm][i])

        # f32 sequential: fastest among the Rust methods in f32.
        fcand = [m for m in METHODS[1:] if m != "auto" and runtime_f32[m][i] == runtime_f32[m][i]]
        if not fcand:
            best_f32_names.append("none")
            best_f32_runtime.append(float("nan"))
            best_f32_err.append(float("nan"))
        else:
            fm = min(fcand, key=lambda m: runtime_f32[m][i])
            best_f32_names.append(fm)
            best_f32_runtime.append(runtime_f32[fm][i])
            best_f32_err.append(err_f32[fm][i])

    # Reference (pure NumPy) as a dashed black line for comparison.
    ax1.plot(ps, np.array(runtime["reference"]), "--", color="black",
             linewidth=1.4, label="reference", markersize=0)
    ax1.plot(ps, best_runtime, "o-", color="tab:blue", linewidth=1.5, markersize=6,
             label="sequential optimal (f64)")
    ax1.plot(ps, best_par_runtime, "s--", color="tab:green", linewidth=1.5, markersize=6,
             label="parallel optimal (f64)")
    ax1.plot(ps, best_f32_runtime, "^-", color="tab:orange", linewidth=1.5, markersize=6,
             label="sequential optimal (f32)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("p")
    ax1.set_ylabel("runtime (µs)")
    ax1.set_title("Fastest method per p")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left")
    for i, p in enumerate(ps):
        ax1.annotate(best_names[i], (p, best_runtime[i]),
                     textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
        ax1.annotate(best_par_names[i], (p, best_par_runtime[i]),
                     textcoords="offset points", xytext=(0, -12), fontsize=7, ha="center",
                     color="tab:green")
        ax1.annotate(best_f32_names[i], (p, best_f32_runtime[i]),
                     textcoords="offset points", xytext=(0, -24), fontsize=7, ha="center",
                     color="tab:orange")

    ax2.plot(ps, best_err, "o-", color="tab:blue", linewidth=1.5, markersize=6,
             label="sequential optimal (f64)")
    ax2.plot(ps, best_par_err, "s--", color="tab:green", linewidth=1.5, markersize=6,
             label="parallel optimal (f64)")
    ax2.plot(ps, best_f32_err, "^-", color="tab:orange", linewidth=1.5, markersize=6,
             label="sequential optimal (f32)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("p")
    ax2.set_ylabel("max relative error")
    ax2.set_title("Error of the fastest method per p")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out3 = os.path.join(args.out_dir, "stieltjes_best.png")
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    print(f"Saved figure -> {out3}")


if __name__ == "__main__":
    main()
