#!/usr/bin/env python3
"""Runtime-vs-p Pareto charts from docs/pareto/*.json.

Produces, for each parallelism (seq, rayon):
  - the full method set,
  - a high-accuracy cut (methods whose worst-case rel L2 error across all
    measured sizes stays <= ACC_CUT),
and one combined grid figure (parallelism x accuracy band).

Usage: python3 scripts/plot_runtime_vs_p.py [bench_after.json] [bench_before.json]
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ACC_CUT = 1e-8
OUT = "docs/pareto"


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path) as fh:
            data = json.load(fh)
        label = data.get("meta", {}).get("label", path)
        for r in data["rows"]:
            rows.append({**r, "label": label})
    return rows


def latest_per_cell(rows):
    """Keep the newest measurement per (label-ordering, method, par, p).

    Rows from earlier files are shadowed by later files (campaign order).
    """
    priority = {}
    for i, r in enumerate(rows):
        priority[r["label"]] = i
    best = {}
    for r in rows:
        key = (r["method"], r["par"], r["p"])
        if key not in best or priority[r["label"]] > priority[best[key]["label"]]:
            best[key] = r
    return list(best.values())


def main():
    paths = sys.argv[1:] or [f"{OUT}/bench_after.json"]
    rows = latest_per_cell(load_rows(paths))
    pars = sorted({r["par"] for r in rows})
    methods = sorted({r["method"] for r in rows})

    def max_err(m):
        return max(r["err"] for r in rows if r["method"] == m)

    bands = {
        "all_methods": lambda m: True,
        f"err_le_{ACC_CUT:g}": lambda m: max_err(m) <= ACC_CUT,
        f"err_le_1e-5": lambda m: max_err(m) <= 1e-5,
    }

    for par in pars:
        sub = [r for r in rows if r["par"] == par]
        # Full chart.
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for m in methods:
            pts = sorted((r["p"], r["ms"]) for r in sub if r["method"] == m)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=3.5, label=m)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("taille du portefeuille $p$")
        ax.set_ylabel("runtime (ms)")
        ax.set_title(f"Stieltjes — runtime vs p ({par})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        out = f"{OUT}/runtime_vs_p_{par}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("wrote", out)

    # Grid: parallelism x accuracy band.
    n_par, n_band = len(pars), len(bands)
    fig, axes = plt.subplots(
        n_par, n_band, figsize=(4.2 * n_band, 3.9 * n_par), squeeze=False
    )
    for pi, par in enumerate(pars):
        sub = [r for r in rows if r["par"] == par]
        for bi, (band, keep) in enumerate(bands.items()):
            ax = axes[pi][bi]
            for m in methods:
                if not keep(m):
                    continue
                pts = sorted((r["p"], r["ms"]) for r in sub if r["method"] == m)
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", ms=3, label=m)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{par} — {band}", fontsize=10)
            ax.grid(True, which="both", alpha=0.25)
            if pi == n_par - 1:
                ax.set_xlabel("$p$", fontsize=9)
            if bi == 0:
                ax.set_ylabel("runtime (ms)", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=min(8, len(labels)), loc="lower center")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = f"{OUT}/runtime_vs_p_grid.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
