#!/usr/bin/env python3
"""Plot before/after Pareto frontiers from pareto_data JSON dumps.

Usage:
    MPLCONFIGDIR=/tmp/mpl scripts/plot_pareto.py docs/pareto/bench_before.json \
        docs/pareto/bench_after.json [--out docs/pareto]

Produces one figure per parallelism mode: log-log runtime vs relative error,
one panel per problem size, with the Pareto staircase highlighted.
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    with open(path) as f:
        d = json.load(f)
    rows = {}
    for r in d["rows"]:
        rows.setdefault((r["par"], r["p"]), []).append(r)
    return d, rows


def pareto_staircase(points):
    """points: list of (err, ms). Lower-left frontier; returns sorted steps."""
    pts = sorted(points)  # by err asc
    best = []
    best_ms = float("inf")
    for err, ms in pts:
        if ms < best_ms:
            best.append((err, ms))
            best_ms = ms
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--out", default="docs/pareto")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    db, rb = load(args.before)
    da, ra = load(args.after)

    sizes = sorted({p for (_, p) in rb.keys()} & {p for (_, p) in ra.keys()})
    pars = ["seq", "rayon"]  # legacy token; new dumps say "parallel"
    colors = plt.get_cmap("tab10").colors
    method_color = {}
    all_methods = []
    for (_, p), rs in list(rb.items()) + list(ra.items()):
        for r in rs:
            if r["method"] not in method_color:
                method_color[r["method"]] = colors[len(method_color) % len(colors)]
                all_methods.append(r["method"])

    for par in pars:
        fig, axes = plt.subplots(
            1, len(sizes), figsize=(4.1 * len(sizes), 3.9), sharey=True
        )
        if len(sizes) == 1:
            axes = [axes]
        for ax, p in zip(axes, sizes):
            for src_rows, style, label in [
                (rb, dict(marker="o", facecolors="none", color="#c0392b"), "before"),
                (ra, dict(marker="o", color="#1e8449"), "after"),
            ]:
                _ = label
                pts = src_rows.get((par, p), [])
                for r in pts:
                    ax.scatter(
                        r["err"], r["ms"], s=34, alpha=0.85,
                        **style, linewidths=1.4,
                    )
                # staircase through Pareto-optimal points of this snapshot
                st = pareto_staircase([(r["err"], r["ms"]) for r in pts])
                if st:
                    errs, mss = zip(*st)
                    ax.step(errs, mss, where="post", alpha=0.55,
                            color=style.get("color"), linewidth=1.6)
                _ = label
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"p = {p}", fontsize=11)
            ax.set_xlabel("relative error (L2)", fontsize=9)
            ax.grid(True, which="both", alpha=0.25)
        axes[0].set_ylabel("runtime (ms)", fontsize=9)
        handles = [
            plt.Line2D([], [], linestyle="", marker="o", mfc="none",
                       color="#c0392b", label="before (HEAD)")
            , plt.Line2D([], [], linestyle="", marker="o",
                         color="#1e8449", label="after (working tree)")
        ]
        fig.suptitle(
            f"Pareto frontier — {par.upper()} "
            f"(open red = before, filled green = after)",
            fontsize=12,
        )
        fig.legend(handles=handles, loc="lower right", ncol=2, fontsize=9)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        out = outdir / f"pareto_{par}.png"
        fig.savefig(out, dpi=140)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
