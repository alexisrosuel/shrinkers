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

from _common import savefig, setup_mpl

plt = setup_mpl()

# The dumps have used several tokens for the same thing over the campaigns
# ("rayon" before, "ray"/"parallel" after). Normalising here keeps the two
# snapshots in the same panels instead of silently drawing empty ones.
PAR_ALIASES = {
    "seq": "seq",
    "sequential": "seq",
    "ray": "rayon",
    "rayon": "rayon",
    "parallel": "rayon",
}


def load(path):
    """Read a dump and group its rows by (normalised par, p)."""
    with open(path) as f:
        d = json.load(f)
    rows = {}
    for r in d["rows"]:
        par = PAR_ALIASES.get(r["par"], r["par"])
        rows.setdefault((par, r["p"]), []).append(r)
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

    _, rb = load(args.before)
    _, ra = load(args.after)

    sizes = sorted({p for (_, p) in rb} & {p for (_, p) in ra})
    # Sequential first, then the threaded mode — the two panels the README
    # artefacts are named after (`pareto_seq.png`, `pareto_rayon.png`).
    pars = sorted(
        {par for (par, _) in rb} | {par for (par, _) in ra},
        key=lambda t: (t != "seq", t),
    )

    for par in pars:
        fig, axes = plt.subplots(
            1, len(sizes), figsize=(4.1 * len(sizes), 3.9), sharey=True
        )
        if len(sizes) == 1:
            axes = [axes]
        for ax, p in zip(axes, sizes):
            for src_rows, style in [
                (rb, {"marker": "o", "facecolors": "none", "color": "#c0392b"}),
                (ra, {"marker": "o", "color": "#1e8449"}),
            ]:
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
                            color=style["color"], linewidth=1.6)
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
        savefig(fig, outdir / f"pareto_{par}.png", dpi=140)


if __name__ == "__main__":
    main()
