#!/usr/bin/env python3
"""Small-p crossover chart: O(p^2) exact kernel vs ChebCode treecode.

Reads docs/pareto/small_p.json (produced by
`cargo run --release --example small_p_crossover`) and plots per-call runtime
in microseconds, log-log, with the measured crossover region highlighted.

Usage: python3 scripts/plot_crossover.py [small_p.json]
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "docs/pareto"

STYLE = {
    # (method, par): (color, linestyle)
    ("exact", "seq"): ("#333333", "-"),
    ("cheb_fast", "seq"): ("#d62728", "-"),
    ("cheb_default", "seq"): ("#1f77b4", "-"),
    ("cheb_xtreme", "seq"): ("#2ca02c", "-"),
    ("exact", "ray"): ("#333333", "--"),
    ("cheb_fast", "ray"): ("#d62728", "--"),
    ("cheb_default", "ray"): ("#1f77b4", "--"),
    ("cheb_xtreme", "ray"): ("#2ca02c", "--"),
}

LABEL = {
    "exact": "exact O(p²) tiled",
    "cheb_fast": "chebcode_fast",
    "cheb_default": "chebcode",
    "cheb_xtreme": "chebcode_xtreme",
}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/small_p.json"
    with open(path) as fh:
        rows = json.load(fh)["rows"]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for key, (color, ls) in STYLE.items():
        pts = sorted(
            (r["p"], r["us"]) for r in rows
            if (r["method"], r["par"]) == key
        )
        if not pts:
            continue
        xs, ys = zip(*pts)
        par_label = "" if ls == "-" else " (rayon)"
        ax.plot(xs, ys, color=color, linestyle=ls, marker="o", ms=3,
                lw=1.8 if ls == "-" else 1.0, alpha=1.0 if ls == "-" else 0.55,
                label=LABEL[key[0]] + par_label)

    # Crossover region measured across two independent sessions:
    # fast/default flip between p=300 and p=400, xtreme between 500 and 600.
    for x0, x1, txt in [(300, 400, "bascule ≈350"), (500, 600, "bascule ≈550")]:
        ax.axvspan(x0, x1, color="orange", alpha=0.15, lw=0)
        ax.annotate(txt, xy=((x0 * x1) ** 0.5, 3.0), fontsize=8,
                    ha="center", color="#a05a00")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("taille du portefeuille $p$")
    ax.set_ylabel("runtime (µs / appel)")
    ax.set_title("Petit p : bascule O(p²) → treecode (traits pleins = seq, défaut Python)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = f"{OUT}/crossover_small_p.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
