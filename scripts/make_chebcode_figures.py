"""Illustration figures for docs/chebcode_algorithms.md.

Generates deterministic schematics (no benchmark data) into docs/img/:

* chebcode_tree.png            - recursive interval splits + Chebyshev panels
* chebcode_traversal.png       - one query's accept/recurse/leaf decision per level
* chebcode_equivalent_densities.png - sources -> panel weights, curves agree
* chebcode_presets.png         - measured error class per preset (log scale)

All text English. Run: .pixi/envs/default/bin/python scripts/make_chebcode_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "img")

GREEN = "#2e7d32"
ORANGE = "#ef6c00"
BLUE = "#1565c0"
GREY = "#9aa5b1"


def cheb_nodes(n: int) -> np.ndarray:
    return np.cos(np.pi * np.arange(n) / max(n - 1, 1))


# ────────────────────────────────────────────────────────────────
# Minimal mirror of the Rust tree (interval-midpoint splits) so the
# traversal figure reflects real decisions, not a cartoon.
# ────────────────────────────────────────────────────────────────


def build_tree(lo: float, hi: float, idx_lo: int, idx_hi: int, src: np.ndarray,
               n_nodes: np.ndarray, leaf_cap: int, depth: int = 0):
    """Returns dict tree over src[idx_lo:idx_hi] covering [lo, hi]."""
    count = idx_hi - idx_lo
    node = {
        "lo": lo, "hi": hi,
        "src": src[idx_lo:idx_hi],
        "nodes": lo + (hi - lo) / 2 * (1 + n_nodes),
        "leaf": count <= leaf_cap,
        "depth": depth,
    }
    if count > leaf_cap:
        mid = lo + 0.5 * (hi - lo)
        seg = src[idx_lo:idx_hi]
        split = int(np.searchsorted(seg, mid, side="right"))
        if split == 0 or split == count:
            split = count // 2
        node["l"] = build_tree(lo, seg[min(split, count - 1)], idx_lo, idx_lo + split,
                               src, n_nodes, leaf_cap, depth + 1)
        node["r"] = build_tree(seg[max(split - 1, 0)] if split > 0 else lo,
                               hi, idx_lo + split, idx_hi, src, n_nodes, leaf_cap, depth + 1)
    return node


def traverse(node, x: float, eta: float, theta: float, out: list, level: int = 0):
    """Record (level, interval, decision) along the query's path."""
    cl = max(node["lo"] - x, x - node["hi"], 0.0)
    d_sq = cl * cl + eta * eta
    hw_sq = ((node["hi"] - node["lo"]) / 2) ** 2
    if node["leaf"]:
        out.append((level, node["lo"], node["hi"], "leaf"))
        return
    if hw_sq < theta * theta * d_sq:
        out.append((level, node["lo"], node["hi"], "accepted"))
        return
    out.append((level, node["lo"], node["hi"], "recursed"))
    traverse(node["l"], x, eta, theta, out, level + 1)
    traverse(node["r"], x, eta, theta, out, level + 1)


def fig_tree_and_traversal() -> None:
    rng = np.random.default_rng(42)
    c = 0.5
    lo, hi = (1 - c ** 0.5) ** 2, (1 + c ** 0.5) ** 2
    src = np.sort(np.concatenate([
        lo + rng.random(120) * (hi - lo),
        [hi * 2.3, lo * 0.35],
    ]))
    n_panel = 9
    tree = build_tree(src.min(), src.max(), 0, len(src), src,
                      cheb_nodes(n_panel), leaf_cap=16)

    # Figure 1: static tree layout (levels of intervals).
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    levels: dict[int, list] = {}

    def collect(node):
        levels.setdefault(node["depth"], []).append(node)
        if not node["leaf"]:
            collect(node["l"])
            collect(node["r"])

    collect(tree)
    max_depth = max(levels)
    for d, nodes in sorted(levels.items()):
        y = -d
        for nd in nodes:
            color = BLUE if nd["leaf"] else GREY
            ax.plot([nd["lo"], nd["hi"]], [y, y], lw=6, color=color,
                    solid_capstyle="butt", alpha=0.85)
            if not nd["leaf"]:
                xs = nd["nodes"]
                ax.plot(xs, np.full_like(xs, y), "|", ms=4, color="#37474f")
    ax.set_yticks(range(0, -max_depth - 1, -1))
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("depth")
    ax.set_title("ChebCode tree: intervals per depth; '|' = Chebyshev panel nodes; "
                 "blue = exact leaves", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chebcode_tree.png"), dpi=150)
    plt.close(fig)

    # Figure 2: one query's decisions.
    x_query = float(src.max()) + 0.35
    eta = 0.1 / np.sqrt(len(src))
    decisions: list = []
    traverse(tree, x_query, eta, 0.5, decisions)
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    style = {"accepted": GREEN, "recursed": ORANGE, "leaf": BLUE}
    seen_levels = sorted({d[0] for d in decisions})
    for level, lo_i, hi_i, kind in decisions:
        y = -level
        ax.plot([lo_i, hi_i], [y, y], lw=8, color=style[kind],
                solid_capstyle="butt", alpha=0.9)
    ax.plot([x_query], [0.4], "v", ms=9, color="#b71c1c")
    ax.annotate("query z", (x_query, 0.4), textcoords="offset points",
                xytext=(6, -2), fontsize=8)
    ax.set_yticks([-d for d in seen_levels])
    ax.set_yticklabels([str(d) for d in seen_levels])
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("depth")
    ax.set_title(f"One query's path: green = far-field accepted (theta test), "
                 f"orange = recursed, blue = exact leaf   (eta={eta:.3f})", fontsize=9)
    handles = [plt.Line2D([], [], lw=6, color=style[k], label=k)
               for k in ("accepted", "recursed", "leaf")]
    ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chebcode_traversal.png"), dpi=150)
    plt.close(fig)


def fig_equivalent_densities() -> None:
    """Real computation: barycentric row update vs direct kernel sum."""
    rng = np.random.default_rng(7)
    lo, hi = -1.0, 1.0
    n_panel = 9
    t = lo + (hi - lo) / 2 * (1 + cheb_nodes(n_panel))
    lam_w = np.array([0.5 if j in (0, n_panel - 1) else (1 if j % 2 == 0 else -1)
                      for j in range(n_panel)])
    src = np.sort(rng.random(60)) * (hi - lo) + lo

    # Equivalent densities: same normalized barycentric row update as fill_weights.
    w = np.zeros(n_panel)
    for x in src:
        d = x - t
        hit = np.where(d == 0)[0]
        if len(hit):
            w[hit[0]] += 1.0
            continue
        v = lam_w / d
        w += v / v.sum()
    w *= len(src) / w.sum()

    # Evaluate only where the theta test would ACCEPT this panel:
    # hw^2 (=1) < theta^2 * ((z-hi)^2 + eta^2) requires z - hi >= ~2.
    z = np.linspace(hi + 2.5, hi + 6.0, 400)
    eta = 0.05
    exact = ((z[:, None] - src[None, :]) / ((z[:, None] - src[None, :]) ** 2 + eta ** 2)).sum(axis=1)
    approx = ((z[:, None] - t[None, :]) / ((z[:, None] - t[None, :]) ** 2 + eta ** 2) @ w)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.4, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.4]})
    ax0.stem(src, np.ones_like(src), markerfmt=",", basefmt=" ", linefmt=f"C0-")
    ax0.stem(t, w / w.max() * 1.0, markerfmt="D", basefmt=" ", linefmt=f"C3-", )
    ax0.set_ylabel("sources / weights")
    ax0.legend(["60 sources (exact side)", "panel weights w_j (scaled)"],
               fontsize=8, loc="upper left")
    ax0.set_title("Equivalent densities: many sources -> few weighted Chebyshev nodes",
                  fontsize=9)

    ax1.plot(z, exact, lw=2, color=BLUE, label="direct sum over sources")
    ax1.plot(z, approx, "--", lw=2, color="#d62728", label="sum over panel nodes")
    ax1.set_xlabel("evaluation point z")
    ax1.set_ylabel("Re S(z)")
    ax1.legend(fontsize=8, loc="upper right")
    err = np.max(np.abs(exact - approx)) / np.max(np.abs(exact))
    ax1.set_title(f"max relative deviation {err:.1e}  (well-separated panel)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chebcode_equivalent_densities.png"), dpi=150)
    plt.close(fig)


def fig_presets() -> None:
    presets = [
        ("chebcode_fast\n(theta .50, n 9)", 1e-8),
        ("chebcode\n(theta .50, n 11)", 5e-10),
        ("chebcode_balanced\n(theta .55, n 11)", 3e-10),
        ("chebcode_xtreme\n(theta .25, n 11)", 1e-12),
    ]
    names = [p[0] for p in presets]
    errs = [p[1] for p in presets]
    colors = ["#ef6c00", BLUE, GREEN, "#6a1b9a"]
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    bars = ax.bar(names, errs, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_ylim(1e-13, 1e-6)
    ax.set_ylabel("rel-L2 error class (measured)")
    ax.set_title("ChebCode* presets: accuracy classes at their shipped parameters",
                 fontsize=9)
    for bar, e in zip(bars, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, e * 1.6, f"{e:.0e}",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chebcode_presets.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    fig_tree_and_traversal()
    fig_equivalent_densities()
    fig_presets()
    print("figures written:", sorted(
        f for f in os.listdir(OUT) if f.startswith("chebcode_")))


if __name__ == "__main__":
    sys.exit(main())
