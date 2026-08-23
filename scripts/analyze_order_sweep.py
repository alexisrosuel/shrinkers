#!/usr/bin/env python3
"""Analyze the order_sweep JSONL dump.

Reports, per (p, order):
  * the empirical convergence slope  d(log err)/d(log m)  over the pre-floor
    region (theory: -N for an N-point stencil),
  * the saturation floor (wrap-around/periodization limit),
and, from the padding experiment, the exponent of the floor
  err_floor ~ pad_mult ** (-alpha).

Finally extracts the cheapest measured (order, m) per target error and
cross-references ChebCode runtimes from a pareto bench JSON if available.

Usage: python3 scripts/analyze_order_sweep.py docs/pareto/order_sweep.jsonl \
           [docs/pareto/bench_after.json]
"""
import json
import math
import sys
from collections import defaultdict

def fit_slope(xs, ys):
    """Least-squares slope of log(ys) vs log(xs) on the strictly-decreasing head."""
    pts = []
    prev = None
    for x, y in zip(xs, ys):
        if y > 0 and (prev is None or y < prev * 0.999):
            pts.append((math.log(x), math.log(y)))
            prev = y
        else:
            break
    if len(pts) < 2:
        return float("nan"), len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    n = len(pts)
    return (n * sxy - sx * sy) / (n * sxx - sx * sx), n


def main():
    rows = []
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    grid = [r for r in rows if r["kind"] == "grid"]
    pad = [r for r in rows if r["kind"] == "pad"]

    by_series = defaultdict(list)
    for r in grid:
        by_series[(r["p"], r["order"])].append(r)

    print("=" * 78)
    print("A. ERROR vs GRID SIZE  (slope should approach -N, the stencil order)")
    print("=" * 78)
    floors = {}
    for (p, order), rs in sorted(by_series.items()):
        rs = sorted(rs, key=lambda r: (r["m"] is not None, r["m"] or 0))
        print(f"\n-- p={p}  order={order}")
        for r in rs:
            mtxt = "auto" if r["m"] is None else str(r["m"])
            print(f"   m={mtxt:>7}  err={r['err']:.3e}  ms={r['ms']:8.3f}")
        ms_ = [r["m"] for r in rs if r["m"] is not None]
        es = [r["err"] for r in rs if r["m"] is not None]
        slope, npts = fit_slope(ms_, es)
        floor = min(es)
        floors[(p, order)] = floor
        print(f"   slope={slope:6.2f} over {npts} pts   floor={floor:.3e}")

    print()
    print("=" * 78)
    print("B. FLOOR vs PADDING  (err_floor ~ pad**-alpha)")
    print("=" * 78)
    by_pad = defaultdict(list)
    for r in pad:
        by_pad[r["order"]].append(r)
    pad_alpha = {}
    for order, rs in sorted(by_pad.items()):
        rs = sorted(rs, key=lambda r: r["pad_mult"])
        print(f"\n-- order={order}  (m=262144, p=5000)")
        for r in rs:
            print(f"   pad_mult={r['pad_mult']:7.0f}  err={r['err']:.3e}  ms={r['ms']:8.3f}")
        xs = [r["pad_mult"] for r in rs]
        ys = [r["err"] for r in rs]
        alpha, npts = fit_slope(xs, ys)
        pad_alpha[order] = alpha
        print(f"   alpha={-alpha:5.2f} over {npts} pts (floor ~ pad^-alpha)")

    # Best measured config per target error, per p.
    cheb = {}
    if len(sys.argv) > 2:
        try:
            with open(sys.argv[2]) as f:
                bench = json.load(f)
            for r in bench["rows"]:
                if r["method"] == "chebcode" and r["par"] == "seq":
                    cheb[r["p"]] = (r["ms"], r["err"])
        except OSError:
            pass

    print()
    print("=" * 78)
    print("C. CHEAPEST MEASURED (order, m) PER TARGET ERROR  [seq timings]")
    print("=" * 78)
    targets = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]
    ps = sorted({r["p"] for r in grid})
    for p in ps:
        rows_p = [r for r in grid if r["p"] == p and r["err"] < 1.0]
        print(f"\n-- p={p}")
        header = f"   {'target':>8} | {'best (order, m)':>26} | {'ms':>8} | {'err':>9}"
        if p in cheb:
            header += f" | {'chebcode ms (err)':>22}"
        print(header)
        for t in targets:
            ok = [r for r in rows_p if r["err"] <= t]
            if not ok:
                print(f"   {t:8.0e} | {'— unreachable in sweep':>26} |")
                continue
            best = min(ok, key=lambda r: (r["m"] is None, r["m"] or 0))
            mtxt = "auto" if best["m"] is None else str(best["m"])
            line = f"   {t:8.0e} | {best['order'] + ', m=' + mtxt:>26} | {best['ms']:8.3f} | {best['err']:9.2e}"
            if p in cheb:
                cms, cerr = cheb[p]
                line += f" | {cms:9.3f} ms ({cerr:.1e})"
            print(line)

    print()
    print("=" * 78)
    print("D. SUMMARY")
    print("=" * 78)
    for (p, order), fl in sorted(floors.items()):
        print(f"   p={p:<6} {order:>8}: floor={fl:.3e}")
    for order, a in pad_alpha.items():
        print(f"   floor(pad) exponent [{order}]: alpha={-a:.2f}")


if __name__ == "__main__":
    main()
