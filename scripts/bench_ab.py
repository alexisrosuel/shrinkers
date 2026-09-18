#!/usr/bin/env python3
"""Interleaved before/after benchmark driver.

Runs two builds of `examples/measure_runtime_audit compare <p>` alternately
(so machine load is shared) and reports the per-key median for each build
plus the ratio. Writes a small markdown table to stdout.

Usage: scripts/bench_ab.py /tmp/audit_before /tmp/audit_after 10000 25000 50000
"""
from __future__ import annotations

import statistics
import subprocess
import sys
from collections import defaultdict

ROUNDS = 3


def run(binary: str, p: int) -> dict[str, float]:
    out = subprocess.run(
        [binary, "compare", str(p)], capture_output=True, text=True, check=True
    ).stdout
    vals: dict[str, float] = {}
    for line in out.splitlines():
        if "\t" not in line:
            continue
        key, val = line.split("\t")
        vals[key.strip()] = float(val)
    return vals


def main() -> None:
    before, after = sys.argv[1], sys.argv[2]
    sizes = [int(x) for x in sys.argv[3:]] or [10_000, 50_000]

    samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"before": [], "after": []})
    for p in sizes:
        # One untimed warm-up round per build, then the interleaved rounds.
        run(before, p)
        run(after, p)
        for _ in range(ROUNDS):
            for tag, binary in (("before", before), ("after", after)):
                for key, val in run(binary, p).items():
                    samples[key][tag].append(val)

    w = max(len(k) for k in samples)
    print(f"{'metric':<{w}}  {'before':>10}  {'after':>10}  {'speedup':>8}")
    print("-" * (w + 34))
    for key in sorted(samples):
        b = statistics.median(samples[key]["before"])
        a = statistics.median(samples[key]["after"])
        print(f"{key:<{w}}  {b:>10.4f}  {a:>10.4f}  {b / a:>7.2f}x")


if __name__ == "__main__":
    main()
