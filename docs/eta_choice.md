# The choice of η (regularization offset)

Every Stieltjes-transform evaluation in this crate computes

    S(x) = (1/p) Σ_j 1 / (x − λ_j − iη),

the empirical transform evaluated *off* the real axis. The eigenvalues λ_j
live exactly on the axis, so some offset η is not optional — without it the
sum hits singular terms whenever two eigenvalues coincide. But its magnitude
is a free parameter, and this document records why the library ships
**η = 0.1/√p** as its default (`stieltjes::default_eta`), what theory does
and does not say, and what the measurements show. Reproduce everything with:

    cargo run --release --example measure_eta_sweep > docs/pareto/eta_sweep.json
    # data referenced below: docs/pareto/eta_sweep.json (c = 0.25)

## Why the form f/√p

Two constraints pin the *scaling*, both classical:

1. **Resolution floor (lower bound).** Inside the bulk, neighbouring
   eigenvalues are spaced ~O(1/p); the principal-value part of the sum only
   stabilizes once η exceeds the local spacing, otherwise near-coincident
   pairs dominate term-by-term. Any η ≫ 1/p satisfies this — 1/√p does,
   with three orders of margin.
2. **Consistency requirement (upper bound).** Nonlinear-shrinkage theory
   (Ledoit–Wolf's direct estimator) proves consistency for η_n → 0 while
   smoothing stays wide relative to spacing; the canonical proven choice is
   η ∝ n^(−1/2). With c = p/n that is √c/√p ≈ 0.5/√p here — i.e. **both**
   shipped conventions (0.1/√p and the benchmark convention 1/√p) lie inside
   the theoretically sanctioned family. Theory fixes the exponent, not the
   constant.

So the form is supported; the constant is an empirical question. That is
what the sweep measures.

## What the measurements say

### Precision, end-to-end (bulk-only RIE, mean |ξ−1| over modes, 12 seeds)

| f (η = f/√p) | p = 2000, ChebCodeFast |
|---|---|
| 0.003 | 0.569 |
| 0.01 | 0.560 |
| 0.03 | 0.474 |
| 0.1 | **0.386** |
| 0.3 | 0.356 |
| 1 | 0.342 |
| 3 | 0.328 |

Pointwise bulk accuracy improves slowly as η grows and saturates past
f ≈ 0.3. Two caveats keep this honest: (a) these are *pointwise* errors —
Ledoit–Wolf's own direct-method tables show the same tens-of-percent
pointwise spread at these sizes even though aggregate quantities (trace,
mean) converge far better; (b) the sweep feeds iid MP-distributed marginals
rather than true Wishart spectra (see caveats), which inflates pointwise
noise via near-degenerate pairs (min gap 10⁻¹¹ at p = 20 000).

### Spiked pipeline (deconvolve_spiked: detection + BBP debiasing + bulk)

Spike relative errors and detection are **flat across the entire sweep**
f ∈ [0.003, 3] at both p = 500 and 2000: k detected in 8/8 seeds everywhere,
spike errors ≈ 1.0 % / 2.0 % / 5.1 % (top/middle/weakest spike) regardless of
η. Detection runs on edge statistics (BEMA fit, Tracy–Widom margins) and the
BBP inverse on the fitted edge — neither consumes η. The offset only touches
the bulk deconvolution.

### Runtime (p = 10 000, sequential, median of 9)

| method | f = 0.01 | f = 0.1 | f = 1 |
|---|---|---|---|
| ChebCodeFast | 3.41 ms | 3.34 ms | 2.73 ms |
| BlockedTiled | 37.6 ms | 37.6 ms | 37.6 ms |

Runtime is essentially η-independent: the exact kernel does identical work;
the treecode's adaptivity barely notices (slightly *faster* at large η since
smoother kernels need fewer nodes).

### Empirical-vs-population gap (part A of the sweep)

Comparing the sample transform against direct quadrature of the exact MP
density above the bulk edge shows an η-*independent* offset (~1.6 vs ~1.05
at x = λ₊+0.06, p = 20 000). This is genuine sampling physics, not kernel
error: the largest sample eigenvalues fluctuate above the population edge
(Tracy–Widom law, ~p^(−2/3)), and that extra mass dominates any η effect.
Practical consequence: shrinking η cannot buy accuracy the sampling noise
has already spent, which is why the knee in the bulk table — not the small-η
limit — is the meaningful target.

## Verdict

* **Form** f/√p: theoretically sanctioned (exponent from Ledoit–Wolf-style
  consistency; constants free).
* **Constant**: 0.1/√p sits at the knee where bulk accuracy approaches its
  saturation plateau (0.386 vs 0.328 at f = 3) while keeping the boundary
  layer narrow. Going to 1/√p buys ≤ 4 pp of pointwise bulk accuracy — and
  nothing else, per the spiked-pipeline and runtime tables. Going below
  f ≈ 0.03 costs real accuracy and buys nothing measurable.
* **Recommendation**: keep `default_eta() = 0.1/√p` for the library. The
  recorded benchmark harnesses use 1/√p (declared in each file's meta); all
  published speed numbers are η-insensitive per part C, so cross-convention
  comparisons remain valid.

## Caveats

Single machine (Apple M1 Max), c = 0.25 only, iid-marginal spectra instead
of full Wishart draws (pointwise metrics pessimistic), 8–12 seeds per cell,
single session. Re-run via the command above before quoting new numbers.
