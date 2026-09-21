# The choice of η (regularization offset)

Every Stieltjes-transform evaluation in this crate computes

    S(x) = (1/p) Σ_j 1 / (x − λ_j − iη),

the empirical transform evaluated *off* the real axis. The eigenvalues λ_j
live exactly on the axis, so some offset η is not optional — without it the
sum hits singular terms whenever two eigenvalues coincide. But its magnitude
is a free parameter, and this document records what the library ships and why:

* **η = 0.1/√p** — the generic default (`stieltjes::default_eta`), used by the
  density drivers (`spectral_deconvolution`, `deconvolve_spiked`,
  `deconvolve_adaptive`) and by the precision-matrix shrinkage;
* **η = 0.4/√p** — the default for the pointwise bulk *eigenvalue*
  deconvolution (`stieltjes::default_eta_bulk`, selected by
  `EtaDefault::Bulk`), used by `rie_shrinkage` / `ledoit_wolf_shrinkage` and
  therefore by `estimate_population_eigenvalues`.

The two constants differ because the optimum depends on the **generator** the
spectrum came from, not only on `(p, c)` — see
[The optimum depends on the generator](#the-optimum-depends-on-the-generator).
Reproduce everything with:

    cargo run --release --example measure_eta_sweep > docs/pareto/eta_sweep.json
    # iid-marginal sweep referenced below: docs/pareto/eta_sweep.json (c = 0.25)

    # the true-Wishart calibration and the generator comparison:
    ./.pixi/envs/default/bin/python coherence_shrinkage/scripts/eta_reconcile.py
    ./.pixi/envs/default/bin/python coherence_shrinkage/scripts/eta_refine.py

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
   η ∝ n^(−1/2). With c = p/n that is √c/√p ≈ 0.5/√p here — i.e. **all three**
   shipped conventions (0.1/√p, 0.4/√p and the benchmark convention 1/√p) lie
   inside the theoretically sanctioned family. Theory fixes the exponent, not
   the constant.

A third argument pins the exponent from the estimator itself, and it is the
one that explains why a constant has to be calibrated at all. For the RIE /
Ledoit–Wolf update `ξ = λ/|1 − c + c λ m(z)|²` at `z = λ + iη`:

* the diagonal term of the empirical average contributes `−i/(pη)` to `m` — a
  deterministic displacement of order `1/(pη)`, which **grows as η shrinks**;
* evaluating off the real axis costs a bias **linear in η** (measured
  ≈ `1.09 η` against the exact MP transform).

The sampling variance is `O(1/p)` and carries no η. Balancing the two biases,
`c₁/(pη) = c₄η`, gives

    η* = √(c₁/c₄) / √p,

so the exponent `1/2` is forced by the structure and only the constant is
empirical.

## What the measurements say

### The optimum depends on the generator

The original sweep below feeds **iid draws from the MP density**. Real callers
hand the library a **Wishart spectrum**, and the two do not agree. Same
operating point (`p = 2000`, `c = 0.25`), same metric, three generators:

| spectrum generator | f\* (MSE-optimal) | E[(ξ−1)²] at f=0.1 → at f\* |
|---|---|---|
| true Wishart, p=2000, c=0.25 | **0.4** | 2.82e-3 → 3.79e-4 (**7.4×**) |
| iid MP marginals, p=2000, c=0.25 | **1.0** | 4.75e-2 → 6.18e-3 (7.7×) |
| deterministic MP quantiles, p=2000 | **0.1** | 2.4e-3 → 2.4e-3 (1.0×) |

`mean |ξ−1|` on true Wishart at `p = 2000, c = 0.25` (6 seeds):

| f | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 0.4 | 0.5 | 1 | 3 |
|---|---|---|---|---|---|---|---|---|---|
| mean \|ξ−1\| | 0.748 | 0.268 | 0.119 | 0.041 | 0.0163 | **0.0146** | 0.0148 | 0.0244 | 0.0683 |
| MSE | 5.7e-1 | 9.7e-2 | 2.2e-2 | 2.8e-3 | 4.6e-4 | **3.8e-4** | 4.0e-4 | 9.1e-4 | 5.7e-3 |

The true-Wishart curve is **not** saturating: it has a sharp minimum at 0.4 and
rises steeply past 0.5 (MSE ×6.5 by f = 3). The earlier "0.1 sits at the knee,
1/√p buys ≤ 4 pp" reading does not transfer, because it was measured on a
generator whose near-degenerate clumping suppresses the small-η penalty — the
caveat at the bottom of this file says as much.

Across 15 `(p, c)` cells with true Wishart draws — `p ∈ {120, 400, 1000, 2000}`,
`c ∈ {0.1, 0.25, 0.4, 0.6, 0.8}` — the optimum is `f* ∈ [0.30, 0.50]`, mean
0.39, with gains of **2.7–8.4×** over `f = 0.1`. The curve is flat over
`[0.4, 0.5]`, so 0.4 is chosen as a robust constant rather than a
`c`-dependent fit.

Why the optimum moves at all, in one line: the H0 error is dominated by how
close the sample eigenvalues come to one another, and iid marginals clump far
more than a β-ensemble spectrum does.

### Spiked pipeline (deconvolve_spiked: detection + BBP debiasing + bulk)

Spike relative errors and detection are **flat across the entire sweep**
f ∈ [0.003, 3] at both p = 500 and 2000: k detected in 8/8 seeds everywhere,
spike errors ≈ 1.0 % / 2.0 % / 5.1 % (top/middle/weakest spike) regardless of
η. Detection runs on edge statistics (BEMA fit, Tracy–Widom margins) and the
BBP inverse on the fitted edge — neither consumes η. The offset only touches
the bulk deconvolution.

Re-measured with the tuned bandwidth on the documented benchmark
(`p = 1000, c = 0.25`, population `[12, 7, 4, 1, …]`), the bulk improves and
the spikes are untouched:

| f | bulk mean dev | bulk med \|dev\| | bulk MSE | spikes |
|---|---|---|---|---|
| 0.1 (previous) | −0.0060 | 0.0452 | 5.36e-3 | 12.12 / 7.03 / 4.00 |
| **0.4 (new)** | +0.0085 | **0.0156** | **6.52e-4** | 11.92 / 7.09 / 4.02 |

### Precision, end-to-end (iid-marginal generator, bulk-only RIE, mean |ξ−1| over modes, 12 seeds)

Kept as measured; note the generator caveat above before quoting it.

| f (η = f/√p) | p = 2000, ChebCodeFast |
|---|---|
| 0.003 | 0.569 |
| 0.01 | 0.560 |
| 0.03 | 0.474 |
| 0.1 | **0.386** |
| 0.3 | 0.356 |
| 1 | 0.342 |
| 3 | 0.328 |

Two caveats keep this honest: (a) these are *pointwise* errors —
Ledoit–Wolf's own direct-method tables show the same tens-of-percent
pointwise spread at these sizes even though aggregate quantities (trace,
mean) converge far better; (b) the sweep feeds iid MP-distributed marginals
rather than true Wishart spectra (see caveats), which inflates pointwise
noise via near-degenerate pairs (min gap 10⁻¹¹ at p = 20 000).

### Why the precision path keeps 0.1

`direct_precision_shrinkage` reads the same `RmtConfig::eta` field but applies
a different update (`precision_factor`, real part of the transform only) for a
different loss. Running *it* at 0.4/√p degrades sharply: its
identity-population test moves from ~1.0 to **1.26**. That is why the two
paths carry separate constants rather than one crate-wide value — the
calibration above is specifically for the bulk covariance-eigenvalue
deconvolution.

### Runtime (p = 10 000, sequential, median of 9)

| method | f = 0.01 | f = 0.1 | f = 1 |
|---|---|---|---|
| ChebCodeFast | 3.41 ms | 3.34 ms | 2.73 ms |
| BlockedTiled | 37.6 ms | 37.6 ms | 37.6 ms |

Runtime is essentially η-independent: the exact kernel does identical work;
the ChebCode treecode's adaptivity barely notices (slightly *faster* at large η
since smoother kernels need fewer nodes).

### Empirical-vs-population gap (part A of the sweep)

Comparing the sample transform against direct quadrature of the exact MP
density above the bulk edge shows an η-*independent* offset (~1.6 vs ~1.05
at x = λ₊+0.06, p = 20 000). This is genuine sampling physics, not kernel
error: the largest sample eigenvalues fluctuate above the population edge
(Tracy–Widom law, ~p^(−2/3)), and that extra mass dominates any η effect.
Practical consequence: shrinking η cannot buy accuracy the sampling noise
has already spent — which is another way of seeing why the *true-Wishart*
optimum sits where the `1/(pη)` and `η` biases balance, not at small η.

## Verdict

* **Form** f/√p: theoretically sanctioned (exponent from Ledoit–Wolf-style
  consistency, and independently from the `1/(pη)` vs `η` bias balance);
  constants free.
* **Generic default** `0.1/√p`: unchanged for the density drivers and the
  precision shrinkage. They were not calibrated against the criterion above,
  and the precision path measurably degrades at 0.4.
* **Bulk eigenvalue deconvolution** `0.4/√p`: calibrated on true Wishart
  draws, `f* ∈ [0.30, 0.50]` over 15 `(p, c)` cells, 2.7–8.4× in `E[(ξ−1)²]`
  and ~3× in median `|ξ−1|` against 0.1, with no effect on spikes. The
  conclusion is generator-specific *by construction*: on iid marginals the
  optimum is ≈1, on deterministic quantiles ≈0.1. Wishart is the generator
  the library actually receives.
* **Recommendation**: keep the two-constant split. Recorded benchmark
  harnesses use 1/√p (declared in each file's meta); all published speed
  numbers are η-insensitive per the runtime table, so cross-convention
  comparisons remain valid.

## Caveats

Single machine (Apple M1 Max), `c ∈ [0.1, 0.8]`, 6–20 seeds per cell, single
session, true-Wishart cells sized `p ≤ 2000` (large-`p` cells use the SVD of
an `n × p` matrix). Both the iid-marginal and the true-Wishart sweeps are
reproduced by the scripts above; re-run them before quoting new numbers.
