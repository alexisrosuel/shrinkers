# Changelog

All notable changes to **shrinkers** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed — rank alignment of the cleaning-quality README figure

`scripts/make_readme_figures.py` sorted the cleaned estimates by value before
plotting them, so the red curve was the *order statistic* of the cleaned
spectrum while the grey sample and black truth curves were in their own rank
order. `estimate_population_eigenvalues` is a pointwise map, so the two
orderings put features at different ranks: the red error curve's minimum sat at
rank ~666 against the sample's rank ~449, which reads as a horizontal shift on
the figure. The cleaned series now stays in the sample-eigenvalue order
(`spikes` descending, `bulk_population` reversed to match the descending bulk
sample). The multiset of cleaned values — and therefore the median error, 1.7 %
— is unchanged; only the alignment is. The caption also said "within 1 %" for
spike estimates whose worst error is 1.7 % (4.07 vs 4.00), now "within 2 %".

### Added — complex Hermitian correlation-matrix cleaning

`clean_correlation_matrix` assumed real symmetric input, which is what a
covariance gives. A **spectral coherence matrix** does not: it is built from
Fourier coefficients and is complex Hermitian, with unit diagonal, the same
Marchenko-Pastur bulk and the same BBP spikes.

New surface:

- `pipeline::complex::hermitian_eigh(re, im)` — eigenvalues and eigenvectors of
  a Hermitian matrix through the standard real embedding
  `H = A + iB -> [[A, -B], [B, A]]`. Reuses the existing `symmetric_eigh`; the
  embedding duplicates each eigenvalue, it does not scale it.
- `pipeline::complex::clean_eigensystem_complex` and
  `clean_correlation_matrix_complex` — the same estimator, with the conjugate
  transpose replacing the transpose in `d_bulk I + sum_t scale_t v_t v_t^H`.
  The RIE shrinkage, the `sigma^2` estimate and the RMT angular overlaps are
  scalar and shared with the real path, so there is no second implementation of
  anything that matters.
- Python: `clean_correlation_matrix_complex(correlation, c)`, which validates
  Hermitian symmetry at the boundary and returns the same dict, with
  `covariance` and `eigenvectors` complex.

Verified: a real matrix passed through the complex path reproduces the real
path's cleaned eigenvalues **bit for bit** (`max |diff| = 0.0`), the cleaned
matrix is Hermitian to machine precision and positive definite, and non-Hermitian
input is rejected with a `ValueError`. 144 Rust tests (4 new), 62 Python.

This is the missing brick for frequency-domain estimation on time series; the
downstream estimator that motivated it lives in its own repository and now
delegates its eigensolver here rather than carrying a duplicate.

### Added — complex Hermitian spiked decomposition

The cleaning entry point above answers the *cleaning* question (RIE on every
eigenvalue, spike directions reweighted by their angular overlap). The
*estimation* question — how many coherent modes are there, what are their
debiased eigenvalues, what does the bulk look like — needs the **spiked split**
instead, and that only existed in the eigenvalue domain
(`deconvolution::estimate_population_eigenvalues`). A caller holding the matrix
had to eigendecompose it, throw the eigenvectors away, and hand the eigenvalues
over.

New surface:

- `pipeline::complex::hermitian_eigh_matrix(h)` — the same eigendecomposition as
  `hermitian_eigh(re, im)`, from a single `Complex64` array. Symmetrises to the
  Hermitian part first, so round-off asymmetry does not matter. Both matrix-level
  entry points now start here; `clean_correlation_matrix_complex` was refactored
  onto it with no numerical change.
- `pipeline::complex::deconvolve_correlation_matrix_complex(correlation, c,
  margin, config)` — one pass from the matrix to the whole answer: sample
  eigenvalues, sample eigenvectors, and the `PopulationEigenvalues` split (BEMA
  detection, inverse-BBP spike debiasing, Ledoit–Wolf bulk deconvolution). The
  eigenvectors are a by-product of the eigendecomposition, so a caller that also
  needs the coherent directions does not pay for a second one — and the split is
  guaranteed to come from the *same* eigenvalues that are returned.

Verified: the matrix entry point reproduces
`estimate_population_eigenvalues` on its own returned eigenvalues exactly
(spikes, bulk, `sigma^2` and `bulk_edge` to 1e-12), and a non-Hermitian input's
antisymmetric part is dropped rather than folded in. 147 Rust tests (3 new).

### Changed — calibrated regularization η for the bulk eigenvalue deconvolution

The pointwise bulk deconvolution (`rie_shrinkage`, `ledoit_wolf_shrinkage`,
hence `estimate_population_eigenvalues`) now falls back on **η = 0.4/√p**
instead of the crate-wide 0.1/√p, through a new
`stieltjes::default_eta_bulk` selected by `EtaDefault::Bulk`. The density
drivers (`spectral_deconvolution`, `deconvolve_spiked`, `deconvolve_adaptive`)
and the precision shrinkage keep 0.1/√p; `direct_precision_shrinkage` is
measurably degraded by the wider bandwidth (its identity-population output
moves from ~1.0 to 1.26), so the two paths stay on separate constants.

The exponent is not new — the RIE/Ledoit–Wolf update is *exact* under H0 when
the Stieltjes transform is the exact MP one (verified to 1e-6 against machine
precision), and the whole error comes from using the empirical transform. Its
diagonal term contributes `−i/(pη)` while the off-axis evaluation costs a bias
linear in `η`; balancing them forces `η* ∝ 1/√p`. The constant was then
calibrated on **true Wishart spectra**, which is what callers actually pass in:
`f* ∈ [0.30, 0.50]` (mean 0.39) across 15 `(p, c)` cells, `p ∈ [120, 2000]`,
`c ∈ [0.1, 0.8]`.

Measured effect:

- H0 bulk bias on the spectral-coherence matrices (M=120, c=0.40) moves from
  **−0.092 to +0.034**, median `|ξ−1|` from 0.140 to 0.060, and the monotone
  tilt across the bulk deciles (from −0.02 up to −0.22) is gone;
- the README's own cleaning benchmark (p=1000, c=0.25, spikes 12/7/4) goes
  from **4.3 % to 1.7 %** median relative error — the figure and
  `docs/img/readme_figures.json` are regenerated;
- spike detection and debiasing are untouched (neither consumes η).

This was found while checking whether the LRV spectral coherence matrix
(arXiv:2501.04371) is a `shrinkers`-shaped estimation problem; the
reconciliation with the earlier iid-marginal η sweep, including the
generator-dependence of the optimum, is in `docs/eta_choice.md`.

### Added — Ledoit–Wolf inverse nonlinear shrinkage (precision matrix)

Two new Python entry points estimate the precision matrix Ω = Σ⁻¹ directly,
instead of inverting a covariance-optimal estimator (which over-inflates the
small inverse eigenvalues):

- `inverse_nonlinear_shrinkage(eigenvalues, c, *, method="qis", parallel=False)`
  returns the optimal precision eigenvalues, the matching covariance
  eigenvalues and the Ledoit–Wolf bandwidth. `method` selects the loss:
  `"qis"` (Frobenius / inverse Stein / minimum variance, default), `"lis"`
  (Stein's loss) and `"gis"` (symmetrized Kullback–Leibler).
- `estimate_precision_matrix(covariance, c, *, method="qis", parallel=False)`
  is the matrix counterpart of `clean_correlation_matrix`: it returns the full
  p×p precision matrix Ω̂ = U diag(ω) U′, the sample eigenvectors and the paired
  eigenvalues.

Rust-side: `deconvolution::InverseShrinkageMethod` /
`deconvolution::inverse_nonlinear_shrinkage` and
`pipeline::{estimate_precision_matrix, precision_from_eigensystem,
PrecisionMatrixResult}`.

**Reuse** — θ and Hθ are exactly the real and imaginary parts of the empirical
Stieltjes transform on the **scaled ray** z_i = λ_i(1 + ih), so the estimator
drives the crate's existing ultra-optimized per-point kernels through the new
`stieltjes::compute_stieltjes_scaled_ray` (η_i = h·λ_i per query point, which
the batched FFT/treecode paths — fixed η — cannot serve; they fall back to the
exact auto-vectorized sum). Total cost stays O(p²), the reference
implementation's own complexity.

**Verified** — the Python eigenvalue path matches the Ledoit & Wolf reference
package (github.com/pald22/covShrinkage, `QIS.py`) to ~5e-16 relative, and a
golden-master Rust test locks the formula.

**Note** — the pre-existing `direct_precision_shrinkage` uses a different,
simpler pointwise formula; these functions implement the published
QIS/LIS/GIS estimators.

## [0.1.2] — 2026-09-20

### ChebCodeFast: relaxed operating point + 4-lane f32 far field

`chebcode_fast` was re-optimized end to end, with accuracy explicitly traded
for speed (the preset's contract moves from a ~1e-8 to a ~1e-5 relative-L2
class; `chebcode`, `chebcode_balanced` and `chebcode_xtreme` keep the
accuracy-grade f64 paths unchanged).

Interleaved A/B, same harness (`examples/measure_chebfast.rs compare`),
p = 10 000 / 50 000, MP spectrum c = 0.25, η = 1/√p, Apple M1 Max, quiet
machine. `all.*` is the public dispatcher (build + evaluation + 1/p scaling):

| metric | p | before | after | speedup |
|---|---|---|---|---|
| `eval.seq` | 10 000 | 1.879 ms | 0.753 ms | **2.49x** |
| `eval.seq` | 50 000 | 10.391 ms | 4.288 ms | **2.42x** |
| `eval.par` | 10 000 | 0.430 ms | 0.191 ms | **2.25x** |
| `eval.par` | 50 000 | 1.914 ms | 0.694 ms | **2.76x** |
| `all.seq` | 10 000 | 2.100 ms | 0.953 ms | **2.20x** |
| `all.seq` | 50 000 | 11.131 ms | 5.148 ms | **2.16x** |
| `all.par` | 50 000 | 2.549 ms | 1.752 ms | **1.45x** |
| `grid.seq` (nq = 200) | 50 000 | 0.0488 ms | 0.0197 ms | **2.48x** |
| `build` | 50 000 | 0.859 ms | 0.803 ms | 1.07x |

`all.par` is capped by the still-serial tree build (~0.8 ms = ~45 % of the
parallel wall clock at p = 50 000); the evaluation itself scales 6.2x over the
sequential one.

**Changed**
- **Preset retune**: `ChebPreset::FAST` goes from `(theta 0.5, n 9, leaf 32,
  f64)` to `(theta 1.0, n 8, leaf 32, f32)`. `theta = 1.0` accepts far more
  panels as far field, `n = 8` is a multiple of the 4-lane f32 width, and
  `leaf_cap = 32` re-swept against the new point (`examples/sweep_leaf.rs`)
  in both η conventions.
- **Four-lane f32 far field** (`F32x4` in `src/stieltjes/simd.rs`, still the
  crate's only `unsafe`). ONLY the well-separated far-field dot product is
  f32: traversal, the distance test, the leaf exact sums and the returned
  accumulators stay f64, and each panel's four lanes are reduced into an f64
  scalar. ~1.3x on the far field with a ~6e-7 error floor; the accuracy
  presets keep the 53-bit f64 reciprocal, which is what lets
  `chebcode_xtreme` still reach ~1e-12.
- **Fixed-size traversal stack.** `TraversalStack` (`[i32; 50]`, inline
  push/pop) replaces every `Vec<i32>`, and the builder now *guarantees* the
  depth (`MAX_TREE_DEPTH = 48`; a deeper node becomes an oversized exact
  leaf) instead of assuming it.
- **In-place parallel output.** `eval_points_parallel` preallocates the
  result and writes it through `par_chunks_mut`, dropping the per-chunk
  `Vec` + flatten (one allocation and one p-element copy per call).
- **A/B-instrumented entry points.** `ChebPreset` now carries its far-field
  `FastMode`, `compute_all_stieltjes_chebcode_impl_f32` and
  `ChebCodeBatch::{evaluate_mode, evaluate_points_mode}` expose both
  arithmetics on one shared tree, and `examples/measure_chebfast.rs` gains an
  interleaved `ab` mode (plus `sweep_leaf.rs`, `validate_fast_preset.rs`).

**Accuracy after the change** (library-default η = 0.1/√p, the worst case):
the raw Stieltjes transform is 1.1e-6 … 1.5e-5 relative-L2 for p = 2 000 …
50 000, and the *product* is essentially untouched — `deconvolve_spiked`
bulk density 1.4e-6 … 3.8e-6 relative-L2 with identical spike recovery
against the exact `blocked` pipeline (`examples/validate_fast_preset.rs`).
The deconvolution grid is dispatched to `ChebCodeFast` only when
`nq·4 < p`, so the f32 far field never sees a near-singular query set.

**Measured negatives (kept on purpose, all reverted — full detail with the
per-experiment numbers lives in
[docs/hardware_optimizations.md](docs/hardware_optimizations.md#chebcodefast-round-what-did-not-work-measured-negatives))**
- **k-ary tree** (k = 4, 8): 1.1–2.0x SLOWER end to end, at equal accuracy.
  Accepted panels per level are O(k), not O(1), which cancels the `log_k`
  shallower tree; and the shallow k = 8 tree dumps the near field into exact
  leaves (17 → 458 sources per query at p = 10 000). The binary tree stays.
- **Packed AoS `Node`** (one cache line per visit): +2.6 % sequential CPU —
  the metadata arrays are walked nearly sequentially, so the SoA layout
  already prefetches well, while AoS adds address arithmetic.
- **Even-`n` zero-weight padding** to kill the scalar tail: +3.3 % sequential
  CPU (the padded lane still pays a full reciprocal). Moot now: `n = 8`.
- **Two-accumulator far-field unroll**: p = 50 000 sequential 9.22 → 9.87 ms.
  The accumulation chain is not the critical path.
- **Two-lane refined-reciprocal `barycentric_row` in the build**: ~20 %
  SLOWER than the scalar `fdiv` it replaces (reciprocal latency chain plus
  extra loads/stores, in a short `n`-loop). The build stays scalar.
- **Parallel tree build** (parallel leaf projections + per-depth parallel
  merges, gated on `parallel = true`): 0.63–0.93x end to end. The per-level
  Rayon barrier, buffer churn and scatter cost more than the ~0.8 ms of build
  work at p = 50 000. Left serial.
- **`F64x2::recip_fast`** (17-bit, 1 Newton step) did land +1.19–1.24x on the
  f64 far field with a ~1.4e-6 floor, but the f32 far field is both faster
  (~1.3x) and more accurate (~6e-7), so it is kept only as a measured
  reference (`#[allow(dead_code)]`).

---

Performance round on the exact kernel, the deconvolution grid path and the
treecode build. All numbers are interleaved before/after medians on the same
Apple M1 Max, MP-like spectra, `mp_spectrum(p, c=0.25)`, η = 0.1/√p,
reproduced by `scripts/bench_ab.py` driving two binaries built from the same
harness (`examples/measure_runtime_audit.rs compare <p>`) — one from HEAD,
one from this revision. Machine load was shared by construction (the driver
alternates builds), so the ratios are the meaningful quantity.

| metric | p | before | after | speedup |
|---|---|---|---|---|
| `auto` all-points, seq | 10 000 | 29.47 ms | 2.75 ms | **10.7×** |
| `auto` all-points, seq | 20 000 | 117.8 ms | 5.68 ms | **20.7×** |
| `auto` all-points, seq | 50 000 | 737.4 ms | 12.58 ms | **58.6×** |
| exact all-points, seq (`blocked`) | 10 000 | 29.53 ms | 25.65 ms | **1.15×** |
| exact all-points, seq | 20 000 | 117.8 ms | 101.9 ms | **1.16×** |
| exact all-points, seq | 50 000 | 737.2 ms | 639.1 ms | **1.15×** |
| exact all-points, Rayon | 10 000 | 5.25 ms | 5.26 ms | 1.00× |
| exact all-points, Rayon | 50 000 | 122.8 ms | 121.7 ms | 1.01× |
| grid `chebcode_fast`, nq = 200 | 10 000 | 0.256 ms | 0.186 ms | **1.37×** |
| grid `chebcode_fast`, nq = 200 | 20 000 | 0.462 ms | 0.313 ms | **1.48×** |
| grid `chebcode_fast`, nq = 200 | 50 000 | 1.034 ms | 0.638 ms | **1.62×** |
| grid `chebcode_balanced`, nq = 200 | 10 000 | 0.323 ms | 0.214 ms | **1.51×** |
| grid `chebcode_balanced`, nq = 200 | 50 000 | 1.344 ms | 0.748 ms | **1.80×** |
| `deconvolve_spiked` default (`auto`) | 10 000 | 0.345 ms | 0.244 ms | **1.42×** |
| `deconvolve_spiked` default | 20 000 | 0.524 ms | 0.368 ms | **1.43×** |
| `deconvolve_spiked` default | 50 000 | 11.86 ms | 0.779 ms | **15.2×** |
| grid exact (`blocked`, control) | 50 000 | 4.157 ms | 4.182 ms | 0.99× |
| exact all-points Rayon (control) | 20 000 | 20.34 ms | 19.18 ms | 1.06× |

The `auto` rows are the dispatcher fix below — the single biggest win, and it
lands on the `stieltjes_transform` default-adjacent path. The `blocked` rows
are the fused-accumulation win. The Rayon rows are flat on purpose: the
parallel exact kernel was already fully fused and still runs its own
output-partitioned full-square body. The `grid.blocked` row is the control —
that kernel was not touched.

### Fixed
- **`Auto` was not resolved by the Stieltjes dispatchers.**
  `StieltjesMethod::Auto` is documented as the speed policy that resolves
  through the measured Pareto table (`config.rs`: "`Auto` is the **speed**
  policy"; `docs/internals.md`: "Auto-selects fastest by p"), and
  `RmtConfig::resolve_auto` did resolve it — but `compute_all_stieltjes` and
  `compute_stieltjes_at_points` only resolved the two explicit `*Auto`
  presets and let plain `Auto` fall into a defensive arm that runs the exact
  O(p²) `Blocked` kernel. The Python `stieltjes_transform(method="auto")`
  binding calls `compute_all_stieltjes` directly, so `method="auto"` was
  silently exact: **102 ms at p = 20 000 instead of 4.6 ms**, against
  `method="speed_auto"` which resolved correctly.
  Resolution now lives in one place, `config::resolve_auto_method`, used by
  `RmtConfig::resolve_auto` *and* both dispatchers; the dead `Auto` arm is
  gone. Interleaved before/after: **10.7×** (p = 10 000), **20.7×**
  (p = 20 000), **58.6×** (p = 50 000); Python-level p = 20 000:
  102.4 → 4.6 ms.
  *Behaviour note*: `method="auto"` now returns the treecode preset's
  ~1e-8-class result rather than a machine-precision exact one. That is what
  the `Auto` contract says, and the exact path remains available and is
  still the default for `stieltjes_transform` (`method="blocked"`) and
  `accuracy_auto`. Regression-tested by
  `test_auto_resolves_and_matches_the_resolved_method`.
- **The deconvolution grid inherited an all-points dispatch decision.**
  `RmtConfig::resolve_auto` consults the measured Pareto table, which is an
  *all-points* table (`nq = p`); its large-p speed pick is `Fft5`, whose cost
  is a whole-grid convolution **independent of the query count**. On the
  `n_points = 200` deconvolution grid used by `deconvolve_spiked` /
  `spectral_deconvolution` (the Python default, `method="auto"`, sequential)
  that spent 11.9 ms at p = 50 000 where `ChebCodeFast` needs 0.64 ms — and
  the FFT bank is ~4e-5 accurate against the treecode's ~1e-8.
  New `RmtConfig::resolve_auto_at_points(p, nq)` redirects the auto presets
  to the ChebCode speed preset when `nq·4 < p` (the measured FFT/treecode
  grid crossover is near `nq ≈ 0.7·p`); above that the table's pick stands.
  Explicit `stieltjes_method` values are never second-guessed.
  `deconvolve_spiked` default path, sequential: p = 20 000 **1.43×**,
  p = 50 000 **15.2×** (11.86 → 0.78 ms), at ~3 orders of magnitude *better*
  accuracy.
  The Pareto table itself is deliberately left untouched: the exact family is
  never the speed pick in any bin (it wins the accuracy column on error, not
  runtime), so making it 15 % faster cannot change a bin.
- **The Python boundary validated less than its error messages promised.** Five
  gaps, each reproduced against a freshly built extension before the fix:
  `deconvolve_spiked` forwarded `eta` raw, so `eta <= 0` or `NaN` was accepted
  there while `stieltjes_transform*` rejected it (the explicit `"inferred"`/None
  sentinel still passes through untouched, because the downstream default
  `0.1/√p_bulk` is deliberately not `0.1/√p_full`); `margin` was checked only in
  `deconvolve_spiked` while `detect_spikes_bema`, `analyze_spikes` and
  `estimate_population_eigenvalues` accepted non-positive or NaN values that
  BEMA's `margin.max(1.0)` then swallowed silently; `FloatOrVec` validated the
  array branch and returned early on the scalar one, so `inverse_bbp(nan)` and
  `inverse_bbp(-1.0)` were accepted while the equivalent array raised; and
  `clean_correlation_matrix` never checked symmetry even though `symmetric_eigh`
  reads and updates both triangles, so a non-symmetric input produced a silently
  wrong eigensystem (now rejected outside a 1e-12 relative tolerance). The
  checks are single-definition helpers (`require_positive_finite`,
  `checked_lambda_hat`, `owned_positive_spectrum`, `sorted_ascending`) shared by
  every entry point. Tests: 39 → 50, `TestBoundaryValidation` pins each gap.
- **`parse_method` matched `"chebcode_balanced" | "chebb"` twice**, making the
  second arm unreachable. Rust CI never saw it: `python.rs` is behind
  `#[cfg(feature = "python")]` and the rust job's clippy ran without that
  feature — the lint job now runs `cargo clippy --features python`.
- **The shipped type stub described a signature the extension no longer had.**
  `shrinkers.pyi` still advertised `parallelism: Parallelism` while the runtime
  had long taken `parallel: bool | None`, and it omitted the
  `chebcode_balanced`/`chebb` methods; `eta`/`cutoff` now document the accepted
  `None` too. Stub and runtime agree again.

### Packaging & CI
- **The wheel is now declared and exercised on Python 3.9–3.15.** The artifact
  was already a single `cp39-abi3` wheel, so 3.11/3.12/3.14/3.15 could load it;
  CI now proves that instead of assuming it, 3.15 included while it is still a
  release candidate (`allow-prereleases`). `Programming Language :: Python ::
  3.9` … `3.15` classifiers added.
- **CI installs the wheel the job just built.** `pip install --find-links dist
  shrinkers` resolved to the published 0.1.1 on PyPI rather than the
  equal-version local wheel, so the Python matrix was silently testing the
  released artifact; the job now installs `dist/*.whl` by explicit path.

### Changed
- **ChebCode leaf capacity now depends on the query count.** A preset's
  `leaf_cap` is tuned for `nq = p`; on the grid path `nq` is `n_points`, the
  call is build-dominated (measured 81 % / 94 % of the total at p = 10 000 /
  50 000) and most of that build is `merge_weights`, whose cost scales as
  `p / leaf_cap`. `compute_stieltjes_at_points` now sizes the leaf from the
  cost model `L* = n·√(2p/nq)`, floored at the preset's own `leaf_cap` so the
  all-points behaviour is untouched. Because leaves are summed **exactly**,
  the relaxed tree is also *more* accurate, not less: p = 50 000, nq = 200,
  `chebcode_fast` 4.1e-9 → 1.3e-9 rel-L2; p = 10 000 3.2e-9 → 2.0e-9.

### Performance
- **Exact symmetric sweep: 9 → 6 FP ops per visited pair.** The sequential
  `Blocked`/`BlockedTiled` hot loop wrote each accumulation as
  `w = d·inv; rr += w; cr -= w` and `v = η·inv; ri += v; ci += v` — two FMULs
  and four FADDs/FSUBs per pair. Rewriting each as one explicit `mul_add`
  (`rr = d.mul_add(inv, rr)`, `cr = (-d).mul_add(inv, cr)`,
  `ri = eta.mul_add(inv, ri)`, `ci = eta.mul_add(inv, ci)`) removes the two
  temporaries and fuses the additions, leaving
  `FSUB + FFMA + FDIV + 4 FFMA`. The kernel already ran at ~92 % of the
  machine's FP issue rate, so op count converts directly into runtime:
  **1.15–1.16×** end-to-end at p = 10 000 / 25 000 / 50 000. `mul_add` is a
  true fused primitive in Rust, so the contraction does not depend on the
  `fp-contract` codegen policy, and fusing makes every accumulation strictly
  more accurate (one rounding instead of two). The same treatment was applied
  to the treecode's barycentric row update (`mass·(1/s)` hoisted, one
  `mul_add` per node).
- `figures/stieltjes_*.png` and `figures/stieltjes_data.csv` regenerated with
  `scripts/bench_stieltjes_all.py` against the new build.

### Measured negatives (kept on purpose)
- **Register-resident target tiles in the parallel `tiled_one_block_no_cutoff`
  body: slower.** Accumulating a 4-row target tile in registers removes the
  two output read-modify-writes per pair, but it also forces the source array
  to be re-streamed once per 4-row tile instead of once per 32-row block:
  p = 10 000 Rayon 4.94 → 5.79 ms, p = 50 000 119.5 → 137 ms. Reverted.
- **A parallel *symmetric* exact kernel: slower per pair.** Splitting the
  strict upper triangle into folded row strips (`[k·c,(k+1)·c)` plus
  `[p−(k+1)·c, p−k·c)`, exactly balanced) and reducing per-worker mirror
  planes halves the pairs *and* the divisions, yet the worker-local sweep
  measured ~1.6× slower per pair than the sequential `SoaSink` sweep even at
  one thread (p = 10 000: 25.2 ms sequential vs 39.7 ms for the same schedule
  through the mirror sink), and never beat the output-partitioned full-square
  kernel at any thread count (10 threads: 133 vs 116 ms at p = 50 000).
  Collapsing the mirror planes onto the owner planes and running a single
  strip changed nothing, so the cost is in the sweep's two-stream write
  pattern rather than in the reduction. Reverted; the parallel exact path
  keeps `compute_all_stieltjes_blocked_tiled_parallel`.
- **PGO remains unavailable offline.** rustc emits raw profile format v10;
  the only `llvm-profdata` on the box (Command Line Tools) reads v8, and
  `rustup component add llvm-tools-preview` fails to download. Same blocker
  as the 0.1.1 round.

## [0.1.1] — 2026-08-26

### Added
- **`docs/hardware_optimizations.md`**: catalogue of the machine-level
  optimizations across the fast O(p^2) exact kernels and ChebCode* —
  hoisted-reciprocal scalar term, pair-once symmetric sweep in
  register-resident 4-row tiles, measured bs8 cache-block sweet spot,
  AoS-vs-SoA small-p crossover, F64x2 refined Newton-Raphson reciprocal
  (AArch64 lacks FP64 vector divide), per-term far-field stability,
  build-side division hoisting and parent composition, SoA layout,
  chunked parallel dispatch — including the documented negative results.
- **`docs/chebcode_algorithms.md`**: full algorithm reference for the
  ChebCode* treecode family — tree layout, Chebyshev equivalent
  densities with barycentric weights and parent composition, the
  branchless opening-angle traversal, SIMD near/far-field loops,
  preset table with measured parameter sensitivities, η coupling,
  batch API and usage sites. Cross-linked from README and internals.
- **`StieltjesMethod::ChebCodeBalanced`** (`"chebcode_balanced"` /
  `"chebb"`): ~3e-10 rel-L2 at roughly FAST+6% runtime — a measured
  intermediate frontier point between FAST (~1e-8) and XTREME (~1e-12).

### Changed
- **HODLR: one ACA compression now serves both cross blocks** via the
  exact kernel relation M(R<-L) = -conj(M(L<-R)^T), transferred at the
  factor level (U2 = -conj(V1), V2 = conj(U1)). Full Hodlr path at
  p=20 000 seq: 82.3 -> 49.9 ms (-41%, 1.68x); rel-L2 vs BlockedTiled
  improves to 6.7e-10 from 1.1e-9 (the transferred adaptive rank is
  whichever side compressed tighter). Both Aca and Random modes.
- **Fixed: Python `method="auto"` resolves through the measured Pareto
  table** instead of the pre-table heuristic that sent p=20k sequential
  to Fft2. Default-path `deconvolve_spiked` at p=20 000: 10.6 ms ->
  0.70 ms (15x). `Auto` now tracks every future re-sweep for free.
- Dispatch bins regenerated from a two-pass merged sweep;
  `chebcode_balanced` wins the speed_seq <=1000 and <=10000 bins.
- Campaign diagnostics: ChebCodeFast tree build measured at 10-11% of
  runtime (below action threshold); windowed ratio 10 confirmed at the
  knee; four symmetry-exploitation attempts measured slower and are
  documented with root causes (see CHANGELOG history below and
  docs/internals.md).
- **BREAKING (Rust API): the individual Stieltjes kernels are no longer
  `pub`.** `compute_all_stieltjes` / `compute_stieltjes_at_points` (the
  dispatcher over `StieltjesMethod`) are the supported entry points; the
  per-family kernels below them had no callers outside the crate. Now
  `pub(crate)`: `naive_stieltjes_sum`, `autovec_stieltjes_sum`,
  `stieltjes_with_deriv_sum`, `ValuesAndDerivs`,
  `compute_all_stieltjes_with_deriv`, `compute_all_stieltjes_blocked`,
  `compute_all_stieltjes_blocked_windowed`,
  `compute_all_stieltjes_blocked_autovec`, `stieltjes_sum_blocked_autovec`,
  `compute_all_stieltjes_adaptive`, `compute_all_stieltjes_ewald`,
  `compute_all_stieltjes_treecode_impl`, `compute_all_stieltjes_fft5`,
  `compute_stieltjes_fft_at_points`, `BLOCK_SZ`, `stieltjes_term_hoisted`,
  `compute_all_stieltjes_f32` (Python-only) and
  `compute_all_stieltjes_chebcode_preset`. Deleted as unused:
  `compute_all_stieltjes_fft5_linear` and
  `compute_all_stieltjes_fft5_with_order`, two thin wrappers over
  `compute_all_stieltjes_fft5_with_options`, which is what the grid-order
  study actually calls. Kept `pub` **only** because the `examples/` and
  `benches/` harnesses call them directly, and documented as such:
  `compute_all_stieltjes_blocked_tiled(_parallel)`,
  `compute_all_stieltjes_chebcode(_impl)`, `chebcode_tree_for_bench`,
  `compute_all_stieltjes_hodlr_impl`,
  `compute_all_stieltjes_fft5_with_options`. The empty `pub use <mod>::*`
  re-exports that resulted are gone.
  Five of those items exist only for the Python bindings or for the unit
  tests (`compute_all_stieltjes_f32`, `stieltjes_with_deriv_sum`,
  `ValuesAndDerivs`, `compute_all_stieltjes_with_deriv`,
  `compute_all_stieltjes_blocked_tiled_f32`); they are additionally gated on
  `feature = "python"` / `test`, so a plain Rust build compiles none of them.
  Without that gate `cargo clippy --all-targets` — the CI rust job, which does
  not enable the `python` feature — reports them as dead code.

### Fixed
- **Changelog correction.** The 0.1.0 "Removed" entry claimed a set of
  "config knobs that did nothing" had been deleted: `Strategy`
  (+ `with_strategy`), `Precision` + `RmtConfig::precision` +
  `with_precision`, `FftGridSize` + `fft_grid_size` + `with_fft_grid`,
  `RmtConfig::label`, `StieltjesMethod::{description, all}` and
  `Parallelism::name`. None were removed — that paragraph came in with a
  changelog-only commit and `src/config.rs` is unchanged since the initial
  import. They are retained deliberately:
  - `FftGridSize`/`fft_grid_size` are read by the dispatcher
    (`grid_points()`), and `FftGridSize::Custom` is reachable through the
    public builder even though no preset uses it;
  - `Strategy`/`with_strategy` are the documented preset entry point
    (`docs/internals.md`) and are covered by the config tests;
  - `StieltjesMethod::all` and `Parallelism::all` drive the exhaustive
    method × parallelism test matrix;
  - `RmtConfig::label`, `StieltjesMethod::description` and
    `Parallelism::name` are public API of a published crate, so dropping
    them is a semver decision rather than a cleanup.
  `Precision`/`RmtConfig::precision`/`with_precision` remain genuinely
  inert (the real f32 path is Python's `precision="f32"` side channel);
  they stay as declared-but-unread so that removing them is a deliberate
  breaking change, not a silent one.

## [0.1.0] — 2026-08-25

### Changed
- **Documented the η choice** (`docs/eta_choice.md`) with a reproducible
  sweep (`examples/measure_eta_sweep.rs`, data under `docs/pareto/`):
  f/√p form anchored to Ledoit–Wolf consistency scaling; 0.1/√p confirmed
  as the knee between bulk accuracy and boundary-layer width; spike
  pipeline and runtimes measured η-insensitive.
- **Breaking (Python API): `parallelism="seq"/"rayon"` replaced by a plain
  `parallel` switch** on `deconvolve_spiked`, `stieltjes_transform` and
  `shrink_eigenvalues`: `False` (default) single-threaded, `True`
  multi-core, `None` = library decides by problem size. The threading
  backend is no longer part of the API surface; the Rust enum variant
  `Parallelism::Rayon` was renamed `Parallelism::Parallel` to match.
- **ChebCode query-path anatomy and first cache-locality pass**
  (`examples/profile_hot.rs` sampling harness + wall-clock ablations at
  p=50k DEFAULT sequential). Cost split of the 14.7 ms evaluation:
  far-field panel sums ≈63%, tree traversal + acceptance tests ≈35%,
  exact leaf sums <1% (they are NOT worth optimizing further). Adopted:
  per-panel squared half-width precomputed at build time
  (`FlatChebTree::hw_sq`) so the acceptance test is one load plus one
  multiply, and a branchless distance clamp `(lo−z).max(z−hi).max(0)`.
  Build 1.41→1.16 ms (−18%), query −2–3%; chebcode seq p=50k
  14.59→14.44 ms. Documented negative: processing accepted far-field
  panels in interleaved PAIRS measured ~10% slower — the n-loop
  iterations are already independent, so the out-of-order core overlaps
  their reciprocal chains without help, and pairing only doubles live
  registers and loads.
- **Per-call overhead removed from the small-p exact path**
  (`SYM_AOS_MAX_P = 64` in `stieltjes/cacheblock.rs`). The symmetric-pair
  schedule is now written once, generically over a `SymSink` trait that is
  monomorphized per output layout: below p=64 the dispatcher's sequential
  no-cutoff path runs it into ONE interleaved buffer scaled in place (the
  old route cost three allocations — kernel reals/imags plus a zip/scale
  collect — and two extra output passes; at p≤5 those dominated, ~2.5×
  end-to-end), above p=64 the dense SoA streams are kept (the interleaved
  layout loses ~17% there). The monomorphized SoA specialization also
  measures faster than the previous hand-written kernel at every size:
  seq p=1000 380→294 µs, seq p=50000 0.938 s → 0.736 s. Crossover
  unchanged in shape: `chebcode_fast` ties exact at p≈500 and wins from
  600; `chebcode_xtreme` needs ≈1000.
- **Exact all-points kernel rewritten as a symmetric-pair sweep**
  (`symmetric_all_points` in `stieltjes/cacheblock.rs`). The sequential
  no-cutoff `BlockedTiled` path previously swept the FULL p×p square,
  computing both orientations of every pair. Because the query set is the
  source set, the pair term satisfies: real part antisymmetric
  (`out_r[i] += d·u`, `out_r[j] -= d·u`), imaginary part symmetric
  (`out_i[i] += η·u`, `out_i[j] += η·u`), reciprocal shared. The new kernel
  visits each unordered pair once in a register-resident 4×4 schedule
  (16 independent divisions per tile; column side accumulated in registers
  and flushed with one read-modify-write per column). Output identical up to
  FP summation order (~1e-15 rel). Measured back-to-back on M1 Max:
  +33% at p=300 (46.7→35.0 µs), ~+31% sustained for p=2000..50000
  (seq 50k: 1.244 s → 0.938 s); parity below p≈50 where call overhead
  dominates. Consequence: the O(p²)→treecode crossover moved OUT from ≈350
  to ≈500–600 (`chebcode_fast` ties at ≈500, `chebcode` from ≈600) and
  ≈1000 (`xtreme`).
  Documented dead ends en route: a naive row-wise triangle loop was SLOWER
  than the full-square kernel (scattered per-pair output updates destroy
  the original 4×4 ILP); doubling to two source quads per pass spills
  registers and loses ~15%; factoring η out of the imaginary accumulators
  saves a multiply per pair but measured neutral-to-slower.
- **ChebCode hot loops vectorized with a Newton-refined reciprocal**
  (`stieltjes::simd::F64x2`): AArch64 NEON has no FP64 vector divide, so the
  per-term `1/(d²+η²)` now runs as a 4-step FRECPE/FRECPS refinement on
  pipelined multiply/add units, with lanes spanning pairs of Chebyshev nodes
  (far field) or source points (leaf near field). Measured on Apple M1 at
  identical outputs (≤1 ulp; test error vectors bit-identical): p=50000
  seq 26.3→23.4 ms (−11%), rayon 10.19→9.3 ms (−9%); ~10% across all sizes.
  The `unsafe` policy is updated: all crate `unsafe` is confined to
  `stieltjes/simd.rs` behind the safe `F64x2` abstraction (README section
  "Unsafe code policy").
- Chebyshev weight build (`fill_weights`) hoists its divisions:
  `v_j = λ_j/(x−t_j)` is computed once and the barycentric update becomes
  `w_j += v_j·(1/s)` — one division per point instead of two per
  (point, node); numerically identical up to ≤1 ulp.

- **`fft5` grid transfer upgraded to higher-order Lagrange stencils** —
  2/4/6/8-point (`Order::Linear…Heptic`), default **heptic**; the linear
  path remains as `compute_all_stieltjes_fft5_linear` and every knob
  (order, forced grid size, padding multipliers) is exposed through
  `Fft5Options` / `compute_all_stieltjes_fft5_with_options`. Measured on
  MP-like spectra: the cubic step moved the frontier ~10× in error at
  equal cost (and ~7× less cost at equal error); going to quintic/heptic
  reaches the same order-independent wrap-around floor one grid-halving
  (~40 % runtime) earlier and measured never worse than narrower stencils,
  so heptic is the free accuracy-insurance default. `Adaptive`, `Dst`,
  `Fft3`/`Fft2` inherit the default through their fft5 core.
- **Presets are now data-driven.** A benchmark harness
  (`examples/pareto_data.rs`, JSON dump in `docs/pareto/`) measures every
  method × {seq, rayon} × p against the exact O(p²) reference;
  `scripts/build_pareto_table.py` derives the per-size winners and emits
  `src/config/pareto_autogen.rs`. New `StieltjesMethod::SpeedAuto` resolves
  Speed to the fastest method per size/parallelism with error ≤ 1e-2;
  `AccuracyAuto` now uses the same table (lowest error, ties within 5%
  broken by runtime) instead of a hard-coded threshold. **Both presets no
  longer override the user's parallelism choice** — Sequential and Rayon
  have independent table columns. Regenerate after re-benchmarking with
  `cargo run --release --example measure_pareto_frontier -- after > docs/pareto/bench_after.json`
  then `python3 scripts/build_pareto_table.py docs/pareto/bench_after.json`.
- **Pareto-frontier plots**: `scripts/plot_pareto.py` renders before/after
  frontiers (`docs/pareto/pareto_{seq,rayon}.png`) from two JSON dumps.
- **FFT plan cache**: one `FftPlanner` per thread (`stieltjes::fftplan`)
  shared by `fft5`/`fft3`/`fft2`/`Ewald`; previously every call constructed a
  fresh planner and re-planned identical transform lengths. Steady-state
  `fft5` at p=4000: 0.91 → 0.58 ms (**1.6×**).
- **`Strategy::Accuracy` is now size-aware** via the new
  [`StieltjesMethod::AccuracyAuto`] policy: exact O(p²) tiled kernel below
  p = 4000 (`ACC_EXACT_MAX_P`, machine precision is free there), ChebCode
  (~1e-10 relative at a small fraction of the cost) above. Previously it
  pinned `AutoVectorized`, which is brutally slow at large p.
- **`Strategy::Speed` block_size fixed**: 128 → 16 (the measured optimum;
  the old value predated the tiling analysis).
- **Parallel exact path rewritten**: `Blocked`/`BlockedTiled` + Rayon now run
  the tiled kernel over disjoint output spans (`par_chunks_mut`, no false
  sharing, no reduction) instead of a per-row single-point scan. Measured
  ~2.5× faster at p=20000 (8-core Apple Silicon); `BlockedTiled` + Rayon is
  now genuinely parallel (previously fell back to sequential).
- **Cutoff dispatch fixed**: with a far-field cutoff enabled, the blocked
  family (sequential and parallel) now routes to the windowed kernels, which
  binary-search each contiguous inclusion window instead of branch-skipping
  an O(p²) sweep. Same included term set, O(p·k) iterations: ~20× faster at
  p=20000 (parallel), results identical up to FP summation order.
- Single-point `stieltjes_sum_cutoff` now binary-searches its window too
  (fixes the same branchy-scan loss for `compute_stieltjes_at_points` and
  per-point parallel queries).
- Tree codes (`ChebCode`, `TreeCode`) skip their defensive O(p log p) sort
  when the input is already sorted (O(p) check) — the pipeline always passes
  sorted eigenvalues.
- `StieltjesMethod::Dst` now delegates to the shared fft5 grid (real part) +
  windowed imaginary part — the same computation with fewer transforms
  (1 forward + 1 inverse vs 2 forward + 1 inverse).

### Changed
- **Fixed: Python `method="auto"` now resolves through the measured
  Pareto table** instead of the pre-table size heuristic that sent
  e.g. p=20k sequential to Fft2. End-to-end `deconvolve_spiked` at
  p=20 000 drops from 10.6 ms to 0.70 ms (15x) with identical detected
  k and spike estimates; `StieltjesMethod::Auto` and the hardcoded
  `resolve()` tests now mirror the regenerated pareto_autogen bins.
- Campaign diagnostics committed without code change: ChebCodeFast
  tree build is 10-11% of end-to-end runtime (below the 15% action
  threshold), and the windowed cutoff-ratio curve at p=10k shows
  rel-L2 stuck near 0.35-0.5 across ratio 4..30 while runtime grows
  linearly - ratio 10 sits at the knee and is kept.
### Changed
- Documented a third negative result: a symmetric privatized parallel
  variant of the AutoVectorized sweep at SMALL p (where the private
  buffers DO fit L2, unlike the large-p attempt below) measured
  136-182% slower at p=1k/2k/5k. Root cause: dual-output inner loops
  (row update + conjugated column update per term) defeat the LLVM
  auto-vectorization that makes autovec competitive - losing SIMD costs
  more than halving the pair arithmetic gains. Symmetry exploitation
  beyond the existing sequential kernels is closed out.
- Documented a second negative result: exploiting K(b,a)=-conj(K(a,b))
  in the PARALLEL tiled path (private full-size buffers per span +
  reduction, each unordered pair computed once) measured 3.5-4.9x
  SLOWER than the duplicated-pair baseline at p=10k/20k/50k - the
  n_spans x p private-buffer footprint (~48 MB at p=50k) streams
  column updates from RAM, far costlier than the halved arithmetic
  buys back. Reverted; sequential paths already exploit the identity.
- Recorded as future work: HODLR off-diagonal blocks satisfy
  M(B,A) = -conj(M(A,B)^T), so one ACA compression could serve both
  orientations if the solver ever needs one.
- Documented a negative result: an F64x2 SIMD rewrite of the
  blocked-tiled symmetric sweep measured 36% SLOWER than the shipped
  scalar kernel at p=5k/20k (scalar fdiv latency is already hidden by
  the ILP of four independent register-resident rows), with rel
  deviation up to 2e-9 from bit-exact outputs. Reverted; the scalar
  kernel stands.

### Added
- **`StieltjesMethod::ChebCodeBalanced`** (`"chebcode_balanced"` / Python
  alias `"chebb"`): new ChebCode preset (theta=0.55, n=11, leaf=32) at
  ~3e-10 rel-L2 error and roughly FAST+6% runtime — a measured operating
  point between FAST (~1e-8) and XTREME (~1e-12); it dominates neither
  and takes no auto-dispatch bin. Duel data in
  `examples/measure_xtreme_duel.rs`.
- **README value-proposition front page with two measured figures**
  (`scripts/make_readme_figures.py`, data `docs/img/readme_figures.json`).
  Figure 1 shows what the cleaning buys on a spiked model (p=1000, c=0.25,
  spikes 12/7/4): all spikes detected and debiased within 1%, sigma^2
  estimated at 1.002, median relative error vs the true population spectrum
  40% -> 4.3%. Figure 2 benchmarks the full Stieltjes transform against a
  textbook pure-Python double loop and a chunked vectorized NumPy baseline
  (same arithmetic, NO FFT, NO scipy): at p=4096, 6.7 s / 104 ms / 1.5 ms
  respectively (~70x over NumPy); at p=50000 NumPy needs 14.9 s vs
  0.15 s exact rayon (~98x) and 3.6 ms chebcode_fast (~4000x).
- **Small-p crossover study** (`examples/small_p_crossover.rs`, data
  `docs/pareto/small_p.json`, chart `docs/pareto/crossover_small_p.png`).
  The frontier sweep starts at p=1000; this companion sweep covers
  p ∈ [1, 1000] log-spaced with noise-resistant batched timing (each point =
  median of nine ≥5 ms batches). Findings on MP spectra, η=1/√p, sequential
  (after the symmetric-pair exact-kernel rewrite below): the exact O(p²)
  kernel wins up to p≈500 (2.6× faster than any preset at p=100);
  `chebcode`/`chebcode_fast` take over from p≈600; `chebcode_xtreme` only
  from p≈1000. Below a preset's leaf cap the
  tree is a single exact leaf — each curve visibly steps when p crosses it.
  Under p=2000 the exact+Rayon route is the per-row fallback
  (`PAR_TILED_MIN_P`) with ~20 µs fixed scheduling overhead, so the
  sequential comparison is the honest algorithmic crossover.
- **`StieltjesMethod::Hodlr` — hierarchical low-rank (HODLR) summation**, a
  fundamentally different paradigm from the analytic compressions already in
  the crate: the kernel matrix `K_ij = 1/(λᵢ−λⱼ−iη)` is applied to the
  all-ones vector over a balanced index tree whose off-diagonal blocks are
  compressed by **adaptive cross approximation** to a requested tolerance.
  ACA pivots actual kernel entries and validates itself block by block — no
  opening-angle parameter, no equivalent densities, no analytic translations
  (the failure mode that sank the FMM prototype). Near-field is exact at the
  leaves; cross terms apply as `U·(V·1)`; factors live only during their
  level pass (`O(rank·p)` peak memory); Rayon-parallel over subtrees.
  Measured (MP spectra, η=1/√p, defaults leaf=256/tol=1e-9/rank≤32):
  accuracy 5e-10..7e-10 rel L2 at every size — ~10× more accurate than
  ChebCode's operating point — at 140 ms seq / 64 ms rayon for p=50000
  (vs ChebCode 14.44 ms / 3.24 ms at the same size). Verdict: dominated
  on this spectrum family
  (ChebCode wins speed-at-accuracy, BlockedTiled wins exact), but kept as
  the portfolio's kernel-agnostic member: it needs nothing but kernel
  evaluations, so it transfers unchanged to future kernels without FFT or
  tree structure. Implementation notes: skeleton pivots must skip previously
  used rows/columns (re-use makes the residual column vanish identically and
  division by the machine-zero pivot destroys the factors), and the stopping
  rule must test the normalized next-term norm ‖u‖·‖v‖/|pivot| — the raw
  product ignores the pivot scale and stopped ~1e4× before tolerance.

- **`HodlrMode::Random` — RandNLA sketching path inside the HODLR driver**
  (Halko–Martinsson–Tropp style double sampling): orthonormalize a
  *stratified* sample of kernel columns (boundary strips + geometric offset
  ladder + uniform fill — measured 1e5× better span than uniform sampling on
  adjacent blocks), fit the row space by complex least squares on stratified
  rows, validate against whole boundary-strip test columns and double the
  rank until tolerance. Complex Cholesky solver unit-tested to 5e-15.
- `scripts/plot_runtime_vs_p.py` — runtime-vs-p charts (log-log), replicated
  per parallelism and per accuracy band (`runtime_vs_p_{seq,rayon}.png`,
  `runtime_vs_p_grid.png`).

- **ChebCode re-tuning (overnight round, M1 Max).** New dispatch default
  θ=0.5, n=11, leaf=32 dominates the historical (0.3, 9, 16) on BOTH axes
  at every size (p=50k: 5.2e-10 @ 3.5 ms rayon vs 9.2e-10 @ 9.5 ms).
  Two measured presets join the enum/Python: `chebcode_fast` (θ.5 n9 L32;
  ~1e-8 band speed king, 3.0 ms rayon) and `chebcode_xtreme` (θ.25 n11
  L16; 5.8e-13 @ 5.2 ms rayon — the 1e-12 class previously cost 124 ms
  via blocked_tiled, −96 %).
- **Hierarchical weight composition in the ChebCode build** — parent
  barycentric weights are merged from children's node masses
  (O(n²)/child) instead of rescanning every source in range
  (O(count·n)); build arithmetic drops ~10×, per-call runtime −20 % seq /
  −50 % parallel, error unchanged.
- **Two-lane pairwise multi-η traversal** (`contribution_x2`) — one η per
  F64x2 lane, sources/nodes splatted across lanes so accumulator lanes
  keep fixed meaning; γ-sweep workflow now measures 6.7–7.5× vs naive
  per-η calls at the DEFAULT preset (8.5–11× at FAST).
- **Chunked parallel queries** for ChebCode single calls (256-query
  blocks): 4.9 → 3.95 ms rayon at p=50k.
- **`ChebCodeBatch` — amortized multi-η driver for γ-sweeps.** The Chebyshev
  tree depends on the spectrum and the interpolation geometry only, not on
  η, so one build serves a whole deconvolution sweep;
  `evaluate_many` parallelizes across the η axis with the tree shared
  read-only. Measured (MP spectra): 16 η at p=20000 → 165 ms naive vs
  25 ms batched (**6.5×**), 32 η at p=50000 → 829 ms vs 126 ms (**6.6×**).
- **Analytic derivative `stieltjes_transform_with_deriv`** (Rust
  `compute_all_stieltjes_with_deriv` + Python binding): one exact pass
  returns S and S′ with `S′ᵣₑ = −Σ (d²−η²)/den²`,
  `S′ᵢₘ = −Σ 2dη/den²` — ready for Newton-style root finding on γ.
- `examples/bench_batch.rs` — sweep-workflow benchmark.

### Rejected (research track — documented negative result)
- **NR-refined vector reciprocals in the exact family.** A near-exact
  blocked kernel replacing scalar FP64 divides with two-lane Newton–Raphson
  reciprocals measured SLOWER than scalar tiled code at p=50k (1155 ms vs
  944 ms seq): Firestorm's divide throughput plus the reciprocal's vector
  bookkeeping (splat/sub/lane-pairing overhead) make it a net loss in this
  elementwise pattern. Exact family keeps true division. Also fixed a
  latent `bench_one` double-scaling bug this experiment uncovered.
- **PGO build.** Blocked offline: rustc emits profraw v10 while the local
  CommandLineTools llvm-profdata reads v8, and the matching
  `llvm-tools-preview` component cannot be downloaded without network.
  Retry once a toolchain with a matching llvm-profdata is available.
- **GPU offload (wgpu/Metal).** Not testable in this environment: adding
  the dependency requires network. fp32 GPU precision would confine it to
  the loose-error band where ChebCode already wins by orders of magnitude,
  so expected Pareto value was low regardless.
- **AMX (Apple Matrix eXtensions).** The Stieltjes kernels are
  elementwise-with-reduction, not matmul-shaped; AMX has no fp64 path that
  helps here and requires inline asm (no std intrinsics). Expected value
  judged low against implementation risk; not attempted beyond analysis.
- **Quantile-quadrature far field (mass-only continuum replacement).**
  Idea: exact near window ±W plus composite Gauss–Legendre over equal-mass
  panels (or geometric annuli centered on each query, weights from exact
  empirical counts) for the smooth far field. Refuted numerically: any
  scheme that replaces discrete sources by their continuous mass carries an
  Euler–Maclaurin-type relative error ~(panel_width/distance)² per panel.
  The singularity follows the query point, so global panels cannot refine
  toward it; annuli can, but the first rings always have width ≈ distance,
  producing an irreducible ~1e-2 floor across p = 5k..50k, insensitive to
  W, q and ring count. Reaching 1e-10 requires high-order local expansions —
  i.e. exactly what ChebCode already does; its order-n interpolation beats
  the (h/d)² wall by construction.
- **Taylor-per-panel amortization on top of that far field** inherits the
  same wall analytically (the Taylor expansion reproduces only the smooth
  part; the lattice discrepancy in the mid-field remains), so it was not
  implemented.
- **Pure-uniform RandNLA sketching for near-field Cauchy blocks.** The
  kernel's interaction mass concentrates on a handful of boundary columns;
  uniform column samples miss it entirely (rank-8 block error ~1e-2 where
  greedy pivoted skeletons reach ~2e-5 — a 500× per-rank efficiency gap that
  widens as η = 1/√p shrinks). Boundary-stratified sampling recovers most of
  the gap at moderate rank but still loses end-to-end to ACA on MP spectra:
  rank-capped accuracy degrades with p (7.5e-8 at p=1000 vs 1e-3 at
  p=50000) while its O(ℓ²(m+n)) machinery costs more than ACA's deflation at
  the same sizes. Adaptive pivoting is not an optimization detail here — it
  is the mechanism that makes low-rank compression viable for this kernel.
  The stratified path is kept behind `HodlrMode::Random` for future kernels
  without boundary concentration; `HodlrMode::Aca` remains the default.

- **Black-box FMM for the Stieltjes transform** (adaptive Chebyshev panels,
  P2M anterpolation, M2M merging, per-leaf well-separated pair DFS, direct
  M2L, barycentric local evaluation). Three variants were built and
  measured; all diverge numerically, and the cause is structural:
  1. index-uniform hierarchy + density-interpolation M2M: on MP-like spectra
     the parent can be dozens of child-widths wide in *value* space, so M2M
     extrapolates a degree-n Lagrange density far outside its interval —
     equivalent densities hit 1e8 and propagate NaN through the barycentric
     sums;
  2. field-evaluation M2M (parent weights = children's field at parent
     nodes): mathematically invalid — parent nodes sit inside the child's
     θ-zone where the anterpolated field is not trustworthy; densities
     grow like width⁻¹ per level and reach 1e17;
  3. adaptive (≤2×) hierarchy + density-interpolation M2M: geometrically
     consistent, but each merge extrapolates one full child width, costing
     the Lagrange Lebesgue factor at |s|=3 (~10³ for n=8) — and that factor
     composes per level (measured 17 → 1.4e4 in one merge), so deep trees
     explode regardless of tuning.
  Conclusion: 1D Coulomb-family kernels with an off-real-axis pole admit no
  stable black-box FMM translation; the treecode evaluation (ChebCode)
  remains the right algorithmic point in this family. The prototype was
  removed; the analysis is preserved here and in the session report.
- **NUFFT evaluation of the Stieltjes transform.** Two formulations were
  implemented end-to-end and benchmarked before removal:

1. *Spectral filtering* on Fourier modes of the density using the analytic
   Cauchy spectra (`πe^(−η|ω|)`, `−iπsign(ω)e^(−η|ω|)`): correct per-stage,
   but the result is the **periodic** summation of S — the Cauchy kernel
   decays only like 1/x, so wrap-around images sit at O(1) relative level.
2. *Laplace quadrature* `1/(z−x) = i∫₀^∞ e^{−is(z−x)}ds` with two NUFFTs
   (image-free in real space): verified to machine-precision consistency
   stage-by-stage, but the s-domain trapezoid computes the same periodized
   kernel through pole-image aliasing (`Σ_m 1/(z'+mL)` terms).

Conclusion: the obstruction is structural — any uniform-grid method for an
algebraically decaying kernel pays un-removable pole/wrap-around images
unless the padding grows like the exact cost. This also identifies the
dominant error source of the uniform-grid FFT family (~1e-4…1e-1). High
accuracy therefore belongs to local approximation (ChebCode/FMM), which has
no global grid; ChebCode remains the approximate-frontier optimum.

### Removed
- `stieltjes::term::fast_reciprocal` and the `stieltjes_term_fast*` variants:
  AArch64 NEON has no vector f64 reciprocal estimate (`vrecpe_f64` is
  scalar-lane), so the Newton-Raphson chain measured **3× slower** than
  hardware `fdiv` and was never wired into any kernel. The crate is back to
  zero `unsafe` (matching the README).
- `stieltjes::dst` module (the DST-I real-part implementation): dominated by
  the fft5 odd-kernel path it duplicated; `StieltjesMethod::Dst` remains as
  an alias for the Adaptive composition (see above).
- `src/stieltjes/fftgrid.rs`: superseded 3-FFT variant that was not even
  declared as a module (dead file).
- `FlatChebTree.lam` field: barycentric weights are build-only data.
- **Finalization sweep (API surface reduction).** Every item below had
  zero callers outside its own definition/tests, or was a literal alias:
  - `deconvolution::deconvolve_density`, `pipeline::estimate_noise_variance`
    (+ its private `median`), `pipeline::clean_covariance_from_data`,
    `rmt::reconstruct_covariance_basic`, `spiked::debias_eigenvector`,
    `deconvolution::rie_shrinkage_naive`, `adaptive::DEFAULT_ETA_LEVELS`
    (documented a default that the required parameter never applied);
  - `src/math/` — the `C64` complex type (4 of 7 methods unused; the
    crate-wide `allow(dead_code)` died with it) and `stieltjes_term_c64`;
    its one consumer test now verifies against plain `(re, im)` arithmetic;
  - `stieltjes::fft2` / `fft3` modules — 100 % aliases of fft5 whose own
    docs said so; the enum variants remain (Python strings + recorded
    tables key on them) and share one dispatch arm;
  - five dead term variants (`_fma`, `_cutoff`, `_cutoff_hoisted`,
    `_symmetric_pair`, `_complex`) and `term::CUTOFF_RATIO`; zero-reference
    wrappers `compute_all_stieltjes_treecode` / `_hodlr` (dispatch defaults
    became named consts next to each kernel: `treecode::DEFAULT_THETA/_ORDER`,
    `hodlr::DEFAULT_LEAF/_ACA_TOL/_ACA_RANK`);
  - **the config knobs were NOT removed.** An earlier revision of this entry
    also listed `Strategy`/`with_strategy`,
    `Precision`/`precision`/`with_precision`,
    `FftGridSize`/`fft_grid_size`/`with_fft_grid`, `RmtConfig::label`,
    `StieltjesMethod::{description, all}` and `Parallelism::name` as removed.
    That never happened: the paragraph was written by a changelog-only commit
    (`871cb01`) that did not touch `src/config.rs`. All of them are still
    present and public. Corrected under 0.1.1 below;
  - campaign probes: benches `chebyshev_fmm`/`local_expansion` (~530 lines
    of prototype FMM living inside bench files), examples
    `profile_cheb`/`fft_bench`/`check_poly` (its Horner-instability proof
    moved into the chebcode module docs), scripts
    `rie_core`/`proto_adaptive`/`proto_split_padding`/
    `check_analytic_kernel`/`measure_real_cutoff`. Cargo.toml carries five
    [[bench]] targets, all real.
- `spiked::detection::mp_upper_edge(sigma2, gamma)`: identical body to
  `estimation::bbp_threshold(gamma, sigma2)` with swapped arguments.
  One formula lives once now, under `bbp_threshold`.
- The ignored `block_size` parameters of `compute_all_stieltjes_blocked`,
  `_blocked_parallel`, `_blocked_windowed_parallel` and
  `_blocked_autovec_parallel`: four functions accepted a knob that did
  nothing while the dispatcher forwarded user values into them. The
  sequential windowed kernel keeps its (it uses it). Sibling signatures no
  longer swap argument positions either.
- Visibility aligned with real use: `stieltjes_sum_for_one`,
  `stieltjes_sum_windowed`, `auto_tiled_block_size` are private;
  dispatcher-only kernels are `pub(crate)` (the fake parallel block-size
  tuner became `PARALLEL_TILED_BS`).

### Fixed
- **Tiny-p Rayon requests no longer pay the thread-pool floor on the exact
  family.** Requesting `parallelism="rayon"` below p≈512 routed to a
  per-row Rayon fallback whose flat join overhead (~20 µs) exceeded the
  entire sequential computation — an end-to-end regression against plain
  vectorized NumPy at p ≤ 128. A parallel request now runs the sequential
  kernel below `RAYON_MIN_P = 512` (identical results, strictly faster);
  measured through the Python boundary: p=4 drops 18.1 µs -> 0.71 µs,
  p=64 46.6 µs -> 2.5 µs, keeping shrinkers 6–10× ahead of NumPy at every
  size instead of losing up to p≈128.
- **`cutoff=None` raised ValueError on documented calls.** The
  `InferredF64` extractor accepted only floats and the string
  `"inferred"`, yet both `deconvolve_spiked` and `stieltjes_transform`
  document `None` as the disabled spelling. `None` is now a synonym of
  `"inferred"`, and `detect_spikes_tracy_widom`'s `sigma2` joins the same
  grammar (it previously typed as bare `Option<f64>`, so `"inferred"`
  TypeError'd). Regression tests in `tests/test_python_api.py`.
- **`deconvolve_spiked` with a ChebCode method was quadratic.**
  `compute_stieltjes_at_points` had no ChebCode arm, so grid evaluations
  fell through to the O(p²) scalar fallback PER QUERY POINT. The tree is
  now built once and serves the whole grid: at p=20000/n_points=200 the
  grid costs 3.3 ms (rayon) at rel-err 4e-10 — previously minutes-scale.
- Tiled-kernel hot body is now a single source of truth
  (`tiled_span_*`/`tiled_one_block_*`) shared by the sequential and parallel
  kernels — the refactor is measured at parity with (or slightly ahead of)
  the original hand-unrolled monolith (159.8 vs 160.0 ms at p=20000).

### Notes
- **Pareto table regenerated against the final kernel set** (fresh full
  sweep through `examples/measure_pareto_frontier` +
  `scripts/build_pareto_table.py`).
  Headline change: `ChebCodeFast` now owns
  every speed-intent bin sequentially (an initial exception at
  p = 50000 for `Fft5` did not reproduce under interleaved re-measurement);
  accuracy-intent flips stay inside the exact
  family on runtime tie-breaks. Single-session sweep — bins decided by
  runtime ties carry normal run-to-run noise.
- **Two η conventions coexist by design, both documented** in the
  stieltjes module Conventions list: the library default is η = 0.1/√p
  (`stieltjes::default_eta`), while every recorded harness measures at
  η = 1/√p (declared in their JSON meta). Unifying them would silently
  invalidate every recorded number, so they are kept and cross-referenced;
  remember that larger η means more smoothing, so approximate-method
  errors recorded at 1/√p are optimistic relative to default-η calls.
- **Rebuild the wheel before any release smoke test**
  (`pixi run build`, or maturin directly with `CONDA_PREFIX` set) — a
  stale installed module is indistinguishable from a code regression.

## Internal history — pre-publication

First release under the new name.

### Changed
- Renamed crate & Python package: `rmt_kernel` / `fast_rmt_shrinkage` → **shrinkers**
  (`import shrinkers`, `pip install shrinkers`, `cargo add shrinkers`)
- Python package version now sourced from `Cargo.toml`
  (`dynamic = ["version"]`) so wheel/crate versions can't drift

### Added
- Spiked + bulk spectral deconvolution entry point `deconvolve_spiked`
  (BEMA spike detection → inverse-BBP debiasing → El Karoui
  Marčenko–Pastur bulk inversion)
- 13 Stieltjes-transform strategies from exact O(p²) SIMD kernels to
  O(p log p) FFT / FMM / DST approximations
- GitHub Actions CI: Rust fmt/clippy/test + Python wheel build &
  pytest suite (CPython 3.10/3.13)
- Release workflow: multi-platform wheels (Linux x86_64/aarch64,
  macOS arm64/x86_64) + sdist, published to PyPI via trusted
  publishing on `v*` tags

