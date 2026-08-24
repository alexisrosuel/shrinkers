# Changelog

All notable changes to **shrinkers** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
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
  `cargo run --release --example pareto_data -- after > docs/pareto/bench_after.json`
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

### Added
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
  - config knobs that did nothing: `Strategy` (+ `with_strategy`),
    `Precision` + `RmtConfig::precision` + `with_precision` (the real f32
    path is Python's `precision="f32"` side channel), `FftGridSize` +
    `fft_grid_size` + `with_fft_grid` (`Custom` was never constructible),
    `RmtConfig::label`, `StieltjesMethod::{description, all}`,
    `Parallelism::name`;
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
- **The Pareto table predates the latest kernel work.**
  `src/config/pareto_autogen.rs` was generated from the overnight
  benchmark round; the symmetric-pair exact rewrite, the `SYM_AOS_MAX_P`
  layout switch and the ChebCode hw_sq/branchless acceptance landed
  afterwards. The Speed/Accuracy picks remain sane (the accuracy ordering
  did not change) but are no longer measured-optimal — regenerate with
  `examples/pareto_data` + `scripts/build_pareto_table.py` before trusting
  `speed_auto`/`accuracy_auto` in production.
- **Cross-harness η conventions differ, by history not by intent:** the
  criterion benches and the library default use η = 0.1/√p, while
  `bench_one` / `pareto_data` / `small_p_crossover` record with η = 1/√p
  (declared in their JSON meta). Numbers from the two families must not be
  read side-by-side as if comparable.
- **The installed wheel is stale relative to this source tree** (it lacks
  the sentinel fix, the newer method strings and `stieltjes_transform_
  with_deriv`). Run `pixi run build` before any release smoke test.

## [0.3.0] — shrinkers

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

[0.3.0]: https://github.com/alexisrosuel/shrinkers/releases/tag/v0.3.0
