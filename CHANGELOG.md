# Changelog

All notable changes to **shrinkers** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
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

### Added
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
  ChebCode's operating point — at 203 ms seq / 89 ms rayon for p=50000
  (~9×/10× slower than ChebCode). Verdict: dominated on this spectrum family
  (ChebCode wins speed-at-accuracy, BlockedTiled wins exact), but kept as
  the portfolio's kernel-agnostic member: it needs nothing but kernel
  evaluations, so it transfers unchanged to future kernels without FFT or
  tree structure. Implementation notes: skeleton pivots must skip previously
  used rows/columns (re-use makes the residual column vanish identically and
  division by the machine-zero pivot destroys the factors), and the stopping
  rule must test the normalized next-term norm ‖u‖·‖v‖/|pivot| — the raw
  product ignores the pivot scale and stopped ~1e4× before tolerance.

### Added
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

### Rejected (research track — documented negative result)
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

### Rejected (research track — documented negative result)
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

### Changed (continued)
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

### Investigated and rejected: NUFFT evaluation of the Stieltjes transform
Two formulations were implemented end-to-end and benchmarked before removal:

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

### Fixed
- Tiled-kernel hot body is now a single source of truth
  (`tiled_span_*`/`tiled_one_block_*`) shared by the sequential and parallel
  kernels — the refactor is measured at parity with (or slightly ahead of)
  the original hand-unrolled monolith (159.8 vs 160.0 ms at p=20000).

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
