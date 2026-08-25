# Hardware-level optimizations

How `shrinkers` extracts performance from the machine, method by
method: the fast O(p²) exact kernels that own the small-p regime and the
ChebCode* treecodes that own everything else. Every claim below traces to
code in `src/` (file references given) or to measured artifacts under
`docs/` and `examples/`. Documented *negative* results are included on
purpose — they are part of why the shipped shape is what it is.

## Build policy: one flag, everywhere

`.cargo/config.toml` sets `-C target-cpu=native`. All auto-vectorized
loops therefore compile against the host ISA (NEON on Apple Silicon,
AVX2-class on x86) without per-target code. The hand-written SIMD layer
(`stieltjes::simd`) additionally provides a portable fallback so wheels
built for generic targets stay correct; the release pipeline compiles on
each platform's runner.

## The exact O(p²) kernels (`src/stieltjes/term.rs`, `cacheblock.rs`, `symmetric.rs`)

### Scalar term: hoisted reciprocal

`stieltjes_term_hoisted(lambda_i, lambda_j, eta)` returns `(re, im)`
with the expensive part structured as one reciprocal plus FMAs instead of
a full complex division:

    inv = 1 / ((λi − λj)² + η²)
    re  = (λi − λj) · inv          via mul_add
    im  = −η · inv                 via mul_add

One divide per term is the floor; everything around it keeps the two FP
pipes busy.

### Symmetric sweep: evaluate each pair once (`symmetric.rs`)

The kernel's antisymmetry `K(b,a) = −conj(K(a,b))` halves the work:
rows are produced in pairs, computing the upper triangle and mirroring.
The loop body is a **register-resident 4-row tile** (`rr[4]`, `ri[4]`
arrays):

* within-tile pairs (same 4×4 block): scalar accumulation with the sign
  flip;
* off-diagonal tiles: 4-column groups loaded as index-based loops —
  measured ~30 % faster than the iterator form (the compiler cannot prove
  non-aliasing through iterators over separately-borrowed rows);
* outputs are written through **SoA sinks** (`col_update` trait), keeping
  real and imaginary planes contiguous for the consumer loops.

This path is branchless after dispatch: `use_cutoff` selects one of two
tight monomorphic loop bodies up front, so the hot loop never tests a
flag (I-cache friendly).

### Blocked tiling: keep the working set in L1 (`cacheblock.rs`)

When a cutoff window is active, the naive row-major traversal streams the
whole matrix per row. `blocked_tiled` instead iterates **output blocks
outermost**, then walks only the source window each block needs. The
block size auto-tunes to the measured sweet spot: the smallest block that
still amortizes per-tile overhead — bs ∈ {4..128} at p = 10 k on M-series
gave 40.7 ms at bs8 rising monotonically to 51.7 ms at bs128 (table in
`cacheblock.rs`); below bs8 the per-tile bookkeeping dominates. The
parallel twin uses a smaller `PARALLEL_TILED_BS = 32`, and rayon spans
are grouped at `n_blocks / (threads·4)` to amortize task scheduling.

Below p ≈ 100 the output array itself already fits in cache, so tiling
buys nothing: the allocation-free AoS variant
(`compute_all_stieltjes_symmetric_scaled_aos`) is ~2.5× faster end-to-end
there (single contiguous output vector, zero setup); from p ≈ 100 on the
dense SoA streams win it back. The dispatch table encodes this crossover.

### What did NOT work (measured, kept as negatives)

* **Hand-SIMD for the symmetric sweep**: NEON two-lane versions of the
  pair-once loop ran *slower* than the scalar register-tile form — the
  mirror write pattern defeats lane pairing.
* **Dual-output symmetry loops defeat LLVM auto-vectorization** (+136–182 %
  on small p): writing `(re, im)` interleaved from a symmetric loop stops
  the auto-vectorizer cold; the sequential symmetric path stays scalar by
  design while the asymmetric blocked paths auto-vectorize cleanly.

## ChebCode* treecodes (`src/stieltjes/chebcode.rs`, `simd.rs`)

Algorithm-level design (tree layout, Chebyshev panels, opening-angle
traversal) is covered in [`chebcode_algorithms.md`](chebcode_algorithms.md);
here only the machine-facing decisions.

### Two-lane SIMD with refined reciprocal (`simd.rs`)

AArch64 has **no FP64 vector divide**, and scalar `f64` division latency
(~13 cycles) would dominate both the leaf sums and the far-field panel
evaluation. The crate's answer is a tiny unsafe core behind a safe type
`F64x2`:

* load/store/fma/splat map to NEON intrinsics (portable fallback for
  other ISAs);
* division is replaced by `recip()`: an approximate reciprocal refined by
  Newton–Raphson steps to full precision, i.e. a fixed chain of
  multiply-adds that pipelines on the FMA pipes instead of stalling on
  the divider.

All unsafe code in the crate lives in this module (see the unsafe-code
policy in `internals.md`).

Both hot loops share one lane layout — leaf near-field sums process
**source pairs per lane load**, far-field processes **panel-node pairs**
— so the same denominator shape `(z−t)² + η² → fma → recip → fma`
streams identically in either context.

### Per-term evaluation instead of polynomial evaluation

The far field could mathematically be evaluated as a degree-n rational
function via one Horner division. Measured reality: forming monomial
coefficients of polynomials whose roots cluster near ±1 amplifies rounding
catastrophically (relative errors of 10²–10³ observed), while the per-term
dot product evaluates every denominator exactly like the leaf loop — same
conditioning, unconditional stability. Speed is comparable; stability wins
outright.

Also documented there: processing panels in pairs (interleaved) measured
~10 % slower — the n-loop iterations are already independent and the
out-of-order core overlaps their reciprocal chains without help; pairing
only doubles live registers.

### Barycentric build: hoist divisions, compose parents

* `fill_weights` computes `v_j = β_j/(x_s − t_j)` once per (source, node)
  and normalizes with a single extra division per source (≤1 ulp change).
* `merge_weights` composes a parent's weights from its children's
  (O(n²) per child) instead of rescanning all parent sources — the build
  ends at 10–11 % of end-to-end runtime (`examples/measure_build_share.rs`)
  rather than dominating it.

### Memory layout and parallelism

* Structure-of-arrays tree vectors, no pointer chasing; child indices are
  `i32`s inside flat arrays.
* Each worker thread owns one reusable stack buffer across consecutive
  queries; queries are dispatched in chunks of **256** (measured plateau
  between 64 and 1024 at p = 50 k on M1 Max). Adjacent sorted eigenvalues
  traverse nearly identical paths, so chunk locality doubles as cache
  warmth.
* The HODLR family shares the SoA philosophy and, since the conj-transpose
  transfer, compresses each cross-block level once instead of twice
  (p = 20 k seq: −41 %; see CHANGELOG 0.1.x).

## Methodology note

Every optimization above survived an interleaved A/B measurement before
shipping; the ones listed as negatives were measured too and documented
rather than deleted. Reproduce with the `examples/measure_*.rs` harnesses
(median ≥9 interleaved repetitions unless stated otherwise).
