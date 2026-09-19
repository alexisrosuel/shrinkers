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

`stieltjes_term_hoisted(lambda_i, lambda_j, eta)` (internal, `pub(crate)`) returns `(re, im)`
with the expensive part structured as one reciprocal plus FMAs instead of
a full complex division:

    inv = 1 / ((λi − λj)² + η²)
    re  = (λi − λj) · inv          via mul_add
    im  = −η · inv                 via mul_add

One divide per term is the floor; everything around it keeps the two FP
pipes busy.

### Symmetric sweep: evaluate each pair once (`cacheblock.rs`)

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

Each visited pair accumulates **four** partial sums — `rr`/`ri` for the
target row and `cr`/`ci` for the mirrored column — so the accumulation
arithmetic dominates the body. It is written as four explicit `mul_add`
calls (`rr = d.mul_add(inv, rr)`, `cr = (-d).mul_add(inv, cr)`,
`ri = eta.mul_add(inv, ri)`, `ci = eta.mul_add(inv, ci)`) rather than as
`w = d·inv; v = η·inv` plus four separate adds: that is 6 FP ops per pair
(`FSUB + FFMA + FDIV + 4 FFMA`) instead of 9, and at ~92 % of the FP issue
rate the saving is the runtime — interleaved before/after **1.15–1.16×** at
p = 10 000 / 25 000 / 50 000. `mul_add` is a true fused primitive in Rust,
so the contraction does not depend on the `fp-contract` codegen policy, and
each fused accumulation carries one rounding instead of two.

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
* **Register-resident target tiles in the *parallel* exact body**: the
  output-partitioned `tiled_one_block_no_cutoff` accumulates into `out_r[i]`
  / `out_i[i]` (two read-modify-writes per pair) inside a 32-row block that
  stays in L1. Moving the four target rows into registers and flushing once
  per source sweep removes those writes but forces the source array to be
  re-streamed once per 4-row tile instead of once per 32-row block:
  p = 10 000 Rayon 4.94 → 5.79 ms, p = 50 000 119.5 → 137 ms. Reverted.
* **A parallel symmetric kernel**: splitting the strict upper triangle into
  folded, exactly-balanced row strips and reducing per-worker mirror planes
  halves both the pairs and the divisions, yet the worker-local sweep ran
  ~1.6× slower per pair than the sequential sink sweep at one thread
  (25.2 → 39.7 ms at p = 10 000) and never beat the full-square parallel
  kernel at any thread count (10 threads, p = 50 000: 133 vs 116 ms).
  Collapsing the mirror planes onto the owner planes and using a single
  strip changed nothing, so the penalty lives in the sweep's two-stream
  write pattern, not in the reduction. Reverted.

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
  The shared `mass·(1/s)` factor is hoisted out of the node loop and applied
  with one fused `w_j = m.mul_add(v_j, w_j)` per node instead of
  `mass · v_j · inv_s` + add.
* `merge_weights` composes a parent's weights from its children's
  (O(n²) per child) instead of rescanning all parent sources — on the
  **all-points** path the build ends at 10–11 % of end-to-end runtime
  (`examples/measure_build_share.rs`) rather than dominating it.
* **The build dominates whenever the tree serves few queries.** At
  `nq = 200` (the deconvolution grid) the build is 81 % / 94 % of the call
  at p = 10 000 / 50 000, and most of it is `merge_weights`
  (cost ∝ `p / leaf_cap`). `compute_stieltjes_at_points` therefore sizes
  the leaf from the query count (`L* = n·√(2p/nq)`, floored at the preset
  value) — a bigger leaf removes merge work *and* leaves more sources in
  the exactly-summed leaves, so it is faster **and** more accurate.
  Interleaved before/after, nq = 200, `chebcode_fast`: **1.37×** at
  p = 10 000 (0.250 → 0.183 ms), **1.45×** at p = 25 000, **1.60×** at
  p = 50 000 (1.017 → 0.636 ms, 4.1e-9 → 1.3e-9 rel-L2);
  `chebcode_balanced` **1.77×** at p = 50 000.
  The all-points path is untouched (`L* = n√2` there, below every preset).

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

### Four-lane f32 far field (`F32x4`, `chebcode_fast` round)

The far-field dot product is already an interpolation (~1e-5 at the speed
preset), so its arithmetic does not need 53 bits. AArch64 gives **4 f32 lanes**
per NEON register against 2 for f64, and an f32 refined reciprocal
(`vrecpeq_f32` + two `vrecpsq_f32`) needs no scalar divide. `F32x4` mirrors
`F64x2` (same safe-abstraction policy, all `unsafe` still only in `simd.rs`),
and the tree keeps an `nodes_f32`/`w_f32` mirror of the flat panel arrays so
one build serves both arithmetics.

Only the well-separated far field is f32: traversal, the distance test, the
leaf exact sums and the returned accumulators stay f64, and each panel's four
lanes are reduced into an f64 scalar. Measured **~1.30x** on the far field at
a ~6e-7 relative-error floor. `n` must be a multiple of 4 to see it (n = 4 →
1.10x, n = 6 = 4 + masked tail → 0.99x, n = 8 → 1.33-1.37x), which is why the
speed preset uses n = 8; a non-multiple-of-4 tail is packed into one masked
4-lane op with zero weights rather than a scalar `1.0f32/x` loop, whose
hardware divide does not pipeline like the estimate. The accuracy presets
(`chebcode`, `chebcode_balanced`, `chebcode_xtreme`) keep the f64 path — a
17-bit `F64x2::recip_fast` (one Newton step) did measure +1.19-1.24x on the
f64 far field with a ~1.4e-6 floor, but it would cap `chebcode_xtreme`'s
~1e-12 class, so it survives only as a measurement reference.

`eval_points_parallel` writes into one preallocated output through
`par_chunks_mut`, and the traversal stack is a fixed inline
`TraversalStack` whose depth bound is *guaranteed* by the builder
(`MAX_TREE_DEPTH`, a deeper node becomes an oversized exact leaf) — together
1.04-1.07x on Rayon at p = 50 000 with seq CPU exactly neutral.

### ChebCodeFast round: what did NOT work (measured negatives)

Six ideas were implemented, measured against the shipped point with the same
harnesses, and reverted. They are recorded here (and in the `[Unreleased]`
CHANGELOG round) so they are not re-attempted.

* **k-ary tree (k = 4, 8)** — generalizing the binary tree to k children
  (flat `children[]`, k-way split, k-way `merge_weights`, k-aware traversal)
  is **1.1-2.0x SLOWER** at equal accuracy (p = 50 000 seq: k=2 13.10 ms,
  k=4 16.65, k=8 21.55). The hypothesis — `log_k` fewer levels means fewer
  accepted panels — is false because accepted panels *per level* are O(k),
  not O(1): an instrumented traversal counted ~178/246/221 far-field terms
  per query for k = 2/4/8 at p = 10 000. Worse, the shallow k = 8 tree dumps
  the near field into exact leaf sums (17 → 40 → **458** leaf sources per
  query). Re-tuning `leaf_cap` to 8/16 never reached parity. Keep the binary
  tree. (k = 2 was kept bit-identical throughout, FNV fingerprint
  `ba33dff5492e5540`.)
* **Packed AoS `Node`** (one 32 B cache line per visit instead of five SoA
  arrays): **+2.6 % sequential CPU**, wall `eval.seq` 0.93-0.96x. The
  metadata arrays are traversed nearly sequentially, so the SoA layout
  already prefetches well, while AoS adds address arithmetic and a
  `sub`+`mul` on the acceptance branch.
* **Even-`n` zero-weight padding** (remove the odd-`n` scalar tail by
  appending a zero-weight node): **+3.3 % sequential CPU**; grid CPU exactly
  unchanged (0.280 s → 0.280 s over fixed work). The padded lane still pays a
  full refined reciprocal and the flat arrays grow from stride 9 to 10.
  Wall-clock "grid wins" seen under load (up to 1.17x) were contention
  artifacts. Moot now: the speed preset uses n = 8.
* **Two-accumulator far-field unroll** (two independent `F64x2` accumulator
  pairs, to halve the serial FMA chain): p = 50 000 sequential
  9.22 → 9.87 ms. The n/2 FMAs were never the critical path; the extra live
  registers cost more than the shortened chain.
* **Two-lane refined-reciprocal `barycentric_row`** (vectorize the build's
  `β_j/(x−t_j)` with `F64x2::recip`): **~20 % SLOWER** than the scalar `fdiv`
  (p = 50 000 build 0.84 → 1.03 ms; `fills-only` 0.38 → 0.47 ms). The build's
  `n`-loop is short, so the reciprocal's serial 7-op latency chain plus the
  extra loads/stores beats any throughput gain. Per-lane `d == 0` checks made
  it worse still (two lane extractions per pair); the exact-node hit is
  instead detected by one post-loop `s.is_finite()` test. The build stays
  scalar.
* **Parallel tree build** — two variants, both gated on `parallel = true` and
  p ≥ 2000: (a) structural DFS + parallel leaf projections + per-depth
  parallel merges, (b) leaf-only parallel projections with serial reverse
  merges. Both **0.63-0.93x** end-to-end (`all.par`). At p = 50 000 the whole
  build is ~0.8 ms, while a Rayon level barrier plus per-level buffer
  allocation and serial scatter costs more than that — the tree is deep
  (≈ 11 levels) and each level is small. The build stays serial, and that is
  what caps `all.par` at ~1.4-1.8x.

## Methodology note

Every optimization above survived an interleaved A/B measurement before
shipping; the ones listed as negatives were measured too and documented
rather than deleted. Reproduce with the `examples/measure_*.rs` harnesses
(median ≥9 interleaved repetitions unless stated otherwise).

**Load caveat.** Several of these decisions were taken while other workstreams
were benchmarking on the same 10-core machine (load average 8-12), where wall
clock swung by up to 2.7x. Wall-clock A/B ratios from interleaved (ABBA) runs
are still usable, but *accept/reject decisions on sub-5 % effects* were taken
on contention-independent evidence: a fixed-work loop wrapped in
`/usr/bin/time -l`, comparing **user CPU seconds** (noise floor ±0.4-1.2 %).
Six of the negatives above (node packing, padding, the accumulator unroll) are
exactly the kind of small effect that wall clock alone would have
mis-attributed — a reminder to reach for CPU time when the expected delta is
a few percent.
