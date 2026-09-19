# f32 4-lane far-field path for the Chebyshev treecode (`chebcode_fast`)

Worktree: `/Users/alexisrosuel/fast_rmt_shrinkage/.wt/f32` (branch `wt/f32`).
Machine: Apple M1 Max (aarch64, 8P+2E), macOS, `-C target-cpu=native`.
**All timings below were taken while other worktrees/builds were running on the
same machine** (load average ~10-13 on 10 cores). Absolute milliseconds are
therefore inflated and volatile; the *interleaved same-binary ratio* is the
evidence, as instructed.

## What changed

### `src/stieltjes/simd.rs`
- New safe `F32x4` type, same style/contract as `F64x2`:
  `splat/zero/load/from_array/lane/mul/fma/sub/recip/hsum`.
  AArch64 backend uses `vdupq_n_f32`, `vld1q_f32`, `vsubq_f32`, `vmulq_f32`,
  `vfmaq_f32`, `vgetq_lane_f32`, `vaddvq_f32`, and a 4-lane reciprocal from
  `vrecpeq_f32` + **two** `vrecpsq_f32` refinement steps (8 → 16 → ≥24 bits).
  Non-aarch64 fallback is `[f32; 4]` with true division. The module docs now
  cover both types; **all `unsafe` in the crate still lives only in this file**.
- Unit tests: refined reciprocal vs division, and lane-wise ops vs scalar.

### `src/stieltjes/chebcode.rs`
- New `pub enum FastMode { F64, F32 }` (default `F64`).
- `FlatChebTree` gains an f32 mirror of the flattened panel nodes/weights
  (`nodes_f32`, `w_f32`), derived once from the f64 arrays after the build.
  The f64 arrays are byte-for-byte unchanged.
- `contribution` was turned into a shared `contribution_mode<const FAST_F32: bool>`
  plus thin `contribution` / `contribution_f32` wrappers. The `false`
  instantiation is the *same statements in the same order* as before, so the
  f64 path is unchanged (and remains the default).
  **Only the well-separated far-field dot product changes**: traversal,
  distance test, leaf exact sums and the returned `(re, im)` accumulators are
  f64 in both modes. f32 lanes are reduced with `hsum()` and cast to f64 per
  panel.
- `n % 4 != 0` tail: the remaining nodes are packed into **one masked 4-lane
  op with zero weights in the dead lanes** (`F32x4::from_array`), rather than a
  scalar `1.0f32/x` tail.
- New public entry point `compute_all_stieltjes_chebcode_impl_f32(...)` (same
  signature/contract as the f64 one) and `ChebCodeBatch::{evaluate_mode,
  evaluate_point_f32, evaluate_points_mode}` so both paths can be A/B'd from
  one binary and one shared tree.
- New test `chebcode_f32_far_field_tracks_f64`: f32-vs-f64 rel-L2 < 1e-4 and
  sequential f32 == parallel f32 bit-for-bit.

### `examples/measure_chebfast.rs`
- New `ab <p> [theta] [n]` mode: **interleaved** f64/f32 A/B (alternating arm
  order, 9 rounds for `eval.seq`, 7 for `eval.par`/`all.*`, 21 for grid),
  emitting `chebf.<metric>.{f64,f32}.p{p}`.
- `err <p> [theta] [n]` now also prints `chebf.rel_l2_f32[_par].p{p}`.

## Exact commands

```sh
cargo build --release --example measure_chebfast
./target/release/examples/measure_chebfast ab  50000            # interleaved f64 vs f32
./target/release/examples/measure_chebfast ab  50000 1.0 8      # preset override
./target/release/examples/measure_chebfast err 20000
cargo test --lib
cargo build --release 2>&1 | tail
```

## Interleaved A/B, shipped FAST preset (theta=0.5, n=9, leaf_cap=32)

Each cell is the **median of 3 harness runs**, each run already a 9-round
interleaved median (7 for par). `all.*` = public entry point incl. tree build.

| p | metric | f64 (ms) | f32 (ms) | speedup |
|---:|---|---:|---:|---:|
| 10 000 | eval.seq | 1.924 | 1.478 | **1.30x** |
| 10 000 | eval.par | 0.523 | 0.409 | **1.28x** |
| 10 000 | all.seq  | 2.100 | 1.710 | 1.23x |
| 10 000 | all.par  | 0.779 | 0.734 | 1.06x |
| 50 000 | eval.seq | 10.565 | 8.110 | **1.30x** |
| 50 000 | eval.par | 2.178 | 1.724 | **1.26x** |
| 50 000 | all.seq  | 11.546 | 9.052 | 1.28x |
| 50 000 | all.par  | 3.294 | 2.912 | 1.13x |
| 100 000 | eval.seq | 23.237 | 17.852 | **1.30x** |
| 100 000 | eval.par | 4.537 | 3.523 | **1.29x** |
| 100 000 | all.seq  | 25.149 | 19.777 | 1.27x |
| 100 000 | all.par  | 6.272 | 5.421 | 1.16x |

200-point deconvolution grid (p=100 000, single representative run):
`grid.seq` 0.0670 → 0.0555 ms (1.21x), `grid.par` 0.0745 → 0.0623 ms (1.20x).

Earlier runs under heavier load showed the same ~1.30x on `eval.seq`; on the
worst-loaded periods the *parallel* arms once inverted (f32 "slower") purely
from Rayon/scheduler noise, e.g. f64 `eval.par` measured 1.8 / 8.4 / 18.3 ms
across three consecutive runs. Treat `eval.par`/`all.par` as ~1.13-1.29x,
not as tight numbers.

## Accuracy (`err` mode, vs the exact autovec sum)

| p | f64 seq/par rel-L2 | f32 seq/par rel-L2 |
|---:|---|---|
| 10 000 | 2.109895e-8 | 6.030180e-7 |
| 20 000 | 8.754425e-9 | 7.802465e-7 |

f32 costs ~30-90x in relative error, but lands at **6-8e-7** — well inside the
relaxed 1e-5..1e-4 budget for the fast preset. Sequential and parallel f32 are
bit-identical.

## Preset sweep (p=50 000, clean run; `eval.seq`; err at p=10 000)

| theta | n | f64 ms | f32 ms | ratio | f64 err | f32 err |
|---:|---:|---:|---:|---:|---|---|
| 0.5 | 9 | 10.381 | 7.978 | 1.30x | 2.11e-8 | 6.03e-7 |
| 0.6 | 4 | 6.101 | 5.586 | 1.09x | 3.38e-4 | 3.38e-4 |
| 0.6 | 6 | 7.237 | 7.307 | 0.99x | 4.48e-6 | 4.57e-6 |
| 0.6 | 8 | 8.379 | 6.101 | 1.37x | 1.64e-7 | 6.79e-7 |
| 0.8 | 4 | 5.332 | 4.829 | 1.10x | 3.03e-3 | 3.03e-3 |
| 0.8 | 6 | 6.179 | 6.255 | 0.99x | 1.14e-4 | 1.14e-4 |
| 0.8 | 8 | 7.086 | 5.234 | 1.35x | 1.35e-5 | 1.35e-5 |
| 1.0 | 4 | 4.296 | 3.890 | 1.10x | 3.96e-3 | 3.96e-3 |
| 1.0 | 6 | 5.104 | 5.136 | 0.99x | 1.58e-4 | 1.58e-4 |
| 1.0 | 8 | 5.810 | 4.356 | 1.33x | 1.94e-5 | 1.94e-5 |

Two clear effects:
- **`n` a multiple of 4 is the f32 sweet spot** (n=8: 1.33-1.37x). n=4 panels
  are too short (per-panel overhead dominates → 1.10x) and n=6 (4+masked-2)
  shows no gain at all.
- **f32 error is invisible until interpolation error drops below f32
  rounding**: at n<=6 the f32 and f64 errors are identical to 3 digits; only
  at n=8 does the f32 rounding (~7e-7) start to add.

## Recommendation

**Integrate the f32 far-field path for `chebcode_fast`.** It is a drop-in,
strictly additive change (f64 stays default and bit-identical), buys a
consistent **~1.30x on evaluation / ~1.27x on the full entry point** at
p=10k-100k, and raises rel-L2 only from ~2e-8 to ~6e-7. The parallel path
gains ~1.13-1.29x (build/scheduling dominate `all.par`).

Two ways to use it:
1. Conservative: same FAST preset (0.5, 9), f32 far field → ~1.3x, 6e-7 error.
2. Aggressive: **f32 + (theta=1.0, n=8)** → 4.36 ms vs 10.38 ms for the current
   f64 FAST at p=50 000 (**~2.4x**), error 1.94e-5 — inside the accepted
   relaxed budget. This is the f32-enabled operating point worth adding as a
   separate "fastest" preset; it should get its own clean-machine confirmation
   before it replaces any shipped default.

## Negative results / things that did not pay off

- **f32 alone did not reach the hoped 1.5-2x on `eval.seq`.** Only the
  far-field dot product got cheaper; traversal (branchy pointer chase), the
  distance test and the exact leaf sums stay f64, and n=9 needs 4 f64 vector
  iterations + a tail vs 2 f32 iterations + a masked tail. ~1.30x is the real
  ceiling at the shipped preset.
- **n=6 gives no f32 speedup** (~0.99x) despite using one fewer vector op than
  f64 — the 4+masked-2 split plus per-panel fixed overhead eats the gain.
- **Scalar f32 tail vs masked zero-weight tail are within noise** (~2%, sign
  flips between runs). The masked tail is kept because it removes the scalar
  f32 hardware divide and makes the panel body uniform; it is not a measurable
  win on its own.
- **Padded zero-weight node arrays were not needed** ("pad with zero-weight
  nodes" done lazily via `from_array` on the last group instead), avoiding a
  padded stride for every panel.
- **`all.par` gains the least (1.06-1.16x)** because the serial tree build and
  Rayon scheduling dominate; parallel numbers on this loaded machine were too
  noisy for a tighter claim.
- `F32x4::lane` is kept for API parity with `F64x2` but is currently unused by
  the kernel (only `from_array`/`hsum` are), so it carries a scoped
  `#[allow(dead_code)]`.
