# Task: Optimize the `deconvolve_spiked` bulk deconvolution (Stieltjes transform bottleneck)

## Context

Repo: `/Users/alexisrosuel/shrinkers` (Rust crate `shrinkers`, PyO3 bindings).

The package's single public Python entry point is `deconvolve_spiked(eigenvalues, c, n_points=200, eta="inferred", margin=1.0)` (Rust: `src/deconvolution/spiked.rs::deconvolve_spiked`). It does:
1. Sort eigenvalues (O(p log p))
2. BEMA spike detection (`crate::spiked::detect_spikes_bema`, O(p))
3. Inverse BBP debiasing (O(k), negligible)
4. **Bulk deconvolution** → `src/deconvolution/mod.rs::spectral_deconvolution` — this is the bottleneck.

## Measured bottleneck (verified)

Profiling shows **~97–98% of wall-clock time is spent in the Stieltjes transform** inside `spectral_deconvolution`. It calls `empirical_stieltjes_at_point(z_real, z_imag, eigenvalues)` — a **naive O(p) scalar loop per grid point** — once per grid point, giving **O(p · n_points)** total. Everything else (sort + BEMA + debias) is ~2–3%.

Measured (p=1000, release build): runtime scales **linearly with `n_points`** (50→800 grid points: 0.057→0.751 ms). Linear fit: Stieltjes ≈ 0.93 µs/grid-point; fixed overhead ≈ 4.7 µs. At n_points=200 the Stieltjes share is ~97.5%.

## The key opportunity

The crate already has a **highly-optimized Stieltjes library** that the RIE/Ledoit–Wolf path uses, but `spectral_deconvolution` does **not** use it — it calls the naive per-point loop directly. This is the single biggest optimization lever.

Fast library entry point (in `src/stieltjes/mod.rs`):
```rust
pub fn compute_all_stieltjes(
    eigenvalues: &[f64],
    eta: f64,
    method: StieltjesMethod,
    fft_grid_size: Option<usize>,
    cutoff: CutoffConfig,
    block_size: usize,
    parallelism: Parallelism,
) -> Vec<(f64, f64)>  // (real, imag) pairs, scaled by 1/p
```
Available methods (see `StieltjesMethod` in `src/config.rs`): `Naive`, `AutoVectorized`, `Blocked`, `BlockedAutoVec`, `BlockedTiled`, `BlockedWindowed`, `Adaptive`, `Fft5`, `Fft3`, `Fft2`, `TreeCode`, `Ewald`, `Dst`, `Auto`. `StieltjesMethod::resolve(p, parallelism)` auto-selects (p≤200 autovec, 200<p<5000 blocked, p≥5000 fft2 seq / treecode par).

## IMPORTANT correctness constraint (read `src/stieltjes/mod.rs` and `/memories/repo/speed_strategy.md`)

The deconvolution needs **BOTH the real and imaginary parts** of the Stieltjes transform accurately. This is critical:
- **Imaginary part** Im[S]=Σ η/((λᵢ-λⱼ)²+η²) is short-range (Lorentzian, decays 1/d²) → windowed/cutoff methods truncate it cleanly.
- **Real part** Re[S]=Σ (λᵢ-λⱼ)/((λᵢ-λⱼ)²+η²) is the **Hilbert kernel, long-range (1/d), log-divergent** → a finite window/cutoff **destroys it** (161× error). `BlockedWindowed` is ONLY valid for the imaginary part — do NOT use it for the full transform.
- FFT odd-kernel computes the real part accurately (~15% grid error, O(p log p)) but needs large padding (grid ~1000·η), so FFT only wins at p≥~5000.
- Treecode/FMM with order ≥4–6 converges for BOTH parts (see shrinkers.md "Treecode performance").

So the naive loop is exact but slow; the fast methods are approximate. **You must preserve numerical correctness** — the deconvolution output (`density`, `w_re`, Stieltjes arrays) must remain accurate. Do not silently swap in an approximate method that breaks the real part.

## What to do

1. **Understand the current code first**: read `src/deconvolution/mod.rs` (esp. `spectral_deconvolution` and `empirical_stieltjes_at_point`), `src/stieltjes/mod.rs` (esp. `compute_all_stieltjes`, `stieltjes_sum_for_one`, `scale_soa`/`scale_aos`), `src/config.rs` (`RmtConfig`, `StieltjesMethod`, `resolve`), and the memory notes `/memories/repo/speed_strategy.md` and `/memories/repo/shrinkers.md`.

2. **Wire the fast Stieltjes library into `spectral_deconvolution`** so it no longer uses the naive per-point loop. The grid points are `z_k = λ_k + i·η` for k=0..n_points-1 (a uniform grid over `[lo, hi]`). Note the grid λ values are NOT the sample eigenvalues — they are a separate uniform grid. `compute_all_stieltjes` computes the transform at the *sample eigenvalues*; you need it at the *grid points*. Design accordingly (e.g. a variant that evaluates at arbitrary query points, or reuse `stieltjes_sum_for_one` per grid point with a fast single-point kernel, or an FFT-grid method that evaluates on a uniform grid — which matches the deconvolution grid naturally).

3. **Respect the config**: `spectral_deconvolution` currently takes `_config: &RmtConfig` (ignored). Use `config.stieltjes_method`, `config.parallelism`, `config.fft_grid_size`, `config.cutoff`, `config.block_size` to select the method, and call `config.resolve_auto(p)` where appropriate. The Python binding `deconvolve_spiked` currently builds `RmtConfig::new(c)` (default = `Blocked`, sequential) — consider whether to expose method/parallel/strategy again or keep it simple; keep the public Python API as `deconvolve_spiked(eigenvalues, c, n_points, eta, margin)` unless you add optional kwargs.

4. **Preserve exactness where it matters**: the default config is `Blocked` (exact O(p²)). Keep the default exact unless you deliberately choose an approximate method. The goal is to make the *same* computation faster, not to change results.

5. **Benchmark before/after**: use `scripts/bench_stieltjes.py` and/or a quick Python timing harness (like the one used to measure this: time `deconvolve_spiked` over `n_points` in [50,100,200,400,800] at p=1000, and over p in [100,500,1000,2000]). Report the speedup and confirm the output density is unchanged (compare against the current naive implementation numerically).

6. **Build & test**: `cargo test` (all tests must pass — there are ~100+ lib tests including deconvolution tests), `cargo build --release` warning-free, and rebuild the Python wheel with `maturin build --release` + `pip install --force-reinstall --no-deps target/wheels/shrinkers-0.1.0-cp314-cp314-macosx_11_0_arm64.whl` (the pixi env python is `/Users/alexisrosuel/shrinkers/.pixi/envs/default/bin/python`). Verify `import shrinkers` exposes `deconvolve_spiked`.

## Constraints / gotchas

- **Zero unsafe code** is a project invariant (all `unsafe` was removed). Do not introduce `unsafe`.
- **No duplicated kernel**: the core term `1/((λᵢ-λⱼ)-iη)` lives ONLY in `src/stieltjes/term.rs`. Reuse existing kernels; do not copy the formula.
- **Module separation**: `deconvolution/*` imports FROM `spiked`/`stieltjes` (one-way dep). Do not inject deconvolution code into stieltjes.
- **Contiguity**: Python arrays must be contiguous (reversed views raise ValueError).
- **Rayon on Apple M-series**: parallel blocked gives ~2.6–5.4× speedup at p≥1000; FFT methods don't benefit from parallelism.
- Do NOT change the public Python signature `deconvolve_spiked(eigenvalues, c, n_points=200, eta="inferred", margin=1.0)` unless adding optional kwargs.

## Deliverables

- A clear explanation of what you changed and why.
- Before/after benchmark numbers (wall-clock for `deconvolve_spiked` across p and n_points).
- Confirmation that the deconvolution output is numerically unchanged (or the error introduced by any approximate method, quantified).
- All tests passing, build warning-free, Python wheel rebuilt and verified.

---

## ✅ Result (2026-08-03)

### What changed

1. **`src/stieltjes/mod.rs`** — added `compute_stieltjes_at_points(query_points, eigenvalues, eta, method, cutoff, parallelism)`: evaluates the raw Stieltjes sum at **arbitrary query points** (not just the sample eigenvalues). For the blocked-family methods it dispatches to the new write-batched query-point kernel; for global/approximate methods (`Adaptive`/`Fft5`/`Fft3`/`Fft2`/`TreeCode`/`Ewald`/`Dst`/`Auto`) it falls back to the exact single-point kernel per query point (preserving correctness).

2. **`src/stieltjes/cacheblock.rs`** — added `compute_stieltjes_blocked_at_points(query_points, eigenvalues, eta, block_size, cutoff)`: the query-point analogue of `compute_all_stieltjes_blocked`. It iterates the **source** eigenvalues λⱼ in pairs (halving write traffic to the output arrays) while sweeping the **target** query points in cache-sized blocks — the same structure that makes `blocked_default` ~2× faster than the single-point autovec loop. Delegates to `term.rs` (no duplicated kernel).

3. **`src/deconvolution/mod.rs`** — `spectral_deconvolution` now builds the uniform grid `λ_k` over `[lo, hi]`, calls `compute_stieltjes_at_points` once (batched), and applies the El Karoui inversion per grid point. It respects the config via `config.clone().resolve_auto(p)` (method, parallelism, cutoff). The default config is `Blocked` (exact O(p²)) — the computation is unchanged, just faster.

   **Convention mapping**: the Stieltjes library computes `S(λ)=Σ 1/((λ-λⱼ)-iη)` (convention B, Im>0), scaled by `1/p`. The old code computed convention A (`1/(z-λⱼ)`, Im<0) then negated. Mapping: `g_real = -s_real·(1/p)`, `g_imag = +s_imag·(1/p)`. The golden-master regression test confirms an exact match.

4. **Cleanup** — fixed a pre-existing broken doctest in `cacheblock.rs` (pseudo-code in a Rust code block → `text`), and removed an unused import in `python.rs`.

### Refactor (2026-08-03): single source of truth for the blocked kernel

The two blocked kernels were structurally identical — `compute_all_stieltjes_blocked`
(target = sample eigenvalues) and `compute_stieltjes_blocked_at_points` (target =
arbitrary query points) differed only in the target array and output length. To
reuse all optimizations (cache blocking, write batching, loop unrolling, FMA,
far-field cutoff) on **both** the grid path and the eigenvalue path, the
arbitrary-points function is now the single source of truth:

- `compute_stieltjes_blocked_at_points(query_points, eigenvalues, ...)` — the
  canonical blocked kernel, evaluating at any set of points.
- `compute_all_stieltjes_blocked(eigenvalues, ...)` — now a thin wrapper that
  delegates to it with `query_points = eigenvalues`.

This removes ~300 lines of duplicated kernel body. Any future optimization to
the blocked kernel automatically benefits both the deconvolution grid and the
full-eigenvalue Stieltjes transform. All 103 lib tests pass; release build is
warning-free.

### Before / after benchmark (p=1000, M-series, release)

| n_points | Before (naive loop) | After (blocked kernel) | Speedup |
|----------|--------------------:|-----------------------:|--------:|
| 50       | 57 µs               | 28.5 µs                | 2.0×    |
| 100      | 96 µs               | 43.0 µs                | 2.2×    |
| 200      | 187 µs              | 80.4 µs                | 2.3×    |
| 400      | 375 µs              | 152 µs                 | 2.5×    |
| 800      | 751 µs              | 306 µs                 | 2.5×    |

| p (n_points=200) | After |
|------------------|------:|
| 100              | 11.6 µs |
| 500              | 40.9 µs |
| 1000             | 77.8 µs |
| 2000             | 152 µs  |

### Numerical correctness

Output is **numerically unchanged** (machine precision). Compared against a
pure-NumPy reference implementation of the El Karoui deconvolution (naive
O(p·n_points) Stieltjes loop) at p=1000, n_points=200:

- `density`: max rel err **3.0e-13**
- `w_re`: max rel err **2.3e-13**
- `sample_stieltjes_real`: max rel err **7.9e-13**
- `sample_stieltjes_imag`: max rel err **5.1e-15**
- `population_stieltjes_real`: max rel err **4.5e-14**
- `population_stieltjes_imag`: max rel err **3.0e-13**

The default config is `Blocked` (exact O(p²), no cutoff), so no approximation
is introduced. The golden-master regression test (`test_golden_master_regression`)
passes unchanged.

### Verification

- `cargo test`: **103 lib tests + doc tests pass** (incl. new
  `test_blocked_at_points_matches_single_point`).
- `cargo build --release`: **warning-free**.
- Wheel rebuilt via `maturin build --release`, reinstalled, and
  `import shrinkers` exposes `deconvolve_spiked` (verified).
- Benchmark script: `scripts/bench_deconvolve_spiked.py`.
