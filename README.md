# shrinkers

[![CI](https://github.com/alexisrosuel/shrinkers/actions/workflows/ci.yml/badge.svg)](https://github.com/alexisrosuel/shrinkers/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/alexisrosuel/shrinkers/blob/main/LICENSE)

**Fast RMT (Random Matrix Theory) Population Eigenvalue Estimation Kernel**

SIMD-accelerated Rust implementation of **spiked + bulk eigenvalue cleaning**
via free-probability deconvolution: detect spikes (BEMA), debias them (inverse
BBP), and deconvolve the bulk (El Karoui). Features O(p log p)
FFT-accelerated Stieltjes transforms, auto-vectorized loops, cache blocking,
and PyO3 bindings.

## Primary entry point: spiked + bulk deconvolution

The **core** of the crate is a single call that cleans the sample eigenvalues
under a spiked covariance model, recovering the population spectrum:

```python
from shrinkers import deconvolve_spiked

res = deconvolve_spiked(evals, c=0.25)
print(res["k"])                 # number of detected spikes
print(res["spikes"])            # debiased population spikes ℓ_i (descending)
print(res["bulk"]["density"])   # deconvolved bulk population density
```

It orchestrates the full pipeline:

1. **Spike detection** (BEMA) — find the sample eigenvalues that escape the
   bulk edge.
2. **Spike debiasing** (inverse BBP / DGJ) — recover the population spike
   eigenvalues $\ell_i$ from the biased sample spikes.
3. **Bulk deconvolution** (El Karoui 2008) — remove the spikes and invert the
   Marčenko–Pastur equation on the remaining bulk to recover the population
   spectral density.

> **Why El Karoui over RIE?** RIE / Ledoit–Wolf shrinkage is the *pointwise*
> special case of free-probability deconvolution: it maps each individual
> sample eigenvalue to a population estimate. El Karoui is the full inversion
> that recovers the entire population spectral density, which is the general
> approach this package uses for the bulk.

## Theory

For sample eigenvalues $\lambda_1, \ldots, \lambda_p$ and concentration ratio $c = p/n$, the RIE shrinks each eigenvalue via the empirical Stieltjes transform:

$$m_g(z) = \frac{1}{p}\sum_{j=1}^p \frac{1}{z - \lambda_j}, \quad z = \lambda_i - i\eta$$

$$\xi(\lambda_i) = \frac{\lambda_i}{\bigl|1 - c + c \cdot \lambda_i \cdot m_g(\lambda_i - i\eta)\bigr|^2}$$

The shrunk eigenvalues are then trace-preserving rescaled.

## Spectral Deconvolution (Marčenko-Pastur Inversion)

The bulk step of `deconvolve_spiked` implements **spectral deconvolution** — the inverse problem of recovering the **population spectral density** $\mu_{\Sigma}$ from sample eigenvalues by inverting the Marčenko-Pastur equation (El Karoui 2008):

$$m_{\hat{\Sigma}}(z) = \int \frac{d\mu_{\Sigma}(x)}{x(1 - c - c z m_{\hat{\Sigma}}(z)) - z}$$

The inversion works by computing $g(z) = m_{\hat{\Sigma}}(z)$ on a grid $z_k = \lambda_k + i\eta$, then using the change-of-variable:

$$a(z) = 1 - c - c z g(z), \quad w = \frac{z}{a(z)}$$

$$m_{\Sigma}(w) = a(z)\,g(z), \quad \rho_{\Sigma}\bigl(\Re(w)\bigr) = \frac{1}{\pi}\,\Im\bigl[m_{\Sigma}(w)\bigr]$$

This is useful for:
- **Model selection**: determine the number of factors/spikes from data
- **Noise model validation**: check if residuals follow the assumed noise distribution
- **Signal extraction**: identify population eigenvalues above the BBP transition
- **Covariance estimation**: when the noise is not i.i.d. white noise

### Hybrid spiked deconvolution

The classical deconvolution above assumes a **continuous** population spectral
density. When the true spectrum contains isolated **spikes** (strong signal
eigenvalues, e.g. market factors), applying it directly distorts both the
spike locations and the adjacent bulk (Gibbs oscillations, BBP bias). The
correct approach is a **two-stage hybrid**:

1. **Spike separation** — detect the sample eigenvalues that escape the bulk
   edge (BEMA) and *debias* each one back to its population value via the
   inverse BBP / DGJ formula.
2. **Bulk deconvolution** — remove the spike eigenvalues and run the El Karoui
   deconvolution on the remaining bulk, where the continuous-density
   assumption holds.

```python
from shrinkers import deconvolve_spiked

res = deconvolve_spiked(evals, c=0.25, n_points=300, eta=0.05)
print(res["k"])                 # number of detected spikes
print(res["spikes"])            # debiased population spikes ℓ_i (descending)
print(res["spike_sample"])      # sample eigenvalues classified as spikes
print(res["bulk"]["density"])   # deconvolved bulk density
```

## Python usage

```python
from shrinkers import deconvolve_spiked
import numpy as np

evals = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0], dtype=np.float64)

# Clean the sample eigenvalues (spiked + bulk deconvolution)
res = deconvolve_spiked(evals, c=0.3)
print(res["k"])                 # number of detected spikes
print(res["spikes"])            # debiased population spikes ℓ_i
print(res["bulk"]["density"])   # deconvolved bulk density
```

See `docs/python_api.md` for the full API reference.

## Stieltjes methods

| Method | Complexity | Type | Accuracy |
|--------|-----------|------|----------|
| `Naive` | O(p²) | Scalar loop | Machine precision |
| `AutoVectorized` | O(p²) | SIMD (NEON/AVX2) | Machine precision |
| `Blocked` | O(p²) | Cache-blocked + unrolled + FMA | Machine precision |
| `BlockedTiled` | O(p²) | 2D cache tiling (output-block-outer) | Machine precision |
| `BlockedWindowed` | O(p·k) | Binary-search far-field window | Imag only (short-range) |
| `Adaptive` | O(p log p) | FFT real + windowed imag | Balanced error |
| `Fft5` | O(p log p) | FFT convolution (5 FFTs) | ~0.15% grid error |
| `Fft3` | O(p log p) | Fused FFT grid (3 FFTs) | ~0.15% grid error |
| `Fft2` | O(p log p) | Packed 2-FFT grid | ~0.15% grid error |
| `TreeCode` | O(p log p) | 1D balanced tree (FMM) | User-controllable |
| `ChebCode` | O(p log p) | Chebyshev-interpolation tree | ~1e-10 rel |
| `Hodlr` | O(r² p log p) | Hierarchical low-rank (ACA / RandNLA modes) | tol-driven (5e-10 default) |
| `Ewald` | O(p·k + M log M) | Near/far splitting + coarse FFT | User-controllable |
| `Dst` | O(p log p) | Alias for the `Adaptive` composition (shared FFT grid) | ~0.15% grid error |
| `Auto` | — | Auto-selects fastest by p | — |

> **Notes.** `Dst` delegates to the same FFT grid as `Adaptive`/`Fft5`
> (identical math, fewer transforms). `Ewald` is dominated by `ChebCode` on
> both axes (faster *and* more accurate) and is kept for reference only.
> With a far-field cutoff enabled, the exact `Blocked` family routes to the
> windowed kernels: identical included terms, O(p·k) instead of O(p²).
> Presets are data-driven: `Strategy::Speed` / `Strategy::Accuracy` resolve
> through a measured Pareto table (`src/config/pareto_autogen.rs`,
> regenerate via `examples/pareto_data.rs` + `scripts/build_pareto_table.py`)
> with independent columns for Sequential/Rayon; the user's parallelism
> choice is respected. NUFFT-based evaluation was investigated and rejected —
> uniform grids cannot suppress the Cauchy kernel's algebraic wrap-around
> images (see CHANGELOG); local approximation wins the accuracy frontier.
>
> `Hodlr` compresses every off-diagonal block of the kernel matrix by adaptive
> cross approximation (skeleton pivots on actual kernel entries,
> self-validating per block; no opening-angle parameter, no analytic
> translations). A randomized RandNLA path ([`HodlrMode::Random`]:
> boundary-stratified column sketches + least-squares row fit, validated on
> whole boundary-strip test columns) shares the same driver. Measured verdict
> on MP spectra: `Hodlr` delivers ~10× lower error than `ChebCode`'s operating
> point at 4–9× its runtime, and is dominated by the exact blocked family in
> the zero-error regime — it wins no preset bin and stays as the
> kernel-agnostic member of the portfolio; the randomized variant loses to
> ACA everywhere on this family because near-field Cauchy mass concentrates
> on a handful of boundary columns that fixed sampling schemes cannot see
> (greedy pivots reach ~2e-5 block error at rank 8 where stratified sampling
> needs ~30 ranks). Runtime-vs-p charts:
> `docs/pareto/runtime_vs_p_{seq,rayon}.png`, banded cuts in
> `runtime_vs_p_grid.png`.

## Performance

### Core Stieltjes kernel (raw, Criterion, Apple M-series)

The direct-sum kernel is the hot path. Hardware optimizations — cache tiling
(output-block-outer), FMA fusion on the real part, eta-hoisting of the
imaginary part, and unrolled inner loops — have made the exact `BlockedTiled`
kernel the fastest exact method:

| p | `tiled_auto` (auto block size) |
|---|-------------------------------|
| 1000 | **293 µs** |
| 10000 | **29.3 ms** |

### Stieltjes method frontier (bench_one, Apple M1, MP spectra, η=1/√p)

Same-binary A/B, rel L2 error vs the exact reference:

| p=50 000 | error | seq | rayon |
|---|---|---|---|
| `ChebCode` | ~9e-9 | 23 ms | 10 ms |
| `Hodlr` (ACA, tol 1e-9) | ~7e-10 | 216 ms | 87 ms |
| `Hodlr` (Random/sketch) | ~1e-3 (rank-capped) | 2.6 s | 1.1 s |
| `BlockedTiled` (exact) | 0 | — | ~185 ms |

ChebCode owns the speed-at-accuracy frontier on this spectrum family; the
exact family owns the zero-error regime; `Hodlr` trades runtime for an
extra digit of accuracy plus kernel-agnosticism. Full curves:
`docs/pareto/runtime_vs_p_{seq,rayon}.png`; accuracy-banded cuts and the
combined grid: `docs/pareto/runtime_vs_p_grid.png`.

### γ-sweeps: build once, evaluate many

RIE deconvolution evaluates the same spectrum for many γ (one η per γ).
`stieltjes::ChebCodeBatch::build(...)` constructs the Chebyshev tree once;
`evaluate_many(&etas)` then runs the sweep with the tree shared read-only,
parallelizing ACROSS the η axis — measured **6.5–6.6×** faster than calling
`compute_all_stieltjes_chebcode` per η (`examples/bench_batch.rs`). For
root-finding on γ, `stieltjes_transform_with_deriv` returns S and its
analytic derivative dS/dx in a single exact pass.

### Pure Rust (Criterion, Apple M-series, p=1000, c=0.5)

Full `rie_shrinkage` pipeline (includes spike detection + shrinkage overhead):

| Method | Time | vs Autovec |
|--------|------|-----------|
| Naive | 962 µs | 1.0× |
| AutoVectorized | 962 µs | 1.0× |
| **Blocked (default)** | **318 µs** | **3.0×** |
| BlockedWindowed (imag only) | 40 µs | 24× |
| TreeCode (FMM) | 748 µs | 1.3× |
| FFT5 (grid) | 1011 µs | 0.95× |
| FFT3 (fused) | 998 µs | 0.96× |
| FFT2 (packed) | 994 µs | 0.97× |

> **The hardware-optimized `Blocked` kernel now beats the FFT methods at
> p=1000.** The FFT grid padding (`1000·η` with `η = 0.1/√p`) forces large
> grids at small p, so the O(p log p) FFT methods only win at very large p.
> For the pure imaginary part (spectral density reconstruction),
> `BlockedWindowed` is far faster still (40 µs).

### Python-level (via PyO3, `deconvolve_spiked`, n_points=200)

| p | time (µs) |
|---|-----------|
| 100 | 9.4 |
| 500 | 33.2 |
| 1000 | 63.2 |
| 2000 | 125.8 |

### Python-level: bulk deconvolution scaling (p=1000)

| n_points | time (µs) |
|----------|-----------|
| 50 | 19.3 |
| 100 | 34.7 |
| 200 | 63.3 |
| 400 | 121.2 |
| 800 | 247.3 |

Numerical correctness is preserved: the fast-kernel output matches a pure-NumPy
reference of the El Karoui deconvolution to ~1e-13 max relative error.

## Comparison with existing packages

The only other Python package implementing RIE shrinkage is **pyRMT** (PyPI). Our comparison (a one-off benchmark script, since removed from `scripts/`; the reference NumPy implementation lives in `scripts/rie_numpy.py`) reveals:

### 🐛 pyRMT has a critical bug

`pyRMT.stieltjes(z, E)` computes $\operatorname{tr}(zI - E)$ instead of the correct $\operatorname{tr}\bigl((zI - E)^{-1}\bigr)$ — the matrix inverse is missing. This makes its RIE shrinkage completely wrong (~360% relative error).

| Method | p=100, c=0.5 | p=500, c=0.5 |
|--------|-------------|-------------|
| `rie_numpy` (ref) | ✅ ground truth | ✅ ground truth |
| `shrinkers` (Rust autovec) | **1.2e-14** max diff | **4.4e-14** max diff |
| `shrinkers` (Rust FFT) | **9.5e-03** (0.15% error) | **3.0e-02** (0.16% error) |
| pyRMT (fixed, same η) | 3.3e-01 (2.5% error, η effect) | 6.2e-01 (1.7% error, η effect) |
| **pyRMT (original buggy)** | **❌ 1.7e+01 (358% error)** | **❌ 1.9e+01 (376% error)** |

Note: pyRMT uses $\eta = 1/\sqrt{p}$ vs our $0.1/\sqrt{p}$. When using the same η, both give identical results up to machine precision. The γ bias correction in pyRMT's `optimalShrinkage` has negligible effect at moderate p.

### Performance vs pyRMT

> Note: the earlier `stieltjes_transform` / RIE entry point has been replaced
> by `deconvolve_spiked` (spiked + bulk deconvolution). The `shrinkers` rows
> below are the current `deconvolve_spiked` timings (n_points=200).

| Method | p=100 | p=500 | p=1000 | Scaling |
|--------|-------|-------|--------|---------|
| **`shrinkers` deconvolve_spiked** | **9.4 µs** | **33.2 µs** | **63.2 µs** | O(p²) |
| `rie_numpy` (pure NumPy) | 60 µs | 1337 µs | 5381 µs | O(p²) |
| pyRMT (fixed, loop-based) | 1164 µs | 2695 µs | 6642 µs | O(p²) |
| **pyRMT (original buggy)** | **928 000 µs** | **1 474 000 µs** | — | **O(p³)** |

Key findings:
- **`shrinkers` is 6–80× faster than `rie_numpy`** and **~100–120× faster than pyRMT (fixed)** at p=100
- The **buggy pyRMT is O(p³)**: 928 ms at p=100, making it unusable beyond tiny dimensions
- The **hardware-optimized `Blocked` kernel** now beats the FFT methods at p=1000 in pure Rust (see the Performance section)

### Why `shrinkers` is the only correct & fast RIE

| Feature | pyRMT | Ledoit-Wolf (official) | **shrinkers** |
|---------|-------|----------------------|----------------|
| RIE algorithm | ✅ (but buggy) | ❌ (different method) | **✅ correct** |
| FFT acceleration | ❌ | ❌ | **✅ O(p log p)** |
| Eigenvalue-only API | ❌ (needs X) | ❌ (needs X) | **✅ yes** |
| SIMD auto-vectorization | ❌ | ❌ | **✅ NEON/AVX2** |
| Multiple strategies | ❌ | ❌ | **✅ 13 methods** |
| Zero unsafe code* | N/A | N/A | **✅ (1 audited module)** |
| Maintenance | Last updated 2017 | Sporadic | **✅ Active** |

## Unsafe code policy

The crate is unsafe-free with **one deliberate, audited exception**:
`src/stieltjes/simd.rs`. AArch64 NEON exposes no FP64 vector divide, so the
hot Stieltjes kernels use a Newton–Raphson refined reciprocal
(`vrecpeq_f64` + three `vrecpsq_f64` steps; ≥53 significant bits) to keep
every lane on pipelined multiply/add units — measured 10–15% end-to-end on
ChebCode at equal accuracy.

All `unsafe` blocks of the entire crate live in that one module, behind the
safe `F64x2` abstraction:

- every SIMD load reads lanes `i..i+2` of a live `&[f64]`; callers guarantee
  the bound by construction (`j + 2 <= n` loop guards, `debug_assert`
  documents it);
- FP64 NEON is architectural on AArch64 and builds pin
  `-C target-cpu=native`, so the intrinsics' feature set is a compile-time
  constant — no runtime dispatch can be missed;
- non-AArch64 targets get an identical-semantics `[f64; 2]` fallback (true
  division), so kernels remain a single portable code path.

Everything outside `src/stieltjes/simd.rs` contains zero `unsafe`. (An even
earlier scalar-lane `fast_reciprocal` was removed: without 128-bit lanes it
measured 3× slower than hardware `fdiv`.)

## Development

The project is managed with **pixi** (conda-forge) for the Python environment and
**Cargo** for the Rust crate. The Python extension is built with **maturin** (PyO3).

### Toolchain

| Tool | Purpose | Managed by |
|------|---------|-----------|
| Rust (stable) | Core SIMD kernel | rustup |
| Cargo | Rust build/bench/test | rustup |
| pixi | Python env + tasks | pixi |
| Python 3.14 | Python bindings & scripts | pixi |
| maturin | PyO3 extension build | pixi |
| numpy / scipy / matplotlib | Python numerics & plotting | pixi |
| pytest / ruff / mypy | Python test, lint, type-check | pixi |

### Setup

```bash
# Create the pixi environment (installs Python, numpy, scipy, maturin, dev tools)
pixi install

# Build the Rust extension into the pixi env (release)
pixi run build
```

### Common tasks (via pixi)

```bash
pixi run build          # maturin develop --release
pixi run build-debug    # maturin develop (debug)
pixi run test           # cargo test
pixi run test-py        # pytest tests (Python API tests; run `pixi run build` first)
pixi run lint           # cargo clippy --all-targets -- -D warnings
pixi run lint-py        # ruff check scripts tests
pixi run fmt            # cargo fmt
pixi run fmt-check      # cargo fmt --check
pixi run type-py        # mypy scripts tests
pixi run bench-stieltjes  # python scripts/bench_stieltjes.py
pixi run measure        # python scripts/measure_current.py
```

### Rust benchmarks

```bash
cargo bench --bench comparison
cargo bench --bench peropt
cargo bench --bench cache_tiling
cargo bench --bench large_p_approx
```

## License

MIT