# shrinkers

[![CI](https://github.com/alexisrosuel/shrinkers/actions/workflows/ci.yml/badge.svg)](https://github.com/alexisrosuel/shrinkers/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/alexisrosuel/shrinkers/blob/main/LICENSE)

**Fast RMT (Random Matrix Theory) Population Eigenvalue Estimation Kernel**

A fast Rust implementation of **spiked + bulk eigenvalue cleaning** via
free-probability deconvolution: detect spikes (BEMA — Bulk Eigenvalue
Matching Analysis), debias them with the inverse BBP map (the
Baik–Ben Arous–Péché phase transition), and deconvolve the bulk (El
Karoui). Features O(p log p)
ChebCode treecodes (Chebyshev interpolation) for the
Stieltjes transform, auto-vectorized loops,
cache blocking, multi-threaded kernels (**~6× on an 8-core machine**, one keyword away),
and PyO3 bindings.

## Install

```bash
uv pip install shrinkers        # or: uv add shrinkers
# pixi users:
pixi add --pypi shrinkers       # in a pixi.toml: [pypi-dependencies]
# plain pip works too
pip install shrinkers
```

Wheels are published for Linux (x86_64, aarch64) and macOS (arm64,
x86_64); Windows and other platforms build from the sdist with a Rust
toolchain. To hack on the crate itself, see [Development](#development).

## Why shrinkers?

**1 · It actually un-distorts the spectrum.** Sample eigenvalues of a
high-dimensional covariance are biased artifacts: the bulk is smeared over
`[(1−√c)², (1+√c)²]·σ²`, genuine spikes are compressed toward the edge.
`shrinkers` inverts that distortion and recovers the population spectrum:

![Cleaning quality](docs/img/cleaning_quality.png)

*Spiked model, p = 1000, c = 0.25, σ² = 1, three spikes injected at 12 / 7 / 4.
All 3 spikes are detected and debiased to within 1 % (12.01 / 7.07 / 4.07),
the noise level is estimated at σ̂² = 1.002, and the median relative error
against the true population eigenvalues drops from **40 % to 4.3 %** in a
single call (`estimate_population_eigenvalues`).*

**2 · It is absurdly fast for what it computes.** Same math as your NumPy
one-liner, orders of magnitude faster — compared below against the two
baselines people actually write: a textbook pure-Python double loop, and a
vectorized NumPy version:

![Performance](docs/img/performance.png)

*Full transform of all p points, log-log over the whole 10⁰–10⁵ range,
η = 1/√p, Apple M1 Max, NumPy 2.5.1.*

| p | Naive Python | NumPy | shrinkers exact ¹ | shrinkers ChebCode ² |
|---|---|---|---|---|
| 1 024 | 0.32 s | 2.3 ms | **0.33 ms** · 7× | **0.22 ms** · 11× |
| 4 096 | 5.2 s | 94 ms | **1.2 ms** · 81× | **0.48 ms** · 195× |
| 50 000 | *(~2 h extrapolated)* | 14.6 s | **0.17 s** · 86× | **3.6 ms** · ≈4000× |

¹ machine precision — the zero-error anchor.  ² `chebcode_fast` preset,
rel. error ~1e-8: giving up four digits of accuracy buys two more orders of
magnitude in speed, and the higher-precision `chebcode` preset (~5e-10)
costs nearly the same (see [docs/internals.md](docs/internals.md)).
At microsecond scales both
shrinkers curves flatten onto their fixed Python-call overhead — a Rayon
request on tiny inputs runs sequential automatically rather than paying the
thread-pool floor (details in [docs/internals.md](docs/internals.md)).
Every number is reproducible: `scripts/make_readme_figures.py`
regenerates both figures end-to-end, and `docs/img/readme_figures.json`
holds the raw measurements.

**3 · It uses every core you give it.** The exact kernel is data-parallel
across cache blocks — flip one argument (`parallel=True`) and the
same call spreads over your cores with no reduction step and no false
sharing. Measured on the 8-core machine above (fresh sweep,
`docs/pareto/bench_after.json`):

| p | exact, 1 thread | exact, all cores | gain |
|---|---|---|---|
| 10 000 | 38.6 ms | 6.4 ms | **×6.0** |
| 50 000 | 0.94 s | 0.16 s | **×5.9** |

ChebCodeFast scales too (×3.3–4.5). Note also that the NumPy baseline in figure 2
is itself single-core — even pinned to one thread, shrinkers still wins
by roughly an order of magnitude (9.1× at p≈5000).

### What's inside

- `deconvolve_spiked(evals, c)` — the one-call pipeline: BEMA detection →
  inverse-BBP spike debiasing → El Karoui bulk deconvolution;
- Stieltjes-transform methods spanning the whole speed/accuracy frontier —
  machine-precision exact kernels, ChebCode treecodes (~1e-8 … ~6e-13),
  HODLR — plus data-driven `speed_auto` / `accuracy_auto` picks;
- correlation-matrix cleaning with eigenvector-overlap correction, direct
  precision-matrix shrinkage, Tracy–Widom spike detection;
- Rust API + PyO3 bindings with the GIL released during computation;
- multi-core execution built in — exact and ChebCode kernels parallelize
  multi-core (`parallel=True`).

## Quickstart

The **core** of the crate is a single call that cleans the sample eigenvalues
under a spiked covariance model, recovering the population spectrum:

```python
import numpy as np
from shrinkers import deconvolve_spiked

# Sample spectrum of a spiked covariance: Marchenko-Pastur bulk
# (sigma^2 = 1, c = 0.25) plus three spikes at 12, 7 and 4, observed
# through n = p/c Gaussian samples.
p, c = 1000, 0.25
pop = np.concatenate([[12.0, 7.0, 4.0], np.ones(p - 3)])
y = np.random.standard_normal((p, round(p / c))) * np.sqrt(pop)[:, None]
evals = np.linalg.eigvalsh(y @ y.T / y.shape[1])     # ascending

res = deconvolve_spiked(evals, c=c)
print(res["k"])        # -> 3 spikes detected
print(res["spikes"])   # -> close to [12, 7, 4] (BBP-debiased)
print(res["bulk"]["lambda_grid"][:4],   # grid + deconvolved bulk density
      res["bulk"]["density"][:4])       # profile on a 200-point grid
```

## How it works (30-second version)

One call, three steps:

1. **Detect** — sample eigenvalues that escape the bulk edge are flagged as
   spikes (real signal, e.g. factors).
2. **Debias** — each spike is mapped back to its true population value,
   undoing the upward push that high-dimensional noise inflicts on large
   eigenvalues.
3. **Deconvolve** — spikes removed, the remaining bulk is inverted through
   the Marčenko–Pastur equation to recover the population density.

This is useful for factor-model selection, noise validation, signal
extraction, and covariance estimation when the noise is not i.i.d. white.
The math (Stieltjes transforms, BBP inverse, El Karoui inversion) lives in
[docs/internals.md](docs/internals.md).

## Python usage

```python
from shrinkers import deconvolve_spiked
import numpy as np

evals = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0], dtype=np.float64)

# Clean the sample eigenvalues (spiked + bulk deconvolution)
res = deconvolve_spiked(evals, c=0.3)
print(res["k"])                 # -> 1 detected spike
print(res["spikes"])            # -> [8.779] debiased population spike
                                #    (sample value was 10; BBP pulls it down)
print(res["bulk"]["density"].shape)     # -> (200,) deconvolved bulk density
```

See `docs/python_api.md` for the full API reference.

## Comparison with existing packages

The only other Python package implementing RIE shrinkage is **pyRMT** (PyPI). Our comparison (a one-off benchmark script, since removed from `scripts/`; the reference NumPy implementation lives in `scripts/rie_numpy.py`) reveals:

### 🐛 pyRMT has a critical bug

`pyRMT.stieltjes(z, E)` computes $\mathrm{tr}(zI - E)$ instead of the correct $\mathrm{tr}\bigl((zI - E)^{-1}\bigr)$ — the matrix inverse is missing. This makes its RIE shrinkage completely wrong (~360% relative error).

| Method | p=100, c=0.5 | p=500, c=0.5 |
|--------|-------------|-------------|
| `rie_numpy` (ref) | ✅ ground truth | ✅ ground truth |
| `shrinkers` (Rust autovec) | **1.2e-14** max diff | **4.4e-14** max diff |
| `shrinkers` (Rust, approximate kernel) | **9.5e-03** (0.15% error) | **3.0e-02** (0.16% error) |
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
- Below the measured small-p crossover (~p≤500 single-core), the exact O(p²)
  kernels are the fastest pick; ChebCode takes over beyond
  ([docs/internals.md](docs/internals.md))

## Documentation

- [`docs/python_api.md`](docs/python_api.md) — full Python API reference;
- [`docs/internals.md`](docs/internals.md) — method map, algorithm math, the
  complete benchmark record and the unsafe-code policy;
- [`CHANGELOG.md`](CHANGELOG.md) — release history and known caveats.

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
cargo bench --bench pipeline_methods_overview  # every method end-to-end (p=500/1000)
cargo bench --bench pipeline_config_sweep      # knob-by-knob comparison at p=1000
cargo bench --bench pipeline_cache_scaling     # blocked vs tiled as output outgrows cache
cargo bench --bench kernel_tiled_blocksize     # raw tiled kernel + block-size landscape
```

## License

MIT