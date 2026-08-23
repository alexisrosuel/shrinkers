# Changelog

All notable changes to **shrinkers** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
