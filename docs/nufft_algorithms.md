# The coarse-grid NUFFT (`Nufft`): algorithm reference

`StieltjesMethod::Nufft` (Python `"nufft"` / `"fft_fast"`) is the large-p
member of the FFT family. It computes the same transform as `fft5` but sizes
the convolution grid from the **kernel width η**, not from the number of
eigenvalues. That keeps the grid at 32k–65k cells from p ≈ 20 000 to p = 10⁶,
so the cost is O(p + M log M) with M independent of p.

This document records the derivation, the implementation in
`src/stieltjes/nufft.rs` (policy) and `src/stieltjes/fft5.rs` (convolution
kernel), the measured accuracy/speed, and the tuning decisions. It mirrors
[`chebcode_algorithms.md`](chebcode_algorithms.md) for the treecode family.

## Problem

All evaluation paths reduce to

    S(x_k) = Σ_j 1 / ((x_k − λ_j) − iη),      k = 1..Q

with `p` sources λ_j, `Q` query points, and η > 0 the imaginary
regularization. Writing `d = x_k − λ_j`:

    Re S(x) = Σ_j d/(d² + η²),      Im S(x) = Σ_j η/(d² + η²).

Both are convolutions of the empirical measure `f = Σ_j δ_{λ_j}` with the
odd kernel `R_η(x) = x/(x²+η²)` and the even kernel `K_η(x) = η/(x²+η²)`.
For all-points evaluation `x_k = λ_k`, for the deconvolution grid `x_k` is a
uniform grid.

## Why the fft5 grid was wasteful

`fft5` runs the convolution on a uniform grid with spacing `dx` and size `m`.
Its adaptive rule was

    dx ≤ η/8      (resolve the Lorentzian)
    m  ≥ 8p       (resolve every eigenvalue individually)

The first bound is about the **kernel**; the second is about the **density**.
The density is a sum of Dirac masses, but the splat stencil already smooths it:
an N-point Lagrange splat produces a band-limited grid density whose spectrum
is the source spectrum times the stencil's transfer function. There is no need
to put each eigenvalue in its own grid cell. With the second bound, p = 50 000
forced `m = 524 288` and a ~12 ms call even though the kernel only needed
`m ≈ 32 768`.

The NUFFT drops `m ≥ 8p` and keeps the kernel-resolution rule with a safety
factor:

    pad   = max(2000·η, 0.75·raw_range)
    range = raw_range + 2·pad
    m     = next_pow2(ceil(5·range/η)),  clamped to [8192, 65 536].

The `5` (rather than `8`) was measured: it stays within ~1e-5 of the finer
grid (5.1e-5 vs 4.3e-5 at p = 10⁶) while choosing `m = 32 768` instead of
`65 536`, saving ~30 % of the FFT work. The `2000·η` padding controls the periodization floor of
the odd (1/x) kernel; 0.75·range covers wide spectra at large η. The cap is
what makes the method scale: at very small η the unconstrained rule would ask
for hundreds of thousands of cells and lose to the treecode.

## The grid convolution

```text
λ_j ──splat──▶ density[i] ──┐
                            ├─ FFT ─▶ D[k] ─┐
R_η sampled ────────────────┘               ├─ D·K, D·R ─▶ packed IFFT ─▶ grid
K_η analytic ───────────────────────────────┘                              │
                                                                           ▼
                                                     S(x_k) ◀── interpolate
```

The implementation reuses the `fft5` dual-convolution kernel unchanged; only
the grid size and the parallel transfer differ.

### 1. Splat

For a source at position `pos = (λ − lo)/dx`, an N-point Lagrange stencil
deposits weights on the N nearest cells:

    density[i0 + off0 + k] += w_k(t),     t = pos − floor(pos).

`N = 4` (Cubic) is the NUFFT default; N = 6/8 are available through
`NufftOptions.order` but cost more transfer work and do not improve the error
once the grid reaches the padding floor.

### 2. Packed forward FFT

The density and the odd kernel are packed into one complex array

    packed[i] = density[i] + i·R_η(x_i)

and transformed with a single forward FFT. Conjugate symmetry recovers the two
real spectra:

    D[k] = (C[k] + conj(C[(m−k) mod m]))/2
    R[k] = −i·(C[k] − conj(C[(m−k) mod m]))/2

The even kernel does not need a transform: its periodic DFT is known
analytically,

    K_even[k] = (π/dx)·exp(−2π·η·|k|/(m·dx)),

built by a recurrence (`r = exp(−2πη/(m dx))`) rather than m transcendental
calls. The odd kernel keeps its exact sampled DFT because the 1/x tail's seam
would otherwise show up at small η.

### 3. Frequency-domain products and one inverse FFT

The two products are packed as `Im_hat + i·Re_hat`, so a single inverse FFT
returns both convolution results on the grid. `packed_out.re = Im[m_g]` and
`packed_out.im = Re[m_g]` before the `1/m` scaling.

### 4. Interpolation

For each query point `x`, an N-point Lagrange interpolation at
`pos = (x − lo)/dx` combines the four neighbouring grid cells (Cubic) into
`(Re, Im)`. The same stencil as the splat is used, so splat and interpolation
are adjoint and mass is preserved.

## Parallel transfer

The FFTs stay sequential (two transforms of ≤65 536 points, ~1 ms total), but
the O(p·N) splat and O(Q·N) interpolation are the p-dependent part:

* sources are split into one contiguous chunk per Rayon worker;
* each worker fills a private `m`-cell density buffer;
* buffers are reduced element-wise (no atomics, no locks);
* interpolation maps independent query points over Rayon.

The reduction order differs from the sequential path, so parallel results are
numerically equal but not bit-identical (`nufft_matches_exact_and_parallel`
checks `max_diff < 1e-9`). At p = 10⁶ the transfer drops from ~9 ms
(sequential) to ~1.5 ms; the whole call is ~2.3 ms in Rust.

## Complexity and measured results

Per call: O(p·N + M log M) with N = 4 and M ≤ 65 536; for all p eigenvalues
`Q = p`. The p-dependent part is linear, unlike the treecode's O(p log p) and
`fft5`'s p-sized grid.

Rust kernel (`examples/measure_nufft.rs`, M1 Max, η = 1/√p, median of 3–7):

| p | sequential | Rayon | rel-L2 |
|---|---|---|---|
| 50 000 | 1.8 ms | 1.5 ms | 2.1e-5 |
| 100 000 | 2.7 ms | 1.4 ms | 2.7e-5 |
| 200 000 | 2.4 ms | 1.2 ms | 3.4e-5 |
| 1 000 000 | 9.9 ms | 2.3 ms | 5.1e-5 |

Python-level `stieltjes_transform` (includes the binding and result
conversion), versus `chebcode_fast`:

| p | ChebCodeFast | `nufft` | speed-up | rel. error |
|---|---|---|---|---|
| 50 000 | 1.9 ms | 1.8 ms | 1.0× | 2.1e-5 |
| 100 000 | 3.5 ms | **2.0 ms** | 1.8× | 2.7e-5 |
| 200 000 | 7.4 ms | **2.9 ms** | 2.5× | 3.4e-5 |
| 1 000 000 | 34.6 ms | **8.5 ms** | 4.0× | 5.1e-5 |

At p = 50k the two methods are a tie; below that the fixed grid cost dominates
and `ChebCodeFast` wins. The measured crossover is ~40k, but the speed table
keeps `ChebCodeFast` through the measured 50k bin rather than splitting a
noisy tie, and selects `Nufft` above it. At the other end,
η = 0.1/√p forces the 65 536-cell cap and the error grows toward ~1e-4, while
the speed advantage remains (p = 200k: 4 ms vs 8 ms at the default η).

## Parameter sensitivity

All knobs are exposed on `NufftOptions`:

| knob | default | effect |
|---|---|---|
| `order` | `Cubic` | transfer stencil; Cubic reaches the floor at the default `m` and is cheapest |
| `resolution` | 5.0 | cells per η; 5 was enough at η = 1/√p, 8 is the fft5-style conservative choice |
| `pad_eta_mult` | 2000 | odd-kernel periodization floor; lowering it raises the floor roughly as pad^-1.55 |
| `pad_range_frac` | 0.75 | padding for wide spectra at large η |
| `max_grid` | 65 536 | speed/accuracy lever at small η; 32 768 gains ~30 % at p = 1M but raises the error to ~7e-4 at η = 0.1/√p |
| `min_grid` | 8192 | keeps tiny-p calls from degenerating |

Measured representative points at p = 50 000:

| grid | order | time | rel-L2 |
|---|---|---|---|
| m = 16 384 | Cubic | 0.7 ms | 9.1e-5 |
| m = 32 768 | Cubic | 1.0 ms | 2.1e-5 |
| m = 65 536 | Cubic | 1.6 ms | 2.1e-5 |
| m = 16 384 | Heptic | 1.3 ms | 2.3e-5 |
| m = 32 768 | Heptic | 1.5 ms | 2.1e-5 |

Cubic at 32 768 is the sweet spot: the error is already at the padding floor,
and Heptic pays ~50 % more transfer time for no gain.

## Relationship to the other methods

* **`fft5`** — same kernel, conservative grid (`m ≥ 8p`), sequential transfer.
  Kept for its historical grid semantics and as the reference implementation.
* **`ChebCodeFast`** — O(p log p) treecode; wins below p ≈ 50k all-points
  and below p ≈ 300k for a few hundred query points.
* **`Nufft`** — linear-in-p transfer with a fixed grid; the large-p speed
  default and the method `auto` selects above p = 50 000.
* On the deconvolution grid (`nq` small), `auto` redirects a NUFFT pick to
  `ChebCodeFast` below p ≈ 300k: paying for a whole grid to serve 200 points
  is wasteful while the tree build is cheap. Above 300k the NUFFT wins even
  there (nq = 200: 1.6 ms vs 2.9 ms at p = 1M), so the redirect is skipped.
  An explicitly requested `nufft` is never second-guessed.

## Implementation map

| file | role |
|---|---|
| `src/stieltjes/nufft.rs` | `NufftOptions`, grid-size rule, parallel entry points, auto-facing API |
| `src/stieltjes/fft5.rs` | shared dual-convolution kernel (`fft_convolution`), splat/interpolation, packed FFTs |
| `src/stieltjes/mod.rs` | dispatch of `StieltjesMethod::Nufft`, all-points and at-points |
| `src/config/pareto_autogen.rs` | speed bins: ChebCodeFast through 50k, Nufft above |
| `examples/measure_nufft.rs` | reproducible speed/error harness |

## Caveats

* **Sorted input.** The grid range is taken from the first and last
  eigenvalue; the caller must pass a sorted spectrum (same contract as `fft5`).
* **Query range.** Interpolation clamps at the grid edges. Query points are
  expected inside the padded spectrum range; wildly outside values are not
  meaningful.
* **Small η.** The 65 536-cell cap trades the last digit for speed; at
  η = 0.1/√p the error is ~1e-4. Below p ≈ 50k, `ChebCodeFast` is faster.
* **Sequential FFTs.** Only the O(p) transfer is parallel; the two FFTs are
  not. At p ≲ 200k the fixed FFT cost limits the speed-up over `ChebCodeFast`;
  it grows to ~4× at 10⁶.

## Reproduce

```bash
cargo run --release --example measure_nufft -- 50000 100000 200000 1000000
cargo run --release --example measure_nufft -- 200000     # one size
cargo run --release --example measure_nufft -- sweep 50000  # parameter table
pytest tests -q -k nufft
cargo test --release --lib nufft
```
