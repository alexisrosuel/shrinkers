//! FFT-based O(p log p) Stieltjes transform via two separate convolutions.
//!
//!   Im[m_g] = (f_emp * K_η)(λ)   where K_η(x) = η/(x²+η²) (even Cauchy kernel)
//!   Re[m_g] = (f_emp * R_η)(λ)   where R_η(x) =  x/(x²+η²) (odd Cauchy kernel)
//!
//! Both convolutions are done via FFT.  With adequate padding, the density is
//! zero at the domain boundaries, so the odd kernel's periodic boundary
//! discontinuity doesn't matter (zero × kernel = zero).
//!
//! # Performance
//!
//! This is the **`fft5`** implementation (the full dual-convolution reference).
//! It uses **2 FFTs** (1 forward + 1 inverse):
//!
//! - **Forward**: the density and the odd (Hilbert) kernel are packed into a
//!   single complex array (`density + i·R_odd`) and transformed with one FFT,
//!   then unpacked via conjugate symmetry. The even (Lorentzian) kernel
//!   spectrum is computed **analytically** in O(m) — its DFT equals the
//!   periodic Poisson-kernel spectrum to exponential accuracy, so no kernel
//!   FFT is needed for it.
//! - **Inverse**: the two frequency-domain products (`Im_hat`, `Re_hat`) are
//!   packed into one complex array (`Im_hat + i·Re_hat`) and recovered from a
//!   single inverse FFT.
//!
//! The padding is `max(1000·η, 0.75·raw_range)` — the `0.75·raw_range` term
//! keeps the odd kernel's long-range 1/x tail from wrapping around, while
//! avoiding the oversized grids that a `2·raw_range` term would force.
//!
//! # Grid-transfer order
//!
//! Splatting and interpolation default to the 8-point heptic stencil
//! ([`Order::Heptic`]); all narrower stencils remain available through
//! [`compute_all_stieltjes_fft5_with_order`] / [`Fft5Options`]. Measured on
//! MP-like spectra (`examples/order_sweep.rs`, error vs exact O(p²)):
//!
//! ```text
//!   error vs grid size m (p=5000):          floor (order-independent):
//!     linear   m=65536 → 3.9e-5              p=1000 : 1.6e-5
//!     cubic    m=16384 → 3.3e-5              p=5000 : 3.0e-5
//!     quintic  m=16384 → 2.9e-5              p=20000: 4.2e-5
//!     heptic   m=16384 → 3.0e-5
//! ```
//!
//! Findings that motivated the default:
//!
//! * **The wrap-around (periodization) floor is order-independent** — no
//!   stencil beats it. At the *default* adaptive grid every order ≥ cubic
//!   sits on it; the choice of default is therefore about robustness at
//!   coarser-than-default grids.
//! * **Wider stencils never lose**: at any fixed grid the error ordering is
//!   heptic ≤ quintic ≤ cubic ≤ linear (up to ~2× better for heptic vs
//!   cubic at coarse grids), and the floor is reached one grid-halving
//!   (~40 % of runtime) earlier than with cubic. The extra O(p·N)
//!   multiply-adds are invisible next to the O(m log m) FFT.
//! * **The floor itself is tunable by padding**: err_floor ∝ pad^-1.55
//!   (measured α = 1.52–1.57). With `pad_eta_mult = 8000` and m=262144 the
//!   p=5000 error drops to 6.7e-7 — but only if the transfer keeps up:
//!   growing the padding at fixed m grows dx, and the linear stencil's
//!   O(dx²) error then *rises* back above the old floor while heptic stays
//!   flat.
//! * **Economics**: below ~3e-5 ChebCode dominates anyway (it reaches
//!   ~1e-9..1e-10 *faster* than big-pad fft5 reaches 1e-6, e.g. 3.0 ms vs
//!   4.7 ms sequential at p=5000). The padding lever is therefore exposed
//!   via [`Fft5Options`] rather than wired into any preset.

use num_complex::Complex64;
use rustfft::FftDirection;

/// Result of the FFT grid convolution: the convolved grid plus the mapping
/// parameters needed to interpolate the Stieltjes transform at arbitrary
/// query points.
struct FftGrid {
    /// Convolved grid values. `packed_out.re = Im[m_g]`, `packed_out.im = Re[m_g]`
    /// (before the `1/m` scaling), one per grid cell.
    packed_out: Vec<Complex64>,
    /// Grid origin (left edge of the padded domain).
    lo: f64,
    /// Grid spacing.
    dx: f64,
    /// Grid size.
    m: usize,
}

/// Grid transfer order used for splatting the density and interpolating
/// the result.
///
/// All stencils are Lagrange interpolants on `N` uniform nodes around the
/// target cell; the transfer error is O(dx^N) for a smooth density.
///
/// * [`Order::Linear`] — the historical 2-point stencil, error O(dx²).
/// * [`Order::Cubic`] — 4-point stencil, error O(dx⁴).
/// * [`Order::Quintic`] — 6-point stencil, error O(dx⁶).
/// * [`Order::Heptic`] — 8-point stencil, error O(dx⁸); the default.
///
/// Higher orders cost only O(p) extra multiply-adds in the splat/interp
/// passes (the FFT dominates), so at a *fixed grid* they are nearly free.
/// Their usefulness is bounded by the periodization (wrap-around) floor:
/// once the transfer error drops below it, extra order buys nothing — see
/// the module docs and `examples/order_sweep.rs` for the measured picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    Linear,
    Cubic,
    Quintic,
    Heptic,
}

impl Order {
    /// Number of stencil nodes (even).
    #[inline]
    pub fn nodes(self) -> usize {
        match self {
            Order::Linear => 2,
            Order::Cubic => 4,
            Order::Quintic => 6,
            Order::Heptic => 8,
        }
    }
}

/// Lagrange basis weights on the `N` uniform nodes
/// `{i₀-(N/2-1), …, i₀, …, i₀+N/2}` for fractional offset `t ∈ [0, 1)`
/// from node `i₀`. Partition of unity; exact for polynomials up to
/// degree `N-1`. Computed in O(N) via prefix/suffix products (the
/// denominator is the O(N²) node product, `N ≤ 8`, negligible).
#[inline]
fn lagrange_weights<const N: usize>(t: f64) -> [f64; N] {
    debug_assert!(N >= 2 && N % 2 == 0);
    let q = (N / 2) as f64;
    // Node l sits at cell offset l - (q-1); numerator factors (t - x_l).
    let mut pref = [1.0; N];
    let mut acc = 1.0;
    for (j, slot) in pref.iter_mut().enumerate() {
        *slot = acc;
        acc *= t - (j as f64) + q - 1.0;
    }
    let mut suf = [1.0; N];
    acc = 1.0;
    for (j, slot) in suf.iter_mut().enumerate().rev() {
        *slot = acc;
        acc *= t - (j as f64) + q - 1.0;
    }
    let mut w = [0.0; N];
    for (j, (w_slot, &pf)) in w.iter_mut().zip(pref.iter()).enumerate() {
        let mut den = 1.0;
        // Signed arithmetic: (j - l) underflows usize for l > j.
        for l in 0..N {
            if l != j {
                den *= (j as isize - l as isize) as f64;
            }
        }
        *w_slot = pf * suf[j] / den;
    }
    w
}

#[inline]
fn stencil_offset0<const N: usize>() -> isize {
    -(N as isize) / 2 + 1
}

/// Adjoint splat of one unit mass at grid position `pos` onto the `N`-point
/// stencil (same weights as interpolation, so splat∘interp is consistent
/// and mass is preserved: the weights sum to 1).
#[inline]
fn splat<const N: usize>(density: &mut [f64], m: usize, pos: f64) {
    let i0 = pos.floor() as isize;
    let t = pos - i0 as f64;
    let w = lagrange_weights::<N>(t);
    let off0 = stencil_offset0::<N>();
    let m_isize = m as isize;
    for (j, &wk) in w.iter().enumerate() {
        let cell = (i0 + off0 + j as isize).clamp(0, m_isize - 1) as usize;
        density[cell] += wk;
    }
}

/// `N`-point stencil interpolation of the packed grid at position `pos`.
/// Returns `(Re, Im)` scaled by `inv_m` (packed_out.re = Im[m_g],
/// packed_out.im = Re[m_g] before scaling).
#[inline]
fn interp_at<const N: usize>(
    packed_out: &[Complex64],
    m: usize,
    pos: f64,
    inv_m: f64,
) -> (f64, f64) {
    let i0 = pos.floor() as isize;
    let t = pos - i0 as f64;
    let w = lagrange_weights::<N>(t);
    let off0 = stencil_offset0::<N>();
    let m_isize = m as isize;
    let mut re_acc = 0.0f64;
    let mut im_acc = 0.0f64;
    for (j, &wk) in w.iter().enumerate() {
        let cell = (i0 + off0 + j as isize).clamp(0, m_isize - 1) as usize;
        let g = packed_out[cell];
        im_acc += g.re * wk;
        re_acc += g.im * wk;
    }
    (re_acc * inv_m, im_acc * inv_m)
}

/// Knobs of the FFT grid convolution beyond the method choice.
///
/// Defaults reproduce the historical adaptive behaviour (cubic transfer,
/// padding `max(1000η, 0.75·raw_range)`, grid `dx ≤ η/8` rounded to a power
/// of two). The fields are exposed for accuracy experiments and for tuning
/// the accuracy/speed trade-off; see `examples/order_sweep.rs`.
#[derive(Debug, Clone, Copy)]
pub struct Fft5Options {
    /// Grid transfer stencil.
    pub order: Order,
    /// Force the grid size (≥ 2, powers of two are fastest). `None` keeps
    /// the adaptive rule. When `Some`, the `dx ≤ η/8` resolution bound is
    /// the caller's responsibility.
    pub m_override: Option<usize>,
    /// Kernel-tail padding as a multiple of η (default 1000). Larger values
    /// push the periodization (wrap-around) floor down at linear cost in
    /// grid size.
    pub pad_eta_mult: f64,
    /// Padding as a fraction of the raw eigenvalue range, covering the odd
    /// kernel's 1/x tail (default 0.75).
    pub pad_range_frac: f64,
}

impl Default for Fft5Options {
    fn default() -> Self {
        Fft5Options {
            // Measured default: the widest stencil never loses to narrower
            // ones (same periodization floor, reached one grid-halving
            // earlier), and costs O(p·N) against O(m log m) FFT work.
            order: Order::Heptic,
            m_override: None,
            pad_eta_mult: 1000.0,
            pad_range_frac: 0.75,
        }
    }
}

/// Run the FFT dual-convolution (even + odd Cauchy kernels) once, producing
/// the convolved grid. Both [`compute_all_stieltjes_fft5`] and
/// [`compute_stieltjes_fft_at_points`] share this core so the O(p log p)
/// convolution is never duplicated.
fn fft_convolution(eigenvalues: &[f64], eta: f64, opts: &Fft5Options) -> FftGrid {
    let p = eigenvalues.len();

    // Adaptive padding: generous enough that the density is zero at the
    // boundaries (so the odd kernel's periodic wrap-around is harmless), but
    // not so large that the grid explodes. The pad_range_frac·raw_range term
    // covers the odd kernel's 1/x tail; pad_eta_mult·η dominates at small p.
    //
    // Padding sensitivity (measured, p=1000, dx=η/8, error vs exact O(p²)):
    //   pad = 2.0·range → m=65536, re_err 0.07%, im_err 0.08%
    //   pad = 0.75·range → m=32768, re_err 0.07%, im_err 0.08%  (HALF the grid)
    //   pad = 0.5·range  → m=16384, re_err 0.19%, im_err 0.18%
    //   pad = 0.25·range → m=16384, re_err 10%,  im_err 0.11%  (real part breaks)
    // The real part (Hilbert 1/d tail) is what forces the padding; the
    // imaginary part (Lorentzian 1/d²) is robust to tiny padding.
    let lo_raw = eigenvalues[0];
    let hi_raw = eigenvalues[p - 1];
    let raw_range = hi_raw - lo_raw;
    let pad = (opts.pad_eta_mult * eta).max(opts.pad_range_frac * raw_range);
    let lo = lo_raw - pad;
    let hi = hi_raw + pad;
    let range = hi - lo;

    // Adaptive grid: dx ≤ η/8 so the kernel is well-resolved.
    let min_grid = (8.0 * range / eta).ceil() as usize;
    let min_grid = min_grid.max(8 * p).min(1024 * 1024);
    let m = opts
        .m_override
        .unwrap_or_else(|| next_pow2(min_grid))
        .max(2);
    let dx = range / (m as f64);
    let half = m / 2;

    // --- Density on grid (order-selected splatting) ---
    let mut density = vec![0.0; m];
    match opts.order {
        Order::Linear => {
            for &lam in eigenvalues {
                splat::<2>(&mut density, m, (lam - lo) / dx);
            }
        }
        Order::Cubic => {
            for &lam in eigenvalues {
                splat::<4>(&mut density, m, (lam - lo) / dx);
            }
        }
        Order::Quintic => {
            for &lam in eigenvalues {
                splat::<6>(&mut density, m, (lam - lo) / dx);
            }
        }
        Order::Heptic => {
            for &lam in eigenvalues {
                splat::<8>(&mut density, m, (lam - lo) / dx);
            }
        }
    }

    // --- Pack density + odd kernel, one forward FFT ---
    // The even (Lorentzian) kernel decays fast, so its DFT equals the periodic
    // Poisson-kernel spectrum to exponential accuracy and is computed
    // analytically below. The odd (Hilbert) kernel has a 1/x tail, so its
    // sampled DFT carries a seam discontinuity that the analytic no-seam
    // formula misses (matters for small η); we therefore keep it exact by
    // packing it with the density into a single complex array and recovering
    // both spectra from one FFT via conjugate symmetry.
    //
    //   packed[i] = density[i] + i·R_odd[i],  R_odd(x) = x/(x²+η²)
    //   C = FFT(packed)  ⇒  D[k] = (C[k]+conj(C[(m-k)%m]))/2
    //                       R[k] = (C[k]-conj(C[(m-k)%m]))/(2i)
    let m_f64 = m as f64;
    let mut packed = vec![Complex64::new(0.0, 0.0); m];
    for (i, slot) in packed.iter_mut().enumerate() {
        let signed_dist = if i <= half {
            i as f64
        } else {
            i as f64 - m_f64
        };
        let x = signed_dist * dx;
        let denom = x * x + eta * eta;
        *slot = Complex64::new(density[i], x / denom);
    }
    super::fftplan::fft_inplace(&mut packed, FftDirection::Forward);
    // Density spectrum + odd-kernel spectrum from the single packed FFT.
    let (dens_freq, ko) = super::fftplan::unpack_packed_real_pair(&packed);

    // --- Analytic even-kernel spectrum (no kernel FFT) ---
    // The DFT of the sampled Lorentzian equals the periodic Poisson-kernel
    // spectrum to exponential accuracy (the kernel decays to zero at the
    // padded boundaries):
    //
    //   K_even[k] = (π/dx) · exp(-2π·η·|k|/(m·dx)),  |k| = min(k, m-k).
    //
    // The decaying factor r^|k| is built by recurrence (r = exp(-2πη/(m·dx)))
    // to avoid m transcendental calls; the accumulated rounding (~1e-10) is
    // far below the ~1e-4 target accuracy.
    let r = (-2.0 * std::f64::consts::PI * eta / (m as f64 * dx)).exp();
    let ke0 = std::f64::consts::PI / dx;
    let mut ke = vec![Complex64::new(0.0, 0.0); m];
    let mut pow = 1.0;
    for ke_k in ke.iter_mut().take(half + 1) {
        *ke_k = Complex64::new(ke0 * pow, 0.0);
        pow *= r;
    }
    pow = r;
    for k in (half + 1..m).rev() {
        ke[k] = Complex64::new(ke0 * pow, 0.0);
        pow *= r;
    }

    // --- Frequency-domain products, packed for one IFFT ---
    // packed_out = im_hat + i·re_hat, where im_hat = D·K_even (→ Im[m_g]) and
    // re_hat = D·R_odd (→ Re[m_g]). One inverse FFT recovers both.
    let mut packed_out = super::fftplan::pack_dual_product(&dens_freq, &ke, &ko);

    // --- Single inverse FFT ---
    super::fftplan::fft_inplace(&mut packed_out, FftDirection::Inverse);

    FftGrid {
        packed_out,
        lo,
        dx,
        m,
    }
}

/// Interpolate the convolved grid at a set of query points.
///
/// Returns `(real, imag)` pairs (the Stieltjes transform, **not** scaled by
/// `1/p`), one per query point. `packed_out.re = Im[m_g]`,
/// `packed_out.im = Re[m_g]` before the `1/m` scaling.
fn interpolate_grid(grid: &FftGrid, query_points: &[f64], order: Order) -> Vec<(f64, f64)> {
    let inv_m = 1.0 / (grid.m as f64);
    let mut result = Vec::with_capacity(query_points.len());
    macro_rules! interp_loop {
        ($n:literal) => {{
            for &q in query_points {
                let pos = (q - grid.lo) / grid.dx;
                let (r, i) = interp_at::<$n>(&grid.packed_out, grid.m, pos, inv_m);
                result.push((r, i));
            }
        }};
    }
    match order {
        Order::Linear => interp_loop!(2),
        Order::Cubic => interp_loop!(4),
        Order::Quintic => interp_loop!(6),
        Order::Heptic => interp_loop!(8),
    }
    result
}

/// Reference implementation of the OLD 3-FFT path: build the packed kernel
/// (even+odd), FFT it, and unpack the two spectra via conjugate symmetry.
/// Kept for the equivalence test. Not part of the public API.
#[cfg(test)]
#[doc(hidden)]
fn fft_convolution_kernel_fft(
    eigenvalues: &[f64],
    eta: f64,
    grid_size_opt: Option<usize>,
) -> FftGrid {
    let p = eigenvalues.len();
    let lo_raw = eigenvalues[0];
    let hi_raw = eigenvalues[p - 1];
    let raw_range = hi_raw - lo_raw;
    let pad = (1000.0 * eta).max(0.75 * raw_range);
    let lo = lo_raw - pad;
    let hi = hi_raw + pad;
    let range = hi - lo;
    let min_grid = (8.0 * range / eta).ceil() as usize;
    let min_grid = min_grid.max(8 * p).min(1024 * 1024);
    let m = grid_size_opt.unwrap_or_else(|| next_pow2(min_grid));
    let dx = range / (m as f64);
    let half = m / 2;
    let m_f64 = m as f64;

    let mut density = vec![0.0; m];
    for &lam in eigenvalues {
        let pos = (lam - lo) / dx;
        let idx = pos as usize;
        let frac = pos - (idx as f64);
        if idx >= m - 1 {
            density[m - 1] += 1.0;
        } else {
            density[idx] += 1.0 - frac;
            density[idx + 1] += frac;
        }
    }

    let mut packed_kernel = vec![Complex64::new(0.0, 0.0); m];
    for (i, slot) in packed_kernel.iter_mut().enumerate() {
        let signed_dist = if i <= half {
            i as f64
        } else {
            i as f64 - m_f64
        };
        let x = signed_dist * dx;
        let denom = x * x + eta * eta;
        *slot = Complex64::new(eta / denom, x / denom);
    }

    let mut dens_freq = density
        .iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect::<Vec<_>>();
    super::fftplan::fft_inplace(&mut dens_freq, FftDirection::Forward);
    super::fftplan::fft_inplace(&mut packed_kernel, FftDirection::Forward);

    let mut packed_out = vec![Complex64::new(0.0, 0.0); m];
    for k in 0..m {
        let ck = packed_kernel[k];
        let cnk = packed_kernel[(m - k) % m];
        let ke = 0.5 * (ck + cnk.conj());
        let ko = -0.5 * Complex64::new(0.0, 1.0) * (ck - cnk.conj());
        let d = dens_freq[k];
        let im_hat = d * ke;
        let re_hat = d * ko;
        packed_out[k] = Complex64::new(im_hat.re - re_hat.im, im_hat.im + re_hat.re);
    }
    super::fftplan::fft_inplace(&mut packed_out, FftDirection::Inverse);

    FftGrid {
        packed_out,
        lo,
        dx,
        m,
    }
}

/// Compute all Stieltjes transforms via two FFT convolutions (even+odd
/// kernel), using the default (heptic, 8-point) grid transfer.
///
/// `grid_size_opt` overrides the adaptive rule `dx ≤ η/8` when `Some`.
pub fn compute_all_stieltjes_fft5(
    eigenvalues: &[f64],
    eta: f64,
    grid_size_opt: Option<usize>,
) -> Vec<(f64, f64)> {
    let opts = Fft5Options {
        m_override: grid_size_opt,
        ..Fft5Options::default()
    };
    compute_all_stieltjes_fft5_with_options(eigenvalues, eta, &opts)
}

/// [`Order::Linear`] variant of [`compute_all_stieltjes_fft5`]: the historical
/// 2-point stencil. Kept for A/B comparisons and as a conservative fallback.
pub fn compute_all_stieltjes_fft5_linear(
    eigenvalues: &[f64],
    eta: f64,
    grid_size_opt: Option<usize>,
) -> Vec<(f64, f64)> {
    compute_all_stieltjes_fft5_with_order(eigenvalues, eta, grid_size_opt, Order::Linear)
}

/// Shared implementation for all transfer orders.
pub fn compute_all_stieltjes_fft5_with_order(
    eigenvalues: &[f64],
    eta: f64,
    grid_size_opt: Option<usize>,
    order: Order,
) -> Vec<(f64, f64)> {
    let opts = Fft5Options {
        order,
        m_override: grid_size_opt,
        ..Fft5Options::default()
    };
    compute_all_stieltjes_fft5_with_options(eigenvalues, eta, &opts)
}

/// Fully configurable entry point: transfer order, forced grid size and
/// padding multipliers. See [`Fft5Options`] for the knobs and
/// `examples/order_sweep.rs` for the measured accuracy/speed landscape.
pub fn compute_all_stieltjes_fft5_with_options(
    eigenvalues: &[f64],
    eta: f64,
    opts: &Fft5Options,
) -> Vec<(f64, f64)> {
    if eigenvalues.is_empty() {
        return Vec::new();
    }
    let grid = fft_convolution(eigenvalues, eta, opts);
    interpolate_grid(&grid, eigenvalues, opts.order)
}

/// Compute the Stieltjes transform at arbitrary query points via the FFT
/// dual-convolution.
///
/// This is the O(p log p) path for evaluating the sample Stieltjes transform
/// on a uniform grid (e.g. the deconvolution grid), where the query points
/// differ from the sample eigenvalues. It runs the convolution once and
/// interpolates at the query points.
///
/// Returns raw sums (not scaled by `1/p`), one `(real, imag)` per query point.
pub fn compute_stieltjes_fft_at_points(
    query_points: &[f64],
    eigenvalues: &[f64],
    eta: f64,
    grid_size_opt: Option<usize>,
) -> Vec<(f64, f64)> {
    if eigenvalues.is_empty() || query_points.is_empty() {
        return Vec::new();
    }
    let order;
    let grid = {
        let opts = Fft5Options {
            m_override: grid_size_opt,
            ..Fft5Options::default()
        };
        order = opts.order;
        fft_convolution(eigenvalues, eta, &opts)
    };
    interpolate_grid(&grid, query_points, order)
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_runs_and_returns_right_length() {
        let p = 512;
        let evals: Vec<f64> = (0..p).map(|i| ((i as f64 + 1.0) / 100.0).ln_1p()).collect();
        let eta = 0.05;

        let fft_results = compute_all_stieltjes_fft5(&evals, eta, None);
        assert_eq!(fft_results.len(), p);
        for (r, i) in &fft_results {
            assert!(r.is_finite());
            assert!(i.is_finite());
        }
    }

    #[test]
    fn test_fft_real_part_sign_and_accuracy() {
        // Verify the FFT odd-kernel real part against the exact method.
        // The real part Re[S] = Σ (λᵢ-λⱼ)/((λᵢ-λⱼ)²+η²) is long-range (1/d).
        // Empirically the FFT odd kernel produces -Re[S] (sign flip), so we
        // check both signs and report which matches.
        let p = 512;
        let evals = crate::stieltjes::testutil::log_spectrum(p);
        let eta = 0.1 / (p as f64).sqrt();

        let fft_results = compute_all_stieltjes_fft5(&evals, eta, None);

        // Exact real part (no cutoff)
        let mut exact_re = vec![0.0_f64; p];
        for i in 0..p {
            let li = evals[i];
            let mut s = 0.0;
            for &lj in &evals {
                let d = li - lj;
                s += d / (d * d + eta * eta);
            }
            exact_re[i] = s / (p as f64);
        }

        let mut err_pos = 0.0_f64;
        let mut err_neg = 0.0_f64;
        let inv_p = 1.0 / (p as f64);
        for i in 0..p {
            let fft_r = fft_results[i].0 * inv_p;
            let scale = exact_re[i].abs().max(1e-12);
            err_pos = err_pos.max((fft_r - exact_re[i]).abs() / scale);
            err_neg = err_neg.max((-fft_r - exact_re[i]).abs() / scale);
        }
        eprintln!("FFT real: err(+fft)={err_pos:.4} err(-fft)={err_neg:.4}");
        // The Rust FFT uses the correct sign (err(+fft) is small). The real
        // part is long-range so the FFT grid resolution limits accuracy.
        assert!(err_pos < 0.5, "FFT real part not accurate: {err_pos}");
    }

    #[test]
    fn test_analytic_kernel_matches_kernel_fft() {
        // The analytic-kernel path must reproduce the old kernel-FFT path to
        // within the ~1e-3 target accuracy. The odd kernel is periodized
        // without its seam, which differs from the FFT path only where the
        // density is zero (thanks to padding); at the interpolated eigenvalues
        // the difference is ~5e-4 relative (measured), well under 1e-3.
        for p in [257, 512, 1024, 2048] {
            for eta in [0.05, 0.2, 0.5] {
                let evals: Vec<f64> = (0..p).map(|i| ((i as f64 + 1.0) / 50.0).ln_1p()).collect();

                let new_res =
                    compute_all_stieltjes_fft5_with_order(&evals, eta, None, Order::Linear);
                let old_grid = fft_convolution_kernel_fft(&evals, eta, None);
                let old_res = interpolate_grid(&old_grid, &evals, Order::Linear);

                let mut max_re = 0.0_f64;
                let mut max_im = 0.0_f64;
                let mut scale = 0.0_f64;
                for i in 0..p {
                    max_re = max_re.max((new_res[i].0 - old_res[i].0).abs());
                    max_im = max_im.max((new_res[i].1 - old_res[i].1).abs());
                    scale = scale.max(old_res[i].0.abs()).max(old_res[i].1.abs());
                }
                let scale = scale.max(1e-12);
                let rel_re = max_re / scale;
                let rel_im = max_im / scale;
                assert!(
                    rel_re < 1e-3,
                    "p={p} eta={eta}: analytic vs kernel-FFT real rel err {rel_re:.2e}"
                );
                assert!(
                    rel_im < 1e-3,
                    "p={p} eta={eta}: analytic vs kernel-FFT imag rel err {rel_im:.2e}"
                );
            }
        }
    }

    /// Regression guard for the cubic grid-transfer upgrade: at the default
    /// adaptive grid, cubic must not be worse than linear against the exact
    /// O(p²) sum (measured: ~8–12× better on MP-like spectra).
    #[test]
    fn test_cubic_not_worse_than_linear() {
        for &p in &[512usize, 2000] {
            let c: f64 = 0.5;
            let lo = (1.0 - c.sqrt()).powi(2);
            let hi = (1.0 + c.sqrt()).powi(2);
            let mut evs: Vec<f64> = (0..p)
                .map(|i| {
                    let x = (i as f64 + 0.5) / p as f64;
                    lo + x * (hi - lo)
                })
                .collect();
            evs.push(hi * 1.9);
            evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let eta = 1.0 / (p as f64).sqrt();

            let mut exact_sq = 0.0f64;
            let mut refs = Vec::with_capacity(p);
            for &li in &evs {
                let mut sr = 0.0;
                let mut si = 0.0;
                for &lj in &evs {
                    let d = li - lj;
                    let den = d * d + eta * eta;
                    sr += d / den;
                    si += eta / den;
                }
                refs.push((sr, si));
                exact_sq += sr * sr + si * si;
            }

            let err_of = |res: Vec<(f64, f64)>| -> f64 {
                let num: f64 = res
                    .iter()
                    .zip(&refs)
                    .map(|(g, r)| (g.0 - r.0).powi(2) + (g.1 - r.1).powi(2))
                    .sum();
                (num / exact_sq).sqrt()
            };

            let lin = err_of(compute_all_stieltjes_fft5_linear(&evs, eta, None));
            let cub = err_of(compute_all_stieltjes_fft5(&evs, eta, None));
            assert!(
                cub <= lin,
                "p={p}: cubic ({cub:.3e}) should beat linear ({lin:.3e})"
            );
        }
    }

    /// Lagrange stencils must form a partition of unity (mass-preserving
    /// splatting) for every supported order.
    #[test]
    fn test_lagrange_partition_of_unity() {
        for order in [Order::Linear, Order::Cubic, Order::Quintic, Order::Heptic] {
            match order.nodes() {
                2 => check_pou::<2>(),
                4 => check_pou::<4>(),
                6 => check_pou::<6>(),
                8 => check_pou::<8>(),
                _ => unreachable!(),
            }
        }
    }

    fn check_pou<const N: usize>() {
        let steps = 97;
        for s in 0..steps {
            let t = (s as f64) / (steps as f64); // includes t=0 exactly
            let w = lagrange_weights::<N>(t);
            let sum: f64 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "N={N} t={t}: partition of unity violated, sum={sum}"
            );
            // Exactness on constants implies the interpolation reproduces
            // constants; also verify symmetry w_j(t) + w_{N-1-j}(1-t) is not
            // needed here, but weights must be finite and bounded.
            for &wj in &w {
                assert!(wj.is_finite() && wj.abs() < 10.0);
            }
        }
    }

    /// Higher-order transfers must slash the interpolation error on a smooth
    /// field, with the O(dx^N) signature of each stencil. Synthetic field,
    /// no convolution involved (the kernel-resolution floor of the full
    /// pipeline would mask the stencil scaling).
    #[test]
    fn test_interp_error_scales_with_stencil_order() {
        let g = |x: f64| (3.0 * x).sin() + 0.5 * (5.0 * x).cos();
        let m = 4096usize;
        let lo = -1.0f64;
        let dx = 0.01f64;
        // packed_out.re plays the Im[m_g] slot, .im the Re[m_g] slot; fill
        // both with g so either component checks the same interpolation.
        // The grid carries UNNORMALIZED sums — interpolate_grid applies the
        // final 1/m — so store g·m.
        let packed_out: Vec<Complex64> = (0..m)
            .map(|i| {
                let x = lo + i as f64 * dx;
                Complex64::new(g(x) * m as f64, g(x) * m as f64)
            })
            .collect();
        let grid = FftGrid {
            packed_out,
            lo,
            dx,
            m,
        };

        let err_of_order = |order: Order| -> f64 {
            // Interior query points at fractional offsets (avoid clamp edges).
            let queries: Vec<f64> = (0..500)
                .map(|s| {
                    let u = 10 + s;
                    lo + (u as f64 + 0.37) * dx
                })
                .collect();
            let res = interpolate_grid(&grid, &queries, order);
            res.iter()
                .zip(&queries)
                .map(|((r, im), &x)| ((r - g(x)).abs()).max((im - g(x)).abs()))
                .fold(0.0f64, f64::max)
        };

        let e_lin = err_of_order(Order::Linear);
        let e_cub = err_of_order(Order::Cubic);
        let e_qui = err_of_order(Order::Quintic);
        let e_hep = err_of_order(Order::Heptic);
        // Each +2 stencil order should gain ≥ ~50× on this field (theory:
        // ×dx⁻² ≈ 10⁴; measured constants make it smaller, keep margin).
        assert!(
            e_cub < e_lin / 50.0,
            "cubic {e_cub:.3e} should crush linear {e_lin:.3e}"
        );
        assert!(
            e_qui < e_cub / 50.0,
            "quintic {e_qui:.3e} should crush cubic {e_cub:.3e}"
        );
        assert!(
            e_hep < e_qui / 30.0,
            "heptic {e_hep:.3e} should crush quintic {e_qui:.3e}"
        );
    }

    /// Pipeline-level guard for the heptic default: on the full FFT
    /// convolution at a forced coarse grid in the transfer-dominated regime,
    /// the error must order as heptic ≤ cubic < linear against the exact
    /// O(p²) sum (measured at this operating point: 5.5e-5 / 3.0e-4 /
    /// 2.0e-3).
    #[test]
    fn test_heptic_default_not_worse_on_pipeline() {
        use crate::config::{CutoffConfig as Cc, Parallelism as Par, StieltjesMethod as Sm};
        let p = 5000;
        let c: f64 = 0.5;
        let lo = (1.0 - c.sqrt()).powi(2);
        let hi = (1.0 + c.sqrt()).powi(2);
        let mut evs: Vec<f64> = (0..p)
            .map(|i| lo + ((i as f64 + 0.5) / p as f64) * (hi - lo))
            .collect();
        evs.push(hi * 2.3);
        evs.push(lo * 0.35);
        evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 1.0 / (p as f64).sqrt();

        let refr = crate::stieltjes::compute_all_stieltjes(
            &evs,
            eta,
            Sm::BlockedTiled,
            None,
            Cc::Disabled,
            32,
            Par::Sequential,
        );

        let err_of_order = |order: Order| -> f64 {
            let opts = Fft5Options {
                order,
                m_override: Some(8192),
                ..Fft5Options::default()
            };
            let res = compute_all_stieltjes_fft5_with_options(&evs, eta, &opts);
            let inv_p = 1.0 / p as f64;
            let num: f64 = res
                .iter()
                .zip(refr.iter())
                .map(|(g, r)| (g.0 * inv_p - r.0).powi(2) + (g.1 * inv_p - r.1).powi(2))
                .sum();
            let den: f64 = refr.iter().map(|(r, i)| r * r + i * i).sum();
            (num / den).sqrt()
        };

        let e_lin = err_of_order(Order::Linear);
        let e_cub = err_of_order(Order::Cubic);
        let e_hep = err_of_order(Order::Heptic);
        assert!(
            e_cub < e_lin && e_hep <= e_cub,
            "ordering violated: lin={e_lin:.3e} cub={e_cub:.3e} hep={e_hep:.3e}"
        );
        // The default entry point must actually BE heptic.
        let dflt = err_of_order(Fft5Options::default().order);
        assert!((dflt - e_hep).abs() < 1e-15);
    }
}
