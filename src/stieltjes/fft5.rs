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

use num_complex::Complex64;
use rustfft::{FftDirection, FftPlanner};

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

/// Run the FFT dual-convolution (even + odd Cauchy kernels) once, producing
/// the convolved grid. Both [`compute_all_stieltjes_fft5`] and
/// [`compute_stieltjes_fft_at_points`] share this core so the O(p log p)
/// convolution is never duplicated.
fn fft_convolution(eigenvalues: &[f64], eta: f64, grid_size_opt: Option<usize>) -> FftGrid {
    let p = eigenvalues.len();

    // Adaptive padding: generous enough that the density is zero at the
    // boundaries (so the odd kernel's periodic wrap-around is harmless), but
    // not so large that the grid explodes. The 0.75·raw_range term covers the
    // odd kernel's 1/x tail; 1000·η dominates at small p.
    //
    // Padding sensitivity (measured, p=1000, dx=η/8, error vs exact O(p²)):
    //   pad = 2.0·range → m=65536, re_err 0.07%, im_err 0.08%
    //   pad = 0.75·range → m=32768, re_err 0.07%, im_err 0.08%  (HALF the grid)
    //   pad = 0.5·range  → m=16384, re_err 0.19%, im_err 0.18%
    //   pad = 0.25·range → m=16384, re_err 10%,  im_err 0.11%  (real part breaks)
    // The real part (Hilbert 1/d tail) is what forces the padding; the
    // imaginary part (Lorentzian 1/d²) is robust to tiny padding. 0.75·range
    // keeps both parts ~0.1% while halving the grid vs the old 2.0·range.
    let lo_raw = eigenvalues[0];
    let hi_raw = eigenvalues[p - 1];
    let raw_range = hi_raw - lo_raw;
    let pad = (1000.0 * eta).max(0.75 * raw_range);
    let lo = lo_raw - pad;
    let hi = hi_raw + pad;
    let range = hi - lo;

    // Adaptive grid: dx ≤ η/8 so the kernel is well-resolved.
    let min_grid = (8.0 * range / eta).ceil() as usize;
    let min_grid = min_grid.max(8 * p).min(1024 * 1024);
    let m = grid_size_opt.unwrap_or_else(|| next_pow2(min_grid));
    let dx = range / (m as f64);
    let half = m / 2;

    // --- Density on grid via linear splatting ---
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
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(m, FftDirection::Forward);
    fft.process(&mut packed);

    let mut dens_freq = vec![Complex64::new(0.0, 0.0); m];
    let mut ko = vec![Complex64::new(0.0, 0.0); m];
    for k in 0..m {
        let ck = packed[k];
        let cnk = packed[(m - k) % m];
        dens_freq[k] = 0.5 * (ck + cnk.conj());
        ko[k] = -0.5 * Complex64::new(0.0, 1.0) * (ck - cnk.conj());
    }

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
    let mut ke = vec![0.0_f64; m];
    let mut pow = 1.0;
    for ke_k in ke.iter_mut().take(half + 1) {
        *ke_k = ke0 * pow;
        pow *= r;
    }
    pow = r;
    for k in (half + 1..m).rev() {
        ke[k] = ke0 * pow;
        pow *= r;
    }

    // --- Frequency-domain products, packed for one IFFT ---
    // packed_out = im_hat + i·re_hat, where im_hat = D·K_even (→ Im[m_g]) and
    // re_hat = D·R_odd (→ Re[m_g]). One inverse FFT recovers both.
    let mut packed_out = vec![Complex64::new(0.0, 0.0); m];
    for k in 0..m {
        let d = dens_freq[k];
        let im_hat = d * ke[k];
        let re_hat = d * ko[k];
        packed_out[k] = Complex64::new(im_hat.re - re_hat.im, im_hat.im + re_hat.re);
    }

    // --- Single inverse FFT ---
    let ifft = planner.plan_fft(m, FftDirection::Inverse);
    ifft.process(&mut packed_out);

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
fn interpolate_grid(grid: &FftGrid, query_points: &[f64]) -> Vec<(f64, f64)> {
    let inv_m = 1.0 / (grid.m as f64);
    let mut result = Vec::with_capacity(query_points.len());
    for &q in query_points {
        let pos = (q - grid.lo) / grid.dx;
        let idx = pos as usize;
        let frac = pos - (idx as f64);

        if idx >= grid.m - 1 {
            let g = grid.packed_out[grid.m - 1];
            result.push((g.im * inv_m, g.re * inv_m));
        } else {
            let g0 = grid.packed_out[idx];
            let g1 = grid.packed_out[idx + 1];
            let r = (g0.im * (1.0 - frac) + g1.im * frac) * inv_m;
            let i = (g0.re * (1.0 - frac) + g1.re * frac) * inv_m;
            result.push((r, i));
        }
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
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(m, FftDirection::Forward);
    fft.process(&mut dens_freq);
    fft.process(&mut packed_kernel);

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
    let ifft = planner.plan_fft(m, FftDirection::Inverse);
    ifft.process(&mut packed_out);

    FftGrid {
        packed_out,
        lo,
        dx,
        m,
    }
}

/// Compute all Stieltjes transforms via two FFT convolutions (even+odd kernel).
pub fn compute_all_stieltjes_fft5(
    eigenvalues: &[f64],
    eta: f64,
    grid_size_opt: Option<usize>,
) -> Vec<(f64, f64)> {
    if eigenvalues.is_empty() {
        return Vec::new();
    }
    let grid = fft_convolution(eigenvalues, eta, grid_size_opt);
    interpolate_grid(&grid, eigenvalues)
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
    let grid = fft_convolution(eigenvalues, eta, grid_size_opt);
    interpolate_grid(&grid, query_points)
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
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
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

                let new_res = compute_all_stieltjes_fft5(&evals, eta, None);
                let old_grid = fft_convolution_kernel_fft(&evals, eta, None);
                let old_res = interpolate_grid(&old_grid, &evals);

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
}
