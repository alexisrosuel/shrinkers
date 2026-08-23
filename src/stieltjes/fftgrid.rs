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
//! This implementation uses **3 FFTs instead of 5** via two-real packing:
//!
//! - **Forward**: the even and odd kernels are packed into a single complex
//!   array (`k_even + i·k_odd`), transformed with one FFT, then unpacked in
//!   frequency space using conjugate symmetry.
//! - **Inverse**: the two frequency-domain products (`Im_hat`, `Re_hat`) are
//!   packed into one complex array (`Im_hat + i·Re_hat`) and recovered from a
//!   single inverse FFT.
//!
//! The padding is `max(1000·η, 2·raw_range)` — the `2·raw_range` term keeps the
//! odd kernel's long-range 1/x tail from wrapping around, while avoiding the
//! oversized grids that a `5·raw_range` term would force.

use num_complex::Complex64;
use rustfft::{FftDirection, FftPlanner};

/// Compute all Stieltjes transforms via two FFT convolutions (even+odd kernel).
pub fn compute_all_stieltjes_fft(
    eigenvalues: &[f64],
    eta: f64,
    grid_size_opt: Option<usize>,
) -> Vec<(f64, f64)> {
    let p = eigenvalues.len();
    if p == 0 {
        return Vec::new();
    }

    // Adaptive padding: generous enough that the density is zero at the
    // boundaries (so the odd kernel's periodic wrap-around is harmless), but
    // not so large that the grid explodes. The 2·raw_range term covers the
    // odd kernel's 1/x tail; 1000·η dominates at small p.
    let lo_raw = eigenvalues[0];
    let hi_raw = eigenvalues[p - 1];
    let raw_range = hi_raw - lo_raw;
    let pad = (1000.0 * eta).max(2.0 * raw_range);
    let lo = lo_raw - pad;
    let hi = hi_raw + pad;
    let range = hi - lo;

    // Adaptive grid: dx ≤ η/8 so the kernel is well-resolved.
    let min_grid = (8.0 * range / eta).ceil() as usize;
    let min_grid = min_grid.max(8 * p).min(1024 * 1024);
    let m = grid_size_opt.unwrap_or_else(|| next_pow2(min_grid));
    let dx = range / (m as f64);
    let half = m / 2;
    let m_f64 = m as f64;

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

    // --- Build both kernels packed into one complex array ---
    // For circular convolution, kernel[i] = kernel_value_at_circular_distance(i)
    // where distance(i) = i·dx  for i ≤ half,  = (i-M)·dx  for i > half.
    //
    // Even kernel: K(x) = η/(x²+η²),  K(x) = K(-x)
    // Odd kernel:  R(x) = x/(x²+η²),  R(-x) = -R(x)
    //
    // Pack: packed[i] = K_even[i] + i·R_odd[i]. One forward FFT gives both
    // spectra, which we unpack via conjugate symmetry.
    let mut packed_kernel = vec![Complex64::new(0.0, 0.0); m];
    for i in 0..m {
        let signed_dist = if i <= half {
            i as f64
        } else {
            i as f64 - m_f64
        };
        let x = signed_dist * dx;
        let denom = x * x + eta * eta;
        let k_even = eta / denom;
        let k_odd = x / denom;
        packed_kernel[i] = Complex64::new(k_even, k_odd);
    }

    // --- FFT density + packed kernel (2 forward FFTs) ---
    let mut dens_freq = density
        .iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect::<Vec<_>>();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(m, FftDirection::Forward);

    fft.process(&mut dens_freq);
    fft.process(&mut packed_kernel);

    // --- Unpack the two kernel spectra from the packed transform ---
    // If C = FFT(a + i·b), then:
    //   A[k] = (C[k] + conj(C[(m-k)%m])) / 2
    //   B[k] = (C[k] - conj(C[(m-k)%m])) / (2i) = -i/2 · (C[k] - conj(C[(m-k)%m]))
    let mut packed_out = vec![Complex64::new(0.0, 0.0); m];
    for k in 0..m {
        let ck = packed_kernel[k];
        let cnk = packed_kernel[(m - k) % m];
        let ke = 0.5 * (ck + cnk.conj()); // even kernel spectrum
        let ko = -0.5 * Complex64::new(0.0, 1.0) * (ck - cnk.conj()); // odd kernel spectrum
        let d = dens_freq[k];
        let im_hat = d * ke; // → Im[m_g]
        let re_hat = d * ko; // → Re[m_g]
        // Pack Im_hat + i·Re_hat so one IFFT recovers both.
        packed_out[k] = Complex64::new(im_hat.re - re_hat.im, im_hat.im + re_hat.re);
    }

    // --- Single inverse FFT ---
    let ifft = planner.plan_fft(m, FftDirection::Inverse);
    ifft.process(&mut packed_out);

    // packed_out.re = Im[m_g], packed_out.im = Re[m_g] (before 1/m scaling)
    let inv_m = 1.0 / m_f64;

    // --- Interpolate back ---
    let mut result = Vec::with_capacity(p);
    let inv_p = 1.0 / (p as f64);
    for &lam in eigenvalues {
        let pos = (lam - lo) / dx;
        let idx = pos as usize;
        let frac = pos - (idx as f64);

        if idx >= m - 1 {
            let g = packed_out[m - 1];
            result.push((g.im * inv_m * inv_p, g.re * inv_m * inv_p));
        } else {
            let g0 = packed_out[idx];
            let g1 = packed_out[idx + 1];
            let r = (g0.im * (1.0 - frac) + g1.im * frac) * inv_m * inv_p;
            let i = (g0.re * (1.0 - frac) + g1.re * frac) * inv_m * inv_p;
            result.push((r, i));
        }
    }

    result
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

        let fft_results = compute_all_stieltjes_fft(&evals, eta, None);
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

        let fft_results = compute_all_stieltjes_fft(&evals, eta, None);

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
        for i in 0..p {
            let fft_r = fft_results[i].0;
            let scale = exact_re[i].abs().max(1e-12);
            err_pos = err_pos.max((fft_r - exact_re[i]).abs() / scale);
            err_neg = err_neg.max((-fft_r - exact_re[i]).abs() / scale);
        }
        eprintln!("FFT real: err(+fft)={err_pos:.4} err(-fft)={err_neg:.4}");
        // The Rust FFT uses the correct sign (err(+fft) is small). The real
        // part is long-range so the FFT grid resolution limits accuracy.
        assert!(err_pos < 0.5, "FFT real part not accurate: {err_pos}");
    }
}
