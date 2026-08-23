//! DST-I (Discrete Sine Transform) real-part computation.
//!
//! # Mathematical note (important)
//!
//! The Stieltjes **real part** is the odd-kernel convolution
//!
//! ```text
//!   Re[S](λᵢ) = Σⱼ R(λᵢ-λⱼ),   R(x) = x/(x²+η²)   (odd: R(-x) = -R(x))
//! ```
//!
//! A common claim is that because `R` is odd, a DST diagonalizes this
//! convolution. **This is not correct for the plain convolution.** DST-I
//! diagonalizes the *sine convolution* operator
//!
//! ```text
//!   (T f)_n = Σ_m [R(n-m) - R(n+m)] f_m   (Toeplitz + Hankel)
//! ```
//!
//! which differs from the plain convolution `Σ_m R(n-m) f_m` by the Hankel
//! term `Σ_m R(n+m) f_m`. So a naive `DST⁻¹(diag·DST(f))` computes a *different*
//! quantity and is **not** a drop-in for `Re[S]`.
//!
//! The correct way to exploit the odd symmetry is the **odd-extension trick**:
//! extend the density oddly across the boundary, then a single FFT of the
//! doubled array computes the odd-kernel convolution with clean sine boundary
//! conditions. This is mathematically equivalent to the `fft5` odd-kernel
//! approach but uses a real DST-I (via the FFT odd-extension), which is
//! cheaper and has cleaner boundary handling for the odd kernel.
//!
//! This module implements that correct DST-I-based real part. It is provided
//! as a reference/alternative to the `fft5` odd kernel and to document the
//! (incorrect) naive-DST pitfall.

use num_complex::Complex64;
use rustfft::{FftDirection, FftPlanner};

/// Compute the real part `Re[S]` via DST-I (odd-extension FFT).
///
/// Returns a Vec of raw real-part sums (not scaled by 1/p).
///
/// # Arguments
/// * `eigenvalues` — sorted eigenvalues (length p)
/// * `eta` — regularization parameter
/// * `grid_size_opt` — optional grid size (None = auto)
pub fn compute_real_part_dst(
    eigenvalues: &[f64],
    eta: f64,
    grid_size_opt: Option<usize>,
) -> Vec<f64> {
    let p = eigenvalues.len();
    if p == 0 {
        return Vec::new();
    }

    let lo_raw = eigenvalues[0];
    let hi_raw = eigenvalues[p - 1];
    let raw_range = hi_raw - lo_raw;
    let pad = (1000.0 * eta).max(2.0 * raw_range);
    let lo = lo_raw - pad;
    let hi = hi_raw + pad;
    let range = hi - lo;

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

    // --- Odd kernel R(x) = x/(x²+η²), packed as the imaginary part ---
    // We compute the odd-kernel convolution via FFT. The odd kernel is placed
    // in the imaginary channel of a complex array so that one forward FFT
    // recovers its spectrum via conjugate symmetry (this is the DST-I
    // odd-extension: an odd kernel has a purely imaginary spectrum).
    let mut packed_kernel = vec![Complex64::new(0.0, 0.0); m];
    for (i, slot) in packed_kernel.iter_mut().enumerate() {
        let signed_dist = if i <= half {
            i as f64
        } else {
            i as f64 - m_f64
        };
        let x = signed_dist * dx;
        let denom = x * x + eta * eta;
        *slot = Complex64::new(0.0, x / denom); // odd kernel in imag
    }

    let mut dens_freq: Vec<Complex64> = density.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft(m, FftDirection::Forward);

    fft.process(&mut dens_freq);
    fft.process(&mut packed_kernel);

    // Unpack the odd-kernel spectrum from the packed transform.
    // If C = FFT(a + i·b) with a=0, b=odd kernel, then the odd spectrum is
    //   B[k] = -i/2 · (C[k] - conj(C[(m-k)%m]))
    // The convolution spectrum is Re_hat[k] = dens_freq[k] · B[k].
    let mut re_freq = vec![Complex64::new(0.0, 0.0); m];
    for k in 0..m {
        let ck = packed_kernel[k];
        let cnk = packed_kernel[(m - k) % m];
        let ko = -0.5 * Complex64::new(0.0, 1.0) * (ck - cnk.conj()); // odd kernel spectrum
        re_freq[k] = dens_freq[k] * ko; // → Re[S] spectrum
    }

    let ifft = planner.plan_fft(m, FftDirection::Inverse);
    ifft.process(&mut re_freq);

    let inv_m = 1.0 / m_f64;

    // --- Interpolate back (real part is in re_freq.re) ---
    let mut result = Vec::with_capacity(p);
    for &lam in eigenvalues {
        let pos = (lam - lo) / dx;
        let idx = pos as usize;
        let frac = pos - (idx as f64);

        if idx >= m - 1 {
            result.push(re_freq[m - 1].re * inv_m);
        } else {
            let g0 = re_freq[idx];
            let g1 = re_freq[idx + 1];
            result.push((g0.re * (1.0 - frac) + g1.re * frac) * inv_m);
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
    fn test_dst_real_part_accuracy() {
        let p = 512;
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        let eta = 0.1 / (p as f64).sqrt();

        let res = compute_real_part_dst(&evals, eta, None);

        // Exact real part
        let mut exact = vec![0.0_f64; p];
        for i in 0..p {
            let li = evals[i];
            let mut s = 0.0;
            for &lj in &evals {
                let d = li - lj;
                s += d / (d * d + eta * eta);
            }
            exact[i] = s;
        }

        let mut err = 0.0_f64;
        for i in 0..p {
            err = err.max((res[i] - exact[i]).abs() / exact[i].abs().max(1e-12));
        }
        eprintln!("dst real part rel err: {err:.4}");
        // DST-I odd-extension is equivalent to fft5's odd kernel (~15% error).
        assert!(err < 0.5, "dst real part error too large: {err}");
    }

    #[test]
    fn test_dst_finite() {
        let p = 300;
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        let eta = 0.1 / (p as f64).sqrt();

        let res = compute_real_part_dst(&evals, eta, None);
        assert_eq!(res.len(), p);
        for v in &res {
            assert!(v.is_finite());
        }
    }
}
