//! Shared rustfft plan cache.
//!
//! [`FftPlanner`] memoizes plans per instance, but every call site used to
//! construct a fresh planner, re-planning the same transform lengths on every
//! invocation (~12% of `fft5` runtime at p=4000). One planner per thread,
//! kept alive for the process lifetime, makes planning a one-off cost.

use num_complex::Complex64;
use rustfft::{Fft, FftDirection, FftPlanner};
use std::cell::RefCell;
use std::sync::Arc;

thread_local! {
    static PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

/// Plan (or fetch a cached) forward/inverse FFT of length `len`.
///
/// The returned plan is shared (`Arc`) and immutable; run it with
/// [`Fft::process`] on a `&mut [Complex64]` buffer.
pub(crate) fn plan_fft(len: usize, direction: FftDirection) -> Arc<dyn Fft<f64>> {
    PLANNER.with(|p| p.borrow_mut().plan_fft(len, direction))
}

/// Convenience: run a cached FFT in place.
pub(crate) fn fft_inplace(buf: &mut [Complex64], direction: FftDirection) {
    plan_fft(buf.len(), direction).process(buf);
}

/// Unpack the two real spectra hidden inside one packed forward FFT.
///
/// If `C = FFT(a + i·b)` for real sequences `a`, `b` of length `m`
/// (the standard packing that halves the forward-FFT count), conjugate
/// symmetry separates them again:
///
/// ```text
/// A[k] = (C[k] + conj(C[(m-k) % m])) / 2        = FFT(a)[k]
/// B[k] = -i · (C[k] - conj(C[(m-k) % m])) / 2   = FFT(b)[k]
/// ```
///
/// Returns `(A, B)`. Used by the fft5 grid builder (density + odd kernel)
/// and by ewald (even + odd far kernels).
pub(crate) fn unpack_packed_real_pair(c: &[Complex64]) -> (Vec<Complex64>, Vec<Complex64>) {
    let m = c.len();
    let mut a = Vec::with_capacity(m);
    let mut b = Vec::with_capacity(m);
    for k in 0..m {
        let ck = c[k];
        let cnk = c[(m - k) % m];
        a.push(0.5 * (ck + cnk.conj()));
        b.push(-0.5 * Complex64::new(0.0, 1.0) * (ck - cnk.conj()));
    }
    (a, b)
}

/// Multiply two kernel spectra against a density spectrum and repack the
/// products as `Im_hat + i·Re_hat`, so ONE inverse FFT recovers both
/// convolutions at once (`packed_out.re` ← imaginary part,
/// `packed_out.im` ← real part, before the usual `1/m` scaling).
///
/// The caller performs the inverse FFT on the returned buffer.
pub(crate) fn pack_dual_product(
    dens_freq: &[Complex64],
    imag_kernel: &[Complex64],
    real_kernel: &[Complex64],
) -> Vec<Complex64> {
    let m = dens_freq.len();
    let mut packed_out = Vec::with_capacity(m);
    for k in 0..m {
        let d = dens_freq[k];
        let im_hat = d * imag_kernel[k]; // → far/grid imaginary part
        let re_hat = d * real_kernel[k]; // → far/grid real part
        packed_out.push(Complex64::new(im_hat.re - re_hat.im, im_hat.im + re_hat.re));
    }
    packed_out
}
