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
