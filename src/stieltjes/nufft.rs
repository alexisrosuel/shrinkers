//! Coarse-grid NUFFT for the Stieltjes transform.
//!
//! `fft5` sizes its convolution grid by `dx <= η/8` **and** `m >= 8p`. The
//! second bound resolves every eigenvalue individually; it is unnecessary
//! because the density is band-limited by the splat stencil before the FFT.
//! This module keeps the same dual-convolution kernel but sizes the grid from
//! the kernel scale only:
//!
//! ```text
//!   pad   = max(pad_eta_mult·η, pad_range_frac·raw_range)
//!   range = raw_range + 2·pad
//!   m     = next_pow2(ceil(resolution·range/η)), clamped to [min_grid, max_grid]
//! ```
//!
//! With `resolution = 5`, `pad_eta_mult = 2000` and `max_grid = 65 536`, `m`
//! stays at 32k–64k from p ≈ 20k to p = 1M — independent of p — so the cost is
//! O(p + m log m) instead of O(p log p) with a p-sized grid.
//!
//! The convolution itself (splat, packed forward FFT with the odd kernel,
//! analytic even kernel, packed inverse FFT, interpolation) lives in
//! [`super::fft5`]; this module owns only the NUFFT policy and the parallel
//! entry points. See `docs/nufft_algorithms.md` for the full derivation and
//! measured accuracy/speed landscape.

use super::fft5::{self, Fft5Options, Order};
use super::next_pow2;

/// Padding before the spectrum, as a multiple of η.
///
/// The odd (Hilbert) kernel has a 1/x tail; periodizing it on the FFT grid
/// otherwise produces a wrap-around floor. 2000·η was measured to put the
/// floor at ~2e-5 (p = 50k) / ~4e-5 (p = 1M) for η = 1/√p.
pub const DEFAULT_PAD_ETA_MULT: f64 = 2000.0;

/// Padding before the spectrum, as a fraction of the raw eigenvalue range.
///
/// Takes over from [`DEFAULT_PAD_ETA_MULT`] when the spectrum is wide and η is
/// large; 0.75 is the measured fft5 default.
pub const DEFAULT_PAD_RANGE_FRAC: f64 = 0.75;

/// Grid cells per η of kernel scale.
///
/// Five cells per η is the measured speed/accuracy knee for the Cubic
/// stencil: it keeps `m = 32 768` (rather than 65 536) at p = 1M, where the
/// error moves only from 4.3e-5 to 5.1e-5 for ~30 % less FFT work.
pub const DEFAULT_RESOLUTION: f64 = 5.0;

/// Smallest grid the NUFFT will build (power-of-two rounding happens after).
pub const DEFAULT_MIN_GRID: usize = 8192;

/// Largest grid the NUFFT will build.
///
/// The cap is what makes the method scale with p: at very small η the
/// unconstrained rule would ask for hundreds of thousands of cells, while
/// 65 536 keeps the two FFTs (~1 ms) and the O(p·N) transfer dominant. The
/// cost is the last digit: at η = 0.1/√p, p = 200k the error is ~1e-4 instead
/// of ~5e-5.
pub const DEFAULT_MAX_GRID: usize = 65_536;

/// Knobs for the coarse-grid NUFFT.
#[derive(Debug, Clone, Copy)]
pub struct NufftOptions {
    /// Grid transfer stencil for splatting and interpolation.
    pub order: Order,
    /// Padding as a multiple of η (see [`DEFAULT_PAD_ETA_MULT`]).
    pub pad_eta_mult: f64,
    /// Padding as a fraction of the raw eigenvalue range.
    pub pad_range_frac: f64,
    /// Grid cells per η (see [`DEFAULT_RESOLUTION`]).
    pub resolution: f64,
    /// Lower bound on the grid size.
    pub min_grid: usize,
    /// Upper bound on the grid size.
    pub max_grid: usize,
    /// Force a grid size, bypassing the sizing rule. `None` (default) uses
    /// [`Self::grid_size_for_range`]. Exposed for parameter sweeps; the
    /// caller owns the accuracy/speed trade-off.
    pub m_override: Option<usize>,
}

impl Default for NufftOptions {
    fn default() -> Self {
        Self {
            order: Order::Cubic,
            pad_eta_mult: DEFAULT_PAD_ETA_MULT,
            pad_range_frac: DEFAULT_PAD_RANGE_FRAC,
            resolution: DEFAULT_RESOLUTION,
            min_grid: DEFAULT_MIN_GRID,
            max_grid: DEFAULT_MAX_GRID,
            m_override: None,
        }
    }
}

impl NufftOptions {
    /// Grid size for a spectrum of width `raw_range` at imaginary offset `eta`.
    ///
    /// This is the pure sizing rule; [`Self::grid_size`] feeds it the actual
    /// sorted spectrum. Exposed for tests and for callers that want to budget
    /// the FFT before building the grid.
    ///
    /// # Panics
    ///
    /// Panics if `eta` is not finite and strictly positive.
    #[inline]
    pub fn grid_size_for_range(&self, raw_range: f64, eta: f64) -> usize {
        assert!(
            eta.is_finite() && eta > 0.0,
            "NUFFT needs a positive, finite eta, got {eta}"
        );
        if let Some(m) = self.m_override {
            return m.max(2);
        }
        let pad = (self.pad_eta_mult * eta).max(self.pad_range_frac * raw_range);
        let range = raw_range + 2.0 * pad;
        // Clamp the target BEFORE rounding up: a tiny eta would otherwise
        // overflow `next_pow2` before the cap could apply.
        let max_grid = self.max_grid.max(self.min_grid).max(2);
        let target = self.resolution * range / eta;
        let target = if target.is_finite() {
            target.min(max_grid as f64)
        } else {
            max_grid as f64
        };
        next_pow2((target.ceil() as usize).max(self.min_grid)).min(max_grid)
    }

    /// Grid size for a sorted spectrum.
    ///
    /// # Panics
    ///
    /// Panics if `eigenvalues` is empty or `eta` is invalid (see
    /// [`Self::grid_size_for_range`]).
    #[inline]
    pub fn grid_size(&self, eigenvalues: &[f64], eta: f64) -> usize {
        assert!(
            !eigenvalues.is_empty(),
            "NUFFT needs at least one eigenvalue"
        );
        let raw_range = eigenvalues[eigenvalues.len() - 1] - eigenvalues[0];
        self.grid_size_for_range(raw_range, eta)
    }

    /// Translate this policy into the underlying fft5 grid options.
    #[inline]
    pub fn fft5_options(&self, eigenvalues: &[f64], eta: f64) -> Fft5Options {
        Fft5Options {
            order: self.order,
            m_override: Some(self.grid_size(eigenvalues, eta)),
            pad_eta_mult: self.pad_eta_mult,
            pad_range_frac: self.pad_range_frac,
        }
    }
}

/// Coarse-grid NUFFT at all sample eigenvalues, with explicit options.
///
/// Returns raw sums (not scaled by `1/p`), one per eigenvalue, in input order.
/// `eigenvalues` must be sorted ascending (the same contract as `fft5`).
///
/// # Panics
///
/// Panics if `eta` is not finite and strictly positive.
pub fn compute_all_stieltjes_nufft_with_options(
    eigenvalues: &[f64],
    eta: f64,
    parallel: bool,
    opts: &NufftOptions,
) -> Vec<(f64, f64)> {
    if eigenvalues.is_empty() {
        return Vec::new();
    }
    let grid_opts = opts.fft5_options(eigenvalues, eta);
    let grid = fft5::fft_convolution(eigenvalues, eta, &grid_opts, parallel);
    fft5::interpolate_grid(&grid, eigenvalues, grid_opts.order, parallel)
}

/// Coarse-grid NUFFT at arbitrary query points, with explicit options.
///
/// Query points are expected to lie inside the padded spectrum range; the
/// interpolation stencil clamps at the grid edges (same contract as `fft5`).
pub fn compute_stieltjes_nufft_at_points_with_options(
    query_points: &[f64],
    eigenvalues: &[f64],
    eta: f64,
    parallel: bool,
    opts: &NufftOptions,
) -> Vec<(f64, f64)> {
    if eigenvalues.is_empty() || query_points.is_empty() {
        return Vec::new();
    }
    let grid_opts = opts.fft5_options(eigenvalues, eta);
    let grid = fft5::fft_convolution(eigenvalues, eta, &grid_opts, parallel);
    fft5::interpolate_grid(&grid, query_points, grid_opts.order, parallel)
}

/// [`compute_all_stieltjes_nufft_with_options`] at [`NufftOptions::default`].
pub fn compute_all_stieltjes_nufft(
    eigenvalues: &[f64],
    eta: f64,
    parallel: bool,
) -> Vec<(f64, f64)> {
    compute_all_stieltjes_nufft_with_options(eigenvalues, eta, parallel, &NufftOptions::default())
}

/// [`compute_stieltjes_nufft_at_points_with_options`] at defaults.
pub fn compute_stieltjes_nufft_at_points(
    query_points: &[f64],
    eigenvalues: &[f64],
    eta: f64,
    parallel: bool,
) -> Vec<(f64, f64)> {
    compute_stieltjes_nufft_at_points_with_options(
        query_points,
        eigenvalues,
        eta,
        parallel,
        &NufftOptions::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CutoffConfig, Parallelism, StieltjesMethod};
    use crate::stieltjes::{compute_all_stieltjes, compute_stieltjes_at_points};

    #[test]
    fn grid_size_is_capped_and_independent_of_p() {
        let d = NufftOptions::default();
        // Wide spectrum / small eta: the cap must win.
        assert_eq!(d.grid_size_for_range(4.0, 1e-4), DEFAULT_MAX_GRID);
        // Moderate eta: the rule lands on a small power of two.
        assert_eq!(d.grid_size_for_range(2.0, 0.01), 32768);
        // Without padding the floor wins.
        let floor = NufftOptions {
            pad_eta_mult: 0.0,
            pad_range_frac: 0.0,
            ..NufftOptions::default()
        };
        assert_eq!(floor.grid_size_for_range(0.1, 1.0), DEFAULT_MIN_GRID);
        // A tiny eta must hit the cap, not overflow the power-of-two rounding.
        assert_eq!(d.grid_size_for_range(2.0, 1e-300), DEFAULT_MAX_GRID);
        // An explicit override bypasses the rule.
        let forced = NufftOptions {
            m_override: Some(12345),
            ..NufftOptions::default()
        };
        assert_eq!(forced.grid_size_for_range(4.0, 1e-4), 12345);
    }

    #[test]
    fn nufft_matches_exact_and_parallel() {
        let p = 8192;
        let evs = crate::stieltjes::testutil::log_spectrum(p);
        let eta = 1.0 / (p as f64).sqrt();
        let inv_p = 1.0 / p as f64;

        let exact = compute_all_stieltjes(
            &evs,
            eta,
            StieltjesMethod::Blocked,
            None,
            CutoffConfig::Disabled,
            64,
            Parallelism::Parallel,
        );
        let got = compute_all_stieltjes_nufft(&evs, eta, false);
        let scaled: Vec<(f64, f64)> = got.iter().map(|&(r, i)| (r * inv_p, i * inv_p)).collect();
        let err: f64 = scaled
            .iter()
            .zip(&exact)
            .map(|(g, e)| (g.0 - e.0).powi(2) + (g.1 - e.1).powi(2))
            .sum::<f64>()
            .sqrt()
            / exact
                .iter()
                .map(|e| e.0 * e.0 + e.1 * e.1)
                .sum::<f64>()
                .sqrt();
        assert!(err < 1e-4, "nufft all-points rel-L2 {err:.3e}");

        // The parallel splat reduces per-worker buffers in a different order,
        // so require numerical agreement rather than bit identity.
        let par = compute_all_stieltjes_nufft(&evs, eta, true);
        let max_diff = got
            .iter()
            .zip(&par)
            .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()))
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-9, "seq/par nufft max diff {max_diff:.3e}");

        let (lo, hi) = (evs[0], evs[p - 1]);
        let grid: Vec<f64> = (0..200)
            .map(|k| lo + (hi - lo) * (k as f64 / 199.0))
            .collect();
        let exact_grid = compute_stieltjes_at_points(
            &grid,
            &evs,
            eta,
            StieltjesMethod::Blocked,
            None,
            Parallelism::Parallel,
            None,
        );
        let got_grid = compute_stieltjes_nufft_at_points(&grid, &evs, eta, false);
        let num: f64 = got_grid
            .iter()
            .zip(&exact_grid)
            .map(|(g, e)| ((g.0 - e.0) * inv_p).powi(2) + ((g.1 - e.1) * inv_p).powi(2))
            .sum();
        let den: f64 = exact_grid
            .iter()
            .map(|e| (e.0 * inv_p).powi(2) + (e.1 * inv_p).powi(2))
            .sum();
        let err_grid = (num / den).sqrt();
        assert!(err_grid < 1e-4, "nufft grid rel-L2 {err_grid:.3e}");
    }
}
