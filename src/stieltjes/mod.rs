//! Stieltjes transform computation methods.
//!
//! Provides multiple algorithms for computing the empirical Stieltjes transform:
//!
//! - `naive`: O(p²) scalar loop (baseline)
//! - `autovec`: O(p²) auto-vectorized loop (LLVM NEON/AVX2)
//! - `blocked`: O(p²) cache-blocked + loop-unrolled + FMA + far-field cutoff
//! - `blocked_autovec`: O(p²) cache-blocked + auto-vectorized + binary-search window
//! - `blocked_tiled`: O(p²) 2D-tiled cache-blocked (output block outer)
//! - `blocked_windowed`: O(p·k) cache-blocked + binary-search far-field window
//! - `adaptive`: balanced real(FFT) + imaginary(windowed)
//! - `fft5`/`fft3`/`fft2`: O(p log p) FFT-based grid convolution
//! - `treecode`: O(p log p) 1D tree code / Fast Multipole Method
//!
//! # Single Responsibility
//!
//! - The core term `1/((λᵢ-λⱼ) - iη)` lives **only** in `term.rs`.
//!   Every direct-sum method delegates there — no duplication of the formula.
//! - Every method returns **raw sums** (not scaled by `1/p`). The `1/p`
//!   scaling is applied exactly once, centrally, in [`compute_all_stieltjes`].
//! - Parallel variants delegate to the same single-point kernels as their
//!   sequential counterparts, so there is no duplicated inner-loop body.

mod adaptive;
mod autovec;
mod blocked_autovec;
mod cacheblock;
mod chebcode;
pub mod ewald;
pub mod fft2;
pub mod fft3;
pub mod fft5;
mod fftplan;
mod hodlr;
mod naive;
mod simd;
pub mod term;
mod treecode;

pub use adaptive::*;
pub use autovec::*;
pub use blocked_autovec::*;
pub use cacheblock::*;
pub use chebcode::*;
pub use ewald::*;
pub use fft2::*;
pub use fft3::*;
pub use fft5::*;
pub use hodlr::*;
pub use naive::*;
pub use term::*;
pub use treecode::*;

use crate::config::{CutoffConfig, Parallelism, StieltjesMethod};
use rayon::prelude::*;

/// Compute the Stieltjes sum S(λᵢ) = Σⱼ 1/((λᵢ-λⱼ) - iη)
/// returning (sum_real, sum_imag) for a single λᵢ using the selected method.
///
/// Note: the result is the raw sum (not multiplied by 1/p).
#[inline(always)]
pub fn stieltjes_sum_for_one(
    lambda_i: f64,
    eigenvalues: &[f64],
    eta: f64,
    method: StieltjesMethod,
    cutoff: Option<f64>,
) -> (f64, f64) {
    match method {
        StieltjesMethod::Naive => naive::naive_stieltjes_sum(lambda_i, eigenvalues, eta),
        StieltjesMethod::AutoVectorized => {
            autovec::autovec_stieltjes_sum(lambda_i, eigenvalues, eta)
        }
        StieltjesMethod::Blocked
        | StieltjesMethod::BlockedTiled
        | StieltjesMethod::BlockedWindowed
        | StieltjesMethod::BlockedHybrid => {
            cacheblock::stieltjes_sum_cutoff(lambda_i, eigenvalues, eta, cutoff)
        }
        StieltjesMethod::BlockedAutoVec => {
            blocked_autovec::stieltjes_sum_blocked_autovec(lambda_i, eigenvalues, eta, cutoff)
        }
        StieltjesMethod::Adaptive
        | StieltjesMethod::Fft5
        | StieltjesMethod::Fft3
        | StieltjesMethod::Fft2
        | StieltjesMethod::TreeCode
        | StieltjesMethod::ChebCode
        | StieltjesMethod::Ewald
        | StieltjesMethod::Dst
        | StieltjesMethod::AccuracyAuto
        | StieltjesMethod::SpeedAuto
        | StieltjesMethod::Hodlr
        | StieltjesMethod::Auto => {
            // FFT/treecode/adaptive methods cannot compute a single point
            // efficiently, so we use the autovec fallback for single-point
            // evaluations. Auto should already be resolved upstream; this is
            // a safety fallback.
            autovec::autovec_stieltjes_sum(lambda_i, eigenvalues, eta)
        }
    }
}

/// Compute the raw Stieltjes sum at arbitrary query points (not necessarily
/// sample eigenvalues) using the selected method.
///
/// Returns a `Vec<(real, imag)>` of **raw sums** (not scaled by `1/p`), one
/// per query point. This is the building block for evaluating the Stieltjes
/// transform on a uniform grid (e.g. the deconvolution grid), where the query
/// points differ from the sample eigenvalues.
///
/// For the direct methods (`Naive`, `AutoVectorized`, `Blocked`,
/// `BlockedTiled`, `BlockedWindowed`, `BlockedAutoVec`) this delegates to the
/// corresponding single-point kernel. For the global/approximate methods
/// (`Adaptive`, `Fft5`, `Fft3`, `Fft2`, `TreeCode`, `Ewald`, `Dst`, `Auto`)
/// which cannot be evaluated efficiently at a single point, it falls back to
/// the exact auto-vectorized kernel — preserving correctness.
pub fn compute_stieltjes_at_points(
    query_points: &[f64],
    eigenvalues: &[f64],
    eta: f64,
    method: StieltjesMethod,
    cutoff: Option<f64>,
    parallelism: Parallelism,
    grid_size_opt: Option<usize>,
) -> Vec<(f64, f64)> {
    let parallel = matches!(parallelism, Parallelism::Rayon);

    // The cache-blocked write-batched kernel is the fastest exact method for
    // evaluating at arbitrary query points (it reuses the same 2-source-per-
    // pass structure as `compute_all_stieltjes_blocked`). Use it for the
    // blocked-family methods when the query points are a uniform grid.
    match method {
        StieltjesMethod::Blocked
        | StieltjesMethod::BlockedTiled
        | StieltjesMethod::BlockedWindowed
        | StieltjesMethod::BlockedHybrid => {
            if parallel {
                // Parallel: each query point is independent → data-parallel
                // over query points, delegating to the single-point kernel.
                let pairs: Vec<(f64, f64)> = query_points
                    .par_iter()
                    .map(|&q| stieltjes_sum_cutoff(q, eigenvalues, eta, cutoff))
                    .collect();
                pairs
            } else {
                let (reals, imags) = cacheblock::compute_stieltjes_blocked_at_points(
                    query_points,
                    eigenvalues,
                    eta,
                    None,
                    cutoff,
                );
                reals.into_iter().zip(imags).collect()
            }
        }
        // The FFT methods evaluate the whole grid in one O(p log p)
        // convolution, then interpolate at the query points. This is the
        // fast path for the deconvolution grid (previously these methods
        // fell back to the exact per-point kernel).
        StieltjesMethod::Fft5 | StieltjesMethod::Fft3 | StieltjesMethod::Fft2 => {
            fft5::compute_stieltjes_fft_at_points(query_points, eigenvalues, eta, grid_size_opt)
        }
        _ => {
            if parallel {
                query_points
                    .par_iter()
                    .map(|&q| stieltjes_sum_for_one(q, eigenvalues, eta, method, cutoff))
                    .collect()
            } else {
                query_points
                    .iter()
                    .map(|&q| stieltjes_sum_for_one(q, eigenvalues, eta, method, cutoff))
                    .collect()
            }
        }
    }
}

/// Scale a raw SoA `(reals, imags)` result by `1/p` and convert to AoS pairs.
#[inline]
fn scale_soa(reals: Vec<f64>, imags: Vec<f64>, inv_p: f64) -> Vec<(f64, f64)> {
    reals
        .into_iter()
        .zip(imags)
        .map(|(r, i)| (r * inv_p, i * inv_p))
        .collect()
}

/// Scale a raw AoS `Vec<(f64, f64)>` result by `1/p`.
#[inline]
fn scale_aos(pairs: Vec<(f64, f64)>, inv_p: f64) -> Vec<(f64, f64)> {
    pairs
        .into_iter()
        .map(|(r, i)| (r * inv_p, i * inv_p))
        .collect()
}

/// Below this p, Rayon scheduling overhead exceeds the tiled-kernel gain and
/// the per-row autovec kernel wins (measured: par-tiled 0.26 ms vs per-row
/// 0.22 ms at p=1000; the ordering flips between p=1000 and p=5000).
const PAR_TILED_MIN_P: usize = 2000;

/// Cheap O(p) sortedness check so the tree-code methods can skip their
/// defensive O(p log p) sort when the caller passes pre-sorted eigenvalues
/// (which the whole pipeline does).
pub(crate) fn is_sorted_ascending(v: &[f64]) -> bool {
    v.windows(2).all(|w| w[0] <= w[1])
}

/// Dispatch the exact blocked/tiled family over sequential and parallel
/// execution — one policy, used by both the `Blocked` and `BlockedTiled`
/// arms of [`compute_all_stieltjes`].
///
/// - Sequential: the 2D-tiled kernel with auto-tuned block size
///   (output-block-outer keeps the output block resident in cache across all
///   source sweeps; `block_size` has always been ignored by this family).
/// - Parallel, large p: the same tiled hot body distributed over disjoint
///   output chunks with `par_chunks_mut` (no false sharing, no reduction).
///   Note this also gives `BlockedTiled` + Rayon a genuinely parallel path —
///   it previously ran sequentially.
/// - Parallel, small p: per-row autovec (scheduling overhead dominates).
///
/// With a far-field cutoff enabled, every path here delegates to the
/// *windowed* implementations (see the branch below).
fn dispatch_exact_blocked(
    eigenvalues: &[f64],
    eta: f64,
    cutoff_ratio: Option<f64>,
    parallel: bool,
) -> (Vec<f64>, Vec<f64>) {
    let p = eigenvalues.len();

    // Cutoff ⇒ windowed kernels: identical included term set (pairs with
    // |λᵢ-λⱼ| ≤ ratio·η), but the contiguous window is located by binary
    // search so far-field iterations are skipped entirely — O(p·k) instead
    // of an O(p²) sweep with per-term branch skips.
    if let Some(cut) = cutoff_ratio {
        return if parallel {
            cacheblock::compute_all_stieltjes_blocked_windowed_parallel(
                eigenvalues,
                eta,
                None,
                Some(cut),
            )
        } else {
            cacheblock::compute_all_stieltjes_blocked_windowed(eigenvalues, eta, None, Some(cut))
        };
    }

    match (parallel, p >= PAR_TILED_MIN_P) {
        (true, true) => {
            cacheblock::compute_all_stieltjes_blocked_tiled_parallel(eigenvalues, eta, None, None)
        }
        (true, false) => {
            cacheblock::compute_all_stieltjes_blocked_parallel(eigenvalues, eta, None, None)
        }
        (false, _) => cacheblock::compute_all_stieltjes_blocked(eigenvalues, eta, None, None),
    }
}

/// Compute the full Stieltjes transform for all eigenvalues.
///
/// Returns a Vec of (real, imag) pairs, one per eigenvalue, scaled by `1/p`.
/// The FFT and TreeCode methods are only beneficial when computing all p
/// values at once, so they are dispatched here.
pub fn compute_all_stieltjes(
    eigenvalues: &[f64],
    eta: f64,
    method: StieltjesMethod,
    fft_grid_size: Option<usize>,
    cutoff: CutoffConfig,
    block_size: usize,
    parallelism: Parallelism,
) -> Vec<(f64, f64)> {
    let p = eigenvalues.len();
    if p == 0 {
        return Vec::new();
    }

    let cutoff_ratio = match cutoff {
        CutoffConfig::Enabled { ratio } => Some(ratio),
        CutoffConfig::Disabled => None,
    };
    let inv_p = 1.0 / (p as f64);
    let parallel = matches!(parallelism, Parallelism::Rayon);
    // Data-driven presets resolve via the measured Pareto table.
    let method = if matches!(
        method,
        StieltjesMethod::AccuracyAuto | StieltjesMethod::SpeedAuto
    ) {
        let speed = method == StieltjesMethod::SpeedAuto;
        let rayon = matches!(parallelism, Parallelism::Rayon);
        crate::config::pareto_autogen::pareto_pick(speed, rayon, p)
    } else {
        method
    };

    match method {
        // Resolved above; kept only for match exhaustiveness.
        StieltjesMethod::AccuracyAuto | StieltjesMethod::SpeedAuto => {
            unreachable!("preset autos resolve to concrete methods before dispatch")
        }
        StieltjesMethod::Adaptive => scale_aos(
            adaptive::compute_all_stieltjes_adaptive(eigenvalues, eta, fft_grid_size, cutoff_ratio),
            inv_p,
        ),
        StieltjesMethod::Fft5 => scale_aos(
            fft5::compute_all_stieltjes_fft5(eigenvalues, eta, fft_grid_size),
            inv_p,
        ),
        StieltjesMethod::Fft3 => scale_aos(
            fft3::compute_all_stieltjes_fft3(eigenvalues, eta, fft_grid_size),
            inv_p,
        ),
        StieltjesMethod::Fft2 => scale_aos(
            fft2::compute_all_stieltjes_fft2(eigenvalues, eta, fft_grid_size),
            inv_p,
        ),
        StieltjesMethod::TreeCode => scale_aos(
            treecode::compute_all_stieltjes_treecode_impl(eigenvalues, eta, 0.5, 6, parallel),
            inv_p,
        ),
        StieltjesMethod::ChebCode => scale_aos(
            chebcode::compute_all_stieltjes_chebcode_impl(eigenvalues, eta, 0.3, 9, 16, parallel),
            inv_p,
        ),
        StieltjesMethod::Ewald => scale_aos(
            ewald::compute_all_stieltjes_ewald(eigenvalues, eta, None, fft_grid_size),
            inv_p,
        ),
        StieltjesMethod::Dst => {
            // The DST-I odd-extension real part is mathematically the same
            // odd-kernel convolution that the shared fft5 grid computes with
            // fewer transforms (1 forward + 1 inverse vs 2 forward + 1
            // inverse). `Dst` therefore delegates to the fft5 real part plus
            // the windowed imaginary part — i.e. the Adaptive composition
            // with a user-configurable window. (The standalone `dst` module
            // was removed; see the CHANGELOG.)
            let reals = {
                let pairs = fft5::compute_all_stieltjes_fft5(eigenvalues, eta, fft_grid_size);
                pairs.into_iter().map(|(r, _)| r).collect::<Vec<f64>>()
            };
            let (_, imags) = cacheblock::compute_all_stieltjes_blocked_windowed(
                eigenvalues,
                eta,
                None,
                cutoff_ratio,
            );
            scale_soa(reals, imags, inv_p)
        }
        StieltjesMethod::Auto => {
            // Auto should already be resolved upstream by rie_shrinkage.
            // If reached here, fall back to Blocked which is the safe default.
            let (reals, imags) = if parallel {
                cacheblock::compute_all_stieltjes_blocked_parallel(
                    eigenvalues,
                    eta,
                    None,
                    Some(block_size),
                )
            } else {
                cacheblock::compute_all_stieltjes_blocked(eigenvalues, eta, None, None)
            };
            scale_soa(reals, imags, inv_p)
        }
        StieltjesMethod::Hodlr => scale_aos(
            hodlr::compute_all_stieltjes_hodlr_impl(
                eigenvalues,
                eta,
                256,
                1e-9,
                32,
                parallel,
                hodlr::HodlrMode::Aca,
            ),
            inv_p,
        ),
        StieltjesMethod::Blocked => {
            let (reals, imags) = dispatch_exact_blocked(eigenvalues, eta, cutoff_ratio, parallel);
            scale_soa(reals, imags, inv_p)
        }
        StieltjesMethod::BlockedAutoVec => {
            let (reals, imags) = if parallel {
                blocked_autovec::compute_all_stieltjes_blocked_autovec_parallel(
                    eigenvalues,
                    eta,
                    Some(block_size),
                    cutoff_ratio,
                )
            } else {
                blocked_autovec::compute_all_stieltjes_blocked_autovec(
                    eigenvalues,
                    eta,
                    Some(block_size),
                    cutoff_ratio,
                )
            };
            scale_soa(reals, imags, inv_p)
        }
        StieltjesMethod::BlockedTiled => {
            let (reals, imags) = dispatch_exact_blocked(eigenvalues, eta, cutoff_ratio, parallel);
            scale_soa(reals, imags, inv_p)
        }
        StieltjesMethod::BlockedWindowed => {
            let (reals, imags) = if parallel {
                cacheblock::compute_all_stieltjes_blocked_windowed_parallel(
                    eigenvalues,
                    eta,
                    Some(block_size),
                    cutoff_ratio,
                )
            } else {
                cacheblock::compute_all_stieltjes_blocked_windowed(
                    eigenvalues,
                    eta,
                    Some(block_size),
                    cutoff_ratio,
                )
            };
            scale_soa(reals, imags, inv_p)
        }
        StieltjesMethod::BlockedHybrid => {
            // Real part: exact blocked/tiled kernel (long-range 1/d tail must
            // not be truncated). Imaginary part: windowed method (short-range,
            // O(p·k)).
            let (reals, _) = if parallel {
                cacheblock::compute_all_stieltjes_blocked_parallel(
                    eigenvalues,
                    eta,
                    None,
                    Some(block_size),
                )
            } else {
                cacheblock::compute_all_stieltjes_blocked(eigenvalues, eta, None, None)
            };
            let (_, imags) = cacheblock::compute_all_stieltjes_blocked_windowed(
                eigenvalues,
                eta,
                Some(block_size),
                cutoff_ratio,
            );
            scale_soa(reals, imags, inv_p)
        }
        StieltjesMethod::Naive | StieltjesMethod::AutoVectorized => {
            let pairs: Vec<(f64, f64)> = if parallel {
                eigenvalues
                    .par_iter()
                    .map(|&li| stieltjes_sum_for_one(li, eigenvalues, eta, method, cutoff_ratio))
                    .collect()
            } else {
                eigenvalues
                    .iter()
                    .map(|&li| stieltjes_sum_for_one(li, eigenvalues, eta, method, cutoff_ratio))
                    .collect()
            };
            scale_aos(pairs, inv_p)
        }
    }
}

// Re-export the classic `stieltjes_term` for backward compat
pub use term::stieltjes_term;

/// Compute the full Stieltjes transform in **single precision (f32)**.
///
/// This is the f32 counterpart of [`compute_all_stieltjes`] for the
/// `Blocked`/`BlockedTiled` methods. It operates on `f32` eigenvalues and
/// returns `(real, imag)` pairs scaled by `1/p`, in f32.
///
/// **Precision tradeoff**: f32 is ~2× faster than f64 (4 elements per NEON
/// instruction vs 2) but has ~1e-2 relative error. Use it when speed matters
/// more than precision (e.g. approximate spectral density reconstruction).
///
/// Only the `Blocked`/`BlockedTiled` methods are supported in f32; other
/// methods fall back to the f64 path.
pub fn compute_all_stieltjes_f32(
    eigenvalues: &[f32],
    eta: f32,
    method: StieltjesMethod,
    cutoff: CutoffConfig,
) -> Vec<(f32, f32)> {
    let p = eigenvalues.len();
    if p == 0 {
        return Vec::new();
    }

    let cutoff_ratio = match cutoff {
        CutoffConfig::Enabled { ratio } => Some(ratio),
        CutoffConfig::Disabled => None,
    };
    let inv_p = 1.0 / (p as f32);

    match method {
        StieltjesMethod::Blocked | StieltjesMethod::BlockedTiled | StieltjesMethod::Auto => {
            let (reals, imags) = cacheblock::compute_all_stieltjes_blocked_tiled_f32(
                eigenvalues,
                eta,
                None,
                cutoff_ratio,
            );
            reals
                .into_iter()
                .zip(imags)
                .map(|(r, i)| (r * inv_p, i * inv_p))
                .collect()
        }
        // Other methods fall back to the f64 path (converted).
        _ => {
            let ev64: Vec<f64> = eigenvalues.iter().map(|&x| x as f64).collect();
            let eta64 = eta as f64;
            compute_all_stieltjes(
                &ev64,
                eta64,
                method,
                None,
                cutoff,
                64,
                Parallelism::Sequential,
            )
            .into_iter()
            .map(|(r, i)| (r as f32, i as f32))
            .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StieltjesMethod;

    #[test]
    fn test_all_methods_agree() {
        let evals: Vec<f64> = (0..50).map(|i| ((i as f64 + 0.5) * 0.2).ln_1p()).collect();
        let eta = 0.05;
        let methods = [
            StieltjesMethod::Naive,
            StieltjesMethod::AutoVectorized,
            StieltjesMethod::Blocked,
            StieltjesMethod::BlockedAutoVec,
            StieltjesMethod::BlockedTiled,
            StieltjesMethod::BlockedWindowed,
            StieltjesMethod::BlockedHybrid,
            StieltjesMethod::Auto,
        ];

        let reference: Vec<(f64, f64)> = evals
            .iter()
            .map(|&li| {
                let (sr, si) = naive::naive_stieltjes_sum(li, &evals, eta);
                (sr / 50.0, si / 50.0)
            })
            .collect();

        for method in &methods {
            let results = compute_all_stieltjes(
                &evals,
                eta,
                *method,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Sequential,
            );
            for (i, (r, i_)) in results.iter().enumerate() {
                let ref_r = reference[i].0;
                let ref_i = reference[i].1;
                assert!(
                    (r - ref_r).abs() < 1e-13,
                    "Method {:?} real mismatch at {i}: {} vs {}",
                    method,
                    r,
                    ref_r
                );
                assert!(
                    (i_ - ref_i).abs() < 1e-13,
                    "Method {:?} imag mismatch at {i}: {} vs {}",
                    method,
                    i_,
                    ref_i
                );
            }
        }
    }

    #[test]
    fn test_blocked_parallel_matches_sequential() {
        use crate::config::CutoffConfig;

        for p in [10, 50, 127, 200] {
            let evals: Vec<f64> = (0..p).map(|i| ((i as f64 + 0.5) * 0.2).ln_1p()).collect();
            let eta = 0.05;

            let seq_results = compute_all_stieltjes(
                &evals,
                eta,
                StieltjesMethod::Blocked,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Sequential,
            );
            let par_results = compute_all_stieltjes(
                &evals,
                eta,
                StieltjesMethod::Blocked,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Rayon,
            );

            for (i, ((sr, si), (pr, pi))) in seq_results.iter().zip(par_results.iter()).enumerate()
            {
                assert!(
                    (sr - pr).abs() < 1e-14,
                    "Real mismatch at p={p} i={i}: seq={sr} vs par={pr}"
                );
                assert!(
                    (si - pi).abs() < 1e-14,
                    "Imag mismatch at p={p} i={i}: seq={si} vs par={pi}"
                );
            }
        }
    }

    #[test]
    fn test_blocked_cutoff_parallel_matches_sequential() {
        // With a far-field cutoff enabled, the parallel blocked path must
        // agree with the sequential one (both skip the same far-field terms;
        // only the summation order differs).
        use crate::config::CutoffConfig;

        for p in [500, 2000, 8000] {
            let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
            evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let eta = 0.1 / (p as f64).sqrt();

            let seq = compute_all_stieltjes(
                &evals,
                eta,
                StieltjesMethod::Blocked,
                None,
                CutoffConfig::Enabled { ratio: 10.0 },
                64,
                Parallelism::Sequential,
            );
            let par = compute_all_stieltjes(
                &evals,
                eta,
                StieltjesMethod::Blocked,
                None,
                CutoffConfig::Enabled { ratio: 10.0 },
                64,
                Parallelism::Rayon,
            );
            for i in 0..p {
                let scale = seq[i].0.abs().max(seq[i].1.abs()).max(1e-12);
                assert!(
                    ((par[i].0 - seq[i].0).abs() + (par[i].1 - seq[i].1).abs()) / scale < 1e-10,
                    "cutoff parallel mismatch at p={p} i={i}: {:?} vs {:?}",
                    par[i],
                    seq[i]
                );
            }
        }
    }

    #[test]
    fn test_tiled_parallel_matches_sequential() {
        // The Rayon-tiled kernel (disjoint output chunks) must reproduce the
        // sequential tiled kernel up to FP summation order.
        use crate::config::CutoffConfig;

        for p in [300, 4000] {
            let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
            evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let eta = 0.1 / (p as f64).sqrt();

            for &cut in &[None, Some(10.0)] {
                let cfg_cut = match cut {
                    Some(r) => CutoffConfig::Enabled { ratio: r },
                    None => CutoffConfig::Disabled,
                };
                let seq = compute_all_stieltjes(
                    &evals,
                    eta,
                    StieltjesMethod::BlockedTiled,
                    None,
                    cfg_cut,
                    64,
                    Parallelism::Sequential,
                );
                let par = compute_all_stieltjes(
                    &evals,
                    eta,
                    StieltjesMethod::BlockedTiled,
                    None,
                    cfg_cut,
                    64,
                    Parallelism::Rayon,
                );
                for i in 0..p {
                    let scale = seq[i].0.abs().max(seq[i].1.abs()).max(1e-12);
                    assert!(
                        ((par[i].0 - seq[i].0).abs() + (par[i].1 - seq[i].1).abs()) / scale < 1e-12,
                        "tiled parallel mismatch at p={p} i={i} cut={cut:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_dst_matches_adaptive_composition() {
        // `Dst` delegates to the shared fft5 grid + windowed imag (the
        // Adaptive composition with a configurable window); with the same
        // window it must equal `Adaptive` exactly.
        use crate::config::CutoffConfig;

        let p = 512;
        let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        let dst_res = compute_all_stieltjes(
            &evals,
            eta,
            StieltjesMethod::Dst,
            None,
            CutoffConfig::Enabled { ratio: 10.0 },
            64,
            Parallelism::Sequential,
        );
        let adaptive = compute_all_stieltjes(
            &evals,
            eta,
            StieltjesMethod::Adaptive,
            None,
            CutoffConfig::Enabled { ratio: 10.0 },
            64,
            Parallelism::Sequential,
        );
        for (a, b) in dst_res.iter().zip(adaptive.iter()) {
            assert!((a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9);
        }
    }

    #[test]
    fn test_blocked_hybrid_matches_exact() {
        // The hybrid mode must recover the exact blocked result: real part
        // from the exact blocked kernel, imaginary part from the windowed
        // method. With a large cutoff the windowed imag is essentially exact,
        // so the whole result should match the exact (no-cutoff) blocked sum.
        use crate::config::CutoffConfig;

        for p in [50, 200, 500] {
            let evals: Vec<f64> = (0..p).map(|i| ((i as f64 + 0.5) * 0.2).ln_1p()).collect();
            let eta = 0.05;

            let exact = compute_all_stieltjes(
                &evals,
                eta,
                StieltjesMethod::Blocked,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Sequential,
            );
            let hybrid = compute_all_stieltjes(
                &evals,
                eta,
                StieltjesMethod::BlockedHybrid,
                None,
                CutoffConfig::Enabled { ratio: 100.0 },
                64,
                Parallelism::Sequential,
            );

            for i in 0..p {
                assert!(
                    (hybrid[i].0 - exact[i].0).abs() < 1e-12,
                    "Hybrid real mismatch at p={p} i={i}: {} vs {}",
                    hybrid[i].0,
                    exact[i].0
                );
                assert!(
                    (hybrid[i].1 - exact[i].1).abs() < 1e-12,
                    "Hybrid imag mismatch at p={p} i={i}: {} vs {}",
                    hybrid[i].1,
                    exact[i].1
                );
            }
        }
    }

    #[test]
    fn test_ewald_and_dst_dispatch() {
        // Ewald and Dst are approximate methods; verify they dispatch through
        // compute_all_stieltjes and return finite, reasonable results.
        use crate::config::CutoffConfig;

        let p = 512;
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        let eta = 0.1 / (p as f64).sqrt();

        for method in [StieltjesMethod::Ewald, StieltjesMethod::Dst] {
            let results = compute_all_stieltjes(
                &evals,
                eta,
                method,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Sequential,
            );
            assert_eq!(results.len(), p);
            for (r, i) in &results {
                assert!(r.is_finite());
                assert!(i.is_finite());
            }
        }
    }
}
