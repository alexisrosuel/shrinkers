//! Cache-blocked **and** auto-vectorized Stieltjes sum.
//!
//! Combines the strengths of the `Blocked` and `AutoVectorized` variants:
//!
//! - **λᵢ-outer loop with local scalar accumulators** (`sum_real`/`sum_imag`):
//!   a branch-free reduction over contiguous memory, which LLVM reliably
//!   auto-vectorizes into NEON (Apple Silicon) / AVX2 (x86). This is the
//!   structure that makes `autovec` fast.
//!
//! - **Cache blocking (tiling)**: the λⱼ inner loop is processed in blocks of
//!   size `BLOCK_SZ`, keeping the access pattern cache-friendly for large `p`.
//!
//! - **FMA (Fused Multiply-Add)**: `diff.mul_add(diff, eta_sq)` compiles to a
//!   single FMA instruction.
//!
//! - **Far-field cutoff via binary search**: because eigenvalues are sorted,
//!   the set of λⱼ within `cut·η` of λᵢ is a *contiguous* window. We locate it
//!   once per λᵢ with `partition_point` and then run a **branch-free**
//!   vectorized reduction over just that window. This avoids the per-element
//!   branches that prevent the original `Blocked` variant from vectorizing.
//!
//! The result is a method that is simultaneously cache-blocked, SIMD-vectorized,
//! FMA-optimized, and (optionally) cutoff-accelerated.

use crate::stieltjes::autovec::autovec_stieltjes_sum;
use crate::stieltjes::term::BLOCK_SZ;
use rayon::prelude::*;

/// Compute all Stieltjes sums with cache blocking + auto-vectorization.
///
/// Returns (real_parts, imag_parts) as separate SoA-style vectors.
///
/// # Arguments
/// * `eigenvalues` — sorted eigenvalues (length p)
/// * `eta` — regularization parameter
/// * `block_size` — cache block size (default: 64)
/// * `cutoff` — far-field cutoff ratio (None = disabled, Some(r) = enabled)
pub fn compute_all_stieltjes_blocked_autovec(
    eigenvalues: &[f64],
    eta: f64,
    block_size: Option<usize>,
    cutoff: Option<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let p = eigenvalues.len();
    if p == 0 {
        return (Vec::new(), Vec::new());
    }

    let bs = block_size.unwrap_or(BLOCK_SZ);
    let eta_sq = eta * eta;
    let cut = cutoff.unwrap_or(f64::INFINITY);
    let use_cutoff = cutoff.is_some();

    let mut reals = vec![0.0_f64; p];
    let mut imags = vec![0.0_f64; p];

    for (i, &lambda_i) in eigenvalues.iter().enumerate() {
        let mut sum_real = 0.0_f64;
        let mut sum_imag = 0.0_f64;

        // Locate the contiguous far-field window [lo, hi) once per λᵢ.
        // Since eigenvalues are sorted, all λⱼ with |λᵢ-λⱼ| <= cut·η form a
        // contiguous slice. This lets the inner loop be branch-free.
        let (lo, hi) = if use_cutoff {
            let window = cut * eta;
            let lo = eigenvalues.partition_point(|&x| x < lambda_i - window);
            let hi = eigenvalues.partition_point(|&x| x <= lambda_i + window);
            (lo, hi)
        } else {
            (0, p)
        };

        // Cache-blocked, 4× unrolled, branch-free inner loop over the window.
        let mut j = lo;
        while j < hi {
            let block_end = (j + bs).min(hi);

            while j + 4 <= block_end {
                let l0 = eigenvalues[j];
                let l1 = eigenvalues[j + 1];
                let l2 = eigenvalues[j + 2];
                let l3 = eigenvalues[j + 3];

                let d0 = lambda_i - l0;
                let denom0 = d0.mul_add(d0, eta_sq);
                let inv0 = 1.0 / denom0;
                sum_real += d0 * inv0;
                sum_imag += inv0;

                let d1 = lambda_i - l1;
                let denom1 = d1.mul_add(d1, eta_sq);
                let inv1 = 1.0 / denom1;
                sum_real += d1 * inv1;
                sum_imag += inv1;

                let d2 = lambda_i - l2;
                let denom2 = d2.mul_add(d2, eta_sq);
                let inv2 = 1.0 / denom2;
                sum_real += d2 * inv2;
                sum_imag += inv2;

                let d3 = lambda_i - l3;
                let denom3 = d3.mul_add(d3, eta_sq);
                let inv3 = 1.0 / denom3;
                sum_real += d3 * inv3;
                sum_imag += inv3;

                j += 4;
            }

            // Remainder (non-unrolled)
            while j < block_end {
                let diff = lambda_i - eigenvalues[j];
                let denom = diff.mul_add(diff, eta_sq);
                let inv = 1.0 / denom;
                sum_real += diff * inv;
                sum_imag += inv;
                j += 1;
            }
        }

        reals[i] = sum_real;
        sum_imag *= eta;
        imags[i] = sum_imag;
    }

    (reals, imags)
}

/// Parallel version: each λᵢ is computed independently by a Rayon thread,
/// delegating to the same single-point kernel [`stieltjes_sum_blocked_autovec`].
/// No duplicated inner-loop body.
pub(crate) fn compute_all_stieltjes_blocked_autovec_parallel(
    eigenvalues: &[f64],
    eta: f64,
    cutoff: Option<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let p = eigenvalues.len();
    if p == 0 {
        return (Vec::new(), Vec::new());
    }

    let results: Vec<(f64, f64)> = eigenvalues
        .par_iter()
        .map(|&lambda_i| stieltjes_sum_blocked_autovec(lambda_i, eigenvalues, eta, cutoff))
        .collect();

    let mut reals = Vec::with_capacity(p);
    let mut imags = Vec::with_capacity(p);
    for (r, i) in results {
        reals.push(r);
        imags.push(i);
    }

    (reals, imags)
}

/// Compute a single Stieltjes sum with far-field cutoff via binary search.
///
/// `None` means "no cutoff" — compute all terms exactly (matching the
/// sequential `compute_all_stieltjes_blocked_autovec` semantics).
#[inline(always)]
pub fn stieltjes_sum_blocked_autovec(
    lambda_i: f64,
    eigenvalues: &[f64],
    eta: f64,
    cutoff: Option<f64>,
) -> (f64, f64) {
    let Some(cut) = cutoff else {
        return autovec_stieltjes_sum(lambda_i, eigenvalues, eta);
    };
    let eta_sq = eta * eta;
    let mut sum_real = 0.0;
    let mut sum_inv = 0.0;

    let window = cut * eta;
    let lo = eigenvalues.partition_point(|&x| x < lambda_i - window);
    let hi = eigenvalues.partition_point(|&x| x <= lambda_i + window);

    for &lambda_j in &eigenvalues[lo..hi] {
        let diff = lambda_i - lambda_j;
        let denom = diff.mul_add(diff, eta_sq);
        let inv = 1.0 / denom;
        sum_real += diff * inv;
        sum_inv += inv;
    }

    (sum_real, eta * sum_inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stieltjes::autovec::autovec_stieltjes_sum;

    #[test]
    fn test_blocked_autovec_matches_autovec() {
        let evals: Vec<f64> = (0..50).map(|i| (i as f64 + 0.5).ln_1p()).collect();
        let eta = 0.05;

        let (reals, imags) = compute_all_stieltjes_blocked_autovec(&evals, eta, Some(16), None);

        for (i, &li) in evals.iter().enumerate() {
            let (ref_r, ref_i) = autovec_stieltjes_sum(li, &evals, eta);
            assert!(
                (reals[i] - ref_r).abs() < 1e-12,
                "Real mismatch at {i}: {} vs {}",
                reals[i],
                ref_r
            );
            assert!(
                (imags[i] - ref_i).abs() < 1e-12,
                "Imag mismatch at {i}: {} vs {}",
                imags[i],
                ref_i
            );
        }
    }

    #[test]
    fn test_blocked_autovec_cutoff_matches_full() {
        // With a very large cutoff ratio, the window covers everything,
        // so the result must match the full (no-cutoff) computation.
        let p = 200;
        let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        let (reals, imags) =
            compute_all_stieltjes_blocked_autovec(&evals, eta, Some(64), Some(1e9));
        let (ref_r, ref_i) = compute_all_stieltjes_blocked_autovec(&evals, eta, Some(64), None);

        for i in 0..p {
            assert!(
                (reals[i] - ref_r[i]).abs() < 1e-12,
                "Real mismatch at {i}: {} vs {}",
                reals[i],
                ref_r[i]
            );
            assert!(
                (imags[i] - ref_i[i]).abs() < 1e-12,
                "Imag mismatch at {i}: {} vs {}",
                imags[i],
                ref_i[i]
            );
        }
    }

    #[test]
    fn test_blocked_autovec_parallel_matches_sequential() {
        let p = 300;
        let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        let (seq_r, seq_i) =
            compute_all_stieltjes_blocked_autovec(&evals, eta, Some(64), Some(10.0));
        let (par_r, par_i) =
            compute_all_stieltjes_blocked_autovec_parallel(&evals, eta, Some(10.0));

        for i in 0..p {
            assert!(
                (seq_r[i] - par_r[i]).abs() < 1e-12,
                "Real mismatch at {i}: {} vs {}",
                seq_r[i],
                par_r[i]
            );
            assert!(
                (seq_i[i] - par_i[i]).abs() < 1e-12,
                "Imag mismatch at {i}: {} vs {}",
                seq_i[i],
                par_i[i]
            );
        }
    }
}
