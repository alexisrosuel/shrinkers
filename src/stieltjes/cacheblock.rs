//! Cache-blocked, loop-unrolled, auto-vectorized Stieltjes sum.
//!
//! Combines multiple hardware-aware optimizations:
//!
//! - **Cache blocking (tiling)**: Process eigenvalues in blocks of size `BLOCK_SZ`.
//!   For each λᵢ, only scan eigenvalues within the cache-friendly block window
//!   (contiguous memory access). This prevents L1/L2 cache thrashing for large p.
//!
//! - **Loop unrolling**: Manually unroll the inner loop 4× or 8× to reduce
//!   loop overhead and expose more ILP (Instruction-Level Parallelism).
//!
//! - **FMA (Fused Multiply-Add)**: Uses `diff.mul_add(diff, eta_sq)` which
//!   compiles to a single FMA instruction (NEON `fmla` / AVX `vfmadd231sd`).
//!
//! - **Far-field cutoff**: When |λᵢ-λⱼ| ≫ η, skip the term entirely.
//!   For η = 0.1/√p and large p, this yields ~O(p·k) effectively.
//!
//! - **Structure-of-Arrays (SoA) output**: Returns separate Vec<f64> for
//!   real and imaginary parts to enable SIMD-friendly downstream processing.

use crate::stieltjes::autovec::autovec_stieltjes_sum;
use crate::stieltjes::term::BLOCK_SZ;
use rayon::prelude::*;

/// Compute all Stieltjes sums with cache blocking and loop unrolling.
///
/// Returns (real_parts, imag_parts) as separate SoA-style vectors.
/// Each vector has length p, one entry per eigenvalue.
///
/// This is the special case of [`compute_stieltjes_blocked_at_points`] where
/// the query points are the sample eigenvalues themselves. It delegates to
/// that function so that **all** optimizations (cache blocking, write
/// batching, loop unrolling, FMA, far-field cutoff) are shared between the
/// eigenvalue path and the arbitrary-query-point (grid) path — a single
/// source of truth for the blocked kernel.
///
/// # Cache-invalidation note
///
/// This delegates to the **2D-tiled** kernel ([`compute_all_stieltjes_blocked_tiled`]),
/// which uses the output-block-outer loop order. That keeps each output block
/// resident in cache while sweeping all source eigenvalues, avoiding the
/// cache thrashing of the λⱼ-outer order at large p (where the output array
/// exceeds cache). Measured 13-22% faster than the λⱼ-outer order across
/// p=1000..50000.
///
/// The `block_size` argument is passed through to the tiled kernel, which
/// auto-tunes it when `None` (see [`auto_tiled_block_size`]). Callers that
/// want the cache-optimal size should pass `None`.
///
/// # Arguments
/// * `eigenvalues` — sorted eigenvalues (length p)
/// * `eta` — regularization parameter
/// * `block_size` — cache block size (None = auto-tuned, Some(s) = explicit)
/// * `cutoff` — far-field cutoff ratio (None = disabled, Some(r) = enabled with ratio r)
pub fn compute_all_stieltjes_blocked(
    eigenvalues: &[f64],
    eta: f64,
    _block_size: Option<usize>,
    cutoff: Option<f64>,
) -> (Vec<f64>, Vec<f64>) {
    // Delegate to the 2D-tiled kernel: output-block-outer loop order keeps the
    // output resident in cache across all source sweeps (minimizes cache
    // invalidation on the output arrays). Pass `None` for block_size so the
    // tiled kernel auto-selects the cache-optimal block size for p.
    compute_all_stieltjes_blocked_tiled(eigenvalues, eta, None, cutoff)
}

/// Compute all Stieltjes sums in parallel using Rayon.
///
/// Restructures the computation so each λᵢ (target eigenvalue) accumulates
/// independently over all λⱼ (source eigenvalues). Each thread computes its
/// own λᵢ sum into local scalar variables — no shared state or reduction needed.
///
/// This differs from `compute_all_stieltjes_blocked` which iterates λⱼ-outer
/// (accumulating into shared arrays) — that structure cannot be parallelized
/// without expensive per-thread partial arrays. The λᵢ-outer structure is
/// trivially data-parallel.
///
/// Delegates to the single-point kernel [`stieltjes_sum_cutoff`] — no
/// duplicated inner-loop body.
///
/// Returns (real_parts, imag_parts) as separate SoA-style vectors.
pub fn compute_all_stieltjes_blocked_parallel(
    eigenvalues: &[f64],
    eta: f64,
    cutoff: Option<f64>,
    _block_size: Option<usize>,
) -> (Vec<f64>, Vec<f64>) {
    let p = eigenvalues.len();
    if p == 0 {
        return (Vec::new(), Vec::new());
    }

    let results: Vec<(f64, f64)> = eigenvalues
        .par_iter()
        .map(|&lambda_i| stieltjes_sum_cutoff(lambda_i, eigenvalues, eta, cutoff))
        .collect();

    // Transpose Vec<(f64, f64)> into two separate Vecs
    let mut reals = Vec::with_capacity(p);
    let mut imags = Vec::with_capacity(p);
    for (r, i) in results {
        reals.push(r);
        imags.push(i);
    }

    (reals, imags)
}

/// Compute a single Stieltjes sum with far-field cutoff.
/// Falls back to full sum if cutoff is None.
///
/// When the cutoff is enabled, the eigenvalues' sortedness is exploited: the
/// λⱼ within `cutoff·η` of λᵢ form a contiguous window, located once with
/// binary search. Only that window is summed — far-field iterations are
/// skipped entirely rather than skipped per-term with a branch. This makes
/// the single-point kernel an order of magnitude faster than a full scan at
/// small η·√p, and it is what the parallel blocked path uses per λᵢ.
#[inline(always)]
pub fn stieltjes_sum_cutoff(
    lambda_i: f64,
    eigenvalues: &[f64],
    eta: f64,
    cutoff: Option<f64>,
) -> (f64, f64) {
    // `None` means "no cutoff" — compute all terms exactly (matching the
    // sequential `compute_all_stieltjes_blocked` semantics).
    let Some(cut) = cutoff else {
        return autovec_stieltjes_sum(lambda_i, eigenvalues, eta);
    };
    let eta_sq = eta * eta;
    let window = cut * eta;
    let lo = eigenvalues.partition_point(|&x| x < lambda_i - window);
    let hi = eigenvalues.partition_point(|&x| x <= lambda_i + window);

    let mut sum_real = 0.0;
    let mut sum_inv = 0.0;
    for &lambda_j in &eigenvalues[lo..hi] {
        let diff = lambda_i - lambda_j;
        let denom = diff.mul_add(diff, eta_sq);
        let inv = 1.0 / denom;
        sum_real += diff * inv;
        sum_inv += inv;
    }

    (sum_real, eta * sum_inv)
}

/// Compute the raw Stieltjes sum at arbitrary query points using the
/// cache-blocked, write-batched structure.
///
/// This is the query-point analogue of [`compute_all_stieltjes_blocked`]: the
/// **source** eigenvalues λⱼ are iterated in pairs (halving write traffic to
/// the output arrays), while the **target** query points are swept in
/// cache-sized blocks. It evaluates S(q) = Σⱼ 1/((q-λⱼ) - iη) at every query
/// point q, returning raw sums (not scaled by 1/p).
///
/// This is used by the deconvolution path, where the query points are a
/// uniform grid over [lo, hi] rather than the sample eigenvalues themselves.
///
/// Returns (real_parts, imag_parts) as SoA vectors, each of length
/// `query_points.len()`.
pub fn compute_stieltjes_blocked_at_points(
    query_points: &[f64],
    eigenvalues: &[f64],
    eta: f64,
    block_size: Option<usize>,
    cutoff: Option<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let nq = query_points.len();
    let p = eigenvalues.len();
    if nq == 0 || p == 0 {
        return (Vec::new(), Vec::new());
    }

    let bs = block_size.unwrap_or(BLOCK_SZ);
    let eta_sq = eta * eta;
    // Effective cutoff distance = cutoff_ratio * eta. When cutoff is None,
    // no term is skipped.
    let cut_dist = cutoff.map(|r| r * eta).unwrap_or(f64::INFINITY);

    let mut reals = vec![0.0_f64; nq];
    let mut imags = vec![0.0_f64; nq];

    // Branch hoisting: dispatch once on whether the far-field cutoff is
    // enabled. Each branch runs a dedicated inner loop with NO per-iteration
    // `use_cutoff` check — the compiler emits a single tight loop body.
    if cutoff.is_some() {
        at_points_inner_loop(
            query_points,
            eigenvalues,
            &mut reals,
            &mut imags,
            bs,
            eta,
            eta_sq,
            cut_dist,
        );
    } else {
        at_points_inner_loop_no_cutoff(
            query_points,
            eigenvalues,
            &mut reals,
            &mut imags,
            bs,
            eta,
            eta_sq,
        );
    }

    // Idea 1 (eta-hoist): inner loops accumulate raw `inv`; scale by eta once.
    for v in imags.iter_mut() {
        *v *= eta;
    }

    (reals, imags)
}

/// Inner loop of the query-point kernel with the far-field cutoff enabled.
///
/// `cut` is the cutoff distance (already `cutoff_ratio * eta`); terms with
/// `|q-λⱼ| > cut` are skipped.
#[inline(always)]
#[allow(clippy::too_many_arguments)] // hot inner loop; grouping args would hurt perf
fn at_points_inner_loop(
    query_points: &[f64],
    eigenvalues: &[f64],
    reals: &mut [f64],
    imags: &mut [f64],
    bs: usize,
    _eta: f64,
    eta_sq: f64,
    cut: f64,
) {
    let nq = query_points.len();
    let p = eigenvalues.len();

    // Outer loop: iterate over source eigenvalues in pairs to halve write
    // traffic to reals[]/imags[]. Each pair of λⱼ values contributes to the
    // same output locations (the query points), so batching reduces memory
    // bandwidth pressure by 2× while keeping the auto-vectorization-friendly
    // inner loop.
    let mut j = 0;
    while j + 2 <= p {
        let lj0 = eigenvalues[j];
        let lj1 = eigenvalues[j + 1];

        // Inner loop: cache-blocked over target query points
        for block_start in (0..nq).step_by(bs) {
            let block_end = (block_start + bs).min(nq);

            // Unrolled inner loop over the block (4× unrolling)
            let mut i = block_start;
            while i + 4 <= block_end {
                let q0 = query_points[i];
                let q1 = query_points[i + 1];
                let q2 = query_points[i + 2];
                let q3 = query_points[i + 3];

                // λⱼ₀
                let d00 = q0 - lj0;
                let d10 = q1 - lj0;
                let d20 = q2 - lj0;
                let d30 = q3 - lj0;
                let a00 = if d00 < 0.0 { -d00 } else { d00 };
                let a10 = if d10 < 0.0 { -d10 } else { d10 };
                let a20 = if d20 < 0.0 { -d20 } else { d20 };
                let a30 = if d30 < 0.0 { -d30 } else { d30 };
                if a00 <= cut {
                    let denom = d00.mul_add(d00, eta_sq);
                    let inv = 1.0 / denom;
                    reals[i] = d00.mul_add(inv, reals[i]);
                    imags[i] += inv;
                }
                if a10 <= cut {
                    let denom = d10.mul_add(d10, eta_sq);
                    let inv = 1.0 / denom;
                    reals[i + 1] = d10.mul_add(inv, reals[i + 1]);
                    imags[i + 1] += inv;
                }
                if a20 <= cut {
                    let denom = d20.mul_add(d20, eta_sq);
                    let inv = 1.0 / denom;
                    reals[i + 2] = d20.mul_add(inv, reals[i + 2]);
                    imags[i + 2] += inv;
                }
                if a30 <= cut {
                    let denom = d30.mul_add(d30, eta_sq);
                    let inv = 1.0 / denom;
                    reals[i + 3] = d30.mul_add(inv, reals[i + 3]);
                    imags[i + 3] += inv;
                }
                // λⱼ₁
                let d01 = q0 - lj1;
                let d11 = q1 - lj1;
                let d21 = q2 - lj1;
                let d31 = q3 - lj1;
                let a01 = if d01 < 0.0 { -d01 } else { d01 };
                let a11 = if d11 < 0.0 { -d11 } else { d11 };
                let a21 = if d21 < 0.0 { -d21 } else { d21 };
                let a31 = if d31 < 0.0 { -d31 } else { d31 };
                if a01 <= cut {
                    let denom = d01.mul_add(d01, eta_sq);
                    let inv = 1.0 / denom;
                    reals[i] = d01.mul_add(inv, reals[i]);
                    imags[i] += inv;
                }
                if a11 <= cut {
                    let denom = d11.mul_add(d11, eta_sq);
                    let inv = 1.0 / denom;
                    reals[i + 1] = d11.mul_add(inv, reals[i + 1]);
                    imags[i + 1] += inv;
                }
                if a21 <= cut {
                    let denom = d21.mul_add(d21, eta_sq);
                    let inv = 1.0 / denom;
                    reals[i + 2] = d21.mul_add(inv, reals[i + 2]);
                    imags[i + 2] += inv;
                }
                if a31 <= cut {
                    let denom = d31.mul_add(d31, eta_sq);
                    let inv = 1.0 / denom;
                    reals[i + 3] = d31.mul_add(inv, reals[i + 3]);
                    imags[i + 3] += inv;
                }

                i += 4;
            }

            // Remainder (non-unrolled) — process both λⱼ₀ and λⱼ₁
            while i < block_end {
                let qi = query_points[i];
                let abs_diff0 = if qi - lj0 < 0.0 {
                    -(qi - lj0)
                } else {
                    qi - lj0
                };
                if abs_diff0 <= cut {
                    let diff = qi - lj0;
                    let denom = diff.mul_add(diff, eta_sq);
                    let inv_denom = 1.0 / denom;
                    reals[i] = diff.mul_add(inv_denom, reals[i]);
                    imags[i] += inv_denom;
                }
                let abs_diff1 = if qi - lj1 < 0.0 {
                    -(qi - lj1)
                } else {
                    qi - lj1
                };
                if abs_diff1 <= cut {
                    let diff = qi - lj1;
                    let denom = diff.mul_add(diff, eta_sq);
                    let inv_denom = 1.0 / denom;
                    reals[i] = diff.mul_add(inv_denom, reals[i]);
                    imags[i] += inv_denom;
                }
                i += 1;
            }
        }

        j += 2;
    }

    // Handle odd p (one remaining λⱼ)
    if j < p {
        let lambda_j = eigenvalues[j];
        for block_start in (0..nq).step_by(bs) {
            let block_end = (block_start + bs).min(nq);

            let mut i = block_start;
            while i + 4 <= block_end {
                let q0 = query_points[i];
                let q1 = query_points[i + 1];
                let q2 = query_points[i + 2];
                let q3 = query_points[i + 3];

                let d0 = q0 - lambda_j;
                let d1 = q1 - lambda_j;
                let d2 = q2 - lambda_j;
                let d3 = q3 - lambda_j;
                let a0 = if d0 < 0.0 { -d0 } else { d0 };
                let a1 = if d1 < 0.0 { -d1 } else { d1 };
                let a2 = if d2 < 0.0 { -d2 } else { d2 };
                let a3 = if d3 < 0.0 { -d3 } else { d3 };
                if a0 <= cut {
                    let denom = d0.mul_add(d0, eta_sq);
                    let inv = 1.0 / denom;
                    reals[i] = d0.mul_add(inv, reals[i]);
                    imags[i] += inv;
                }
                if a1 <= cut {
                    let denom = d1.mul_add(d1, eta_sq);
                    let inv = 1.0 / denom;
                    reals[i + 1] = d1.mul_add(inv, reals[i + 1]);
                    imags[i + 1] += inv;
                }
                if a2 <= cut {
                    let denom = d2.mul_add(d2, eta_sq);
                    let inv = 1.0 / denom;
                    reals[i + 2] = d2.mul_add(inv, reals[i + 2]);
                    imags[i + 2] += inv;
                }
                if a3 <= cut {
                    let denom = d3.mul_add(d3, eta_sq);
                    let inv = 1.0 / denom;
                    reals[i + 3] = d3.mul_add(inv, reals[i + 3]);
                    imags[i + 3] += inv;
                }

                i += 4;
            }

            while i < block_end {
                let diff = query_points[i] - lambda_j;
                let abs_diff = if diff < 0.0 { -diff } else { diff };
                if abs_diff <= cut {
                    let denom = diff.mul_add(diff, eta_sq);
                    let inv_denom = 1.0 / denom;
                    reals[i] = diff.mul_add(inv_denom, reals[i]);
                    imags[i] += inv_denom;
                }
                i += 1;
            }
        }
    }
}

/// Inner loop of the query-point kernel with the far-field cutoff disabled.
///
/// Computes every term exactly (no branch, no skip). This is a separate
/// function so the hot loop body has no `use_cutoff` branch at all.
#[inline(always)]
fn at_points_inner_loop_no_cutoff(
    query_points: &[f64],
    eigenvalues: &[f64],
    reals: &mut [f64],
    imags: &mut [f64],
    bs: usize,
    _eta: f64,
    eta_sq: f64,
) {
    let nq = query_points.len();
    let p = eigenvalues.len();

    // Outer loop: iterate over source eigenvalues in pairs to halve write
    // traffic to reals[]/imags[].
    let mut j = 0;
    while j + 2 <= p {
        let lj0 = eigenvalues[j];
        let lj1 = eigenvalues[j + 1];

        // Inner loop: cache-blocked over target query points
        for block_start in (0..nq).step_by(bs) {
            let block_end = (block_start + bs).min(nq);

            // Unrolled inner loop over the block (4× unrolling)
            let mut i = block_start;
            while i + 4 <= block_end {
                let q0 = query_points[i];
                let q1 = query_points[i + 1];
                let q2 = query_points[i + 2];
                let q3 = query_points[i + 3];

                // λⱼ₀
                let d00 = q0 - lj0;
                let denom00 = d00.mul_add(d00, eta_sq);
                let inv00 = 1.0 / denom00;
                reals[i] = d00.mul_add(inv00, reals[i]);
                imags[i] += inv00;

                let d10 = q1 - lj0;
                let denom10 = d10.mul_add(d10, eta_sq);
                let inv10 = 1.0 / denom10;
                reals[i + 1] = d10.mul_add(inv10, reals[i + 1]);
                imags[i + 1] += inv10;

                let d20 = q2 - lj0;
                let denom20 = d20.mul_add(d20, eta_sq);
                let inv20 = 1.0 / denom20;
                reals[i + 2] = d20.mul_add(inv20, reals[i + 2]);
                imags[i + 2] += inv20;

                let d30 = q3 - lj0;
                let denom30 = d30.mul_add(d30, eta_sq);
                let inv30 = 1.0 / denom30;
                reals[i + 3] = d30.mul_add(inv30, reals[i + 3]);
                imags[i + 3] += inv30;

                // λⱼ₁
                let d01 = q0 - lj1;
                let denom01 = d01.mul_add(d01, eta_sq);
                let inv01 = 1.0 / denom01;
                reals[i] = d01.mul_add(inv01, reals[i]);
                imags[i] += inv01;

                let d11 = q1 - lj1;
                let denom11 = d11.mul_add(d11, eta_sq);
                let inv11 = 1.0 / denom11;
                reals[i + 1] = d11.mul_add(inv11, reals[i + 1]);
                imags[i + 1] += inv11;

                let d21 = q2 - lj1;
                let denom21 = d21.mul_add(d21, eta_sq);
                let inv21 = 1.0 / denom21;
                reals[i + 2] = d21.mul_add(inv21, reals[i + 2]);
                imags[i + 2] += inv21;

                let d31 = q3 - lj1;
                let denom31 = d31.mul_add(d31, eta_sq);
                let inv31 = 1.0 / denom31;
                reals[i + 3] = d31.mul_add(inv31, reals[i + 3]);
                imags[i + 3] += inv31;

                i += 4;
            }

            // Remainder (non-unrolled) — process both λⱼ₀ and λⱼ₁
            while i < block_end {
                let qi = query_points[i];
                let diff0 = qi - lj0;
                let denom0 = diff0.mul_add(diff0, eta_sq);
                let inv0 = 1.0 / denom0;
                reals[i] = diff0.mul_add(inv0, reals[i]);
                imags[i] += inv0;

                let diff1 = qi - lj1;
                let denom1 = diff1.mul_add(diff1, eta_sq);
                let inv1 = 1.0 / denom1;
                reals[i] = diff1.mul_add(inv1, reals[i]);
                imags[i] += inv1;
                i += 1;
            }
        }

        j += 2;
    }

    // Handle odd p (one remaining λⱼ)
    if j < p {
        let lambda_j = eigenvalues[j];
        for block_start in (0..nq).step_by(bs) {
            let block_end = (block_start + bs).min(nq);

            let mut i = block_start;
            while i + 4 <= block_end {
                let q0 = query_points[i];
                let q1 = query_points[i + 1];
                let q2 = query_points[i + 2];
                let q3 = query_points[i + 3];

                let d0 = q0 - lambda_j;
                let denom0 = d0.mul_add(d0, eta_sq);
                let inv0 = 1.0 / denom0;
                reals[i] = d0.mul_add(inv0, reals[i]);
                imags[i] += inv0;

                let d1 = q1 - lambda_j;
                let denom1 = d1.mul_add(d1, eta_sq);
                let inv1 = 1.0 / denom1;
                reals[i + 1] = d1.mul_add(inv1, reals[i + 1]);
                imags[i + 1] += inv1;

                let d2 = q2 - lambda_j;
                let denom2 = d2.mul_add(d2, eta_sq);
                let inv2 = 1.0 / denom2;
                reals[i + 2] = d2.mul_add(inv2, reals[i + 2]);
                imags[i + 2] += inv2;

                let d3 = q3 - lambda_j;
                let denom3 = d3.mul_add(d3, eta_sq);
                let inv3 = 1.0 / denom3;
                reals[i + 3] = d3.mul_add(inv3, reals[i + 3]);
                imags[i + 3] += inv3;

                i += 4;
            }

            while i < block_end {
                let diff = query_points[i] - lambda_j;
                let denom = diff.mul_add(diff, eta_sq);
                let inv_denom = 1.0 / denom;
                reals[i] = diff.mul_add(inv_denom, reals[i]);
                imags[i] += inv_denom;
                i += 1;
            }
        }
    }
}

/// Windowed cache-blocked Stieltjes sum.
///
/// Exploits the fact that eigenvalues are **sorted**: for each source λⱼ, the
/// set of target λᵢ within the far-field cutoff `cut·η` forms a *contiguous*
/// window `[lo, hi)`. We locate it once per λⱼ with `partition_point` (binary
/// search) and only iterate over that window.
///
/// This turns the O(p²) iteration into O(p·k) where k is the average window
/// size — skipping far-field iterations entirely (not just skipping the write,
/// which is what the branch-based cutoff in `compute_all_stieltjes_blocked`
/// does). For η = 0.1/√p and large p, k ≪ p, so this is a large win.
///
/// Requires `cutoff` to be `Some` (the window is the whole point). If `None`,
/// falls back to the full blocked computation.
pub fn compute_all_stieltjes_blocked_windowed(
    eigenvalues: &[f64],
    eta: f64,
    block_size: Option<usize>,
    cutoff: Option<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let p = eigenvalues.len();
    if p == 0 {
        return (Vec::new(), Vec::new());
    }
    let Some(cut) = cutoff else {
        return compute_all_stieltjes_blocked(eigenvalues, eta, block_size, None);
    };

    let bs = block_size.unwrap_or(BLOCK_SZ);
    let eta_sq = eta * eta;
    let window = cut * eta;

    let mut reals = vec![0.0_f64; p];
    let mut imags = vec![0.0_f64; p];

    // λⱼ-outer loop (source eigenvalues). For each λⱼ, binary-search the
    // contiguous window of λᵢ within `window`, then accumulate into the
    // output arrays only for that window.
    for &lj in eigenvalues.iter() {
        let lo = eigenvalues.partition_point(|&x| x < lj - window);
        let hi = eigenvalues.partition_point(|&x| x <= lj + window);

        // Cache-blocked inner loop over the window [lo, hi).
        let mut i = lo;
        while i < hi {
            let block_end = (i + bs).min(hi);

            // 4× unrolled inner loop
            while i + 4 <= block_end {
                let l0 = eigenvalues[i];
                let l1 = eigenvalues[i + 1];
                let l2 = eigenvalues[i + 2];
                let l3 = eigenvalues[i + 3];

                let d0 = l0 - lj;
                let denom0 = d0.mul_add(d0, eta_sq);
                let inv0 = 1.0 / denom0;
                reals[i] = d0.mul_add(inv0, reals[i]);
                imags[i] = eta.mul_add(inv0, imags[i]);

                let d1 = l1 - lj;
                let denom1 = d1.mul_add(d1, eta_sq);
                let inv1 = 1.0 / denom1;
                reals[i + 1] = d1.mul_add(inv1, reals[i + 1]);
                imags[i + 1] = eta.mul_add(inv1, imags[i + 1]);

                let d2 = l2 - lj;
                let denom2 = d2.mul_add(d2, eta_sq);
                let inv2 = 1.0 / denom2;
                reals[i + 2] = d2.mul_add(inv2, reals[i + 2]);
                imags[i + 2] = eta.mul_add(inv2, imags[i + 2]);

                let d3 = l3 - lj;
                let denom3 = d3.mul_add(d3, eta_sq);
                let inv3 = 1.0 / denom3;
                reals[i + 3] = d3.mul_add(inv3, reals[i + 3]);
                imags[i + 3] = eta.mul_add(inv3, imags[i + 3]);

                i += 4;
            }

            // Remainder
            while i < block_end {
                let diff = eigenvalues[i] - lj;
                let denom = diff.mul_add(diff, eta_sq);
                let inv = 1.0 / denom;
                reals[i] = diff.mul_add(inv, reals[i]);
                imags[i] = eta.mul_add(inv, imags[i]);
                i += 1;
            }
        }
    }

    (reals, imags)
}

/// Single-point windowed Stieltjes sum: binary-search the contiguous window
/// of λⱼ within `cut·η` of `lambda_i` and sum only those terms.
#[inline(always)]
pub fn stieltjes_sum_windowed(
    lambda_i: f64,
    eigenvalues: &[f64],
    eta: f64,
    cut: f64,
) -> (f64, f64) {
    let eta_sq = eta * eta;
    let window = cut * eta;
    let lo = eigenvalues.partition_point(|&x| x < lambda_i - window);
    let hi = eigenvalues.partition_point(|&x| x <= lambda_i + window);

    let mut sum_real = 0.0;
    let mut sum_imag = 0.0;
    for &lambda_j in &eigenvalues[lo..hi] {
        let diff = lambda_i - lambda_j;
        let denom = diff.mul_add(diff, eta_sq);
        let inv = 1.0 / denom;
        sum_real = diff.mul_add(inv, sum_real);
        sum_imag = eta.mul_add(inv, sum_imag);
    }
    (sum_real, sum_imag)
}

/// Parallel windowed variant: each λᵢ is processed independently by a Rayon
/// thread, delegating to the single-point kernel [`stieltjes_sum_windowed`].
/// No duplicated inner-loop body.
pub fn compute_all_stieltjes_blocked_windowed_parallel(
    eigenvalues: &[f64],
    eta: f64,
    _block_size: Option<usize>,
    cutoff: Option<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let p = eigenvalues.len();
    if p == 0 {
        return (Vec::new(), Vec::new());
    }
    let Some(cut) = cutoff else {
        return compute_all_stieltjes_blocked_parallel(eigenvalues, eta, None, None);
    };

    let results: Vec<(f64, f64)> = eigenvalues
        .par_iter()
        .map(|&lambda_i| stieltjes_sum_windowed(lambda_i, eigenvalues, eta, cut))
        .collect();

    let mut reals = Vec::with_capacity(p);
    let mut imags = Vec::with_capacity(p);
    for (r, i) in results {
        reals.push(r);
        imags.push(i);
    }

    (reals, imags)
}

/// Auto-select the optimal cache block size for the tiled variant.
///
/// The tiled variant keeps the output block resident in cache across all
/// source sweeps. The sweet spot is the smallest block that still amortizes
/// loop overhead — measured at ~8 for large p (output block = 2·8·8 = 128
/// bytes, fits in L1). For small p the output array already fits in cache, so
/// a larger block avoids loop overhead.
///
/// Empirically (M-series, p=10000): bs8=40.7ms, bs16=46.3ms, bs32=49.1ms,
/// bs64=50.7ms, bs128=51.7ms, bs4=57.7ms. So bs8 is optimal at large p.
pub fn auto_tiled_block_size(p: usize) -> usize {
    if p <= 256 {
        // Small p: output fits in L1/L2 regardless; use a moderate block to
        // avoid loop overhead.
        32
    } else {
        // Mid/large p: bs8 (128-byte output block) is the measured sweet spot.
        // It keeps the output block resident in L1 while sweeping all source
        // eigenvalues, and beats bs16/bs32 at both p=1000 and p=10000
        // (e.g. p=1000: bs8 459µs vs bs16 511µs — ~11% faster).
        8
    }
}

/// 2D-tiled cache-blocked Stieltjes sum.
///
/// The original `compute_all_stieltjes_blocked` is λⱼ-outer: for each source
/// λⱼ it sweeps the ENTIRE output array. When the output exceeds cache size
/// (large p), each output line is evicted and reloaded p/2 times — cache
/// thrashing on the output arrays.
///
/// This variant performs **loop interchange**: the output block is the OUTER
/// loop, so it stays resident in cache while sweeping all source eigenvalues.
/// Each output block is loaded from memory once and reused across all j.
///
/// ```text
/// for i_block (target, OUTER):   # stays in cache
///     for j (source, step 2):    # INNER
///         reals[i] += ...        # block resident across all j
///         imags[i] += ...
/// ```
///
/// **Branch hoisting**: the `use_cutoff` decision is loop-invariant, so it is
/// hoisted out of the innermost loop. The function dispatches ONCE to either
/// a cutoff or a non-cutoff inner loop, so the hot loop body contains no
/// per-iteration `if use_cutoff` branch and the compiler emits a single
/// (smaller, I-cache-friendly) loop body instead of two inlined copies.
///
/// Auto-selects the block size via [`auto_tiled_block_size`] when `block_size`
/// is `None`.
pub fn compute_all_stieltjes_blocked_tiled(
    eigenvalues: &[f64],
    eta: f64,
    block_size: Option<usize>,
    cutoff: Option<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let p = eigenvalues.len();
    if p == 0 {
        return (Vec::new(), Vec::new());
    }

    let eta_sq = eta * eta;
    // Effective cutoff distance = cutoff_ratio * eta (matches the original
    // `a <= cut * eta` comparison). When cutoff is None, no term is skipped.
    let cut_dist = cutoff.map(|r| r * eta);

    let mut reals = vec![0.0_f64; p];
    let mut imags = vec![0.0_f64; p];

    // Branch hoisting: dispatch once on whether the far-field cutoff is
    // enabled. Each branch runs a dedicated inner loop with NO per-iteration
    // `use_cutoff` check — the compiler emits a single tight loop body.
    //
    // No cutoff + sequential: the query set IS the source set, so the
    // symmetric kernel visits each unordered pair once (half the divisions);
    // cache blocking is irrelevant there, so `block_size` only applies to
    // the cutoff sweep.
    if let Some(cut_dist) = cut_dist {
        let bs = block_size.unwrap_or_else(|| auto_tiled_block_size(p));
        tiled_inner_loop(
            eigenvalues,
            &mut reals,
            &mut imags,
            bs,
            eta,
            eta_sq,
            cut_dist,
        );
    } else {
        symmetric_all_points(eigenvalues, &mut reals, &mut imags, eta, eta_sq);
    }

    (reals, imags)
}

/// Symmetric all-points evaluation: visit each unordered source pair ONCE.
///
/// This entry point always evaluates the transform AT the eigenvalues
/// themselves, and for a pair (i, j) the term
/// `1/(λᵢ−λⱼ−iη) = (d + iη)·u` with `d = λᵢ−λⱼ`, `u = 1/(d²+η²)` satisfies:
///
/// - real part **antisymmetric**: `out_r[i] += d·u`, `out_r[j] -= d·u`
/// - imaginary part **symmetric**: `out_i[i] += η·u`, `out_i[j] += η·u`
/// - reciprocal shared between both orientations
///
/// The full-square kernel therefore computes every pair TWICE (two sweeps,
/// two divisions, two round-trips to the output arrays). This kernel visits
/// each unordered pair once with a **register-resident 4×4 schedule** that
/// keeps the original kernel's ILP: 16 independent divisions per tile, row
/// side accumulated in registers, column side accumulated in registers and
/// flushed with a single read-modify-write per column (8 per tile instead
/// of 32). Diagonal term `1/(0−iη) = i/η` folded into the initialization.
///
/// Output identical to the full-square kernel up to FP summation order
/// (~1e-15 relative).
#[allow(clippy::needless_range_loop)]
fn symmetric_all_points(evs: &[f64], out_r: &mut [f64], out_i: &mut [f64], eta: f64, eta_sq: f64) {
    let p = evs.len();
    let diag_im = 1.0 / eta;
    out_r.fill(0.0);
    out_i.fill(diag_im);

    // Target tiles of 4 rows; row-side sums live in registers for the whole
    // tile lifetime.
    let mut t0 = 0;
    while t0 < p {
        let t1 = (t0 + 4).min(p);
        let mut rr = [0.0_f64; 4];
        let mut ri = [0.0_f64; 4];

        // Within-tile pairs (strict upper triangle of the diagonal tile):
        // rare and tiny; plain scalar loops.
        for i in t0..t1 {
            let li = evs[i];
            let ii = i - t0;
            for j in (i + 1)..t1 {
                let d = li - evs[j];
                let inv = 1.0 / d.mul_add(d, eta_sq);
                let w = d * inv;
                let v = eta * inv;
                rr[ii] += w;
                ri[ii] += v;
                out_r[j] -= w;
                out_i[j] += v;
            }
        }

        // Off-diagonal full 4-column source tiles: the hot path.
        let mut j0 = t1;
        while j0 + 4 <= p {
            let l0 = evs[j0];
            let l1 = evs[j0 + 1];
            let l2 = evs[j0 + 2];
            let l3 = evs[j0 + 3];
            let mut cr = [0.0_f64; 4];
            let mut ci = [0.0_f64; 4];

            // Deliberately index-based (see the allow on this function): the
            // explicit `ii = i - t0` register indexing compiles to measurably
            // better code than the iterator form here (~30% at p=1000).
            for i in t0..t1 {
                let li = evs[i];
                let ii = i - t0;

                let d = li - l0;
                let inv = 1.0 / d.mul_add(d, eta_sq);
                let w = d * inv;
                let v = eta * inv;
                rr[ii] += w;
                ri[ii] += v;
                cr[0] -= w;
                ci[0] += v;

                let d = li - l1;
                let inv = 1.0 / d.mul_add(d, eta_sq);
                let w = d * inv;
                let v = eta * inv;
                rr[ii] += w;
                ri[ii] += v;
                cr[1] -= w;
                ci[1] += v;

                let d = li - l2;
                let inv = 1.0 / d.mul_add(d, eta_sq);
                let w = d * inv;
                let v = eta * inv;
                rr[ii] += w;
                ri[ii] += v;
                cr[2] -= w;
                ci[2] += v;

                let d = li - l3;
                let inv = 1.0 / d.mul_add(d, eta_sq);
                let w = d * inv;
                let v = eta * inv;
                rr[ii] += w;
                ri[ii] += v;
                cr[3] -= w;
                ci[3] += v;
            }

            // One read-modify-write per column instead of one per pair.
            out_r[j0] += cr[0];
            out_i[j0] += ci[0];
            out_r[j0 + 1] += cr[1];
            out_i[j0 + 1] += ci[1];
            out_r[j0 + 2] += cr[2];
            out_i[j0 + 2] += ci[2];
            out_r[j0 + 3] += cr[3];
            out_i[j0 + 3] += ci[3];

            j0 += 4;
        }

        // Remainder columns (1–3), scalar.
        while j0 < p {
            let lj = evs[j0];
            for i in t0..t1 {
                let d = evs[i] - lj;
                let inv = 1.0 / d.mul_add(d, eta_sq);
                let w = d * inv;
                let v = eta * inv;
                rr[i - t0] += w;
                ri[i - t0] += v;
                out_r[j0] -= w;
                out_i[j0] += v;
            }
            j0 += 1;
        }

        // Flush the tile's row-side sums.
        for i in t0..t1 {
            out_r[i] += rr[i - t0];
            out_i[i] += ri[i - t0];
        }
        t0 = t1;
    }
}

/// Process a RANGE of target blocks of the tiled kernel (far-field cutoff
/// enabled). Same structure as [`tiled_range_no_cutoff`]; terms with
/// `|λᵢ-λⱼ| > cut·η` are skipped.
///
/// NOTE: the windowed kernels ([`compute_all_stieltjes_blocked_windowed`])
/// compute the same included term set in O(p·k) instead of this O(p²)
/// branch-skip sweep; the dispatcher routes cutoff work there. This kernel
/// remains for direct callers of [`compute_all_stieltjes_blocked_tiled`].
#[allow(clippy::too_many_arguments)]
fn tiled_span_cutoff(
    targets: &[f64],
    eigenvalues: &[f64],
    out_r: &mut [f64],
    out_i: &mut [f64],
    bs: usize,
    eta: f64,
    eta_sq: f64,
    cut: f64,
) {
    let n = out_r.len();
    let mut b = 0;
    while b < n {
        let bend = (b + bs).min(n);
        tiled_one_block_cutoff(
            targets,
            eigenvalues,
            out_r,
            out_i,
            b,
            bend,
            eta,
            eta_sq,
            cut,
        );
        b += bs;
    }
}

/// Sweep ALL source eigenvalues against the target block
/// `[block_start, block_end)`, skipping far-field terms.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tiled_one_block_cutoff(
    targets: &[f64],
    eigenvalues: &[f64],
    out_r: &mut [f64],
    out_i: &mut [f64],
    block_start: usize,
    block_end: usize,
    eta: f64,
    eta_sq: f64,
    cut: f64,
) {
    let p = eigenvalues.len();
    let mut j = 0;
    while j + 4 <= p {
        let lj0 = eigenvalues[j];
        let lj1 = eigenvalues[j + 1];
        let lj2 = eigenvalues[j + 2];
        let lj3 = eigenvalues[j + 3];

        let mut i = block_start;
        while i + 4 <= block_end {
            let l0 = targets[i];
            let l1 = targets[i + 1];
            let l2 = targets[i + 2];
            let l3 = targets[i + 3];

            let d00 = l0 - lj0;
            if (if d00 < 0.0 { -d00 } else { d00 }) <= cut {
                let inv00 = 1.0 / d00.mul_add(d00, eta_sq);
                out_r[i] = d00.mul_add(inv00, out_r[i]);
                out_i[i] = eta.mul_add(inv00, out_i[i]);
            }
            let d10 = l1 - lj0;
            if (if d10 < 0.0 { -d10 } else { d10 }) <= cut {
                let inv10 = 1.0 / d10.mul_add(d10, eta_sq);
                out_r[i + 1] = d10.mul_add(inv10, out_r[i + 1]);
                out_i[i + 1] = eta.mul_add(inv10, out_i[i + 1]);
            }
            let d20 = l2 - lj0;
            if (if d20 < 0.0 { -d20 } else { d20 }) <= cut {
                let inv20 = 1.0 / d20.mul_add(d20, eta_sq);
                out_r[i + 2] = d20.mul_add(inv20, out_r[i + 2]);
                out_i[i + 2] = eta.mul_add(inv20, out_i[i + 2]);
            }
            let d30 = l3 - lj0;
            if (if d30 < 0.0 { -d30 } else { d30 }) <= cut {
                let inv30 = 1.0 / d30.mul_add(d30, eta_sq);
                out_r[i + 3] = d30.mul_add(inv30, out_r[i + 3]);
                out_i[i + 3] = eta.mul_add(inv30, out_i[i + 3]);
            }

            let d01 = l0 - lj1;
            if (if d01 < 0.0 { -d01 } else { d01 }) <= cut {
                let inv01 = 1.0 / d01.mul_add(d01, eta_sq);
                out_r[i] = d01.mul_add(inv01, out_r[i]);
                out_i[i] = eta.mul_add(inv01, out_i[i]);
            }
            let d11 = l1 - lj1;
            if (if d11 < 0.0 { -d11 } else { d11 }) <= cut {
                let inv11 = 1.0 / d11.mul_add(d11, eta_sq);
                out_r[i + 1] = d11.mul_add(inv11, out_r[i + 1]);
                out_i[i + 1] = eta.mul_add(inv11, out_i[i + 1]);
            }
            let d21 = l2 - lj1;
            if (if d21 < 0.0 { -d21 } else { d21 }) <= cut {
                let inv21 = 1.0 / d21.mul_add(d21, eta_sq);
                out_r[i + 2] = d21.mul_add(inv21, out_r[i + 2]);
                out_i[i + 2] = eta.mul_add(inv21, out_i[i + 2]);
            }
            let d31 = l3 - lj1;
            if (if d31 < 0.0 { -d31 } else { d31 }) <= cut {
                let inv31 = 1.0 / d31.mul_add(d31, eta_sq);
                out_r[i + 3] = d31.mul_add(inv31, out_r[i + 3]);
                out_i[i + 3] = eta.mul_add(inv31, out_i[i + 3]);
            }

            let d02 = l0 - lj2;
            if (if d02 < 0.0 { -d02 } else { d02 }) <= cut {
                let inv02 = 1.0 / d02.mul_add(d02, eta_sq);
                out_r[i] = d02.mul_add(inv02, out_r[i]);
                out_i[i] = eta.mul_add(inv02, out_i[i]);
            }
            let d12 = l1 - lj2;
            if (if d12 < 0.0 { -d12 } else { d12 }) <= cut {
                let inv12 = 1.0 / d12.mul_add(d12, eta_sq);
                out_r[i + 1] = d12.mul_add(inv12, out_r[i + 1]);
                out_i[i + 1] = eta.mul_add(inv12, out_i[i + 1]);
            }
            let d22 = l2 - lj2;
            if (if d22 < 0.0 { -d22 } else { d22 }) <= cut {
                let inv22 = 1.0 / d22.mul_add(d22, eta_sq);
                out_r[i + 2] = d22.mul_add(inv22, out_r[i + 2]);
                out_i[i + 2] = eta.mul_add(inv22, out_i[i + 2]);
            }
            let d32 = l3 - lj2;
            if (if d32 < 0.0 { -d32 } else { d32 }) <= cut {
                let inv32 = 1.0 / d32.mul_add(d32, eta_sq);
                out_r[i + 3] = d32.mul_add(inv32, out_r[i + 3]);
                out_i[i + 3] = eta.mul_add(inv32, out_i[i + 3]);
            }

            let d03 = l0 - lj3;
            if (if d03 < 0.0 { -d03 } else { d03 }) <= cut {
                let inv03 = 1.0 / d03.mul_add(d03, eta_sq);
                out_r[i] = d03.mul_add(inv03, out_r[i]);
                out_i[i] = eta.mul_add(inv03, out_i[i]);
            }
            let d13 = l1 - lj3;
            if (if d13 < 0.0 { -d13 } else { d13 }) <= cut {
                let inv13 = 1.0 / d13.mul_add(d13, eta_sq);
                out_r[i + 1] = d13.mul_add(inv13, out_r[i + 1]);
                out_i[i + 1] = eta.mul_add(inv13, out_i[i + 1]);
            }
            let d23 = l2 - lj3;
            if (if d23 < 0.0 { -d23 } else { d23 }) <= cut {
                let inv23 = 1.0 / d23.mul_add(d23, eta_sq);
                out_r[i + 2] = d23.mul_add(inv23, out_r[i + 2]);
                out_i[i + 2] = eta.mul_add(inv23, out_i[i + 2]);
            }
            let d33 = l3 - lj3;
            if (if d33 < 0.0 { -d33 } else { d33 }) <= cut {
                let inv33 = 1.0 / d33.mul_add(d33, eta_sq);
                out_r[i + 3] = d33.mul_add(inv33, out_r[i + 3]);
                out_i[i + 3] = eta.mul_add(inv33, out_i[i + 3]);
            }

            i += 4;
        }

        // Remainder rows of the block.
        while i < block_end {
            let li = targets[i];
            for lj in [lj0, lj1, lj2, lj3] {
                let d = li - lj;
                if (if d < 0.0 { -d } else { d }) <= cut {
                    let inv = 1.0 / d.mul_add(d, eta_sq);
                    out_r[i] = d.mul_add(inv, out_r[i]);
                    out_i[i] = eta.mul_add(inv, out_i[i]);
                }
            }
            i += 1;
        }
        j += 4;
    }

    // Remaining 1–3 sources.
    while j < p {
        let lambda_j = eigenvalues[j];
        for k in block_start..block_end {
            let diff = eigenvalues[k] - lambda_j;
            if (if diff < 0.0 { -diff } else { diff }) <= cut {
                let inv_denom = 1.0 / diff.mul_add(diff, eta_sq);
                out_r[k] = diff.mul_add(inv_denom, out_r[k]);
                out_i[k] = eta.mul_add(inv_denom, out_i[k]);
            }
        }
        j += 1;
    }
}

/// Tiled inner loop with far-field cutoff: process all blocks.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tiled_inner_loop(
    eigenvalues: &[f64],
    reals: &mut [f64],
    imags: &mut [f64],
    bs: usize,
    eta: f64,
    eta_sq: f64,
    cut: f64,
) {
    tiled_span_cutoff(eigenvalues, eigenvalues, reals, imags, bs, eta, eta_sq, cut);
}

/// Process a RANGE of target blocks of the tiled kernel exactly (no cutoff).
///
/// Blocks `[blk_lo*bs .. min(blk_hi*bs, p))` of the output arrays are
/// computed; every source eigenvalue is swept for each block. This is the
/// SINGLE source of truth for the exact tiled hot body: the sequential
/// kernel calls it once over all blocks (identical codegen to the original
/// monolithic loop — absolute indices, `noalias` output params), and the
/// Rayon-parallel kernel calls it once per contiguous chunk of blocks, so
/// each thread accumulates into a disjoint, cache-aligned output span with
/// no reduction and no false sharing.
#[allow(clippy::too_many_arguments)]
fn tiled_span_no_cutoff(
    // Eigenvalue sub-slice this span of outputs corresponds to
    // (`targets.len() == out_r.len()`); passed separately so all indices are
    // provably in-bounds from loop guards alone.
    targets: &[f64],
    // All source eigenvalues.
    eigenvalues: &[f64],
    out_r: &mut [f64],
    out_i: &mut [f64],
    bs: usize,
    eta: f64,
    eta_sq: f64,
) {
    // OUTER loop over this span of target blocks — each block stays resident
    // in cache while we sweep all source eigenvalues.
    let n = out_r.len();
    let mut b = 0;
    while b < n {
        let bend = (b + bs).min(n);
        tiled_one_block_no_cutoff(targets, eigenvalues, out_r, out_i, b, bend, eta, eta_sq);
        b += bs;
    }
}

/// Sweep ALL source eigenvalues against the target block
/// `[block_start, block_end)`, accumulating in place (exact, branch-free).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tiled_one_block_no_cutoff(
    targets: &[f64],
    eigenvalues: &[f64],
    out_r: &mut [f64],
    out_i: &mut [f64],
    block_start: usize,
    block_end: usize,
    eta: f64,
    eta_sq: f64,
) {
    let p = eigenvalues.len();
    // INNER loop over source eigenvalues (quads to halve write traffic and
    // amortize the target-block loads over 4 sources).
    let mut j = 0;
    while j + 4 <= p {
        let lj0 = eigenvalues[j];
        let lj1 = eigenvalues[j + 1];
        let lj2 = eigenvalues[j + 2];
        let lj3 = eigenvalues[j + 3];

        // Unrolled inner loop over the block (4× unrolling)
        let mut i = block_start;
        while i + 4 <= block_end {
            let l0 = targets[i];
            let l1 = targets[i + 1];
            let l2 = targets[i + 2];
            let l3 = targets[i + 3];

            let d00 = l0 - lj0;
            let inv00 = 1.0 / d00.mul_add(d00, eta_sq);
            out_r[i] = d00.mul_add(inv00, out_r[i]);
            out_i[i] = eta.mul_add(inv00, out_i[i]);

            let d10 = l1 - lj0;
            let inv10 = 1.0 / d10.mul_add(d10, eta_sq);
            out_r[i + 1] = d10.mul_add(inv10, out_r[i + 1]);
            out_i[i + 1] = eta.mul_add(inv10, out_i[i + 1]);

            let d20 = l2 - lj0;
            let inv20 = 1.0 / d20.mul_add(d20, eta_sq);
            out_r[i + 2] = d20.mul_add(inv20, out_r[i + 2]);
            out_i[i + 2] = eta.mul_add(inv20, out_i[i + 2]);

            let d30 = l3 - lj0;
            let inv30 = 1.0 / d30.mul_add(d30, eta_sq);
            out_r[i + 3] = d30.mul_add(inv30, out_r[i + 3]);
            out_i[i + 3] = eta.mul_add(inv30, out_i[i + 3]);

            let d01 = l0 - lj1;
            let inv01 = 1.0 / d01.mul_add(d01, eta_sq);
            out_r[i] = d01.mul_add(inv01, out_r[i]);
            out_i[i] = eta.mul_add(inv01, out_i[i]);

            let d11 = l1 - lj1;
            let inv11 = 1.0 / d11.mul_add(d11, eta_sq);
            out_r[i + 1] = d11.mul_add(inv11, out_r[i + 1]);
            out_i[i + 1] = eta.mul_add(inv11, out_i[i + 1]);

            let d21 = l2 - lj1;
            let inv21 = 1.0 / d21.mul_add(d21, eta_sq);
            out_r[i + 2] = d21.mul_add(inv21, out_r[i + 2]);
            out_i[i + 2] = eta.mul_add(inv21, out_i[i + 2]);

            let d31 = l3 - lj1;
            let inv31 = 1.0 / d31.mul_add(d31, eta_sq);
            out_r[i + 3] = d31.mul_add(inv31, out_r[i + 3]);
            out_i[i + 3] = eta.mul_add(inv31, out_i[i + 3]);

            let d02 = l0 - lj2;
            let inv02 = 1.0 / d02.mul_add(d02, eta_sq);
            out_r[i] = d02.mul_add(inv02, out_r[i]);
            out_i[i] = eta.mul_add(inv02, out_i[i]);

            let d12 = l1 - lj2;
            let inv12 = 1.0 / d12.mul_add(d12, eta_sq);
            out_r[i + 1] = d12.mul_add(inv12, out_r[i + 1]);
            out_i[i + 1] = eta.mul_add(inv12, out_i[i + 1]);

            let d22 = l2 - lj2;
            let inv22 = 1.0 / d22.mul_add(d22, eta_sq);
            out_r[i + 2] = d22.mul_add(inv22, out_r[i + 2]);
            out_i[i + 2] = eta.mul_add(inv22, out_i[i + 2]);

            let d32 = l3 - lj2;
            let inv32 = 1.0 / d32.mul_add(d32, eta_sq);
            out_r[i + 3] = d32.mul_add(inv32, out_r[i + 3]);
            out_i[i + 3] = eta.mul_add(inv32, out_i[i + 3]);

            let d03 = l0 - lj3;
            let inv03 = 1.0 / d03.mul_add(d03, eta_sq);
            out_r[i] = d03.mul_add(inv03, out_r[i]);
            out_i[i] = eta.mul_add(inv03, out_i[i]);

            let d13 = l1 - lj3;
            let inv13 = 1.0 / d13.mul_add(d13, eta_sq);
            out_r[i + 1] = d13.mul_add(inv13, out_r[i + 1]);
            out_i[i + 1] = eta.mul_add(inv13, out_i[i + 1]);

            let d23 = l2 - lj3;
            let inv23 = 1.0 / d23.mul_add(d23, eta_sq);
            out_r[i + 2] = d23.mul_add(inv23, out_r[i + 2]);
            out_i[i + 2] = eta.mul_add(inv23, out_i[i + 2]);

            let d33 = l3 - lj3;
            let inv33 = 1.0 / d33.mul_add(d33, eta_sq);
            out_r[i + 3] = d33.mul_add(inv33, out_r[i + 3]);
            out_i[i + 3] = eta.mul_add(inv33, out_i[i + 3]);

            i += 4;
        }

        // Remainder rows of the block (all four λⱼ).
        while i < block_end {
            let li = targets[i];
            for lj in [lj0, lj1, lj2, lj3] {
                let d = li - lj;
                let inv = 1.0 / d.mul_add(d, eta_sq);
                out_r[i] = d.mul_add(inv, out_r[i]);
                out_i[i] = eta.mul_add(inv, out_i[i]);
            }
            i += 1;
        }
        j += 4;
    }

    // Remaining 1–3 sources.
    while j < p {
        let lambda_j = eigenvalues[j];
        for k in block_start..block_end {
            let diff = eigenvalues[k] - lambda_j;
            let inv_denom = 1.0 / diff.mul_add(diff, eta_sq);
            out_r[k] = diff.mul_add(inv_denom, out_r[k]);
            out_i[k] = eta.mul_add(inv_denom, out_i[k]);
        }
        j += 1;
    }
}

/// Auto-select the cache block size for the **parallel** tiled kernel.
///
/// With several threads each owning a distinct output chunk, slightly larger
/// blocks amortize the per-chunk source sweep better than the sequential
/// optimum (8): measured at p=20000 on an 8-core M-series, bs=32–128 are
/// within noise of each other and ~15% ahead of the per-row autovec path.
pub fn auto_tiled_block_size_parallel(_p: usize) -> usize {
    32
}

/// Parallel (Rayon) 2D-tiled Stieltjes sum.
///
/// Contiguous groups of output blocks are distributed over threads; every
/// thread runs the full-square hot body
/// ([`tiled_span_no_cutoff`] / [`tiled_span_cutoff`]) over its own span,
/// so outputs are accumulated in disjoint, cache-aligned regions — no
/// reduction, no false sharing. (The symmetric half-work pairing of the
/// sequential no-cutoff kernel needs scattered cross-chunk updates and does
/// not survive this partitioning.)
///
/// This replaces the former parallel strategy (per-row single-point autovec),
/// which ignored tiling entirely; it also gives `BlockedTiled` + Rayon a
/// genuinely parallel path where it previously ran sequentially. Measured
/// ~2.5× faster than the old parallel path at p=20000 (8-core M-series).
pub fn compute_all_stieltjes_blocked_tiled_parallel(
    eigenvalues: &[f64],
    eta: f64,
    block_size: Option<usize>,
    cutoff: Option<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let p = eigenvalues.len();
    if p == 0 {
        return (Vec::new(), Vec::new());
    }

    let bs = block_size.unwrap_or_else(|| auto_tiled_block_size_parallel(p));
    let eta_sq = eta * eta;
    let cut_dist = cutoff.map(|r| r * eta).unwrap_or(f64::INFINITY);

    let mut reals = vec![0.0_f64; p];
    let mut imags = vec![0.0_f64; p];

    // Split the block range into a few groups per thread so each worker gets
    // one call with a wide span (amortizes scheduling to noise level).
    let n_blocks = p.div_ceil(bs);
    let span_len = bs * (n_blocks.div_ceil(rayon::current_num_threads().max(1) * 4)).max(1);

    if cutoff.is_some() {
        reals
            .par_chunks_mut(span_len)
            .zip(imags.par_chunks_mut(span_len))
            .enumerate()
            .for_each(|(k, (out_r, out_i))| {
                tiled_span_cutoff(
                    &eigenvalues[k * span_len..k * span_len + out_r.len()],
                    eigenvalues,
                    out_r,
                    out_i,
                    bs,
                    eta,
                    eta_sq,
                    cut_dist,
                );
            });
    } else {
        reals
            .par_chunks_mut(span_len)
            .zip(imags.par_chunks_mut(span_len))
            .enumerate()
            .for_each(|(k, (out_r, out_i))| {
                tiled_span_no_cutoff(
                    &eigenvalues[k * span_len..k * span_len + out_r.len()],
                    eigenvalues,
                    out_r,
                    out_i,
                    bs,
                    eta,
                    eta_sq,
                );
            });
    }

    (reals, imags)
}

/// Float32 (single-precision) 2D-tiled Stieltjes sum.
///
/// Same structure as [`compute_all_stieltjes_blocked_tiled`] but operates on
/// `f32`. On NEON this processes **4 elements per instruction** (vs 2 for
/// f64), giving ~2× the throughput. The tradeoff is precision: ~1e-2 relative
/// error vs ~1e-16 for f64. Suitable for the approximate methods or when
/// speed matters more than precision.
///
/// Returns `(reals, imags)` as `Vec<f32>`.
pub fn compute_all_stieltjes_blocked_tiled_f32(
    eigenvalues: &[f32],
    eta: f32,
    block_size: Option<usize>,
    cutoff: Option<f64>,
) -> (Vec<f32>, Vec<f32>) {
    let p = eigenvalues.len();
    if p == 0 {
        return (Vec::new(), Vec::new());
    }

    let bs = block_size.unwrap_or_else(|| auto_tiled_block_size(p));
    let eta_sq = eta * eta;
    let cut_dist = cutoff
        .map(|r| (r * eta as f64) as f32)
        .unwrap_or(f32::INFINITY);

    let mut reals = vec![0.0_f32; p];
    let mut imags = vec![0.0_f32; p];

    if cutoff.is_some() {
        tiled_inner_loop_f32(
            eigenvalues,
            &mut reals,
            &mut imags,
            bs,
            eta,
            eta_sq,
            cut_dist,
        );
    } else {
        tiled_inner_loop_no_cutoff_f32(eigenvalues, &mut reals, &mut imags, bs, eta, eta_sq);
    }

    (reals, imags)
}

/// Float32 tiled inner loop with far-field cutoff.
#[inline(always)]
fn tiled_inner_loop_f32(
    eigenvalues: &[f32],
    reals: &mut [f32],
    imags: &mut [f32],
    bs: usize,
    eta: f32,
    eta_sq: f32,
    cut: f32,
) {
    let p = eigenvalues.len();
    for block_start in (0..p).step_by(bs) {
        let block_end = (block_start + bs).min(p);
        let mut j = 0;
        while j + 4 <= p {
            let lj0 = eigenvalues[j];
            let lj1 = eigenvalues[j + 1];
            let lj2 = eigenvalues[j + 2];
            let lj3 = eigenvalues[j + 3];
            let mut i = block_start;
            while i + 4 <= block_end {
                let l0 = eigenvalues[i];
                let l1 = eigenvalues[i + 1];
                let l2 = eigenvalues[i + 2];
                let l3 = eigenvalues[i + 3];

                let d00 = l0 - lj0;
                let a00 = if d00 < 0.0 { -d00 } else { d00 };
                if a00 <= cut {
                    let denom00 = d00.mul_add(d00, eta_sq);
                    let inv00 = 1.0 / denom00;
                    reals[i] = d00.mul_add(inv00, reals[i]);
                    imags[i] = eta.mul_add(inv00, imags[i]);
                }
                let d10 = l1 - lj0;
                let a10 = if d10 < 0.0 { -d10 } else { d10 };
                if a10 <= cut {
                    let denom10 = d10.mul_add(d10, eta_sq);
                    let inv10 = 1.0 / denom10;
                    reals[i + 1] = d10.mul_add(inv10, reals[i + 1]);
                    imags[i + 1] = eta.mul_add(inv10, imags[i + 1]);
                }
                let d20 = l2 - lj0;
                let a20 = if d20 < 0.0 { -d20 } else { d20 };
                if a20 <= cut {
                    let denom20 = d20.mul_add(d20, eta_sq);
                    let inv20 = 1.0 / denom20;
                    reals[i + 2] = d20.mul_add(inv20, reals[i + 2]);
                    imags[i + 2] = eta.mul_add(inv20, imags[i + 2]);
                }
                let d30 = l3 - lj0;
                let a30 = if d30 < 0.0 { -d30 } else { d30 };
                if a30 <= cut {
                    let denom30 = d30.mul_add(d30, eta_sq);
                    let inv30 = 1.0 / denom30;
                    reals[i + 3] = d30.mul_add(inv30, reals[i + 3]);
                    imags[i + 3] = eta.mul_add(inv30, imags[i + 3]);
                }

                let d01 = l0 - lj1;
                let a01 = if d01 < 0.0 { -d01 } else { d01 };
                if a01 <= cut {
                    let denom01 = d01.mul_add(d01, eta_sq);
                    let inv01 = 1.0 / denom01;
                    reals[i] = d01.mul_add(inv01, reals[i]);
                    imags[i] = eta.mul_add(inv01, imags[i]);
                }
                let d11 = l1 - lj1;
                let a11 = if d11 < 0.0 { -d11 } else { d11 };
                if a11 <= cut {
                    let denom11 = d11.mul_add(d11, eta_sq);
                    let inv11 = 1.0 / denom11;
                    reals[i + 1] = d11.mul_add(inv11, reals[i + 1]);
                    imags[i + 1] = eta.mul_add(inv11, imags[i + 1]);
                }
                let d21 = l2 - lj1;
                let a21 = if d21 < 0.0 { -d21 } else { d21 };
                if a21 <= cut {
                    let denom21 = d21.mul_add(d21, eta_sq);
                    let inv21 = 1.0 / denom21;
                    reals[i + 2] = d21.mul_add(inv21, reals[i + 2]);
                    imags[i + 2] = eta.mul_add(inv21, imags[i + 2]);
                }
                let d31 = l3 - lj1;
                let a31 = if d31 < 0.0 { -d31 } else { d31 };
                if a31 <= cut {
                    let denom31 = d31.mul_add(d31, eta_sq);
                    let inv31 = 1.0 / denom31;
                    reals[i + 3] = d31.mul_add(inv31, reals[i + 3]);
                    imags[i + 3] = eta.mul_add(inv31, imags[i + 3]);
                }

                let d02 = l0 - lj2;
                let a02 = if d02 < 0.0 { -d02 } else { d02 };
                if a02 <= cut {
                    let denom02 = d02.mul_add(d02, eta_sq);
                    let inv02 = 1.0 / denom02;
                    reals[i] = d02.mul_add(inv02, reals[i]);
                    imags[i] = eta.mul_add(inv02, imags[i]);
                }
                let d12 = l1 - lj2;
                let a12 = if d12 < 0.0 { -d12 } else { d12 };
                if a12 <= cut {
                    let denom12 = d12.mul_add(d12, eta_sq);
                    let inv12 = 1.0 / denom12;
                    reals[i + 1] = d12.mul_add(inv12, reals[i + 1]);
                    imags[i + 1] = eta.mul_add(inv12, imags[i + 1]);
                }
                let d22 = l2 - lj2;
                let a22 = if d22 < 0.0 { -d22 } else { d22 };
                if a22 <= cut {
                    let denom22 = d22.mul_add(d22, eta_sq);
                    let inv22 = 1.0 / denom22;
                    reals[i + 2] = d22.mul_add(inv22, reals[i + 2]);
                    imags[i + 2] = eta.mul_add(inv22, imags[i + 2]);
                }
                let d32 = l3 - lj2;
                let a32 = if d32 < 0.0 { -d32 } else { d32 };
                if a32 <= cut {
                    let denom32 = d32.mul_add(d32, eta_sq);
                    let inv32 = 1.0 / denom32;
                    reals[i + 3] = d32.mul_add(inv32, reals[i + 3]);
                    imags[i + 3] = eta.mul_add(inv32, imags[i + 3]);
                }

                let d03 = l0 - lj3;
                let a03 = if d03 < 0.0 { -d03 } else { d03 };
                if a03 <= cut {
                    let denom03 = d03.mul_add(d03, eta_sq);
                    let inv03 = 1.0 / denom03;
                    reals[i] = d03.mul_add(inv03, reals[i]);
                    imags[i] = eta.mul_add(inv03, imags[i]);
                }
                let d13 = l1 - lj3;
                let a13 = if d13 < 0.0 { -d13 } else { d13 };
                if a13 <= cut {
                    let denom13 = d13.mul_add(d13, eta_sq);
                    let inv13 = 1.0 / denom13;
                    reals[i + 1] = d13.mul_add(inv13, reals[i + 1]);
                    imags[i + 1] = eta.mul_add(inv13, imags[i + 1]);
                }
                let d23 = l2 - lj3;
                let a23 = if d23 < 0.0 { -d23 } else { d23 };
                if a23 <= cut {
                    let denom23 = d23.mul_add(d23, eta_sq);
                    let inv23 = 1.0 / denom23;
                    reals[i + 2] = d23.mul_add(inv23, reals[i + 2]);
                    imags[i + 2] = eta.mul_add(inv23, imags[i + 2]);
                }
                let d33 = l3 - lj3;
                let a33 = if d33 < 0.0 { -d33 } else { d33 };
                if a33 <= cut {
                    let denom33 = d33.mul_add(d33, eta_sq);
                    let inv33 = 1.0 / denom33;
                    reals[i + 3] = d33.mul_add(inv33, reals[i + 3]);
                    imags[i + 3] = eta.mul_add(inv33, imags[i + 3]);
                }

                i += 4;
            }
            while i < block_end {
                let li = eigenvalues[i];
                let d0 = li - lj0;
                if d0.abs() <= cut {
                    let denom0 = d0.mul_add(d0, eta_sq);
                    let inv0 = 1.0 / denom0;
                    reals[i] = d0.mul_add(inv0, reals[i]);
                    imags[i] = eta.mul_add(inv0, imags[i]);
                }
                let d1 = li - lj1;
                if d1.abs() <= cut {
                    let denom1 = d1.mul_add(d1, eta_sq);
                    let inv1 = 1.0 / denom1;
                    reals[i] = d1.mul_add(inv1, reals[i]);
                    imags[i] = eta.mul_add(inv1, imags[i]);
                }
                let d2 = li - lj2;
                if d2.abs() <= cut {
                    let denom2 = d2.mul_add(d2, eta_sq);
                    let inv2 = 1.0 / denom2;
                    reals[i] = d2.mul_add(inv2, reals[i]);
                    imags[i] = eta.mul_add(inv2, imags[i]);
                }
                let d3 = li - lj3;
                if d3.abs() <= cut {
                    let denom3 = d3.mul_add(d3, eta_sq);
                    let inv3 = 1.0 / denom3;
                    reals[i] = d3.mul_add(inv3, reals[i]);
                    imags[i] = eta.mul_add(inv3, imags[i]);
                }
                i += 1;
            }
            j += 4;
        }
        while j < p {
            let lambda_j = eigenvalues[j];
            let mut i = block_start;
            while i + 4 <= block_end {
                let l0 = eigenvalues[i];
                let l1 = eigenvalues[i + 1];
                let l2 = eigenvalues[i + 2];
                let l3 = eigenvalues[i + 3];

                let d0 = l0 - lambda_j;
                if d0.abs() <= cut {
                    let denom0 = d0.mul_add(d0, eta_sq);
                    let inv0 = 1.0 / denom0;
                    reals[i] = d0.mul_add(inv0, reals[i]);
                    imags[i] = eta.mul_add(inv0, imags[i]);
                }
                let d1 = l1 - lambda_j;
                if d1.abs() <= cut {
                    let denom1 = d1.mul_add(d1, eta_sq);
                    let inv1 = 1.0 / denom1;
                    reals[i + 1] = d1.mul_add(inv1, reals[i + 1]);
                    imags[i + 1] = eta.mul_add(inv1, imags[i + 1]);
                }
                let d2 = l2 - lambda_j;
                if d2.abs() <= cut {
                    let denom2 = d2.mul_add(d2, eta_sq);
                    let inv2 = 1.0 / denom2;
                    reals[i + 2] = d2.mul_add(inv2, reals[i + 2]);
                    imags[i + 2] = eta.mul_add(inv2, imags[i + 2]);
                }
                let d3 = l3 - lambda_j;
                if d3.abs() <= cut {
                    let denom3 = d3.mul_add(d3, eta_sq);
                    let inv3 = 1.0 / denom3;
                    reals[i + 3] = d3.mul_add(inv3, reals[i + 3]);
                    imags[i + 3] = eta.mul_add(inv3, imags[i + 3]);
                }
                i += 4;
            }
            while i < block_end {
                let diff = eigenvalues[i] - lambda_j;
                if diff.abs() <= cut {
                    let denom = diff.mul_add(diff, eta_sq);
                    let inv_denom = 1.0 / denom;
                    reals[i] = diff.mul_add(inv_denom, reals[i]);
                    imags[i] = eta.mul_add(inv_denom, imags[i]);
                }
                i += 1;
            }
            j += 1;
        }
    }
}

/// Float32 tiled inner loop without far-field cutoff (exact).
#[inline(always)]
fn tiled_inner_loop_no_cutoff_f32(
    eigenvalues: &[f32],
    reals: &mut [f32],
    imags: &mut [f32],
    bs: usize,
    eta: f32,
    eta_sq: f32,
) {
    let p = eigenvalues.len();
    for block_start in (0..p).step_by(bs) {
        let block_end = (block_start + bs).min(p);
        let mut j = 0;
        while j + 4 <= p {
            let lj0 = eigenvalues[j];
            let lj1 = eigenvalues[j + 1];
            let lj2 = eigenvalues[j + 2];
            let lj3 = eigenvalues[j + 3];
            let mut i = block_start;
            while i + 4 <= block_end {
                let l0 = eigenvalues[i];
                let l1 = eigenvalues[i + 1];
                let l2 = eigenvalues[i + 2];
                let l3 = eigenvalues[i + 3];

                let d00 = l0 - lj0;
                let denom00 = d00.mul_add(d00, eta_sq);
                let inv00 = 1.0 / denom00;
                reals[i] = d00.mul_add(inv00, reals[i]);
                imags[i] = eta.mul_add(inv00, imags[i]);

                let d10 = l1 - lj0;
                let denom10 = d10.mul_add(d10, eta_sq);
                let inv10 = 1.0 / denom10;
                reals[i + 1] = d10.mul_add(inv10, reals[i + 1]);
                imags[i + 1] = eta.mul_add(inv10, imags[i + 1]);

                let d20 = l2 - lj0;
                let denom20 = d20.mul_add(d20, eta_sq);
                let inv20 = 1.0 / denom20;
                reals[i + 2] = d20.mul_add(inv20, reals[i + 2]);
                imags[i + 2] = eta.mul_add(inv20, imags[i + 2]);

                let d30 = l3 - lj0;
                let denom30 = d30.mul_add(d30, eta_sq);
                let inv30 = 1.0 / denom30;
                reals[i + 3] = d30.mul_add(inv30, reals[i + 3]);
                imags[i + 3] = eta.mul_add(inv30, imags[i + 3]);

                let d01 = l0 - lj1;
                let denom01 = d01.mul_add(d01, eta_sq);
                let inv01 = 1.0 / denom01;
                reals[i] = d01.mul_add(inv01, reals[i]);
                imags[i] = eta.mul_add(inv01, imags[i]);

                let d11 = l1 - lj1;
                let denom11 = d11.mul_add(d11, eta_sq);
                let inv11 = 1.0 / denom11;
                reals[i + 1] = d11.mul_add(inv11, reals[i + 1]);
                imags[i + 1] = eta.mul_add(inv11, imags[i + 1]);

                let d21 = l2 - lj1;
                let denom21 = d21.mul_add(d21, eta_sq);
                let inv21 = 1.0 / denom21;
                reals[i + 2] = d21.mul_add(inv21, reals[i + 2]);
                imags[i + 2] = eta.mul_add(inv21, imags[i + 2]);

                let d31 = l3 - lj1;
                let denom31 = d31.mul_add(d31, eta_sq);
                let inv31 = 1.0 / denom31;
                reals[i + 3] = d31.mul_add(inv31, reals[i + 3]);
                imags[i + 3] = eta.mul_add(inv31, imags[i + 3]);

                let d02 = l0 - lj2;
                let denom02 = d02.mul_add(d02, eta_sq);
                let inv02 = 1.0 / denom02;
                reals[i] = d02.mul_add(inv02, reals[i]);
                imags[i] = eta.mul_add(inv02, imags[i]);

                let d12 = l1 - lj2;
                let denom12 = d12.mul_add(d12, eta_sq);
                let inv12 = 1.0 / denom12;
                reals[i + 1] = d12.mul_add(inv12, reals[i + 1]);
                imags[i + 1] = eta.mul_add(inv12, imags[i + 1]);

                let d22 = l2 - lj2;
                let denom22 = d22.mul_add(d22, eta_sq);
                let inv22 = 1.0 / denom22;
                reals[i + 2] = d22.mul_add(inv22, reals[i + 2]);
                imags[i + 2] = eta.mul_add(inv22, imags[i + 2]);

                let d32 = l3 - lj2;
                let denom32 = d32.mul_add(d32, eta_sq);
                let inv32 = 1.0 / denom32;
                reals[i + 3] = d32.mul_add(inv32, reals[i + 3]);
                imags[i + 3] = eta.mul_add(inv32, imags[i + 3]);

                let d03 = l0 - lj3;
                let denom03 = d03.mul_add(d03, eta_sq);
                let inv03 = 1.0 / denom03;
                reals[i] = d03.mul_add(inv03, reals[i]);
                imags[i] = eta.mul_add(inv03, imags[i]);

                let d13 = l1 - lj3;
                let denom13 = d13.mul_add(d13, eta_sq);
                let inv13 = 1.0 / denom13;
                reals[i + 1] = d13.mul_add(inv13, reals[i + 1]);
                imags[i + 1] = eta.mul_add(inv13, imags[i + 1]);

                let d23 = l2 - lj3;
                let denom23 = d23.mul_add(d23, eta_sq);
                let inv23 = 1.0 / denom23;
                reals[i + 2] = d23.mul_add(inv23, reals[i + 2]);
                imags[i + 2] = eta.mul_add(inv23, imags[i + 2]);

                let d33 = l3 - lj3;
                let denom33 = d33.mul_add(d33, eta_sq);
                let inv33 = 1.0 / denom33;
                reals[i + 3] = d33.mul_add(inv33, reals[i + 3]);
                imags[i + 3] = eta.mul_add(inv33, imags[i + 3]);

                i += 4;
            }
            while i < block_end {
                let li = eigenvalues[i];
                let d0 = li - lj0;
                let denom0 = d0.mul_add(d0, eta_sq);
                let inv0 = 1.0 / denom0;
                reals[i] = d0.mul_add(inv0, reals[i]);
                imags[i] = eta.mul_add(inv0, imags[i]);

                let d1 = li - lj1;
                let denom1 = d1.mul_add(d1, eta_sq);
                let inv1 = 1.0 / denom1;
                reals[i] = d1.mul_add(inv1, reals[i]);
                imags[i] = eta.mul_add(inv1, imags[i]);

                let d2 = li - lj2;
                let denom2 = d2.mul_add(d2, eta_sq);
                let inv2 = 1.0 / denom2;
                reals[i] = d2.mul_add(inv2, reals[i]);
                imags[i] = eta.mul_add(inv2, imags[i]);

                let d3 = li - lj3;
                let denom3 = d3.mul_add(d3, eta_sq);
                let inv3 = 1.0 / denom3;
                reals[i] = d3.mul_add(inv3, reals[i]);
                imags[i] = eta.mul_add(inv3, imags[i]);
                i += 1;
            }
            j += 4;
        }
        while j < p {
            let lambda_j = eigenvalues[j];
            let mut i = block_start;
            while i + 4 <= block_end {
                let l0 = eigenvalues[i];
                let l1 = eigenvalues[i + 1];
                let l2 = eigenvalues[i + 2];
                let l3 = eigenvalues[i + 3];

                let d0 = l0 - lambda_j;
                let denom0 = d0.mul_add(d0, eta_sq);
                let inv0 = 1.0 / denom0;
                reals[i] = d0.mul_add(inv0, reals[i]);
                imags[i] = eta.mul_add(inv0, imags[i]);

                let d1 = l1 - lambda_j;
                let denom1 = d1.mul_add(d1, eta_sq);
                let inv1 = 1.0 / denom1;
                reals[i + 1] = d1.mul_add(inv1, reals[i + 1]);
                imags[i + 1] = eta.mul_add(inv1, imags[i + 1]);

                let d2 = l2 - lambda_j;
                let denom2 = d2.mul_add(d2, eta_sq);
                let inv2 = 1.0 / denom2;
                reals[i + 2] = d2.mul_add(inv2, reals[i + 2]);
                imags[i + 2] = eta.mul_add(inv2, imags[i + 2]);

                let d3 = l3 - lambda_j;
                let denom3 = d3.mul_add(d3, eta_sq);
                let inv3 = 1.0 / denom3;
                reals[i + 3] = d3.mul_add(inv3, reals[i + 3]);
                imags[i + 3] = eta.mul_add(inv3, imags[i + 3]);

                i += 4;
            }
            while i < block_end {
                let diff = eigenvalues[i] - lambda_j;
                let denom = diff.mul_add(diff, eta_sq);
                let inv_denom = 1.0 / denom;
                reals[i] = diff.mul_add(inv_denom, reals[i]);
                imags[i] = eta.mul_add(inv_denom, imags[i]);
                i += 1;
            }
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stieltjes::autovec::autovec_stieltjes_sum;

    #[test]
    fn test_symmetric_all_points_matches_reference() {
        // The sequential no-cutoff tiled path is the symmetric half-work
        // kernel: each unordered pair visited once, antisymmetric real part,
        // symmetric imaginary part, diagonal folded into the init. It must
        // reproduce the per-row autovec sum for tiny sizes, unsorted input,
        // and the p=1 pure-diagonal edge.
        let evals = vec![3.0, -1.5, 2.25, 0.7, 4.2, -0.3, 1.1];
        let eta = 0.37;

        let (reals, imags) = compute_all_stieltjes_blocked_tiled(&evals, eta, None, None);
        for (i, &li) in evals.iter().enumerate() {
            let (ref_r, ref_i) = autovec_stieltjes_sum(li, &evals, eta);
            assert!(
                (reals[i] - ref_r).abs() < 1e-12,
                "sym real mismatch at {i}: {} vs {ref_r}",
                reals[i]
            );
            assert!(
                (imags[i] - ref_i).abs() < 1e-12,
                "sym imag mismatch at {i}: {} vs {ref_i}",
                imags[i]
            );
        }

        // p=1: only the diagonal term survives: Re = 0, Im = η/η² = 1/η.
        let (r1, i1) = compute_all_stieltjes_blocked_tiled(&[2.0], 0.5, None, None);
        assert_eq!(r1[0], 0.0);
        assert!((i1[0] - 2.0).abs() < 1e-15);

        // Symmetric (sequential) vs full-square (parallel chunked) on a
        // larger unsorted sample: agreement far inside the dispatch-level
        // tolerances used by the cross-method tests.
        let mut big: Vec<f64> = (0..1000)
            .map(|i| ((i * 7919) % 9973) as f64 / 9973.0 * 4.0 - 2.0)
            .collect();
        big.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta_b = 0.11;
        let (sr, si) = compute_all_stieltjes_blocked_tiled(&big, eta_b, None, None);
        let (pr, pi) = compute_all_stieltjes_blocked_tiled_parallel(&big, eta_b, None, None);
        for i in 0..big.len() {
            let scale = sr[i].abs().max(si[i].abs()).max(1e-12);
            let diff = ((pr[i] - sr[i]).abs() + ((pi[i] - si[i]).abs())) / scale;
            assert!(diff < 1e-12, "sym vs full-square mismatch at {i}: {diff}");
        }
    }

    #[test]
    fn test_blocked_matches_autovec_small() {
        let evals: Vec<f64> = (0..50).map(|i| (i as f64 + 0.5).ln_1p()).collect();
        let eta = 0.05;

        let (reals, imags) = compute_all_stieltjes_blocked(&evals, eta, Some(16), None);

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
    fn test_blocked_at_points_matches_single_point() {
        // The query-point blocked kernel must agree with the single-point
        // autovec kernel at every query point (same math, batched structure).
        let p = 300;
        let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        // Query points are a uniform grid (like the deconvolution grid), NOT
        // the sample eigenvalues.
        let lo = evals[0] - 0.5;
        let hi = evals[p - 1] + 0.5;
        let nq = 200;
        let query: Vec<f64> = (0..nq)
            .map(|k| lo + (hi - lo) * (k as f64) / (nq as f64 - 1.0))
            .collect();

        for &cut in &[None, Some(10.0)] {
            let (br, bi) = compute_stieltjes_blocked_at_points(&query, &evals, eta, None, cut);
            for (i, &q) in query.iter().enumerate() {
                // Reference: the single-point kernel with the SAME cutoff
                // semantics (None = all terms, Some = skip far-field terms).
                let (rr, ri) = stieltjes_sum_cutoff(q, &evals, eta, cut);
                assert!(
                    (br[i] - rr).abs() < 1e-12,
                    "Real mismatch at {i} (cut={cut:?}): {} vs {}",
                    br[i],
                    rr
                );
                assert!(
                    (bi[i] - ri).abs() < 1e-12,
                    "Imag mismatch at {i} (cut={cut:?}): {} vs {}",
                    bi[i],
                    ri
                );
            }
        }
    }

    #[test]
    fn test_tiled_matches_blocked() {
        // The tiled variant must produce identical results to the original
        // blocked variant (same math, different loop order).
        let p = 300;
        let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        for &cut in &[None, Some(10.0)] {
            let (tiled_r, tiled_i) =
                compute_all_stieltjes_blocked_tiled(&evals, eta, Some(64), cut);
            let (ref_r, ref_i) = compute_all_stieltjes_blocked(&evals, eta, Some(64), cut);
            for i in 0..p {
                assert!(
                    (tiled_r[i] - ref_r[i]).abs() < 1e-12,
                    "Real mismatch at {i} (cut={cut:?}): {} vs {}",
                    tiled_r[i],
                    ref_r[i]
                );
                assert!(
                    (tiled_i[i] - ref_i[i]).abs() < 1e-12,
                    "Imag mismatch at {i} (cut={cut:?}): {} vs {}",
                    tiled_i[i],
                    ref_i[i]
                );
            }
        }
    }

    #[test]
    fn test_tiled_auto_block_size() {
        // The auto block size should be smaller for larger p (cache pressure).
        assert!(auto_tiled_block_size(100) >= auto_tiled_block_size(1000));
        assert!(auto_tiled_block_size(1000) >= auto_tiled_block_size(10000));
        // All returned sizes must be multiples of 4 (required by the unrolled loop).
        for p in [100, 1000, 10000, 100000] {
            assert_eq!(auto_tiled_block_size(p) % 4, 0);
        }
    }

    #[test]
    fn test_tiled_auto_matches_explicit() {
        // Using None (auto) must produce the same result as using the
        // auto-selected block size explicitly.
        let p = 5000;
        let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        let (auto_r, auto_i) = compute_all_stieltjes_blocked_tiled(&evals, eta, None, None);
        let bs = auto_tiled_block_size(p);
        let (exp_r, exp_i) = compute_all_stieltjes_blocked_tiled(&evals, eta, Some(bs), None);
        for i in 0..p {
            assert_eq!(auto_r[i], exp_r[i]);
            assert_eq!(auto_i[i], exp_i[i]);
        }
    }

    #[test]
    fn test_blocked_with_cutoff_large() {
        let p = 5000;
        let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        let (reals, imags) = compute_all_stieltjes_blocked(&evals, eta, Some(64), Some(10.0));

        // Check that results are finite and plausible
        for (r, i) in reals.iter().zip(imags.iter()).take(10) {
            assert!(r.is_finite());
            assert!(i.is_finite());
            assert!(*i > 0.0); // Imaginary part should be positive
        }
    }

    #[test]
    fn test_f32_matches_f64_within_tolerance() {
        // The f32 kernel must be close to the f64 kernel (within ~1e-2,
        // the documented f32 precision), and much faster.
        let p = 500;
        let mut evals64: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        evals64.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta64 = 0.1 / (p as f64).sqrt();

        let (r64, i64) = compute_all_stieltjes_blocked_tiled(&evals64, eta64, None, None);

        let evals32: Vec<f32> = evals64.iter().map(|&x| x as f32).collect();
        let eta32 = eta64 as f32;
        let (r32, i32) = compute_all_stieltjes_blocked_tiled_f32(&evals32, eta32, None, None);

        let mut max_err: f64 = 0.0;
        for i in 0..p {
            let scale_r = r64[i].abs().max(1e-12);
            let scale_i = i64[i].abs().max(1e-12);
            max_err = max_err.max((r64[i] - r32[i] as f64).abs() / scale_r);
            max_err = max_err.max((i64[i] - i32[i] as f64).abs() / scale_i);
        }
        // f32 has ~1e-2 relative error; assert it's within 5e-2.
        assert!(
            max_err < 5e-2,
            "f32 vs f64 max rel err too large: {max_err}"
        );
    }

    #[test]
    fn test_windowed_matches_blocked_cutoff() {
        // The windowed variant must agree with the branch-based cutoff variant
        // (both skip the same far-field terms).
        let p = 2000;
        let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();
        let cut = 10.0;

        let (win_r, win_i) =
            compute_all_stieltjes_blocked_windowed(&evals, eta, Some(64), Some(cut));
        let (ref_r, ref_i) = compute_all_stieltjes_blocked(&evals, eta, Some(64), Some(cut));

        for i in 0..p {
            assert!(
                (win_r[i] - ref_r[i]).abs() < 1e-12,
                "Real mismatch at {i}: {} vs {}",
                win_r[i],
                ref_r[i]
            );
            assert!(
                (win_i[i] - ref_i[i]).abs() < 1e-12,
                "Imag mismatch at {i}: {} vs {}",
                win_i[i],
                ref_i[i]
            );
        }
    }

    #[test]
    fn test_windowed_parallel_matches_sequential() {
        let p = 1000;
        let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        let (seq_r, seq_i) =
            compute_all_stieltjes_blocked_windowed(&evals, eta, Some(64), Some(10.0));
        let (par_r, par_i) =
            compute_all_stieltjes_blocked_windowed_parallel(&evals, eta, Some(64), Some(10.0));

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

    #[test]
    fn test_windowed_accuracy_vs_exact() {
        // Quantify the error introduced by the far-field cutoff.
        //
        // IMPORTANT: the real part Re[S] = Σ (λᵢ-λⱼ)/((λᵢ-λⱼ)²+η²) decays as
        // 1/r (long-range), so a hard cutoff destroys it. The imaginary part
        // Im[S] = Σ η/((λᵢ-λⱼ)²+η²) decays as 1/r² (short-range) and truncates
        // cleanly. This test documents that the windowed method is only
        // accurate for the imaginary part.
        let p = 2000;
        let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        let (exact_r, exact_i) = compute_all_stieltjes_blocked(&evals, eta, Some(64), None);
        let (win_r, win_i) =
            compute_all_stieltjes_blocked_windowed(&evals, eta, Some(64), Some(10.0));

        let mut max_rel_r = 0.0_f64;
        let mut max_rel_i = 0.0_f64;
        for i in 0..p {
            let scale_r = exact_r[i].abs().max(1e-12);
            let scale_i = exact_i[i].abs().max(1e-12);
            max_rel_r = max_rel_r.max((win_r[i] - exact_r[i]).abs() / scale_r);
            max_rel_i = max_rel_i.max((win_i[i] - exact_i[i]).abs() / scale_i);
        }
        eprintln!("max relative error real={max_rel_r:.4} imag={max_rel_i:.4}");
        // The imaginary part is dominated by near-field terms, so the cutoff
        // error should be well under 10%. The real part is long-range (1/r),
        // so it is NOT accurately truncated — we only assert on the imaginary part.
        assert!(
            max_rel_i < 0.10,
            "Windowed imaginary relative error too large: {max_rel_i}"
        );
    }
}
