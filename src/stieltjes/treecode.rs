//! O(p log p) 1D tree code / Fast Multipole Method (FMM) for the Stieltjes
//! transform.
//!
//! The Stieltjes term is `1/((λᵢ-λⱼ) - iη) = 1/(z - λⱼ)` with `z = λᵢ - iη`.
//! This is exactly the 2D Coulomb/log-potential kernel in the complex plane —
//! the classic FMM setting. For a cluster of eigenvalues `{x_k}` centered at
//! `μ`, the far-field contribution at query `z` is expanded in multipoles:
//!
//! ```text
//!   Σ_k 1/(z - x_k) = Σ_{n=0}^{P} M_n / (z - μ)^{n+1},
//!   M_n = Σ_k (x_k - μ)^n
//! ```
//!
//! This is an *exact* geometric-series expansion that converges when
//! `|z - μ| > max_k |x_k - μ|`. Because the eigenvalues are real, the moments
//! `M_n` are real; only the evaluation `M_n/(z-μ)^{n+1}` is complex. This
//! handles **both** the real (Hilbert, 1/d) and imaginary (Lorentzian, 1/d²)
//! parts accurately — unlike a naive center-of-mass (order-0) treecode, which
//! only captures the imaginary part well.
//!
//! # Performance design
//!
//! - **Flat, structure-of-arrays tree** (parallel `Vec`s) instead of a
//!   recursive `Box<TreeNode>` — cache-friendly, iterative traversal.
//! - **Higher-order multipole moments** (order `P`) with correct moment
//!   shifting via binomial coefficients.
//! - **FMA-friendly arithmetic**: the opening-angle test is done in squared
//!   form (no `sqrt`), and the complex evaluation uses real/imag parts
//!   directly (no `Complex64::norm()` / complex division).
//! - **Reusable stack** across queries; optional **Rayon parallelism** over
//!   the independent query points.

use rayon::prelude::*;

/// Flat, structure-of-arrays binary tree.
///
/// Node `i` has bounds `[lo[i], hi[i]]`, `count[i]` eigenvalues with center of
/// mass `mu[i]`, multipole moments `mom[i*(P+1) .. i*(P+1)+P]`, and children
/// `left[i]` / `right[i]` (`-1` = none). The root is node `0`.
struct FlatTree {
    lo: Vec<f64>,
    hi: Vec<f64>,
    count: Vec<usize>,
    mu: Vec<f64>,
    /// Multipole moments, flattened: `mom[node*(order+1) + n]`.
    mom: Vec<f64>,
    left: Vec<i32>,
    right: Vec<i32>,
    order: usize,
}

impl FlatTree {
    /// Compute the Stieltjes contribution of the whole tree at query point
    /// `lambda_i`, using an iterative walk with a caller-provided reusable
    /// stack. Returns (real, imag) of the raw sum (not scaled by 1/p).
    #[inline]
    fn contribution(
        &self,
        lambda_i: f64,
        eta: f64,
        theta_sq: f64,
        stack: &mut Vec<i32>,
    ) -> (f64, f64) {
        let p1 = self.order + 1;
        stack.clear();
        stack.push(0);

        let mut re = 0.0;
        let mut im = 0.0;

        while let Some(node) = stack.pop() {
            let n = node as usize;
            let count = self.count[n];
            if count == 0 {
                continue;
            }

            let mu = self.mu[n];
            let diff = lambda_i - mu;
            let dist_sq = diff * diff + eta * eta;

            let is_leaf = self.left[n] < 0 && self.right[n] < 0;
            let cluster_size = self.hi[n] - self.lo[n];

            if is_leaf || cluster_size * cluster_size < theta_sq * dist_sq {
                // Leaf (exact) or far cluster (multipole expansion).
                // w = 1/(z - mu) = 1/(diff - i·eta) = (diff + i·eta)/dist_sq
                let inv = 1.0 / dist_sq;
                let wr = diff * inv;
                let wi = eta * inv;

                // s = Σ_{k=0}^{P} M_k · w^{k+1}
                let base = n * p1;
                let mut wpr = wr; // w^1
                let mut wpi = wi;
                for k in 0..p1 {
                    let mk = self.mom[base + k];
                    re += mk * wpr;
                    im += mk * wpi;
                    // wp *= w
                    let nr = wpr * wr - wpi * wi;
                    let ni = wpr * wi + wpi * wr;
                    wpr = nr;
                    wpi = ni;
                }
            } else {
                // Close cluster: descend into children.
                let r = self.right[n];
                let l = self.left[n];
                if r >= 0 {
                    stack.push(r);
                }
                if l >= 0 {
                    stack.push(l);
                }
            }
        }

        (re, im)
    }
}

/// Build a flat balanced tree from a sorted slice via divide-and-conquer,
/// computing multipole moments about each node's center of mass.
/// `binom` is a flat `(P+1)×(P+1)` Pascal triangle: `binom[n*(P+1)+k] = C(n,k)`.
fn build_flat(sorted: &[f64], lo: f64, hi: f64, tree: &mut FlatTree, binom: &[f64]) -> i32 {
    if sorted.is_empty() {
        return -1;
    }

    let p1 = tree.order + 1;
    let idx = tree.lo.len() as i32;
    tree.lo.push(lo);
    tree.hi.push(hi);
    tree.count.push(0);
    tree.mu.push(0.0);
    for _ in 0..p1 {
        tree.mom.push(0.0);
    }
    tree.left.push(-1);
    tree.right.push(-1);

    if sorted.len() == 1 {
        let n = idx as usize;
        tree.count[n] = 1;
        tree.mu[n] = sorted[0];
        tree.mom[n * p1] = 1.0; // M_0 = 1
        return idx;
    }

    let mid = (lo + hi) * 0.5;
    let mut split = sorted.partition_point(|&x| x <= mid);
    // Guard against degenerate splits (duplicates / rounding).
    if split == 0 || split == sorted.len() {
        split = sorted.len() / 2;
    }
    let (ls, rs) = sorted.split_at(split);

    let li = if ls.is_empty() {
        -1
    } else {
        build_flat(ls, lo, mid, tree, binom)
    };
    let ri = if rs.is_empty() {
        -1
    } else {
        build_flat(rs, mid, hi, tree, binom)
    };

    let n = idx as usize;
    tree.left[n] = li;
    tree.right[n] = ri;

    let cnt_l = if li >= 0 { tree.count[li as usize] } else { 0 };
    let cnt_r = if ri >= 0 { tree.count[ri as usize] } else { 0 };
    tree.count[n] = cnt_l + cnt_r;

    let mu_l = if li >= 0 { tree.mu[li as usize] } else { 0.0 };
    let mu_r = if ri >= 0 { tree.mu[ri as usize] } else { 0.0 };
    let mu = if tree.count[n] > 0 {
        (mu_l * cnt_l as f64 + mu_r * cnt_r as f64) / (tree.count[n] as f64)
    } else {
        0.0
    };
    tree.mu[n] = mu;

    // Shift child moments to this node's center of mass.
    // M_new[n] = Σ_{k=0}^{n} C(n,k) · M_old[k] · (-Δ)^(n-k),  Δ = μ_new - μ_old
    let base = n * p1;
    for child in [li, ri] {
        if child < 0 {
            continue;
        }
        let c = child as usize;
        let cbase = c * p1;
        let delta = mu - tree.mu[c];
        for nn in 0..p1 {
            let mut acc = 0.0;
            for k in 0..=nn {
                let term =
                    tree.mom[cbase + k] * binom[nn * p1 + k] * (-delta).powi((nn - k) as i32);
                acc += term;
            }
            tree.mom[base + nn] += acc;
        }
    }
    idx
}

/// Compute all Stieltjes transforms using a 1D tree code / FMM.
///
/// Returns **raw sums** (not scaled by `1/p`); the caller applies scaling.
///
/// # Arguments
/// Dispatch default opening angle and multipole order of the
/// `TreeCode` variant (~5e-4 relative error class on both parts). Named
/// here so the dispatcher carries no magic numbers.
pub(crate) const DEFAULT_THETA: f64 = 0.5;
/// Dispatch default multipole order.
pub(crate) const DEFAULT_ORDER: usize = 6;

/// Treecode/FMM with explicit opening-angle `theta`, multipole `order`, and
/// optional Rayon parallelism.
///
/// Returns **raw sums** (not scaled by `1/p`); the caller applies scaling.
///
/// Smaller `theta` / larger `order` = more accurate but more work.
/// `parallel = true` parallelizes over query points (each is independent).
pub fn compute_all_stieltjes_treecode_impl(
    eigenvalues: &[f64],
    eta: f64,
    theta: f64,
    order: usize,
    parallel: bool,
) -> Vec<(f64, f64)> {
    let p = eigenvalues.len();
    if p == 0 {
        return Vec::new();
    }

    // The tree is built over the multiset of eigenvalues, which must be
    // sorted for the divide-and-conquer split to be correct. The input is
    // not guaranteed sorted, so sort a copy (results are still returned in
    // the original query order) — skipped when already sorted.
    let mut sorted_buf: Vec<f64>;
    let sorted: &[f64] = if super::is_sorted_ascending(eigenvalues) {
        eigenvalues
    } else {
        sorted_buf = eigenvalues.to_vec();
        sorted_buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        &sorted_buf
    };

    let lo = sorted[0];
    let hi = sorted[p - 1];
    let padding = (hi - lo).max(1.0) * 0.1;

    // Pascal triangle for binomial coefficients, flat (P+1)×(P+1).
    let p1 = order + 1;
    let mut binom = vec![0.0_f64; p1 * p1];
    for n in 0..p1 {
        binom[n * p1] = 1.0;
        binom[n * p1 + n] = 1.0;
        for k in 1..n {
            binom[n * p1 + k] = binom[(n - 1) * p1 + k - 1] + binom[(n - 1) * p1 + k];
        }
    }

    let mut tree = FlatTree {
        lo: Vec::with_capacity(2 * p),
        hi: Vec::with_capacity(2 * p),
        count: Vec::with_capacity(2 * p),
        mu: Vec::with_capacity(2 * p),
        mom: Vec::with_capacity(2 * p * p1),
        left: Vec::with_capacity(2 * p),
        right: Vec::with_capacity(2 * p),
        order,
    };
    build_flat(sorted, lo - padding, hi + padding, &mut tree, &binom);

    let theta_sq = theta * theta;

    if parallel {
        eigenvalues
            .par_iter()
            .map(|&lambda_i| {
                let mut stack = Vec::with_capacity(64);
                tree.contribution(lambda_i, eta, theta_sq, &mut stack)
            })
            .collect()
    } else {
        let mut stack = Vec::with_capacity(64);
        let mut result = Vec::with_capacity(p);
        for &lambda_i in eigenvalues {
            let (re, im) = tree.contribution(lambda_i, eta, theta_sq, &mut stack);
            result.push((re, im));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stieltjes::autovec::autovec_stieltjes_sum;

    #[test]
    fn test_treecode_agrees_with_autovec() {
        let p = 1500;
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        let eta = 0.15;

        // Both treecode and autovec return raw sums (not scaled by 1/p).
        let tc_results =
            compute_all_stieltjes_treecode_impl(&evals, eta, DEFAULT_THETA, DEFAULT_ORDER, false);
        let ref_results: Vec<(f64, f64)> = evals
            .iter()
            .take(100)
            .map(|&li| autovec_stieltjes_sum(li, &evals, eta))
            .collect();

        for (i, (tc_r, tc_i)) in tc_results.iter().take(100).enumerate() {
            let (ref_r, ref_i) = ref_results[i];
            let tol = 0.5;
            assert!(
                (tc_r - ref_r).abs() < tol,
                "Treecode real mismatch at {i}: {tc_r} vs {ref_r}"
            );
            assert!(
                (tc_i - ref_i).abs() < tol,
                "Treecode imag mismatch at {i}: {tc_i} vs {ref_i}"
            );
        }
    }

    #[test]
    fn test_parallel_matches_sequential() {
        let p = 2000;
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        let eta = 0.15;

        let seq = compute_all_stieltjes_treecode_impl(&evals, eta, 0.5, 6, false);
        let par = compute_all_stieltjes_treecode_impl(&evals, eta, 0.5, 6, true);

        for i in 0..p {
            assert!(
                (seq[i].0 - par[i].0).abs() < 1e-14,
                "Real mismatch at {i}: {} vs {}",
                seq[i].0,
                par[i].0
            );
            assert!(
                (seq[i].1 - par[i].1).abs() < 1e-14,
                "Imag mismatch at {i}: {} vs {}",
                seq[i].1,
                par[i].1
            );
        }
    }

    #[test]
    fn test_multipole_accuracy() {
        // Verify the multipole treecode is accurate on BOTH real and imaginary
        // parts (the real part is the long-range 1/d Hilbert kernel that a
        // naive center-of-mass treecode gets wrong).
        let p = 512;
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        let eta = 0.1 / (p as f64).sqrt();

        let tc = compute_all_stieltjes_treecode_impl(&evals, eta, 0.3, 8, false);

        // Exact reference (raw sums, matching treecode's raw output).
        let mut max_err_r = 0.0_f64;
        let mut max_err_i = 0.0_f64;
        for i in 0..p {
            let li = evals[i];
            let mut sr = 0.0;
            let mut si = 0.0;
            for &lj in &evals {
                let d = li - lj;
                let denom = d * d + eta * eta;
                sr += d / denom;
                si += eta / denom;
            }
            max_err_r = max_err_r.max((tc[i].0 - sr).abs() / sr.abs().max(1e-12));
            max_err_i = max_err_i.max((tc[i].1 - si).abs() / si.abs().max(1e-12));
        }
        eprintln!("multipole real max rel err: {max_err_r:.6}, imag: {max_err_i:.6}");
        assert!(
            max_err_r < 1e-3,
            "multipole real error too large: {max_err_r}"
        );
        assert!(
            max_err_i < 1e-3,
            "multipole imag error too large: {max_err_i}"
        );
    }
}
