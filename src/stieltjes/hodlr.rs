//! Hierarchical low-rank (HODLR) summation for the Stieltjes transform.
//!
//! # Paradigm — algebraic, self-validating, kernel-agnostic
//!
//! The Stieltjes row sums are `Sᵢ = Σ_j K(λᵢ, λⱼ)` with the Cauchy/Lorentz
//! kernel `K(x, y) = 1/((x − y) − iη)`. Collect them as `K·1`: a matrix–vector
//! product against the all-ones right-hand side. The previous fast methods
//! compress this product *analytically* — FFT grids exploit translation
//! structure; ChebCode interpolates equivalent densities; the rejected FMM
//! attempt showed that analytic translations of this off-axis pole are
//! numerically unstable in 1D.
//!
//! HODLR takes the opposite road: partition the sorted index set into a
//! balanced binary tree and observe that every **off-diagonal block**
//! `K(A, B)` with A, B separated index ranges is *numerically low rank* —
//! provably for asymptotically smooth kernels, with the rank growing only
//! logarithmically in (block distance)/(block size). Each block is compressed
//! by **adaptive cross approximation** (ACA): alternating skeleton rows and
//! columns pivoted at the largest residual entry, stopped when the standard
//! residual-product estimator drops below the tolerance. ACA touches only
//! actual kernel entries and validates itself block by block — there is no
//! extrapolation that can blow up, no opening-angle parameter to tune.
//!
//! # Assembly
//!
//! Recursion over a node's index range `[a, b)` split at `(a+b)/2`:
//!
//! ```text
//!   sums(L ← L∪R) = sums(L ← L) + K[L,R]·1      // cross term via factors
//!   sums(R ← L∪R) = sums(R ← R) + K[R,L]·1
//! ```
//!
//! Leaves compute their diagonal blocks exactly. Cross terms use the ACA
//! factors: `K·1 ≈ U·(V·1)` — two skinny products, `O(rank·(m+n))`.
//!
//! # Cost model
//!
//! Exact work: `Σ_leaves m² = p·leaf`. Compression: per sibling pair,
//! `O(rank·(m+n))` kernel evaluations plus deflation arithmetic; summed over
//! the `log₂(p/leaf)` levels: `O(rank·p·log p)`. Factors live only during
//! their own level pass, so peak memory is `O(rank·p)`.
//!
//! # Numerical notes
//!
//! - The kernel is complex; pivoting uses magnitudes |z| and both parts are
//!   carried through U/V explicitly (no num_complex in the hot path).
//! - Degenerate panels (repeated eigenvalues) need no special handling:
//!   entries stay finite (`d = 0 ⇒ i/η`) and ACA simply converges faster.
//! - The stopping rule ‖u_{k+1}‖·‖v_k‖ ≤ tol·‖K̂‖_F uses the interlacing
//!   lower bound Σ_t ‖U_t‖²‖V_t‖² ≤ ‖K‖²_F for the norm estimate.

/// Adaptive cross approximation of the complex kernel block
/// `{K(tgt[i], src[j])}`.
///
/// Returns the column-major factor pair `(u_re, u_im)` of shape
/// `[rank × n_tgt]` and `(v_rows_re, v_rows_im)` of shape `[rank × n_src]`
/// (each rank row is one skeleton row), plus the achieved rank.
///
/// Standard ACA loop: pick a seed target row, pivot on the largest residual
/// entry alternately in column and row space, deflate through the growing
/// factors, stop on the Frobenius-norm residual estimator.
fn aca(
    tgt: &[f64],
    src: &[f64],
    eta: f64,
    eta_sq: f64,
    tol: f64,
    max_rank: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    let m = tgt.len();
    let n = src.len();
    debug_assert!(m > 0 && n > 0);

    let mut u_re: Vec<f64> = Vec::with_capacity(max_rank * m);
    let mut u_im: Vec<f64> = Vec::with_capacity(max_rank * m);
    let mut v_re: Vec<f64> = Vec::with_capacity(max_rank * n);
    let mut v_im: Vec<f64> = Vec::with_capacity(max_rank * n);

    // Kernel entry helper (inline closure for clarity; hot enough to keep).
    #[inline(always)]
    fn entry(x: f64, y: f64, eta: f64, eta_sq: f64) -> (f64, f64) {
        let d = x - y;
        let inv = 1.0 / (d * d + eta_sq);
        (d * inv, eta * inv)
    }

    // Seed: middle target row (fixed — later pivot rows come from the
    // residual columns, so the seed index is never re-used).
    let cur_row_i = m / 2;
    // Residual estimate of that row (no ranks yet → exact row).
    let mut row_re: Vec<f64> = Vec::with_capacity(n);
    let mut row_im: Vec<f64> = Vec::with_capacity(n);
    for &y in src.iter() {
        let (r, i) = entry(tgt[cur_row_i], y, eta, eta_sq);
        row_re.push(r);
        row_im.push(i);
    }

    let mut norm_est_sq = 0.0f64;
    let mut col_re = vec![0.0f64; m];
    let mut col_im = vec![0.0f64; m];

    // Previously used skeleton pivots. Reusing one makes the corresponding
    // residual column vanish identically (the rank-1 update reproduces that
    // column exactly), and dividing by the resulting ~machine-zero pivot
    // destroys the factors — so argmax selections must skip them.
    let mut used_cols: Vec<usize> = Vec::new();
    let mut used_rows: Vec<usize> = Vec::new();

    for _rank in 0..max_rank {
        // Pivot column: largest |row| entry among unused columns.
        let j_star = {
            let mut best = usize::MAX;
            let mut best_v = -1.0f64;
            for (k, (&r, &i)) in row_re.iter().zip(row_im.iter()).enumerate() {
                if used_cols.contains(&k) {
                    continue;
                }
                let mag = r * r + i * i;
                if mag > best_v {
                    best_v = mag;
                    best = k;
                }
            }
            if best == usize::MAX {
                break; // every column already a skeleton pivot
            }
            best
        };

        // Residual column at j_star, deflated through current factors.
        // Deflation runs rank-outer (streaming the contiguous factor rows —
        // auto-vectorizable AXPYs) rather than entry-outer with stride-m
        // gathers.
        let y_star = src[j_star];
        let rk = u_re.len() / m;
        for (i, &x) in tgt.iter().enumerate() {
            let (r, im) = entry(x, y_star, eta, eta_sq);
            col_re[i] = r;
            col_im[i] = im;
        }
        for t in 0..rk {
            let wr = v_re[t * n + j_star];
            let wi = v_im[t * n + j_star];
            let ur = &u_re[t * m..t * m + m];
            let ui = &u_im[t * m..t * m + m];
            for i in 0..m {
                col_re[i] -= ur[i] * wr - ui[i] * wi;
                col_im[i] -= ur[i] * wi + ui[i] * wr;
            }
        }
        let mut col_norm_sq = 0.0f64;
        for i in 0..m {
            col_norm_sq += col_re[i] * col_re[i] + col_im[i] * col_im[i];
        }

        // Pivot row: largest |col| entry among unused rows.
        let mut i_star = usize::MAX;
        let mut best_v = -1.0f64;
        for (i, (&r, &im)) in col_re.iter().zip(col_im.iter()).enumerate() {
            if used_rows.contains(&i) {
                continue;
            }
            let mag = r * r + im * im;
            if mag > best_v {
                best_v = mag;
                i_star = i;
            }
        }
        if i_star == usize::MAX {
            break;
        }
        let pivot = (col_re[i_star], col_im[i_star]);
        let pivot_mag2 = pivot.0 * pivot.0 + pivot.1 * pivot.1;
        if pivot_mag2 <= 0.0 || !pivot_mag2.is_finite() {
            break; // exhausted (zero or non-finite pivot)
        }

        // New residual row at i_star, deflated through current factors
        // (BEFORE appending rank k — u_new must exclude the new rank).
        let x_star = tgt[i_star];
        let mut new_row_re = vec![0.0f64; n];
        let mut new_row_im = vec![0.0f64; n];
        for (jj, &y) in src.iter().enumerate() {
            let (r, im) = entry(x_star, y, eta, eta_sq);
            new_row_re[jj] = r;
            new_row_im[jj] = im;
        }
        let mut row_norm_sq = 0.0f64;
        for t in 0..rk {
            let ur_s = u_re[t * m + i_star];
            let ui_s = u_im[t * m + i_star];
            let vr = &v_re[t * n..t * n + n];
            let vi = &v_im[t * n..t * n + n];
            for j in 0..n {
                new_row_re[j] -= ur_s * vr[j] - ui_s * vi[j];
                new_row_im[j] -= ur_s * vi[j] + ui_s * vr[j];
            }
        }
        for j in 0..n {
            row_norm_sq += new_row_re[j] * new_row_re[j] + new_row_im[j] * new_row_im[j];
        }

        // Stopping rule: the next rank-1 term would have norm
        // ‖col‖·‖row‖/|pivot|; the true remaining error tracks this "term"
        // quantity up to a small constant (measured on Cauchy blocks), while
        // the raw product ignores the pivot normalization and can stop
        // ~1e4× early. Checked BEFORE appending: breaking here means the
        // existing factors already meet the tolerance.
        if col_norm_sq.sqrt() * row_norm_sq.sqrt() / pivot_mag2.sqrt() <= tol * norm_est_sq.sqrt() {
            break;
        }

        // Append normalized rank-1 factor: U[:, k] = col/pivot, V[k, :] = row.
        let inv_p = 1.0 / pivot_mag2;
        let pr = pivot.0 * inv_p;
        let pi = -pivot.1 * inv_p; // conjugate division: col * conj(pivot)/|pivot|²
        for i in 0..m {
            let c_r = col_re[i];
            let c_i = col_im[i];
            u_re.push(c_r * pr - c_i * pi);
            u_im.push(c_r * pi + c_i * pr);
        }
        for j in 0..n {
            v_re.push(new_row_re[j]);
            v_im.push(new_row_im[j]);
        }

        // Interlacing lower bound on ‖K‖_F².
        norm_est_sq += col_norm_sq * row_norm_sq * inv_p;

        used_cols.push(j_star);
        used_rows.push(i_star);

        row_re = new_row_re;
        row_im = new_row_im;
    }

    let rank = u_re.len() / m.max(1);
    (u_re, u_im, v_re, v_im, rank)
}

/// ACA factor pair for one block: `U` is `[rank × n_tgt]` (row-major by
/// rank), `V` is `[rank × n_src]`.
struct AcaFactors {
    u_re: Vec<f64>,
    u_im: Vec<f64>,
    v_re: Vec<f64>,
    v_im: Vec<f64>,
}

/// Apply the stored factors to the all-ones vector:
/// `out[out_off..][0..m] += U·(V·1)`.
#[inline]
fn apply_cross(f: &AcaFactors, m: usize, n: usize, out: &mut [(f64, f64)], out_off: usize) {
    let u_re = &f.u_re;
    let u_im = &f.u_im;
    let v_re = &f.v_re;
    let v_im = &f.v_im;
    let rank = u_re.len() / m.max(1);
    // rhs[r] = Σ_j V[r, j]
    let mut rhs_re = vec![0.0f64; rank];
    let mut rhs_im = vec![0.0f64; rank];
    for t in 0..rank {
        let vr = &v_re[t * n..t * n + n];
        let vi = &v_im[t * n..t * n + n];
        let (mut ar, mut ai) = (0.0, 0.0);
        for k in 0..n {
            ar += vr[k];
            ai += vi[k];
        }
        rhs_re[t] = ar;
        rhs_im[t] = ai;
    }
    for t in 0..rank {
        let ur = &u_re[t * m..t * m + m];
        let ui = &u_im[t * m..t * m + m];
        let br = rhs_re[t];
        let bi = rhs_im[t];
        for i in 0..m {
            let o = &mut out[out_off + i];
            o.0 += ur[i] * br - ui[i] * bi;
            o.1 += ur[i] * bi + ui[i] * br;
        }
    }
}

/// Leaf exact sums over the node's own sources (local indexing: `out[k]`
/// corresponds to `seg[k]`; callers pass an output slice already re-based on
/// the node's index range).
fn leaf_exact(seg: &[f64], eta: f64, eta_sq: f64, out: &mut [(f64, f64)]) {
    for (k, &x) in seg.iter().enumerate() {
        let mut sr = 0.0;
        let mut si = 0.0;
        for &y in seg {
            let d = x - y;
            let inv = 1.0 / (d * d + eta_sq);
            sr += d * inv;
            si += eta * inv;
        }
        out[k] = (sr, si);
    }
}

/// Compression settings shared by every block of one run.
#[derive(Copy, Clone)]
struct HodlrSettings {
    leaf_cap: usize,
    eta: f64,
    eta_sq: f64,
    tol: f64,
    max_rank: usize,
}

/// Recursive assembly over [lo, hi). Writes raw sums into `out[lo..hi)`
/// (the slice is re-based on the range at every level).
fn rec(
    sorted: &[f64],
    lo: usize,
    hi: usize,
    s: &HodlrSettings,
    depth_budget: u32,
    out: &mut [(f64, f64)],
) {
    if hi - lo <= s.leaf_cap {
        leaf_exact(&sorted[lo..hi], s.eta, s.eta_sq, out);
        return;
    }
    let mid_val = sorted[lo] + 0.5 * (sorted[hi - 1] - sorted[lo]);
    let mut mid = sorted[lo..hi].partition_point(|&x| x <= mid_val) + lo;
    if mid == lo || mid == hi {
        mid = (lo + hi) / 2;
    }

    {
        let (out_l, out_r) = out.split_at_mut(mid - lo);
        if depth_budget > 0 {
            rayon::join(
                || rec(sorted, lo, mid, s, depth_budget - 1, out_l),
                || rec(sorted, mid, hi, s, depth_budget - 1, out_r),
            );
        } else {
            rec(sorted, lo, mid, s, 0, out_l);
            rec(sorted, mid, hi, s, 0, out_r);
        }
    }

    // Cross terms L←R and R←L (`out` is already re-based on [lo, hi)).
    let tgt_l = &sorted[lo..mid];
    let src_r = &sorted[mid..hi];
    {
        let (u_re, u_im, v_re, v_im, _rk) = aca(tgt_l, src_r, s.eta, s.eta_sq, s.tol, s.max_rank);
        let f = AcaFactors {
            u_re,
            u_im,
            v_re,
            v_im,
        };
        apply_cross(&f, tgt_l.len(), src_r.len(), out, 0);
    }
    let tgt_r = &sorted[mid..hi];
    let src_l = &sorted[lo..mid];
    {
        let (u_re, u_im, v_re, v_im, _rk) = aca(tgt_r, src_l, s.eta, s.eta_sq, s.tol, s.max_rank);
        let f = AcaFactors {
            u_re,
            u_im,
            v_re,
            v_im,
        };
        apply_cross(&f, tgt_r.len(), src_l.len(), out, mid - lo);
    }
}

/// Compute all Stieltjes transforms with the hierarchical low-rank method.
///
/// Returns **raw sums** (not scaled by `1/p`). `tol` bounds each off-diagonal
/// block's estimated relative Frobenius residual; `max_rank` caps every block
/// (adjacent blocks may saturate it — they are small near the leaves).
pub fn compute_all_stieltjes_hodlr_impl(
    eigenvalues: &[f64],
    eta: f64,
    leaf_cap: usize,
    tol: f64,
    max_rank: usize,
    parallel: bool,
) -> Vec<(f64, f64)> {
    let p = eigenvalues.len();
    if p == 0 {
        return Vec::new();
    }
    let already_sorted = super::is_sorted_ascending(eigenvalues);
    let mut sorted_buf: Vec<f64>;
    let sorted: &[f64] = if already_sorted {
        eigenvalues
    } else {
        sorted_buf = eigenvalues.to_vec();
        sorted_buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        &sorted_buf
    };

    let settings = HodlrSettings {
        leaf_cap,
        eta,
        eta_sq: eta * eta,
        tol,
        max_rank,
    };
    let mut out: Vec<(f64, f64)> = vec![(0.0, 0.0); p];
    let depth_budget = if parallel { 3 } else { 0 };
    rec(sorted, 0, p, &settings, depth_budget, &mut out);

    if already_sorted {
        out
    } else {
        let perm: Vec<u32> = {
            let mut idx: Vec<u32> = (0..p as u32).collect();
            idx.sort_unstable_by(|&i, &j| {
                eigenvalues[i as usize]
                    .partial_cmp(&eigenvalues[j as usize])
                    .unwrap()
            });
            idx
        };
        // Scatter: `out[s]` is the sum of the s-th smallest eigenvalue,
        // which came from original index `perm[s]`.
        let mut ordered = vec![(0.0f64, 0.0f64); p];
        for (s, orig) in perm.into_iter().enumerate() {
            ordered[orig as usize] = out[s];
        }
        ordered
    }
}

/// Default-settings wrapper: raw sums, sequential.
pub fn compute_all_stieltjes_hodlr(eigenvalues: &[f64], eta: f64) -> Vec<(f64, f64)> {
    compute_all_stieltjes_hodlr_impl(eigenvalues, eta, 256, 1e-9, 32, false)
}

#[cfg(test)]
mod tests {
    struct Lcg(u64);
    impl Iterator for Lcg {
        type Item = f64;
        fn next(&mut self) -> Option<f64> {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            Some((self.0 >> 11) as f64 / (1u64 << 53) as f64)
        }
    }

    /// MP-like spectrum with two outliers (benchmark harness convention).
    fn spectrum(p: usize) -> Vec<f64> {
        let c: f64 = 0.5;
        let lo = (1.0 - c.sqrt()).powi(2);
        let hi = (1.0 + c.sqrt()).powi(2);
        let mut v: Vec<f64> = Lcg(42)
            .take(p.saturating_sub(2))
            .map(|x| lo + x * (hi - lo))
            .collect();
        v.push(hi * 2.3);
        v.push(lo * 0.35);
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    #[test]
    fn hodlr_matches_exact() {
        let crate_ref = |evs: &[f64], eta: f64| -> Vec<(f64, f64)> {
            evs.iter()
                .map(|&li| {
                    let mut sr = 0.0;
                    let mut si = 0.0;
                    for &lj in evs {
                        let d = li - lj;
                        let inv = 1.0 / (d * d + eta * eta);
                        sr += d * inv;
                        si += eta * inv;
                    }
                    (sr, si)
                })
                .collect()
        };
        for p in [256usize, 1000, 4000] {
            let evs = spectrum(p);
            let eta = 1.0 / (p as f64).sqrt();
            let got = super::compute_all_stieltjes_hodlr_impl(&evs, eta, 32, 1e-10, 64, false);
            let exact = crate_ref(&evs, eta);
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..p {
                num += (got[i].0 - exact[i].0).powi(2) + (got[i].1 - exact[i].1).powi(2);
                den += exact[i].0.powi(2) + exact[i].1.powi(2);
            }
            let rel = (num / den).sqrt();
            eprintln!("hodlr p={p} rel_l2={rel:.3e}");
            assert!(rel < 5e-9, "p={p} rel {rel:.3e}");
        }
    }

    #[test]
    fn hodlr_parallel_matches_sequential() {
        let p = 2000;
        let evs = spectrum(p);
        let eta = 1.0 / (p as f64).sqrt();
        let seq = super::compute_all_stieltjes_hodlr_impl(&evs, eta, 32, 1e-10, 64, false);
        let par = super::compute_all_stieltjes_hodlr_impl(&evs, eta, 32, 1e-10, 64, true);
        for i in 0..p {
            assert!(
                (seq[i].0 - par[i].0).abs() < 1e-7 && (seq[i].1 - par[i].1).abs() < 1e-7,
                "mismatch at {i}: seq={} par={}",
                seq[i].0,
                par[i].0
            );
        }
    }

    #[test]
    fn hodlr_handles_unsorted_input() {
        // Unsorted input must return sums indexed by the ORIGINAL order and
        // match the exact O(p²) sums at those same (shuffled) points.
        let p = 500;
        let mut evs = spectrum(p);
        // Shuffle deterministically.
        let mut s = Lcg(7u64);
        for i in (1..p).rev() {
            let j = (s.next().unwrap() * (i as f64 + 1.0)) as usize % (i + 1);
            evs.swap(i, j);
        }
        let eta = 1.0 / (p as f64).sqrt();
        let got = super::compute_all_stieltjes_hodlr_impl(&evs, eta, 16, 1e-10, 64, false);
        // Ground truth per ORIGINAL index.
        let mut worst = 0.0f64;
        let mut worst_i = 0usize;
        for i in 0..p {
            let mut sr = 0.0;
            let mut si = 0.0;
            for &lj in &evs {
                let d = evs[i] - lj;
                let inv = 1.0 / (d * d + eta * eta);
                sr += d * inv;
                si += eta * inv;
            }
            let e = ((got[i].0 - sr).powi(2) + (got[i].1 - si).powi(2)).sqrt();
            if e > worst {
                worst = e;
                worst_i = i;
            }
        }
        eprintln!("worst abs err vs truth = {worst:.3e} at i={worst_i}");
        assert!(worst < 1e-4, "unsorted-vs-truth err {worst:.3e}");
    }
}

#[cfg(test)]
mod aca_tests {
    struct Lcg(u64);
    impl Iterator for Lcg {
        type Item = f64;
        fn next(&mut self) -> Option<f64> {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            Some((self.0 >> 11) as f64 / (1u64 << 53) as f64)
        }
    }

    #[test]
    fn aca_block_reconstruction() {
        // Two separated halves of an MP-like spectrum.
        let p = 256;
        let lo = (1.0 - 0.5f64.sqrt()).powi(2);
        let hi = (1.0 + 0.5f64.sqrt()).powi(2);
        let mut evs: Vec<f64> = Lcg(42).take(p).map(|x| lo + x * (hi - lo)).collect();
        evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 1.0 / (p as f64).sqrt();

        let tgt = &evs[0..64];
        let src = &evs[128..256];
        let (ur, ui, vr, vi, rank) = super::aca(tgt, src, eta, eta * eta, 1e-10, 64);
        eprintln!("rank={rank}");

        // Reconstruct and compare against the exact kernel.
        let m = tgt.len();
        let n = src.len();
        let rk = ur.len() / m;
        let mut worst = 0.0f64;
        let mut knorm = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let d = tgt[i] - src[j];
                let inv = 1.0 / (d * d + eta * eta);
                let er = d * inv;
                let ei = eta * inv;
                let mut ar = 0.0;
                let mut ai = 0.0;
                for t in 0..rk {
                    ar += ur[t * m + i] * vr[t * n + j] - ui[t * m + i] * vi[t * n + j];
                    ai += ur[t * m + i] * vi[t * n + j] + ui[t * m + i] * vr[t * n + j];
                }
                worst = worst.max(((ar - er).powi(2) + (ai - ei).powi(2)).sqrt());
                knorm += er * er + ei * ei;
            }
        }
        eprintln!("block max_abs_err={worst:.3e} fro={:.3e}", knorm.sqrt());
        assert!(worst < 1e-7, "ACA block reconstruction err {worst:.3e}");
    }
}
