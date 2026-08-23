//! Benchmark: hierarchical FMM with LOCAL expansions (the "downward pass" /
//! Chebyshev-style interpolation the user described) vs the existing multipole
//! treecode.
//!
//! The user's idea: for eigenvalues far from the query point, use a polynomial
//! (local) expansion instead of evaluating each term directly. The classic FMM
//! "downward pass" does exactly this: it computes a local expansion about each
//! leaf's center ONCE (from all well-separated source clusters via M2L), then
//! evaluates that polynomial at every query point in the leaf. This amortizes
//! the far-field work over all query points in a leaf, instead of walking the
//! tree per query point like the multipole treecode does.
//!
//! Kernel: 1/(z - x), z = λ - iη. Multipole moments M_n = Σ(x-μ)^n are real;
//! local coefficients L_n = -Σ 1/(x-μ_t)^{n+1} are also real. Only the final
//! polynomial evaluation at complex z is complex.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::RngExt;
use shrinkers::stieltjes::compute_all_stieltjes_treecode_impl;
use std::hint::black_box;

/// Generate a Marchenko-Pastur-like eigenvalue spectrum.
fn generate_mp_spectrum(p: usize, c: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    let lam_min = (1.0 - c.sqrt()).max(0.01).powi(2);
    let lam_max = (1.0 + c.sqrt()).powi(2);
    (0..p)
        .map(|_| lam_min + rng.random::<f64>() * (lam_max - lam_min) + rng.random::<f64>() * 0.1)
        .collect()
}

// ---------------------------------------------------------------------------
// Local-expansion FMM
// ---------------------------------------------------------------------------

struct LNode {
    lo: f64,
    hi: f64,
    count: usize,
    mu: f64,
    /// Multipole moments M_n about `mu` (real).
    mom: Vec<f64>,
    left: i32,
    right: i32,
    /// Eigenvalue indices in this node (leaves only).
    points: Vec<usize>,
}

struct LocalFmm {
    nodes: Vec<LNode>,
    order: usize,
    theta_sq: f64,
}

// Explicit bounds keep the recursive build readable; a params struct would
// only obscure the lo/hi split recursion.
#[allow(clippy::too_many_arguments)]
fn build_node(
    sorted: &[f64],
    lo_idx: usize,
    hi_idx: usize,
    lo: f64,
    hi: f64,
    p1: usize,
    leaf_cap: usize,
    nodes: &mut Vec<LNode>,
) -> i32 {
    let idx = nodes.len() as i32;
    let count = hi_idx - lo_idx;
    nodes.push(LNode {
        lo,
        hi,
        count,
        mu: 0.0,
        mom: vec![0.0; p1],
        left: -1,
        right: -1,
        points: Vec::new(),
    });

    if count <= leaf_cap {
        let mu: f64 = sorted[lo_idx..hi_idx].iter().sum::<f64>() / count as f64;
        let n = idx as usize;
        nodes[n].mu = mu;
        nodes[n].points = (lo_idx..hi_idx).collect();
        return idx;
    }

    let mid = (lo + hi) * 0.5;
    let mut split = sorted[lo_idx..hi_idx].partition_point(|&x| x <= mid) + lo_idx;
    if split == lo_idx || split == hi_idx {
        split = (lo_idx + hi_idx) / 2;
    }
    let li = build_node(sorted, lo_idx, split, lo, mid, p1, leaf_cap, nodes);
    let ri = build_node(sorted, split, hi_idx, mid, hi, p1, leaf_cap, nodes);
    let n = idx as usize;
    nodes[n].left = li;
    nodes[n].right = ri;
    let cnt_l = nodes[li as usize].count;
    let cnt_r = nodes[ri as usize].count;
    nodes[n].count = cnt_l + cnt_r;
    let mu = (nodes[li as usize].mu * cnt_l as f64 + nodes[ri as usize].mu * cnt_r as f64)
        / (nodes[n].count as f64);
    nodes[n].mu = mu;
    idx
}

impl LocalFmm {
    fn build(sorted: &[f64], order: usize, theta: f64, leaf_cap: usize) -> Self {
        let p1 = order + 1;
        let mut nodes = Vec::with_capacity(2 * sorted.len());
        build_node(
            sorted,
            0,
            sorted.len(),
            sorted[0],
            sorted[sorted.len() - 1],
            p1,
            leaf_cap,
            &mut nodes,
        );
        LocalFmm {
            nodes,
            order,
            theta_sq: theta * theta,
        }
    }

    /// Upward pass: compute multipole moments bottom-up.
    fn upward(&mut self, sorted: &[f64]) {
        let p1 = self.order + 1;
        // Nodes are built pre-order (parent before children), so reverse order
        // processes children before parents.
        for i in (0..self.nodes.len()).rev() {
            let left = self.nodes[i].left;
            let right = self.nodes[i].right;
            if left < 0 && right < 0 {
                // Leaf: M_n = Σ (x_k - mu)^n
                let mu = self.nodes[i].mu;
                let mut acc = vec![0.0; p1];
                for &idx in &self.nodes[i].points {
                    let d = sorted[idx] - mu;
                    let mut pw = 1.0;
                    for a_n in acc.iter_mut() {
                        *a_n += pw;
                        pw *= d;
                    }
                }
                self.nodes[i].mom = acc;
            } else {
                // Internal: shift children moments to this node's mu.
                let mu = self.nodes[i].mu;
                let mut acc = vec![0.0; p1];
                for child in [left, right] {
                    if child < 0 {
                        continue;
                    }
                    let c = child as usize;
                    let delta = mu - self.nodes[c].mu;
                    // M_new[n] = Σ_{k=0}^n C(n,k) M_old[k] (-delta)^{n-k}
                    // Indexed loop is clearest here: `n` participates in the
                    // binomial coefficient and the (n-k) power, not just as an
                    // index.
                    #[allow(clippy::needless_range_loop)]
                    for n in 0..p1 {
                        let mut s = 0.0;
                        for k in 0..=n {
                            let ck = binom(n, k);
                            s += self.nodes[c].mom[k] * ck * (-delta).powi((n - k) as i32);
                        }
                        acc[n] += s;
                    }
                }
                self.nodes[i].mom = acc;
            }
        }
    }

    /// Downward pass + evaluation. For each leaf, compute its local expansion
    /// from all well-separated source clusters (M2L), then evaluate the
    /// polynomial at each query point in the leaf, plus direct near-field.
    fn evaluate(&self, sorted: &[f64], eta: f64) -> Vec<(f64, f64)> {
        let p = sorted.len();
        let p1 = self.order + 1;
        let mut result = vec![(0.0, 0.0); p];

        for t in 0..self.nodes.len() {
            let tn = &self.nodes[t];
            if tn.left >= 0 || tn.right >= 0 {
                continue; // only leaves are query targets
            }
            let mu_t = tn.mu;

            // Local expansion coefficients about mu_t (real).
            let mut loc = vec![0.0; p1];
            // Per-query-point direct near-field accumulators.
            let nq = tn.points.len();
            let mut dir_re = vec![0.0; nq];
            let mut dir_im = vec![0.0; nq];

            // Walk the source tree.
            let mut stack = vec![0i32];
            while let Some(s) = stack.pop() {
                let sn = &self.nodes[s as usize];
                if sn.count == 0 {
                    continue;
                }
                let mu_s = sn.mu;
                let diff = mu_t - mu_s;
                let dist_sq = diff * diff + eta * eta;
                let cluster_size = sn.hi - sn.lo;
                let is_leaf = sn.left < 0 && sn.right < 0;

                if is_leaf || cluster_size * cluster_size < self.theta_sq * dist_sq {
                    if is_leaf {
                        // Direct near-field: add sn.points to each query point.
                        for &idx in &sn.points {
                            let x = sorted[idx];
                            for (q, &qidx) in tn.points.iter().enumerate() {
                                let d = sorted[qidx] - x;
                                let denom = d * d + eta * eta;
                                let inv = 1.0 / denom;
                                dir_re[q] += d * inv;
                                dir_im[q] += eta * inv;
                            }
                        }
                    } else {
                        // M2L: L_n += Σ_m M_m (-1)^n C(m+n,n) / diff^{m+n+1}
                        let inv = 1.0 / diff;
                        let mut inv_pow = vec![1.0; 2 * p1 + 1];
                        for k in 1..=2 * p1 {
                            inv_pow[k] = inv_pow[k - 1] * inv;
                        }
                        for n in 0..p1 {
                            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                            let mut acc = 0.0;
                            for m in 0..p1 {
                                let mm = sn.mom[m];
                                if mm == 0.0 {
                                    continue;
                                }
                                acc += mm * binom(m + n, n) * inv_pow[m + n + 1];
                            }
                            loc[n] += sign * acc;
                        }
                    }
                } else {
                    // Descend.
                    if sn.right >= 0 {
                        stack.push(sn.right);
                    }
                    if sn.left >= 0 {
                        stack.push(sn.left);
                    }
                }
            }

            // Evaluate local expansion + direct at each query point.
            for (q, &qidx) in tn.points.iter().enumerate() {
                let lambda = sorted[qidx];
                let wr = lambda - mu_t;
                let wi = -eta;
                let mut wpr = 1.0;
                let mut wpi = 0.0;
                let mut re = 0.0;
                let mut im = 0.0;
                for &l_n in loc.iter() {
                    re += l_n * wpr;
                    im += l_n * wpi;
                    let nr = wpr * wr - wpi * wi;
                    let ni = wpr * wi + wpi * wr;
                    wpr = nr;
                    wpi = ni;
                }
                result[qidx] = (re + dir_re[q], im + dir_im[q]);
            }
        }
        result
    }
}

/// Binomial coefficient C(n,k) computed directly (small n, prototype).
fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut r = 1.0;
    for i in 0..k {
        r *= (n - i) as f64 / (i + 1) as f64;
    }
    r
}

/// Full local-expansion FMM entry point.
fn compute_local_fmm(
    sorted: &[f64],
    eta: f64,
    theta: f64,
    order: usize,
    leaf_cap: usize,
) -> Vec<(f64, f64)> {
    let mut fmm = LocalFmm::build(sorted, order, theta, leaf_cap);
    fmm.upward(sorted);
    fmm.evaluate(sorted, eta)
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_local_expansion(c: &mut Criterion) {
    let conc = 0.5;

    for p in [1000, 5000, 20000] {
        let mut evals = generate_mp_spectrum(p, conc);
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        let mut group = c.benchmark_group(format!("p={p},local_expansion"));
        group.sample_size(10);

        // Existing multipole treecode (baseline), same config as local.
        group.bench_function("treecode_multipole", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| black_box(compute_all_stieltjes_treecode_impl(&e, eta, 0.3, 8, false)),
                BatchSize::SmallInput,
            )
        });

        // Parallel multipole treecode (production large-p winner).
        group.bench_function("treecode_multipole_par", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| black_box(compute_all_stieltjes_treecode_impl(&e, eta, 0.3, 8, true)),
                BatchSize::SmallInput,
            )
        });

        // New local-expansion FMM, same (theta, order).
        group.bench_function("local_fmm", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| black_box(compute_local_fmm(&e, eta, 0.3, 8, 16)),
                BatchSize::SmallInput,
            )
        });

        group.finish();
    }
}

// ---------------------------------------------------------------------------
// Correctness check (not a criterion bench; run via `cargo test --release`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn local_fmm_matches_autovec() {
        use super::compute_local_fmm;
        use shrinkers::stieltjes::autovec_stieltjes_sum;
        for p in [256, 1000, 4000] {
            let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
            evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let eta = 0.1 / (p as f64).sqrt();

            let lf = compute_local_fmm(&evals, eta, 0.5, 6, 16);

            let mut max_err_r = 0.0;
            let mut max_err_i = 0.0;
            for i in 0..p {
                let (rr, ri) = autovec_stieltjes_sum(evals[i], &evals, eta);
                let denom = rr.abs().max(1e-12);
                max_err_r = max_err_r.max((lf[i].0 - rr).abs() / denom);
                max_err_i = max_err_i.max((lf[i].1 - ri).abs() / ri.abs().max(1e-12));
            }
            eprintln!("p={p}: local_fmm real max rel err {max_err_r:.6}, imag {max_err_i:.6}");
            assert!(max_err_r < 1e-2, "p={p} real err too large: {max_err_r}");
            assert!(max_err_i < 1e-2, "p={p} imag err too large: {max_err_i}");
        }
    }
}

criterion_group!(benches, bench_local_expansion);
criterion_main!(benches);
