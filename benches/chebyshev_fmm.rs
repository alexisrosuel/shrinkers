//! Benchmark: TRUE Chebyshev-interpolation treecode vs the existing multipole
//! treecode.
//!
//! The user's idea: for eigenvalues far from the query point, decouple via
//! Chebyshev interpolation of the kernel rather than multipole moments:
//!
//! ```text
//!   1/(z - x) ≈ Σ_{j=1}^{n} ℓ_j(x) · 1/(z - t_j),   t_j = Chebyshev nodes on [lo,hi]
//! ```
//!
//! where ℓ_j are the Lagrange basis polynomials. Summing over sources gives:
//!
//!   Σ_k 1/(z - x_k) ≈ Σ_j  w_j / (z - t_j),   w_j = Σ_k ℓ_j(x_k)  (REAL, fixed)
//!
//! This mirrors the multipole treecode structure (aggregate sources once per
//! node, evaluate per query) but:
//!   - **no binomial moment-shifting** in the upward pass (weights are just the
//!     aggregated Lagrange basis at fixed nodes),
//!   - the pole `z = λ - iη` sits **off the real axis by η**, so interpolation
//!     of `1/(z-x)` over the real interval `[lo,hi]` converges fast,
//!   - evaluation is a plain dot product `Σ w_j/(z-t_j)` (no power recurrence).
//!
//! Treecode variant: per-query walk (like `treecode.rs`). Leaves evaluate
//! exactly (handles the `z ≈ x` diagonal / singular near-field); well-separated
//! internal nodes use the Chebyshev far-field.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::RngExt;
use rayon::prelude::*;
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
// Chebyshev-interpolation treecode
// ---------------------------------------------------------------------------

struct ChebNode {
    lo: f64,
    hi: f64,
    /// Chebyshev nodes t_j on [lo,hi] (interpolate of degree n-1).
    nodes: Vec<f64>,
    /// Source weights w_j = Σ_{x in node} ℓ_j(x), real, aligned with `nodes`.
    w: Vec<f64>,
    left: i32,
    right: i32,
    /// Exact eigenvalues held by a leaf (empty for internal nodes).
    pts: Vec<f64>,
}

struct ChebTree {
    nodes: Vec<ChebNode>,
    n: usize,
    theta_sq: f64,
}

/// Second-kind Chebyshev nodes on [-1,1]: s_j = cos(jπ/(n-1)), j=0..n-1.
/// Barycentric weights: λ_0 = 1/2, λ_j = (-1)^j (interior), λ_{n-1} = (-1)^(n-1)/2.
fn cheb_nodes_m1(n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| (j as f64 * std::f64::consts::PI / (n - 1) as f64).cos())
        .collect()
}

/// Fill Chebyshev nodes + source weights (Lagrange basis, barycentric form)
/// for node `idx` using `sorted[lo_idx..hi_idx]`.
fn fill_weights(nodes: &mut [ChebNode], idx: usize, sorted: &[f64], lo_idx: usize, hi_idx: usize) {
    let n = nodes[idx].nodes.len();
    let lo = nodes[idx].lo;
    let hi = nodes[idx].hi;
    let c = 0.5 * (lo + hi);
    let l = 0.5 * (hi - lo);

    let sm1 = cheb_nodes_m1(n);
    let mut t = vec![0.0; n];
    let mut lam = vec![0.0; n];
    for j in 0..n {
        t[j] = c + l * sm1[j];
        // Barycentric weights for 2nd-kind nodes.
        lam[j] = if j == 0 || j == n - 1 {
            0.5 * if j % 2 == 0 { 1.0 } else { -1.0 }
        } else if j % 2 == 0 {
            1.0
        } else {
            -1.0
        };
    }
    nodes[idx].nodes = t.clone();
    let mut w = vec![0.0; n];

    // ℓ_j(x) = (λ_j/(x-t_j)) / (Σ_i λ_i/(x-t_i)); if x hits a node exactly,
    // that single basis evaluates to 1.
    for &x in &sorted[lo_idx..hi_idx] {
        let mut s = 0.0;
        let mut hit = usize::MAX;
        for j in 0..n {
            let d = x - t[j];
            if d == 0.0 {
                hit = j;
                break;
            }
            s += lam[j] / d;
        }
        if hit != usize::MAX {
            w[hit] += 1.0;
            continue;
        }
        for j in 0..n {
            w[j] += (lam[j] / (x - t[j])) / s;
        }
    }
    nodes[idx].w = w;
}

fn build_cheb(
    sorted: &[f64],
    n: usize,
    leaf_cap: usize,
    lo_idx: usize,
    hi_idx: usize,
    nodes: &mut Vec<ChebNode>,
) -> i32 {
    let idx = nodes.len() as i32;
    let count = hi_idx - lo_idx;
    let lo = sorted[lo_idx];
    let hi = sorted[hi_idx - 1];

    nodes.push(ChebNode {
        lo,
        hi,
        nodes: vec![0.0; n],
        w: Vec::new(),
        left: -1,
        right: -1,
        pts: if count <= leaf_cap {
            sorted[lo_idx..hi_idx].to_vec()
        } else {
            Vec::new()
        },
    });

    if count <= leaf_cap {
        fill_weights(nodes, idx as usize, sorted, lo_idx, hi_idx);
        return idx;
    }

    let mid = lo + 0.5 * (hi - lo);
    let mut split = sorted[lo_idx..hi_idx].partition_point(|&x| x <= mid) + lo_idx;
    if split == lo_idx || split == hi_idx {
        split = (lo_idx + hi_idx) / 2;
    }
    let li = build_cheb(sorted, n, leaf_cap, lo_idx, split, nodes);
    let ri = build_cheb(sorted, n, leaf_cap, split, hi_idx, nodes);

    let ni = idx as usize;
    nodes[ni].left = li;
    nodes[ni].right = ri;
    fill_weights(nodes, ni, sorted, lo_idx, hi_idx);
    idx
}

impl ChebTree {
    fn build(sorted: &[f64], n: usize, theta: f64, leaf_cap: usize) -> Self {
        let mut nodes = Vec::with_capacity(2 * sorted.len());
        build_cheb(sorted, n, leaf_cap, 0, sorted.len(), &mut nodes);
        ChebTree {
            nodes,
            n,
            theta_sq: theta * theta,
        }
    }

    /// Stieltjes contribution at query z=(lambda_i, -eta): per-query walk.
    #[inline]
    fn contribution(&self, lambda_i: f64, eta: f64, stack: &mut Vec<i32>) -> (f64, f64) {
        let n = self.n;
        stack.clear();
        stack.push(0);

        let mut re = 0.0;
        let mut im = 0.0;

        while let Some(node) = stack.pop() {
            let ni = node as usize;
            let nd = &self.nodes[ni];
            let is_leaf = !nd.pts.is_empty();

            if is_leaf {
                // Exact pairwise sum over the leaf's sources.
                let mut lr = 0.0;
                let mut li = 0.0;
                for &x in &nd.pts {
                    let d = lambda_i - x;
                    let inv = 1.0 / (d * d + eta * eta);
                    lr += d * inv;
                    li += eta * inv;
                }
                re += lr;
                im += li;
                continue;
            }

            // Distance from z to the interval [lo,hi].
            let lo = nd.lo;
            let hi = nd.hi;
            let cl = if lambda_i < lo {
                lo - lambda_i
            } else if lambda_i > hi {
                lambda_i - hi
            } else {
                0.0
            };
            let d_sq = cl * cl + eta * eta;
            let half_w = 0.5 * (hi - lo);

            if half_w * half_w < self.theta_sq * d_sq {
                // Well-separated: Chebyshev far-field  Σ_j w_j/(z-t_j).
                for j in 0..n {
                    let t = nd.nodes[j];
                    let wr = lambda_i - t;
                    let wi = eta; // z = λ - iη → Im 1/(z-t) = +η/|z-t|²
                    let inv = 1.0 / (wr * wr + wi * wi);
                    let wj = nd.w[j];
                    re += wj * wr * inv;
                    im += wj * wi * inv;
                }
            } else {
                // Otherwise descend.
                let r = nd.right;
                let l = nd.left;
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

fn compute_cheb_treecode(
    sorted: &[f64],
    eta: f64,
    theta: f64,
    n: usize,
    leaf_cap: usize,
) -> Vec<(f64, f64)> {
    let tree = ChebTree::build(sorted, n, theta, leaf_cap);
    let mut stack = Vec::with_capacity(64);
    sorted
        .iter()
        .map(|&li| tree.contribution(li, eta, &mut stack))
        .collect()
}

/// Parallel Chebyshev treecode: Rayon over query points (each independent).
fn compute_cheb_treecode_par(
    sorted: &[f64],
    eta: f64,
    theta: f64,
    n: usize,
    leaf_cap: usize,
) -> Vec<(f64, f64)> {
    let tree = ChebTree::build(sorted, n, theta, leaf_cap);
    sorted
        .par_iter()
        .map(|&li| {
            let mut stack = Vec::with_capacity(64);
            tree.contribution(li, eta, &mut stack)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_chebyshev(c: &mut Criterion) {
    let conc = 0.5;

    for p in [1000, 5000, 20000] {
        let mut evals = generate_mp_spectrum(p, conc);
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        let mut group = c.benchmark_group(format!("p={p},chebyshev"));
        group.sample_size(10);

        group.bench_function("treecode_multipole", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| black_box(compute_all_stieltjes_treecode_impl(&e, eta, 0.3, 8, false)),
                BatchSize::SmallInput,
            )
        });
        group.bench_function("treecode_multipole_par", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| black_box(compute_all_stieltjes_treecode_impl(&e, eta, 0.3, 8, true)),
                BatchSize::SmallInput,
            )
        });
        group.bench_function("treecode_chebyshev", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| black_box(compute_cheb_treecode(&e, eta, 0.3, 9, 16)),
                BatchSize::SmallInput,
            )
        });
        group.bench_function("treecode_chebyshev_par", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| black_box(compute_cheb_treecode_par(&e, eta, 0.3, 9, 16)),
                BatchSize::SmallInput,
            )
        });

        group.finish();
    }
}

// ---------------------------------------------------------------------------
// Correctness check (run via `cargo test --release`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn chebyshev_treecode_matches_autovec() {
        use super::compute_cheb_treecode;
        use shrinkers::stieltjes::autovec_stieltjes_sum;
        for p in [512, 2000, 8000] {
            let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
            evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let eta = 0.1 / (p as f64).sqrt();
            let p_ = p;

            let cheb = compute_cheb_treecode(&evals, eta, 0.3, 9, 16);

            let mut max_r: f64 = 0.0;
            let mut max_i: f64 = 0.0;
            for i in 0..p_ {
                let (rr, ri) = autovec_stieltjes_sum(evals[i], &evals, eta);
                max_r = max_r.max((cheb[i].0 - rr).abs() / rr.abs().max(1e-12));
                max_i = max_i.max((cheb[i].1 - ri).abs() / ri.abs().max(1e-12));
            }
            eprintln!("p={p_}: cheb real max rel err {max_r:.6}, imag {max_i:.6}");
            assert!(max_r < 1e-2, "p={p_} real err too large: {max_r}");
            assert!(max_i < 1e-2, "p={p_} imag err too large: {max_i}");
        }
    }

    #[test]
    fn chebyshev_parallel_matches_sequential() {
        use super::{compute_cheb_treecode, compute_cheb_treecode_par};
        let p = 4000;
        let mut evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        let seq = compute_cheb_treecode(&evals, eta, 0.3, 9, 16);
        let par = compute_cheb_treecode_par(&evals, eta, 0.3, 9, 16);
        for i in 0..p {
            assert!(
                (seq[i].0 - par[i].0).abs() < 1e-9 && (seq[i].1 - par[i].1).abs() < 1e-9,
                "mismatch at {i}: seq=({},{}) par=({},{})",
                seq[i].0,
                seq[i].1,
                par[i].0,
                par[i].1
            );
        }
    }
}

criterion_group!(benches, bench_chebyshev);
criterion_main!(benches);
