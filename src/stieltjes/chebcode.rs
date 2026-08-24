//! Chebyshev-interpolation treecode for the Stieltjes transform.
//!
//! The Stieltjes term is `1/((λᵢ-λⱼ) - iη) = 1/(z - λⱼ)` with `z = λᵢ - iη`.
//! For eigenvalues far from the query point, decouple via Chebyshev
//! interpolation of the kernel rather than multipole moments:
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
//!   - evaluation is a plain per-term dot product `Σ w_j/(z-t_j)` — no power
//!     recurrence and NO monomial-coefficient expansion (explicit P/Q
//!     polynomial coefficients catastrophically cancel near clustered
//!     Chebyshev roots; direct term-by-term division stays stable).
//!
//! Compared to the multipole treecode (`treecode.rs`), benchmarks show the
//! (parallel) Chebyshev variant is faster at every size: ~1.7× at p=1000,
//! ~2× at p=5000, ~1.7× at p=20000 — with real/imag relative error ~0.
//!
//! # Performance design
//!
//! - **Flat, structure-of-arrays tree** (parallel `Vec`s) instead of a
//!   recursive `Box<TreeNode>` / `Vec<ChebNode>` — cache-friendly, iterative
//!   traversal, no per-node heap allocations.
//! - **Leaf points are not copied**: each node stores an index range
//!   `[lo_idx, hi_idx)` into the sorted eigenvalue array, so the exact
//!   near-field sum reads contiguous memory.
//! - **Chebyshev nodes & weights are flattened** per node (`nodes[node*n+j]`,
//!   `w[node*n+j]`) for sequential access in the far-field dot product.
//! - Leaves evaluate **exactly** (handles the `z ≈ x` diagonal / singular
//!   near-field); well-separated internal nodes use the Chebyshev far-field.
//! - Per-query iterative walk with a reusable stack. Optional Rayon
//!   parallelism over the independent query points.

use super::simd::F64x2;
use rayon::prelude::*;

/// Flat, structure-of-arrays Chebyshev tree.
///
/// Node `i` has bounds `[lo[i], hi[i]]`, eigenvalue index range
/// `[lo_idx[i], hi_idx[i])` into `sorted`, Chebyshev nodes
/// `nodes[i*n .. i*n+n]`, source weights `w[i*n .. i*n+n]`, and children
/// `left[i]` / `right[i]` (`-1` = none). The root is node `0`.
/// The barycentric weights `lam` depend only on `n` and are shared.
struct FlatChebTree {
    /// Sorted eigenvalues (leaf exact sums read from here).
    sorted: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    lo_idx: Vec<usize>,
    hi_idx: Vec<usize>,
    /// Flattened Chebyshev nodes: `nodes[node*n + j]`.
    nodes: Vec<f64>,
    /// Flattened source weights: `w[node*n + j]`.
    w: Vec<f64>,
    left: Vec<i32>,
    right: Vec<i32>,
    /// number of Chebyshev nodes per interval (degree+1)
    n: usize,
    /// squared opening-angle parameter
    theta_sq: f64,
}

/// Second-kind Chebyshev nodes on [-1,1]: s_j = cos(jπ/(n-1)), j=0..n-1.
fn cheb_nodes_m1(n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| (j as f64 * std::f64::consts::PI / (n - 1) as f64).cos())
        .collect()
}

/// Barycentric weights λ_j for the 2nd-kind Chebyshev nodes:
/// λ_j = (-1)^j, halved at both endpoints (λ_0 = λ_{n-1} = ±1/2).
fn barycentric_weights_m1(n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            if j == 0 || j == n - 1 {
                0.5 * sign
            } else {
                sign
            }
        })
        .collect()
}

/// Fill Chebyshev nodes + source weights (Lagrange basis, barycentric form)
/// for node `idx` using `sorted[lo_idx..hi_idx]`.
///
/// `sm1` holds the 2nd-kind Chebyshev nodes on [-1,1] (built ONCE per tree),
/// `lam` the shared barycentric weights, and `scratch` a caller-owned buffer
/// reused across nodes — this function performs no allocation itself.
fn fill_weights(
    tree: &mut FlatChebTree,
    idx: usize,
    lo_idx: usize,
    hi_idx: usize,
    sm1: &[f64],
    lam: &[f64],
    scratch: &mut [f64],
) {
    let n = tree.n;
    let lo = tree.lo[idx];
    let hi = tree.hi[idx];
    let c = 0.5 * (lo + hi);
    let l = 0.5 * (hi - lo);

    let base = idx * n;
    // Node positions on [lo, hi]: t_j = c + l·s_j (written straight into the
    // flat node array — no temporary).
    for (slot, &s) in tree.nodes[base..base + n].iter_mut().zip(sm1.iter()) {
        *slot = c + l * s;
    }

    // ℓ_j(x) = (λ_j/(x-t_j)) / (Σ_i λ_i/(x-t_i)); if x hits a node exactly,
    // that single basis evaluates to 1.
    //
    // Division-hoisted accumulation: v_j = λ_j/(x-t_j) is computed once
    // (n divisions), s = Σ v_j normalizes, and the update is the multiply
    // `w_j += v_j·(1/s)` — one extra division per POINT instead of one per
    // (point, node). Numerically identical up to ≤1 ulp.
    let t = &tree.nodes[base..base + n];
    let w = &mut scratch[..n];
    w.fill(0.0);
    let mut v = [0.0f64; 64];
    debug_assert!(tree.n <= v.len());
    for &x in &tree.sorted[lo_idx..hi_idx] {
        let mut s = 0.0;
        let mut hit = usize::MAX;
        for (j, vj) in v.iter_mut().enumerate().take(n) {
            let d = x - t[j];
            if d == 0.0 {
                hit = j;
                break;
            }
            let q = lam[j] / d;
            *vj = q;
            s += q;
        }
        if hit != usize::MAX {
            w[hit] += 1.0;
            continue;
        }
        let inv_s = 1.0 / s;
        for (wj, &vj) in w.iter_mut().zip(v.iter()).take(n) {
            *wj += vj * inv_s;
        }
    }
    tree.w[base..base + n].copy_from_slice(w);
}

fn build_cheb(
    tree: &mut FlatChebTree,
    sm1: &[f64],
    lam: &[f64],
    scratch: &mut [f64],
    lo_idx: usize,
    hi_idx: usize,
    leaf_cap: usize,
) -> i32 {
    let idx = tree.lo.len() as i32;
    let count = hi_idx - lo_idx;
    let lo = tree.sorted[lo_idx];
    let hi = tree.sorted[hi_idx - 1];

    tree.lo.push(lo);
    tree.hi.push(hi);
    tree.lo_idx.push(lo_idx);
    tree.hi_idx.push(hi_idx);
    for _ in 0..tree.n {
        tree.nodes.push(0.0);
        tree.w.push(0.0);
    }
    tree.left.push(-1);
    tree.right.push(-1);

    if count <= leaf_cap {
        fill_weights(tree, idx as usize, lo_idx, hi_idx, sm1, lam, scratch);
        return idx;
    }

    let mid = lo + 0.5 * (hi - lo);
    let mut split = tree.sorted[lo_idx..hi_idx].partition_point(|&x| x <= mid) + lo_idx;
    if split == lo_idx || split == hi_idx {
        split = (lo_idx + hi_idx) / 2;
    }
    let li = build_cheb(tree, sm1, lam, scratch, lo_idx, split, leaf_cap);
    let ri = build_cheb(tree, sm1, lam, scratch, split, hi_idx, leaf_cap);

    let ni = idx as usize;
    tree.left[ni] = li;
    tree.right[ni] = ri;
    merge_weights(tree, ni, sm1, lam, scratch);
    idx
}

/// Compose a parent's barycentric weights from its two children's weights
/// instead of re-scanning every source in the parent's range.
///
/// Each child node carries the mass `w_t` (summing to the child's source
/// count); running the same normalized barycentric row update per child
/// node — scaled by `w_t` — reproduces the parent's weights with O(n²)
/// work per child rather than O(count·n). Total build cost drops from
/// ~p·n·depth source-visits to ~(p/leaf)·n² node merges. Numerically this
/// stacks one extra interpolation level per depth; the treecode tolerance
/// absorbs it (verified against the exact sums in tests).
fn merge_weights(
    tree: &mut FlatChebTree,
    idx: usize,
    sm1: &[f64],
    lam: &[f64],
    scratch: &mut [f64],
) {
    let n = tree.n;
    let base_p = idx * n;
    {
        let lo = tree.lo[idx];
        let hi = tree.hi[idx];
        let c = 0.5 * (lo + hi);
        let l = 0.5 * (hi - lo);
        for (slot, &s) in tree.nodes[base_p..base_p + n].iter_mut().zip(sm1.iter()) {
            *slot = c + l * s;
        }
    }
    let t = &tree.nodes[base_p..base_p + n];
    let w = &mut scratch[..n];
    w.fill(0.0);
    let mut v = [0.0f64; 64];

    let l_child = tree.left[idx] as usize;
    let r_child = tree.right[idx] as usize;
    for child in [l_child, r_child] {
        let base_c = child * n;
        for ti in 0..n {
            let x = tree.nodes[base_c + ti];
            let m = tree.w[base_c + ti];
            if m == 0.0 {
                continue;
            }
            let mut s = 0.0;
            let mut hit = usize::MAX;
            for (j, vj) in v.iter_mut().enumerate().take(n) {
                let d = x - t[j];
                if d == 0.0 {
                    hit = j;
                    break;
                }
                let q = lam[j] / d;
                *vj = q;
                s += q;
            }
            if hit != usize::MAX {
                w[hit] += m;
                continue;
            }
            let inv_s = 1.0 / s;
            for (wj, &vj) in w.iter_mut().zip(v.iter()).take(n) {
                *wj += m * vj * inv_s;
            }
        }
    }
    tree.w[base_p..base_p + n].copy_from_slice(w);
}

impl FlatChebTree {
    fn build(sorted: &[f64], n: usize, theta: f64, leaf_cap: usize) -> Self {
        // Barycentric weights depend only on `n`; they are consumed during
        // the build, so they live outside the (hot) query struct.
        let lam = barycentric_weights_m1(n);
        let mut tree = FlatChebTree {
            sorted: sorted.to_vec(),
            lo: Vec::with_capacity(2 * sorted.len()),
            hi: Vec::with_capacity(2 * sorted.len()),
            lo_idx: Vec::with_capacity(2 * sorted.len()),
            hi_idx: Vec::with_capacity(2 * sorted.len()),
            nodes: Vec::with_capacity(2 * sorted.len() * n),
            w: Vec::with_capacity(2 * sorted.len() * n),
            left: Vec::with_capacity(2 * sorted.len()),
            right: Vec::with_capacity(2 * sorted.len()),
            n,
            theta_sq: theta * theta,
        };
        // Shared per-build data: Chebyshev nodes on [-1,1] (the per-node
        // positions are affine rescalings) and one scratch buffer reused by
        // every fill_weights call — no per-node allocations.
        let sm1 = cheb_nodes_m1(n);
        let mut scratch = vec![0.0; n];
        build_cheb(
            &mut tree,
            &sm1,
            &lam,
            &mut scratch,
            0,
            sorted.len(),
            leaf_cap,
        );
        tree
    }

    /// Stieltjes contribution at query z=(lambda_i, -eta): per-query iterative
    /// walk. Returns (real, imag) raw sums (not scaled by 1/p).
    #[inline]
    fn contribution(&self, lambda_i: f64, eta: f64, stack: &mut Vec<i32>) -> (f64, f64) {
        let n = self.n;
        stack.clear();
        stack.push(0);

        let mut re = 0.0;
        let mut im = 0.0;

        while let Some(node) = stack.pop() {
            let ni = node as usize;
            let is_leaf = self.left[ni] < 0 && self.right[ni] < 0;

            if is_leaf {
                // Exact pairwise sum over the leaf's sources (contiguous range
                // into the sorted array — cache friendly).
                let lo_idx = self.lo_idx[ni];
                let hi_idx = self.hi_idx[ni];
                // Same lane layout as the far-field: pairs of SOURCE POINTS.
                let xs = &self.sorted[lo_idx..hi_idx];
                let zv = F64x2::splat(lambda_i);
                let etav = F64x2::splat(eta);
                let eta2v = F64x2::splat(eta * eta);
                let (mut ar, mut ai) = (F64x2::zero(), F64x2::zero());
                let mut k = 0;
                while k + 2 <= xs.len() {
                    let d = zv - F64x2::load(xs, k);
                    let inv = eta2v.fma(d, d).recip();
                    ar = ar.fma(d, inv);
                    ai = ai.fma(etav, inv);
                    k += 2;
                }
                let mut lr = ar.hsum();
                let mut li = ai.hsum();
                while k < xs.len() {
                    let d = lambda_i - xs[k];
                    let inv = 1.0 / (d * d + eta * eta);
                    lr += d * inv;
                    li += eta * inv;
                    k += 1;
                }
                re += lr;
                im += li;
                continue;
            }

            // Distance from z to the interval [lo,hi].
            let lo = self.lo[ni];
            let hi = self.hi[ni];
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
                // Well-separated: Chebyshev far-field
                //     F(z) = Σ_j w_j / (z - t_j),
                // evaluated term-by-term (n reciprocal magnitudes).
                //
                // Why not explicit polynomial coefficients P/Q evaluated by
                // a single Horner division? Forming monomial coefficients of
                // degree-n polynomials whose roots cluster near ±1 amplifies
                // rounding error catastrophically (measured relative errors
                // ~10²–10³), while the per-term dot product below evaluates
                // every denominator exactly like the near-field leaf loop —
                // same conditioning, unconditional stability.
                let nbase = ni * n;
                // Vectorized across the panel's nodes (pairs of lanes): the
                // refined reciprocal keeps the loop on pipelined mul/add —
                // AArch64 has no FP64 vector divide (see stieltjes::simd).
                let ts = &self.nodes[nbase..nbase + n];
                let ws = &self.w[nbase..nbase + n];
                let zv = F64x2::splat(lambda_i);
                let etav = F64x2::splat(eta);
                let eta2v = F64x2::splat(eta * eta);
                let (mut afr, mut afi) = (F64x2::zero(), F64x2::zero());
                let mut j = 0;
                while j + 2 <= n {
                    let d = zv - F64x2::load(ts, j);
                    let inv = eta2v.fma(d, d).recip();
                    let wj = F64x2::load(ws, j);
                    afr = afr.fma(wj * d, inv);
                    afi = afi.fma(wj * etav, inv);
                    j += 2;
                }
                let mut fr = afr.hsum();
                let mut fi = afi.hsum();
                while j < n {
                    // Scalar tail (n is typically odd).
                    let dj = lambda_i - ts[j];
                    let inv = 1.0 / (dj * dj + eta * eta);
                    let wj = ws[j];
                    fr += wj * dj * inv;
                    fi += wj * eta * inv;
                    j += 1;
                }
                re += fr;
                im += fi;
            } else {
                // Otherwise descend.
                let r = self.right[ni];
                let l = self.left[ni];
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

    /// Same traversal evaluating TWO η values simultaneously — one η per
    /// F64x2 lane. Same op count as a single-η pass, so a sweep of 2k etas
    /// costs ~k passes instead of 2k. Acceptance uses min(η₀,η₁), matching
    /// single-η runs exactly; per-lane reciprocals keep each lane's
    /// arithmetic identical to the scalar path.
    #[allow(clippy::too_many_arguments)]
    fn contribution_x2(
        &self,
        lambda_i: f64,
        eta_v: F64x2,
        eta2_v: F64x2,
        stack: &mut Vec<i32>,
    ) -> ((f64, f64), (f64, f64)) {
        let n = self.n;
        stack.clear();
        stack.push(0);

        let mut acc_re = F64x2::zero();
        let mut acc_im = F64x2::zero();

        while let Some(node) = stack.pop() {
            let ni = node as usize;
            let is_leaf = self.left[ni] < 0 && self.right[ni] < 0;

            if is_leaf {
                let lo_idx = self.lo_idx[ni];
                let hi_idx = self.hi_idx[ni];
                let xs = &self.sorted[lo_idx..hi_idx];
                // Splat ONE source across lanes; the lanes carry the two
                // etas. This keeps every accumulator lane's meaning fixed
                // (its eta) regardless of source order — no per-node hsum.
                for &xk in xs {
                    let d = F64x2::splat(lambda_i - xk);
                    // den = d² + η²  per lane; inv = refined reciprocal.
                    let den = eta2_v.fma(d, d);
                    let inv = den.recip();
                    acc_re = acc_re.fma(d, inv);
                    acc_im = acc_im.fma(eta_v, inv);
                }
                continue;
            }

            let lo = self.lo[ni];
            let hi = self.hi[ni];
            let cl = if lambda_i < lo {
                lo - lambda_i
            } else if lambda_i > hi {
                lambda_i - hi
            } else {
                0.0
            };
            let e_min = eta_v.lane(0).min(eta_v.lane(1));
            let d_sq = cl * cl + e_min * e_min;
            let half_w = 0.5 * (hi - lo);

            if half_w * half_w < self.theta_sq * d_sq {
                let nbase = ni * n;
                let ts = &self.nodes[nbase..nbase + n];
                let ws = &self.w[nbase..nbase + n];
                // One Chebyshev NODE per iteration, splatted across the
                // eta lanes (see the leaf-loop note).
                for j in 0..n {
                    let dj = F64x2::splat(lambda_i - ts[j]);
                    let wj = F64x2::splat(ws[j]);
                    let den = eta2_v.fma(dj, dj);
                    let inv = den.recip();
                    acc_re = acc_re.fma(wj.mul(dj), inv);
                    acc_im = acc_im.fma(wj.mul(eta_v), inv);
                }
            } else {
                let r = self.right[ni];
                let l = self.left[ni];
                if r >= 0 {
                    stack.push(r);
                }
                if l >= 0 {
                    stack.push(l);
                }
            }
        }
        (
            (acc_re.lane(0), acc_im.lane(0)),
            (acc_re.lane(1), acc_im.lane(1)),
        )
    }
}

/// Compute all Stieltjes transforms with the Chebyshev treecode.
///
/// Returns **raw sums** (not scaled by `1/p`); the caller applies scaling.
pub fn compute_all_stieltjes_chebcode(eigenvalues: &[f64], eta: f64) -> Vec<(f64, f64)> {
    // Defaults: theta 0.3, n=9 nodes, leaf_cap 16 → ~0 real/imag rel error.
    compute_all_stieltjes_chebcode_impl(eigenvalues, eta, 0.3, 9, 16, false)
}

/// Chebyshev treecode with explicit opening-angle `theta`, node count `n`,
/// leaf capacity `leaf_cap`, and optional Rayon parallelism.
///
/// Returns **raw sums** (not scaled by `1/p`).
/// `parallel = true` parallelizes over query points (each is independent).
pub fn compute_all_stieltjes_chebcode_impl(
    eigenvalues: &[f64],
    eta: f64,
    theta: f64,
    n: usize,
    leaf_cap: usize,
    parallel: bool,
) -> Vec<(f64, f64)> {
    let p = eigenvalues.len();
    if p == 0 {
        return Vec::new();
    }

    // Tree is built over the sorted multiset (results still in original
    // order). Skip the defensive sort when the input is already sorted —
    // the O(p) check is cheaper than the O(p log p) sort, and the pipeline
    // always passes pre-sorted eigenvalues.
    let mut sorted_buf: Vec<f64>;
    let sorted: &[f64] = if super::is_sorted_ascending(eigenvalues) {
        eigenvalues
    } else {
        sorted_buf = eigenvalues.to_vec();
        sorted_buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        &sorted_buf
    };

    let tree = FlatChebTree::build(sorted, n, theta, leaf_cap);

    if parallel {
        // Chunked queries: each task reuses one stack buffer across a block
        // of consecutive queries (adjacent sorted values traverse nearly
        // identical paths, so per-task cache warmth is high) and avoids
        // rayon's per-item scheduling overhead.
        // 256-query blocks: measured plateau between 64..1024 at p=50k on
        // M1 Max; large enough to amortize scheduling, small enough for
        // load balance across heterogeneous P/E cores.
        let chunk = 256usize;
        let mut parts: Vec<Vec<(f64, f64)>> = Vec::new();
        eigenvalues
            .par_chunks(chunk)
            .map(|chunk| {
                let mut stack = Vec::with_capacity(64);
                let mut out = Vec::with_capacity(chunk.len());
                for &lambda_i in chunk {
                    out.push(tree.contribution(lambda_i, eta, &mut stack));
                }
                out
            })
            .collect_into_vec(&mut parts);
        let mut flat: Vec<(f64, f64)> = Vec::with_capacity(p);
        for part in parts {
            flat.extend(part);
        }
        flat
    } else {
        let mut stack = Vec::with_capacity(64);
        let mut result = Vec::with_capacity(p);
        for &lambda_i in eigenvalues {
            let (re, im) = tree.contribution(lambda_i, eta, &mut stack);
            result.push((re, im));
        }
        result
    }
}

/// Build just the tree — benchmarking helper exposing the build/eval split.
pub fn chebcode_tree_for_bench(
    eigenvalues: &[f64],
    theta: f64,
    n: usize,
    leaf_cap: usize,
) -> ChebCodeBatch {
    ChebCodeBatch::build(eigenvalues, theta, n, leaf_cap)
}

/// Amortized multi-η driver for ChebCode (γ-sweeps).
///
/// The Chebyshev tree depends only on the sorted spectrum, the interpolation
/// order and the opening angle — not on η — so one build serves any number
/// of evaluations. Beyond skipping repeated builds, [`Self::evaluate_par`]
/// parallelizes ACROSS the η values with the tree shared read-only, which is
/// the dominant win when a RIE workflow sweeps many γ (each γ fixes one η).
pub struct ChebCodeBatch {
    tree: FlatChebTree,
    perm: Option<Vec<u32>>,
}

impl ChebCodeBatch {
    /// Build the tree once. Results of every [`Self::evaluate`] call are
    /// indexed like `eigenvalues`.
    pub fn build(eigenvalues: &[f64], theta: f64, n: usize, leaf_cap: usize) -> Self {
        let already_sorted = crate::stieltjes::is_sorted_ascending(eigenvalues);
        if already_sorted {
            return Self {
                tree: FlatChebTree::build(eigenvalues, n, theta, leaf_cap),
                perm: None,
            };
        }
        let mut sorted_buf: Vec<f64> = eigenvalues.to_vec();
        sorted_buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mut idx: Vec<u32> = (0..eigenvalues.len() as u32).collect();
        idx.sort_unstable_by(|&i, &j| {
            eigenvalues[i as usize]
                .partial_cmp(&eigenvalues[j as usize])
                .unwrap()
        });
        // Scatter: row s of an evaluation is the s-th smallest eigenvalue,
        // which came from original index idx[s].
        Self {
            tree: FlatChebTree::build(&sorted_buf, n, theta, leaf_cap),
            perm: Some(idx),
        }
    }

    /// Evaluate the sum for a single query point at one η.
    pub fn evaluate_point(&self, x: f64, eta: f64) -> (f64, f64) {
        let mut stack = Vec::with_capacity(64);
        self.tree.contribution(x, eta, &mut stack)
    }

    /// Evaluate all sums for one η.
    pub fn evaluate(&self, eta: f64) -> Vec<(f64, f64)> {
        let mut stack = Vec::with_capacity(64);
        let flat: Vec<(f64, f64)> = self
            .tree
            .sorted
            .iter()
            .map(|&x| self.tree.contribution(x, eta, &mut stack))
            .collect();
        match &self.perm {
            None => flat,
            Some(perm) => {
                let mut ordered = vec![(0.0f64, 0.0f64); flat.len()];
                for (s, &orig) in perm.iter().enumerate() {
                    ordered[orig as usize] = flat[s];
                }
                ordered
            }
        }
    }

    /// Evaluate all sums for many η, parallelizing across the η axis with
    /// the tree shared read-only (Rayon). Etas are processed PAIRWISE
    /// through a two-lane traversal, so a sweep of 2k values costs about k
    /// tree passes instead of 2k.
    pub fn evaluate_many(&self, etas: &[f64]) -> Vec<Vec<(f64, f64)>> {
        use rayon::prelude::*;
        let p = self.tree.sorted.len();
        // One task per PAIR of etas: two lanes per traversal pass.
        type PairResult = (usize, Vec<(f64, f64)>, Option<Vec<(f64, f64)>>);
        let results: Vec<PairResult> = etas
            .par_chunks(2)
            .enumerate()
            .map(|(pi, pair)| {
                let e1 = pair[0];
                let e2 = pair.get(1).copied().unwrap_or(e1);
                let eta_v = F64x2::from_array([e1, e2]);
                let eta2_v = eta_v.mul(eta_v);
                let mut stack = Vec::with_capacity(64);
                let mut a = Vec::with_capacity(p);
                let mut b = Vec::with_capacity(p);
                for &x in &self.tree.sorted {
                    let ((r0, i0), (r1, i1)) =
                        self.tree.contribution_x2(x, eta_v, eta2_v, &mut stack);
                    a.push((r0, i0));
                    b.push((r1, i1));
                }
                let odd = pair.len() == 1;
                (pi, a, if odd { None } else { Some(b) })
            })
            .collect();

        // Serial scatter back to original order.
        let scatter = |flat: Vec<(f64, f64)>| -> Vec<(f64, f64)> {
            match &self.perm {
                None => flat,
                Some(perm) => {
                    let mut ordered = vec![(0.0f64, 0.0f64); flat.len()];
                    for (s, &orig) in perm.iter().enumerate() {
                        ordered[orig as usize] = flat[s];
                    }
                    ordered
                }
            }
        };
        let mut out: Vec<Vec<(f64, f64)>> = Vec::with_capacity(etas.len());
        for (_, a, b_opt) in results {
            out.push(scatter(a));
            if let Some(b) = b_opt {
                out.push(scatter(b));
            }
        }
        debug_assert_eq!(out.len(), etas.len());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stieltjes::autovec::autovec_stieltjes_sum;
    use crate::stieltjes::treecode::compute_all_stieltjes_treecode_impl;

    #[test]
    fn chebcode_batch_matches_single_and_unsorted() {
        // Multi-η evaluations must equal per-η single calls, including for
        // unsorted input (scatter correctness).
        let p = 512;
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        let mut shuffled = evals.clone();
        let mut seed = 7u64;
        for i in (1..p).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (seed >> 33) as usize % (i + 1);
            shuffled.swap(i, j);
        }
        for build_input in [&evals, &shuffled] {
            let batch = ChebCodeBatch::build(build_input, 0.3, 9, 16);
            let etas = [0.02, 0.05, 0.11];
            let many = batch.evaluate_many(&etas);
            for (k, &eta) in etas.iter().enumerate() {
                let single = compute_all_stieltjes_chebcode(build_input, eta);
                // Pairwise two-lane evaluation accepts a node only if BOTH
                // lanes' criteria pass (eta_min); at acceptance boundaries
                // this can differ from a lane's own single-eta traversal by
                // one node, bounded by that node's far-field error (~1e-9).
                for i in 0..p {
                    let dr = (many[k][i].0 - single[i].0).abs();
                    let di = (many[k][i].1 - single[i].1).abs();
                    assert!(
                        dr <= 5e-9 * (1.0 + single[i].0.abs()) && di <= 5e-9,
                        "batch mismatch at eta={eta} i={i}: {dr} {di}"
                    );
                }
            }
        }
    }

    #[test]
    fn chebcode_agrees_with_autovec() {
        for p in [512, 2000, 8000] {
            let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
            let eta = 0.1 / (p as f64).sqrt();

            let cb = compute_all_stieltjes_chebcode(&evals, eta);

            let mut max_r: f64 = 0.0;
            let mut max_i: f64 = 0.0;
            for i in 0..p {
                let (rr, ri) = autovec_stieltjes_sum(evals[i], &evals, eta);
                max_r = max_r.max((cb[i].0 - rr).abs() / rr.abs().max(1e-12));
                max_i = max_i.max((cb[i].1 - ri).abs() / ri.abs().max(1e-12));
            }
            assert!(max_r < 1e-2, "p={p} real err too large: {max_r}");
            assert!(max_i < 1e-2, "p={p} imag err too large: {max_i}");
        }
    }

    #[test]
    fn chebcode_parallel_matches_sequential() {
        let p = 4000;
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        let eta = 0.1 / (p as f64).sqrt();

        let seq = compute_all_stieltjes_chebcode_impl(&evals, eta, 0.3, 9, 16, false);
        let par = compute_all_stieltjes_chebcode_impl(&evals, eta, 0.3, 9, 16, true);
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

    #[test]
    fn chebcode_beats_treecode_accuracy() {
        let p = 3000;
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        let eta = 0.1 / (p as f64).sqrt();

        // Both return raw sums; compare against exact autovec reference.
        let cb = compute_all_stieltjes_chebcode_impl(&evals, eta, 0.3, 9, 16, false);
        let tc = compute_all_stieltjes_treecode_impl(&evals, eta, 0.3, 8, false);

        let mut cb_max_i: f64 = 0.0;
        let mut cb_max_r: f64 = 0.0;
        let mut tc_max_i: f64 = 0.0;
        for i in 0..p {
            let (rr, ri) = autovec_stieltjes_sum(evals[i], &evals, eta);
            cb_max_r = cb_max_r.max((cb[i].0 - rr).abs() / rr.abs().max(1e-12));
            cb_max_i = cb_max_i.max((cb[i].1 - ri).abs() / ri.abs().max(1e-12));
            tc_max_i = tc_max_i.max((tc[i].1 - ri).abs() / ri.abs().max(1e-12));
        }
        // Both are extremely accurate; ChebCode should be at least as good as
        // the multipole treecode used in production at the same settings.
        assert!(
            cb_max_i <= tc_max_i + 1e-6,
            "cb imag {cb_max_i} > tc {tc_max_i}"
        );
    }
}

#[cfg(test)]
mod tests_x2 {
    use super::*;

    /// The two-lane traversal must reproduce the scalar traversal when both
    /// lanes carry the same eta (up to acceptance-boundary noise), and stay
    /// within the far-field error bound against each eta's own single run.
    #[test]
    fn contribution_x2_matches_single_with_equal_lanes() {
        let p = 256;
        let evals: Vec<f64> = (0..p).map(|i| (i as f64 + 1.0).ln()).collect();
        let batch = ChebCodeBatch::build(&evals, 0.3, 9, 16);
        let eta = 0.05;
        let eta_v = F64x2::from_array([eta, eta]);
        let mut stack = Vec::with_capacity(64);
        for &x in evals.iter().step_by(17) {
            let s = batch.evaluate_point(x, eta);
            let ((r0, i0), _) = batch
                .tree
                .contribution_x2(x, eta_v, eta_v.mul(eta_v), &mut stack);
            assert!(
                (s.0 - r0).abs() < 1e-9 && (s.1 - i0).abs() < 1e-9,
                "x={x}: single=({},{}) x2=({r0},{i0})",
                s.0,
                s.1
            );
        }
    }
}
