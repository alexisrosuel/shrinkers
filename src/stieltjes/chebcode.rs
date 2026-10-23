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
//! - **Flat, structure-of-arrays tree** (parallel `Vec`s) for the precision
//!   path, plus a compact 24-byte AoS node record and interleaved 64-byte
//!   Chebyshev panels for the f32 speed path — cache-friendly, iterative
//!   traversal, no per-node heap allocations.
//! - **Leaf points are not copied**: each node stores an index range
//!   `[lo_idx, hi_idx)` into the sorted eigenvalue array, so the exact
//!   near-field sum reads contiguous memory.
//! - **Chebyshev nodes & weights are flattened** per node (`nodes[node*n+j]`,
//!   `w[node*n+j]`) for sequential access in the far-field dot product.
//! - Leaves evaluate **exactly** (handles the `z ≈ x` diagonal / singular
//!   near-field); well-separated internal nodes use the Chebyshev far-field.
//! - **Pre-order child links need no left pointer**: an internal node's left
//!   child is always `node + 1`, so the hot traversal loads one child index
//!   and one leaf flag per node instead of two.
//! - Per-query iterative walk with a reusable stack. Optional Rayon
//!   parallelism over the independent query points; the tree build's
//!   independent leaf-fill phase is parallelized too.

use super::simd::{F32x4, F64x2};
use rayon::prelude::*;

/// Arithmetic precision of the FAR FIELD and traversal metadata.
///
/// Leaves always sum exactly in f64, and both modes fold their partials into
/// f64 before returning. In [`FastMode::F32`] the well-separated Chebyshev
/// dot product `Σ_j w_j/(z−t_j)` and the node distance/acceptance state use
/// four-lane f32. The far field is already an interpolation carrying ~1e-5
/// relative error at the FAST preset, so f32's ~1e-7 per-term rounding is a
/// small addition — measured in `RESULT.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FastMode {
    /// Bit-identical to the historical f64 path.
    #[default]
    F64,
    /// Four-lane f32 far field (AArch64 NEON), f64 elsewhere.
    F32,
}

/// Chunk size for parallel query evaluation.
///
/// 128-query blocks: measured faster than 256 at p = 50k on M1 Max with the
/// f32 fast path (the plateau spans 64..1024), large enough to amortize
/// scheduling and small enough for load balance across the heterogeneous
/// P/E cores. Each task reuses one traversal stack across the block, and
/// adjacent (sorted) values traverse nearly identical paths, so per-task
/// cache warmth is high.
const EVAL_CHUNK: usize = 128;

/// Minimum query count for a Rayon evaluation of the ChebCode tree.
///
/// Below this, the thread-pool join dominates: the deconvolution grid serves
/// `nq = 200` queries by default, where sequential evaluation measured
/// 0.021 ms against 0.037 ms for a parallel request at p = 50 000. Same
/// policy as the exact family's `RAYON_MIN_P`.
const MIN_PARALLEL_QUERIES: usize = 2048;

/// Maximum recursion depth of the tree build — and therefore the hard bound on
/// the traversal stack.
///
/// Realistic spectra bottom out at depth ~10-17 for p <= 1e6, so this bound is
/// never reached in practice. It is a GUARANTEE for pathologically clustered
/// inputs: a node at this depth becomes an (oversized) exact leaf instead of
/// splitting further, so [`TraversalStack`]'s fixed buffer cannot overflow.
const MAX_TREE_DEPTH: usize = 48;

/// Squared opening-angle multiplier for f32 far-field panels that do NOT
/// contain the query point.
///
/// A panel containing `z` sees the pole at distance η, so its interpolation
/// error is the accuracy-critical one: those panels keep the preset's own θ.
/// A panel outside the query is separated by a real distance, which tolerates
/// a wider angle. `1.3²` was measured on MP, log-spaced and clustered spectra
/// at p = 5k..50k: it accepts ~10 % fewer panels (worth ~2–5 % end-to-end)
/// while leaving the worst-case relative-L2 error at the θ = 1 level
/// (~2e-5). Larger multipliers buy little more time and start moving the
/// error, so this is deliberately a constant rather than a preset knob.
const OUTSIDE_THETA2: f32 = 1.69;

/// Fixed-capacity DFS stack, replacing `Vec<i32>`.
///
/// A root-to-leaf walk pushes at most one pending sibling per level, so
/// `MAX_TREE_DEPTH + 1` slots always suffice ([`MAX_TREE_DEPTH`] is enforced
/// by the builder). The buffer lives inline: no heap indirection and no
/// capacity check on push. One instance is reused across a whole chunk of
/// queries.
#[derive(Clone)]
struct TraversalStack {
    buf: [i32; MAX_TREE_DEPTH + 2],
    top: usize,
}

impl TraversalStack {
    #[inline]
    fn new() -> Self {
        Self {
            buf: [0; MAX_TREE_DEPTH + 2],
            top: 0,
        }
    }
    #[inline]
    fn clear(&mut self) {
        self.top = 0;
    }
    #[inline]
    fn push(&mut self, v: i32) {
        self.buf[self.top] = v;
        self.top += 1;
    }
    #[inline]
    fn pop(&mut self) -> Option<i32> {
        if self.top == 0 {
            None
        } else {
            self.top -= 1;
            Some(self.buf[self.top])
        }
    }
}

/// Evaluate one tree at many query points, in parallel chunks.
///
/// Shared by `compute_all_stieltjes_chebcode_impl` and
/// [`ChebCodeBatch::evaluate_points`]; the output order matches `points`.
/// `FAST_F32` selects the far-field arithmetic (see [`FastMode`]).
fn eval_points_parallel<const FAST_F32: bool>(
    tree: &FlatChebTree,
    points: &[f64],
    eta: f64,
) -> Vec<(f64, f64)> {
    // Preallocate once and write in place: the previous form collected a
    // `Vec` per chunk and then flattened them (one extra allocation plus a
    // full p-element copy per call).
    let mut out = vec![(0.0f64, 0.0f64); points.len()];
    points
        .par_chunks(EVAL_CHUNK)
        .zip(out.par_chunks_mut(EVAL_CHUNK))
        .for_each(|(chunk, dst)| {
            let mut stack = TraversalStack::new();
            for (&x, d) in chunk.iter().zip(dst.iter_mut()) {
                *d = tree.contribution_mode::<FAST_F32>(x, eta, &mut stack);
            }
        });
    out
}

/// Compact f32 node record used by the [`FastMode::F32`] traversal.
///
/// `right == -1` marks a leaf; the left child of an internal node is the next
/// pre-order node (`+1`), so no left link is stored. `lo_idx`/`hi_idx` are
/// u32 because the tree never holds more sources than a slice can index
/// anyway; this keeps the record at 24 bytes, i.e. 5 records per 128-byte
/// Apple cache line.
#[derive(Clone, Copy)]
struct NodeF32 {
    lo: f32,
    hi: f32,
    accept_hw: f32,
    lo_idx: u32,
    hi_idx: u32,
    /// Right child, or -1 for a leaf. The left child of an internal node is
    /// always the next node in pre-order, so it needs no stored link.
    right: i32,
}

/// Flat, structure-of-arrays Chebyshev tree.
///
/// Node `i` has bounds `[lo[i], hi[i]]`, eigenvalue index range
/// `[lo_idx[i], hi_idx[i])` into `sorted`, Chebyshev nodes
/// `nodes[i*n .. i*n+n]`, source weights `w[i*n .. i*n+n]`, and children
/// `right[i]` (`-1` = leaf). The left child of an internal node is always
/// the next node in pre-order (`i + 1`). The root is node `0`.
/// The barycentric weights `lam` depend only on `n` and are shared.
struct FlatChebTree {
    /// Sorted eigenvalues (leaf exact sums read from here).
    sorted: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    lo_idx: Vec<usize>,
    hi_idx: Vec<usize>,
    /// Acceptance threshold per panel: `(0.5·(hi−lo))² / θ²`, precomputed so
    /// the per-visit test is `threshold < cl² + η²` — one load + one compare,
    /// with θ folded into the stored value at build time.
    accept_hw: Vec<f64>,
    /// Flattened Chebyshev nodes: `nodes[node*n + j]`.
    nodes: Vec<f64>,
    /// Flattened source weights: `w[node*n + j]`.
    w: Vec<f64>,
    /// f32 mirror of the traversal metadata for [`FastMode::F32`].
    ///
    /// One record per node packs the interval, acceptance test input, source
    /// range and right-child link into 24 bytes, so the whole hot decision
    /// state of a node fits in one cache line (the f64 arrays remain for the
    /// precision path and are never touched by the f32 traversal).
    node_f32: Vec<NodeF32>,
    /// f32 far-field panels, `2n` floats per node: the `n` Chebyshev nodes
    /// followed by the `n` weights.
    ///
    /// Keeping a panel's nodes and weights adjacent (64 bytes for n = 8)
    /// means the far-field loop touches one cache line per accepted panel
    /// instead of one line in each of two arrays.
    panel_f32: Vec<f32>,
    /// Right child per node (`-1` = leaf); see the struct docs.
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

/// Write node `idx`'s `n` Chebyshev positions `t_j = c + l·s_j`.
///
/// Split out of the weight passes so the topology builder can set every
/// position once; leaf fills and parent merges then only read them.
#[inline]
fn set_node_positions(tree: &mut FlatChebTree, idx: usize, sm1: &[f64]) {
    let n = tree.n;
    let lo = tree.lo[idx];
    let hi = tree.hi[idx];
    let c = 0.5 * (lo + hi);
    let l = 0.5 * (hi - lo);
    let base = idx * n;
    for (slot, &s) in tree.nodes[base..base + n].iter_mut().zip(sm1.iter()) {
        *slot = c + l * s;
    }
}

/// Accumulate one leaf's sources into its `n` barycentric weights.
///
/// `w` is the destination panel (overwritten), `t` its Chebyshev nodes,
/// `sources` this leaf's contiguous eigenvalue range and `v` an `n`-sized
/// scratch. A leaf's sources each carry mass 1.
#[inline]
fn fill_leaf_weights(w: &mut [f64], t: &[f64], sources: &[f64], lam: &[f64], v: &mut [f64]) {
    w.fill(0.0);
    for &x in sources {
        barycentric_row(x, t, lam, w, v, 1.0);
    }
}

/// One barycentric row update: accumulate the Lagrange basis at `x` into `w`,
/// scaled by `mass`.
///
/// `t[j]` are the panel's Chebyshev nodes, `lam[j]` the shared barycentric
/// weights, `v` a caller-owned `n`-sized scratch (no allocation here). The
/// two callers differ only in `mass`: a leaf passes `1.0` for every source it
/// owns ([`fill_leaf_weights`]), a parent passes each child's weight `w_t`
/// ([`merge_all_weights`]).
///
/// Division-hoisted: `v_j = λ_j/(x-t_j)` is computed once (n divisions),
/// `s = Σ v_j` normalizes, and the update is `w_j += mass·v_j·(1/s)` — one
/// extra division per POINT instead of one per (point, node). If `x` hits a
/// node exactly, the whole mass goes to that single basis. With `mass = 1.0`
/// this is bit-identical to the inlined form it replaces (`1.0 * v_j == v_j`).
///
#[inline(always)]
fn barycentric_row(x: f64, t: &[f64], lam: &[f64], w: &mut [f64], v: &mut [f64], mass: f64) {
    let n = t.len();
    debug_assert_eq!(lam.len(), n);
    debug_assert!(w.len() >= n && v.len() >= n);

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
        w[hit] += mass;
        return;
    }
    // `mass*(1/s)` is common to every node: hoisting it turns the original
    // `w_j += mass * v_j * inv_s` (FMUL + FMUL + FADD) into one hoisted FMUL
    // plus a single fused `mul_add` per node - one fewer rounding step too.
    let m = mass * (1.0 / s);
    for (wj, &vj) in w.iter_mut().zip(v.iter()).take(n) {
        *wj = m.mul_add(vj, *wj);
    }
}

/// Recursive TOPOLOGY pass: append this node, its interval metadata, its
/// child links and its Chebyshev positions.
///
/// Weights are computed afterwards (leaf fills and bottom-up merges), which
/// is what lets the weight phase run the independent leaf work in parallel
/// and the parent merges in a flat reverse-index sweep. Node indices are
/// pre-order, so every child has a larger index than its parent.
#[allow(clippy::too_many_arguments)]
fn build_topology(
    tree: &mut FlatChebTree,
    sm1: &[f64],
    lo_idx: usize,
    hi_idx: usize,
    leaf_cap: usize,
    depth: usize,
) -> i32 {
    let idx = tree.lo.len() as i32;
    let count = hi_idx - lo_idx;
    let lo = tree.sorted[lo_idx];
    let hi = tree.sorted[hi_idx - 1];

    tree.lo.push(lo);
    tree.hi.push(hi);
    tree.accept_hw
        .push((0.5 * (hi - lo)).powi(2) / tree.theta_sq);
    tree.lo_idx.push(lo_idx);
    tree.hi_idx.push(hi_idx);
    for _ in 0..tree.n {
        tree.nodes.push(0.0);
        tree.w.push(0.0);
    }
    tree.right.push(-1);
    set_node_positions(tree, idx as usize, sm1);

    if count <= leaf_cap || depth >= MAX_TREE_DEPTH {
        return idx;
    }

    let mid = lo + 0.5 * (hi - lo);
    let mut split = tree.sorted[lo_idx..hi_idx].partition_point(|&x| x <= mid) + lo_idx;
    if split == lo_idx || split == hi_idx {
        split = (lo_idx + hi_idx) / 2;
    }
    let li = build_topology(tree, sm1, lo_idx, split, leaf_cap, depth + 1);
    let ri = build_topology(tree, sm1, split, hi_idx, leaf_cap, depth + 1);
    debug_assert_eq!(li, idx + 1, "pre-order left child must be next");

    tree.right[idx as usize] = ri;
    idx
}

/// Maximum `n` whose per-leaf barycentric scratch fits on the stack.
///
/// Every shipped preset is far below this; the heap fallback keeps arbitrary
/// interpolation orders correct.
const STACK_SCRATCH: usize = 64;

/// Fill every leaf's weights, optionally in parallel.
///
/// Leaves write disjoint `n`-element slices of `w` and only read `nodes` /
/// `sorted`, so the parallel form needs no locks or atomics: one
/// `par_chunks_mut` over the panel array with non-leaves skipped.
fn fill_leaves(tree: &mut FlatChebTree, lam: &[f64], parallel: bool) {
    let n = tree.n;
    let use_parallel = parallel && tree.lo.len() >= 1024;
    if use_parallel {
        use rayon::prelude::*;
        // Group panels so each Rayon job owns several leaves: one task per
        // leaf (thousands of ~0.5 µs jobs) spends more time in the scheduler
        // than in the barycentric divisions it is meant to parallelize.
        const GROUP: usize = 32;
        let node_count = tree.lo.len();
        let FlatChebTree {
            w,
            nodes,
            sorted,
            lo_idx,
            hi_idx,
            right,
            ..
        } = tree;
        w.par_chunks_mut(n * GROUP)
            .enumerate()
            .for_each(|(g, chunk)| {
                let first = g * GROUP;
                let mut vb = [0.0f64; STACK_SCRATCH];
                let mut vheap: Vec<f64> = Vec::new();
                for k in 0..GROUP {
                    let idx = first + k;
                    if idx >= node_count {
                        break;
                    }
                    if right[idx] >= 0 {
                        continue;
                    }
                    let base = idx * n;
                    let t = &nodes[base..base + n];
                    let src = &sorted[lo_idx[idx]..hi_idx[idx]];
                    let wchunk = &mut chunk[k * n..(k + 1) * n];
                    if n <= STACK_SCRATCH {
                        fill_leaf_weights(wchunk, t, src, lam, &mut vb[..n]);
                    } else {
                        if vheap.len() < n {
                            vheap.resize(n, 0.0);
                        }
                        fill_leaf_weights(wchunk, t, src, lam, &mut vheap);
                    }
                }
            });
    } else {
        let mut v = vec![0.0f64; n];
        for idx in 0..tree.lo.len() {
            if tree.right[idx] >= 0 {
                continue;
            }
            let base = idx * n;
            let t = &tree.nodes[base..base + n];
            let src = &tree.sorted[tree.lo_idx[idx]..tree.hi_idx[idx]];
            fill_leaf_weights(&mut tree.w[base..base + n], t, src, lam, &mut v);
        }
    }
}

/// Compose every internal node's weights from its two children's weights.
///
/// Node indices are pre-order, so iterating them in REVERSE order is a valid
/// bottom-up sweep: both children of a node always have larger indices, and
/// are therefore already merged. This replaces the old recursive post-order
/// merge with a flat loop (same arithmetic, same order within each merge).
///
/// Each child node carries mass `w_t` (summing to its source count);
/// running the normalized barycentric row update per child node — scaled by
/// `w_t` — reproduces the parent's weights with O(n²) work per child rather
/// than O(count·n), which is why build cost scales with `p/leaf_cap` rather
/// than `p·depth`.
fn merge_all_weights(tree: &mut FlatChebTree, lam: &[f64]) {
    let n = tree.n;
    let node_count = tree.lo.len();
    let FlatChebTree {
        w, nodes, right, ..
    } = tree;
    let mut v = vec![0.0f64; n];
    for pi in (0..node_count).rev() {
        if right[pi] < 0 {
            continue;
        }
        // Destination panel, plus the tail that holds every later node's
        // weights. Both children have index > pi, so their panels live in
        // `rest` (child `ci` starts at offset `(ci - pi - 1)*n`).
        let (parent_w, rest) = w[pi * n..].split_at_mut(n);
        parent_w.fill(0.0);
        let t = &nodes[pi * n..pi * n + n];
        for child in [pi as i32 + 1, right[pi]] {
            let ci = child as usize;
            debug_assert!(ci > pi);
            let cbase = ci * n;
            let off = cbase - (pi * n + n);
            for ti in 0..n {
                let m = rest[off + ti];
                if m == 0.0 {
                    continue;
                }
                let x = nodes[cbase + ti];
                barycentric_row(x, t, lam, parent_w, &mut v, m);
            }
        }
    }
}

impl FlatChebTree {
    fn build(sorted: &[f64], n: usize, theta: f64, leaf_cap: usize, parallel_build: bool) -> Self {
        // Barycentric weights depend only on `n`; they are consumed during
        // the build, so they live outside the (hot) query struct.
        let lam = barycentric_weights_m1(n);
        // Reserve for the TYPICAL tree, not the pathological one: a balanced
        // value-split tree has ~p/leaf_cap leaves, so ~2p/leaf_cap nodes.
        // Reserving `2*p` for every array (the old form) over-allocated by
        // ~20x at p=50k, i.e. ~28 MB of untouched address space per build.
        // The 4x safety factor covers value splits that produce more leaves
        // than the count-balanced estimate; Vec growth is the rare fallback.
        let est_nodes = (4 * sorted.len() / leaf_cap.max(1) + 32)
            .min(2 * sorted.len())
            .max(1);
        let mut tree = FlatChebTree {
            sorted: sorted.to_vec(),
            lo: Vec::with_capacity(est_nodes),
            hi: Vec::with_capacity(est_nodes),
            lo_idx: Vec::with_capacity(est_nodes),
            hi_idx: Vec::with_capacity(est_nodes),
            accept_hw: Vec::with_capacity(est_nodes),
            nodes: Vec::with_capacity(est_nodes * n),
            w: Vec::with_capacity(est_nodes * n),
            node_f32: Vec::new(),
            panel_f32: Vec::with_capacity(est_nodes * n * 2),
            right: Vec::with_capacity(est_nodes),
            n,
            theta_sq: theta * theta,
        };
        // Shared per-build data: Chebyshev nodes on [-1,1] (the per-node
        // positions are affine rescalings).
        assert!(n >= 2, "ChebCode needs at least 2 Chebyshev nodes, got {n}");
        let sm1 = cheb_nodes_m1(n);
        build_topology(&mut tree, &sm1, 0, sorted.len(), leaf_cap, 0);
        // Leaf fills are independent; parent merges need their children, so
        // they run as a flat reverse-index sweep afterwards. Ordering within
        // each merge is unchanged, so this is the same arithmetic as the old
        // recursive post-order build.
        fill_leaves(&mut tree, &lam, parallel_build);
        merge_all_weights(&mut tree, &lam);
        // f32 mirrors of the traversal metadata and panels for
        // `FastMode::F32`. One extra pass over the (small) node/panel arrays;
        // the f64 arrays are unchanged, so the f64 path stays bit-identical.
        tree.node_f32 = (0..tree.lo.len())
            .map(|i| NodeF32 {
                lo: tree.lo[i] as f32,
                hi: tree.hi[i] as f32,
                accept_hw: tree.accept_hw[i] as f32,
                lo_idx: tree.lo_idx[i] as u32,
                hi_idx: tree.hi_idx[i] as u32,
                right: tree.right[i],
            })
            .collect();
        let node_count = tree.lo.len();
        tree.panel_f32 = vec![0.0f32; node_count * 2 * n];
        for i in 0..node_count {
            let src = i * n;
            let dst = i * 2 * n;
            for j in 0..n {
                tree.panel_f32[dst + j] = tree.nodes[src + j] as f32;
                tree.panel_f32[dst + n + j] = tree.w[src + j] as f32;
            }
        }
        tree
    }

    /// Stieltjes contribution at query z=(lambda_i, -eta): per-query iterative
    /// walk. Returns (real, imag) raw sums (not scaled by 1/p).
    #[inline]
    fn contribution(&self, lambda_i: f64, eta: f64, stack: &mut TraversalStack) -> (f64, f64) {
        self.contribution_mode::<false>(lambda_i, eta, stack)
    }

    /// [`Self::contribution`] with the four-lane f32 far field
    /// ([`FastMode::F32`]). Leaves, traversal and accumulators are unchanged.
    #[inline]
    fn contribution_f32(&self, lambda_i: f64, eta: f64, stack: &mut TraversalStack) -> (f64, f64) {
        self.contribution_mode::<true>(lambda_i, eta, stack)
    }

    /// Shared traversal; `FAST_F32` selects the far-field arithmetic width.
    /// Monomorphized, so the `false` instantiation compiles to exactly the
    /// historical f64 code (same operations, same order → bit-identical).
    #[inline]
    fn contribution_mode<const FAST_F32: bool>(
        &self,
        lambda_i: f64,
        eta: f64,
        stack: &mut TraversalStack,
    ) -> (f64, f64) {
        if FAST_F32 {
            self.contribution_fast32(lambda_i, eta, stack)
        } else {
            self.contribution_f64(lambda_i, eta, stack)
        }
    }

    /// Full-precision traversal: f64 near field and f64 far field.
    #[inline]
    fn contribution_f64(&self, lambda_i: f64, eta: f64, stack: &mut TraversalStack) -> (f64, f64) {
        let n = self.n;
        stack.clear();
        stack.push(0);

        let mut re = 0.0;
        let mut im = 0.0;
        let eta2 = eta * eta;

        while let Some(node) = stack.pop() {
            let ni = node as usize;
            // The builder writes `right = -1` only for leaves, so one sign
            // test is enough to identify one.
            if self.right[ni] < 0 {
                // Exact pairwise sum over the leaf's sources (contiguous range
                // into the sorted array — cache friendly).
                let lo_idx = self.lo_idx[ni];
                let hi_idx = self.hi_idx[ni];
                let xs = &self.sorted[lo_idx..hi_idx];
                let zv = F64x2::splat(lambda_i);
                let eta2v = F64x2::splat(eta2);
                let (mut ar, mut ai) = (F64x2::zero(), F64x2::zero());
                let mut k = 0;
                while k + 2 <= xs.len() {
                    let d = zv - F64x2::load(xs, k);
                    let inv = eta2v.fma(d, d).recip();
                    ar = ar.fma(d, inv);
                    ai = ai.add(inv);
                    k += 2;
                }
                let mut lr = ar.hsum();
                let mut li = ai.hsum();
                while k < xs.len() {
                    let d = lambda_i - xs[k];
                    let inv = 1.0 / (d * d + eta2);
                    lr += d * inv;
                    li += inv;
                    k += 1;
                }
                re += lr;
                im += eta * li;
                continue;
            }

            // Distance from z to the interval [lo,hi] — branchless clamp.
            let lo = self.lo[ni];
            let hi = self.hi[ni];
            let cl = (lo - lambda_i).max(lambda_i - hi).max(0.0);
            let d_sq = cl.mul_add(cl, eta2);

            if self.accept_hw[ni] < d_sq {
                // Well-separated: Chebyshev far-field
                //     F(z) = Σ_j w_j / (z - t_j),
                // evaluated term-by-term (n reciprocal magnitudes).
                //
                // Why per-term evaluation rather than polynomial
                // coefficients P/Q via a single Horner division? Monomial
                // coefficients of degree-n polynomials whose roots cluster
                // near ±1 amplify rounding catastrophically (measured
                // relative errors ~10²–10³), while this dot product
                // evaluates every denominator exactly like the leaf
                // near-field loop — same conditioning, unconditional
                // stability.
                let nbase = ni * n;
                let ts = &self.nodes[nbase..nbase + n];
                let ws = &self.w[nbase..nbase + n];
                let zv = F64x2::splat(lambda_i);
                let eta2v = F64x2::splat(eta2);
                let (mut afr, mut afi) = (F64x2::zero(), F64x2::zero());
                let mut j = 0;
                while j + 2 <= n {
                    let d = zv - F64x2::load(ts, j);
                    let inv = eta2v.fma(d, d).recip();
                    let q = F64x2::load(ws, j) * inv;
                    afr = afr.fma(q, d);
                    afi = afi.add(q);
                    j += 2;
                }
                let mut fr = afr.hsum();
                let mut fi = afi.hsum();
                while j < n {
                    // Scalar tail (n is typically odd).
                    let dj = lambda_i - ts[j];
                    let q = ws[j] / (dj * dj + eta2);
                    fr += q * dj;
                    fi += q;
                    j += 1;
                }
                re += fr;
                im += eta * fi;
            } else {
                // Otherwise descend. Push right first so the left (lower
                // values) subtree is visited first: queries arrive sorted, so
                // this keeps source reads closer to sequential.
                // Pre-order layout: the left child is node+1.
                let r = self.right[ni];
                if r >= 0 {
                    stack.push(r);
                    stack.push(node + 1);
                }
            }
        }

        (re, im)
    }

    /// f32 far-field traversal (the FAST preset).
    ///
    /// Everything the branchy traversal touches — node bounds, the
    /// acceptance threshold, the right-child link and the source range —
    /// lives in one 24-byte [`NodeF32`] record, and each accepted panel's
    /// nodes and weights are adjacent. Far-field partial sums stay in
    /// `F32x4` registers across panels and are folded into f64 once per
    /// query; leaves still sum exactly in f64.
    #[inline]
    fn contribution_fast32(
        &self,
        lambda_i: f64,
        eta: f64,
        stack: &mut TraversalStack,
    ) -> (f64, f64) {
        let n = self.n;
        stack.clear();
        stack.push(0);

        let x = lambda_i as f32;
        let e32 = eta as f32;
        let eta2 = e32 * e32;
        let eta2v = F32x4::splat(eta2);
        let zv = F32x4::splat(x);

        let mut far_re = F32x4::zero();
        let mut far_im = F32x4::zero();
        let mut leaf_re = 0.0f64;
        let mut leaf_im = 0.0f64;

        while let Some(node) = stack.pop() {
            let nd = &self.node_f32[node as usize];
            if nd.right < 0 {
                let lo_idx = nd.lo_idx as usize;
                let hi_idx = nd.hi_idx as usize;
                let xs = &self.sorted[lo_idx..hi_idx];
                let zvd = F64x2::splat(lambda_i);
                let eta2vd = F64x2::splat(eta * eta);
                let (mut ar, mut ai) = (F64x2::zero(), F64x2::zero());
                let mut k = 0;
                while k + 2 <= xs.len() {
                    let d = zvd - F64x2::load(xs, k);
                    let inv = eta2vd.fma(d, d).recip();
                    ar = ar.fma(d, inv);
                    ai = ai.add(inv);
                    k += 2;
                }
                let mut lr = ar.hsum();
                let mut li = ai.hsum();
                while k < xs.len() {
                    let d = lambda_i - xs[k];
                    let inv = 1.0 / (d * d + eta * eta);
                    lr += d * inv;
                    li += inv;
                    k += 1;
                }
                leaf_re += lr;
                leaf_im += eta * li;
                continue;
            }

            let cl = (nd.lo - x).max(x - nd.hi).max(0.0);
            let d_sq = cl.mul_add(cl, eta2);
            // Panels that CONTAIN the query are the accuracy-critical ones;
            // panels outside it tolerate a wider angle (see the constant).
            let outside = if cl == 0.0 {
                d_sq
            } else {
                OUTSIDE_THETA2 * d_sq
            };
            if nd.accept_hw < outside {
                let base = node as usize * 2 * n;
                let ts = &self.panel_f32[base..base + n];
                let ws = &self.panel_f32[base + n..base + 2 * n];
                // n is a runtime preset value. n = 8 (the FAST preset) is
                // two independent 4-lane iterations; a dedicated unrolled
                // n=8 branch measured within noise of this loop, so the
                // generic form is kept (less code, better I-cache).
                let mut j = 0;
                while j + 4 <= n {
                    let d = zv - F32x4::load(ts, j);
                    let inv = eta2v.fma(d, d).recip_fast();
                    let q = F32x4::load(ws, j) * inv;
                    far_re = far_re.fma(q, d);
                    far_im = far_im.add(q);
                    j += 4;
                }
                // n not a multiple of 4: pack the remaining nodes into
                // ONE more 4-lane op with zero weights in the dead lanes
                // (measured faster than a scalar `1.0f32/x` tail, whose
                // hardware divide does not pipeline like FRECPE).
                if j < n {
                    let mut dt = [0.0f32; 4];
                    let mut wt = [0.0f32; 4];
                    for k in 0..(n - j) {
                        dt[k] = x - ts[j + k];
                        wt[k] = ws[j + k];
                    }
                    let d = F32x4::from_array(dt);
                    let inv = eta2v.fma(d, d).recip_fast();
                    let q = F32x4::from_array(wt) * inv;
                    far_re = far_re.fma(q, d);
                    far_im = far_im.add(q);
                }
            } else {
                // Pre-order layout: the left child is node+1.
                stack.push(nd.right);
                stack.push(node + 1);
            }
        }

        (
            leaf_re + far_re.hsum() as f64,
            leaf_im + eta * far_im.hsum() as f64,
        )
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
        stack: &mut TraversalStack,
    ) -> ((f64, f64), (f64, f64)) {
        let n = self.n;
        stack.clear();
        stack.push(0);

        let mut acc_re = F64x2::zero();
        let mut acc_im = F64x2::zero();

        while let Some(node) = stack.pop() {
            let ni = node as usize;
            let is_leaf = self.right[ni] < 0;

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
                    // Full 53-bit reciprocal: `evaluate_many` is the
                    // γ-sweep path, where the batch result must stay
                    // accuracy-grade.
                    let inv = den.recip();
                    acc_re = acc_re.fma(wj.mul(dj), inv);
                    acc_im = acc_im.fma(wj.mul(eta_v), inv);
                }
            } else {
                // Pre-order layout: the left child is node+1.
                let r = self.right[ni];
                if r >= 0 {
                    stack.push(r);
                    stack.push(node + 1);
                }
            }
        }
        (
            (acc_re.lane(0), acc_im.lane(0)),
            (acc_re.lane(1), acc_im.lane(1)),
        )
    }
}

/// ChebCode operating point: opening angle, interpolation order, leaf capacity.
///
/// These three numbers fully parameterize the treecode's speed/accuracy
/// trade-off. The three associated constants are the MEASURED presets
/// (Apple M1 Max, MP spectra — see CHANGELOG); they are the single source
/// of truth used by the dispatch table, the Python bindings and the
/// benchmarks alike. Do not hardcode tuples elsewhere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChebPreset {
    /// Opening angle: larger = more far-field acceptance = faster, less accurate.
    pub theta: f64,
    /// Chebyshev interpolation order per panel.
    pub n: usize,
    /// Maximum sources per leaf (leaves are summed exactly).
    pub leaf_cap: usize,
    /// Far-field arithmetic. [`FastMode::F32`] is 4-lane NEON f32 (≈1.3x on
    /// the far field, ~1e-6 relative-error floor); [`FastMode::F64`] is the
    /// full-precision path every accuracy preset needs.
    pub mode: FastMode,
}

impl ChebPreset {
    /// Dispatch default (`method = "chebcode"`): dominates the historical
    /// (0.3, 9, 16) on BOTH runtime and error at every size.
    pub const DEFAULT: Self = Self {
        theta: 0.5,
        n: 11,
        leaf_cap: 32,
        mode: FastMode::F64,
    };
    /// Speed preset (`"chebcode_fast"` / `"chebf"`): the fastest measured
    /// operating point of the family, ~2.4x the historical (0.5, 9) point at
    /// a ~2e-5-class relative-L2 error (theta 1.0, n 8 — a multiple of the
    /// 4-lane f32 width — leaf 32, f32 far field).
    pub const FAST: Self = Self {
        theta: 1.0,
        n: 8,
        leaf_cap: 32,
        mode: FastMode::F32,
    };
    /// Precision preset (`"chebcode_xtreme"` / `"chebx"`): ~1e-12 class
    /// without paying the full exact O(p²).
    pub const XTREME: Self = Self {
        theta: 0.25,
        n: 11,
        leaf_cap: 16,
        mode: FastMode::F64,
    };
    /// Balanced preset (`"chebcode_balanced"` / `"chebb"`): ~3e-10 error at
    /// roughly the FAST price — measured round-2 operating point.
    pub const BALANCED: Self = Self {
        theta: 0.55,
        n: 11,
        leaf_cap: 32,
        mode: FastMode::F64,
    };

    #[inline]
    pub const fn parts(self) -> (f64, usize, usize) {
        (self.theta, self.n, self.leaf_cap)
    }

    /// `parts()` plus the far-field arithmetic mode.
    #[inline]
    pub const fn parts_mode(self) -> (f64, usize, usize, FastMode) {
        (self.theta, self.n, self.leaf_cap, self.mode)
    }

    /// The preset attached to a `ChebCode*` dispatch variant, or `None` for
    /// any other method. Single resolution point for the dispatcher and the
    /// point-evaluation path (previously duplicated in both).
    pub const fn from_method(method: crate::config::StieltjesMethod) -> Option<Self> {
        match method {
            crate::config::StieltjesMethod::ChebCode => Some(Self::DEFAULT),
            crate::config::StieltjesMethod::ChebCodeFast => Some(Self::FAST),
            crate::config::StieltjesMethod::ChebCodeXtreme => Some(Self::XTREME),
            crate::config::StieltjesMethod::ChebCodeBalanced => Some(Self::BALANCED),
            _ => None,
        }
    }
}

/// Compute all Stieltjes transforms with the Chebyshev treecode at the
/// [`ChebPreset::DEFAULT`] operating point.
///
/// Returns **raw sums** (not scaled by `1/p`); the caller applies scaling.
///
/// Public for the benchmark harnesses in `examples/` and `benches/`, which
/// call one kernel directly to A/B it. The supported entry point for
/// callers is the dispatcher (`compute_all_stieltjes`), which resolves
/// `StieltjesMethod` to the right kernel and applies the `1/p` scaling.
pub fn compute_all_stieltjes_chebcode(eigenvalues: &[f64], eta: f64) -> Vec<(f64, f64)> {
    let (theta, n, leaf) = ChebPreset::DEFAULT.parts();
    compute_all_stieltjes_chebcode_impl(eigenvalues, eta, theta, n, leaf, false)
}

/// Same at an explicit [`ChebPreset`], sequential or Rayon-parallel.
pub(crate) fn compute_all_stieltjes_chebcode_preset(
    eigenvalues: &[f64],
    eta: f64,
    preset: ChebPreset,
    parallel: bool,
) -> Vec<(f64, f64)> {
    let (theta, n, leaf, mode) = preset.parts_mode();
    match mode {
        FastMode::F64 => {
            compute_all_stieltjes_chebcode_impl(eigenvalues, eta, theta, n, leaf, parallel)
        }
        FastMode::F32 => {
            compute_all_stieltjes_chebcode_impl_f32(eigenvalues, eta, theta, n, leaf, parallel)
        }
    }
}

/// Chebyshev treecode with explicit opening-angle `theta`, node count `n`,
/// leaf capacity `leaf_cap`, and optional Rayon parallelism.
///
/// Returns **raw sums** (not scaled by `1/p`).
/// `parallel = true` parallelizes over query points (each is independent).
///
/// Public for the benchmark harnesses in `examples/` and `benches/`, which
/// call one kernel directly to A/B it. The supported entry point for
/// callers is the dispatcher (`compute_all_stieltjes`), which resolves
/// `StieltjesMethod` to the right kernel and applies the `1/p` scaling.
pub fn compute_all_stieltjes_chebcode_impl(
    eigenvalues: &[f64],
    eta: f64,
    theta: f64,
    n: usize,
    leaf_cap: usize,
    parallel: bool,
) -> Vec<(f64, f64)> {
    compute_all_stieltjes_chebcode_impl_mode::<false>(
        eigenvalues,
        eta,
        theta,
        n,
        leaf_cap,
        parallel,
    )
}

/// [`compute_all_stieltjes_chebcode_impl`] with the four-lane **f32
/// far-field** ([`FastMode::F32`]) and everything else f64.
///
/// Same arguments and same return contract (raw sums). The tree is built
/// exactly as in the f64 entry point and the f32 node/weight mirror is
/// derived from it, so the two entry points differ ONLY in the far-field
/// arithmetic — which is what makes them A/B-able from one binary.
pub fn compute_all_stieltjes_chebcode_impl_f32(
    eigenvalues: &[f64],
    eta: f64,
    theta: f64,
    n: usize,
    leaf_cap: usize,
    parallel: bool,
) -> Vec<(f64, f64)> {
    compute_all_stieltjes_chebcode_impl_mode::<true>(eigenvalues, eta, theta, n, leaf_cap, parallel)
}

/// Shared implementation for both [`FastMode`]s; see the wrappers above.
fn compute_all_stieltjes_chebcode_impl_mode<const FAST_F32: bool>(
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

    let tree = FlatChebTree::build(sorted, n, theta, leaf_cap, parallel);

    // Below `MIN_PARALLEL_QUERIES`, a Rayon request runs sequentially: the
    // join costs more than the whole evaluation.
    if parallel && p >= MIN_PARALLEL_QUERIES {
        eval_points_parallel::<FAST_F32>(&tree, eigenvalues, eta)
    } else {
        let mut stack = TraversalStack::new();
        let mut result = Vec::with_capacity(p);
        for &lambda_i in eigenvalues {
            let (re, im) = tree.contribution_mode::<FAST_F32>(lambda_i, eta, &mut stack);
            result.push((re, im));
        }
        result
    }
}

/// Build just the tree at an explicit preset — benchmarking helper exposing
/// the build/eval split.
pub fn chebcode_tree_for_bench(eigenvalues: &[f64], preset: ChebPreset) -> ChebCodeBatch {
    ChebCodeBatch::build_preset(eigenvalues, preset)
}

/// [`chebcode_tree_for_bench`] with an explicit build-parallelism choice.
///
/// Benchmark helper: lets the harnesses separate the sequential and parallel
/// build phases without going through a full evaluation.
pub fn chebcode_tree_for_bench_parallel(
    eigenvalues: &[f64],
    preset: ChebPreset,
    parallel_build: bool,
) -> ChebCodeBatch {
    let (theta, n, leaf_cap, mode) = preset.parts_mode();
    ChebCodeBatch::build_mode_parallel(eigenvalues, theta, n, leaf_cap, mode, parallel_build)
}

/// Amortized multi-η driver for ChebCode (γ-sweeps).
///
/// The Chebyshev tree depends only on the sorted spectrum, the interpolation
/// order and the opening angle — not on η — so one build serves any number
/// of evaluations. Beyond skipping repeated builds, [`Self::evaluate_many`]
/// parallelizes ACROSS the η values with the tree shared read-only, which is
/// the dominant win when a RIE workflow sweeps many γ (each γ fixes one η).
pub struct ChebCodeBatch {
    tree: FlatChebTree,
    perm: Option<Vec<u32>>,
    /// Far-field mode this batch was built for; every `evaluate*` call that
    /// does not take an explicit [`FastMode`] uses it.
    mode: FastMode,
}

impl ChebCodeBatch {
    /// Build the tree once at an explicit [`ChebPreset`].
    pub fn build_preset(eigenvalues: &[f64], preset: ChebPreset) -> Self {
        let (theta, n, leaf_cap, mode) = preset.parts_mode();
        Self::build_mode(eigenvalues, theta, n, leaf_cap, mode)
    }

    /// Build the tree once. Results of every [`Self::evaluate`] call are
    /// indexed like `eigenvalues`. Full-precision far field ([`FastMode::F64`]).
    pub fn build(eigenvalues: &[f64], theta: f64, n: usize, leaf_cap: usize) -> Self {
        Self::build_mode(eigenvalues, theta, n, leaf_cap, FastMode::F64)
    }

    /// [`Self::build`] with an explicit far-field arithmetic mode.
    pub fn build_mode(
        eigenvalues: &[f64],
        theta: f64,
        n: usize,
        leaf_cap: usize,
        mode: FastMode,
    ) -> Self {
        Self::build_mode_parallel(eigenvalues, theta, n, leaf_cap, mode, false)
    }

    /// [`Self::build_mode`] with an explicit build-parallelism choice.
    ///
    /// Only the dispatcher's parallel branches pass `true`: the leaf-weight
    /// phase (the largest part of the build) then runs on the Rayon pool.
    /// Public batch builders stay sequential so constructing a tree never
    /// spawns threads behind the caller's back.
    pub(crate) fn build_mode_parallel(
        eigenvalues: &[f64],
        theta: f64,
        n: usize,
        leaf_cap: usize,
        mode: FastMode,
        parallel_build: bool,
    ) -> Self {
        let already_sorted = crate::stieltjes::is_sorted_ascending(eigenvalues);
        if already_sorted {
            return Self {
                tree: FlatChebTree::build(eigenvalues, n, theta, leaf_cap, parallel_build),
                perm: None,
                mode,
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
            tree: FlatChebTree::build(&sorted_buf, n, theta, leaf_cap, parallel_build),
            perm: Some(idx),
            mode,
        }
    }

    /// The far-field mode this batch uses by default.
    pub fn mode(&self) -> FastMode {
        self.mode
    }

    /// Evaluate the sum for a single query point at one η.
    pub fn evaluate_point(&self, x: f64, eta: f64) -> (f64, f64) {
        let mut stack = TraversalStack::new();
        match self.mode {
            FastMode::F64 => self.tree.contribution(x, eta, &mut stack),
            FastMode::F32 => self.tree.contribution_f32(x, eta, &mut stack),
        }
    }

    /// [`Self::evaluate_point`] with the four-lane f32 far field.
    pub fn evaluate_point_f32(&self, x: f64, eta: f64) -> (f64, f64) {
        let mut stack = TraversalStack::new();
        self.tree.contribution_f32(x, eta, &mut stack)
    }

    /// Evaluate the sum at ARBITRARY query points (e.g. a deconvolution
    /// grid), one tree serving them all. Chunked Rayon optional.
    pub fn evaluate_points(&self, points: &[f64], eta: f64, parallel: bool) -> Vec<(f64, f64)> {
        self.evaluate_points_mode(points, eta, parallel, self.mode)
    }

    /// [`Self::evaluate_points`] with an explicit [`FastMode`].
    ///
    /// A `parallel` request below `MIN_PARALLEL_QUERIES` queries runs
    /// sequentially: the Rayon join costs more than the whole evaluation at
    /// the deconvolution grid's default `nq = 200` (measured 0.037 ms
    /// parallel vs 0.021 ms sequential at p = 50 000).
    pub fn evaluate_points_mode(
        &self,
        points: &[f64],
        eta: f64,
        parallel: bool,
        mode: FastMode,
    ) -> Vec<(f64, f64)> {
        let parallel = parallel && points.len() >= MIN_PARALLEL_QUERIES;
        match (mode, parallel) {
            (FastMode::F64, false) => {
                let mut stack = TraversalStack::new();
                points
                    .iter()
                    .map(|&x| self.tree.contribution(x, eta, &mut stack))
                    .collect()
            }
            (FastMode::F32, false) => {
                let mut stack = TraversalStack::new();
                points
                    .iter()
                    .map(|&x| self.tree.contribution_f32(x, eta, &mut stack))
                    .collect()
            }
            (FastMode::F64, true) => eval_points_parallel::<false>(&self.tree, points, eta),
            (FastMode::F32, true) => eval_points_parallel::<true>(&self.tree, points, eta),
        }
    }

    /// Evaluate all sums for one η.
    pub fn evaluate(&self, eta: f64) -> Vec<(f64, f64)> {
        self.evaluate_mode(eta, self.mode)
    }

    /// [`Self::evaluate`] with an explicit [`FastMode`].
    pub fn evaluate_mode(&self, eta: f64, mode: FastMode) -> Vec<(f64, f64)> {
        let mut stack = TraversalStack::new();
        let flat: Vec<(f64, f64)> = self
            .tree
            .sorted
            .iter()
            .map(|&x| match mode {
                FastMode::F64 => self.tree.contribution(x, eta, &mut stack),
                FastMode::F32 => self.tree.contribution_f32(x, eta, &mut stack),
            })
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
                let mut stack = TraversalStack::new();
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
        let evals = crate::stieltjes::testutil::log_spectrum(p);
        let shuffled = crate::stieltjes::testutil::shuffled(&evals, 7);
        for build_input in [&evals, &shuffled] {
            // Same preset on both sides: the wrapper now runs DEFAULT.
            let batch = ChebCodeBatch::build_preset(build_input, ChebPreset::DEFAULT);
            let etas = [0.02, 0.05, 0.11];
            let many = batch.evaluate_many(&etas);
            for (k, &eta) in etas.iter().enumerate() {
                let single = compute_all_stieltjes_chebcode(build_input, eta);
                // Pairwise two-lane evaluation accepts a node only if BOTH
                // lanes' criteria pass (eta_min); at acceptance boundaries
                // this can differ from a lane's own single-eta traversal by
                // one node, bounded by that node's far-field error — i.e.
                // by the preset's OWN accuracy scale (~1e-8 here).
                for i in 0..p {
                    let dr = (many[k][i].0 - single[i].0).abs();
                    let di = (many[k][i].1 - single[i].1).abs();
                    assert!(
                        dr <= 5e-8 * (1.0 + single[i].0.abs()) && di <= 5e-8,
                        "batch mismatch at eta={eta} i={i}: {dr} {di}"
                    );
                }
            }
        }
    }

    #[test]
    fn chebcode_agrees_with_autovec() {
        for p in [512, 2000, 8000] {
            let evals = crate::stieltjes::testutil::log_spectrum(p);
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
    fn chebcode_supports_node_count_above_legacy_scratch() {
        // Regression: the barycentric scratch used to be a fixed [f64; 64]
        // guarded only by a debug_assert, so n > 64 silently summed just the
        // first 64 Chebyshev nodes in release builds.
        let p = 500;
        let evals = crate::stieltjes::testutil::log_spectrum(p);
        let eta = 0.1 / (p as f64).sqrt();
        let exact = crate::stieltjes::testutil::exact_stieltjes(&evals, eta);

        for n in [64usize, 65, 96] {
            let got = compute_all_stieltjes_chebcode_impl(&evals, eta, 0.5, n, 32, false);
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for i in 0..p {
                num += (got[i].0 - exact[i].0).powi(2) + (got[i].1 - exact[i].1).powi(2);
                den += exact[i].0.powi(2) + exact[i].1.powi(2);
            }
            let rel = (num / den).sqrt();
            assert!(rel < 1e-6, "n={n} rel-L2 {rel:.3e}");
        }
    }

    #[test]
    fn chebcode_parallel_matches_sequential() {
        let p = 4000;
        let evals = crate::stieltjes::testutil::log_spectrum(p);
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
    fn chebcode_f32_far_field_tracks_f64() {
        // The f32 path differs from f64 ONLY in the far field; leaves and the
        // traversal are shared. Its deviation from f64 must stay inside the
        // documented relaxed budget, and both must remain near the exact sum.
        let p = 4000;
        let evals = crate::stieltjes::testutil::log_spectrum(p);
        let eta = 0.1 / (p as f64).sqrt();
        let a = compute_all_stieltjes_chebcode_impl(&evals, eta, 0.5, 9, 32, false);
        let b = compute_all_stieltjes_chebcode_impl_f32(&evals, eta, 0.5, 9, 32, false);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            num += (x.0 - y.0).powi(2) + (x.1 - y.1).powi(2);
            den += x.0.powi(2) + x.1.powi(2);
        }
        let rel = (num / den).sqrt();
        assert!(rel < 1e-4, "f32 vs f64 rel-L2 {rel:.3e}");

        // Parallel f32 must be bit-identical to sequential f32 (same kernel).
        let bp = compute_all_stieltjes_chebcode_impl_f32(&evals, eta, 0.5, 9, 32, true);
        for i in 0..p {
            assert_eq!(b[i], bp[i], "seq/par f32 mismatch at {i}");
        }
    }

    #[test]
    fn parallel_build_is_bit_identical() {
        // Leaf fills are disjoint, so the parallel build must produce exactly
        // the sequential tree — for both far-field arithmetics.
        let p = 50_000;
        let evals = crate::stieltjes::testutil::log_spectrum(p);
        let eta = 1.0 / (p as f64).sqrt();
        for mode in [FastMode::F64, FastMode::F32] {
            let seq = ChebCodeBatch::build_mode(&evals, 1.0, 8, 32, mode);
            let par = ChebCodeBatch::build_mode_parallel(&evals, 1.0, 8, 32, mode, true);
            assert_eq!(
                seq.evaluate(eta),
                par.evaluate(eta),
                "parallel build changed {mode:?} results"
            );
        }
    }

    #[test]
    fn chebcode_beats_treecode_accuracy() {
        let p = 3000;
        let evals = crate::stieltjes::testutil::log_spectrum(p);
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
        let evals = crate::stieltjes::testutil::log_spectrum(p);
        let batch = ChebCodeBatch::build(&evals, 0.3, 9, 16);
        let eta = 0.05;
        let eta_v = F64x2::from_array([eta, eta]);
        let mut stack = TraversalStack::new();
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
