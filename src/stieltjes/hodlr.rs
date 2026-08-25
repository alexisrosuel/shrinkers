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

/// RandNLA compression of one kernel block (Randomized SVD with sampled
/// validation, Halko–Martinsson–Tropp flavor adapted to implicit blocks):
///
/// 1. evaluate ℓ random columns of the block and orthonormalize them
///    (modified Gram–Schmidt with reorthogonalization) → `Q` spans the
///    dominant column space;
/// 2. fit the row space on 2ℓ random rows by least squares in the normal
///    equations (`Qᵣ` is orthonormal-column, well conditioned);
/// 3. validate against fresh random probe entries disjoint from the fitting
///    rows; double ℓ while the estimated relative residual exceeds `tol`.
///
/// Deterministic for a given block (RNG seeded from `seed`) so sequential
/// and parallel runs produce identical factors.
fn rand_block(
    tgt: &[f64],
    src: &[f64],
    eta: f64,
    eta_sq: f64,
    tol: f64,
    max_rank: usize,
    seed: u64,
) -> AcaFactors {
    #[inline(always)]
    fn entry(x: f64, y: f64, eta: f64, eta_sq: f64) -> (f64, f64) {
        let d = x - y;
        let inv = 1.0 / (d * d + eta_sq);
        (d * inv, eta * inv)
    }

    let m = tgt.len();
    let n = src.len();
    let mut rng = Rng::new(seed);

    // Extra random test columns for validation (besides the deterministic
    // boundary strips).
    const N_PROBE_COLS: usize = 8;

    let start_rank = if tol >= 1e-5 {
        8
    } else if tol >= 1e-8 {
        16
    } else {
        24
    };
    // `req` is the requested size fed to sampling; `ell` is what actually
    // came back after deduplication. The exit condition tracks `req` so a
    // saturated dedup cannot spin forever.
    let mut req = start_rank.min(max_rank).min(m.max(1) * n.max(1));
    loop {
        req = req.min(max_rank);
        // --- Q: orthonormal basis of ell stratified columns ---
        let cols = stratified_sample(n, req, &mut rng);
        // Deduplication may shrink the request; drive every allocation off
        // the actual count.
        let ell = cols.len().max(1);
        let mut q_re = Vec::with_capacity(ell * m);
        let mut q_im = Vec::with_capacity(ell * m);
        for &j in &cols {
            let y = src[j];
            for &x in tgt.iter() {
                let (r, i) = entry(x, y, eta, eta_sq);
                q_re.push(r);
                q_im.push(i);
            }
        }
        // Modified Gram–Schmidt, twice for numerical orthogonality.
        for t in 0..ell {
            for _pass in 0..2 {
                for s in 0..t {
                    // q_t -= q_s * <q_s, q_t>
                    let (mut dr, mut di) = (0.0, 0.0);
                    for i in 0..m {
                        dr += q_re[s * m + i] * q_re[t * m + i] + q_im[s * m + i] * q_im[t * m + i];
                        di += q_re[s * m + i] * q_im[t * m + i] - q_im[s * m + i] * q_re[t * m + i];
                    }
                    for i in 0..m {
                        q_re[t * m + i] -= q_re[s * m + i] * dr - q_im[s * m + i] * di;
                        q_im[t * m + i] -= q_re[s * m + i] * di + q_im[s * m + i] * dr;
                    }
                }
            }
            // Normalize.
            let mut nr = 0.0f64;
            for i in 0..m {
                nr += q_re[t * m + i] * q_re[t * m + i] + q_im[t * m + i] * q_im[t * m + i];
            }
            let nrm = nr.sqrt();
            if nrm <= 1e-300 {
                // Degenerate column (repeated structure): replace by a unit
                // coordinate direction to keep Q well defined.
                q_re[t * m] += 1.0;
            } else {
                let inv = 1.0 / nrm;
                for i in 0..m {
                    q_re[t * m + i] *= inv;
                    q_im[t * m + i] *= inv;
                }
            }
        }

        // --- Least-squares row-space fit: boundary rows (kernel mass sits
        // at the shared edge of adjacent ranges) + random rows ---
        let edge = 4.min(m / 2);
        let mut rows: Vec<usize> = (0..edge).chain(m - edge..m).collect();
        let extra = (2 * ell).saturating_sub(rows.len()).min(m);
        rows.extend(stratified_sample(m, extra, &mut rng));
        rows.sort_unstable();
        rows.dedup();
        // G = Qr^H Qr (ell x ell), B = Qr^H K(IR,:) (ell x n).
        let mut g_re = vec![0.0f64; ell * ell];
        let mut g_im = vec![0.0f64; ell * ell];
        let mut b_re = vec![0.0f64; ell * n];
        let mut b_im = vec![0.0f64; ell * n];
        let mut krow_re = vec![0.0f64; n];
        let mut krow_im = vec![0.0f64; n];
        for &i in rows.iter() {
            // Materialize the sampled kernel row once, then stream it
            // against every factor row (contiguous AXPYs).
            let x = tgt[i];
            for (tj, &y) in src.iter().enumerate() {
                let (r, im) = entry(x, y, eta, eta_sq);
                krow_re[tj] = r;
                krow_im[tj] = im;
            }
            for tt in 0..ell {
                let qr = q_re[tt * m + i];
                let qi = q_im[tt * m + i];
                let br = &mut b_re[tt * n..tt * n + n];
                let bi = &mut b_im[tt * n..tt * n + n];
                for tj in 0..n {
                    br[tj] += qr * krow_re[tj] + qi * krow_im[tj];
                    bi[tj] += qr * krow_im[tj] - qi * krow_re[tj];
                }
            }
            for ta in 0..ell {
                for tb in 0..ell {
                    let ar = q_re[ta * m + i];
                    let ai = q_im[ta * m + i];
                    let br = q_re[tb * m + i];
                    let bi = q_im[tb * m + i];
                    // conj(a)*b
                    g_re[ta * ell + tb] += ar * br + ai * bi;
                    g_im[ta * ell + tb] += ar * bi - ai * br;
                }
            }
        }
        // Solve (G + λI) W = B — tiny SPD system (Gaussian elimination with
        // partial pivoting), Tikhonov floor for safety.
        let lambda = 1e-12;
        for ta in 0..ell {
            g_re[ta * ell + ta] += lambda;
        }
        let mut aug_re = vec![0.0f64; ell * 2 * ell];
        let mut aug_im = vec![0.0f64; ell * 2 * ell];
        for ta in 0..ell {
            for tb in 0..ell {
                aug_re[ta * 2 * ell + tb] = g_re[ta * ell + tb];
                aug_im[ta * 2 * ell + tb] = g_im[ta * ell + tb];
            }
            for tb in 0..n.min(ell) {
                aug_re[ta * 2 * ell + ell + tb] = b_re[ta * n + tb];
                aug_im[ta * 2 * ell + ell + tb] = b_im[ta * n + tb];
            }
        }
        // The RHS must hold all n columns; solve directly on (G, B) instead
        // of an augmented copy sized for n.
        let w_re = solve_spd_complex(&g_re, &g_im, ell, &b_re, &b_im, n);
        let w_im = {
            // solve_spd_complex returns both parts flattened [ell x n].
            let off = ell * n;
            w_re[off..].to_vec()
        };
        let w_re = {
            let mut r = vec![0.0f64; ell * n];
            r.copy_from_slice(&w_re[..ell * n]);
            r
        };
        let _ = &aug_re;
        let _ = &aug_im;

        // --- Validation: evaluate whole test columns so boundary-localized
        // error cannot hide between sparse point probes ---
        let edge_j = 4.min(n / 2);
        let mut test_cols: Vec<usize> = (0..edge_j).chain(n - edge_j..n).collect();
        for _ in 0..N_PROBE_COLS {
            test_cols.push(rng.below(n));
        }

        // Residual accumulation over the test columns.
        let eval_col = |jset: &[usize], err: &mut f64, refs: &mut f64| {
            for &pj in jset {
                let y = src[pj];
                for (pi, &x) in tgt.iter().enumerate() {
                    let (kr, ki) = entry(x, y, eta, eta_sq);
                    let (mut ar, mut ai) = (0.0, 0.0);
                    for tt in 0..ell {
                        ar += q_re[tt * m + pi] * w_re[tt * n + pj]
                            - q_im[tt * m + pi] * w_im[tt * n + pj];
                        ai += q_re[tt * m + pi] * w_im[tt * n + pj]
                            + q_im[tt * m + pi] * w_re[tt * n + pj];
                    }
                    *err += (ar - kr).powi(2) + (ai - ki).powi(2);
                    *refs += kr.powi(2) + ki.powi(2);
                }
            }
        };
        let (mut err_sq, mut ref_sq) = (0.0f64, 0.0f64);
        eval_col(&test_cols, &mut err_sq, &mut ref_sq);
        let rel = (err_sq / ref_sq.max(1e-300)).sqrt();
        #[allow(clippy::let_and_return)]
        let dbg_rel = rel;
        if std::env::var("HODLR_DEBUG").is_ok() {
            eprintln!("[rand] m={m} n={n} req={req} ell={ell} rel={dbg_rel:.3e} tol={tol:.1e}");
        }
        let saturated = req >= max_rank || req >= m.min(n);
        if rel <= tol || saturated {
            return AcaFactors {
                u_re: q_re,
                u_im: q_im,
                v_re: w_re,
                v_im: w_im,
            };
        }
        req = req.saturating_mul(2).min(max_rank);
    }
}

/// Solve the small Hermitian positive-definite system G W = B (G: ell×ell,
/// B: ell×n) by Cholesky in complex arithmetic. Returns [Re(W); Im(W)]
/// concatenated (each ell×n row-major).
fn solve_spd_complex(
    g_re: &[f64],
    g_im: &[f64],
    ell: usize,
    b_re: &[f64],
    b_im: &[f64],
    n: usize,
) -> Vec<f64> {
    // Cholesky: G = L L^H.
    let mut l_re = vec![0.0f64; ell * ell];
    let mut l_im = vec![0.0f64; ell * ell];
    // Row-major layout: L[row, col] lives at [row * ell + col]. The
    // diagonal update consumes row j (L[j,k], k<j); the column update
    // subtracts L[i,k] * conj(L[j,k]).
    for j in 0..ell {
        let mut d = g_re[j * ell + j];
        for k in 0..j {
            let lr = l_re[j * ell + k];
            let li = l_im[j * ell + k];
            d -= lr * lr + li * li;
        }
        let dn = d.max(1e-300).sqrt();
        l_re[j * ell + j] = dn;
        for i in (j + 1)..ell {
            let mut sr = g_re[i * ell + j];
            let mut si = g_im[i * ell + j];
            for k in 0..j {
                let ar = l_re[i * ell + k];
                let ai = l_im[i * ell + k];
                let br = l_re[j * ell + k];
                let bi = l_im[j * ell + k];
                sr -= ar * br + ai * bi;
                si -= ai * br - ar * bi;
            }
            let inv = 1.0 / dn;
            l_re[i * ell + j] = sr * inv;
            l_im[i * ell + j] = si * inv;
        }
    }
    // Forward/back substitution per RHS column.
    let mut out = vec![0.0f64; 2 * ell * n];
    {
        let (z_re, z_im) = out.split_at_mut(ell * n);
        z_re.copy_from_slice(b_re);
        z_im.copy_from_slice(b_im);
        for j in 0..ell {
            let dj = l_re[j * ell + j];
            for c in 0..n {
                let xr = z_re[j * n + c] / dj;
                let xi = z_im[j * n + c] / dj;
                z_re[j * n + c] = xr;
                z_im[j * n + c] = xi;
            }
            for i in (j + 1)..ell {
                let lr = l_re[i * ell + j];
                let li = l_im[i * ell + j];
                for c in 0..n {
                    let xr = z_re[j * n + c];
                    let xi = z_im[j * n + c];
                    // z_i -= l_ij * z_j
                    z_re[i * n + c] -= lr * xr - li * xi;
                    z_im[i * n + c] -= lr * xi + li * xr;
                }
            }
        }
        // Solve L^H W = z: process rows bottom-up, subtracting each solved
        // W_j's contribution from earlier rows via COLUMN j of L.
        for j in (0..ell).rev() {
            let dj = l_re[j * ell + j];
            for c in 0..n {
                let mut xr = z_re[j * n + c];
                let mut xi = z_im[j * n + c];
                // Subtract conj(L[k,j]) * W_k (L^H appears in L^H W = z).
                for k in (j + 1)..ell {
                    let lr = l_re[k * ell + j];
                    let li = l_im[k * ell + j];
                    xr -= lr * z_re[k * n + c] + li * z_im[k * n + c];
                    xi -= lr * z_im[k * n + c] - li * z_re[k * n + c];
                }
                z_re[j * n + c] = xr / dj;
                z_im[j * n + c] = xi / dj;
            }
        }
    }
    out
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

/// Off-diagonal block compression strategy.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HodlrMode {
    /// Deterministic adaptive cross approximation: alternating skeleton
    /// rows/columns pivoted at the largest residual entry. Tightest
    /// tolerance control; deflation costs `O(rank²·(m+n))` per block.
    Aca,
    /// RandNLA sketching (Halko–Martinsson–Tropp style): orthonormalize a
    /// stratified sample of kernel columns (boundary strips + geometric
    /// ladder + uniform fill), fit the row space by least squares on
    /// stratified rows, validate against whole boundary-strip test columns
    /// and double ℓ until the estimated relative residual meets the
    /// tolerance. Linear `O(ℓ·(m+n))` per block with tiny constants.
    ///
    /// Operating envelope (measured): excellent for smooth, uniformly
    /// decaying interactions. On sharp near-field Cauchy spectra (this
    /// crate's MP family, η = 1/√p) adaptive pivoting dominates: greedy
    /// skeleton selection reaches ~2e-5 block error at rank 8 where any
    /// fixed sampling scheme needs ~30 ranks for the same block, and the
    /// gap widens as p grows. Kept for kernel families without boundary
    /// concentration; [`HodlrMode::Aca`] is the default dispatch.
    Random,
}

/// Compression settings shared by every block of one run.
#[derive(Copy, Clone)]
struct HodlrSettings {
    leaf_cap: usize,
    eta: f64,
    eta_sq: f64,
    tol: f64,
    max_rank: usize,
    mode: HodlrMode,
}

/// Tiny deterministic PRNG (xorshift64*) — seeded from the block's index
/// range so sequential and parallel runs draw identical samples.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // SplitMix-style warmup so nearby seeds decorrelate.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Rng(z ^ (z >> 31))
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let x = self.0;
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let z = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB)
    }
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() >> 11) as usize % n.max(1)
    }
}

/// Draw `k` distinct indices from `0..n` via partial Fisher–Yates.
fn sample_without_replacement(n: usize, k: usize, rng: &mut Rng) -> Vec<usize> {
    let k = k.min(n);
    let mut idx: Vec<usize> = (0..n).collect();
    for t in 0..k {
        let j = t + rng.below(n - t);
        idx.swap(t, j);
    }
    idx.truncate(k);
    idx.sort_unstable();
    idx
}

/// Stratified index set for Cauchy-type blocks: the dominant singular
/// directions concentrate near the shared boundary of adjacent ranges, so
/// probe both boundary strips, a geometric ladder of offsets covering the
/// decay profile, then fill the rest uniformly at random. Measured on MP
/// spectra this reaches ~1e-8 block error at rank 16 where pure-uniform
/// sampling stays at ~1e-3.
fn stratified_sample(n: usize, k: usize, rng: &mut Rng) -> Vec<usize> {
    if k >= n {
        return (0..n).collect();
    }
    let mut out: Vec<usize> = Vec::with_capacity(k);
    let edge = 2.min(n / 2);
    out.extend(0..edge);
    out.extend(n - edge..n);
    // Geometric ladder of offsets from each boundary.
    let mut off = 2usize;
    while off < n / 2 && out.len() + 2 <= k {
        out.push(off.saturating_sub(1));
        out.push(n - off.min(n));
        off = off.saturating_mul(2);
    }
    // Uniform fill.
    if out.len() < k {
        let extra = sample_without_replacement(n, k - out.len(), rng);
        out.extend(extra);
    }
    out.truncate(k);
    out.sort_unstable();
    out.dedup();
    out
}

/// Recursive assembly over [lo, hi). Writes raw sums into `out[lo..hi)`
/// (the slice is re-based on the range at every level).
fn rec(
    sorted: &[f64],
    lo: usize,
    hi: usize,
    s: &HodlrSettings,
    depth_budget: u32,
    seed: u64,
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

    // Deterministic per-subtree seeds: sequential and parallel traversals
    // visit identical seeds regardless of join order.
    let child_seed = |left: bool| seed.wrapping_mul(2).wrapping_add(left as u64);
    {
        let (out_l, out_r) = out.split_at_mut(mid - lo);
        if depth_budget > 0 {
            rayon::join(
                || {
                    rec(
                        sorted,
                        lo,
                        mid,
                        s,
                        depth_budget - 1,
                        child_seed(true),
                        out_l,
                    )
                },
                || {
                    rec(
                        sorted,
                        mid,
                        hi,
                        s,
                        depth_budget - 1,
                        child_seed(false),
                        out_r,
                    )
                },
            );
        } else {
            rec(sorted, lo, mid, s, 0, child_seed(true), out_l);
            rec(sorted, mid, hi, s, 0, child_seed(false), out_r);
        }
    }

    // Cross terms L←R and R←L (`out` is already re-based on [lo, hi)).
    //
    // The kernel satisfies M(R←L)[j,i] = -conj(M(L←R)[i,j]) for the same
    // two halves with roles swapped (d -> -d flips Re, keeps Im), so ONE
    // compression serves both blocks: U2 = -conj(V1), V2 = conj(U1) under
    // the bilinear U·Vᵀ convention of `apply_cross`. This halves the
    // off-diagonal compression work at every level.
    let tgt_l = &sorted[lo..mid];
    let src_r = &sorted[mid..hi];
    let f = match s.mode {
        HodlrMode::Aca => {
            let (u_re, u_im, v_re, v_im, _rk) =
                aca(tgt_l, src_r, s.eta, s.eta_sq, s.tol, s.max_rank);
            AcaFactors {
                u_re,
                u_im,
                v_re,
                v_im,
            }
        }
        HodlrMode::Random => rand_block(tgt_l, src_r, s.eta, s.eta_sq, s.tol, s.max_rank, seed),
    };
    apply_cross(&f, tgt_l.len(), src_r.len(), out, 0);

    // Transferred factors for R←L: U2[t,j] = -conj(V1[t,i]),
    // V2[t,i] = conj(U1[t,j]).
    let n_src_r = src_r.len();
    let mut u2_re = Vec::with_capacity(f.v_re.len());
    let mut u2_im = Vec::with_capacity(f.v_im.len());
    let rank_v = f.v_re.len().checked_div(n_src_r).unwrap_or(0);
    for t in 0..rank_v {
        for k in 0..n_src_r {
            u2_re.push(-f.v_re[t * n_src_r + k]);
            u2_im.push(f.v_im[t * n_src_r + k]);
        }
    }
    let m_tgt_l = tgt_l.len();
    let mut v2_re = Vec::with_capacity(f.u_re.len());
    let mut v2_im = Vec::with_capacity(f.u_im.len());
    let rank_u = f.u_re.len().checked_div(m_tgt_l).unwrap_or(0);
    for t in 0..rank_u {
        for k in 0..m_tgt_l {
            v2_re.push(f.u_re[t * m_tgt_l + k]);
            v2_im.push(-f.u_im[t * m_tgt_l + k]);
        }
    }
    apply_cross(
        &AcaFactors {
            u_re: u2_re,
            u_im: u2_im,
            v_re: v2_re,
            v_im: v2_im,
        },
        src_r.len(),
        tgt_l.len(),
        out,
        mid - lo,
    );
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
    mode: HodlrMode,
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
        mode,
    };
    let mut out: Vec<(f64, f64)> = vec![(0.0, 0.0); p];
    let depth_budget = if parallel { 3 } else { 0 };
    rec(sorted, 0, p, &settings, depth_budget, 1, &mut out);

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

/// Dispatch defaults of the `Hodlr` variant: near-field leaf size, ACA
/// compression tolerance, ACA rank cap. Named here so the dispatcher
/// carries no magic numbers.
pub(crate) const DEFAULT_LEAF: usize = 256;
/// Dispatch default ACA tolerance.
pub(crate) const DEFAULT_ACA_TOL: f64 = 1e-9;
/// Dispatch default ACA rank cap.
pub(crate) const DEFAULT_ACA_RANK: usize = 32;

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
            let got = super::compute_all_stieltjes_hodlr_impl(
                &evs,
                eta,
                32,
                1e-10,
                64,
                false,
                super::HodlrMode::Aca,
            );
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
        // Parallel traversal must reproduce the sequential sums (identical
        // seeds and identical per-block arithmetic order in both modes).
        let p = 1000;
        let evs = spectrum(p);
        let eta = 1.0 / (p as f64).sqrt();
        for mode in [super::HodlrMode::Aca, super::HodlrMode::Random] {
            let seq = super::compute_all_stieltjes_hodlr_impl(&evs, eta, 32, 1e-9, 32, false, mode);
            let par = super::compute_all_stieltjes_hodlr_impl(&evs, eta, 32, 1e-9, 32, true, mode);
            let max_diff = seq
                .iter()
                .zip(&par)
                .map(|(a, b)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt())
                .fold(0.0f64, f64::max);
            assert!(
                max_diff < 1e-7,
                "par/seq mismatch for {mode:?}: {max_diff:.3e}"
            );
        }
    }

    #[test]
    fn solve_spd_complex_correctness() {
        // G W = B with a random well-conditioned Hermitian G.
        let ell = 6usize;
        let n = 4usize;
        let mut rng = super::Rng::new(99);
        // Build M first, then G = M^H M + 2I (properly Hermitian, HPD).
        let mut m_re = vec![0.0f64; ell * ell];
        let mut m_im = vec![0.0f64; ell * ell];
        for v in m_re.iter_mut() {
            *v = rng.below(100) as f64 / 50.0 - 1.0;
        }
        for v in m_im.iter_mut() {
            *v = rng.below(100) as f64 / 50.0 - 1.0;
        }
        let mut g_re = vec![0.0f64; ell * ell];
        let mut g_im = vec![0.0f64; ell * ell];
        for i in 0..ell {
            for j in 0..ell {
                for k in 0..ell {
                    let ar = m_re[k * ell + i];
                    let ai = m_im[k * ell + i];
                    let br = m_re[k * ell + j];
                    let bi = m_im[k * ell + j];
                    g_re[i * ell + j] += ar * br + ai * bi;
                    g_im[i * ell + j] += ai * br - ar * bi;
                }
            }
            g_re[i * ell + i] += 2.0;
        }
        // Reference solve in f64 complex via manual Gaussian elimination on
        // the augmented system, then compare against our Cholesky path.
        let b_re: Vec<f64> = (0..ell * n).map(|_| rng.below(10) as f64).collect();
        let b_im: Vec<f64> = (0..ell * n).map(|_| rng.below(10) as f64).collect();
        let sol = super::solve_spd_complex(&g_re, &g_im, ell, &b_re, &b_im, n);
        // Verify G W = B by direct multiplication.
        let mut worst = 0.0f64;
        for j in 0..n {
            for i in 0..ell {
                let (mut sr, mut si) = (0.0f64, 0.0f64);
                for k in 0..ell {
                    let gr = g_re[i * ell + k];
                    let gi = g_im[i * ell + k];
                    sr += gr * sol[k * n + j] - gi * sol[ell * n + k * n + j];
                    si += gr * sol[ell * n + k * n + j] + gi * sol[k * n + j];
                }
                let _ = si;
                worst = worst.max((sr - b_re[i * n + j]).abs());
            }
        }
        eprintln!("solve_spd residual={worst:.3e}");
        assert!(worst < 1e-8, "solver residual {worst:.3e}");
    }

    #[test]
    fn rand_block_reconstruction() {
        // Direct block-level check: factors must reproduce the kernel on
        // held-out entries to the advertised tolerance band.
        // Sibling-like pair from a balanced tree over 512 sorted points:
        // left grandchild [0..64) vs right grandchild [192..256).
        let evs = spectrum(512);
        let tgt: Vec<f64> = evs[..64].to_vec();
        let src: Vec<f64> = evs[192..256].to_vec();
        let eta = 0.05;
        let f = super::rand_block(&tgt, &src, eta, eta * eta, 1e-6, 32, 7);
        let m = tgt.len();
        let nn = src.len();
        let mut worst = 0.0f64;
        let mut ref_scale = 0.0f64;
        for i in (0..m).step_by(7) {
            for j in (0..nn).step_by(5) {
                let d = tgt[i] - src[j];
                let inv = 1.0 / (d * d + eta * eta);
                let kr = d * inv;
                let ki = eta * inv;
                let (mut ar, mut ai) = (0.0, 0.0);
                for t in 0..f.u_re.len() / m {
                    ar += f.u_re[t * m + i] * f.v_re[t * nn + j]
                        - f.u_im[t * m + i] * f.v_im[t * nn + j];
                    ai += f.u_re[t * m + i] * f.v_im[t * nn + j]
                        + f.u_im[t * m + i] * f.v_re[t * nn + j];
                }
                worst = worst.max(((ar - kr).powi(2) + (ai - ki).powi(2)).sqrt());
                ref_scale = ref_scale.max((kr * kr + ki * ki).sqrt());
            }
        }
        eprintln!("rand_block worst={worst:.3e} scale={ref_scale:.3e}");
        assert!(worst / ref_scale < 1e-3, "block rel err {worst:.3e}");
    }

    #[test]
    fn hodlr_rand_mode_accuracy() {
        // Sketching trades a controlled amount of accuracy for speed: at
        // tol=1e-6 the randomized path must stay within ~1e-5 relative L2.
        for &p in &[256usize, 1000] {
            let evs = spectrum(p);
            let eta = 1.0 / (p as f64).sqrt();
            let got = super::compute_all_stieltjes_hodlr_impl(
                &evs,
                eta,
                256,
                1e-6,
                32,
                false,
                super::HodlrMode::Random,
            );
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for (i, &x) in evs.iter().enumerate() {
                let (mut sr, mut si) = (0.0, 0.0);
                for &y in &evs {
                    let d = x - y;
                    let inv = 1.0 / (d * d + eta * eta);
                    sr += d * inv;
                    si += eta * inv;
                }
                num += (got[i].0 - sr).powi(2) + (got[i].1 - si).powi(2);
                den += sr.powi(2) + si.powi(2);
            }
            let rel = (num / den).sqrt();
            eprintln!("hodlr-rand p={p} rel_l2={rel:.3e}");
            assert!(rel < 1e-4, "rand mode p={p}: {rel:.3e}");
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
        let got = super::compute_all_stieltjes_hodlr_impl(
            &evs,
            eta,
            16,
            1e-10,
            64,
            false,
            super::HodlrMode::Aca,
        );
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
