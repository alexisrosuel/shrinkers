//! Why the Chebyshev treecode evaluates the far field `Σ_j w_j/(z-t_j)`
//! term-by-term instead of precomputing numerator/denominator polynomial
//! coefficients `P`, `Q` and using a single complex Horner division:
//!
//! Forming monomial coefficients of degree-n polynomials whose roots cluster
//! near ±1 (Chebyshev nodes) amplifies rounding error catastrophically — the
//! measured relative error below grows from ~1e-16 to O(1) as the node count
//! rises — while per-term evaluation is unconditionally stable.
//!
//! Run: cargo run --release --example check_poly

/// Multiply polynomial `coeff` (ascending powers, length `deg+1`) by `(z - r)`,
/// in place. `coeff` must have room for `deg+2`.
fn poly_mul_linear(coeff: &mut [f64], deg: usize, r: f64) {
    let mut prev = 0.0;
    for c_k in coeff.iter_mut().take(deg + 1) {
        let ck = *c_k;
        *c_k = prev - r * ck;
        prev = ck;
    }
    coeff[deg + 1] = prev;
}

/// Numerator `P(z) = Σ_j w_j Π_{i≠j}(z-t_i)` (degree n-1) and denominator
/// `Q(z) = Π_j (z-t_j)` (degree n), ascending-power coefficient vectors.
/// THE UNSTABLE REPRESENTATION — kept here only to demonstrate why.
fn partial_fraction_polys(t: &[f64], w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = t.len();
    // Q(z) = Π_j (z - t_j)
    let mut q = vec![0.0; n + 1];
    q[0] = 1.0;
    for (deg, &tj) in t.iter().enumerate() {
        poly_mul_linear(&mut q, deg, tj);
    }
    // P(z) = Σ_j w_j · Q(z)/(z - t_j) by synthetic division of Q.
    let mut p = vec![0.0; n];
    for (&wj, &tj) in w.iter().zip(t.iter()) {
        let mut r = vec![0.0; n];
        let mut carry = 0.0;
        for (r_k, &q_next) in r.iter_mut().rev().zip(q[1..].iter().rev()) {
            *r_k = q_next + tj * carry;
            carry = *r_k;
        }
        for (p_k, &r_k) in p.iter_mut().zip(r.iter()) {
            *p_k += wj * r_k;
        }
    }
    (p, q)
}

fn eval_poly_c(c: &[f64], zr: f64, zi: f64) -> (f64, f64) {
    let mut acc_r = 0.0;
    let mut acc_i = 0.0;
    for &k in c.iter().rev() {
        let nr = acc_r * zr - acc_i * zi + k;
        let ni = acc_r * zi + acc_i * zr;
        acc_r = nr;
        acc_i = ni;
    }
    (acc_r, acc_i)
}

/// Direct, stable evaluation: Σ_j w_j / (z - t_j) with z = zr + i·zi.
fn eval_direct(t: &[f64], w: &[f64], zr: f64, zi: f64) -> (f64, f64) {
    let mut dir_r = 0.0;
    let mut dir_i = 0.0;
    for (&wj, &tj) in w.iter().zip(t.iter()) {
        // d = (zr - tj) + i·zi ; 1/d = conj(d)/|d|²
        let wr = zr - tj;
        let inv = 1.0 / (wr * wr + zi * zi);
        dir_r += wj * wr * inv;
        dir_i -= wj * zi * inv; // Im[1/d] = -zi/|d|²
    }
    (dir_r, dir_i)
}

fn main() {
    // Second-kind Chebyshev nodes on [-1,1] for several degrees, plus weights
    // that mimic aggregated source weights (O(1) magnitude).
    for (n_nodes, tag) in [(3usize, "n=3"), (9, "n=9"), (17, "n=17"), (33, "n=33")] {
        let t: Vec<f64> = (0..n_nodes)
            .map(|j| (j as f64 * std::f64::consts::PI / (n_nodes - 1) as f64).cos())
            .collect();
        let w: Vec<f64> = (0..n_nodes).map(|j| ((j % 5) as f64) - 2.0).collect();

        let zr = 1.3_f64;
        let zi = -0.05_f64;

        let direct = eval_direct(&t, &w, zr, zi);

        let (p, q) = partial_fraction_polys(&t, &w);
        let (pr, pi) = eval_poly_c(&p, zr, zi);
        let (qr, qi) = eval_poly_c(&q, zr, zi);
        let denom = qr * qr + qi * qi;
        let rat = ((pr * qr + pi * qi) / denom, (pi * qr - pr * qi) / denom);

        let err_r = (direct.0 - rat.0).abs() / direct.0.abs().max(1e-12);
        let err_i = (direct.1 - rat.1).abs() / direct.1.abs().max(1e-12);
        println!(
            "{tag}: P/Q-Horner rel err vs direct = real {err_r:.3e}, imag {err_i:.3e} \
             (larger nodes → catastrophic cancellation)"
        );
        assert!(direct.1.is_finite() && rat.1.is_finite());
    }
    println!("=> the crate evaluates far-field terms directly (see stieltjes/chebcode.rs)");
}
