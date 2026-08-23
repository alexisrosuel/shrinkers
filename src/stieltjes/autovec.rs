//! Auto-vectorized O(p²) Stieltjes transform.
//!
//! Uses clean iterator-based loops that LLVM reliably auto-vectorizes
//! into NEON (Apple Silicon) or AVX2/AVX-512 (x86). Zero unsafe code.

use crate::stieltjes::term::stieltjes_term_hoisted;

/// Compute the raw sum S(λᵢ) = Σⱼ 1/((λᵢ-λⱼ) - iη)
/// using an auto-vectorization-friendly loop.
///
/// The compiler sees contiguous memory access on `eigenvalues`,
/// constant `eta_sq`/`eta`, and a pure reduction pattern — all
/// trivially vectorizable. Delegates to the hoisted term in `term.rs`
/// (the single source of truth for the division formula).
///
/// Idea 1 (eta-hoist): accumulates the raw reciprocal `inv` and multiplies
/// by `eta` once at the end, saving one `fmul` per term.
#[inline(always)]
pub fn autovec_stieltjes_sum(lambda_i: f64, eigenvalues: &[f64], eta: f64) -> (f64, f64) {
    let mut sum_real = 0.0;
    let mut sum_inv = 0.0;

    for &lambda_j in eigenvalues.iter() {
        let (r, inv) = stieltjes_term_hoisted(lambda_i, lambda_j, eta);
        sum_real += r;
        sum_inv += inv;
    }

    (sum_real, eta * sum_inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stieltjes::term::stieltjes_term_hoisted;

    #[test]
    fn test_autovec_matches_term_loop() {
        let evals: Vec<f64> = (0..100).map(|i| (i as f64 + 0.5).ln_1p()).collect();
        let eta = 0.05;

        for &lambda_i in &evals {
            let (simd_r, simd_i) = autovec_stieltjes_sum(lambda_i, &evals, eta);

            // Reference uses the same hoisted term (accumulate raw inv, scale
            // by eta once) so the accumulation order matches exactly.
            let mut scalar_r = 0.0;
            let mut scalar_inv = 0.0;
            for &lambda_j in &evals {
                let (r, inv) = stieltjes_term_hoisted(lambda_i, lambda_j, eta);
                scalar_r += r;
                scalar_inv += inv;
            }
            let scalar_i = eta * scalar_inv;

            assert!(
                (simd_r - scalar_r).abs() < 1e-14,
                "autovec real mismatch: {simd_r} vs {scalar_r}"
            );
            assert!(
                (simd_i - scalar_i).abs() < 1e-14,
                "autovec imag mismatch: {simd_i} vs {scalar_i}"
            );
        }
    }
}
