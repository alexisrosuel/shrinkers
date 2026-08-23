//! Squared angular overlaps α_i² = cos²(θ_i) for eigenvector shrinkage.
//!
//! Under the Spiked Covariance Model (BBP transition), the squared cosine
//! of the angle between the i-th sample eigenvector v̂_i and the population
//! eigenvector v_i is:
//!
//! α_i² = max(0,  (1 - c·σ⁴ / (λ̃_i - σ²)²) / (1 + c·σ² / (λ̃_i - σ²)))

/// Compute squared angular overlaps α_i² = cos²(θ_i) for all eigenvalues.
///
/// - When λ̃_i ≤ σ² (bulk / noise eigenvalue), α_i² = 0.
/// - When c → 0 (N << T), α_i² → 1 (eigenvectors are consistent).
///
/// # Arguments
///
/// * `lambda_rie` — RIE-cleaned eigenvalues, sorted descending.
/// * `c` — Concentration ratio N / T.
/// * `sigma2` — Estimated noise variance σ².
///
/// # Returns
///
/// A vector of squared angular overlaps, same length as `lambda_rie`.
pub fn compute_angular_overlaps(lambda_rie: &[f64], c: f64, sigma2: f64) -> Vec<f64> {
    let p = lambda_rie.len();
    let mut alpha2 = vec![0.0; p];

    for i in 0..p {
        let l_rie = lambda_rie[i];
        let diff = l_rie - sigma2;

        // Only spikes (eigenvalues above the noise floor) can have non-zero overlap
        if diff > 1e-12 {
            let num = 1.0 - (c * (sigma2 * sigma2)) / (diff * diff);
            let den = 1.0 + (c * sigma2) / diff;
            if num > 0.0 {
                let raw = num / den;
                alpha2[i] = raw.min(1.0 - 1e-15); // clamp just below 1
            }
        }
    }

    alpha2
}
