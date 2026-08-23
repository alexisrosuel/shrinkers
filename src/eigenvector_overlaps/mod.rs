//! Theoretical eigenvector alignment (Spiked Covariance Model / RMT).
//!
//! Given RIE-cleaned eigenvalues, computes the squared angular overlaps
//! α_i² = cos²(θ_i) between each sample eigenvector and its (unknown)
//! population counterpart.
//!
//! # Theory
//!
//! Under the Spiked Covariance Model (BBP transition), the squared cosine of
//! the angle between the i-th sample eigenvector v̂_i and the population
//! eigenvector v_i is:
//!
//! α_i² = max(0,  (1 - c·σ⁴ / (λ̃_i - σ²)²) / (1 + c·σ² / (λ̃_i - σ²)))
//!
//! where λ̃_i is the RIE-cleaned eigenvalue, σ² the noise variance (median
//! of sample eigenvalues), and c = N/T the concentration ratio.

pub use overlaps::*;

mod overlaps;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_angular_overlaps_c_small() {
        let lambda_rie = vec![5.0, 4.0, 3.0, 2.0];
        let alpha2 = compute_angular_overlaps(&lambda_rie, 1e-6, 0.5);
        for &a in &alpha2 {
            assert_relative_eq!(a, 1.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_angular_overlaps_noise_floor() {
        let lambda_rie = vec![5.0, 4.0, 3.0, 2.0, 0.5, 0.3];
        let sigma2 = 1.0;
        let alpha2 = compute_angular_overlaps(&lambda_rie, 0.5, sigma2);
        assert!(alpha2[0] > 0.0);
        assert!(alpha2[1] > 0.0);
        assert_eq!(alpha2[4], 0.0);
        assert_eq!(alpha2[5], 0.0);
    }
}
