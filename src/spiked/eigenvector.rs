//! Spike eigenvector estimation: BBP angle formula (Benaych-Georges &
//! Nadakuditi) and debiased projection.
//!
//! Above the BBP threshold, the sample eigenvector $\hat u_i$ is not aligned
//! with the population eigenvector $v_i$, but the asymptotic squared overlap
//! is known deterministically:
//!
//! $$\bigl|\langle \hat u_i, v_i\rangle\bigr|^2 \to
//!   \frac{1 - \frac{\gamma}{(\ell_i - 1)^2}}{1 + \frac{\gamma}{\ell_i - 1}}$$
//!
//! where $\ell_i$ is the population spike (in units of $\sigma^2 = 1$).
//! This module computes that overlap from an estimated population spike and
//! uses it to rescale factor loadings / project back into a debiased subspace
//! (the S-POET / debiased-PCA idea).

/// Compute the asymptotic squared angular overlap
/// $\alpha_i^2 = |\langle \hat u_i, v_i\rangle|^2$ for a population spike
/// $\ell_i$ (in units of $\sigma^2 = 1$) and concentration ratio $\gamma$.
///
/// Returns a value in $[0, 1]$. If the spike is at or below the BBP threshold
/// ($\ell_i \le 1 + \sqrt\gamma$), the overlap is 0 (the eigenvector carries
/// no signal).
#[inline(always)]
pub fn bbp_angle_overlap(ell: f64, gamma: f64) -> f64 {
    // In units of σ²=1, the BBP threshold is 1 + √γ.
    if ell <= 1.0 + gamma.sqrt() {
        return 0.0;
    }
    let num = 1.0 - gamma / (ell - 1.0).powi(2);
    let den = 1.0 + gamma / (ell - 1.0);
    if num <= 0.0 {
        0.0
    } else {
        (num / den).min(1.0)
    }
}

/// Compute the squared angular overlaps for a set of estimated population
/// spikes (in units of $\sigma^2 = 1$).
///
/// # Arguments
///
/// * `spikes` — Estimated population spikes $\ell_i$ (in units of $\sigma^2$).
/// * `gamma` — Concentration ratio $p/n$.
///
/// # Returns
///
/// A `Vec<f64>` of squared overlaps, same length as `spikes`.
pub fn bbp_angle_overlaps(spikes: &[f64], gamma: f64) -> Vec<f64> {
    spikes
        .iter()
        .map(|&ell| bbp_angle_overlap(ell, gamma))
        .collect()
}

/// Debiased projection: rescale a sample eigenvector's loadings by the
/// inverse of its asymptotic overlap, producing a debiased direction.
///
/// Given a sample eigenvector $\hat u$ (length p) and its squared overlap
/// $\alpha^2$, the debiased vector is $\hat u / \alpha$. This is the
/// S-POET / debiased-PCA rescaling that corrects the understated projection
/// alignment of sample eigenvectors.
///
/// # Returns
///
/// A new `Vec<f64>` of length p. If $\alpha^2 \le 0$, returns a zero vector
/// (no signal to debias).
pub fn debias_eigenvector(u_hat: &[f64], alpha2: f64) -> Vec<f64> {
    if alpha2 <= 0.0 {
        return vec![0.0; u_hat.len()];
    }
    let scale = 1.0 / alpha2.sqrt();
    u_hat.iter().map(|&x| x * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bbp_angle_overlap_large_spike() {
        // As ℓ → ∞, overlap → 1.
        let gamma = 0.5;
        let a = bbp_angle_overlap(100.0, gamma);
        assert!(a > 0.99 && a <= 1.0);
    }

    #[test]
    fn test_bbp_angle_overlap_below_threshold() {
        let gamma = 0.5;
        // Threshold is 1 + √0.5 ≈ 1.707
        assert_eq!(bbp_angle_overlap(1.5, gamma), 0.0);
        assert_eq!(bbp_angle_overlap(1.0 + gamma.sqrt(), gamma), 0.0);
    }

    #[test]
    fn test_bbp_angle_overlap_monotone() {
        let gamma = 0.5;
        let mut prev = 0.0;
        for ell in (2..=20).map(|e| e as f64) {
            let a = bbp_angle_overlap(ell, gamma);
            assert!(a >= prev);
            prev = a;
        }
    }

    #[test]
    fn test_debias_eigenvector() {
        let u = vec![0.6, 0.8];
        let alpha2 = 0.25; // α = 0.5
        let debiased = debias_eigenvector(&u, alpha2);
        // scale = 1/0.5 = 2
        assert_relative_eq!(debiased[0], 1.2, epsilon = 1e-12);
        assert_relative_eq!(debiased[1], 1.6, epsilon = 1e-12);
    }

    #[test]
    fn test_debias_eigenvector_no_signal() {
        let u = vec![0.6, 0.8];
        let debiased = debias_eigenvector(&u, 0.0);
        assert_eq!(debiased, vec![0.0, 0.0]);
    }
}
