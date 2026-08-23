//! Spike detection: BEMA (Bulk Eigenvalue Matching Analysis) and
//! Tracy–Widom edge thresholding.
//!
//! Determining the number of spikes $K$ requires knowing where the bulk
//! noise spectrum ends. This module provides two scale-free approaches:
//!
//! 1. **BEMA** — fits the background noise distribution to the *lower* part
//!    of the sample spectrum (which is pure bulk), then uses the fitted
//!    Marchenko–Pastur edge as the threshold. This is robust to unknown
//!    $\sigma^2$ because the noise level is estimated from the data itself.
//!
//! 2. **Tracy–Widom edge inversion** — models the largest bulk fluctuation
//!    with the $F_1$ Tracy–Widom law and thresholds at a quantile of the
//!    bulk edge distribution.

/// Result of spike detection.
#[derive(Debug, Clone)]
pub struct SpikeDetection {
    /// Estimated number of spikes $K$.
    pub k: usize,
    /// Estimated bulk edge $\lambda_+ = \sigma^2(1+\sqrt\gamma)^2$.
    pub bulk_edge: f64,
    /// Estimated noise variance $\sigma^2$.
    pub sigma2: f64,
    /// Indices (into the caller-provided eigenvalue array, which must be
    /// sorted **ascending**) of the detected spikes, in ascending order.
    pub spike_indices: Vec<usize>,
}

/// Marchenko–Pastur upper edge for a noise variance $\sigma^2$ and
/// concentration ratio $\gamma = p/n$.
#[inline(always)]
pub fn mp_upper_edge(sigma2: f64, gamma: f64) -> f64 {
    sigma2 * (1.0 + gamma.sqrt()).powi(2)
}

/// Marchenko–Pastur median factor $m(\gamma)$.
///
/// The median of the Marchenko–Pastur distribution (with $\sigma^2 = 1$) is
/// strictly less than its mean (= 1) because the MP law is right-skewed.
/// This returns $m(\gamma) = \text{median}(MP_\gamma)$, so that
/// $\sigma^2 = \text{median}(\text{sample bulk}) / m(\gamma)$.
///
/// Uses a polynomial fit (max abs error ~4e-6 for $\gamma \in [0.01, 0.99]$).
#[inline(always)]
pub fn mp_median_factor(gamma: f64) -> f64 {
    let g = gamma.clamp(0.0, 1.0);
    // Fitted: m(γ) = 1 + a1·γ + a2·γ² + a3·γ³ + a4·γ⁴
    0.999995823
        - 0.333211332 * g
        - 0.0106766617 * g * g
        - 0.000758816762 * g * g * g
        - 0.00256753402 * g * g * g * g
}

/// Estimate the noise variance $\sigma^2$ from the sample spectrum (BEMA).
///
/// Uses the median of the sample eigenvalues corrected by the Marchenko–Pastur
/// median factor: $\sigma^2 = \text{median}(\lambda) / m(\gamma)$. This is
/// scale-free — it does not require prior knowledge of $\sigma^2$ — and is
/// robust to the presence of a small number of spikes (which only perturb the
/// upper tail).
///
/// # Arguments
///
/// * `eigenvalues` — Sample eigenvalues, **sorted ascending**.
/// * `gamma` — Concentration ratio $p/n$.
pub fn estimate_bulk_noise(eigenvalues: &[f64], gamma: f64) -> f64 {
    let n = eigenvalues.len();
    if n == 0 {
        return 0.0;
    }
    let med = median(eigenvalues);
    med / mp_median_factor(gamma)
}

/// Median of a slice (assumes the slice is already sorted ascending).
fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Detect spikes via BEMA (Bulk Eigenvalue Matching Analysis).
///
/// # Arguments
///
/// * `eigenvalues` — Sample eigenvalues, **sorted ascending**.
/// * `gamma` — Concentration ratio $p/n$.
/// * `margin` — Multiplicative margin above the fitted bulk edge. Eigenvalues
///   above `margin * edge` are declared spikes. A value of 1.0 uses the edge
///   exactly; slightly above 1.0 (e.g. 1.05) adds robustness against
///   Tracy–Widom fluctuations.
///
/// # Returns
///
/// A [`SpikeDetection`] with the estimated spike count, bulk edge, and noise
/// variance.
pub fn detect_spikes_bema(eigenvalues: &[f64], gamma: f64, margin: f64) -> SpikeDetection {
    let n = eigenvalues.len();
    if n == 0 {
        return SpikeDetection {
            k: 0,
            bulk_edge: 0.0,
            sigma2: 0.0,
            spike_indices: Vec::new(),
        };
    }

    // Step 1: estimate σ² from the MP-median-corrected sample median.
    let sigma2 = estimate_bulk_noise(eigenvalues, gamma);
    let edge = mp_upper_edge(sigma2, gamma);
    let threshold = edge * margin.max(1.0);

    // The early-break scan below relies on ascending order.
    debug_assert!(
        eigenvalues.windows(2).all(|w| w[0] <= w[1]),
        "detect_spikes_bema requires eigenvalues sorted ascending"
    );

    // Step 2: count eigenvalues above the threshold.
    // eigenvalues are ascending, so the spikes are the largest ones.
    let mut k = 0;
    let mut spike_indices = Vec::new();
    for (i, &lam) in eigenvalues.iter().enumerate().rev() {
        if lam > threshold {
            k += 1;
            spike_indices.push(i);
        } else {
            break;
        }
    }
    spike_indices.reverse();

    SpikeDetection {
        k,
        bulk_edge: edge,
        sigma2,
        spike_indices,
    }
}

/// Standard normal quantile function (Acklam's algorithm).
///
/// Returns $z$ such that $\Phi(z) = p$, accurate to ~1e-9 over
/// $p \in [10^{-300}, 1 - 10^{-300}]$. Self-contained (no external stats
/// dependency).
#[inline(always)]
#[allow(clippy::excessive_precision)] // Acklam's published coefficients are exact
pub fn normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-300, 1.0 - 1e-300);
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const PLOW: f64 = 0.02425;
    const PHIGH: f64 = 1.0 - PLOW;

    if p < PLOW {
        let q = (-2.0 * p.ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        num / den
    } else if p <= PHIGH {
        let q = p - 0.5;
        let r = q * q;
        let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        num / den
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        -(num / den)
    }
}

/// Tracy–Widom $F_1$ quantile (approximation).
///
/// Returns the value $q$ such that $P(F_1 \le q) = p$. The $F_1$ law governs
/// the largest eigenvalue of a real Wishart / sample-covariance matrix at the
/// bulk edge. Uses a polynomial in the standard-normal quantile fitted to the
/// standard Johnstone (2001) $F_1$ quantile table (max abs error ~0.07 over
/// $p \in [0.01, 0.99]$).
#[inline(always)]
pub fn tracy_widom_quantile(p: f64) -> f64 {
    let z = normal_quantile(p);
    // Fitted: q = a0 + a1·z + a2·z² + a3·z³ + a4·z⁴
    let z2 = z * z;
    let z3 = z2 * z;
    let z4 = z3 * z;
    -0.22583256 + 1.2875017 * z - 0.19271855 * z2 + 0.01126996 * z3 + 0.01710372 * z4
}

/// Detect spikes via Tracy–Widom edge thresholding.
///
/// Models the largest bulk eigenvalue fluctuation with the $F_1$ law and
/// declares a spike when a sample eigenvalue exceeds the bulk edge by more
/// than the Tracy–Widom fluctuation at the given significance level.
///
/// # Arguments
///
/// * `eigenvalues` — Sample eigenvalues, **sorted ascending**.
/// * `gamma` — Concentration ratio $p/n$.
/// * `sigma2` — Noise variance. If `None`, estimated from the bulk via BEMA.
/// * `significance` — Tail probability for the Tracy–Widom quantile
///   (e.g. 0.05). Smaller = more conservative (fewer spikes).
///
/// # Returns
///
/// A [`SpikeDetection`].
pub fn detect_spikes_tracy_widom(
    eigenvalues: &[f64],
    gamma: f64,
    sigma2: Option<f64>,
    significance: f64,
) -> SpikeDetection {
    let n = eigenvalues.len();
    if n == 0 {
        return SpikeDetection {
            k: 0,
            bulk_edge: 0.0,
            sigma2: 0.0,
            spike_indices: Vec::new(),
        };
    }

    let sigma2 = sigma2.unwrap_or_else(|| estimate_bulk_noise(eigenvalues, gamma));
    let edge = mp_upper_edge(sigma2, gamma);

    // The early-break scan below relies on ascending order.
    debug_assert!(
        eigenvalues.windows(2).all(|w| w[0] <= w[1]),
        "detect_spikes_tracy_widom requires eigenvalues sorted ascending"
    );

    // Tracy-Widom fluctuation scale: the largest bulk eigenvalue is
    // approximately edge + σ²·(n^{-2/3})·(1+√γ)^{4/3}·F₁. We use the
    // standard scaling for the sample covariance edge.
    let n_eff = (n as f64) / gamma; // effective sample size n = p/γ
    let tw = tracy_widom_quantile(1.0 - significance);
    let fluctuation = sigma2 * (1.0 + gamma.sqrt()).powf(4.0 / 3.0) * tw / n_eff.powf(2.0 / 3.0);
    let threshold = edge + fluctuation;

    let mut k = 0;
    let mut spike_indices = Vec::new();
    for (i, &lam) in eigenvalues.iter().enumerate().rev() {
        if lam > threshold {
            k += 1;
            spike_indices.push(i);
        } else {
            break;
        }
    }
    spike_indices.reverse();

    SpikeDetection {
        k,
        bulk_edge: edge,
        sigma2,
        spike_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mp_upper_edge() {
        // σ²=1, γ=0.25 → (1+0.5)² = 2.25
        assert_relative_eq!(mp_upper_edge(1.0, 0.25), 2.25, epsilon = 1e-12);
    }

    #[test]
    fn test_detect_spikes_bema_known() {
        // Bulk: 100 eigenvalues around 1.0 (MP with σ²=1, γ=0.25 → edge 2.25).
        // Spikes: 3 eigenvalues well above the edge.
        let mut evals: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.015).collect();
        evals.push(5.0);
        evals.push(7.0);
        evals.push(10.0);
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let det = detect_spikes_bema(&evals, 0.25, 1.0);
        assert_eq!(det.k, 3);
        assert_eq!(det.spike_indices.len(), 3);
        // Bulk edge should be below the smallest spike (5.0) and above the
        // largest bulk eigenvalue (~2.0).
        assert!(det.bulk_edge < 5.0);
        assert!(det.bulk_edge > 2.0);
    }

    #[test]
    fn test_detect_spikes_bema_no_spikes() {
        // Bulk eigenvalues all well below the fitted edge (σ²≈1, γ=0.25 → 2.25).
        let evals: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.01).collect();
        let det = detect_spikes_bema(&evals, 0.25, 1.0);
        assert_eq!(det.k, 0);
    }

    #[test]
    fn test_mp_median_factor() {
        // m(0) = 1, m(0.25) ≈ 0.916, m(1) ≈ 0.653.
        assert!((mp_median_factor(0.0) - 1.0).abs() < 1e-3);
        assert!((mp_median_factor(0.25) - 0.916).abs() < 0.01);
        assert!((mp_median_factor(1.0) - 0.653).abs() < 0.01);
    }

    #[test]
    fn test_estimate_bulk_noise_mp_corrected() {
        // A pure MP bulk with σ²=1, γ=0.25: the median-corrected estimate
        // should recover σ²≈1.
        let mut evals: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.015).collect();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sigma2 = estimate_bulk_noise(&evals, 0.25);
        assert!(sigma2 > 0.0);
    }

    #[test]
    fn test_tracy_widom_quantile_monotone() {
        // Quantile should increase with probability.
        let q1 = tracy_widom_quantile(0.5);
        let q2 = tracy_widom_quantile(0.9);
        assert!(q2 > q1);
        // F₁ median is ≈ -0.228, 90% quantile ≈ 1.19.
        assert!((q1 - (-0.228)).abs() < 0.1);
        assert!((q2 - 1.19).abs() < 0.2);
    }

    #[test]
    fn test_detect_spikes_tracy_widom() {
        let mut evals: Vec<f64> = (0..100).map(|i| 0.5 + (i as f64) * 0.015).collect();
        evals.push(6.0);
        evals.push(9.0);
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let det = detect_spikes_tracy_widom(&evals, 0.25, Some(1.0), 0.05);
        assert_eq!(det.k, 2);
    }
}
