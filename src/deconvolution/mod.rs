//! Spectral deconvolution — Marčenko-Pastur inversion (El Karoui 2008).
//!
//! Recovers the population spectral distribution $\mu_{\Sigma}$ from sample
//! eigenvalues by inverting the Marčenko-Pastur equation:
//!
//! $$m_{\hat{\Sigma}}(z) = \int \frac{d\mu_{\Sigma}(x)}{x(1 - c - c z m_{\hat{\Sigma}}(z)) - z}$$
//!
//! where $m_{\hat{\Sigma}}$ is the Stieltjes transform of the sample covariance
//! matrix and $\mu_{\Sigma}$ is the population spectral distribution.
//!
//! # Inversion formula
//!
//! Let $g(z) = m_{\hat{\Sigma}}(z)$. The MP equation can be rewritten as:
//!
//! $$m_{\Sigma}\!\left(\frac{z}{a(z)}\right) = a(z)\,g(z), \quad
//!   a(z) = 1 - c - c z g(z)$$
//!
//! For each $z = \lambda + i\eta$ on a grid, we obtain a pair
//! $(w, m_{\Sigma}(w))$ where $w = z / a(z)$.
//! The population spectral density is then recovered as:
//!
//! $$\rho_{\Sigma}\bigl(\Re(w)\bigr) = \frac{1}{\pi}\,\Im\bigl[m_{\Sigma}(w)\bigr]$$

pub mod adaptive;
/// Hybrid spike+bulk composition (renamed from `spiked` so the root
/// namespace stops shadowing the `crate::spiked` toolkit module with a
/// different meaning).
pub mod hybrid;
pub mod population;
pub mod precision;
pub mod shrinkage;

pub use adaptive::*;
pub use hybrid::*;
pub use population::*;
pub use precision::*;
pub use shrinkage::*;

use crate::config::RmtConfig;

/// Result of the spectral deconvolution.
#[derive(Debug, Clone)]
pub struct DeconvolutionResult {
    /// Grid points $\lambda$ (real axis) where the density is evaluated.
    pub lambda_grid: Vec<f64>,
    /// Population spectral density $\rho_{\Sigma}(\lambda)$.
    pub density: Vec<f64>,
    /// Real part of the transformed argument $\Re(w)$ for each grid point.
    pub w_re: Vec<f64>,
    /// Sample Stieltjes transform $g(z)$ at each grid point.
    pub sample_stieltjes_real: Vec<f64>,
    /// Sample Stieltjes transform $g(z)$ at each grid point.
    pub sample_stieltjes_imag: Vec<f64>,
    /// Population Stieltjes transform $m_{\Sigma}(w)$ at each grid point.
    pub population_stieltjes_real: Vec<f64>,
    /// Population Stieltjes transform $m_{\Sigma}(w)$ at each grid point.
    pub population_stieltjes_imag: Vec<f64>,
}

/// Compute the empirical Stieltjes transform $g(z) = \frac{1}{p}\sum_{j=1}^p \frac{1}{z - \lambda_j}$
/// for a single complex point $z = \lambda + i\eta$.
///
/// Returns $(g_\text{real}, g_\text{imag})$.
#[inline(always)]
pub fn empirical_stieltjes_at_point(z_real: f64, z_imag: f64, eigenvalues: &[f64]) -> (f64, f64) {
    let mut sum_real = 0.0;
    let mut sum_imag = 0.0;
    let p = eigenvalues.len() as f64;

    for &lambda_j in eigenvalues {
        let diff = z_real - lambda_j;
        let denom = diff * diff + z_imag * z_imag;
        let inv_denom = 1.0 / denom;
        // 1 / (z - λⱼ) = (diff - i·η) / (diff² + η²) = diff/denom - i·η/denom
        // But z = λ + iη, so z - λⱼ = (λ-λⱼ) + iη
        // 1/((λ-λⱼ) + iη) = (λ-λⱼ)/(diff²+η²) - i·η/(diff²+η²)
        sum_real += diff * inv_denom;
        sum_imag -= z_imag * inv_denom; // negative because imag part of 1/(a+ib) = -b/(a²+b²)
    }

    (sum_real / p, sum_imag / p)
}

/// Perform spectral deconvolution (MP inversion) to recover the population
/// spectral distribution from sample eigenvalues.
///
/// # Arguments
///
/// * `eigenvalues` — Sample eigenvalues (sorted, length p)
/// * `c` — Concentration ratio p / n
/// * `n_points` — Number of grid points for the density evaluation
/// * `eta` — Regularization parameter (imaginary shift). Default: 0.1 / sqrt(p)
/// * `lambda_min` — Minimum lambda for the grid. If None, inferred from eigenvalues.
/// * `lambda_max` — Maximum lambda for the grid. If None, inferred from eigenvalues.
/// * `config` — `RmtConfig` (used for eta and consistency with RIE)
///
/// # Returns
///
/// `DeconvolutionResult` containing the population spectral density and
/// intermediate Stieltjes transforms.
///
/// # Example
///
/// ```rust,ignore
/// use shrinkers::{spectral_deconvolution, DeconvolutionResult, RmtConfig};
/// let evals = vec![0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
/// let config = RmtConfig::new(0.3);
/// let result = spectral_deconvolution(&evals, 0.3, 200, None, None, None, &config);
/// ```
pub fn spectral_deconvolution(
    eigenvalues: &[f64],
    c: f64,
    n_points: usize,
    eta: Option<f64>,
    lambda_min: Option<f64>,
    lambda_max: Option<f64>,
    config: &RmtConfig,
) -> DeconvolutionResult {
    let p = eigenvalues.len();
    if p == 0 {
        return DeconvolutionResult {
            lambda_grid: Vec::new(),
            density: Vec::new(),
            w_re: Vec::new(),
            sample_stieltjes_real: Vec::new(),
            sample_stieltjes_imag: Vec::new(),
            population_stieltjes_real: Vec::new(),
            population_stieltjes_imag: Vec::new(),
        };
    }

    let eta_val = eta.unwrap_or_else(|| crate::stieltjes::default_eta(p));

    // Determine the grid range from eigenvalues with margins
    let min_ev = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ev = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let range = if (max_ev - min_ev) > 0.0 {
        max_ev - min_ev
    } else {
        1.0
    };
    let margin = 0.2 * range;
    let lo = lambda_min.unwrap_or((min_ev - margin).max(0.0));
    let hi = lambda_max.unwrap_or(max_ev + margin);

    // Build the grid: z_k = λ_k + i·η for k = 0..n_points-1
    let step = if n_points > 1 {
        (hi - lo) / (n_points as f64 - 1.0)
    } else {
        0.0
    };

    let mut lambda_grid = Vec::with_capacity(n_points);
    for k in 0..n_points {
        lambda_grid.push(lo + step * (k as f64));
    }

    // Resolve Auto config based on the problem size p, then select the
    // Stieltjes method. The grid points are a uniform grid over [lo, hi] —
    // NOT the sample eigenvalues — so we evaluate the fast Stieltjes kernel
    // at those arbitrary query points.
    let resolved = config.resolve_auto(p);
    let method = resolved.stieltjes_method;
    let parallelism = resolved.parallelism;
    let cutoff_ratio = resolved.cutoff.ratio();
    let fft_grid_size = resolved.fft_grid_size.grid_points();

    // Step 1: Compute g(z) = sample Stieltjes transform at every grid point
    // in one batched call to the fast Stieltjes library.
    //
    // The library computes S(λ) = Σⱼ 1/((λ-λⱼ) - iη) = Σⱼ (λ-λⱼ)/((λ-λⱼ)²+η²)
    // + i·Σⱼ η/((λ-λⱼ)²+η²), i.e. convention B (Im[S] > 0 for Im[z] > 0),
    // scaled by 1/p. This matches the convention used by the rest of the
    // deconvolution (see below), so no sign flip is needed.
    let raw = crate::stieltjes::compute_stieltjes_at_points(
        &lambda_grid,
        eigenvalues,
        eta_val,
        method,
        cutoff_ratio,
        parallelism,
        fft_grid_size,
    );
    let inv_p = 1.0 / (p as f64);

    let mut density = Vec::with_capacity(n_points);
    let mut w_re = Vec::with_capacity(n_points);
    let mut sample_stieltjes_real = Vec::with_capacity(n_points);
    let mut sample_stieltjes_imag = Vec::with_capacity(n_points);
    let mut population_stieltjes_real = Vec::with_capacity(n_points);
    let mut population_stieltjes_imag = Vec::with_capacity(n_points);

    for (k, &lambda_k) in lambda_grid.iter().enumerate() {
        let z_real = lambda_k;
        let z_imag = eta_val;

        // g(z) in convention B (Im[g] > 0 for Im[z] > 0), scaled by 1/p.
        let (s_real, s_imag) = raw[k];
        let g_real = -s_real * inv_p;
        let g_imag = s_imag * inv_p;

        // Step 2: Compute a(z) = 1 - c - c·z·g(z)   [convention B]
        // z·g(z) = (λ + iη)·(g_real + i·g_imag)
        //        = (λ·g_real - η·g_imag) + i·(λ·g_imag + η·g_real)
        let zg_real = z_real * g_real - z_imag * g_imag;
        let zg_imag = z_real * g_imag + z_imag * g_real;

        let a_real = 1.0 - c - c * zg_real;
        let a_imag = -c * zg_imag;

        // Step 3: Compute w = z / a(z) = z · conj(a) / |a|²
        let a_norm_sq = a_real * a_real + a_imag * a_imag;
        let w_real = if a_norm_sq > 0.0 {
            // z / a = (z_real + i·z_imag) · (a_real - i·a_imag) / |a|²
            (z_real * a_real + z_imag * a_imag) / a_norm_sq
        } else {
            z_real
        };

        // Step 4: Compute m_Σ(w) = a(z) · g(z)   [convention B]
        // a·g = (a_real + i·a_imag) · (g_real + i·g_imag)
        let m_real = a_real * g_real - a_imag * g_imag;
        let m_imag = a_real * g_imag + a_imag * g_real;

        // Step 5: Population spectral density at w_re:
        // ρ_Σ(Re(w)) = (1/π) · Im[m_Σ(w)]   [convention B: Im[m_Σ] > 0]
        let rho = m_imag / std::f64::consts::PI;

        density.push(rho);
        w_re.push(w_real);
        sample_stieltjes_real.push(g_real);
        sample_stieltjes_imag.push(g_imag);
        population_stieltjes_real.push(m_real);
        population_stieltjes_imag.push(m_imag);
    }

    DeconvolutionResult {
        lambda_grid,
        density,
        w_re,
        sample_stieltjes_real,
        sample_stieltjes_imag,
        population_stieltjes_real,
        population_stieltjes_imag,
    }
}

// ──────────────────────────────────────────────
//  Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Parallelism, StieltjesMethod};
    use approx::assert_relative_eq;

    // ── Golden master: deterministic eigenvalues, exact expected output ──

    /// Deterministic regression test.
    ///
    /// Golden-master values computed via the Python binding of
    /// `spectral_deconvolution` on 2026-07-28 (the function has since been
    /// un-exported from the pymodule; the Rust entry point is unchanged).
    /// If this test fails after a code change, the change altered the
    /// numerical output — verify it's intentional.
    #[test]
    fn test_golden_master_regression() {
        let evals = vec![6.0, 4.0, 2.5, 1.5, 1.0, 0.6, 0.3, 0.1];
        let c = 0.3;
        let n_points = 20;
        let eta = 0.1;
        let config = RmtConfig::new(c);

        let result = spectral_deconvolution(&evals, c, n_points, Some(eta), None, None, &config);

        // Expected values (golden master)
        let expected_density: Vec<f64> = vec![
            0.163_697_932_028_836,
            0.2851819271746207,
            0.1765006334786763,
            0.2192264328067419,
            0.5010071341958172,
            0.0518501085056742,
            0.0428118110149010,
            0.2663275661292691,
            0.0293582653925036,
            0.0183543672485776,
            0.0148125784602854,
            0.305_443_223_319_169,
            0.0288178793959170,
            0.0130051927802807,
            0.0119488084994027,
            0.0123759126259342,
            1.0623683376384705,
            0.0546501666657708,
            0.0140263964367322,
            0.0065402373029941,
        ];
        let expected_w_re: Vec<f64> = vec![
            -0.008612973476024624,
            0.4676581369959773,
            0.9127431252274725,
            1.1509410150776325,
            1.0980131742086798,
            2.015_262_766_173_354,
            3.3014493733074897,
            1.8406120930787837,
            2.9633761996543644,
            4.066_581_734_185_9,
            6.063_997_173_543_702,
            2.323_830_655_535_765,
            3.7558706910379773,
            4.7566929520083105,
            6.053_556_618_771_238,
            9.419_904_323_527_75,
            1.5989059507638193,
            3.956_052_912_810_217,
            4.943_382_490_552_525,
            5.615_818_422_806_383,
        ];
        let expected_si: Vec<f64> = vec![
            0.8048156877928192,
            1.1774049311429964,
            0.6579616820030227,
            0.615_606_401_335_683,
            1.3243311654680288,
            0.1454015894475605,
            0.2389652287481778,
            0.4310299782667528,
            0.0721153387019700,
            0.0606054510235282,
            0.2296864984302125,
            0.3750905916017916,
            0.0555390537296531,
            0.0311437458978916,
            0.0364949698437683,
            0.1125484113783906,
            1.0354366712416319,
            0.0706938370526611,
            0.0231108620281907,
            0.0122351457852784,
        ];
        let expected_pi: Vec<f64> = vec![
            0.5142722206696324,
            0.8959254473483677,
            0.5544930934905543,
            0.6887201507783569,
            1.573_960_332_185_655,
            0.1628919199692596,
            0.1344972709712876,
            0.8366927252001617,
            0.0922317108792288,
            0.0576619453094206,
            0.0465350876715550,
            0.9595781864682879,
            0.0905340382022495,
            0.0408570180970488,
            0.0375382890008749,
            0.0388800761871040,
            3.3375285649314197,
            0.1716885621146434,
            0.0440652240019759,
            0.0205467614638201,
        ];

        assert_eq!(result.density.len(), n_points);
        for (&got, &exp) in result.density.iter().zip(expected_density.iter()) {
            assert_relative_eq!(got, exp, epsilon = 1e-14);
        }
        for (&got, &exp) in result.w_re.iter().zip(expected_w_re.iter()) {
            assert_relative_eq!(got, exp, epsilon = 1e-14);
        }
        for (&got, &exp) in result.sample_stieltjes_imag.iter().zip(expected_si.iter()) {
            assert_relative_eq!(got, exp, epsilon = 1e-14);
        }
        for (&got, &exp) in result
            .population_stieltjes_imag
            .iter()
            .zip(expected_pi.iter())
        {
            assert_relative_eq!(got, exp, epsilon = 1e-14);
        }
    }

    // ── Convention tests ──

    /// Verify that `empirical_stieltjes_at_point` uses convention A (Im[g] < 0).
    #[test]
    fn test_convention_a_imag_negative() {
        let evals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let eta = 0.1;
        for &lambda_i in &evals {
            let (_re, im) = empirical_stieltjes_at_point(lambda_i, eta, &evals);
            assert!(
                im < 0.0,
                "Convention A should give Im[g] < 0, got Im[g] = {im} at λ = {lambda_i}"
            );
        }
    }

    /// Verify that `spectral_deconvolution` returns convention B (Im[g] > 0).
    #[test]
    fn test_convention_b_imag_positive() {
        let evals = vec![0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
        let config = RmtConfig::new(0.3);
        let result = spectral_deconvolution(&evals, 0.3, 50, Some(0.1), None, None, &config);
        let n_pos = result
            .sample_stieltjes_imag
            .iter()
            .filter(|&&im| im > 0.0)
            .count();
        assert_eq!(
            n_pos,
            result.sample_stieltjes_imag.len(),
            "All sample Stieltjes imag parts should be positive (convention B)"
        );
    }

    // ── Mathematical invariant tests ──

    /// Density should integrate to approximately 1 over λ.
    #[test]
    fn test_density_integral_over_lambda() {
        let evals = vec![0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
        let config = RmtConfig::new(0.3);
        let result = spectral_deconvolution(&evals, 0.3, 200, Some(0.05), None, None, &config);

        // Trapezoidal integration over λ
        let lam = &result.lambda_grid;
        let rho = &result.density;
        let mut integral = 0.0;
        for i in 1..lam.len() {
            let h = lam[i] - lam[i - 1];
            integral += 0.5 * h * (rho[i] + rho[i - 1]);
        }
        assert!(
            (integral - 1.0).abs() < 0.05,
            "Density should integrate to ≈1 over λ, got {integral}"
        );
    }

    /// Density should be mostly non-negative (small negative values may appear
    /// at the grid edges where the finite-η regularization dominates).
    #[test]
    fn test_density_non_negative() {
        let evals = vec![0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
        let config = RmtConfig::new(0.3);
        // Use larger eta for smoother results
        let result = spectral_deconvolution(&evals, 0.3, 200, Some(0.2), None, None, &config);

        let n_neg = result.density.iter().filter(|&&d| d < -1e-10).count();
        let n_total = result.density.len();
        // Some small negative values can appear from finite-η numerical noise.
        // At most 30% of entries should be negative.
        assert!(
            n_neg < n_total * 3 / 10,
            "Density should be mostly non-negative, found {n_neg}/{n_total} negative entries"
        );
    }

    /// For Σ = I (pure noise), the deconvolved density should peak near λ = 1.
    #[test]
    fn test_white_noise_peak_near_one() {
        let c = 0.5_f64;
        let p = 500;
        let lam_mp_min = (1.0 - c.sqrt()).powi(2);
        let lam_mp_max = (1.0 + c.sqrt()).powi(2);

        // Generate MP-distributed eigenvalues
        use rand::RngExt;
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(21);
        let mut evals: Vec<f64> = Vec::with_capacity(p);
        while evals.len() < p {
            let lam = rng.random::<f64>() * (lam_mp_max - lam_mp_min) + lam_mp_min;
            let pdf = ((lam_mp_max - lam) * (lam - lam_mp_min)).sqrt()
                / (2.0 * std::f64::consts::PI * c * lam);
            if rng.random::<f64>() < pdf / 2.5 {
                evals.push(lam);
            }
        }
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let config = RmtConfig::new(c);
        let result = spectral_deconvolution(&evals, c, 300, Some(0.05), None, None, &config);

        // Find peak density
        let max_idx = result
            .density
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // The peak should be near λ = 1 (the population eigenvalue for Σ = I)
        let peak_w = result.w_re[max_idx];
        assert!(
            (peak_w - 1.0).abs() < 0.5,
            "Peak should be near 1.0 for Σ = I, got {peak_w}"
        );
    }

    /// For a spiked model, the deconvolved density should show peaks
    /// near the spike locations.
    #[test]
    fn test_spiked_model_peaks() {
        let c = 0.2_f64;
        let p = 300;
        let spikes = [5.0, 2.5];
        let lam_mp_min = (1.0 - c.sqrt()).powi(2);
        let lam_mp_max = (1.0 + c.sqrt()).powi(2);

        use rand::RngExt;
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(22);
        let mut evals: Vec<f64> = Vec::with_capacity(p);
        while evals.len() < p - spikes.len() {
            let lam = rng.random::<f64>() * (lam_mp_max - lam_mp_min) + lam_mp_min;
            let pdf = ((lam_mp_max - lam) * (lam - lam_mp_min)).sqrt()
                / (2.0 * std::f64::consts::PI * c * lam);
            if rng.random::<f64>() < pdf / 2.5 {
                evals.push(lam);
            }
        }
        for &s in &spikes {
            evals.push(s);
        }
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let config = RmtConfig::new(c);
        let result = spectral_deconvolution(&evals, c, 300, Some(0.05), None, None, &config);

        // The density should have positive values near the spike locations
        for &spike in &spikes {
            let near_spike: Vec<f64> = result
                .w_re
                .iter()
                .zip(result.density.iter())
                .filter(|&(w, _)| (w - spike).abs() < 0.5)
                .map(|(_, &d)| d)
                .collect();
            if !near_spike.is_empty() {
                let max_near = near_spike.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                assert!(
                    max_near > 0.0,
                    "Density should be positive near spike at λ = {spike}"
                );
            }
        }
    }

    // ── Edge case tests ──

    #[test]
    fn test_deconvolution_empty() {
        let config = RmtConfig::new(0.5);
        let result = spectral_deconvolution(&[], 0.5, 100, None, None, None, &config);
        assert!(result.lambda_grid.is_empty());
        assert!(result.density.is_empty());
    }

    #[test]
    fn test_deconvolution_single_eigenvalue() {
        let evals = vec![1.0];
        let config = RmtConfig::new(0.1);
        let result =
            spectral_deconvolution(&evals, 0.1, 50, Some(0.1), Some(0.0), Some(2.0), &config);
        assert_eq!(result.lambda_grid.len(), 50);
        assert_eq!(result.density.len(), 50);
        // Peak should be near 1.0
        let max_idx = result
            .density
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let peak_lambda = result.lambda_grid[max_idx];
        assert!(
            (peak_lambda - 1.0).abs() < 0.5,
            "Peak should be near 1.0, got {peak_lambda}"
        );
    }

    #[test]
    fn test_all_equal_eigenvalues() {
        let evals = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let config = RmtConfig::new(0.5);
        let result =
            spectral_deconvolution(&evals, 0.5, 50, Some(0.1), Some(0.0), Some(2.0), &config);
        assert_eq!(result.lambda_grid.len(), 50);
        // All density should be finite
        for &d in &result.density {
            assert!(d.is_finite(), "Density should be finite");
        }
    }

    #[test]
    fn test_extreme_c_values() {
        let evals = vec![0.5, 1.0, 1.5, 2.0, 3.0, 5.0];

        // Very small c (p ≪ n)
        let config_small = RmtConfig::new(0.01);
        let result_small =
            spectral_deconvolution(&evals, 0.01, 100, Some(0.1), None, None, &config_small);
        assert_eq!(result_small.lambda_grid.len(), 100);
        assert!(result_small.density.iter().all(|&d| d.is_finite()));

        // Very large c (p ≈ n)
        let config_large = RmtConfig::new(0.95);
        let result_large =
            spectral_deconvolution(&evals, 0.95, 100, Some(0.1), None, None, &config_large);
        assert_eq!(result_large.lambda_grid.len(), 100);
        assert!(result_large.density.iter().all(|&d| d.is_finite()));
    }

    #[test]
    fn test_custom_grid_bounds() {
        let evals = vec![0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
        let config = RmtConfig::new(0.3);
        let result =
            spectral_deconvolution(&evals, 0.3, 100, Some(0.1), Some(0.0), Some(10.0), &config);
        assert_eq!(result.lambda_grid.len(), 100);
        assert!((result.lambda_grid[0] - 0.0).abs() < 1e-10);
        assert!((result.lambda_grid[99] - 10.0).abs() < 1e-10);
    }

    // ── Stieltjes transform tests ──

    #[test]
    fn test_empirical_stieltjes_simple() {
        let evals = vec![1.0, 2.0, 3.0];
        // empirical_stieltjes_at_point uses convention A: 1/(z - λⱼ)
        // z = 2 + 0.1i
        // 1/(z-1) = 1/(1+0.1i) = (1-0.1i)/1.01 ≈ 0.990099 - 0.099010i
        // 1/(z-2) = 1/(0+0.1i) = -10i
        // 1/(z-3) = 1/(-1+0.1i) = (-1-0.1i)/1.01 ≈ -0.990099 - 0.099010i
        // Sum = (0.990099 + 0 - 0.990099) + i(-0.099010 - 10 - 0.099010) = 0 - 10.19802i
        // g_A = sum/3 ≈ 0 - 3.39934i
        let (ga_real, ga_imag) = empirical_stieltjes_at_point(2.0, 0.1, &evals);
        let expected_ga_imag = -10.198019801980198_f64 / 3.0;
        assert_relative_eq!(ga_real, 0.0, epsilon = 1e-12);
        assert_relative_eq!(ga_imag, expected_ga_imag, epsilon = 1e-10);

        // Convention B: g_B = -g_A, so Im[g_B] > 0
        let gb_real = -ga_real;
        let gb_imag = -ga_imag;
        assert_relative_eq!(gb_real, 0.0, epsilon = 1e-12);
        assert!(gb_imag > 0.0, "Convention B should have Im[g] > 0");
    }

    /// Verify that the Stieltjes transform is consistent with the
    /// definition: g(z) = (1/p) Σ 1/(λⱼ - z) for convention B.
    #[test]
    fn test_stieltjes_definition_consistency() {
        let evals = vec![0.5, 1.0, 2.0, 3.0, 5.0];
        let eta = 0.05;
        let p = evals.len() as f64;

        for &lambda_i in &evals {
            // Compute g_B(z) = (1/p) Σ 1/(λⱼ - z) directly with plain
            // (re, im) arithmetic — independent of any complex helper.
            let (mut sum_re, mut sum_im) = (0.0_f64, 0.0_f64);
            for &lambda_j in &evals {
                // 1/(λⱼ - z) = 1/((λⱼ - λ_i) - iη) = diff/denom + i·η/denom
                let diff = lambda_j - lambda_i;
                let denom = diff * diff + eta * eta;
                let inv_denom = 1.0 / denom;
                sum_re += diff * inv_denom;
                sum_im += eta * inv_denom;
            }
            let (gb_direct_re, gb_direct_im) = (sum_re / p, sum_im / p);

            // Get g_A from our function, convert to g_B
            let (ga_re, ga_im) = empirical_stieltjes_at_point(lambda_i, eta, &evals);
            let gb_re = -ga_re;
            let gb_im = -ga_im;

            assert_relative_eq!(gb_re, gb_direct_re, epsilon = 1e-14);
            assert_relative_eq!(gb_im, gb_direct_im, epsilon = 1e-14);
        }
    }

    // ── Round-trip consistency: deconvolution of MP eigenvalues ──

    /// For Σ = I, the deconvolution should recover a density whose
    /// integral over λ is approximately 1.
    #[test]
    fn test_mp_round_trip_integral() {
        let c = 0.5_f64;
        let p = 500;
        let lam_mp_min = (1.0 - c.sqrt()).powi(2);
        let lam_mp_max = (1.0 + c.sqrt()).powi(2);

        use rand::RngExt;
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(23);
        let mut evals: Vec<f64> = Vec::with_capacity(p);
        while evals.len() < p {
            let lam = rng.random::<f64>() * (lam_mp_max - lam_mp_min) + lam_mp_min;
            let pdf = ((lam_mp_max - lam) * (lam - lam_mp_min)).sqrt()
                / (2.0 * std::f64::consts::PI * c * lam);
            if rng.random::<f64>() < pdf / 2.5 {
                evals.push(lam);
            }
        }
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let config = RmtConfig::new(c);
        let result = spectral_deconvolution(&evals, c, 300, Some(0.05), None, None, &config);

        // Integral over λ
        let lam = &result.lambda_grid;
        let rho = &result.density;
        let mut integral = 0.0;
        for i in 1..lam.len() {
            let h = lam[i] - lam[i - 1];
            integral += 0.5 * h * (rho[i] + rho[i - 1]);
        }
        assert!(
            (integral - 1.0).abs() < 0.1,
            "MP round-trip integral should be ≈1, got {integral}"
        );
    }

    // ── Smoke test (existing, kept for backward compat) ──

    #[test]
    fn test_deconvolution_smoke() {
        let c = 0.5_f64;
        let lam_mp_min = (1.0 - c.sqrt()).powi(2);
        let lam_mp_max = (1.0 + c.sqrt()).powi(2);
        let p = 500;

        use rand::RngExt;
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(24);
        let mut mp_evals: Vec<f64> = Vec::with_capacity(p);

        while mp_evals.len() < p - 1 {
            let lambda = rng.random::<f64>() * (lam_mp_max - lam_mp_min) + lam_mp_min;
            let pdf = ((lam_mp_max - lambda) * (lambda - lam_mp_min)).sqrt()
                / (2.0 * std::f64::consts::PI * c * lambda);
            if rng.random::<f64>() < pdf / 2.5 {
                mp_evals.push(lambda);
            }
        }
        mp_evals.push(5.0);
        mp_evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let config = RmtConfig::new(c);
        let result = spectral_deconvolution(&mp_evals, c, 300, Some(0.05), None, None, &config);

        // Convention B: Im[g] > 0
        let pos_imag = result
            .sample_stieltjes_imag
            .iter()
            .filter(|&&im| im > 0.0)
            .count();
        assert!(
            pos_imag > result.sample_stieltjes_imag.len() / 2,
            "Most sample Stieltjes imag parts should be positive (convention B)"
        );

        // Density should be finite and positive somewhere
        let max_density = result
            .density
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(max_density.is_finite(), "Density max should be finite");
        assert!(max_density > 0.0, "Density should be positive somewhere");
    }

    // ── FFT path: the deconvolution grid should use the O(p log p) FFT
    //    kernel and produce a density consistent with the exact Blocked path. ──

    #[test]
    fn test_fft_deconvolution_matches_blocked() {
        let c = 0.5_f64;
        let p = 2000;
        let lam_mp_min = (1.0 - c.sqrt()).powi(2);
        let lam_mp_max = (1.0 + c.sqrt()).powi(2);

        use rand::RngExt;
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(25);
        let mut evals: Vec<f64> = Vec::with_capacity(p);
        while evals.len() < p {
            let lam = rng.random::<f64>() * (lam_mp_max - lam_mp_min) + lam_mp_min;
            let pdf = ((lam_mp_max - lam) * (lam - lam_mp_min)).sqrt()
                / (2.0 * std::f64::consts::PI * c * lam);
            if rng.random::<f64>() < pdf / 2.5 {
                evals.push(lam);
            }
        }
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let n_points = 200;
        let eta = 0.05;

        // Exact Blocked path.
        let blocked_cfg = RmtConfig::new(c)
            .with_stieltjes(StieltjesMethod::Blocked)
            .with_parallelism(Parallelism::Sequential);
        let blocked =
            spectral_deconvolution(&evals, c, n_points, Some(eta), None, None, &blocked_cfg);

        // FFT path (O(p log p) on the grid).
        let fft_cfg = RmtConfig::new(c)
            .with_stieltjes(StieltjesMethod::Fft5)
            .with_parallelism(Parallelism::Sequential);
        let fft = spectral_deconvolution(&evals, c, n_points, Some(eta), None, None, &fft_cfg);

        assert_eq!(blocked.density.len(), fft.density.len());

        // The FFT is approximate (~15% grid error on the long-range real part),
        // so we check the densities are of the same order of magnitude and
        // both integrate to ≈1, rather than exact equality.
        let integrate = |rho: &[f64], lam: &[f64]| -> f64 {
            let mut s = 0.0;
            for i in 1..lam.len() {
                let h = lam[i] - lam[i - 1];
                s += 0.5 * h * (rho[i] + rho[i - 1]);
            }
            s
        };
        let int_blocked = integrate(&blocked.density, &blocked.lambda_grid);
        let int_fft = integrate(&fft.density, &fft.lambda_grid);

        assert!(
            (int_blocked - 1.0).abs() < 0.1,
            "Blocked integral should be ≈1, got {int_blocked}"
        );
        assert!(
            (int_fft - 1.0).abs() < 0.15,
            "FFT integral should be ≈1, got {int_fft}"
        );

        // Both should be finite and positive somewhere.
        for (name, rho) in [("blocked", &blocked.density), ("fft", &fft.density)] {
            let max = rho.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!(max.is_finite(), "{name} density max should be finite");
            assert!(max > 0.0, "{name} density should be positive somewhere");
        }
    }
}
