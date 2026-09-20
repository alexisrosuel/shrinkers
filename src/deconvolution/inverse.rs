//! Inverse nonlinear shrinkage — estimate the **precision matrix** eigenvalues
//! directly (Ledoit & Wolf, *Quadratic shrinkage for large covariance
//! matrices*, Bernoulli 28(3), 2022).
//!
//! # Why not just invert a cleaned covariance?
//!
//! Nonlinear shrinkage of the *covariance* eigenvalues answers the wrong
//! question when the object of interest is $\Omega = \Sigma^{-1}$ (minimum
//! variance portfolios, partial correlations, Gaussian graphical models):
//! inverting a covariance-optimal estimator systematically **over-inflates the
//! small inverted eigenvalues**, because the covariance loss does not penalise
//! errors on the inverse scale. The *inverse shrinkage* family instead derives
//! the optimal eigenvalue of the inverse directly, under three losses:
//!
//! | Method | Loss the eigenvalues are optimal for |
//! |---|---|
//! | [`InverseShrinkageMethod::Qis`] | Frobenius / inverse Stein / minimum-variance |
//! | [`InverseShrinkageMethod::Lis`] | Stein's loss (linear-inverse shrinkage) |
//! | [`InverseShrinkageMethod::Gis`] | symmetrized Kullback–Leibler (geometric mean of QIS and LIS) |
//!
//! # Estimator
//!
//! With $\lambda_i$ the sample eigenvalues, $x_i = 1/\lambda_i$,
//! $p$ the dimension and $c = p/n \le 1$ the concentration ratio, the
//! Ledoit–Wolf bandwidth is
//!
//! $$h = \frac{\min(c^2, c^{-2})^{0.35}}{p^{0.35}}$$
//!
//! and the smoothed Stein shrinker and its conjugate are
//!
//! $$\theta_i = \frac{1}{p}\sum_j \frac{x_j (x_j - x_i)}{(x_j - x_i)^2 + x_j^2 h^2},
//! \qquad
//! H\theta_i = \frac{1}{p}\sum_j \frac{x_j^2 h}{(x_j - x_i)^2 + x_j^2 h^2},$$
//!
//! from which $A_i = \theta_i^2 + H\theta_i^2$. The quadratic-inverse
//! covariance eigenvalue and the linear-inverse precision eigenvalue are
//!
//! $$\delta_i^{\text{QIS}} = \Bigl[(1-c)^2 x_i + 2c(1-c)\,x_i\theta_i
//!   + c^2 x_i A_i\Bigr]^{-1}, \qquad
//!   \omega_i^{\text{LIS}} = (1-c)x_i + 2c\,x_i\theta_i .$$
//!
//! QIS then rescales $\delta^{\text{QIS}}$ to preserve the trace
//! ($\sum_i \delta_i = \sum_i \lambda_i$, i.e. $\sum_i 1/\omega_i$ is fixed),
//! LIS clips $\omega^{\text{LIS}}$ from below at $\min_j x_j$, and GIS
//! reconstructs the covariance as the geometric mean of the QIS and LIS
//! covariance estimates, giving $\omega_i^{\text{GIS}} =
//! \sqrt{\omega_i^{\text{LIS}} / \delta_i^{\text{QIS}}}$.
//!
//! # Reuse of the Stieltjes kernel
//!
//! $\theta$ and $H\theta$ are exactly the real and imaginary parts of the
//! empirical Stieltjes transform evaluated on the **scaled ray**
//! $z_i = \lambda_i(1 + ih)$:
//!
//! $$\theta_i = \lambda_i\,\Re\,m(z_i), \qquad H\theta_i = \lambda_i\,\Im\,m(z_i).$$
//!
//! This crate's pointwise estimators use a *constant* imaginary shift $\eta$;
//! the inverse-shrinkage bandwidth is proportional to the eigenvalue instead,
//! so the per-point kernels are reused through
//! [`crate::stieltjes::compute_stieltjes_scaled_ray`] with $\eta_i = h\lambda_i$.
//! The identity above is what that function exploits, keeping the estimator on
//! the same optimized kernels as the rest of the crate.
//!
//! # Domain
//!
//! Only the non-singular regime $c \le 1$ ($p \le n$) is implemented — the
//! regime where all three losses are defined and where the crate's
//! concentration-ratio contract lives. Eigenvalues must be strictly positive.

use crate::config::RmtConfig;

/// Which inverse-shrinkage loss the precision eigenvalues are optimal for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InverseShrinkageMethod {
    /// Quadratic-inverse shrinkage (Frobenius, inverse Stein's, minimum
    /// variance). The default and the best all-round choice.
    #[default]
    Qis,
    /// Linear-inverse shrinkage (Stein's loss).
    Lis,
    /// Geometric-inverse shrinkage (symmetrized Kullback–Leibler): the
    /// geometric mean of the QIS and LIS covariance estimates.
    Gis,
}

impl InverseShrinkageMethod {
    /// Lower-case Python-facing name.
    pub fn as_str(self) -> &'static str {
        match self {
            InverseShrinkageMethod::Qis => "qis",
            InverseShrinkageMethod::Lis => "lis",
            InverseShrinkageMethod::Gis => "gis",
        }
    }

    /// Parse the Python-facing name (case-insensitive).
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "qis" | "quadratic" | "quadratic_inverse" => Some(InverseShrinkageMethod::Qis),
            "lis" | "linear" | "linear_inverse" => Some(InverseShrinkageMethod::Lis),
            "gis" | "geometric" | "geometric_inverse" => Some(InverseShrinkageMethod::Gis),
            _ => None,
        }
    }
}

/// Eigenvalues of the inverse-shrinkage precision estimate, plus the matching
/// covariance eigenvalues and the intermediate shrunken-Stein quantities.
#[derive(Debug, Clone)]
pub struct InverseShrinkageResult {
    /// Eigenvalues of the precision estimate $\hat{\Omega} = \hat{\Sigma}^{-1}$,
    /// order parallel to the input eigenvalues.
    pub precision_eigenvalues: Vec<f64>,
    /// Eigenvalues of the covariance estimate $\hat{\Sigma}$ (the reciprocals,
    /// trace-rescaled for QIS). Order parallel to the input eigenvalues.
    pub covariance_eigenvalues: Vec<f64>,
    /// The Ledoit–Wolf bandwidth $h$.
    pub smoothing: f64,
    /// Smoothed Stein shrinker $\theta_i$ (exposed for diagnostics and tests).
    pub theta: Vec<f64>,
    /// Conjugate shrinker $H\theta_i$ (exposed for diagnostics and tests).
    pub htheta: Vec<f64>,
}

impl InverseShrinkageResult {
    fn empty() -> Self {
        Self {
            precision_eigenvalues: Vec::new(),
            covariance_eigenvalues: Vec::new(),
            smoothing: 0.0,
            theta: Vec::new(),
            htheta: Vec::new(),
        }
    }
}

/// Estimate the precision-matrix eigenvalues directly by inverse nonlinear
/// shrinkage (Ledoit & Wolf 2022).
///
/// # Arguments
///
/// * `eigenvalues` — sample eigenvalues (length p), strictly positive.
/// * `c` — concentration ratio `p / n`, in `(0, 1]`.
/// * `method` — which inverse-shrinkage loss to use (see
///   [`InverseShrinkageMethod`]).
/// * `config` — `RmtConfig`; its `stieltjes_method`, `cutoff` and `parallelism`
///   select the kernel that evaluates the ray transform. Its `eta` field is
///   **ignored**: the imaginary shift is the estimator's own `h·λ_i`.
///
/// # Returns
///
/// An [`InverseShrinkageResult`]. The precision eigenvalues are, for QIS and
/// LIS, directly the optimal inverse eigenvalues; for GIS they are the inverse
/// of the geometric-mean covariance eigenvalues.
pub fn inverse_nonlinear_shrinkage(
    eigenvalues: &[f64],
    c: f64,
    method: InverseShrinkageMethod,
    config: &RmtConfig,
) -> InverseShrinkageResult {
    let p = eigenvalues.len();
    if p == 0 {
        return InverseShrinkageResult::empty();
    }
    let p_f = p as f64;
    let inv_p = 1.0 / p_f;

    // Ledoit–Wolf bandwidth, shared by every method in the family.
    let smoothing = (c * c).min(1.0 / (c * c)).powf(0.35) / p_f.powf(0.35);

    // θ and Hθ are the real/imaginary parts of the Stieltjes transform on the
    // scaled ray z_i = λ_i(1 + i·h) — reuse the crate's per-point kernels.
    let raw = crate::stieltjes::compute_stieltjes_scaled_ray(
        eigenvalues,
        smoothing,
        config.stieltjes_method,
        config.cutoff.ratio(),
        config.parallelism,
    );

    let mut theta = vec![0.0_f64; p];
    let mut htheta = vec![0.0_f64; p];
    let mut inv_lambda = vec![0.0_f64; p];
    let mut min_inv_lambda = f64::INFINITY;
    let mut trace_lambda = 0.0_f64;
    for i in 0..p {
        let lambda_i = eigenvalues[i];
        theta[i] = lambda_i * raw[i].0 * inv_p;
        htheta[i] = lambda_i * raw[i].1 * inv_p;
        let x = 1.0 / lambda_i;
        inv_lambda[i] = x;
        min_inv_lambda = min_inv_lambda.min(x);
        trace_lambda += lambda_i;
    }

    let one_minus_c = 1.0 - c;
    let mut delta_qis = vec![0.0_f64; p]; // QIS covariance eigenvalues (unscaled)
    let mut omega_lis = vec![0.0_f64; p]; // LIS precision eigenvalues (clipped)
    for i in 0..p {
        let x = inv_lambda[i];
        let th = theta[i];
        let amplitude2 = theta[i] * theta[i] + htheta[i] * htheta[i];
        let denom =
            one_minus_c * one_minus_c * x + 2.0 * c * one_minus_c * x * th + c * c * x * amplitude2;
        // Numerical safety: on a well-conditioned spectrum the denominator is
        // strictly positive; if round-off ever pushes it off, fall back to the
        // raw sample eigenvalue rather than emitting a non-finite estimate.
        delta_qis[i] = if denom > 0.0 && denom.is_finite() {
            1.0 / denom
        } else {
            eigenvalues[i]
        };

        let lis = one_minus_c * x + 2.0 * c * x * th;
        omega_lis[i] = if lis > min_inv_lambda {
            lis
        } else {
            min_inv_lambda
        };
    }

    let (covariance_eigenvalues, precision_eigenvalues) = match method {
        InverseShrinkageMethod::Qis => {
            // Preserve the trace: Σδ = Σλ.
            let sum_delta: f64 = delta_qis.iter().sum();
            let scale = if sum_delta > 0.0 && sum_delta.is_finite() {
                trace_lambda / sum_delta
            } else {
                1.0
            };
            let cov: Vec<f64> = delta_qis.iter().map(|&d| d * scale).collect();
            let prec: Vec<f64> = cov.iter().map(|&d| safe_reciprocal(d)).collect();
            (cov, prec)
        }
        InverseShrinkageMethod::Lis => {
            let cov: Vec<f64> = omega_lis.iter().map(|&d| safe_reciprocal(d)).collect();
            (cov, omega_lis)
        }
        InverseShrinkageMethod::Gis => {
            // Geometric mean of the QIS and LIS covariance estimates.
            let prec: Vec<f64> = (0..p)
                .map(|i| {
                    let ratio = omega_lis[i] / delta_qis[i];
                    if ratio > 0.0 && ratio.is_finite() {
                        ratio.sqrt()
                    } else {
                        safe_reciprocal(eigenvalues[i])
                    }
                })
                .collect();
            let cov: Vec<f64> = prec.iter().map(|&w| safe_reciprocal(w)).collect();
            (cov, prec)
        }
    };

    InverseShrinkageResult {
        precision_eigenvalues,
        covariance_eigenvalues,
        smoothing,
        theta,
        htheta,
    }
}

/// `1/x` with a guard, so a pathological input yields a finite (if extreme)
/// value instead of `inf`/`NaN` that would poison a matrix reconstruction.
#[inline]
fn safe_reciprocal(x: f64) -> f64 {
    if x > 0.0 && x.is_finite() {
        1.0 / x
    } else {
        f64::MAX
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Parallelism, StieltjesMethod};
    use approx::assert_relative_eq;

    /// Reference θ/Hθ computed straight from the definition, independent of the
    /// Stieltjes ray kernel — the two must agree to machine precision.
    fn reference_theta(lambda: &[f64], h: f64) -> (Vec<f64>, Vec<f64>) {
        let p = lambda.len();
        let x: Vec<f64> = lambda.iter().map(|&l| 1.0 / l).collect();
        let mut theta = vec![0.0; p];
        let mut htheta = vec![0.0; p];
        for i in 0..p {
            for j in 0..p {
                let diff = x[j] - x[i];
                let den = diff * diff + x[j] * x[j] * h * h;
                theta[i] += x[j] * diff / den;
                htheta[i] += x[j] * x[j] * h / den;
            }
            theta[i] /= p as f64;
            htheta[i] /= p as f64;
        }
        (theta, htheta)
    }

    fn spectrum(p: usize) -> Vec<f64> {
        (0..p).map(|i| 0.4 + (i as f64 + 1.0) * 0.05).collect()
    }

    #[test]
    fn ray_kernel_matches_definition() {
        let lambda = spectrum(61);
        let c = 0.35;
        let config = RmtConfig::new(c);
        let nls = inverse_nonlinear_shrinkage(&lambda, c, InverseShrinkageMethod::Qis, &config);
        let (theta_ref, htheta_ref) = reference_theta(&lambda, nls.smoothing);
        for i in 0..lambda.len() {
            // Absolute tolerance: θ can cross zero at the spectrum centre,
            // where a purely relative comparison is unstable.
            assert_relative_eq!(nls.theta[i], theta_ref[i], epsilon = 1e-12);
            assert_relative_eq!(nls.htheta[i], htheta_ref[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn qis_preserves_trace() {
        let lambda = spectrum(80);
        let c = 0.5;
        let nls = inverse_nonlinear_shrinkage(
            &lambda,
            c,
            InverseShrinkageMethod::Qis,
            &RmtConfig::new(c),
        );
        let sum_lambda: f64 = lambda.iter().sum();
        let sum_cov: f64 = nls.covariance_eigenvalues.iter().sum();
        assert_relative_eq!(sum_cov, sum_lambda, max_relative = 1e-12);
        // Precision eigenvalues are the reciprocals of the covariance ones.
        for (w, d) in nls
            .precision_eigenvalues
            .iter()
            .zip(nls.covariance_eigenvalues.iter())
        {
            assert_relative_eq!(*w, 1.0 / d, max_relative = 1e-12);
        }
    }

    #[test]
    fn all_methods_are_finite_and_positive() {
        let lambda = spectrum(120);
        let c = 0.6;
        for method in [
            InverseShrinkageMethod::Qis,
            InverseShrinkageMethod::Lis,
            InverseShrinkageMethod::Gis,
        ] {
            let nls = inverse_nonlinear_shrinkage(&lambda, c, method, &RmtConfig::new(c));
            for &w in &nls.precision_eigenvalues {
                assert!(w.is_finite() && w > 0.0, "{method:?} produced {w}");
            }
            for &d in &nls.covariance_eigenvalues {
                assert!(d.is_finite() && d > 0.0, "{method:?} produced {d}");
            }
        }
    }

    #[test]
    fn gis_is_geometric_mean_of_qis_and_lis() {
        let lambda = spectrum(90);
        let c = 0.4;
        let config = RmtConfig::new(c);
        let qis = inverse_nonlinear_shrinkage(&lambda, c, InverseShrinkageMethod::Qis, &config);
        let lis = inverse_nonlinear_shrinkage(&lambda, c, InverseShrinkageMethod::Lis, &config);
        let gis = inverse_nonlinear_shrinkage(&lambda, c, InverseShrinkageMethod::Gis, &config);
        // GIS reconstructs the covariance as the geometric mean of the *raw*
        // (non-trace-rescaled) QIS covariance eigenvalue and the LIS covariance
        // eigenvalue, so recompute the raw QIS diagonal from θ/Hθ here.
        let one_minus_c = 1.0 - c;
        for (i, &lambda_i) in lambda.iter().enumerate() {
            let x = 1.0 / lambda_i;
            let a = qis.theta[i] * qis.theta[i] + qis.htheta[i] * qis.htheta[i];
            let raw_qis = 1.0
                / (one_minus_c * one_minus_c * x
                    + 2.0 * c * one_minus_c * x * qis.theta[i]
                    + c * c * x * a);
            let expected = (raw_qis * lis.covariance_eigenvalues[i]).sqrt();
            assert_relative_eq!(
                gis.covariance_eigenvalues[i],
                expected,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn exact_kernels_agree() {
        let lambda = spectrum(70);
        let c = 0.45;
        let reference = inverse_nonlinear_shrinkage(
            &lambda,
            c,
            InverseShrinkageMethod::Qis,
            &RmtConfig::new(c),
        );
        for method in [
            StieltjesMethod::Naive,
            StieltjesMethod::AutoVectorized,
            StieltjesMethod::Blocked,
            StieltjesMethod::BlockedTiled,
        ] {
            for &par in Parallelism::all() {
                let config = RmtConfig::new(c)
                    .with_stieltjes(method)
                    .with_parallelism(par);
                let got =
                    inverse_nonlinear_shrinkage(&lambda, c, InverseShrinkageMethod::Qis, &config);
                for i in 0..lambda.len() {
                    assert_relative_eq!(
                        got.precision_eigenvalues[i],
                        reference.precision_eigenvalues[i],
                        max_relative = 1e-9
                    );
                }
            }
        }
    }

    #[test]
    fn qis_golden_master() {
        // Golden values produced by the reference implementation
        // (pald22/covShrinkage, QIS.py) on the deterministic spectrum
        // λ_i = 0.4 + (i+1)·0.05, p = 61, c = 0.35. Locked so an accidental
        // change to the estimator (or to the ray-kernel identity behind it)
        // cannot pass unnoticed.
        let lambda = spectrum(61);
        let c = 0.35;
        let nls = inverse_nonlinear_shrinkage(
            &lambda,
            c,
            InverseShrinkageMethod::Qis,
            &RmtConfig::new(c),
        );
        assert_relative_eq!(nls.smoothing, 0.113_758_063_061_085_6, max_relative = 1e-12);
        let expected: [(usize, f64, f64); 3] = [
            (0, -0.588_589_990_089_405_9, 0.521_551_868_512_624_1),
            (10, -0.482_925_991_176_981_6, 0.392_573_983_404_983_5),
            (60, 2.317_970_746_477_298, 0.813_377_702_386_151_3),
        ];
        for (i, theta, prec) in expected {
            assert_relative_eq!(nls.theta[i], theta, max_relative = 1e-11);
            assert_relative_eq!(nls.precision_eigenvalues[i], prec, max_relative = 1e-11);
        }
    }
}
