//! Full covariance cleaning pipeline: RIE eigenvalue shrinkage + eigenvector shrinkage.
//!
//! This module combines:
//! 1. RIE eigenvalue shrinkage (from [`crate::shrinkage`])
//! 2. Noise variance estimation + angular overlaps (from [`crate::eigenvector_overlaps`])
//! 3. Covariance matrix reconstruction (implemented here)
//!
//! # Usage
//!
//! The pipeline takes the **eigensystem** (eigenvectors + eigenvalues of the sample
//! covariance matrix) as input. The spectral decomposition is expected to be done
//! on the Python/NumPy side via LAPACK (scipy/numpy), which is already optimal.
//! Rust takes it from there for the RIE shrinkage, angular overlap correction, and
//! covariance reconstruction.
//!
//! For the truly all-Rust path (no Python at all), the pipeline can also accept
//! the data matrix directly and use a simple symmetric eigendecomposition — but
//! for maximum performance, LAPACK's `dsyevd` is significantly faster.

use crate::config::RmtConfig;
use crate::deconvolution::rie_shrinkage;
use crate::eigenvector_overlaps::compute_angular_overlaps;
use ndarray::Array2;

// ──────────────────────────────────────────────
//  Noise variance estimation (median of sample eigenvalues)
// ──────────────────────────────────────────────

/// Estimate the noise variance σ² from the sample eigenvalue spectrum.
///
/// Uses the **raw median** of the sample eigenvalues — kept for
/// compatibility. Note this is *uncorrected*: the Marchenko–Pastur law is
/// right-skewed, so the raw median overestimates σ². The cleaning pipeline
/// ([`clean_eigensystem`]) instead uses the MP-median-corrected estimator
/// [`crate::spiked::estimate_bulk_noise`].
pub fn estimate_noise_variance(sample_evals: &[f64]) -> f64 {
    median(sample_evals)
}

/// Compute the median of a slice of f64 values.
fn median(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

// ──────────────────────────────────────────────
//  Covariance reconstruction from eigenvectors + overlaps
// ──────────────────────────────────────────────

/// Reconstruct the cleaned covariance matrix from eigenvectors, RIE eigenvalues,
/// and angular overlaps.
///
/// # Arguments
///
/// * `V` — Sample eigenvectors, shape (p, p), columns = eigenvectors, sorted
///   descending by eigenvalue.
/// * `lambda_rie` — RIE-cleaned eigenvalues (sorted descending), length p.
/// * `alpha2` — Squared angular overlaps, length p.
///
/// # Returns
///
/// Cleaned covariance matrix, shape (p, p), symmetric & positive definite.
pub fn reconstruct_covariance(
    eigenvectors: &Array2<f64>,
    lambda_rie: &[f64],
    alpha2: &[f64],
) -> Array2<f64> {
    let p = eigenvectors.ncols();
    debug_assert_eq!(eigenvectors.nrows(), p);
    debug_assert_eq!(lambda_rie.len(), p);
    debug_assert_eq!(alpha2.len(), p);

    let d_bulk = compute_d_bulk(lambda_rie, alpha2);

    // Count active spikes
    let k = alpha2.iter().filter(|&&a| a > 0.0).count();

    // Fast path: no spikes
    if k == 0 {
        let mut sigma = vec![0.0_f64; p * p];
        for i in 0..p {
            sigma[i * p + i] = d_bulk;
        }
        return Array2::from_shape_vec((p, p), sigma).unwrap();
    }

    // ── Pack active columns: V_packed[p][k] (row-major, stride-1 over k) ──
    // v_packed[r * k + t] = V[r, active[t]]
    let mut v_packed = vec![0.0_f64; p * k];
    let mut scales = Vec::with_capacity(k);
    let mut t = 0;
    for i in 0..p {
        let a2 = alpha2[i];
        if a2 > 0.0 {
            scales.push(a2 * (lambda_rie[i] - d_bulk));
            for r in 0..p {
                v_packed[r * k + t] = eigenvectors[[r, i]];
            }
            t += 1;
        }
    }

    // ── Sequential SYRK ──
    // Each row is computed independently: Σ[r,c] = d_bulk·δ(r,c) + Σ ₜ scales[t]·V[r,t]·V[c,t]
    let mut sigma = vec![0.0_f64; p * p];

    for r in 0..p {
        let row_r_base = r * k;

        let mut diag = d_bulk;
        for t in 0..k {
            let vr = v_packed[row_r_base + t];
            diag += scales[t] * vr * vr;
        }
        sigma[r * p + r] = diag;

        for c in (r + 1)..p {
            let row_c_base = c * k;
            let mut val = 0.0;
            for t in 0..k {
                val += scales[t] * v_packed[row_r_base + t] * v_packed[row_c_base + t];
            }
            sigma[r * p + c] = val;
            sigma[c * p + r] = val;
        }
    }

    Array2::from_shape_vec((p, p), sigma).unwrap()
}

/// Compute d_bulk: average of RIE eigenvalues for which α² = 0.
fn compute_d_bulk(lambda_rie: &[f64], alpha2: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..lambda_rie.len() {
        if alpha2[i] == 0.0 {
            sum += lambda_rie[i];
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        lambda_rie.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

// ──────────────────────────────────────────────
//  Pipeline: from eigensystem to cleaned covariance
// ──────────────────────────────────────────────

/// Clean covariance matrix from an already-computed eigensystem.
///
/// This is the main entry point: given the sample eigenvectors and eigenvalues,
/// computes the RIE shrinkage + eigenvector angular overlap correction and
/// returns a cleaned covariance matrix.
///
/// # Arguments
///
/// * `eigenvectors` — Sample eigenvectors, shape (p, p), columns = eigenvectors.
///   Will be sorted descending by eigenvalue internally.
/// * `eigenvalues` — Sample eigenvalues (length p), parallel to `eigenvectors` columns.
/// * `c` — Concentration ratio N / T.
/// * `config` — `RmtConfig` controlling RIE shrinkage method.
///
/// # Returns
///
/// Cleaned covariance matrix, shape (p, p), symmetric & positive definite.
pub fn clean_covariance_from_eigensystem(
    eigenvectors: &Array2<f64>,
    eigenvalues: &[f64],
    c: f64,
    config: &RmtConfig,
) -> Array2<f64> {
    clean_eigensystem(eigenvectors, eigenvalues, c, config).covariance
}

/// Result of cleaning an eigensystem: the cleaned covariance matrix plus the
/// cleaned eigenvalues, the (sorted) eigenvectors, and their theoretical
/// alignment with the population eigenvectors.
#[derive(Debug, Clone)]
pub struct CleanedEigensystem {
    /// Cleaned covariance matrix, shape (p, p), symmetric & positive definite.
    pub covariance: Array2<f64>,
    /// Sample eigenvectors, columns sorted descending by eigenvalue.
    pub eigenvectors: Array2<f64>,
    /// RIE-cleaned eigenvalues, sorted descending.
    pub eigenvalues: Vec<f64>,
    /// Squared angular overlaps α_i² = cos²(θ_i) between each sample
    /// eigenvector and its population counterpart (parallel to `eigenvalues`).
    pub overlaps: Vec<f64>,
    /// Estimated noise variance σ², MP-median-corrected
    /// (see [`crate::spiked::estimate_bulk_noise`]).
    pub sigma2: f64,
}

/// Clean an eigensystem and return the cleaned covariance matrix together with
/// the cleaned eigenvalues, sorted eigenvectors, and their theoretical
/// alignment with the population eigenvectors.
///
/// # Arguments
///
/// * `eigenvectors` — Sample eigenvectors, shape (p, p), columns = eigenvectors.
///   Will be sorted descending by eigenvalue internally.
/// * `eigenvalues` — Sample eigenvalues (length p), parallel to `eigenvectors` columns.
/// * `c` — Concentration ratio N / T.
/// * `config` — `RmtConfig` controlling RIE shrinkage method.
///
/// # Returns
///
/// A [`CleanedEigensystem`] with the cleaned covariance, eigenvalues,
/// eigenvectors, and angular overlaps.
pub fn clean_eigensystem(
    eigenvectors: &Array2<f64>,
    eigenvalues: &[f64],
    c: f64,
    config: &RmtConfig,
) -> CleanedEigensystem {
    let p = eigenvalues.len();

    // ── 1. Sort descending ──
    let mut idx: Vec<usize> = (0..p).collect();
    idx.sort_unstable_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    let sorted_evals: Vec<f64> = idx.iter().map(|&i| eigenvalues[i]).collect();

    // Permute columns of eigenvectors
    let sorted_eigenvectors = permute_eigenvectors(eigenvectors, &idx);

    // ── 2. RIE eigenvalue shrinkage (pointwise deconvolution) ──
    let lambda_rie = rie_shrinkage(&sorted_evals, config);
    let lambda_rie_vec = lambda_rie.as_slice().unwrap().to_vec();

    // ── 3. Noise variance estimation ──
    // MP-median-corrected estimator: the raw median overestimates σ² because
    // the Marchenko–Pastur law is right-skewed; dividing by the fitted MP
    // median factor removes that bias (see crate::spiked::estimate_bulk_noise).
    let sigma2 = crate::spiked::estimate_bulk_noise(&sorted_evals, c);

    // ── 4. Angular overlaps ──
    let alpha2 = compute_angular_overlaps(&lambda_rie_vec, c, sigma2);

    // ── 5. Covariance reconstruction ──
    let covariance = reconstruct_covariance(&sorted_eigenvectors, &lambda_rie_vec, &alpha2);

    CleanedEigensystem {
        covariance,
        eigenvectors: sorted_eigenvectors,
        eigenvalues: lambda_rie_vec,
        overlaps: alpha2,
        sigma2,
    }
}

/// Clean covariance matrix from a data matrix X (T, N).
///
/// Computes the empirical covariance, spectral decomposition, then cleans
/// via RIE + eigenvector shrinkage. Uses LAPACK-style symmetric eigen-
/// decomposition for the spectral step.
///
/// NOTE: This function allocates the p×p empirical covariance matrix.
/// For very large p, prefer `clean_covariance_from_eigensystem` if you
/// already have the eigensystem from Python/scipy.
///
/// # Arguments
///
/// * `X` — Data matrix, shape (T, N), should be mean-centered.
/// * `c` — Concentration ratio N/T. If None, computed as N/T.
/// * `config` — `RmtConfig` controlling RIE shrinkage method.
///
/// # Returns
///
/// Cleaned covariance matrix, shape (N, N).
pub fn clean_covariance_from_data(
    data: &Array2<f64>,
    c: Option<f64>,
    config: &RmtConfig,
) -> Array2<f64> {
    let (t, n) = data.dim();
    let c = c.unwrap_or_else(|| n as f64 / t as f64);

    // ── 1. Empirical covariance Σ = X^T X / T ──
    let sigma_emp = data.t().dot(data) / t as f64;

    // ── 2. Spectral decomposition ──
    let (eigenvalues, eigenvectors) = symmetric_eigh(&sigma_emp);

    // ── 3. Clean ──
    clean_covariance_from_eigensystem(&eigenvectors, &eigenvalues, c, config)
}

/// Clean a correlation matrix and return the cleaned covariance matrix plus
/// the cleaned eigenvalues, sorted eigenvectors, and their theoretical
/// alignment with the population eigenvectors.
///
/// The spectral decomposition is computed internally with a symmetric
/// eigendecomposition (Jacobi iteration). For very large p, prefer
/// `clean_eigensystem` if you already have the eigensystem from Python/scipy.
///
/// # Arguments
///
/// * `correlation` — Sample correlation matrix, shape (p, p), symmetric.
/// * `c` — Concentration ratio N / T.
/// * `config` — `RmtConfig` controlling RIE shrinkage method.
///
/// # Returns
///
/// A [`CleanedEigensystem`] with the cleaned covariance, eigenvalues,
/// eigenvectors, and angular overlaps.
pub fn clean_correlation_matrix(
    correlation: &Array2<f64>,
    c: f64,
    config: &RmtConfig,
) -> CleanedEigensystem {
    // ── 1. Spectral decomposition ──
    let (eigenvalues, eigenvectors) = symmetric_eigh(correlation);

    // ── 2. Clean ──
    clean_eigensystem(&eigenvectors, &eigenvalues, c, config)
}

// ──────────────────────────────────────────────
//  Helper: permute eigenvectors by column index
// ──────────────────────────────────────────────

/// Permute columns of a matrix according to index ordering.
fn permute_eigenvectors(eigenvectors: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    let (rows, cols) = eigenvectors.dim();
    let mut result = Array2::zeros((rows, cols));
    for (j, &i) in idx.iter().enumerate() {
        for r in 0..rows {
            result[[r, j]] = eigenvectors[[r, i]];
        }
    }
    result
}

// ──────────────────────────────────────────────
//  Helper: symmetric eigendecomposition (Jacobi iteration)
// ──────────────────────────────────────────────

/// Simple symmetric eigendecomposition using Jacobi iteration.
///
/// This is a fallback for when no LAPACK library is available.
/// For production use with p > 200, prefer LAPACK (via ndarray-linalg or Python/scipy).
///
/// Returns (eigenvalues_ascending, eigenvectors) where columns of eigenvectors
/// are the eigenvectors.
fn symmetric_eigh(matrix: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let p = matrix.nrows();
    let mut a = matrix.to_owned();
    let mut eigvec_matrix = Array2::eye(p);

    let eps = 1e-15;
    let max_sweeps = 100;

    for _sweep in 0..max_sweeps {
        // Find the largest off-diagonal element.
        let mut max_off = 0.0;
        for i in 0..p {
            for j in (i + 1)..p {
                let aij = a[[i, j]].abs();
                if aij > max_off {
                    max_off = aij;
                }
            }
        }

        if max_off < eps {
            break;
        }

        // Threshold Jacobi: in each sweep, rotate every off-diagonal element
        // whose magnitude is at least 10% of the current maximum. This lets
        // many rotations happen per sweep (fast convergence).
        let threshold = max_off * 0.1;
        for i in 0..p {
            for j in (i + 1)..p {
                let aij = a[[i, j]];
                if aij.abs() < threshold {
                    continue;
                }

                // Numerical Recipes Jacobi rotation (Golub & Van Loan).
                let diff = a[[j, j]] - a[[i, i]];
                let theta = 0.5 * diff / aij;
                let t = 1.0 / (theta.abs() + (1.0 + theta * theta).sqrt());
                let t = if theta < 0.0 { -t } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);
                let h = t * aij;

                a[[i, i]] -= h;
                a[[j, j]] += h;
                a[[i, j]] = 0.0;
                a[[j, i]] = 0.0;

                for k in 0..p {
                    if k != i && k != j {
                        let g = a[[k, i]];
                        let hh = a[[k, j]];
                        a[[k, i]] = g - s * (hh + g * tau);
                        a[[k, j]] = hh + s * (g - hh * tau);
                        a[[i, k]] = a[[k, i]];
                        a[[j, k]] = a[[k, j]];
                    }
                }

                for k in 0..p {
                    let g = eigvec_matrix[[k, i]];
                    let hh = eigvec_matrix[[k, j]];
                    eigvec_matrix[[k, i]] = g - s * (hh + g * tau);
                    eigvec_matrix[[k, j]] = hh + s * (g - hh * tau);
                }
            }
        }
    }

    let mut eigenvalues: Vec<f64> = (0..p).map(|i| a[[i, i]]).collect();
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_unstable_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());
    eigenvalues.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let sorted_eigenvectors = permute_eigenvectors(&eigvec_matrix, &order);

    (eigenvalues, sorted_eigenvectors)
}

// ──────────────────────────────────────────────
//  Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_permute_eigenvectors() {
        let v = Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
        let idx = vec![2, 1, 0];
        let result = permute_eigenvectors(&v, &idx);
        assert_eq!(result.column(0)[2], 1.0);
        assert_eq!(result.column(2)[0], 1.0);
    }

    #[test]
    fn test_symmetric_eigh_identity() {
        let mat = Array2::eye(4);
        let (evals, evecs) = symmetric_eigh(&mat);
        for &v in &evals {
            assert_relative_eq!(v, 1.0, epsilon = 1e-10);
        }
        let vtv = evecs.t().dot(&evecs);
        for i in 0..4 {
            assert_relative_eq!(vtv[[i, i]], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_symmetric_eigh_diagonal() {
        let vals = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let mat = Array2::from_diag(&ndarray::Array1::from_vec(vals.clone()));
        let (evals, _evecs) = symmetric_eigh(&mat);
        for (i, &val) in evals.iter().enumerate() {
            assert_relative_eq!(val, (i + 1) as f64, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_clean_covariance_from_eigensystem_trivial() {
        let p = 3;
        let v = Array2::eye(p);
        let evals = vec![3.0, 2.0, 1.0];
        let config = crate::config::RmtConfig::new(0.3);
        let sigma = clean_covariance_from_eigensystem(&v, &evals, 0.3, &config);
        for i in 0..p {
            for j in 0..p {
                if i == j {
                    assert!(sigma[[i, j]] > 0.0);
                } else {
                    assert_relative_eq!(sigma[[i, j]], 0.0, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_symmetric_eigh_spiked() {
        // A random symmetric matrix plus a strong rank-1 spike.
        let p = 20;
        let mut rng = rand::rng();
        use rand::RngExt;
        let mut a = Array2::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                a[[i, j]] = rng.random::<f64>() * 2.0 - 1.0;
            }
        }
        // Symmetrize
        for i in 0..p {
            for j in (i + 1)..p {
                let avg = (a[[i, j]] + a[[j, i]]) / 2.0;
                a[[i, j]] = avg;
                a[[j, i]] = avg;
            }
        }
        // Add a strong rank-1 spike
        let mut v: Vec<f64> = (0..p).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in v.iter_mut() {
            *x /= norm;
        }
        for i in 0..p {
            for j in 0..p {
                a[[i, j]] += 20.0 * v[i] * v[j];
            }
        }
        let (evals, _evecs) = symmetric_eigh(&a);
        // Largest eigenvalue should be ~20 + O(1)
        assert!(evals[p - 1] > 15.0, "largest eval = {}", evals[p - 1]);
        // Trace preserved
        let trace: f64 = evals.iter().sum();
        let orig_trace: f64 = (0..p).map(|i| a[[i, i]]).sum();
        assert!((trace - orig_trace).abs() < 1e-8);
    }
}
