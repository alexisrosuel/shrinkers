//! Complex Hermitian correlation-matrix cleaning.
//!
//! Everything in [`super`] assumes real symmetric input, which is what a
//! covariance or correlation matrix estimated from real observations gives.
//! A **spectral coherence matrix** does not: it is built from Fourier
//! coefficients and is complex Hermitian,
//!
//! ```text
//!     S(nu)  = (1/B) sum_b xi(nu + b/N) xi(nu + b/N)^*
//!     C(nu)  = dg(S)^{-1/2} S(nu) dg(S)^{-1/2}      (unit diagonal)
//! ```
//!
//! so it is a *complex correlation matrix* of `M` channels built from `B`
//! observations, with the same Marchenko–Pastur bulk and the same BBP spikes
//! as the real case. Only the eigensolver and the reconstruction change:
//!
//! * the eigendecomposition goes through the standard real embedding
//!   `H = A + iB  ->  [[A, -B], [B, A]]`, whose eigenvalues are each of `H`'s
//!   with multiplicity two and whose eigenvector `[u; v]` gives `u + iv`;
//! * the reconstruction is `d_bulk I + sum_t scale_t v_t v_t^H` instead of the
//!   transpose form.
//!
//! Everything else — the RIE eigenvalue shrinkage, the `sigma^2` estimate, the
//! RMT angular overlaps — is scalar and shared with the real path.

use ndarray::Array2;
use num_complex::Complex64;

use crate::config::RmtConfig;
use crate::deconvolution::{PopulationEigenvalues, estimate_population_eigenvalues, rie_shrinkage};

use super::{compute_angular_overlaps, compute_d_bulk};

/// Result of cleaning a complex Hermitian eigensystem.
#[derive(Debug, Clone)]
pub struct CleanedEigensystemComplex {
    /// Cleaned correlation matrix, shape `(p, p)`, Hermitian.
    pub covariance: Array2<Complex64>,
    /// Sample eigenvectors, columns sorted descending by eigenvalue.
    pub eigenvectors: Array2<Complex64>,
    /// RIE-cleaned eigenvalues, sorted descending.
    pub eigenvalues: Vec<f64>,
    /// Squared angular overlaps `alpha_i^2` between each sample eigenvector and
    /// its population counterpart, parallel to `eigenvalues`.
    pub overlaps: Vec<f64>,
    /// Estimated noise variance `sigma^2`.
    pub sigma2: f64,
}

/// Eigendecomposition of a Hermitian matrix `H = re + i*im`.
///
/// `re` must be symmetric and `im` antisymmetric. Returns
/// `(eigenvalues_ascending, eigenvectors_as_columns)`, matching
/// the layout of the real `symmetric_eigh`.
///
/// The real embedding duplicates every eigenvalue, so the sorted spectrum is
/// `[l1, l1, l2, l2, ...]` and taking every second entry recovers it. The
/// embedding duplicates; it does not scale.
pub fn hermitian_eigh(re: &Array2<f64>, im: &Array2<f64>) -> (Vec<f64>, Array2<Complex64>) {
    let n = re.nrows();
    assert_eq!(n, re.ncols(), "hermitian_eigh needs a square matrix");
    assert_eq!((n, n), (im.nrows(), im.ncols()));
    if n == 0 {
        return (Vec::new(), Array2::zeros((0, 0)));
    }

    let mut s = Array2::<f64>::zeros((2 * n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            let (a, b) = (re[[i, j]], im[[i, j]]);
            s[[i, j]] = a;
            s[[i, n + j]] = -b;
            s[[n + i, j]] = b;
            s[[n + i, n + j]] = a;
        }
    }
    let (doubled, vecs) = super::symmetric_eigh(&s);

    let evals: Vec<f64> = (0..n).map(|i| doubled[2 * i]).collect();
    let mut out = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        let col = 2 * i;
        for r in 0..n {
            out[[r, i]] = Complex64::new(vecs[[r, col]], vecs[[n + r, col]]);
        }
    }
    (evals, out)
}

/// Eigendecomposition of a complex Hermitian matrix held as a single
/// `Complex64` array.
///
/// The matrix is first symmetrised to its Hermitian part `(H + H^H) / 2`, so
/// round-off asymmetry in the imaginary part does not matter.  This is the
/// convenient front door to [`hermitian_eigh`] when the caller already has the
/// matrix rather than its real/imaginary halves; both matrix-level entry points
/// of this module ([`clean_correlation_matrix_complex`] and
/// [`deconvolve_correlation_matrix_complex`]) start here.
pub fn hermitian_eigh_matrix(h: &Array2<Complex64>) -> (Vec<f64>, Array2<Complex64>) {
    let p = h.nrows();
    assert_eq!(p, h.ncols(), "hermitian_eigh_matrix needs a square matrix");

    let mut re = Array2::<f64>::zeros((p, p));
    let mut im = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in i..p {
            let a = h[[i, j]];
            let b = h[[j, i]];
            // Hermitian part: (H + H^H) / 2
            re[[i, j]] = 0.5 * (a.re + b.re);
            re[[j, i]] = re[[i, j]];
            let m = 0.5 * (a.im - b.im);
            im[[i, j]] = m;
            im[[j, i]] = -m;
        }
    }
    hermitian_eigh(&re, &im)
}

fn permute_eigenvectors_complex(
    eigenvectors: &Array2<Complex64>,
    idx: &[usize],
) -> Array2<Complex64> {
    let (rows, cols) = eigenvectors.dim();
    let mut result = Array2::zeros((rows, cols));
    for (j, &i) in idx.iter().enumerate() {
        for r in 0..rows {
            result[[r, j]] = eigenvectors[[r, i]];
        }
    }
    result
}

/// `d_bulk I + sum_t scale_t v_t v_t^H`, the complex counterpart of
/// `reconstruct_covariance`.
fn reconstruct_covariance_complex(
    eigenvectors: &Array2<Complex64>,
    lambda_rie: &[f64],
    alpha2: &[f64],
) -> Array2<Complex64> {
    let p = eigenvectors.ncols();
    let d_bulk = compute_d_bulk(lambda_rie, alpha2);
    let active: Vec<usize> = (0..p).filter(|&i| alpha2[i] > 0.0).collect();
    let k = active.len();

    let mut sigma = Array2::<Complex64>::zeros((p, p));
    if k == 0 {
        for i in 0..p {
            sigma[[i, i]] = Complex64::new(d_bulk, 0.0);
        }
        return sigma;
    }

    let scales: Vec<f64> = active
        .iter()
        .map(|&i| alpha2[i] * (lambda_rie[i] - d_bulk))
        .collect();

    for r in 0..p {
        let mut diag = d_bulk;
        for (t, &i) in active.iter().enumerate() {
            diag += scales[t] * eigenvectors[[r, i]].norm_sqr();
        }
        sigma[[r, r]] = Complex64::new(diag, 0.0);

        for c in (r + 1)..p {
            let mut val = Complex64::new(0.0, 0.0);
            for (t, &i) in active.iter().enumerate() {
                val += scales[t] * eigenvectors[[r, i]] * eigenvectors[[c, i]].conj();
            }
            sigma[[r, c]] = val;
            sigma[[c, r]] = val.conj();
        }
    }
    sigma
}

/// Clean a complex Hermitian eigensystem.
///
/// Identical to [`super::clean_eigensystem`] except that the eigenvectors are
/// complex and the reconstruction uses the conjugate transpose.
pub fn clean_eigensystem_complex(
    eigenvectors: &Array2<Complex64>,
    eigenvalues: &[f64],
    c: f64,
    config: &RmtConfig,
) -> CleanedEigensystemComplex {
    let p = eigenvalues.len();

    let mut idx: Vec<usize> = (0..p).collect();
    idx.sort_unstable_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
    let sorted_evals: Vec<f64> = idx.iter().map(|&i| eigenvalues[i]).collect();
    let sorted_eigenvectors = permute_eigenvectors_complex(eigenvectors, &idx);

    let lambda_rie = rie_shrinkage(&sorted_evals, config);
    let lambda_rie_vec = lambda_rie.as_slice().unwrap().to_vec();

    let sigma2 = crate::spiked::estimate_bulk_noise(&sorted_evals, c);
    let alpha2 = compute_angular_overlaps(&lambda_rie_vec, c, sigma2);
    let covariance = reconstruct_covariance_complex(&sorted_eigenvectors, &lambda_rie_vec, &alpha2);

    CleanedEigensystemComplex {
        covariance,
        eigenvectors: sorted_eigenvectors,
        eigenvalues: lambda_rie_vec,
        overlaps: alpha2,
        sigma2,
    }
}

/// Clean a **complex correlation matrix** (a spectral coherence matrix, or any
/// Hermitian matrix with unit diagonal).
///
/// The input is symmetrised to Hermitian form first, so round-off asymmetry
/// does not matter. For very large `p` prefer [`clean_eigensystem_complex`] if
/// the eigensystem is already available from LAPACK.
pub fn clean_correlation_matrix_complex(
    correlation: &Array2<Complex64>,
    c: f64,
    config: &RmtConfig,
) -> CleanedEigensystemComplex {
    let (eigenvalues, eigenvectors) = hermitian_eigh_matrix(correlation);
    clean_eigensystem_complex(&eigenvectors, &eigenvalues, c, config)
}

/// Result of the **spiked** decomposition of a complex Hermitian correlation
/// matrix: the sample eigensystem and the population split of the spectrum.
#[derive(Debug, Clone)]
pub struct DeconvolvedCorrelationComplex {
    /// Sample eigenvalues of the input matrix, ascending.
    pub eigenvalues: Vec<f64>,
    /// Sample eigenvectors as columns, ascending, matching `eigenvalues`.
    /// Produced by the same eigendecomposition, so a caller that needs the
    /// coherent directions does not pay for a second one.
    pub eigenvectors: Array2<Complex64>,
    /// The spiked decomposition: BEMA spike detection, inverse-BBP debiasing of
    /// the spikes, and Ledoit–Wolf / RIE deconvolution of the bulk.
    pub population: PopulationEigenvalues,
}

/// Eigendecompose a **complex correlation matrix** and split its spectrum into
/// spikes and bulk.
///
/// This is [`clean_correlation_matrix_complex`]'s sibling for the *estimation*
/// problem rather than the *cleaning* one.  The two answer different questions
/// on the same matrix:
///
/// * `clean_correlation_matrix_complex` runs the RIE map on **every**
///   eigenvalue and returns a reconstructed matrix whose spike directions are
///   reweighted by their RMT angular overlap `alpha^2`;
/// * this function instead **detects** the spikes (BEMA), debiases them with
///   inverse BBP, and deconvolves only the **bulk**, so the caller gets the
///   population *spectrum* -- the number of coherent modes, their debiased
///   eigenvalues, and a bulk estimate per remaining sample eigenvalue.
///
/// Both start from the same `hermitian_eigh_matrix`, so a caller that needs the
/// matrix *and* the spectrum should use this one and then feed
/// `eigenvectors` / `eigenvalues` into
/// [`clean_eigensystem_complex`], rather than pay for two
/// eigendecompositions.
///
/// # Arguments
///
/// * `correlation` — Hermitian matrix, shape `(p, p)`. Symmetrised internally.
/// * `c` — Concentration ratio `p / n`.
/// * `margin` — Multiplicative margin above the fitted bulk edge for BEMA spike
///   detection (use `>= 1.05` unless a signal is known to be present).
/// * `config` — `RmtConfig` controlling the Stieltjes method used for the bulk
///   deconvolution.
pub fn deconvolve_correlation_matrix_complex(
    correlation: &Array2<Complex64>,
    c: f64,
    margin: f64,
    config: &RmtConfig,
) -> DeconvolvedCorrelationComplex {
    let (eigenvalues, eigenvectors) = hermitian_eigh_matrix(correlation);
    let population = estimate_population_eigenvalues(&eigenvalues, c, margin, config);

    DeconvolvedCorrelationComplex {
        eigenvalues,
        eigenvectors,
        population,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// `U diag(evals) U^H` with `U` the DFT matrix (unitary, genuinely
    /// complex, and guaranteed full rank whatever `n` is).
    fn hermite(evals: &[f64]) -> Array2<Complex64> {
        let n = evals.len();
        let mut u = Array2::<Complex64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let theta = 2.0 * std::f64::consts::PI * (i * j) as f64 / n as f64;
                u[[i, j]] = Complex64::from_polar(1.0 / (n as f64).sqrt(), theta);
            }
        }
        let mut h = Array2::<Complex64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                h[[i, j]] = (0..n)
                    .map(|k| u[[i, k]] * evals[k] * u[[j, k]].conj())
                    .sum();
            }
        }
        h
    }

    #[test]
    fn hermitian_eigh_recovers_a_known_spectrum() {
        let evals = [0.3, 2.0, 5.0, 9.0];
        let h = hermite(&evals);
        let n = 4;
        let re = Array2::from_shape_fn((n, n), |(i, j)| h[[i, j]].re);
        let im = Array2::from_shape_fn((n, n), |(i, j)| h[[i, j]].im);
        let (got, vecs) = hermitian_eigh(&re, &im);
        for (g, w) in got.iter().zip(evals.iter()) {
            assert_relative_eq!(g, w, epsilon = 1e-9);
        }
        // U^H U = I
        for i in 0..n {
            for j in 0..n {
                let dot: Complex64 = (0..n).map(|k| vecs[[k, i]].conj() * vecs[[k, j]]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(dot.re, want, epsilon = 1e-9);
                assert_relative_eq!(dot.im, 0.0, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn real_input_agrees_with_the_real_path() {
        // A real symmetric matrix embedded as complex must give the same
        // cleaned eigenvalues as `clean_correlation_matrix`.
        let p = 6;
        let raw = Array2::from_shape_fn((p, p), |(i, j)| {
            if i == j {
                1.0
            } else {
                0.3 / (1.0 + (i as f64 - j as f64).abs())
            }
        });
        let cplx = raw.mapv(|v| Complex64::new(v, 0.0));
        let c = 0.3;
        let cfg = RmtConfig::new(c);

        let real = super::super::clean_correlation_matrix(&raw, c, &cfg);
        let got = clean_correlation_matrix_complex(&cplx, c, &cfg);

        for (a, b) in real.eigenvalues.iter().zip(got.eigenvalues.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-8);
        }
        for i in 0..p {
            for j in 0..p {
                assert_relative_eq!(
                    got.covariance[[i, j]].re,
                    real.covariance[[i, j]],
                    epsilon = 1e-7
                );
                assert_relative_eq!(got.covariance[[i, j]].im, 0.0, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn hermitian_eigh_matrix_agrees_with_the_split_interface() {
        let h = hermite(&[0.5, 1.0, 3.0, 4.0]);
        let n = 4;
        let re = Array2::from_shape_fn((n, n), |(i, j)| h[[i, j]].re);
        let im = Array2::from_shape_fn((n, n), |(i, j)| h[[i, j]].im);
        let (split, _) = hermitian_eigh(&re, &im);
        let (joined, _) = hermitian_eigh_matrix(&h);
        for (a, b) in split.iter().zip(joined.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn hermitian_eigh_matrix_symmetrises_a_non_hermitian_input() {
        // Only the Hermitian part may matter; the antisymmetric part is dropped.
        let mut h = Array2::<Complex64>::eye(3);
        h[[0, 1]] = Complex64::new(0.0, 0.5);
        h[[1, 0]] = Complex64::new(0.0, 0.25); // deliberately not -0.5i
        let (evals, _) = hermitian_eigh_matrix(&h);

        // (H + H^H)/2 leaves 0.125i on [0,1] and -0.125i on [1,0], so the
        // spectrum is {1, 1 - 0.125, 1 + 0.125}.
        assert_relative_eq!(evals[0], 0.875, epsilon = 1e-12);
        assert_relative_eq!(evals[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(evals[2], 1.125, epsilon = 1e-12);
    }

    #[test]
    fn deconvolve_correlation_matrix_complex_matches_the_eigenvalue_entry_point() {
        // The matrix entry point must return exactly what the eigenvalue-only
        // entry point returns on the same spectrum -- it is the same pipeline
        // with the eigendecomposition folded in.
        let p = 7;
        let h = hermite(&[0.4, 0.6, 0.9, 1.0, 1.1, 2.5, 6.0]);
        let (c, margin) = (0.4, 1.05);
        let cfg = RmtConfig::new(c);

        let got = deconvolve_correlation_matrix_complex(&h, c, margin, &cfg);
        let want = estimate_population_eigenvalues(&got.eigenvalues, c, margin, &cfg);

        assert_eq!(got.population.k, want.k);
        for (a, b) in got.population.spikes.iter().zip(want.spikes.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
        for (a, b) in got
            .population
            .bulk_population
            .iter()
            .zip(want.bulk_population.iter())
        {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
        assert_relative_eq!(got.population.sigma2, want.sigma2, epsilon = 1e-12);
        assert_relative_eq!(got.population.bulk_edge, want.bulk_edge, epsilon = 1e-12);

        // The eigenvectors come free and are unitary.
        for i in 0..p {
            let norm: f64 = (0..p).map(|k| got.eigenvectors[[k, i]].norm_sqr()).sum();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn cleaning_an_identity_returns_the_identity() {
        let p = 5;
        let ident = Array2::<Complex64>::eye(p);
        let res = clean_correlation_matrix_complex(&ident, 0.4, &RmtConfig::new(0.4));
        for i in 0..p {
            for j in 0..p {
                let want = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(res.covariance[[i, j]].re, want, epsilon = 1e-8);
                assert_relative_eq!(res.covariance[[i, j]].im, 0.0, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn cleaned_matrix_is_hermitian_and_positive_definite() {
        let p = 8;
        // A Hermitian correlation matrix with a genuine spike.
        let mut h = Array2::<Complex64>::eye(p);
        let v: Vec<Complex64> = (0..p)
            .map(|i| Complex64::from_polar(1.0, 0.7 * i as f64))
            .collect();
        let norm: f64 = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        for i in 0..p {
            for j in 0..p {
                h[[i, j]] += (0.25 / (norm * norm)) * v[i] * v[j].conj() * (p as f64);
            }
        }
        let res = clean_correlation_matrix_complex(&h, 0.3, &RmtConfig::new(0.3));
        for i in 0..p {
            for j in 0..p {
                let a = res.covariance[[i, j]];
                let b = res.covariance[[j, i]];
                assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
                assert_relative_eq!(a.im, -b.im, epsilon = 1e-10);
            }
        }
        // Positive spectrum
        let re = Array2::from_shape_fn((p, p), |(i, j)| res.covariance[[i, j]].re);
        let im = Array2::from_shape_fn((p, p), |(i, j)| res.covariance[[i, j]].im);
        let (evals, _) = hermitian_eigh(&re, &im);
        for e in evals {
            assert!(e > -1e-9, "non-positive eigenvalue {e}");
        }
    }
}
