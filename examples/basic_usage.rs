//! End-to-end tour of the public Rust API — the "how do I actually use
//! this crate?" example. Run it:
//!
//!     cargo run --release --example basic_usage
//!
//! Three stops, mirroring the README story:
//!   1. Stieltjes transform of a spectrum at a point above the bulk edge;
//!   2. RIE deconvolution: eigenvalues -> optimal shrinkage factors;
//!   3. correlation-matrix cleaning from noisy sample data.

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use shrinkers::pipeline::clean_correlation_matrix;
use shrinkers::stieltjes::compute_stieltjes_at_points;
use shrinkers::{Parallelism, RmtConfig, StieltjesMethod, rie_shrinkage};

fn main() {
    // ── A synthetic spiked sample-covariance eigenspectrum ──────────────
    // Bulk: Marchenko–Pastur-like for c = p_dim/n_samples = 0.25, plus two
    // clear spikes above the bulk edge (the "signal" we want to keep).
    let p = 400usize;
    let n = (p as f64 / 0.25) as usize; // 1600 samples
    let c_ratio = p as f64 / n as f64;
    let bulk_edge = (1.0 + c_ratio.sqrt()).powi(2);
    let mut rng = StdRng::seed_from_u64(0);
    let mut eigenvalues: Vec<f64> = (0..p)
        .map(|_| 0.2 + rng.random::<f64>() * (bulk_edge - 0.2))
        .collect();
    eigenvalues[p - 2] = 5.0; // spike 1
    eigenvalues[p - 1] = 8.0; // spike 2
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "spectrum: p={p}, c={c_ratio:.2}, spikes at {:.1} and {:.1}",
        eigenvalues[p - 2],
        eigenvalues[p - 1]
    );

    // ── 1. Stieltjes transform just above the bulk edge ─────────────────
    // NOTE: the raw kernels (and this entry point) return the UNNORMALIZED
    // sum Σ_j 1/(z - λ_j); divide by p to get the standard m(z).
    let x = bulk_edge + 1.0;
    let eta = 0.05;
    let sum = compute_stieltjes_at_points(
        &[x],
        &eigenvalues,
        eta,
        StieltjesMethod::AutoVectorized,
        None,
        Parallelism::Sequential,
        None,
    )[0];
    let (m_re, m_im) = (sum.0 / p as f64, sum.1 / p as f64);
    println!("\n[1] Stieltjes transform at x={x:.2} (eta={eta}):");
    println!("    m(x + i*eta) = {m_re:+.6} {m_im:+.6}i");

    // ── 2. RIE deconvolution: eigenvalues -> optimal shrinkage factors ──
    let config = RmtConfig::new(c_ratio);
    let factors = rie_shrinkage(&eigenvalues, &config);
    let bulk_max = factors.iter().take(p - 2).cloned().fold(f64::MIN, f64::max);
    println!("\n[2] RIE optimal-shrinkage factors:");
    println!(
        "    sample spikes 5.0 -> {:.4}, 8.0 -> {:.4}",
        factors[p - 2],
        factors[p - 1]
    );
    println!("    bulk max       : {bulk_max:.4}");
    println!("    (optimal shrinkage pulls every mode toward its population value)");

    // ── 3. Clean a correlation matrix built from noisy data ────────────
    // Draw iid standard normals X: (n x p); its sample correlation matrix
    // carries MP noise around identity — exactly what cleaning removes.
    let x_mat = Array2::from_shape_fn((n, p), |_| rng.random::<f64>() - 0.5);
    let cov = x_mat.t().dot(&x_mat) / (n as f64);
    // Normalise to unit diagonal => correlation matrix.
    let mut corr = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            corr[[i, j]] = cov[[i, j]] / (cov[[i, i]] * cov[[j, j]]).sqrt();
        }
    }

    let cleaned = clean_correlation_matrix(&corr, c_ratio, &RmtConfig::new(c_ratio));
    let mean_abs_offdiag = |mat: &Array2<f64>| {
        let s: f64 = (0..p)
            .map(|i| {
                (0..p)
                    .filter(|&j| j != i)
                    .map(|j| mat[[i, j]].abs())
                    .sum::<f64>()
            })
            .sum::<f64>();
        s / (p * (p - 1)) as f64
    };
    println!("\n[3] Correlation cleaning (mean |off-diagonal| entry):");
    println!("    sample : {:.5}", mean_abs_offdiag(&corr));
    println!("    cleaned: {:.5}", mean_abs_offdiag(&cleaned.covariance));
}
