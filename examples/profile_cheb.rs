//! Profile the sequential Chebyshev treecode vs the multipole treecode.
//! Run: cargo run --release --example profile_cheb
use shrinkers::config::{CutoffConfig, Parallelism, StieltjesMethod};
use shrinkers::stieltjes::compute_all_stieltjes;
use std::time::Instant;

fn mp_spectrum(p: usize, c: f64) -> Vec<f64> {
    let lam_min = (1.0 - c.sqrt()).max(0.01).powi(2);
    let lam_max = (1.0 + c.sqrt()).powi(2);
    (0..p)
        .map(|i| {
            let t = i as f64 / p as f64;
            lam_min + t * (lam_max - lam_min) + 0.05
        })
        .collect()
}

fn main() {
    for p in [1000, 5000, 20000] {
        let mut evals = mp_spectrum(p, 0.5);
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();
        let cutoff = CutoffConfig::Disabled;

        // Warm up.
        compute_all_stieltjes(
            &evals,
            eta,
            StieltjesMethod::ChebCode,
            None,
            cutoff,
            0,
            Parallelism::Sequential,
        );

        let t0 = Instant::now();
        let _ = compute_all_stieltjes(
            &evals,
            eta,
            StieltjesMethod::ChebCode,
            None,
            cutoff,
            0,
            Parallelism::Sequential,
        );
        let t_cheb = t0.elapsed();

        let t1 = Instant::now();
        let _ = compute_all_stieltjes(
            &evals,
            eta,
            StieltjesMethod::TreeCode,
            None,
            cutoff,
            0,
            Parallelism::Sequential,
        );
        let t_tree = t1.elapsed();

        println!(
            "p={p}: chebcode_seq {:.3}ms  treecode_seq {:.3}ms  ratio {:.2}x",
            t_cheb.as_secs_f64() * 1e3,
            t_tree.as_secs_f64() * 1e3,
            t_tree.as_secs_f64() / t_cheb.as_secs_f64()
        );
    }
}
