//! Benchmark the large-p approximate Stieltjes methods:
//! Ewald (near/far splitting) vs Fft2 (grid FFT) vs TreeCode (FMM) vs Blocked (exact).
//!
//! These are the O(p log p) / O(p·k) methods that matter at large p where the
//! exact O(p²) Blocked method becomes prohibitive. Ewald is the new method
//! (smooth far kernel → coarse grid), so we compare it head-to-head against
//! the established Fft2 and TreeCode.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::RngExt;
use shrinkers::*;
use std::hint::black_box;

/// Generate a Marchenko-Pastur-like eigenvalue spectrum.
fn generate_mp_spectrum(p: usize, c: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    let lam_min = (1.0 - c.sqrt()).max(0.01).powi(2);
    let lam_max = (1.0 + c.sqrt()).powi(2);
    (0..p)
        .map(|_| lam_min + rng.random::<f64>() * (lam_max - lam_min) + rng.random::<f64>() * 0.1)
        .collect()
}

fn bench_large_p_approx(c: &mut Criterion) {
    let conc = 0.5;

    // Large p where O(p²) Blocked is prohibitive and the approximate methods
    // compete. TreeCode is parallel-friendly; Fft2/Ewald are sequential.
    for p in [5000, 20000, 100000] {
        let evals = generate_mp_spectrum(p, conc);

        let configs: Vec<(&str, RmtConfig)> = vec![
            (
                "blocked",
                RmtConfig::new(conc)
                    .with_stieltjes(StieltjesMethod::Blocked)
                    .with_parallelism(Parallelism::Sequential),
            ),
            (
                "fft2",
                RmtConfig::new(conc)
                    .with_stieltjes(StieltjesMethod::Fft2)
                    .with_parallelism(Parallelism::Sequential),
            ),
            (
                "ewald",
                RmtConfig::new(conc)
                    .with_stieltjes(StieltjesMethod::Ewald)
                    .with_parallelism(Parallelism::Sequential),
            ),
            (
                "treecode",
                RmtConfig::new(conc)
                    .with_stieltjes(StieltjesMethod::TreeCode)
                    .with_parallelism(Parallelism::Sequential),
            ),
            (
                "treecode_par",
                RmtConfig::new(conc)
                    .with_stieltjes(StieltjesMethod::TreeCode)
                    .with_parallelism(Parallelism::Rayon),
            ),
        ];

        for (name, config) in &configs {
            let bench_name = format!("p={p},{}", name);
            let mut group = c.benchmark_group(&bench_name);
            group.sample_size(10);

            group.bench_function("run", |b| {
                b.iter_batched(
                    || evals.clone(),
                    |e| rie_shrinkage(black_box(&e), black_box(config)),
                    BatchSize::SmallInput,
                )
            });

            group.finish();
        }
    }
}

criterion_group!(benches, bench_large_p_approx);
criterion_main!(benches);
