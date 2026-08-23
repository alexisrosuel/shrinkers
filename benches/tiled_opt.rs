//! Focused benchmark for optimizing the blocked_tiled Stieltjes kernel.
//!
//! Benchmarks the RAW kernel directly (bypassing rie_shrinkage overhead) so
//! we can isolate and iterate on the inner-loop constant factor.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::RngExt;
use shrinkers::stieltjes::compute_all_stieltjes_blocked_tiled;
use std::hint::black_box;

fn generate_mp_spectrum(p: usize, c: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    let lam_min = (1.0 - c.sqrt()).max(0.01).powi(2);
    let lam_max = (1.0 + c.sqrt()).powi(2);
    (0..p)
        .map(|_| lam_min + rng.random::<f64>() * (lam_max - lam_min) + rng.random::<f64>() * 0.1)
        .collect()
}

fn bench_tiled(c: &mut Criterion) {
    let conc = 0.5;

    for p in [1000, 10000] {
        let mut evals = generate_mp_spectrum(p, conc);
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        let mut group = c.benchmark_group(format!("p={p},tiled_opt"));

        // Default auto block size (None) is the real hot path.
        group.bench_function("tiled_auto", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| black_box(compute_all_stieltjes_blocked_tiled(&e, eta, None, None)),
                BatchSize::SmallInput,
            )
        });

        // Fixed block sizes for sensitivity.
        for bs in [4usize, 8, 16, 32, 64, 128] {
            group.bench_function(format!("tiled_bs{bs}"), |b| {
                b.iter_batched(
                    || evals.clone(),
                    |e| black_box(compute_all_stieltjes_blocked_tiled(&e, eta, Some(bs), None)),
                    BatchSize::SmallInput,
                )
            });
        }

        group.finish();
    }
}

criterion_group!(benches, bench_tiled);
criterion_main!(benches);
