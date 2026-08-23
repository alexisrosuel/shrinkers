//! Benchmark the cache-tiling optimization at LARGE p where the output array
//! exceeds cache size. This is where cache invalidation matters most.
//!
//! Compares blocked_default (λⱼ-outer) vs blocked_tiled (output-block outer).

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::RngExt;
use shrinkers::*;
use std::hint::black_box;

fn generate_mp_spectrum(p: usize, c: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    let lam_min = (1.0 - c.sqrt()).max(0.01).powi(2);
    let lam_max = (1.0 + c.sqrt()).powi(2);
    (0..p)
        .map(|_| lam_min + rng.random::<f64>() * (lam_max - lam_min) + rng.random::<f64>() * 0.1)
        .collect()
}

fn bench_cache_tiling(c: &mut Criterion) {
    let conc = 0.5;

    // Test at multiple sizes to show where cache effects kick in.
    // Output array = 2 * p * 8 bytes. L2 is ~4MB on M-series.
    // p=1000: 16KB (fits L1) | p=10000: 160KB (fits L2) | p=50000: 800KB
    for p in [1000, 10000] {
        let evals = generate_mp_spectrum(p, conc);

        let configs: Vec<(&str, RmtConfig)> = vec![
            (
                "blocked_default",
                RmtConfig::new(conc)
                    .with_stieltjes(StieltjesMethod::Blocked)
                    .with_parallelism(Parallelism::Sequential),
            ),
            (
                "blocked_tiled",
                RmtConfig::new(conc)
                    .with_stieltjes(StieltjesMethod::BlockedTiled)
                    .with_parallelism(Parallelism::Sequential),
            ),
            (
                "blocked_tiled_bs32",
                RmtConfig::new(conc)
                    .with_stieltjes(StieltjesMethod::BlockedTiled)
                    .with_block_size(32)
                    .with_parallelism(Parallelism::Sequential),
            ),
            (
                "blocked_tiled_bs16",
                RmtConfig::new(conc)
                    .with_stieltjes(StieltjesMethod::BlockedTiled)
                    .with_block_size(16)
                    .with_parallelism(Parallelism::Sequential),
            ),
            (
                "blocked_tiled_bs8",
                RmtConfig::new(conc)
                    .with_stieltjes(StieltjesMethod::BlockedTiled)
                    .with_block_size(8)
                    .with_parallelism(Parallelism::Sequential),
            ),
            (
                "blocked_tiled_bs4",
                RmtConfig::new(conc)
                    .with_stieltjes(StieltjesMethod::BlockedTiled)
                    .with_block_size(4)
                    .with_parallelism(Parallelism::Sequential),
            ),
            (
                "blocked_tiled_bs128",
                RmtConfig::new(conc)
                    .with_stieltjes(StieltjesMethod::BlockedTiled)
                    .with_block_size(128)
                    .with_parallelism(Parallelism::Sequential),
            ),
        ];

        for (name, config) in &configs {
            let bench_name = format!("p={p},{}", name);
            let mut group = c.benchmark_group(&bench_name);
            group.sample_size(10);
            group.measurement_time(std::time::Duration::from_secs(5));
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

criterion_group!(benches, bench_cache_tiling);
criterion_main!(benches);
