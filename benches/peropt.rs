//! Compare each independent hardware optimization.
//!
//! Tests the Blocked method with each optimization toggled ON/OFF.
//! This is only possible from Rust, not from Python.

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

fn bench_optimizations(c: &mut Criterion) {
    let p = 1000;
    let conc = 0.5;
    let evals = generate_mp_spectrum(p, conc);

    // Test each optimization independently with Blocked method
    let configs: Vec<(&str, RmtConfig)> = vec![
        (
            "baseline_autovec",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::AutoVectorized)
                .with_parallelism(Parallelism::Sequential),
        ),
        (
            "blocked_default",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::Blocked)
                .with_parallelism(Parallelism::Sequential),
        ),
        (
            "blocked_autovec",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::BlockedAutoVec)
                .with_parallelism(Parallelism::Sequential),
        ),
        (
            "blocked_tiled",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::BlockedTiled)
                .with_parallelism(Parallelism::Sequential),
        ),
        (
            "blocked_windowed",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::BlockedWindowed)
                .with_cutoff(CutoffConfig::Enabled { ratio: 10.0 })
                .with_parallelism(Parallelism::Sequential),
        ),
        (
            "adaptive",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::Adaptive)
                .with_cutoff(CutoffConfig::Enabled { ratio: 10.0 })
                .with_parallelism(Parallelism::Sequential),
        ),
        (
            "block32",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::Blocked)
                .with_block_size(32)
                .with_parallelism(Parallelism::Sequential),
        ),
        (
            "block128",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::Blocked)
                .with_block_size(128)
                .with_parallelism(Parallelism::Sequential),
        ),
        (
            "cutoff_8",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::Blocked)
                .with_cutoff(CutoffConfig::Enabled { ratio: 8.0 })
                .with_parallelism(Parallelism::Sequential),
        ),
        (
            "cutoff_15",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::Blocked)
                .with_cutoff(CutoffConfig::Enabled { ratio: 15.0 })
                .with_parallelism(Parallelism::Sequential),
        ),
        (
            "blocked_rayon",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::Blocked)
                .with_parallelism(Parallelism::Rayon),
        ),
        (
            "fft5",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::Fft5)
                .with_parallelism(Parallelism::Sequential),
        ),
        (
            "fft3",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::Fft3)
                .with_parallelism(Parallelism::Sequential),
        ),
        (
            "fft2",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::Fft2)
                .with_parallelism(Parallelism::Sequential),
        ),
        (
            "treecode",
            RmtConfig::new(conc)
                .with_stieltjes(StieltjesMethod::TreeCode)
                .with_parallelism(Parallelism::Sequential),
        ),
    ];

    for (name, config) in &configs {
        let mut group = c.benchmark_group(name.to_string());
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

criterion_group!(benches, bench_optimizations);
criterion_main!(benches);
