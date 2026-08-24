//! Focused benchmark for cache-invalidation and branch-prediction behavior
//! of the Stieltjes direct-sum kernels.
//!
//! Benchmarks the RAW kernels directly (bypassing `rie_shrinkage` overhead)
//! so we can isolate the effect of:
//!   - loop-invariant branch hoisting (`use_cutoff` out of the inner loop)
//!   - cache-blocking structure (λⱼ-outer vs output-block-outer)
//!   - software prefetching
//!
//! Run: cargo bench --bench cache_branch

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::RngExt;
use shrinkers::stieltjes::{
    compute_all_stieltjes_blocked, compute_all_stieltjes_blocked_tiled,
    compute_all_stieltjes_blocked_windowed,
};
use std::hint::black_box;

fn generate_mp_spectrum(p: usize, c: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    let lam_min = (1.0 - c.sqrt()).max(0.01).powi(2);
    let lam_max = (1.0 + c.sqrt()).powi(2);
    (0..p)
        .map(|_| lam_min + rng.random::<f64>() * (lam_max - lam_min) + rng.random::<f64>() * 0.1)
        .collect()
}

fn bench_kernels(c: &mut Criterion) {
    let conc = 0.5;

    // p=1000 (output fits L1) and p=10000 (output exceeds L1, tests cache
    // invalidation). p=50000 is omitted: at ~1s/iteration it dominates the
    // benchmark runtime for little extra signal.
    for p in [1000, 10000] {
        let mut evals = generate_mp_spectrum(p, conc);
        evals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let eta = 0.1 / (p as f64).sqrt();

        // Exact (no cutoff) — isolates cache-blocking structure.
        let mut group = c.benchmark_group(format!("p={p},exact"));
        group.sample_size(10);
        group.measurement_time(std::time::Duration::from_secs(1));

        group.bench_function("blocked_default", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| black_box(compute_all_stieltjes_blocked(&e, eta, None)),
                BatchSize::SmallInput,
            )
        });
        group.bench_function("blocked_tiled", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| black_box(compute_all_stieltjes_blocked_tiled(&e, eta, None, None)),
                BatchSize::SmallInput,
            )
        });
        group.finish();

        // With cutoff — isolates branch-prediction behavior.
        let mut group = c.benchmark_group(format!("p={p},cutoff10"));
        group.sample_size(10);
        group.measurement_time(std::time::Duration::from_secs(1));

        group.bench_function("blocked_default", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| black_box(compute_all_stieltjes_blocked(&e, eta, Some(10.0))),
                BatchSize::SmallInput,
            )
        });
        group.bench_function("blocked_tiled", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| {
                    black_box(compute_all_stieltjes_blocked_tiled(
                        &e,
                        eta,
                        None,
                        Some(10.0),
                    ))
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("blocked_windowed", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| {
                    black_box(compute_all_stieltjes_blocked_windowed(
                        &e,
                        eta,
                        Some(64),
                        Some(10.0),
                    ))
                },
                BatchSize::SmallInput,
            )
        });
        group.finish();
    }
}

criterion_group!(benches, bench_kernels);
criterion_main!(benches);
