//! RAW tiled-kernel micro-benchmark — bypasses `rie_shrinkage` entirely to
//! expose the inner-loop constant factor and the block-size landscape.
//!
//! This is the benchmark to run while touching
//! `src/stieltjes/cacheblock.rs`; everything pipeline-level is noise here.

mod support;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use shrinkers::stieltjes::compute_all_stieltjes_blocked_tiled;
use std::hint::black_box;
use support::mp_spectrum;

fn bench_tiled_blocksize(c: &mut Criterion) {
    let c_ratio = 0.5;

    for p in [1000usize, 10_000] {
        let evals = mp_spectrum(p, c_ratio, p as u64);
        let eta = 0.1 / (p as f64).sqrt();

        let mut group = c.benchmark_group(format!("kernel/tiled/p={p}"));

        // Auto block size is the shipped default.
        group.bench_function("auto", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| black_box(compute_all_stieltjes_blocked_tiled(&e, eta, None, None)),
                BatchSize::SmallInput,
            )
        });

        // Fixed block sizes for sensitivity.
        for bs in [4usize, 8, 16, 32, 64, 128] {
            group.bench_function(format!("bs{bs}"), |b| {
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

criterion_group!(benches, bench_tiled_blocksize);
criterion_main!(benches);
