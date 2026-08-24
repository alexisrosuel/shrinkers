//! Cache-scaling behaviour of the exact family: the row-major `Blocked`
//! sweep versus the output-block-outer `BlockedTiled` tiling, at sizes where
//! the output arrays stop fitting in cache (output = 2·p·8 bytes; L2 is
//! ~4 MB on M-series). Full `rie_shrinkage` entry point.

mod support;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use shrinkers::*;
use std::hint::black_box;
use support::mp_spectrum;

fn bench_cache_scaling(c: &mut Criterion) {
    let c_ratio = 0.5;
    // p=1000: 16 KB output (fits L1) | p=10000: 160 KB (fits L2).
    for p in [1000usize, 10_000] {
        let evals = mp_spectrum(p, c_ratio, p as u64);

        let mut configs: Vec<(String, RmtConfig)> = vec![
            (
                "blocked_rowmajor".to_string(),
                RmtConfig::new(c_ratio)
                    .with_stieltjes(StieltjesMethod::Blocked)
                    .with_parallelism(Parallelism::Sequential),
            ),
            (
                "blocked_tiled".to_string(),
                RmtConfig::new(c_ratio)
                    .with_stieltjes(StieltjesMethod::BlockedTiled)
                    .with_parallelism(Parallelism::Sequential),
            ),
        ];
        // Block-size sensitivity of the tiled kernel at cache-stressing sizes.
        for bs in [8usize, 16, 32] {
            configs.push((
                format!("blocked_tiled_bs{bs}"),
                RmtConfig::new(c_ratio)
                    .with_stieltjes(StieltjesMethod::BlockedTiled)
                    .with_block_size(bs)
                    .with_parallelism(Parallelism::Sequential),
            ));
        }

        for (name, config) in &configs {
            let group_id = format!("cache/p={p}/{name}");
            let mut group = c.benchmark_group(&group_id);
            group.sample_size(10);
            group.measurement_time(std::time::Duration::from_secs(5));
            group.bench_function("rie_shrinkage", |b| {
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

criterion_group!(benches, bench_cache_scaling);
criterion_main!(benches);
