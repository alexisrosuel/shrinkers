//! One place to measure every configuration knob of the pipeline at a fixed
//! size (p=1000): exact-family variants, windowed cutoff ratios, block-size
//! sensitivity on the families that honor it, Rayon, and the approximate
//! methods. All through the full `rie_shrinkage` entry point.

mod support;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use shrinkers::*;
use std::hint::black_box;
use support::mp_spectrum;

fn bench_config_sweep(c: &mut Criterion) {
    let p = 1000;
    let c_ratio = 0.5;
    let evals = mp_spectrum(p, c_ratio, 42);

    let seq = |m: StieltjesMethod| {
        RmtConfig::new(c_ratio)
            .with_stieltjes(m)
            .with_parallelism(Parallelism::Sequential)
    };

    let configs: Vec<(&str, RmtConfig)> = vec![
        // Exact-family ladder: scalar -> SIMD -> blocked -> tiled.
        ("autovec", seq(StieltjesMethod::AutoVectorized)),
        ("blocked", seq(StieltjesMethod::Blocked)),
        ("blocked_autovec", seq(StieltjesMethod::BlockedAutoVec)),
        ("blocked_tiled", seq(StieltjesMethod::BlockedTiled)),
        // Cutoff/windowed pair (imag-only short-range path).
        (
            "windowed_cut10",
            seq(StieltjesMethod::BlockedWindowed)
                .with_cutoff(CutoffConfig::Enabled { ratio: 10.0 }),
        ),
        (
            "adaptive_cut10",
            seq(StieltjesMethod::Adaptive).with_cutoff(CutoffConfig::Enabled { ratio: 10.0 }),
        ),
        // Block-size sensitivity — BlockedAutoVec is the family that honors
        // config.block_size (Blocked/BlockedTiled auto-tune and ignore it).
        (
            "blocked_autovec_bs32",
            seq(StieltjesMethod::BlockedAutoVec).with_block_size(32),
        ),
        (
            "blocked_autovec_bs128",
            seq(StieltjesMethod::BlockedAutoVec).with_block_size(128),
        ),
        // Cutoff-ratio sensitivity on the windowed path.
        (
            "windowed_cut8",
            seq(StieltjesMethod::BlockedWindowed).with_cutoff(CutoffConfig::Enabled { ratio: 8.0 }),
        ),
        (
            "windowed_cut15",
            seq(StieltjesMethod::BlockedWindowed)
                .with_cutoff(CutoffConfig::Enabled { ratio: 15.0 }),
        ),
        // Parallel.
        (
            "blocked_rayon",
            RmtConfig::new(c_ratio)
                .with_stieltjes(StieltjesMethod::Blocked)
                .with_parallelism(Parallelism::Parallel),
        ),
        // Approximate methods for reference.
        ("fft5", seq(StieltjesMethod::Fft5)),
        ("treecode", seq(StieltjesMethod::TreeCode)),
    ];

    for (name, config) in &configs {
        let group_id = format!("pipeline/{name}");
        let mut group = c.benchmark_group(group_id);
        group.sample_size(10);
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

criterion_group!(benches, bench_config_sweep);
criterion_main!(benches);
