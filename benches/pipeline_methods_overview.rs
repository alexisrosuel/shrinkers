//! Overview of every Stieltjes method through the FULL `rie_shrinkage`
//! pipeline at two representative sizes — the "which method should I pick?"
//! benchmark, run end-to-end exactly as a user would.

mod support;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use shrinkers::*;
use std::hint::black_box;
use support::mp_spectrum;

fn bench_methods_overview(c: &mut Criterion) {
    // One row per (p, method). Fft2/Fft3 alias the same kernel as Fft5, so
    // only Fft5 is measured here.
    let cases: &[(usize, f64, StieltjesMethod)] = &[
        (500, 0.5, StieltjesMethod::Naive),
        (500, 0.5, StieltjesMethod::AutoVectorized),
        (500, 0.5, StieltjesMethod::Blocked),
        (500, 0.5, StieltjesMethod::BlockedAutoVec),
        (500, 0.5, StieltjesMethod::Fft5),
        (1000, 0.5, StieltjesMethod::Naive),
        (1000, 0.5, StieltjesMethod::AutoVectorized),
        (1000, 0.5, StieltjesMethod::Blocked),
        (1000, 0.5, StieltjesMethod::BlockedAutoVec),
        (1000, 0.5, StieltjesMethod::Fft5),
    ];

    for &(p, cratio, method) in cases {
        let evals = mp_spectrum(p, cratio, p as u64);
        let config = RmtConfig::new(cratio)
            .with_stieltjes(method)
            .with_parallelism(Parallelism::Sequential);

        let group_id = format!("pipeline/p={p}/{}", method.name());
        let mut group = c.benchmark_group(&group_id);
        group.sample_size(10);
        group.bench_function("rie_shrinkage", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| rie_shrinkage(black_box(&e), black_box(&config)),
                BatchSize::SmallInput,
            )
        });
        group.finish();
    }
}

criterion_group!(benches, bench_methods_overview);
criterion_main!(benches);
