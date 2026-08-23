use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::RngExt;
use shrinkers::*;
use std::hint::black_box;

/// Generate a Marchenko-Pastur-like eigenvalue spectrum.
fn generate_mp_spectrum(p: usize, c: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    let lambda_min = (1.0 - c.sqrt()).max(0.01).powi(2);
    let lambda_max = (1.0 + c.sqrt()).powi(2);

    (0..p)
        .map(|_| {
            let u: f64 = rng.random();
            let t = lambda_min + u * (lambda_max - lambda_min);
            t + rng.random::<f64>() * 0.1
        })
        .collect::<Vec<_>>()
}

fn bench_comparison(c: &mut Criterion) {
    // Reduced set: only p=500,1000 and c=0.5 (most representative)
    // Methods: Naive (baseline), Autovec (O(p²) ref), Blocked, FFT, FFT Fused (the winners)
    let cases = [
        (500, 0.5, StieltjesMethod::Naive, Parallelism::Sequential),
        (
            500,
            0.5,
            StieltjesMethod::AutoVectorized,
            Parallelism::Sequential,
        ),
        (500, 0.5, StieltjesMethod::Blocked, Parallelism::Sequential),
        (
            500,
            0.5,
            StieltjesMethod::BlockedAutoVec,
            Parallelism::Sequential,
        ),
        (500, 0.5, StieltjesMethod::Fft5, Parallelism::Sequential),
        (500, 0.5, StieltjesMethod::Fft3, Parallelism::Sequential),
        (500, 0.5, StieltjesMethod::Fft2, Parallelism::Sequential),
        (1000, 0.5, StieltjesMethod::Naive, Parallelism::Sequential),
        (1000, 0.5, StieltjesMethod::Blocked, Parallelism::Sequential),
        (
            1000,
            0.5,
            StieltjesMethod::BlockedAutoVec,
            Parallelism::Sequential,
        ),
        (1000, 0.5, StieltjesMethod::Fft5, Parallelism::Sequential),
        (1000, 0.5, StieltjesMethod::Fft3, Parallelism::Sequential),
        (1000, 0.5, StieltjesMethod::Fft2, Parallelism::Sequential),
        // Quick check: smart default (auto-tune)
        (1000, 0.5, StieltjesMethod::Blocked, Parallelism::Sequential),
    ];

    for &(p, conc, method, par) in &cases {
        let evals = generate_mp_spectrum(p, conc);
        let config = RmtConfig::new(conc)
            .with_stieltjes(method)
            .with_parallelism(par);

        let bench_name = format!("p={},c={:.1},{}", p, conc, method.name());

        let mut group = c.benchmark_group(&bench_name);
        group.sample_size(10);

        group.bench_function("run", |b| {
            b.iter_batched(
                || evals.clone(),
                |e| rie_shrinkage(black_box(&e), black_box(&config)),
                BatchSize::SmallInput,
            )
        });

        group.finish();
    }
}

criterion_group!(benches, bench_comparison);
criterion_main!(benches);
