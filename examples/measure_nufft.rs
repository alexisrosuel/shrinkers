//! A/B harness for the coarse-grid NUFFT (`method = "nufft"`).
//!
//! ```text
//! cargo run --release --example measure_nufft -- 50000
//! cargo run --release --example measure_nufft -- 50000 100000 200000 1000000
//! ```
//!
//! Emits `KEY<TAB>value` lines so two builds can be interleaved:
//!   nufft.seq.p{p}   coarse-grid NUFFT, sequential
//!   nufft.par.p{p}   coarse-grid NUFFT, Rayon
//!   cheb.par.p{p}    ChebCodeFast end-to-end through the dispatcher
//!   nufft.err.p{p}   rel-L2 vs exact (p <= 50 000)

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::{CutoffConfig, Parallelism, StieltjesMethod};
use shrinkers::stieltjes::fft5;
use std::time::Instant;
use support::{median, mp_spectrum, rel_l2};

fn bench<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    f();
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    median(&mut ts)
}

fn main() {
    let mut sizes: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    if sizes.is_empty() {
        sizes = vec![10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000];
    }
    for p in sizes {
        let evs = mp_spectrum(p, 0.25, 7);
        let eta = 1.0 / (p as f64).sqrt();
        let reps = if p >= 200_000 { 3 } else { 7 };

        let seq = bench(reps, || {
            let _ = fft5::compute_all_stieltjes_nufft(&evs, eta, false);
        });
        let par = bench(reps, || {
            let _ = fft5::compute_all_stieltjes_nufft(&evs, eta, true);
        });
        let cheb = bench(reps, || {
            let _ = shrinkers::stieltjes::compute_all_stieltjes(
                &evs,
                eta,
                StieltjesMethod::ChebCodeFast,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Parallel,
            );
        });
        println!("nufft.seq.p{p}\t{seq:.4}");
        println!("nufft.par.p{p}\t{par:.4}");
        println!("cheb.par.p{p}\t{cheb:.4}");

        if p <= 50_000 {
            let exact = shrinkers::stieltjes::compute_all_stieltjes(
                &evs,
                eta,
                StieltjesMethod::AutoVectorized,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Parallel,
            );
            let got = fft5::compute_all_stieltjes_nufft(&evs, eta, false);
            let inv_p = 1.0 / p as f64;
            let scaled: Vec<(f64, f64)> =
                got.iter().map(|&(r, i)| (r * inv_p, i * inv_p)).collect();
            println!("nufft.err.p{p}\t{:.4e}", rel_l2(&scaled, &exact));
        }
    }
}
