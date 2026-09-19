//! Dedicated A/B harness for the `chebcode_fast` preset.
//!
//! Machine-readable `KEY<TAB>value` lines so two builds can be interleaved by
//! `scripts/bench_ab.py` and diffed:
//!
//! ```text
//! cargo run --release --example measure_chebfast -- compare 50000
//! cargo run --release --example measure_chebfast -- err     20000
//! cargo run --release --example measure_chebfast -- build   50000
//! ```
//!
//! Keys emitted by `compare`:
//!   chebf.build.p{p}      tree build only (FAST preset)
//!   chebf.eval.seq.p{p}   evaluate all p points, sequential
//!   chebf.eval.par.p{p}   evaluate all p points, Rayon
//!   chebf.all.seq.p{p}    impl entry point (build + seq eval)
//!   chebf.all.par.p{p}    impl entry point (build + par eval)
//!   chebf.grid.seq.p{p}   n_points = 200 deconvolution grid, sequential
//!   chebf.grid.par.p{p}   n_points = 200 deconvolution grid, Rayon

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::{CutoffConfig, Parallelism, StieltjesMethod};
use shrinkers::stieltjes::{
    ChebCodeBatch, ChebPreset, compute_all_stieltjes, compute_all_stieltjes_chebcode_impl,
};
use std::time::Instant;
use support::{median, mp_spectrum, rel_l2};

/// Benchmark convention: eta = 1/sqrt(p).
fn eta_for(p: usize) -> f64 {
    1.0 / (p as f64).sqrt()
}

fn bench<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let mut ts = Vec::with_capacity(reps);
    for r in 0..=reps {
        let t = Instant::now();
        f();
        let dt = t.elapsed().as_secs_f64() * 1e3;
        if r > 0 {
            ts.push(dt);
        }
    }
    median(&mut ts)
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "compare".into());
    let p: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    let c = 0.25;
    let evs = mp_spectrum(p, c, 7);
    let eta = eta_for(p);
    let preset = ChebPreset::FAST;

    match which.as_str() {
        "err" => {
            // Accuracy vs the exact autovec reference (raw sums on both sides).
            let exact = compute_all_stieltjes(
                &evs,
                eta,
                StieltjesMethod::AutoVectorized,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Sequential,
            );
            // `compute_all_stieltjes` already returns the 1/p-scaled transform;
            // the impl path returns raw sums.
            let inv_p = 1.0 / p as f64;
            let got = compute_all_stieltjes_chebcode_impl(&evs, eta, preset.theta, preset.n, preset.leaf_cap, false);
            let scaled: Vec<(f64, f64)> = got.iter().map(|&(r, i)| (r * inv_p, i * inv_p)).collect();
            println!("chebf.rel_l2.p{p}\t{:.6e}", rel_l2(&scaled, &exact));
            let gotp = compute_all_stieltjes_chebcode_impl(&evs, eta, preset.theta, preset.n, preset.leaf_cap, true);
            let scaledp: Vec<(f64, f64)> = gotp.iter().map(|&(r, i)| (r * inv_p, i * inv_p)).collect();
            println!("chebf.rel_l2_par.p{p}\t{:.6e}", rel_l2(&scaledp, &exact));
        }
        "build" => {
            let t = bench(15, || {
                let b = ChebCodeBatch::build_preset(&evs, preset);
                std::hint::black_box(b);
            });
            println!("chebf.build.p{p}\t{t:.4}");
        }
        _ => {
            let reps = if p >= 40_000 { 5 } else { 11 };
            let tree = ChebCodeBatch::build_preset(&evs, preset);

            let t = bench(15, || {
                let b = ChebCodeBatch::build_preset(&evs, preset);
                std::hint::black_box(b);
            });
            println!("chebf.build.p{p}\t{t:.4}");

            let t = bench(reps, || {
                let _ = tree.evaluate(eta);
            });
            println!("chebf.eval.seq.p{p}\t{t:.4}");

            let t = bench(reps, || {
                let _ = tree.evaluate_points(&evs, eta, true);
            });
            println!("chebf.eval.par.p{p}\t{t:.4}");

            let t = bench(reps, || {
                let _ = compute_all_stieltjes_chebcode_impl(
                    &evs, eta, preset.theta, preset.n, preset.leaf_cap, false,
                );
            });
            println!("chebf.all.seq.p{p}\t{t:.4}");

            let t = bench(reps, || {
                let _ = compute_all_stieltjes_chebcode_impl(
                    &evs, eta, preset.theta, preset.n, preset.leaf_cap, true,
                );
            });
            println!("chebf.all.par.p{p}\t{t:.4}");

            // 200-point deconvolution grid (the at-points path).
            let range = evs[p - 1] - evs[0];
            let lo = (evs[0] - 0.2 * range).max(0.0);
            let hi = evs[p - 1] + 0.2 * range;
            let grid: Vec<f64> = (0..200)
                .map(|k| lo + (hi - lo) * k as f64 / 199.0)
                .collect();
            let t = bench(21, || {
                let _ = tree.evaluate_points(&grid, eta, false);
            });
            println!("chebf.grid.seq.p{p}\t{t:.4}");
            let t = bench(21, || {
                let _ = tree.evaluate_points(&grid, eta, true);
            });
            println!("chebf.grid.par.p{p}\t{t:.4}");
        }
    }
}
