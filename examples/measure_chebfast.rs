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
    ChebCodeBatch, ChebPreset, FastMode, compute_all_stieltjes,
    compute_all_stieltjes_chebcode_impl, compute_all_stieltjes_chebcode_impl_f32,
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

/// One timed run of a closure, in ms.
fn time<F: FnMut()>(mut f: F) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1e3
}

/// Interleaved A/B: alternating order across `rounds` (>=7 recommended), so
/// drift in machine load hits both arms roughly equally. Warmup excludes the
/// first touch of each arm. Returns (median_a_ms, median_b_ms).
fn interleaved<FA: FnMut() -> (), FB: FnMut() -> ()>(rounds: usize, mut a: FA, mut b: FB) -> (f64, f64) {
    a();
    b();
    let mut ta = Vec::with_capacity(rounds);
    let mut tb = Vec::with_capacity(rounds);
    for r in 0..rounds {
        if r % 2 == 0 {
            ta.push(time(&mut a));
            tb.push(time(&mut b));
        } else {
            tb.push(time(&mut b));
            ta.push(time(&mut a));
        }
    }
    (median(&mut ta), median(&mut tb))
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "compare".into());
    let p: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    let c = 0.25;
    let evs = mp_spectrum(p, c, 7);
    // Optional `eta=SCALE` argument (η = SCALE/√p); default 1.0.
    let eta_scale: f64 = std::env::args()
        .find_map(|a| a.strip_prefix("eta=").and_then(|v| v.parse().ok()))
        .unwrap_or(1.0);
    let eta = eta_scale / (p as f64).sqrt();
    let _ = eta_for;
    let mut preset = ChebPreset::FAST;
    if let Some(theta) = std::env::args().nth(3).and_then(|s| s.parse::<f64>().ok()) {
        preset.theta = theta;
    }
    if let Some(n) = std::env::args().nth(4).and_then(|s| s.parse::<usize>().ok()) {
        preset.n = n;
    }

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

            // f32 far-field path, same reference and scaling.
            let got32 = compute_all_stieltjes_chebcode_impl_f32(&evs, eta, preset.theta, preset.n, preset.leaf_cap, false);
            let scaled32: Vec<(f64, f64)> = got32.iter().map(|&(r, i)| (r * inv_p, i * inv_p)).collect();
            println!("chebf.rel_l2_f32.p{p}\t{:.6e}", rel_l2(&scaled32, &exact));
            let got32p = compute_all_stieltjes_chebcode_impl_f32(&evs, eta, preset.theta, preset.n, preset.leaf_cap, true);
            let scaled32p: Vec<(f64, f64)> = got32p.iter().map(|&(r, i)| (r * inv_p, i * inv_p)).collect();
            println!("chebf.rel_l2_f32_par.p{p}\t{:.6e}", rel_l2(&scaled32p, &exact));
        }
        "build" => {
            let t = bench(15, || {
                let b = ChebCodeBatch::build_preset(&evs, preset);
                std::hint::black_box(b);
            });
            println!("chebf.build.p{p}\t{t:.4}");
        }
        "ab" => {
            // Interleaved f64-vs-f32 FAR-FIELD A/B on one shared tree.
            // Keys: chebf.<metric>.<f64|f32>.p{p}
            let rounds = 9;
            let tree = ChebCodeBatch::build_preset(&evs, preset);

            let (a, b) = interleaved(
                rounds,
                || {
                    let _ = tree.evaluate_mode(eta, FastMode::F64);
                },
                || {
                    let _ = tree.evaluate_mode(eta, FastMode::F32);
                },
            );
            println!("chebf.eval.seq.f64.p{p}\t{a:.4}");
            println!("chebf.eval.seq.f32.p{p}\t{b:.4}");

            let (a, b) = interleaved(
                7,
                || {
                    let _ = tree.evaluate_points_mode(&evs, eta, true, FastMode::F64);
                },
                || {
                    let _ = tree.evaluate_points_mode(&evs, eta, true, FastMode::F32);
                },
            );
            println!("chebf.eval.par.f64.p{p}\t{a:.4}");
            println!("chebf.eval.par.f32.p{p}\t{b:.4}");

            let (a, b) = interleaved(
                7,
                || {
                    let _ = compute_all_stieltjes_chebcode_impl(
                        &evs, eta, preset.theta, preset.n, preset.leaf_cap, false,
                    );
                },
                || {
                    let _ = compute_all_stieltjes_chebcode_impl_f32(
                        &evs, eta, preset.theta, preset.n, preset.leaf_cap, false,
                    );
                },
            );
            println!("chebf.all.seq.f64.p{p}\t{a:.4}");
            println!("chebf.all.seq.f32.p{p}\t{b:.4}");

            let (a, b) = interleaved(
                7,
                || {
                    let _ = compute_all_stieltjes_chebcode_impl(
                        &evs, eta, preset.theta, preset.n, preset.leaf_cap, true,
                    );
                },
                || {
                    let _ = compute_all_stieltjes_chebcode_impl_f32(
                        &evs, eta, preset.theta, preset.n, preset.leaf_cap, true,
                    );
                },
            );
            println!("chebf.all.par.f64.p{p}\t{a:.4}");
            println!("chebf.all.par.f32.p{p}\t{b:.4}");

            // 200-point deconvolution grid (the at-points path).
            let range = evs[p - 1] - evs[0];
            let lo = (evs[0] - 0.2 * range).max(0.0);
            let hi = evs[p - 1] + 0.2 * range;
            let grid: Vec<f64> = (0..200)
                .map(|k| lo + (hi - lo) * k as f64 / 199.0)
                .collect();
            let (a, b) = interleaved(
                21,
                || {
                    let _ = tree.evaluate_points_mode(&grid, eta, false, FastMode::F64);
                },
                || {
                    let _ = tree.evaluate_points_mode(&grid, eta, false, FastMode::F32);
                },
            );
            println!("chebf.grid.seq.f64.p{p}\t{a:.4}");
            println!("chebf.grid.seq.f32.p{p}\t{b:.4}");

            let (a, b) = interleaved(
                21,
                || {
                    let _ = tree.evaluate_points_mode(&grid, eta, true, FastMode::F64);
                },
                || {
                    let _ = tree.evaluate_points_mode(&grid, eta, true, FastMode::F32);
                },
            );
            println!("chebf.grid.par.f64.p{p}\t{a:.4}");
            println!("chebf.grid.par.f32.p{p}\t{b:.4}");
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

            // Shipped path: the public dispatcher resolves ChebCodeFast to the
            // preset (f32 far field included), then scales by 1/p.
            let t = bench(reps, || {
                let _ = compute_all_stieltjes(
                    &evs,
                    eta,
                    StieltjesMethod::ChebCodeFast,
                    None,
                    CutoffConfig::Disabled,
                    64,
                    Parallelism::Sequential,
                );
            });
            println!("chebf.all.seq.p{p}\t{t:.4}");

            let t = bench(reps, || {
                let _ = compute_all_stieltjes(
                    &evs,
                    eta,
                    StieltjesMethod::ChebCodeFast,
                    None,
                    CutoffConfig::Disabled,
                    64,
                    Parallelism::Parallel,
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
