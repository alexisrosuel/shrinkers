//! Lever C: windowed cutoff-ratio sweep — time + error curve.
//! Run: cargo run --release --example measure_windowed_ratio

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::CutoffConfig;
use shrinkers::stieltjes::compute_all_stieltjes;
use shrinkers::{Parallelism, StieltjesMethod};
use std::time::Instant;

fn main() {
    let c = 0.25;
    let p = 10_000usize;
    let lam = support::mp_spectrum(p, c, p as u64);
    let eta = 1.0 / (p as f64).sqrt();
    let refr = compute_all_stieltjes(
        &lam,
        eta,
        StieltjesMethod::AutoVectorized,
        None,
        CutoffConfig::Disabled,
        64,
        Parallelism::Sequential,
    );
    let den: f64 = refr.iter().map(|&(r, i)| r * r + i * i).sum();
    for ratio in [4.0f64, 8.0, 10.0, 14.0, 20.0, 30.0] {
        let mut ts = Vec::new();
        let err = loop {
            let t = Instant::now();
            let r = compute_all_stieltjes(
                &lam,
                eta,
                StieltjesMethod::BlockedWindowed,
                None,
                CutoffConfig::Enabled { ratio },
                64,
                Parallelism::Sequential,
            );
            ts.push(t.elapsed().as_secs_f64() * 1e3);
            if ts.len() == 9 {
                break (r.iter())
                    .zip(refr.iter())
                    .map(|(&(a, b), &(x, y))| (a - x) * (a - x) + (b - y) * (b - y))
                    .sum::<f64>()
                    / den;
            }
        }
        .sqrt();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("ratio={ratio:<5} {:>7.2} ms | rel_l2 {err:.2e}", ts[4]);
    }
}
