//! Long-running ChebCodeFast loop for `sample`-based profiling.
//!
//! Prints its PID first, then alternates:
//!   - `seq` mode: build + sequential evaluation at all p points
//!   - `par` mode: build + Rayon evaluation at all p points
//!   - `grid` mode: build + 200-point grid evaluation
//!
//! Usage:
//!   cargo run --release --example profile_chebfast -- seq  [p] [seconds]
//!   cargo run --release --example profile_chebfast -- par  [p] [seconds]

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::stieltjes::{ChebCodeBatch, ChebPreset};
use std::time::Instant;
use support::mp_spectrum;

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "seq".into());
    let p: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    let secs: f64 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20.0);

    let evs = mp_spectrum(p, 0.25, 7);
    let eta = 1.0 / (p as f64).sqrt();
    let preset = ChebPreset::FAST;

    println!("PID={} mode={mode} p={p}", std::process::id());

    let range = evs[p - 1] - evs[0];
    let lo = (evs[0] - 0.2 * range).max(0.0);
    let hi = evs[p - 1] + 0.2 * range;
    let grid: Vec<f64> = (0..200)
        .map(|k| lo + (hi - lo) * k as f64 / 199.0)
        .collect();

    let start = Instant::now();
    let mut iters = 0u64;
    let mut sink = 0.0f64;
    while start.elapsed().as_secs_f64() < secs {
        let tree = ChebCodeBatch::build_preset(&evs, preset);
        let r = match mode.as_str() {
            "par" => tree.evaluate_points(&evs, eta, true),
            "grid" => tree.evaluate_points(&grid, eta, false),
            _ => tree.evaluate(eta),
        };
        sink += r[r.len() / 2].0;
        iters += 1;
    }
    println!("done iters={iters} sink={sink}");
}
