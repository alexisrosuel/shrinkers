//! Backlog item 1: re-verify the speed_seq p<=50000 bin (fft5 by a thin
//! single-session margin). Uses the EXACT harness spectrum construction.
//! Run: cargo run --release --example measure_fft5_duel

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::CutoffConfig;
use shrinkers::stieltjes::compute_all_stieltjes;
use shrinkers::{Parallelism, StieltjesMethod};
use std::time::Instant;
use support::{harness_spectrum, median};

fn main() {
    let p = 50_000usize;
    let lam = harness_spectrum(p);
    let eta = 1.0 / (p as f64).sqrt();
    let reps = 15usize;
    let mut tf = Vec::new();
    let mut tx = Vec::new();
    for rep in 0..reps {
        // Alternate order every rep (thermal-drift cancelation).
        let pair = if rep % 2 == 0 {
            [
                (StieltjesMethod::ChebCodeFast, false),
                (StieltjesMethod::Fft5, true),
            ]
        } else {
            [
                (StieltjesMethod::Fft5, true),
                (StieltjesMethod::ChebCodeFast, false),
            ]
        };
        for &(m, is_x) in pair.iter() {
            let t = Instant::now();
            let _ = compute_all_stieltjes(
                &lam,
                eta,
                m,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Sequential,
            );
            let dt = t.elapsed().as_secs_f64() * 1e3;
            if is_x {
                tx.push(dt);
            } else {
                tf.push(dt);
            }
        }
    }
    let f_fast = median(&mut tf);
    let f_fft = median(&mut tx);
    println!(
        "p=50000 seq | chebcode_fast {f_fast:>7.2} ms | fft5 {f_fft:>7.2} ms ({:+.1}% vs fast)",
        (f_fft / f_fast - 1.0) * 100.0
    );
}
