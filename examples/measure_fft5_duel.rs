//! Backlog item 1: re-verify the speed_seq p<=50000 bin (fft5 by a thin
//! single-session margin). Uses the EXACT harness spectrum construction.
//! Run: cargo run --release --example measure_fft5_duel

struct Lcg(u64);
impl Iterator for Lcg {
    type Item = f64;
    fn next(&mut self) -> Option<f64> {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        Some((self.0 >> 11) as f64 / (1u64 << 53) as f64)
    }
}

fn harness_spectrum(p: usize) -> Vec<f64> {
    let c: f64 = 0.5;
    let lo = (1.0 - c.sqrt()).powi(2);
    let hi = (1.0 + c.sqrt()).powi(2);
    let mut v: Vec<f64> = Lcg(42)
        .take(p.saturating_sub(2))
        .map(|x| lo + x * (hi - lo))
        .collect();
    while v.len() < p - 2 {
        v.push(lo + Lcg(7).next().unwrap() * (hi - lo));
    }
    v.truncate(p - 2);
    v.push(hi * 2.3);
    v.push(lo * 0.35);
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

use shrinkers::config::CutoffConfig;
use shrinkers::stieltjes::compute_all_stieltjes;
use shrinkers::{Parallelism, StieltjesMethod};
use std::time::Instant;

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
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tx.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "p=50000 seq | chebcode_fast {:>7.2} ms | fft5 {:>7.2} ms ({:+.1}% vs fast)",
        tf[tf.len() / 2],
        tx[tx.len() / 2],
        (tx[tx.len() / 2] / tf[tf.len() / 2] - 1.0) * 100.0
    );
}
