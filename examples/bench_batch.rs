//! Batch-vs-loop benchmark for the ChebCode γ-sweep workflow.
//!
//! Simulates a RIE deconvolution sweep: the same spectrum, many η values
//! (one per γ). Naive: one full `compute_all_stieltjes` call per η.
//! Batch: one `ChebCodeBatch::build`, then `evaluate_many`.
//!
//! Usage: cargo run --release --example bench_batch -- [p] [n_eta]

use shrinkers::stieltjes::ChebCodeBatch;
use std::time::Instant;

fn mp_spectrum(p: usize) -> Vec<f64> {
    let mut x = 42u64;
    let mut out = Vec::with_capacity(p);
    for _ in 0..p {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push((x >> 11) as f64 / (1u64 << 53) as f64);
    }
    let lo = (1.0 - 0.5f64.sqrt()).powi(2);
    let hi = (1.0 + 0.5f64.sqrt()).powi(2);
    let mut v: Vec<f64> = out.into_iter().map(|u| lo + u * (hi - lo)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn main() {
    let p: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(20000);
    let n_eta: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(16);
    let evs = mp_spectrum(p);
    let etas: Vec<f64> = (1..=n_eta)
        .map(|k| 1.0 / ((p as f64).sqrt() * k as f64))
        .collect();

    // Naive: full call per eta (build + eval each time).
    let t0 = Instant::now();
    let mut sink = 0.0f64;
    for &eta in &etas {
        let res = shrinkers::stieltjes::compute_all_stieltjes_chebcode(&evs, eta);
        sink += res[p / 2].1;
    }
    let loop_ms = t0.elapsed().as_secs_f64() * 1e3;

    // Batch: build once, evaluate_many across etas.
    let t1 = Instant::now();
    let batch = ChebCodeBatch::build_preset(&evs, shrinkers::stieltjes::ChebPreset::DEFAULT);
    let build_ms = t1.elapsed().as_secs_f64() * 1e3;
    let t2 = Instant::now();
    let all = batch.evaluate_many(&etas);
    let eval_ms = t2.elapsed().as_secs_f64() * 1e3;
    for r in &all {
        sink += r[p / 2].1;
    }
    println!(
        "{{\"p\": {p}, \"n_eta\": {n_eta}, \"loop_ms\": {loop_ms:.2}, \"batch_build_ms\": {build_ms:.2}, \"batch_eval_ms\": {eval_ms:.2}, \"speedup_vs_loop\": {:.2}}}",
        loop_ms / (build_ms + eval_ms)
    );
    std::hint::black_box(sink);
}
