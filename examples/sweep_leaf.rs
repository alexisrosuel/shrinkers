//! Focused leaf-capacity sweep for the shipped `chebcode_fast` point.
//!
//! Run:
//!   cargo run --release --example sweep_leaf -- 10000 50000           # f32, theta 1.0, n 8
//!   cargo run --release --example sweep_leaf -- 50000 eta=0.1 f64

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::{CutoffConfig, Parallelism, StieltjesMethod};
use shrinkers::stieltjes::{
    compute_all_stieltjes, compute_all_stieltjes_chebcode_impl,
    compute_all_stieltjes_chebcode_impl_f32,
};
use std::time::Instant;
use support::{median, mp_spectrum};

fn main() {
    let mut ps: Vec<usize> = Vec::new();
    let mut eta_scale = 1.0f64;
    let mut theta = 1.0f64;
    let mut n = 8usize;
    let mut mode32 = true;
    for a in std::env::args().skip(1) {
        if let Some(v) = a.strip_prefix("eta=") {
            eta_scale = v.parse().unwrap();
        } else if let Some(v) = a.strip_prefix("theta=") {
            theta = v.parse().unwrap();
        } else if let Some(v) = a.strip_prefix("n=") {
            n = v.parse().unwrap();
        } else if a == "f64" {
            mode32 = false;
        } else if a == "f32" {
            mode32 = true;
        } else if let Ok(v) = a.parse() {
            ps.push(v);
        }
    }
    let ps = if ps.is_empty() { vec![10_000, 50_000] } else { ps };
    let leaves = [8usize, 12, 16, 24, 32, 48, 64, 96, 128, 256];

    for &p in &ps {
        let lam = mp_spectrum(p, 0.25, p as u64);
        let eta = eta_scale / (p as f64).sqrt();
        let inv_p = 1.0 / p as f64;
        let exact = compute_all_stieltjes(
            &lam,
            eta,
            StieltjesMethod::AutoVectorized,
            None,
            CutoffConfig::Disabled,
            64,
            Parallelism::Sequential,
        );
        let run = |leaf: usize| -> Vec<(f64, f64)> {
            if mode32 {
                compute_all_stieltjes_chebcode_impl_f32(&lam, eta, theta, n, leaf, false)
            } else {
                compute_all_stieltjes_chebcode_impl(&lam, eta, theta, n, leaf, false)
            }
        };
        println!(
            "== p={p} eta={eta:.3e} theta={theta:.2} n={n} mode={} ==",
            if mode32 { "f32" } else { "f64" }
        );
        for &leaf in &leaves {
            let mut ts = Vec::new();
            let mut res = None;
            for r in 0..5 {
                let t = Instant::now();
                let got = run(leaf);
                let dt = t.elapsed().as_secs_f64() * 1e3;
                if r > 0 {
                    ts.push(dt);
                }
                res = Some(got);
            }
            let ms = median(&mut ts);
            let got = res.unwrap();
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for (g, e) in got.iter().zip(exact.iter()) {
                num += (g.0 * inv_p - e.0).powi(2) + (g.1 * inv_p - e.1).powi(2);
                den += e.0 * e.0 + e.1 * e.1;
            }
            println!(
                "  leaf={leaf:<4} {ms:>8.3} ms   rel-L2 {:.3e}",
                (num / den).sqrt()
            );
        }
    }
}
