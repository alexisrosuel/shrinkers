//! Aggressive (theta, n, leaf_cap) sweep for `chebcode_fast`, with the
//! accuracy budget made explicit. Sequential; one exact reference per size.
//!
//! Run: cargo run --release --example sweep_chebfast -- [p ...]

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::{CutoffConfig, Parallelism, StieltjesMethod};
use shrinkers::stieltjes::{compute_all_stieltjes, compute_all_stieltjes_chebcode_impl};
use std::time::Instant;
use support::{median, mp_spectrum};

fn main() {
    let mut ps: Vec<usize> = Vec::new();
    let mut eta_scale = 1.0f64;
    for a in std::env::args().skip(1) {
        if let Some(v) = a.strip_prefix("eta=") {
            eta_scale = v.parse().unwrap();
        } else if let Ok(v) = a.parse() {
            ps.push(v);
        }
    }
    let ps = if ps.is_empty() {
        vec![10_000, 50_000]
    } else {
        ps
    };

    let thetas = [0.5, 0.6, 0.7, 0.8, 1.0];
    let ns = [4usize, 5, 6, 7, 8, 9, 11];
    let leaves = [32usize, 64, 128, 256];

    for &p in &ps {
        let lam = mp_spectrum(p, 0.25, p as u64);
        let eta = eta_scale / (p as f64).sqrt();
        let exact = compute_all_stieltjes(
            &lam,
            eta,
            StieltjesMethod::AutoVectorized,
            None,
            CutoffConfig::Disabled,
            64,
            Parallelism::Sequential,
        );

        // Baseline: shipped FAST preset.
        let mut base_t = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            let _ = compute_all_stieltjes_chebcode_impl(&lam, eta, 0.5, 9, 32, false);
            base_t.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let base = median(&mut base_t);

        let mut rows: Vec<(f64, f64, f64, usize, usize)> = Vec::new(); // (ms, err, theta, n, leaf)
        for &theta in &thetas {
            for &n in &ns {
                for &leaf in &leaves {
                    let mut ts = Vec::new();
                    let mut res = None;
                    for r in 0..4 {
                        let t = Instant::now();
                        let got =
                            compute_all_stieltjes_chebcode_impl(&lam, eta, theta, n, leaf, false);
                        let dt = t.elapsed().as_secs_f64() * 1e3;
                        if r > 0 {
                            ts.push(dt);
                        }
                        res = Some(got);
                    }
                    let ms = median(&mut ts);
                    // rel-L2 (both sides scaled by 1/p; exact already is).
                    let inv_p = 1.0 / p as f64;
                    let got = res.unwrap();
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    for (g, e) in got.iter().zip(exact.iter()) {
                        let gr = g.0 * inv_p - e.0;
                        let gi = g.1 * inv_p - e.1;
                        num += gr * gr + gi * gi;
                        den += e.0 * e.0 + e.1 * e.1;
                    }
                    rows.push((ms, (num / den).sqrt(), theta, n, leaf));
                }
            }
        }
        rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        println!("== p={p}  eta={eta:.3e}  FAST baseline {base:.3} ms ==");
        println!(
            "{:>8} {:>10} {:>6} {:>3} {:>5}  {:>7}",
            "ms", "err", "theta", "n", "leaf", "vs base"
        );
        for (ms, err, theta, n, leaf) in rows.iter().take(30) {
            println!(
                "{ms:>8.3} {err:>10.2e} {theta:>6.2} {n:>3} {leaf:>5}  {:>6.2}x",
                base / ms
            );
        }
        // Best preset at each accuracy budget.
        println!("-- best per budget --");
        for budget in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8] {
            if let Some((ms, err, theta, n, leaf)) = rows.iter().find(|r| r.1 <= budget) {
                println!(
                    "  err<={budget:.0e}: {ms:.3} ms ({:.2}x) err={err:.2e} theta={theta:.2} n={n} leaf={leaf}",
                    base / ms
                );
            }
        }
    }
}
