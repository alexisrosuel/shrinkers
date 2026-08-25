//! Round-2 duel: shipped XTREME preset vs the round-1 discovery
//! (theta=0.55, n=11). Interleaved A/B timing + error vs exact kernel.
//!
//! Run: cargo run --release --example measure_xtreme_duel

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::CutoffConfig;
use shrinkers::stieltjes::{compute_all_stieltjes, compute_all_stieltjes_chebcode_impl};
use shrinkers::{Parallelism, StieltjesMethod};
use std::time::Instant;
use support::mp_spectrum;

fn main() {
    let c = 0.25;
    let reps = 9usize;
    for &p in &[2_000usize, 5_000, 20_000] {
        let lam = mp_spectrum(p, c, p as u64);
        let eta = 0.1 / (p as f64).sqrt();
        let refr = compute_all_stieltjes(
            &lam,
            eta,
            StieltjesMethod::AutoVectorized,
            None,
            CutoffConfig::Disabled,
            64,
            Parallelism::Sequential,
        );
        let inv_p = 1.0 / p as f64;
        let err_of = |r: &[(f64, f64)]| -> f64 {
            let mut num = 0.0;
            let mut den = 0.0;
            for ((rr, ri), &(gr, gi)) in r.iter().zip(refr.iter()) {
                num +=
                    (rr * inv_p - gr) * (rr * inv_p - gr) + (ri * inv_p - gi) * (ri * inv_p - gi);
                den += gr * gr + gi * gi;
            }
            (num / den).sqrt()
        };
        // XTREME = (0.25, 11, 16); cand = (0.55, 11, 32).
        let mut tx: Vec<f64> = Vec::with_capacity(reps);
        let mut tc: Vec<f64> = Vec::with_capacity(reps);
        let mut ex = 0.0f64;
        let mut ec = 0.0f64;
        for rep in 0..reps {
            let pair = if rep % 2 == 0 {
                [(0.25f64, 11usize, 16usize, false), (0.55, 11, 32, true)]
            } else {
                [(0.55, 11, 32, true), (0.25, 11, 16, false)]
            };
            for &(th, nn, lf, is_c) in pair.iter() {
                let t = Instant::now();
                let r = compute_all_stieltjes_chebcode_impl(&lam, eta, th, nn, lf, false);
                let dt = t.elapsed().as_secs_f64() * 1e3;
                if is_c {
                    tc.push(dt);
                    ec = err_of(&r);
                } else {
                    tx.push(dt);
                    ex = err_of(&r);
                }
            }
        }
        tx.sort_by(|a, b| a.partial_cmp(b).unwrap());
        tc.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "p={p:<6} xtreme(0.25,11,16): {:>7.2} ms err {ex:.1e} | cand(0.55,11,32): {:>7.2} ms ({:+.1}%) err {ec:.1e}",
            tx[reps / 2],
            tc[reps / 2],
            (tc[reps / 2] / tx[reps / 2] - 1.0) * 100.0,
        );
    }
}
