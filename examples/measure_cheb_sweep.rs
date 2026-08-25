//! Parameter-sweep for the ChebCode speed frontier: grid over
//! (theta, n, leaf_cap), interleaved A/B against the shipped FAST preset.
//!
//! Run: cargo run --release --example measure_cheb_sweep

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::CutoffConfig;
use shrinkers::stieltjes::{compute_all_stieltjes, compute_all_stieltjes_chebcode_impl};
use shrinkers::{Parallelism, StieltjesMethod};
use std::time::Instant;
use support::mp_spectrum;

type PairResults = ((f64, f64), (Vec<(f64, f64)>, Vec<(f64, f64)>));

fn bench_pair(
    lam: &[f64],
    eta: f64,
    base: (f64, usize, usize),
    cand: (f64, usize, usize),
    reps: usize,
) -> PairResults {
    // Interleaved order flips each rep to cancel thermal drift.
    let mut tb = Vec::with_capacity(reps);
    let mut tc = Vec::with_capacity(reps);
    let mut rb = None;
    let mut rc = None;
    for rep in 0..reps {
        let pair = if rep % 2 == 0 {
            [(base, false), (cand, true)]
        } else {
            [(cand, true), (base, false)]
        };
        for &(m, is_cand) in pair.iter() {
            let t = Instant::now();
            let r = compute_all_stieltjes_chebcode_impl(lam, eta, m.0, m.1, m.2, false);
            let dt = t.elapsed().as_secs_f64() * 1e3;
            if is_cand {
                tc.push(dt);
                rc = Some(r);
            } else {
                tb.push(dt);
                rb = Some(r);
            }
        }
    }
    tb.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tc.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ((tb[reps / 2], tc[reps / 2]), (rb.unwrap(), rc.unwrap()))
}

fn main() {
    let c = 0.25;
    let fast: (f64, usize, usize) = (0.5, 9, 32);
    let grid: Vec<(f64, usize, usize)> = vec![
        (0.5, 8, 32),
        (0.55, 9, 32),
        (0.6, 9, 32),
        (0.55, 8, 32),
        (0.5, 9, 16),
        (0.5, 9, 64),
        (0.55, 9, 64),
        (0.45, 9, 32),
    ];

    for &p in &[5_000usize, 20_000] {
        let lam = mp_spectrum(p, c, p as u64);
        let eta = 0.1 / (p as f64).sqrt();
        // Reference once per size (exact kernel, machine precision).
        let refr = compute_all_stieltjes(
            &lam,
            eta,
            StieltjesMethod::AutoVectorized,
            None,
            CutoffConfig::Disabled,
            64,
            Parallelism::Sequential,
        );
        // The impl path returns RAW sums; the dispatcher normalizes by 1/p.
        let inv_p = 1.0 / p as f64;
        let err_of = |r: &[(f64, f64)]| -> f64 {
            let mut num = 0.0;
            let mut den = 0.0;
            for ((rr, ri), &(gr, gi)) in r.iter().zip(refr.iter()) {
                let (rr, ri) = (rr * inv_p, ri * inv_p);
                num += (rr - gr) * (rr - gr) + (ri - gi) * (ri - gi);
                den += gr * gr + gi * gi;
            }
            (num / den).sqrt()
        };
        println!("== p={p}, eta={eta:.3e} ==");
        let ((tb, _), (rb, _)) = bench_pair(&lam, eta, fast, fast, 9);
        eprintln!("base sanity {tb:.2} ms");
        for g in &grid {
            let ((t_base, t_cand), (r_base, r_cand)) = bench_pair(&lam, eta, fast, *g, 9);
            let eb = err_of(&r_base);
            let ec = err_of(&r_cand);
            println!(
                "cand theta={:.2} n={} leaf={:<3} | {:>7.2} ms vs {:>7.2} ({:+.1}%) | err {:.1e} vs {:.1e}",
                g.0,
                g.1,
                g.2,
                t_cand,
                t_base,
                (t_cand / t_base - 1.0) * 100.0,
                ec,
                eb,
            );
            let _ = rb;
        }
    }
}
