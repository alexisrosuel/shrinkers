//! A/B harness for the coarse-grid NUFFT (`method = "nufft"`).
//!
//! ```text
//! cargo run --release --example measure_nufft -- 50000
//! cargo run --release --example measure_nufft -- 50000 100000 200000 1000000
//! cargo run --release --example measure_nufft -- sweep 50000
//! ```
//!
//! Emits `KEY<TAB>value` lines so two builds can be interleaved:
//!   nufft.seq.p{p}   coarse-grid NUFFT, sequential
//!   nufft.par.p{p}   coarse-grid NUFFT, Rayon
//!   cheb.par.p{p}    ChebCodeFast end-to-end through the dispatcher
//!   nufft.err.p{p}   rel-L2 vs exact (p <= 50 000; subset Blocked above)
//!
//! The `sweep` mode prints `sweep.order=<Order>.m=<m>\t<ms>\t<rel-L2>` for
//! the parameter table in `docs/nufft_algorithms.md`.

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::{CutoffConfig, Parallelism, StieltjesMethod};
use shrinkers::stieltjes::{
    NufftOptions, Order, compute_all_stieltjes_nufft, compute_all_stieltjes_nufft_with_options,
};
use std::time::Instant;
use support::{median, mp_spectrum, rel_l2};

fn bench<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    f();
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    median(&mut ts)
}

fn run_sweep(p: usize) {
    let evs = mp_spectrum(p, 0.25, 7);
    let eta = 1.0 / (p as f64).sqrt();
    let exact = shrinkers::stieltjes::compute_all_stieltjes(
        &evs,
        eta,
        StieltjesMethod::AutoVectorized,
        None,
        CutoffConfig::Disabled,
        64,
        Parallelism::Parallel,
    );
    let inv_p = 1.0 / p as f64;
    for order in [Order::Cubic, Order::Heptic] {
        for m in [16_384usize, 32_768, 65_536] {
            let opts = NufftOptions {
                order,
                m_override: Some(m),
                ..NufftOptions::default()
            };
            let t = bench(5, || {
                let _ = compute_all_stieltjes_nufft_with_options(&evs, eta, false, &opts);
            });
            let got = compute_all_stieltjes_nufft_with_options(&evs, eta, false, &opts);
            let scaled: Vec<(f64, f64)> =
                got.iter().map(|&(r, i)| (r * inv_p, i * inv_p)).collect();
            println!(
                "sweep.order={order:?}.m={m}\t{t:.4}\t{:.4e}",
                rel_l2(&scaled, &exact)
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("sweep") {
        let p = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50_000);
        run_sweep(p);
        return;
    }
    let mut sizes: Vec<usize> = args.iter().filter_map(|s| s.parse().ok()).collect();
    if sizes.is_empty() {
        sizes = vec![10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000];
    }
    for p in sizes {
        let evs = mp_spectrum(p, 0.25, 7);
        let eta = 1.0 / (p as f64).sqrt();
        let reps = if p >= 200_000 { 3 } else { 7 };

        let seq = bench(reps, || {
            let _ = compute_all_stieltjes_nufft(&evs, eta, false);
        });
        let par = bench(reps, || {
            let _ = compute_all_stieltjes_nufft(&evs, eta, true);
        });
        let cheb = bench(reps, || {
            let _ = shrinkers::stieltjes::compute_all_stieltjes(
                &evs,
                eta,
                StieltjesMethod::ChebCodeFast,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Parallel,
            );
        });
        println!("nufft.seq.p{p}\t{seq:.4}");
        println!("nufft.par.p{p}\t{par:.4}");
        println!("cheb.par.p{p}\t{cheb:.4}");

        let got = compute_all_stieltjes_nufft(&evs, eta, false);
        if p <= 50_000 {
            let exact = shrinkers::stieltjes::compute_all_stieltjes(
                &evs,
                eta,
                StieltjesMethod::AutoVectorized,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Parallel,
            );
            let inv_p = 1.0 / p as f64;
            let scaled: Vec<(f64, f64)> =
                got.iter().map(|&(r, i)| (r * inv_p, i * inv_p)).collect();
            println!("nufft.err.p{p}\t{:.4e}", rel_l2(&scaled, &exact));
        } else {
            // Exact O(p*Q) reference on a subset of queries, so the large-p
            // points carry an error number without an O(p^2) sweep.
            let q: Vec<usize> = (0..128).map(|k| (k * 7919 + 13) % p).collect();
            let qpts: Vec<f64> = q.iter().map(|&i| evs[i]).collect();
            let refr = shrinkers::stieltjes::compute_stieltjes_at_points(
                &qpts,
                &evs,
                eta,
                StieltjesMethod::Blocked,
                None,
                Parallelism::Parallel,
                None,
            );
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for (k, &i) in q.iter().enumerate() {
                let (rr, ri) = refr[k];
                let (gr, gi) = got[i];
                num += (gr - rr).powi(2) + (gi - ri).powi(2);
                den += rr * rr + ri * ri;
            }
            println!("nufft.err.p{p}\t{:.4e}", (num / den).sqrt());
        }
    }
}
