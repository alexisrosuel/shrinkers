//! Phase breakdown of deconvolve_spiked at p=20k (campaign 3, lever D).
//! Run: cargo run --release --example measure_pipeline_phases

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::RmtConfig;
use shrinkers::deconvolution::{deconvolve_spiked, hybrid};
use shrinkers::spiked;
use std::time::Instant;

fn main() {
    let c = 0.25;
    let p = 20_000usize;
    // Same spectrum family as the python probe: MP bulk + 3 spikes.
    let lam_full = support::mp_spectrum(p - 3, c, p as u64);
    let mut lam: Vec<f64> = lam_full;
    lam.push(12.0);
    lam.push(7.0);
    lam.push(4.0);
    lam.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let config = RmtConfig::new(c);

    let reps = 15usize;
    let mut t_all = Vec::new();
    let mut t_sep = Vec::new();
    let mut t_bulk = Vec::new();
    for rep in 0..reps {
        let t = Instant::now();
        let _ = deconvolve_spiked(&lam, c, 200, Some(0.1 / (p as f64).sqrt()), 1.0, &config);
        let a = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let _ = spiked::separate_spikes(&lam, c, 1.0);
        let s = t.elapsed().as_secs_f64() * 1e3;
        let sep = spiked::separate_spikes(&lam, c, 1.0);
        let t = Instant::now();
        let _ = hybrid::deconvolve_bulk(
            &sep.bulk_evals,
            c,
            200,
            Some(0.1 / (p as f64).sqrt()),
            &config,
        );
        let b = t.elapsed().as_secs_f64() * 1e3;
        if rep >= 1 {
            t_all.push(a);
            t_sep.push(s);
            t_bulk.push(b);
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    println!(
        "total {:.2} ms | separate_spikes {:.3} ms | deconvolve_bulk {:.2} ms",
        med(&mut t_all),
        med(&mut t_sep),
        med(&mut t_bulk)
    );
}
