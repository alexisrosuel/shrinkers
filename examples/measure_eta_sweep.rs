//! η (regularization offset) sensitivity study — measures, not folklore.
//!
//! Experiments, JSON rows on stdout:
//!
//! A. Kernel/discretization error: empirical Stieltjes transform of a large
//!    MP-distributed spectrum vs DIRECT QUADRATURE of the exact MP density
//!    at the same offsets — isolates how η trades quadrature noise against
//!    boundary-layer bias, with no sampling noise involved.
//! B1. Bulk-only cleaning: how the FINAL RIE factors deviate from the true
//!     population (all ones) as η scales through f/√p, per method.
//!     Stability = spread across seeds.
//! B2. Spiked model through the FULL `deconvolve_spiked` pipeline (detection
//!     + BBP debiasing + bulk deconvolution), sweeping its η argument.
//! C. Runtime vs η at fixed size.
//!
//! Run: cargo run --release --example measure_eta_sweep > docs/pareto/eta_sweep.json

#[path = "../benches/support/mod.rs"]
mod support;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;
use support::mp_spectrum;

use shrinkers::config::CutoffConfig;
use shrinkers::deconvolution::deconvolve_spiked;
use shrinkers::stieltjes::compute_all_stieltjes;
use shrinkers::{Parallelism, RmtConfig, StieltjesMethod, rie_shrinkage};

const C: f64 = 0.25; // concentration ratio p/n
const K: usize = 3; // number of injected spikes
const LO: f64 = 0.25; // MP bulk edges for c = 0.25, σ² = 1
const HI: f64 = 2.25;

/// Global comma bookkeeping across experiment parts.
struct Comma(bool);

impl Comma {
    fn lead(&mut self) -> &'static str {
        let s = if self.0 { "," } else { "" };
        self.0 = true;
        s
    }
}

/// Direct quadrature of m(x - iη) = ∫ ρ(t)/(t - x + iη) dt for the MP
/// density (σ²=1, ratio C), with a cosine substitution concentrating nodes
/// near the square-root endpoints. Ground truth for part A.
fn mp_m_quadrature(x: f64, eta: f64, n: usize) -> (f64, f64) {
    let mid = (LO + HI) / 2.0;
    let half = (HI - LO) / 2.0;
    let du = std::f64::consts::PI / n as f64;
    let mut acc = (0.0, 0.0);
    for i in 0..n {
        let u = (i as f64 + 0.5) * du;
        let t = mid - half * u.cos();
        let weight = half * u.sin() * du;
        let rho = ((HI - t) * (t - LO)).sqrt() / (2.0 * std::f64::consts::PI * C * t);
        let dr = t - x;
        let den = dr * dr + eta * eta;
        acc.0 += weight * rho * dr / den;
        acc.1 += weight * rho * (-eta) / den;
    }
    acc
}

/// A: kernel error vs quadrature reference, sweeping η.
fn run_a(cm: &mut Comma) {
    let p = 20_000usize;
    let lam = mp_spectrum(p, C, 11);
    let min_gap = lam
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    // Evaluate strictly ABOVE the bulk edge: no singular self-structure
    // there, so crate-vs-quadrature isolates the eta smoothing bias plus
    // a small sampling floor. Inside the support, principal-value
    // fluctuations of the empirical measure dominate and would confound
    // the number.
    let mut pts: Vec<f64> = Vec::new();
    let mut x = HI + 0.02;
    while x < HI + 0.8 {
        pts.push(x);
        x += 0.004;
    }

    for &f in &[0.0003f64, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] {
        let eta = f / (p as f64).sqrt();
        // Match the crate's imaginary-sign convention once per η.
        let probe_head = compute_all_stieltjes(
            &lam[..100],
            eta,
            StieltjesMethod::AutoVectorized,
            None,
            CutoffConfig::Disabled,
            64,
            Parallelism::Sequential,
        );
        let want_pos = probe_head[0].1 > 0.0;
        let got = compute_all_stieltjes(
            &pts,
            eta,
            StieltjesMethod::AutoVectorized,
            None,
            CutoffConfig::Disabled,
            64,
            Parallelism::Sequential,
        );
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &x) in pts.iter().enumerate() {
            let (mr, mi) = mp_m_quadrature(x, eta, 200_000);
            let (mr, mi) = if want_pos { (mr, -mi) } else { (mr, mi) };
            let dr = got[i].0 - mr;
            let di = got[i].1 - mi;
            num += dr * dr + di * di;
            den += mr * mr + mi * mi;
        }
        let rel = (num / den).sqrt();
        eprintln!("A f={f:<7} rel_l2={rel:.3e}");
        println!(
            "{}{{\"part\": \"A\", \"factor\": {f}, \"eta\": {eta:.3e}, \
             \"min_gap\": {min_gap:.3e}, \"rel_l2\": {rel:.3e}}}",
            cm.lead()
        );
    }
}

/// IID draws from the MP(σ²=1,C) density on [λ₋,λ₊] via rejection, plus
/// deterministic BBP-forward-mapped sample spikes on top when requested.
/// RIE factors depend only on the empirical measure, so iid marginals
/// suffice and avoid materializing Wishart matrices inside a sweep.
fn mp_like_sample(pop_desc_len: usize, seed: u64, with_spikes: bool) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(77));
    let mut v: Vec<f64> = if with_spikes {
        // BBP forward map: population spike ℓ > 1+√c appears in the sample
        // at λ = ℓ·(1 + c/(ℓ−1)). Deterministic → η is the only variable.
        [12.0f64, 7.0, 4.0]
            .iter()
            .map(|&s| s * (1.0 + C / (s - 1.0)))
            .collect()
    } else {
        Vec::new()
    };
    while v.len() < pop_desc_len {
        let u = rng.random::<f64>();
        let x = LO + u * (HI - LO);
        let pdf = ((HI - x) * (x - LO)).sqrt();
        if rng.random::<f64>() < pdf * 2.5 {
            v.push(x);
        }
    }
    v.sort_by(|a, b| b.partial_cmp(a).unwrap());
    v
}

/// B1: bulk-only RIE cleaning error vs η, many seeds. The truth is σ²=1 for
/// every mode, so |ξ−1| is the absolute error.
fn run_b1(cm: &mut Comma) {
    let seeds: u64 = 12;
    for &p in &[500usize, 2000] {
        for &f in &[0.003f64, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0] {
            for &method in &[StieltjesMethod::ChebCodeFast, StieltjesMethod::BlockedTiled] {
                let mut errs = Vec::new();
                for seed in 0..seeds {
                    let lam = mp_like_sample(p, p as u64 * 1000 + seed, false);
                    let eta = f / (p as f64).sqrt();
                    let cfg = RmtConfig::new(C)
                        .with_stieltjes(method)
                        .with_parallelism(Parallelism::Sequential)
                        .with_eta(eta);
                    let xi = rie_shrinkage(&lam, &cfg);
                    let mut v: Vec<f64> = xi.iter().map(|&x| (x - 1.0).abs()).collect();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    errs.push(xi.iter().map(|&x| (x - 1.0).abs()).sum::<f64>() / p as f64);
                }
                let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
                let std = |v: &[f64]| {
                    let m = mean(v);
                    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
                };
                eprintln!("B1 p={p} f={f:<5} {method:?} bulk={:.3e}", mean(&errs));
                let name = format!("{method:?}");
                println!(
                    "{}{{\"part\": \"B1\", \"p\": {p}, \"factor\": {f}, \"method\": \"{name}\", \
                     \"mean_abs_err_mean\": {:.3e}, \"mean_abs_err_std\": {:.3e}, \
                     \"seeds\": {seeds}}}",
                    cm.lead(),
                    mean(&errs),
                    std(&errs),
                );
            }
        }
    }
}

/// B2: full spiked pipeline (`deconvolve_spiked`) vs η — detection, BBP
/// debiasing and bulk deconvolution all see the same swept offset.
fn run_b2(cm: &mut Comma) {
    let seeds: u64 = 8;
    for &p in &[500usize, 2000] {
        for &f in &[0.003f64, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0] {
            for &method in &[StieltjesMethod::ChebCodeFast, StieltjesMethod::BlockedTiled] {
                let mut k_ok = 0u32;
                let mut spike_errs = vec![Vec::new(); K];
                let pop_desc: Vec<f64> = [12.0f64, 7.0, 4.0]
                    .iter()
                    .copied()
                    .chain(std::iter::repeat_n(1.0, p - K))
                    .collect();
                for seed in 0..seeds {
                    let lam = mp_like_sample(p, p as u64 * 5000 + seed, true);
                    let eta = f / (p as f64).sqrt();
                    let cfg = RmtConfig::new(C)
                        .with_stieltjes(method)
                        .with_parallelism(Parallelism::Sequential);
                    let res = deconvolve_spiked(&lam, C, 200, Some(eta), 1.0, &cfg);
                    if res.k == K {
                        k_ok += 1;
                        for s in 0..K {
                            spike_errs[s].push((res.spikes[s] - pop_desc[s]).abs() / pop_desc[s]);
                        }
                    }
                }
                let mean = |v: &[f64]| v.iter().sum::<f64>() / (v.len().max(1)) as f64;
                eprintln!(
                    "B2 p={p} f={f:<5} {method:?} k_ok={k_ok}/{seeds} sp0={:.3}",
                    mean(&spike_errs[0])
                );
                let name = format!("{method:?}");
                println!(
                    "{}{{\"part\": \"B2\", \"p\": {p}, \"factor\": {f}, \"method\": \"{name}\", \
                     \"k_ok\": {k_ok}, \"seeds\": {seeds}, \
                     \"spike_relerr_mean\": {:?}}}",
                    cm.lead(),
                    spike_errs.iter().map(|v| mean(v)).collect::<Vec<_>>(),
                );
            }
        }
    }
}

/// C: runtime vs η at fixed size.
fn run_c(cm: &mut Comma) {
    let p = 10_000usize;
    let lam = mp_spectrum(p, C, 3);
    for &f in &[0.01f64, 0.1, 1.0] {
        for &method in &[StieltjesMethod::ChebCodeFast, StieltjesMethod::BlockedTiled] {
            let eta = f / (p as f64).sqrt();
            let cfg = RmtConfig::new(C)
                .with_stieltjes(method)
                .with_parallelism(Parallelism::Sequential)
                .with_eta(eta);
            let mut ts = Vec::new();
            for _ in 0..9 {
                let t0 = Instant::now();
                let _ = rie_shrinkage(&lam, &cfg);
                ts.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!("C f={f:<5} {method:?} {:.2} ms", ts[ts.len() / 2]);
            let name = format!("{method:?}");
            println!(
                "{}{{\"part\": \"C\", \"p\": {p}, \"factor\": {f}, \"method\": \"{name}\", \
                 \"median_ms\": {:.3}}}",
                cm.lead(),
                ts[ts.len() / 2]
            );
        }
    }
}

fn main() {
    println!(
        "{{\"meta\": {{\"date_epoch_s\": {}, \"c\": {C}, \"eta_rule\": \"factor/sqrt(p)\", \
         \"note\": \"generated by examples/measure_eta_sweep.rs\"}}, \"rows\": [",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    let mut cm = Comma(false);
    run_a(&mut cm);
    run_b1(&mut cm);
    run_b2(&mut cm);
    run_c(&mut cm);
    println!("]}}");
}
