//! Accuracy/speed landscape of the `fft5` grid convolution as a function of
//! the grid-transfer order (linear/cubic/quintic/heptic), grid size and
//! padding. Dumps JSON lines on stdout for `scripts/analyze_order_sweep.py`.
//!
//! Usage:
//!   cargo run --release --example order_sweep > docs/pareto/order_sweep.jsonl
//!
//! Two experiments, both against the exact O(p²) sequential reference:
//!   A. "grid"  — error & runtime vs forced grid size m, one series per
//!                transfer order (reveals the empirical order of accuracy
//!                and where each series hits the wrap-around floor);
//!   B. "pad"   — error vs the kernel-tail padding multiplier
//!                (pad = pad_mult·η) at a grid large enough that the
//!                transfer error is negligible (reveals how the periodization
//!                floor scales with the image-pole distance).
use shrinkers::config::{CutoffConfig, Parallelism, StieltjesMethod};
use shrinkers::stieltjes::fft5::{Fft5Options, Order};
use std::time::Instant;

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

/// MP-like spectrum (c=0.5 bulk + two spikes) — same recipe as pareto_data.
fn spectrum(p: usize) -> Vec<f64> {
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

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn bench<F: FnMut()>(mut f: F) -> f64 {
    f(); // warmup
    let probe = {
        let st = Instant::now();
        f();
        st.elapsed().as_secs_f64() * 1e3
    };
    let reps = if probe < 2.0 {
        15
    } else if probe < 20.0 {
        7
    } else {
        3
    };
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let st = Instant::now();
        f();
        ts.push(st.elapsed().as_secs_f64() * 1e3);
    }
    median(&mut ts)
}

fn rel_l2(res: &[(f64, f64)], refr: &[(f64, f64)], p: f64) -> f64 {
    // fft5 returns raw sums; the exact reference is averaged over p.
    let num: f64 = res
        .iter()
        .zip(refr.iter())
        .map(|((gr, gi), (rr, ri))| (gr / p - rr).powi(2) + (gi / p - ri).powi(2))
        .sum();
    let den: f64 = refr.iter().map(|(rr, ri)| rr * rr + ri * ri).sum();
    (num / den).sqrt()
}

const ORDER_NAMES: &[(&str, Order)] = &[
    ("linear", Order::Linear),
    ("cubic", Order::Cubic),
    ("quintic", Order::Quintic),
    ("heptic", Order::Heptic),
];

/// Experiment A: error/runtime vs forced grid size, per order.
const M_GRID: &[usize] = &[
    2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
];

/// Experiment B: kernel-tail padding multipliers (pad = mult·η).
const PAD_MULTS: &[f64] = &[250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];

fn main() {
    eprintln!("# experiment A: grid sweep");
    for &p in &[1000usize, 5000, 20000] {
        let evs = spectrum(p);
        let eta = 1.0 / (p as f64).sqrt();

        // Exact reference (sequential tiled kernel).
        let refr = shrinkers::stieltjes::compute_all_stieltjes(
            &evs,
            eta,
            StieltjesMethod::BlockedTiled,
            None,
            CutoffConfig::Disabled,
            32,
            Parallelism::Sequential,
        );

        for &(oname, order) in ORDER_NAMES {
            // Adaptive-grid anchor point (m = null in JSON).
            let opts = Fft5Options {
                order,
                ..Fft5Options::default()
            };
            let mut res = Vec::new();
            let ms = bench(|| {
                res = shrinkers::stieltjes::fft5::compute_all_stieltjes_fft5_with_options(
                    &evs, eta, &opts,
                )
            });
            println!(
                "{{\"kind\":\"grid\",\"p\":{},\"order\":\"{}\",\"m\":null,\"ms\":{:.4},\"err\":{:.3e}}}",
                p,
                oname,
                ms,
                rel_l2(&res, &refr, evs.len() as f64)
            );
            eprintln!("  done {} auto p={}", oname, p);

            for &m in M_GRID {
                if m < 2 * p.min(1024) {
                    continue; // absurdly undersampled grids only waste time
                }
                let opts = Fft5Options {
                    order,
                    m_override: Some(m),
                    ..Fft5Options::default()
                };
                let mut res = Vec::new();
                let ms = bench(|| {
                    res = shrinkers::stieltjes::fft5::compute_all_stieltjes_fft5_with_options(
                        &evs, eta, &opts,
                    )
                });
                println!(
                    "{{\"kind\":\"grid\",\"p\":{},\"order\":\"{}\",\"m\":{},\"ms\":{:.4},\"err\":{:.3e}}}",
                    p,
                    oname,
                    m,
                    ms,
                    rel_l2(&res, &refr, evs.len() as f64)
                );
                eprintln!("  done {} m={} p={}", oname, m, p);
            }
        }
    }

    eprintln!("# experiment B: padding sweep (floor vs image-pole distance)");
    for &p in &[5000usize] {
        let evs = spectrum(p);
        let eta = 1.0 / (p as f64).sqrt();
        let refr = shrinkers::stieltjes::compute_all_stieltjes(
            &evs,
            eta,
            StieltjesMethod::BlockedTiled,
            None,
            CutoffConfig::Disabled,
            32,
            Parallelism::Sequential,
        );
        for &(oname, order) in ORDER_NAMES {
            for &mult in PAD_MULTS {
                let opts = Fft5Options {
                    order,
                    m_override: Some(262_144),
                    pad_eta_mult: mult,
                    ..Fft5Options::default()
                };
                let mut res = Vec::new();
                let ms = bench(|| {
                    res = shrinkers::stieltjes::fft5::compute_all_stieltjes_fft5_with_options(
                        &evs, eta, &opts,
                    )
                });
                println!(
                    "{{\"kind\":\"pad\",\"p\":{},\"order\":\"{}\",\"pad_mult\":{},\"m\":262144,\"ms\":{:.4},\"err\":{:.3e}}}",
                    p,
                    oname,
                    mult,
                    ms,
                    rel_l2(&res, &refr, evs.len() as f64)
                );
                eprintln!("  done pad×{} {} p={}", mult, oname, p);
            }
        }
    }
}
