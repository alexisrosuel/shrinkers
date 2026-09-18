//! Accuracy/speed landscape of the `fft5` grid convolution as a function of
//! the grid-transfer order (linear/cubic/quintic/heptic), grid size and
//! padding. Dumps JSON lines on stdout for `scripts/analyze_order_sweep.py`.
//!
//! Usage:
//!   cargo run --release --example measure_fft_order_sweep > docs/pareto/order_sweep.jsonl
//!
//! Two experiments, both against the exact O(p²) sequential reference:
//!   A. "grid"  — error & runtime vs forced grid size m, one series per
//!                transfer order (reveals the empirical order of accuracy
//!                and where each series hits the wrap-around floor);
//!   B. "pad"   — error vs the kernel-tail padding multiplier
//!                (pad = pad_mult·η) at a grid large enough that the
//!                transfer error is negligible (reveals how the periodization
//!                floor scales with the image-pole distance).
#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::{CutoffConfig, Parallelism, StieltjesMethod};
use shrinkers::stieltjes::fft5::{Fft5Options, Order};
use support::{bench_ms, harness_spectrum, rel_l2_scaled};

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
        let evs = harness_spectrum(p);
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
            let ms = bench_ms(|| {
                res = shrinkers::stieltjes::fft5::compute_all_stieltjes_fft5_with_options(
                    &evs, eta, &opts,
                )
            });
            println!(
                "{{\"kind\":\"grid\",\"p\":{},\"order\":\"{}\",\"m\":null,\"ms\":{:.4},\"err\":{:.3e}}}",
                p,
                oname,
                ms,
                rel_l2_scaled(&res, &refr, evs.len())
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
                let ms = bench_ms(|| {
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
                    rel_l2_scaled(&res, &refr, evs.len())
                );
                eprintln!("  done {} m={} p={}", oname, m, p);
            }
        }
    }

    eprintln!("# experiment B: padding sweep (floor vs image-pole distance)");
    for &p in &[5000usize] {
        let evs = harness_spectrum(p);
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
                let ms = bench_ms(|| {
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
                    rel_l2_scaled(&res, &refr, evs.len())
                );
                eprintln!("  done pad×{} {} p={}", mult, oname, p);
            }
        }
    }
}
