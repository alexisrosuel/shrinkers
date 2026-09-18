//! Quick single-method benchmark for A/B iteration.
//!
//! Usage: cargo run --release --example ab_quick_timing -- <method> [extra args]
//!   methods: cheb <theta> <n> <leaf> | treecode | tiled | autovec
//!
//! Prints JSON rows compatible with scripts/build_pareto_table.py inputs.

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::{CutoffConfig, Parallelism, StieltjesMethod};
use shrinkers::stieltjes;
use support::{bench_ms, harness_spectrum, rel_l2_scaled};

const P_SIZES: &[usize] = &[1000, 2000, 5000, 10000, 20000, 50000];

fn main() {
    let what = std::env::args().nth(1).unwrap_or_else(|| "cheb".into());

    for &p in P_SIZES {
        let evs = harness_spectrum(p);
        let eta = 1.0 / (p as f64).sqrt();

        let refr = stieltjes::compute_all_stieltjes(
            &evs,
            eta,
            StieltjesMethod::BlockedTiled,
            None,
            CutoffConfig::Disabled,
            32,
            Parallelism::Sequential,
        );

        for &(par_name, par) in &[
            ("seq", Parallelism::Sequential),
            ("rayon", Parallelism::Parallel),
        ] {
            let mut res = Vec::new();
            let ms = match what.as_str() {
                "cheb" => {
                    // optional: theta n leaf — defaults are the measured
                    // ChebPreset::DEFAULT, not hand-copied numbers.
                    let preset = shrinkers::stieltjes::ChebPreset::DEFAULT;
                    let theta: f64 = std::env::args()
                        .nth(2)
                        .map(|s| s.parse().unwrap())
                        .unwrap_or(preset.theta);
                    let n: usize = std::env::args()
                        .nth(3)
                        .map(|s| s.parse().unwrap())
                        .unwrap_or(preset.n);
                    let leaf: usize = std::env::args()
                        .nth(4)
                        .map(|s| s.parse().unwrap())
                        .unwrap_or(preset.leaf_cap);
                    bench_ms(|| {
                        res = shrinkers::stieltjes::compute_all_stieltjes_chebcode_impl(
                            &evs,
                            eta,
                            theta,
                            n,
                            leaf,
                            matches!(par, Parallelism::Parallel),
                        );
                    })
                }
                "hodlr" => {
                    let leaf: usize = std::env::args()
                        .nth(2)
                        .map(|s| s.parse().unwrap())
                        .unwrap_or(256);
                    let tol: f64 = std::env::args()
                        .nth(3)
                        .map(|s| s.parse().unwrap())
                        .unwrap_or(1e-9);
                    let max_rank: usize = std::env::args()
                        .nth(4)
                        .map(|s| s.parse().unwrap())
                        .unwrap_or(32);
                    let mode = std::env::args()
                        .nth(5)
                        .unwrap_or_else(|| "rand".to_string());
                    bench_ms(|| {
                        res = shrinkers::stieltjes::compute_all_stieltjes_hodlr_impl(
                            &evs,
                            eta,
                            leaf,
                            tol,
                            max_rank,
                            matches!(par, Parallelism::Parallel),
                            if mode == "aca" {
                                shrinkers::stieltjes::HodlrMode::Aca
                            } else {
                                shrinkers::stieltjes::HodlrMode::Random
                            },
                        );
                    })
                }
                "treecode" => bench_ms(|| {
                    res = stieltjes::compute_all_stieltjes(
                        &evs,
                        eta,
                        StieltjesMethod::TreeCode,
                        None,
                        CutoffConfig::Disabled,
                        0,
                        par,
                    );
                }),
                "tiled" => bench_ms(|| {
                    // Raw kernels: this tool scales by inv_p itself.
                    let (reals, imags) = if matches!(par, Parallelism::Parallel) {
                        stieltjes::compute_all_stieltjes_blocked_tiled_parallel(
                            &evs, eta, None, None,
                        )
                    } else {
                        stieltjes::compute_all_stieltjes_blocked_tiled(&evs, eta, None, None)
                    };
                    res = reals.into_iter().zip(imags).collect();
                }),
                other => panic!("unknown method: {other}"),
            };
            let err = rel_l2_scaled(&res, &refr, p);
            println!(
                "{{\"method\":\"{}\",\"par\":\"{}\",\"p\":{},\"ms\":{:.4},\"err\":{:.3e}}}",
                what, par_name, p, ms, err
            );
            eprintln!("  done {} {} p={}", what, par_name, p);
        }
    }
}
