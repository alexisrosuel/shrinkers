//! Quick single-method benchmark for A/B iteration.
//!
//! Usage: cargo run --release --example ab_quick_timing -- <method> [extra args]
//!   methods: cheb <theta> <n> <leaf> | treecode | tiled | autovec
//!
//! Prints JSON rows compatible with scripts/build_pareto_table.py inputs.
use shrinkers::config::{CutoffConfig, Parallelism, StieltjesMethod};
use shrinkers::stieltjes;
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
    f();
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

const P_SIZES: &[usize] = &[1000, 2000, 5000, 10000, 20000, 50000];

fn main() {
    let what = std::env::args().nth(1).unwrap_or_else(|| "cheb".into());

    for &p in P_SIZES {
        let evs = spectrum(p);
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
                    bench(|| {
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
                    bench(|| {
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
                "treecode" => bench(|| {
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
                "tiled" => bench(|| {
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
            let inv_p = 1.0 / p as f64;
            let num: f64 = res
                .iter()
                .zip(refr.iter())
                .map(|((gr, gi), (rr, ri))| (gr * inv_p - rr).powi(2) + (gi * inv_p - ri).powi(2))
                .sum();
            let den: f64 = refr.iter().map(|(rr, ri)| rr * rr + ri * ri).sum();
            println!(
                "{{\"method\":\"{}\",\"par\":\"{}\",\"p\":{},\"ms\":{:.4},\"err\":{:.3e}}}",
                what,
                par_name,
                p,
                ms,
                (num / den).sqrt()
            );
            eprintln!("  done {} {} p={}", what, par_name, p);
        }
    }
}
