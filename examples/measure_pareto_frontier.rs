//! Dump benchmark data for the Pareto-frontier analysis (runtime × accuracy
//! × parallelism) as JSON on stdout.
//!
//! Stable across the optimization campaign: only uses the pre-existing
//! public API so the SAME script measures the "before" (git stash) and
//! "after" (working tree) states.
//!
//! Usage: cargo run --release --example measure_pareto_frontier -- after|before
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

/// MP-like spectrum (c=0.5 bulk + two spikes) — representative deconv input.
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
    f(); // warmup + correctness of closure
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
    let label = std::env::args().nth(1).unwrap_or_else(|| "after".into());

    // (name, method, cutoff) — names stable across the campaign.
    let methods: &[(&str, StieltjesMethod, CutoffConfig)] = &[
        (
            "autovec",
            StieltjesMethod::AutoVectorized,
            CutoffConfig::Disabled,
        ),
        ("blocked", StieltjesMethod::Blocked, CutoffConfig::Disabled),
        (
            "blocked_tiled",
            StieltjesMethod::BlockedTiled,
            CutoffConfig::Disabled,
        ),
        (
            "windowed_cut10",
            StieltjesMethod::BlockedWindowed,
            CutoffConfig::Enabled { ratio: 10.0 },
        ),
        (
            "adaptive",
            StieltjesMethod::Adaptive,
            CutoffConfig::Disabled,
        ),
        ("fft5", StieltjesMethod::Fft5, CutoffConfig::Disabled),
        (
            "chebcode",
            StieltjesMethod::ChebCode,
            CutoffConfig::Disabled,
        ),
        (
            "chebcode_fast",
            StieltjesMethod::ChebCodeFast,
            CutoffConfig::Disabled,
        ),
        (
            "chebcode_xtreme",
            StieltjesMethod::ChebCodeXtreme,
            CutoffConfig::Disabled,
        ),
        (
            "chebcode_balanced",
            StieltjesMethod::ChebCodeBalanced,
            CutoffConfig::Disabled,
        ),
        ("hodlr", StieltjesMethod::Hodlr, CutoffConfig::Disabled),
    ];

    println!(
        "{{\"meta\": {{\"label\": \"{}\", \"eta_rule\": \"1/sqrt(p)\", \"spectrum\": \"mp_c05_spikes\", \"error\": \"rel_l2_re_im_vs_exact\"}}, \"rows\": [",
        label
    );

    let mut first = true;
    for &p in P_SIZES {
        let evs = spectrum(p);
        // Benchmark convention (NOT the library default 0.1/sqrt(p)) — see
        // the Conventions list in src/stieltjes/mod.rs before changing.
        let eta = 1.0 / (p as f64).sqrt();

        // Exact reference (sequential tiled kernel).
        let refr = stieltjes::compute_all_stieltjes(
            &evs,
            eta,
            StieltjesMethod::BlockedTiled,
            None,
            CutoffConfig::Disabled,
            32,
            Parallelism::Sequential,
        );

        // Extra direct-call rows that bypass the dispatch enum.
        type ExtraRow<'a> = (&'a str, f64, Vec<(f64, f64)>);
        let mut extra_rows: Vec<ExtraRow> = Vec::new();
        for &(par_name, par) in &[("seq", false), ("parallel", true)] {
            let mut res = Vec::new();
            let ms = bench(|| {
                res = stieltjes::compute_all_stieltjes_hodlr_impl(
                    &evs,
                    eta,
                    256,
                    1e-6,
                    32,
                    par,
                    stieltjes::HodlrMode::Random,
                );
            });
            extra_rows.push((par_name, ms, res));
        }

        for &(name, method, cutoff) in methods {
            for &(par_name, par) in &[
                ("seq", Parallelism::Sequential),
                ("parallel", Parallelism::Parallel),
            ] {
                let mut res = Vec::new();
                let ms = bench(|| {
                    res =
                        stieltjes::compute_all_stieltjes(&evs, eta, method, None, cutoff, 32, par);
                });
                let num: f64 = res
                    .iter()
                    .zip(refr.iter())
                    .map(|((gr, gi), (rr, ri))| (gr - rr).powi(2) + (gi - ri).powi(2))
                    .sum();
                let den: f64 = refr.iter().map(|(rr, ri)| rr * rr + ri * ri).sum();
                let err = (num / den).sqrt();
                eprintln!("  done {} {} p={}", name, par_name, p);
                let comma = if first { "" } else { "," };
                first = false;
                println!(
                    "{}  {{\"method\": \"{}\", \"par\": \"{}\", \"p\": {}, \"ms\": {:.4}, \"err\": {:.3e}}}",
                    comma, name, par_name, p, ms, err
                );
            }
        }

        for (par_name, ms_rand, res) in extra_rows {
            // The raw-sum impl skips the dispatcher's 1/p scaling.
            let inv_p = 1.0 / p as f64;
            let scaled: Vec<(f64, f64)> = res.iter().map(|(a, b)| (a * inv_p, b * inv_p)).collect();
            let num: f64 = scaled
                .iter()
                .zip(refr.iter())
                .map(|((gr, gi), (rr, ri))| (gr - rr).powi(2) + (gi - ri).powi(2))
                .sum();
            let den: f64 = refr.iter().map(|(rr, ri)| rr * rr + ri * ri).sum();
            let err = (num / den).sqrt();
            eprintln!("  done hodlr_rand {} p={}", par_name, p);
            let comma = if first { "" } else { "," };
            first = false;
            println!(
                "{}  {{\"method\": \"hodlr_rand\", \"par\": \"{}\", \"p\": {}, \"ms\": {:.4}, \"err\": {:.3e}}}",
                comma, par_name, p, ms_rand, err
            );
        }
    }
    println!("]}}");
}
