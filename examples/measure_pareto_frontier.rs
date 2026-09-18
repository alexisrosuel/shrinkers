//! Dump benchmark data for the Pareto-frontier analysis (runtime × accuracy
//! × parallelism) as JSON on stdout.
//!
//! Stable across the optimization campaign: only uses the pre-existing
//! public API so the SAME script measures the "before" (git stash) and
//! "after" (working tree) states.
//!
//! Usage: cargo run --release --example measure_pareto_frontier -- after|before

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::{CutoffConfig, Parallelism, StieltjesMethod};
use shrinkers::stieltjes;
use support::{bench_ms, harness_spectrum, rel_l2, rel_l2_scaled};

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
        let evs = harness_spectrum(p);
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
            let ms = bench_ms(|| {
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
                let ms = bench_ms(|| {
                    res =
                        stieltjes::compute_all_stieltjes(&evs, eta, method, None, cutoff, 32, par);
                });
                let err = rel_l2(&res, &refr);
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
            let err = rel_l2_scaled(&res, &refr, p);
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
