//! End-to-end validation of the relaxed `chebcode_fast` preset.
//!
//! The preset now trades ~4 orders of magnitude of Stieltjes-transform
//! accuracy (~1e-8 -> ~1e-5..1e-3) for ~2.3x speed. This example checks that
//! the *product* is unaffected: the deconvolved bulk density and the recovered
//! spikes from `deconvolve_spiked` must stay close to the exact
//! (`StieltjesMethod::Blocked`) answer.
//!
//! Run: cargo run --release --example validate_fast_preset

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::{CutoffConfig, Parallelism, RmtConfig, StieltjesMethod};
use shrinkers::deconvolution::deconvolve_spiked;
use shrinkers::stieltjes::{ChebPreset, compute_all_stieltjes};
use support::harness_spectrum;

fn main() {
    let c = 0.25;
    for &p in &[2_000usize, 10_000, 50_000] {
        // `harness_spectrum` = MP bulk (c=0.5) + two isolated outliers, so
        // spike detection is exercised too.
        let evals = harness_spectrum(p);
        let eta = 0.1 / (p as f64).sqrt(); // library default (worst case)

        // Ground truth: exact O(p^2) transform through the same pipeline.
        let exact_cfg = RmtConfig::new(c).with_stieltjes(StieltjesMethod::Blocked);
        let exact = deconvolve_spiked(&evals, c, 200, Some(eta), 1.0, &exact_cfg);

        // The shipped speed preset.
        let fast_cfg = RmtConfig::new(c).with_stieltjes(StieltjesMethod::ChebCodeFast);
        let fast = deconvolve_spiked(&evals, c, 200, Some(eta), 1.0, &fast_cfg);

        // Density relative L2 on the shared grid.
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (a, b) in fast.bulk.density.iter().zip(exact.bulk.density.iter()) {
            num += (a - b) * (a - b);
            den += b * b;
        }
        let dens_err = (num / den).sqrt();

        // Spike count + worst relative spike error.
        let mut spike_err = 0.0f64;
        if fast.spikes.len() == exact.spikes.len() {
            for (a, b) in fast.spikes.iter().zip(exact.spikes.iter()) {
                spike_err = spike_err.max((a - b).abs() / b.abs().max(1e-12));
            }
        }

        // Raw transform error at the eigenvalues (rel L2), for context.
        let ex = compute_all_stieltjes(
            &evals,
            eta,
            StieltjesMethod::Blocked,
            None,
            CutoffConfig::Disabled,
            64,
            Parallelism::Sequential,
        );
        let got = compute_all_stieltjes(
            &evals,
            eta,
            StieltjesMethod::ChebCodeFast,
            None,
            CutoffConfig::Disabled,
            64,
            Parallelism::Sequential,
        );
        let (mut n2, mut d2) = (0.0f64, 0.0f64);
        for (a, b) in got.iter().zip(ex.iter()) {
            n2 += (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2);
            d2 += b.0.powi(2) + b.1.powi(2);
        }
        let transform_err = (n2 / d2).sqrt();

        let pr = ChebPreset::FAST;
        println!(
            "p={p:<6} preset=(theta {:.2}, n {}, leaf {}, {:?}) | transform rel-L2 {:.2e} | bulk density rel-L2 {:.2e} | spikes exact={} fast={} worst rel diff {:.2e}",
            pr.theta,
            pr.n,
            pr.leaf_cap,
            pr.mode,
            transform_err,
            dens_err,
            exact.spikes.len(),
            fast.spikes.len(),
            spike_err,
        );
    }
}
