//! Runtime audit of the Stieltjes / deconvolution entry points.
//!
//! Reproduces the measurements quoted in the CHANGELOG for the two kernel
//! optimizations (the exact symmetric sweep's fused accumulations and the
//! query-count-aware treecode leaf capacity) plus the at-points dispatch fix.
//!
//! Run one section at a time; the first positional argument selects it and
//! the second is `p` (default 10 000):
//!
//! ```text
//! cargo run --release --example measure_runtime_audit -- allpts 10000
//! cargo run --release --example measure_runtime_audit -- grid   50000
//! cargo run --release --example measure_runtime_audit -- build  50000
//! cargo run --release --example measure_runtime_audit -- top    50000
//! cargo run --release --example measure_runtime_audit -- par    10000
//! cargo run --release --example measure_runtime_audit -- ab     50000
//! ```
//!
//! Sections: `allpts` (all-points kernels), `grid` (nq = 200 query points),
//! `build` (treecode build / grid / all-points split), `top` (the high-level
//! deconvolution entry points), `par` (Rayon all-points), `ab` (interleaved
//! `fft5` vs `chebcode_fast` A/B), `compare` (one `KEY<TAB>ms` line per
//! headline metric, for before/after binary diffing), `all` (everything).
//!
//! # Before/after comparison
//!
//! `compare` prints stable machine-readable keys so two binaries built from
//! different revisions can be interleaved and diffed:
//!
//! ```text
//! git stash push -- src/ && cargo build --release --example measure_runtime_audit
//! cp target/release/examples/measure_runtime_audit /tmp/audit_before
//! git stash pop && cargo build --release --example measure_runtime_audit
//! cp target/release/examples/measure_runtime_audit /tmp/audit_after
//! for i in 1 2 3; do
//!   /tmp/audit_before compare 10000; /tmp/audit_after compare 10000
//! done
//! ```

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::{CutoffConfig, Parallelism, RmtConfig, StieltjesMethod};
use shrinkers::deconvolution::{deconvolve_spiked, spectral_deconvolution};
use shrinkers::stieltjes::{compute_all_stieltjes, compute_stieltjes_at_points};
use std::time::Instant;

/// Median of `reps` timings, discarding the first (warm-up) run.
fn bench(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut ts = Vec::with_capacity(reps);
    for r in 0..=reps {
        let t = Instant::now();
        f();
        let dt = t.elapsed().as_secs_f64() * 1e3;
        if r > 0 {
            ts.push(dt);
        }
    }
    median(ts)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let p: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let c = 0.25;
    let evals = support::mp_spectrum(p, c, 7);
    let eta = 0.1 / (p as f64).sqrt();
    let label = format!("p={p}");
    // A 200-point deconvolution grid over the spectrum plus the usual margin.
    let nq = 200usize;
    let range = evals[p - 1] - evals[0];
    let lo = (evals[0] - 0.2 * range).max(0.0);
    let hi = evals[p - 1] + 0.2 * range;
    let grid: Vec<f64> = (0..nq)
        .map(|k| lo + (hi - lo) * k as f64 / (nq as f64 - 1.0))
        .collect();

    if which == "compare" {
        // One machine-readable line per headline metric: `KEY<TAB>ms`.
        // Reps are scaled so the whole section stays a few seconds.
        let big = p >= 40_000;
        let reps_exact = if big { 5 } else { 15 };

        // 1. Exact all-points kernel (sequential) — the fused-accumulation win.
        let t = bench(reps_exact, || {
            let _ = compute_all_stieltjes(
                &evals,
                eta,
                StieltjesMethod::Blocked,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Sequential,
            );
        });
        println!("allpts.blocked.seq.p{p}\t{t:.4}");

        // 1b. The same call through the `auto` dispatch, which is what the
        // Python `stieltjes_transform(method="auto")` binding uses. It used
        // to fall through to the exact `Blocked` kernel.
        let t = bench(reps_exact, || {
            let _ = compute_all_stieltjes(
                &evals,
                eta,
                StieltjesMethod::Auto,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Sequential,
            );
        });
        println!("allpts.auto.seq.p{p}\t{t:.4}");

        // 2. Rayon all-points exact — inherits the same win per pair.
        let t = bench(11, || {
            let _ = compute_all_stieltjes(
                &evals,
                eta,
                StieltjesMethod::Blocked,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Parallel,
            );
        });
        println!("allpts.blocked.par.p{p}\t{t:.4}");

        // 3. Treecode on the 200-point grid — the leaf-capacity win.
        for m in [
            StieltjesMethod::ChebCodeFast,
            StieltjesMethod::ChebCodeBalanced,
        ] {
            let t = bench(21, || {
                let _ = compute_stieltjes_at_points(
                    &grid,
                    &evals,
                    eta,
                    m,
                    None,
                    Parallelism::Sequential,
                    None,
                );
            });
            println!("grid.{}.nq{nq}.p{p}\t{t:.4}", m.name());
        }

        // 4. Python-default deconvolution path — the at-points dispatch fix.
        let auto = RmtConfig::new(c).with_stieltjes(StieltjesMethod::Auto);
        let t = bench(21, || {
            let _ = deconvolve_spiked(&evals, c, nq, Some(eta), 1.0, &auto);
        });
        println!("deconvolve_spiked.auto.nq{nq}.p{p}\t{t:.4}");

        // 5. Reference: the exact method on the same grid, for context.
        let t = bench(21, || {
            let _ = compute_stieltjes_at_points(
                &grid,
                &evals,
                eta,
                StieltjesMethod::Blocked,
                None,
                Parallelism::Sequential,
                None,
            );
        });
        println!("grid.blocked.nq{nq}.p{p}\t{t:.4}");
        return;
    }

    if which == "readme_par" {
        // Reproduces the README "It uses every core you give it" table: the
        // recorded `harness_spectrum` at eta = 1/sqrt(p), exact `blocked`
        // sequential vs Rayon. Prints `KEY<TAB>ms` for before/after diffing.
        let ev = support::harness_spectrum(p);
        let eta = 1.0 / (p as f64).sqrt();
        for (tag, par) in [
            ("seq", Parallelism::Sequential),
            ("par", Parallelism::Parallel),
        ] {
            let t = bench(11, || {
                let _ = compute_all_stieltjes(
                    &ev,
                    eta,
                    StieltjesMethod::Blocked,
                    None,
                    CutoffConfig::Disabled,
                    64,
                    par,
                );
            });
            println!("readme.exact.{tag}.p{p}\t{t:.4}");
        }
        return;
    }

    if which == "all" || which == "allpts" {
        eprintln!("== all-points Stieltjes ({label}) ==");
        for m in [
            StieltjesMethod::Blocked,
            StieltjesMethod::BlockedTiled,
            StieltjesMethod::AutoVectorized,
            StieltjesMethod::Fft5,
            StieltjesMethod::ChebCodeFast,
            StieltjesMethod::ChebCodeBalanced,
        ] {
            let t = bench(21, || {
                let _ = compute_all_stieltjes(
                    &evals,
                    eta,
                    m,
                    None,
                    CutoffConfig::Disabled,
                    64,
                    Parallelism::Sequential,
                );
            });
            println!("  seq  {:<20} {:>10.3} ms", m.name(), t);
        }
    }

    if which == "all" || which == "grid" {
        eprintln!("== at-points Stieltjes, nq={nq} ({label}) ==");
        for m in [
            StieltjesMethod::Blocked,
            StieltjesMethod::Fft5,
            StieltjesMethod::ChebCodeFast,
            StieltjesMethod::ChebCodeBalanced,
        ] {
            let t = bench(21, || {
                let _ = compute_stieltjes_at_points(
                    &grid,
                    &evals,
                    eta,
                    m,
                    None,
                    Parallelism::Sequential,
                    None,
                );
            });
            println!("  grid {:<20} {:>10.4} ms", m.name(), t);
        }
    }

    if which == "all" || which == "build" {
        use shrinkers::stieltjes::{ChebCodeBatch, ChebPreset};
        eprintln!("== ChebCode build / eval split ({label}, nq={nq}) ==");
        for (name, preset) in [
            ("fast", ChebPreset::FAST),
            ("balanced", ChebPreset::BALANCED),
            ("default", ChebPreset::DEFAULT),
        ] {
            let tb = bench(21, || {
                let _ = ChebCodeBatch::build_preset(&evals, preset);
            });
            let batch = ChebCodeBatch::build_preset(&evals, preset);
            let tg = bench(21, || {
                let _ = batch.evaluate_points(&grid, eta, false);
            });
            let ta = bench(11, || {
                let _ = batch.evaluate(eta);
            });
            // Fill-only cost: one leaf, so every source is interpolated but
            // no parent weights are merged.
            let tf = bench(21, || {
                let _ = ChebCodeBatch::build(&evals, preset.theta, preset.n, evals.len());
            });
            println!(
                "  {name:<9} build {:>8.4} ms | grid{nq} {:>8.4} ms | all-p {:>8.3} ms | fills-only {:>8.4} ms",
                tb, tg, ta, tf
            );
        }
    }

    if which == "all" || which == "top" {
        eprintln!("== high-level deconvolution ({label}, nq={nq}) ==");
        // Python defaults: method="auto" (speed preset), sequential.
        let auto = RmtConfig::new(c).with_stieltjes(StieltjesMethod::Auto);
        let t = bench(21, || {
            let _ = deconvolve_spiked(&evals, c, nq, Some(eta), 1.0, &auto);
        });
        println!("  deconvolve_spiked(auto)       {:>10.4} ms", t);
        let t = bench(21, || {
            let _ = spectral_deconvolution(&evals, c, nq, Some(eta), None, None, &auto);
        });
        println!("  spectral_deconvolution(auto)  {:>10.4} ms", t);
        let blocked = RmtConfig::new(c);
        let t = bench(21, || {
            let _ = deconvolve_spiked(&evals, c, nq, Some(eta), 1.0, &blocked);
        });
        println!("  deconvolve_spiked(blocked)    {:>10.4} ms", t);
    }

    if which == "all" || which == "par" {
        eprintln!(
            "== parallel all-points ({label}, {} threads) ==",
            rayon::current_num_threads()
        );
        for m in [
            StieltjesMethod::Blocked,
            StieltjesMethod::BlockedTiled,
            StieltjesMethod::ChebCodeFast,
        ] {
            let t = bench(11, || {
                let _ = compute_all_stieltjes(
                    &evals,
                    eta,
                    m,
                    None,
                    CutoffConfig::Disabled,
                    64,
                    Parallelism::Parallel,
                );
            });
            println!("  par  {:<20} {:>10.3} ms", m.name(), t);
        }
    }

    if which == "all" || which == "ab" {
        // Interleaved A/B (alternating order): the all-points speed pick for
        // p > 20 000 is fft5, but the FFT pays for the whole grid, so it is
        // the wrong choice on a 200-point deconvolution grid.
        let a = StieltjesMethod::Fft5;
        let b = StieltjesMethod::ChebCodeFast;
        let (mut ta, mut tb) = (Vec::new(), Vec::new());
        for r in 0..13 {
            let order = if r % 2 == 0 { [a, b] } else { [b, a] };
            for m in order {
                let t = Instant::now();
                let _ = compute_all_stieltjes(
                    &evals,
                    eta,
                    m,
                    None,
                    CutoffConfig::Disabled,
                    64,
                    Parallelism::Sequential,
                );
                let dt = t.elapsed().as_secs_f64() * 1e3;
                if r > 0 {
                    if m == a {
                        ta.push(dt);
                    } else {
                        tb.push(dt);
                    }
                }
            }
        }
        println!(
            "  AB all-points  {:<14} {:>9.3} ms | {:<14} {:>9.3} ms",
            a.name(),
            median(ta),
            b.name(),
            median(tb)
        );
    }
}
