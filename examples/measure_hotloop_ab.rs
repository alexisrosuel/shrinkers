//! A/B guard for the cache-blocked hot loops and the ChebCode tree build.
//!
//! Two modes, both machine-readable so two builds of this file can be
//! compared directly:
//!
//!   --checksum   `name<TAB>checksum`  — bit-level output identity
//!   (default)    `name<TAB>median_ms` — timing
//!
//! The modes are separate on purpose: folding the checksum into the timed
//! closure would add a p-element reduction to every repetition and distort the
//! cheap cases (the whole ChebCode tree build is ~0.5 ms at p=20k).
//!
//! Workflow for a kernel refactor: build this example from the pre-refactor
//! revision and from the post-refactor one, then
//!   1. diff the `--checksum` output — it must be IDENTICAL; and
//!   2. run the two binaries alternately for several rounds and compare the
//!      per-case medians (a single round is not evidence; the machine drifts).
//!
//! Cases (largest first, so a slow case cannot hide behind a warm cache):
//!   tiled_exact_p20k   compute_all_stieltjes_blocked_tiled, no cutoff
//!   tiled_cut_p20k     same, with a far-field cutoff (the guarded body)
//!   points_exact_p20k  compute_stieltjes_at_points, blocked family, no cutoff
//!   points_cut_p20k    same, with cutoff
//!   cheb_build_p20k    ChebPreset::DEFAULT tree build (fill + merge weights)
//!   cheb_eval_p20k     that tree evaluated at all p points
//!   ... and the same set at p=4k, where loop/branch overhead dominates.
//!
//! Usage:
//!   cargo run --release --example measure_hotloop_ab -- [filter] [--checksum]

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::config::{Parallelism, StieltjesMethod};
use shrinkers::stieltjes::{self, ChebPreset};
use std::time::Instant;
use support::{harness_spectrum, median};

/// Benchmark convention: η = 1/√p (see the Conventions list in
/// src/stieltjes/mod.rs). Not the library default 0.1/√p.
fn eta_for(p: usize) -> f64 {
    1.0 / (p as f64).sqrt()
}

const CUT_RATIO: f64 = 10.0;
const REPS: usize = 7;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let checksum_mode = args.iter().any(|a| a == "--checksum");
    let filter = args.iter().find(|a| !a.starts_with("--")).cloned();

    for &p in &[20_000usize, 4_000] {
        let evs = harness_spectrum(p);
        let eta = eta_for(p);

        // Query grid for the at-points cases: the deconvolution-style uniform
        // grid over the eigenvalue range (not the eigenvalues themselves).
        let (lo, hi) = (evs[0], evs[p - 1]);
        let grid: Vec<f64> = (0..p)
            .map(|k| lo + (hi - lo) * (k as f64) / (p as f64 - 1.0))
            .collect();

        let tag = |base: &str| format!("{base}_p{}k", p / 1000);

        // Each case: a closure returning a checksum of its result.
        let cases: Vec<(String, Box<dyn FnMut() -> f64>)> = vec![
            (
                tag("tiled_exact"),
                Box::new(|| {
                    let (r, i) =
                        stieltjes::compute_all_stieltjes_blocked_tiled(&evs, eta, None, None);
                    sum_pairs(&r, &i)
                }),
            ),
            (
                tag("tiled_cut"),
                Box::new(|| {
                    let (r, i) = stieltjes::compute_all_stieltjes_blocked_tiled(
                        &evs,
                        eta,
                        None,
                        Some(CUT_RATIO),
                    );
                    sum_pairs(&r, &i)
                }),
            ),
            // The parallel kernel drives the same `tiled_span` body over
            // disjoint output chunks, so a change there must be measured here
            // too (its remainder loop is what the p % 4 regression hit).
            (
                tag("tiled_par_exact"),
                Box::new(|| {
                    let (r, i) = stieltjes::compute_all_stieltjes_blocked_tiled_parallel(
                        &evs, eta, None, None,
                    );
                    sum_pairs(&r, &i)
                }),
            ),
            (
                tag("tiled_par_cut"),
                Box::new(|| {
                    let (r, i) = stieltjes::compute_all_stieltjes_blocked_tiled_parallel(
                        &evs,
                        eta,
                        None,
                        Some(CUT_RATIO),
                    );
                    sum_pairs(&r, &i)
                }),
            ),
            (
                tag("points_exact"),
                Box::new(|| {
                    let out = stieltjes::compute_stieltjes_at_points(
                        &grid,
                        &evs,
                        eta,
                        StieltjesMethod::Blocked,
                        None,
                        Parallelism::Sequential,
                        None,
                    );
                    sum_aos(&out)
                }),
            ),
            (
                tag("points_cut"),
                Box::new(|| {
                    let out = stieltjes::compute_stieltjes_at_points(
                        &grid,
                        &evs,
                        eta,
                        StieltjesMethod::Blocked,
                        Some(CUT_RATIO),
                        Parallelism::Sequential,
                        None,
                    );
                    sum_aos(&out)
                }),
            ),
            (
                tag("cheb_build"),
                Box::new(|| {
                    let tree = stieltjes::chebcode_tree_for_bench(&evs, ChebPreset::DEFAULT);
                    std::hint::black_box(tree);
                    // The build exposes no cheap scalar; its correctness is
                    // covered by cheb_eval's checksum.
                    0.0
                }),
            ),
            (
                tag("cheb_eval"),
                Box::new({
                    let tree = stieltjes::chebcode_tree_for_bench(&evs, ChebPreset::DEFAULT);
                    move || sum_aos(&tree.evaluate(eta))
                }),
            ),
        ];

        for (name, mut f) in cases {
            if filter
                .as_ref()
                .is_some_and(|fl| !name.contains(fl.as_str()))
            {
                continue;
            }
            if checksum_mode {
                println!("{name}\t{:.17e}", f());
            } else {
                f(); // warm-up
                let mut ts = Vec::with_capacity(REPS);
                for _ in 0..REPS {
                    let st = Instant::now();
                    std::hint::black_box(f());
                    ts.push(st.elapsed().as_secs_f64() * 1e3);
                }
                println!("{name}\t{:.4}", median(&mut ts));
            }
        }
    }
}

/// Order-independent checksum over SoA real/imag streams.
fn sum_pairs(re: &[f64], im: &[f64]) -> f64 {
    re.iter().sum::<f64>() + im.iter().map(|v| v * 3.0).sum::<f64>()
}

/// Order-independent checksum over AoS pairs (the imaginary part is weighted
/// so a real/imag lane swap cannot cancel out).
fn sum_aos(pairs: &[(f64, f64)]) -> f64 {
    pairs.iter().map(|(r, i)| r + i * 3.0).sum()
}
