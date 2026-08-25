//! Diagnostic: ChebCodeFast tree-build share vs evaluation at scale.
//! Run: cargo run --release --example measure_build_share

#[path = "../benches/support/mod.rs"]
mod support;

use shrinkers::stieltjes::ChebPreset;
use shrinkers::stieltjes::chebcode_tree_for_bench;
use std::time::Instant;

fn main() {
    let c = 0.25;
    for &p in &[20_000usize, 50_000] {
        let lam = support::mp_spectrum(p, c, p as u64);
        let eta = 1.0 / (p as f64).sqrt();
        let mut tb = Vec::new();
        let mut te = Vec::new();
        for rep in 0..11 {
            let t = Instant::now();
            let tree = chebcode_tree_for_bench(&lam, ChebPreset::FAST);
            let b = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let _ = tree.evaluate_points(&lam, eta, false);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if rep >= 1 {
                tb.push(b);
                te.push(e);
            }
        }
        tb.sort_by(|a, b| a.partial_cmp(b).unwrap());
        te.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (b, e) = (tb[5], te[5]);
        println!(
            "p={p:<6} build {b:>6.2} ms ({:.0}%) | eval {e:>6.2} ms | total {:.2}",
            b / (b + e) * 100.0,
            b + e
        );
    }
}
