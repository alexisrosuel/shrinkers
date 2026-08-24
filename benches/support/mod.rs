//! Shared helpers for the criterion benchmarks.
//!
//! Included by each benchmark crate root via `#[path = "bench_support.rs"]`.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Deterministic Marchenko–Pastur-like spectrum: uniform bulk on
/// `[λmin(c), λmax(c)]` plus small jitter, sorted ascending.
///
/// Seeded, so every run of every benchmark feeds identical input —
/// comparisons across commits stay apples-to-apples.
pub(crate) fn mp_spectrum(p: usize, c: f64, seed: u64) -> Vec<f64> {
    let lam_min = (1.0 - c.sqrt()).max(0.01).powi(2);
    let lam_max = (1.0 + c.sqrt()).powi(2);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<f64> = (0..p)
        .map(|_| {
            let t = lam_min + rng.random::<f64>() * (lam_max - lam_min);
            t + rng.random::<f64>() * 0.1
        })
        .collect();
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    v
}
