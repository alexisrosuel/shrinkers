//! Shared helpers for the criterion benchmarks **and** the measurement
//! examples.
//!
//! Benches include this via `mod support;`, examples via
//! `#[path = "../benches/support/mod.rs"] mod support;`. Everything here is
//! `pub(crate)`, so each target compiles only into itself.
#![allow(dead_code)] // every includer uses a different subset of these helpers

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

/// Deterministic 64-bit LCG yielding uniforms in `[0, 1)`.
///
/// No external RNG dependency and identical values on every platform, which
/// is what makes the recorded harness spectra reproducible.
pub(crate) struct Lcg(u64);

impl Lcg {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }
}

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

/// The recorded benchmark-harness spectrum (`"mp_c05_spikes"`).
///
/// Uniform bulk on `[(1−√c)², (1+√c)²]` with c = 0.5 plus two outliers
/// (`λmax·2.3`, `λmin·0.35`), sorted ascending. For `p < 3` there is no room
/// for the outliers, so a pure bulk is returned.
///
/// This is the exact construction declared in the `docs/pareto/*.json` meta,
/// so every consumer stays comparable with the recorded tables. Do not change
/// it without re-benchmarking and regenerating those tables.
pub(crate) fn harness_spectrum(p: usize) -> Vec<f64> {
    let c: f64 = 0.5;
    let lo = (1.0 - c.sqrt()).powi(2);
    let hi = (1.0 + c.sqrt()).powi(2);
    if p < 3 {
        return Lcg::new(42).take(p).map(|x| lo + x * (hi - lo)).collect();
    }
    let mut v: Vec<f64> = Lcg::new(42)
        .take(p - 2)
        .map(|x| lo + x * (hi - lo))
        .collect();
    v.push(hi * 2.3);
    v.push(lo * 0.35);
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

/// Median of a slice (sorts it in place).
pub(crate) fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// Time one closure in milliseconds: one warm-up call, then a median over
/// 3–15 repetitions chosen from a probe run.
pub(crate) fn bench_ms<F: FnMut()>(mut f: F) -> f64 {
    f(); // warm-up
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

/// Relative L2 error of raw (unscaled) Stieltjes sums against a reference.
pub(crate) fn rel_l2(got: &[(f64, f64)], exact: &[(f64, f64)]) -> f64 {
    let num: f64 = got
        .iter()
        .zip(exact)
        .map(|((gr, gi), (rr, ri))| (gr - rr).powi(2) + (gi - ri).powi(2))
        .sum();
    let den: f64 = exact.iter().map(|(rr, ri)| rr * rr + ri * ri).sum();
    (num / den).sqrt()
}

/// Relative L2 error after scaling `got` by `1/p` — for kernels that return
/// **raw sums** while the reference is already averaged.
pub(crate) fn rel_l2_scaled(got: &[(f64, f64)], exact: &[(f64, f64)], p: usize) -> f64 {
    let inv_p = 1.0 / p as f64;
    let num: f64 = got
        .iter()
        .zip(exact)
        .map(|((gr, gi), (rr, ri))| (gr * inv_p - rr).powi(2) + (gi * inv_p - ri).powi(2))
        .sum();
    let den: f64 = exact.iter().map(|(rr, ri)| rr * rr + ri * ri).sum();
    (num / den).sqrt()
}
