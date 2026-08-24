//! Locate the O(p^2) vs ChebCode crossover for small problem sizes.
//!
//! The main Pareto sweep (`measure_pareto_frontier.rs`) starts at p=1000; below that the
//! per-call times drop into the microsecond range where single-shot timing is
//! mostly scheduler noise. This example measures log-spaced sizes from p=1 to
//! p=1000 with batched repetition (each timed batch lasts a few milliseconds,
//! the reported value is the median batch divided by its repetition count).
//!
//! Accuracy is still recorded (rel L2 vs the sequential BlockedTiled kernel)
//! but the point here is runtime: where does building+querying the tree stop
//! paying for itself compared to the plain quadratic sum?
//!
//! Usage: cargo run --release --example measure_small_p_crossover > docs/pareto/small_p.json
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

/// Same spectrum shape as `measure_pareto_frontier.rs` (MP c=0.5 bulk + two spikes) but
/// total-count safe down to p=1, where the spike bookkeeping would underflow.
fn spectrum(p: usize) -> Vec<f64> {
    let c: f64 = 0.5;
    let lo = (1.0 - c.sqrt()).powi(2);
    let hi = (1.0 + c.sqrt()).powi(2);
    if p < 3 {
        return Lcg(42).take(p).map(|x| lo + x * (hi - lo)).collect();
    }
    let bulk: Vec<f64> = Lcg(42).take(p - 2).map(|x| lo + x * (hi - lo)).collect();
    let mut v = bulk;
    v.push(hi * 2.3);
    v.push(lo * 0.35);
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

/// Per-call time in microseconds, noise-resistant at sub-10us scale:
/// calibrate an in-batch repetition count so one batch runs ~5 ms, then take
/// the median of several such batches.
fn bench_us<F: FnMut()>(mut f: F) -> f64 {
    f(); // warmup (also warms allocator / rayon pools)
    let mut reps = 1usize;
    let mut batch_ms;
    loop {
        let st = Instant::now();
        for _ in 0..reps {
            f();
        }
        batch_ms = st.elapsed().as_secs_f64() * 1e3;
        if batch_ms >= 2.0 {
            break;
        }
        let grown = (reps as f64 * (2.5 / batch_ms.max(0.002))).ceil() as usize;
        reps = grown.min(1 << 26).max(reps + 1);
    }
    let target = ((5.0 / batch_ms) * reps as f64).ceil() as usize;
    let mut samples = Vec::with_capacity(9);
    for _ in 0..9 {
        let st = Instant::now();
        for _ in 0..target {
            f();
        }
        samples.push(st.elapsed().as_secs_f64() * 1e6 / target as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

const P_SIZES: &[usize] = &[
    1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 750, 1000,
];

fn main() {
    println!(
        "{{\"meta\": {{\"label\": \"small_p\", \"eta_rule\": \"1/sqrt(p)\", \
         \"spectrum\": \"mp_c05_spikes\", \"error\": \"rel_l2_re_im_vs_exact_seq\", \
         \"unit\": \"us_per_call_median_of_9_batches\"}}, \"rows\": ["
    );

    let methods: &[(&str, &str, StieltjesMethod, Parallelism)] = &[
        (
            "exact",
            "seq",
            StieltjesMethod::BlockedTiled,
            Parallelism::Sequential,
        ),
        (
            "exact",
            "ray",
            StieltjesMethod::BlockedTiled,
            Parallelism::Rayon,
        ),
        (
            "cheb_fast",
            "seq",
            StieltjesMethod::ChebCodeFast,
            Parallelism::Sequential,
        ),
        (
            "cheb_fast",
            "ray",
            StieltjesMethod::ChebCodeFast,
            Parallelism::Rayon,
        ),
        (
            "cheb_default",
            "seq",
            StieltjesMethod::ChebCode,
            Parallelism::Sequential,
        ),
        (
            "cheb_default",
            "ray",
            StieltjesMethod::ChebCode,
            Parallelism::Rayon,
        ),
        (
            "cheb_xtreme",
            "seq",
            StieltjesMethod::ChebCodeXtreme,
            Parallelism::Sequential,
        ),
        (
            "cheb_xtreme",
            "ray",
            StieltjesMethod::ChebCodeXtreme,
            Parallelism::Rayon,
        ),
    ];

    let mut first = true;
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

        for &(name, par_name, method, parallelism) in methods {
            let us = bench_us(|| {
                let _ = stieltjes::compute_all_stieltjes(
                    &evs,
                    eta,
                    method,
                    None,
                    CutoffConfig::Disabled,
                    32,
                    parallelism,
                );
            });
            let res = stieltjes::compute_all_stieltjes(
                &evs,
                eta,
                method,
                None,
                CutoffConfig::Disabled,
                32,
                parallelism,
            );
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
                "{}  {{\"method\": \"{}\", \"par\": \"{}\", \"p\": {}, \"us\": {:.3}, \"err\": {:.3e}}}",
                comma, name, par_name, p, us, err
            );
        }
    }
    println!("]}}");
}
