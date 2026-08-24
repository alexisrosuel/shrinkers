//! Long-running loop for `sample`-based profiling of the ChebCode path.
//!
//! Alternates tree build and sequential evaluation on the DEFAULT preset at
//! p=50 000 so the profiler sees both phases. Prints its own PID first —
//! backgrounding a compound command makes `$!` the shell's PID, not this
//! binary's (see the profiling playbook in CHANGELOG).
//!
//! Usage: cargo run --release --example profile_hot_loop
use shrinkers::stieltjes::{ChebPreset, chebcode_tree_for_bench};

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
    let mut v: Vec<f64> = Lcg(42).take(p).map(|x| lo + x * (hi - lo)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn main() {
    println!("PID={}", std::process::id());
    let evs = spectrum(50_000);
    let eta = 1.0 / (50_000_f64).sqrt();
    // Warmup outside timing.
    let warm = chebcode_tree_for_bench(&evs, ChebPreset::DEFAULT);
    drop(warm.evaluate(eta));

    let mut sink = 0.0_f64;
    for _ in 0..2000 {
        let batch = chebcode_tree_for_bench(&evs, ChebPreset::DEFAULT);
        let res = batch.evaluate(eta);
        sink += res[1234].0;
        if sink.is_infinite() {
            println!("{sink}");
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    println!("done {sink}");
}
