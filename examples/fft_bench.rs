use rand::RngExt;
use shrinkers::*;
use std::time::Instant;

fn make_evals(p: usize, c: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    let lm = (1.0 - c.sqrt()).max(0.01).powi(2);
    let l_m = (1.0 + c.sqrt()).powi(2);
    let mut v: Vec<f64> = (0..p)
        .map(|_| {
            let u: f64 = rng.random();
            let t = lm + u * (l_m - lm);
            t + rng.random::<f64>() * 0.1
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn main() {
    for p in [500usize, 1000, 2000, 5000] {
        let evals = make_evals(p, 0.5);
        let eta = 0.1 / (p as f64).sqrt();
        // warmup
        let _ = stieltjes::compute_all_stieltjes(
            &evals,
            eta,
            StieltjesMethod::Fft2,
            None,
            CutoffConfig::Disabled,
            64,
            Parallelism::Sequential,
        );
        let n = 20;
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = stieltjes::compute_all_stieltjes(
                &evals,
                eta,
                StieltjesMethod::Fft2,
                None,
                CutoffConfig::Disabled,
                64,
                Parallelism::Sequential,
            );
        }
        let dt = t0.elapsed().as_secs_f64() / n as f64 * 1e6;
        println!("p={}: fft2 pure Rust = {:.0} us", p, dt);
    }
}
