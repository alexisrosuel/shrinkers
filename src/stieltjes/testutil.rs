//! Shared test scaffolding for the stieltjes test modules.
//!
//! Every helper here replaced a verbatim copy that used to live inside an
//! individual module's `#[cfg(test)]` block; the recipes themselves are
//! unchanged so existing tolerance-based assertions keep their meaning.

/// The canonical synthetic spectrum of the stieltjes test-suite: sorted
/// log-spaced eigenvalues `ln(1), ln(2), …, ln(p)` spanning several orders
/// of magnitude (stresses both near-field clustering and far-field decay).
pub(crate) fn log_spectrum(p: usize) -> Vec<f64> {
    (0..p).map(|i| (i as f64 + 1.0).ln()).collect()
}

/// Brute-force O(p²) Stieltjes transform reference in plain f64 loops —
/// the definition every approximate kernel is checked against.
pub(crate) fn exact_stieltjes(evals: &[f64], eta: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(evals.len());
    for &li in evals {
        let mut sr = 0.0;
        let mut si = 0.0;
        for &lj in evals {
            let d = li - lj;
            let denom = d * d + eta * eta;
            sr += d / denom;
            si += eta / denom;
        }
        out.push((sr, si));
    }
    out
}
