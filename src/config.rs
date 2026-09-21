//! Configuration system for the RMT Shrinkage Kernel.
//!
//! Defines all optimization toggles so we can benchmark 2^N combinations
//! and identify the best configuration for each problem size.

pub(crate) mod pareto_autogen;

/// Method for computing the Stieltjes transform.
///
/// Families, in rough accuracy order:
/// - **exact O(p²)**: `Blocked*` variants — bit-stable zero-error anchor;
/// - **FFT grid**: `Fft5/Fft3/Fft2`, `Adaptive`, `Ewald`, `Dst` —
///   O(p log p) but floor ~1e-4..1e-5 (dominated by ChebCode today);
/// - **treecodes**: `TreeCode`, then the `ChebCode*` preset family —
///   the speed-at-accuracy frontier;
/// - **meta**: `Auto`, `SpeedAuto`, `AccuracyAuto` resolve to a concrete
///   method before dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StieltjesMethod {
    /// O(p²) naive scalar loop — no SIMD, no parallel
    Naive,
    /// O(p²) auto-vectorized loop — LLVM auto-vectorizes (NEON/AVX2)
    AutoVectorized,
    /// O(p²) cache-blocked + loop-unrolled + FMA + far-field cutoff
    Blocked,
    /// O(p²) cache-blocked + auto-vectorized (λᵢ-outer local accumulators,
    /// binary-search cutoff window → branch-free SIMD reduction)
    BlockedAutoVec,
    /// O(p²) 2D-tiled cache-blocked (output block outer → stays in cache
    /// across all source sweeps, minimizing cache invalidation)
    BlockedTiled,
    /// O(p·k) cache-blocked + binary-search far-field window (skips far-field
    /// iterations entirely instead of just skipping writes)
    BlockedWindowed,
    /// Hybrid: real part via the exact blocked/tiled kernel (long-range 1/d
    /// tail, cannot be windowed), imaginary part via the windowed method
    /// (short-range, O(p·k)). Keeps the real part exact while saving the
    /// far-field imaginary iterations.
    BlockedHybrid,
    /// Balanced-error adaptive: real part via FFT odd-kernel (global, handles
    /// the long-range 1/d tail), imaginary part via windowed method (short-range)
    Adaptive,
    /// O(p log p) FFT-based convolution on a grid (full dual-convolution)
    Fft5,
    /// O(p log p) fused FFT grid convolution (3 FFTs instead of 5)
    Fft3,
    /// O(p log p) 2-FFT grid convolution (packed real + Hilbert packing)
    Fft2,
    /// O(p log p) 1D tree-code / Fast Multipole Method
    TreeCode,
    /// O(p log p) Chebyshev-interpolation treecode (faster than the multipole
    /// treecode at every size, especially when parallelized)
    ChebCode,
    /// Speed-tuned ChebCode preset (theta=1.0, n=8, leaf=32, 4-lane f32
    /// far field): ~2.3x the historical (0.5, 9) f64 point at a
    /// ~1e-5-class relative-L2 error. `chebcode`/`chebcode_balanced` keep
    /// the accuracy-grade operating points.
    ChebCodeFast,
    /// Precision-tuned ChebCode preset (theta=0.25, n=11, leaf=16):
    /// ~1e-12/1e-13 class without paying the full exact O(p²).
    ChebCodeXtreme,
    /// Middle ChebCode preset (theta=0.55, n=11, leaf=32): ~3e-10 class at
    /// roughly the FAST price (+6%) — measured round-2 operating point.
    ChebCodeBalanced,
    /// O(p·k + M log M) Ewald near/far splitting: exact near window +
    /// coarse-grid FFT far part (smooth kernel, small grid)
    Ewald,
    /// O(p log p) DST-I real part (odd-extension FFT)
    Dst,
    /// Auto-select the fastest method based on problem size $p$.
    Auto,
    /// Hierarchical low-rank (HODLR) summation: off-diagonal kernel blocks
    /// compressed by adaptive cross approximation to a requested tolerance,
    /// exact near-field at the leaves. Algebraic and self-validating — no
    /// geometric opening-angle parameter, no analytic translations.
    Hodlr,
    /// Data-driven maximum-speed preset: resolves via the measured Pareto
    /// table (`pareto_autogen`) to the fastest method per size and
    /// parallelism whose error stays under a sane cap. Regenerate with
    /// `scripts/build_pareto_table.py` after re-benchmarking.
    SpeedAuto,
    /// Data-driven accuracy-first preset: lowest measured error per size and
    /// parallelism, ties broken by runtime (`pareto_autogen`).
    AccuracyAuto,
}

impl StieltjesMethod {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::AutoVectorized => "autovec",
            Self::Blocked => "blocked",
            Self::BlockedAutoVec => "blocked_autovec",
            Self::BlockedTiled => "blocked_tiled",
            Self::BlockedWindowed => "blocked_windowed",
            Self::BlockedHybrid => "blocked_hybrid",
            Self::Adaptive => "adaptive",
            Self::Fft5 => "fft5",
            Self::Fft3 => "fft3",
            Self::Fft2 => "fft2",
            Self::TreeCode => "fmm",
            Self::ChebCode => "chebcode",
            Self::ChebCodeFast => "chebcode_fast",
            Self::ChebCodeXtreme => "chebcode_xtreme",
            Self::ChebCodeBalanced => "chebcode_balanced",
            Self::Ewald => "ewald",
            Self::Dst => "dst",
            Self::Auto => "auto",
            Self::Hodlr => "hodlr",
            Self::SpeedAuto => "speed_auto",
            Self::AccuracyAuto => "accuracy_auto",
        }
    }

    pub const fn description(&self) -> &'static str {
        match self {
            Self::Naive => "O(p²) scalar loop",
            Self::AutoVectorized => "O(p²) auto-vectorized loop",
            Self::Blocked => "O(p²) cache-blocked + unrolled + FMA",
            Self::BlockedAutoVec => "O(p²) cache-blocked + auto-vectorized",
            Self::BlockedTiled => "O(p²) 2D-tiled cache-blocked",
            Self::BlockedWindowed => "O(p·k) cache-blocked + binary-search window",
            Self::BlockedHybrid => "exact real(blocked) + windowed imag",
            Self::Adaptive => "balanced real(FFT)+imag(windowed)",
            Self::Fft5 => "O(p log p) FFT-grid convolution (5 FFTs)",
            Self::Fft3 => "O(p log p) fused FFT grid (3 FFTs)",
            Self::Fft2 => "O(p log p) 2-FFT grid (packed real + Hilbert)",
            Self::TreeCode => "O(p log p) 1D tree code (FMM)",
            Self::ChebCode => "O(p log p) Chebyshev-interpolation treecode",
            Self::ChebCodeFast => "ChebCode speed preset (theta 1.0, n 8, leaf 32, f32 far field)",
            Self::ChebCodeXtreme => "ChebCode precision preset (theta .25, n 11, leaf 16)",
            Self::ChebCodeBalanced => "ChebCode balanced preset (theta .55, n 11, leaf 32)",
            Self::Ewald => "O(p·k+M log M) Ewald near/far splitting",
            Self::Dst => "O(p log p) DST-I real part (odd-extension)",
            Self::Auto => "Auto-select based on problem size",
            Self::Hodlr => "O(p·r·log p) hierarchical low-rank (ACA) sums",
            Self::SpeedAuto => "Data-driven max-speed pick from the measured Pareto table",
            Self::AccuracyAuto => "Accuracy-first: exact O(p²) when cheap, ChebCode beyond",
        }
    }

    /// Return every non-auto, non-`Hodlr` variant for exhaustive benchmarking.
    ///
    /// `Hodlr` is excluded because it is an algebraic method with its own
    /// accuracy knobs rather than a member of the size/accuracy frontier
    /// this list feeds; the meta methods (`Auto`, `SpeedAuto`,
    /// `AccuracyAuto`) are excluded because they resolve to another entry.
    pub const fn all() -> &'static [Self] {
        &[
            Self::Naive,
            Self::AutoVectorized,
            Self::Blocked,
            Self::BlockedAutoVec,
            Self::BlockedTiled,
            Self::BlockedWindowed,
            Self::BlockedHybrid,
            Self::Adaptive,
            Self::Fft5,
            Self::Fft3,
            Self::Fft2,
            Self::TreeCode,
            Self::ChebCode,
            Self::ChebCodeFast,
            Self::ChebCodeXtreme,
            Self::ChebCodeBalanced,
            Self::Ewald,
            Self::Dst,
        ]
    }

    /// Select a method for a given problem size $p$ and parallelism.
    ///
    /// `Auto` is the **speed** policy: it resolves through the measured
    /// Pareto table (`pareto_autogen`, regenerated by
    /// `scripts/build_pareto_table.py` from `docs/pareto/bench_after.json`),
    /// taking the fastest method whose measured error stays under the cap.
    /// [`StieltjesMethod::SpeedAuto`] resolves identically; the two differ
    /// only in that `Auto` is the implicit default, while `SpeedAuto` is the
    /// explicit opt-in used by [`Strategy::Speed`].
    ///
    /// The historical hand-tuned thresholds (sequential `Fft2` at
    /// p ≥ 5000, `Blocked` in between, …) were retired: they sent e.g.
    /// p = 20k sequential to `Fft2`, measured ~12× slower than
    /// `ChebCodeFast` on the deconvolution grid path.
    pub fn resolve(p: usize, parallelism: Parallelism) -> Self {
        let parallel = matches!(parallelism, Parallelism::Parallel);
        pareto_autogen::pareto_pick(true, parallel, p)
    }
}

/// Parallelism strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parallelism {
    /// Single-threaded execution
    Sequential,
    /// Multi-threaded execution (data-parallel over eigenvalues)
    Parallel,
    /// Auto-select based on problem size and method
    Auto,
}

impl Parallelism {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Sequential => "seq",
            Self::Parallel => "parallel",
            Self::Auto => "auto",
        }
    }

    pub const fn all() -> &'static [Self] {
        &[Self::Sequential, Self::Parallel, Self::Auto]
    }

    /// Resolve Auto parallelism — always uses Sequential to avoid consuming
    /// all machine resources. Users who want multi-core must opt in explicitly.
    pub fn resolve(_p: usize, _method: StieltjesMethod) -> Self {
        Self::Sequential
    }
}

/// FFT grid sizing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftGridSize {
    /// Use p points (same as eigenvalue count)
    Auto,
    /// Explicit number of grid points
    Custom(usize),
}

impl FftGridSize {
    /// The explicit grid point count, or `None` for `Auto` (the kernel
    /// picks) — the `Option<usize>` form the kernels take.
    pub fn grid_points(self) -> Option<usize> {
        match self {
            Self::Auto => None,
            Self::Custom(n) => Some(n),
        }
    }
}

/// Far-field cutoff configuration.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CutoffConfig {
    /// No cutoff — compute all terms exactly
    #[default]
    Disabled,
    /// Skip terms where |λᵢ-λⱼ| > ratio · η.
    /// ratio=10 => ~1% max error per term, ratio=20 => ~0.25%
    Enabled { ratio: f64 },
}

impl CutoffConfig {
    /// The cutoff ratio, or `None` when disabled — the `Option<f64>` form
    /// the kernels take. Single conversion point for every caller that
    /// unpacks a config into kernel arguments.
    pub fn ratio(self) -> Option<f64> {
        match self {
            Self::Enabled { ratio } => Some(ratio),
            Self::Disabled => None,
        }
    }
}

/// Intermediate config presets between default and fully manual.
///
/// These set multiple options at once — no p-dependent logic.
/// You can still override individual fields after applying a strategy.
///
/// Note: all presets use sequential execution by default. Rayon parallelism
/// must be opted-in explicitly via `with_parallelism(Parallelism::Parallel)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Balanced default (same as `RmtConfig::new`)
    Default,
    /// Maximise speed — uses FFT-fused, enables far-field cutoff.
    Speed,
    /// Maximise accuracy — exact direct method, no cutoff.
    Accuracy,
}

impl Strategy {
    /// Apply this strategy to an `RmtConfig`, setting multiple fields at once.
    pub fn apply(self, cfg: &mut RmtConfig) {
        match self {
            Strategy::Default => {
                cfg.stieltjes_method = StieltjesMethod::Blocked;
            }
            Strategy::Speed => {
                // Data-driven max-speed pick (see `pareto_autogen`). The
                // user's parallelism choice is respected — Sequential and
                // Rayon have independent table columns.
                cfg.stieltjes_method = StieltjesMethod::SpeedAuto;
                // ratio=10 → ~1% error per skipped far term; benefits the
                // windowed family when the table picks it.
                cfg.cutoff = CutoffConfig::Enabled { ratio: 10.0 };
                // Inner blocking of the windowed/blocked_autovec kernels;
                // measured optimum region (8–16 at large p; 128 was stale).
                cfg.block_size = 16;
            }
            Strategy::Accuracy => {
                // Accuracy-first: lowest measured error first, ties broken by
                // runtime (see `pareto_autogen`). The user's parallelism
                // choice is respected. (Historically this pinned sequential
                // AutoVectorized — brutal at large p and blind to Rayon.)
                cfg.stieltjes_method = StieltjesMethod::AccuracyAuto;
                cfg.cutoff = CutoffConfig::Disabled;
                cfg.block_size = 32;
            }
        }
    }
}

/// Numeric precision for the Stieltjes kernel.
///
/// `Float64` is the default and is exact to machine precision (~1e-16).
/// `Float32` is ~2× faster (4 elements per NEON instruction vs 2) but has
/// ~1e-2 relative error — suitable only for the approximate methods
/// (FFT/treecode/windowed) or when speed matters more than precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Double precision (f64) — exact, default.
    #[default]
    Float64,
    /// Single precision (f32) — ~2× faster, ~1e-2 relative error.
    Float32,
}

/// Complete configuration for the RMT shrinkage kernel.
///
/// Controls all algorithm and optimization settings.
///
/// Fields are used as given — there is no hidden tuning. Only the two `Auto`
/// sentinels (`stieltjes_method`, `parallelism`) are resolved, and only when a
/// kernel calls [`RmtConfig::resolve_auto`] with a known problem size.
#[derive(Debug, Clone)]
pub struct RmtConfig {
    // === Core parameters ===
    /// Concentration ratio p / n
    pub c: f64,
    /// Regularization parameter; `None` = the path's default η, which is
    /// **0.1/√p** (`crate::stieltjes::default_eta`) in general and
    /// **0.4/√p** (`crate::stieltjes::default_eta_bulk`) for the pointwise
    /// bulk eigenvalue deconvolution. See `docs/eta_choice.md`.
    ///
    /// Channel note: the pointwise paths (`rie_shrinkage`,
    /// `direct_precision_shrinkage`, `estimate_population_eigenvalues`)
    /// read THIS field, while the grid/deconvolution drivers
    /// (`spectral_deconvolution`, `deconvolve_spiked`, `deconvolve_adaptive`)
    /// take an explicit `eta: Option<f64>` argument and ignore it.
    pub eta: Option<f64>,

    // === Algorithm selection ===
    /// Stieltjes transform method
    pub stieltjes_method: StieltjesMethod,
    /// Parallelism strategy
    pub parallelism: Parallelism,

    // === FFT-specific ===
    /// FFT grid size (only used with StieltjesMethod::Fft5 / Fft3 / Fft2)
    pub fft_grid_size: FftGridSize,

    // === Hardware optimizations ===
    /// Cache block size (used by the Blocked method)
    pub block_size: usize,
    /// Far-field cutoff configuration
    pub cutoff: CutoffConfig,
    /// Numeric precision (f64 exact, or f32 ~2× faster / ~1e-2 error)
    pub precision: Precision,
}

impl RmtConfig {
    /// Create a new config with sensible defaults.
    ///
    /// The defaults pin a concrete method (`Blocked`) and require an explicit
    /// opt-in to Rayon; use [`RmtConfig::resolve_auto`] to let the size-aware
    /// policies choose instead.
    pub fn new(c: f64) -> Self {
        Self {
            c,
            eta: None,
            stieltjes_method: StieltjesMethod::Blocked,
            parallelism: Parallelism::Sequential,
            fft_grid_size: FftGridSize::Auto,
            block_size: 64,
            cutoff: CutoffConfig::default(),
            precision: Precision::Float64,
        }
    }

    /// Create a config with all optimizations disabled (fully naive).
    pub fn fully_naive(c: f64) -> Self {
        Self {
            c,
            eta: None,
            stieltjes_method: StieltjesMethod::Naive,
            parallelism: Parallelism::Sequential,
            fft_grid_size: FftGridSize::Auto,
            block_size: 64,
            cutoff: CutoffConfig::Disabled,
            precision: Precision::Float64,
        }
    }

    // === Builder-style setters ===

    pub fn with_stieltjes(mut self, method: StieltjesMethod) -> Self {
        self.stieltjes_method = method;
        self
    }

    pub fn with_parallelism(mut self, p: Parallelism) -> Self {
        self.parallelism = p;
        self
    }

    pub fn with_eta(mut self, eta: f64) -> Self {
        self.eta = Some(eta);
        self
    }

    pub fn with_fft_grid(mut self, size: FftGridSize) -> Self {
        self.fft_grid_size = size;
        self
    }

    pub fn with_cutoff(mut self, cutoff: CutoffConfig) -> Self {
        self.cutoff = cutoff;
        self
    }

    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    /// Apply a strategy preset (Default / Speed / Accuracy).
    /// Sets multiple fields at once. Individual `.with_*` calls after this override them.
    pub fn with_strategy(mut self, strategy: Strategy) -> Self {
        strategy.apply(&mut self);
        self
    }

    /// Human-readable label for this config.
    pub fn label(&self) -> String {
        let st = self.stieltjes_method.name();
        let par = self.parallelism.name();
        format!("st={},par={}", st, par)
    }

    /// Resolve `Auto` to concrete settings based on the problem size $p$.
    ///
    /// Returns a new `RmtConfig` with:
    /// - `parallelism` resolved from `Auto` to Sequential/Rayon
    /// - `stieltjes_method` resolved from `Auto` to a concrete method,
    ///   **taking the resolved parallelism into account** (the best method
    ///   differs between sequential and parallel execution).
    ///
    /// Takes `&self` (rather than `self`) so callers can resolve a config
    /// without cloning it first.
    pub fn resolve_auto(&self, p: usize) -> Self {
        let mut resolved = self.clone();
        // Resolve parallelism first so the method choice can depend on it.
        if resolved.parallelism == Parallelism::Auto {
            resolved.parallelism = Parallelism::resolve(p, resolved.stieltjes_method);
        }
        let parallel_rayon = matches!(resolved.parallelism, Parallelism::Parallel);
        resolved.stieltjes_method =
            resolve_auto_method(resolved.stieltjes_method, parallel_rayon, p);
        resolved
    }

    /// [`Self::resolve_auto`] for the **at-points** driver
    /// ([`crate::stieltjes::compute_stieltjes_at_points`]), which evaluates
    /// the transform at `nq` arbitrary query points instead of at the `p`
    /// sample eigenvalues.
    ///
    /// The measured Pareto table behind `resolve_auto` is an *all-points*
    /// table (`nq = p`). One of its large-p speed picks is the FFT family,
    /// whose cost is a whole-grid convolution **independent of `nq`** — a
    /// sensible trade at `nq = p`, and the wrong one on a deconvolution grid.
    /// Measured, p = 50 000 sequential, `nq = 200`:
    ///
    /// | method | grid runtime | rel error |
    /// |---|---|---|
    /// | `Fft5` (the table pick) | 11.7 ms | ~4e-5 |
    /// | `ChebCodeFast` | 0.64 ms | ~1e-8 |
    ///
    /// So the auto preset is redirected to the ChebCode speed preset when the
    /// query count is small (`nq·4 < p`; the measured FFT/ChebCode grid
    /// crossover sits near `nq ≈ 0.7·p`). Above that the table's pick stands.
    ///
    /// Only the auto presets are second-guessed: an explicit
    /// `stieltjes_method` is always honoured as given.
    pub fn resolve_auto_at_points(&self, p: usize, nq: usize) -> Self {
        if !is_auto(self.stieltjes_method) {
            return self.resolve_auto(p);
        }
        let resolved = self.resolve_auto(p);
        Self {
            stieltjes_method: grid_appropriate(resolved.stieltjes_method, p, nq),
            ..resolved
        }
    }
}

/// Is `method` one of the three unresolved auto presets?
pub(crate) const fn is_auto(method: StieltjesMethod) -> bool {
    matches!(
        method,
        StieltjesMethod::Auto | StieltjesMethod::SpeedAuto | StieltjesMethod::AccuracyAuto
    )
}

/// Resolve an auto preset to a concrete method for the **all-points**
/// problem (`nq = p`).
///
/// `Auto` and `SpeedAuto` are the same policy — the fastest method whose
/// measured error stays under the cap — and `AccuracyAuto` is the
/// lowest-error one with ties broken by runtime. An already-concrete method
/// is returned unchanged.
///
/// This is THE single resolution point: [`RmtConfig::resolve_auto`] and the
/// Stieltjes dispatchers (`compute_all_stieltjes`,
/// `compute_stieltjes_at_points`) all route through it, so `method="auto"`
/// means the same thing however it is reached. It previously did not:
/// `compute_all_stieltjes` resolved only the two explicit `*Auto` presets and
/// silently fell back to the exact `Blocked` kernel for plain `Auto`, so the
/// Python `stieltjes_transform(method="auto")` paid O(p²) — 102 ms at
/// p = 20 000 instead of 4.6 ms, a ~22× miss against the documented policy.
pub(crate) fn resolve_auto_method(
    method: StieltjesMethod,
    parallel: bool,
    p: usize,
) -> StieltjesMethod {
    match method {
        StieltjesMethod::Auto | StieltjesMethod::SpeedAuto => {
            pareto_autogen::pareto_pick(true, parallel, p)
        }
        StieltjesMethod::AccuracyAuto => pareto_autogen::pareto_pick(false, parallel, p),
        concrete => concrete,
    }
}

/// Redirect a **resolved** method that pays for the whole grid to the
/// treecode when only a few query points are needed.
///
/// The FFT/Adaptive/Dst families evaluate a full uniform-grid convolution
/// whose cost ignores `nq`; `ChebCodeFast` serves exactly `nq` points. The
/// threshold is `nq·4 < p` against a measured crossover near `nq ≈ 0.7·p`.
/// Concrete methods are only redirected when they belong to that family, and
/// callers apply this to *auto-resolved* methods only.
pub(crate) fn grid_appropriate(method: StieltjesMethod, p: usize, nq: usize) -> StieltjesMethod {
    if nq.saturating_mul(4) >= p {
        return method;
    }
    match method {
        StieltjesMethod::Fft5
        | StieltjesMethod::Fft3
        | StieltjesMethod::Fft2
        | StieltjesMethod::Adaptive
        | StieltjesMethod::Dst => StieltjesMethod::ChebCodeFast,
        concrete => concrete,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_resolution_is_parallelism_aware() {
        // (parallelism, p, expected pick) — mirrors the regenerated Pareto
        // speed bins; update together with pareto_autogen.rs.
        // Every speed bin now resolves to `ChebCodeFast` in both columns:
        // the retuned preset (theta 1.0, n 8, f32 far field) is the fastest
        // measured point at every size, so the parallelism dimension no
        // longer changes the pick.
        for (par, p, expected) in [
            (
                Parallelism::Sequential,
                10_000usize,
                StieltjesMethod::ChebCodeFast,
            ),
            (
                Parallelism::Parallel,
                10_000usize,
                StieltjesMethod::ChebCodeFast,
            ),
            (
                Parallelism::Sequential,
                100usize,
                StieltjesMethod::ChebCodeFast,
            ),
            (
                Parallelism::Parallel,
                100usize,
                StieltjesMethod::ChebCodeFast,
            ),
        ] {
            let resolved = RmtConfig::new(0.5)
                .with_stieltjes(StieltjesMethod::Auto)
                .with_parallelism(par)
                .resolve_auto(p);
            assert_eq!(resolved.stieltjes_method, expected, "par={par:?} p={p}");
        }
    }

    #[test]
    fn test_auto_parallelism_resolves_to_sequential() {
        // Auto parallelism always resolves to Sequential (Rayon must be opted in).
        let cfg = RmtConfig::new(0.5)
            .with_stieltjes(StieltjesMethod::Auto)
            .with_parallelism(Parallelism::Auto)
            .resolve_auto(10000);
        assert_eq!(cfg.parallelism, Parallelism::Sequential);
        // Method resolved based on the resolved (sequential) parallelism.
        assert_eq!(cfg.stieltjes_method, StieltjesMethod::ChebCodeFast);
    }

    #[test]
    fn test_at_points_resolution_avoids_the_whole_grid_fft() {
        // The retuned table now picks `ChebCodeFast` for the all-points speed
        // intent at every size, so the deconvolution grid must be left alone
        // (the historical redirect existed because the large-p pick was
        // `Fft5`, whose cost ignores the query count).
        let cfg = RmtConfig::new(0.5).with_stieltjes(StieltjesMethod::Auto);
        assert_eq!(
            cfg.resolve_auto(50_000).stieltjes_method,
            StieltjesMethod::ChebCodeFast,
            "precondition: the all-points table pick is ChebCodeFast here"
        );
        assert_eq!(
            cfg.resolve_auto_at_points(50_000, 200).stieltjes_method,
            StieltjesMethod::ChebCodeFast
        );
        // Explicitly-requested FFT methods are still redirected on a small
        // grid (that rule is independent of the table).
        let fft = RmtConfig::new(0.5).with_stieltjes(StieltjesMethod::Fft5);
        assert_eq!(
            fft.resolve_auto_at_points(50_000, 200).stieltjes_method,
            StieltjesMethod::Fft5,
            "an explicit method is never second-guessed"
        );
        // The same redirection applies to the explicit speed preset...
        let speed = RmtConfig::new(0.5).with_stieltjes(StieltjesMethod::SpeedAuto);
        assert_eq!(
            speed.resolve_auto_at_points(50_000, 200).stieltjes_method,
            StieltjesMethod::ChebCodeFast
        );
        // ...but never to a method the caller named explicitly.
        for explicit in [
            StieltjesMethod::Fft5,
            StieltjesMethod::ChebCode,
            StieltjesMethod::Blocked,
        ] {
            let cfg = RmtConfig::new(0.5).with_stieltjes(explicit);
            assert_eq!(
                cfg.resolve_auto_at_points(50_000, 200).stieltjes_method,
                explicit
            );
        }
    }

    #[test]
    fn test_at_points_resolution_leaves_small_p_alone() {
        // Every small-p speed bin already resolves to a ChebCode preset, so
        // the at-points resolver is a no-op there.
        let cfg = RmtConfig::new(0.5).with_stieltjes(StieltjesMethod::Auto);
        for p in [100usize, 1_000, 10_000, 20_000] {
            assert_eq!(
                cfg.resolve_auto_at_points(p, 200).stieltjes_method,
                cfg.resolve_auto(p).stieltjes_method,
                "p={p}"
            );
        }
    }
}

#[cfg(test)]
mod preset_tests {
    use super::*;

    #[test]
    fn pareto_pick_returns_concrete_methods_everywhere() {
        for &p in &[1usize, 500, 1000, 1500, 4000, 9000, 15000, 30000, 80000] {
            for parallel in [false, true] {
                for speed in [false, true] {
                    let m = pareto_autogen::pareto_pick(speed, parallel, p);
                    assert!(
                        !matches!(
                            m,
                            StieltjesMethod::Auto
                                | StieltjesMethod::SpeedAuto
                                | StieltjesMethod::AccuracyAuto
                        ),
                        "p={p} par={parallel} speed={speed}: unresolved {m:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn speed_preset_respects_parallelism_choice() {
        let mut cfg = RmtConfig::new(0.5);
        cfg.parallelism = Parallelism::Parallel;
        Strategy::Speed.apply(&mut cfg);
        assert_eq!(cfg.parallelism, Parallelism::Parallel);
        assert_eq!(cfg.stieltjes_method, StieltjesMethod::SpeedAuto);

        let mut cfg = RmtConfig::new(0.5);
        cfg.parallelism = Parallelism::Sequential;
        Strategy::Accuracy.apply(&mut cfg);
        assert_eq!(cfg.parallelism, Parallelism::Sequential);
    }

    #[test]
    fn resolve_auto_resolves_presets() {
        for (method, par) in [
            (StieltjesMethod::SpeedAuto, Parallelism::Sequential),
            (StieltjesMethod::SpeedAuto, Parallelism::Parallel),
            (StieltjesMethod::AccuracyAuto, Parallelism::Sequential),
            (StieltjesMethod::AccuracyAuto, Parallelism::Parallel),
        ] {
            let cfg = RmtConfig {
                stieltjes_method: method,
                parallelism: par,
                ..RmtConfig::new(0.5)
            };
            let r = cfg.resolve_auto(12000);
            assert!(
                !matches!(
                    r.stieltjes_method,
                    StieltjesMethod::SpeedAuto | StieltjesMethod::AccuracyAuto
                ),
                "{method:?} did not resolve"
            );
        }
    }
}
