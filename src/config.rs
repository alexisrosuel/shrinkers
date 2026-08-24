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
    /// Speed-tuned ChebCode preset (theta=0.5, n=9, leaf=32): ~1e-8 error
    /// class at the lowest measured runtime of the family.
    ChebCodeFast,
    /// Precision-tuned ChebCode preset (theta=0.25, n=11, leaf=16):
    /// ~1e-12/1e-13 class without paying the full exact O(p²).
    ChebCodeXtreme,
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
    /// table ([`pareto_autogen`]) to the fastest method per size and
    /// parallelism whose error stays under a sane cap. Regenerate with
    /// `scripts/build_pareto_table.py` after re-benchmarking.
    SpeedAuto,
    /// Data-driven accuracy-first preset: lowest measured error per size and
    /// parallelism, ties broken by runtime ([`pareto_autogen`]).
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
            Self::ChebCodeFast => "ChebCode speed preset (theta .5, n 9, leaf 32)",
            Self::ChebCodeXtreme => "ChebCode precision preset (theta .25, n 11, leaf 16)",
            Self::Ewald => "O(p·k+M log M) Ewald near/far splitting",
            Self::Dst => "O(p log p) DST-I real part (odd-extension)",
            Self::Auto => "Auto-select based on problem size",
            Self::Hodlr => "O(p·r·log p) hierarchical low-rank (ACA) sums",
            Self::SpeedAuto => "Data-driven max-speed pick from the measured Pareto table",
            Self::AccuracyAuto => "Accuracy-first: exact O(p²) when cheap, ChebCode beyond",
        }
    }

    /// Return all non-auto variants for exhaustive benchmarking.
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
            Self::Ewald,
            Self::Dst,
        ]
    }

    /// Select the fastest method for a given problem size $p$ and parallelism.
    ///
    /// The best method depends on whether Rayon parallelism is available,
    /// because the O(p²) direct methods and the O(p log p) treecode both
    /// parallelize well, while the FFT methods do not benefit from Rayon.
    ///
    /// Based on benchmarks on Apple M-series (M3 Max):
    ///
    /// **Sequential:**
    /// - p ≤ 200:   `AutoVectorized` (low overhead, compiler auto-vec)
    /// - 200 < p < 5000: `Blocked` (sequential O(p²), no core contention)
    /// - p ≥ 5000:  `Fft2` (O(p log p), sequential, predictable)
    ///
    /// **Rayon (parallel):**
    /// - p ≤ 200:   `AutoVectorized` (parallel overhead not worth it)
    /// - 200 < p < 5000: `Blocked` (parallel λᵢ-outer, ~2.6-5.4× speedup)
    /// - p ≥ 5000:  `ChebCode` (parallel Chebyshev treecode, ~1.7-2× faster
    ///   than the multipole `TreeCode` at every size, with ~0 error)
    pub fn resolve(p: usize, parallelism: Parallelism) -> Self {
        match parallelism {
            Parallelism::Rayon => {
                if p <= 200 {
                    Self::AutoVectorized
                } else if p < 5000 {
                    Self::Blocked
                } else {
                    Self::ChebCode
                }
            }
            Parallelism::Sequential | Parallelism::Auto => {
                if p <= 200 {
                    Self::AutoVectorized
                } else if p < 5000 {
                    Self::Blocked
                } else {
                    Self::Fft2
                }
            }
        }
    }
}

/// Parallelism strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parallelism {
    /// Single-threaded execution
    Sequential,
    /// Parallel using Rayon (data-parallel over eigenvalues)
    Rayon,
    /// Auto-select based on problem size and method
    Auto,
}

impl Parallelism {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Sequential => "seq",
            Self::Rayon => "rayon",
            Self::Auto => "auto",
        }
    }

    pub const fn all() -> &'static [Self] {
        &[Self::Sequential, Self::Rayon, Self::Auto]
    }

    /// Resolve Auto parallelism — always uses Sequential to avoid consuming
    /// all machine resources. Users who want Rayon must opt-in explicitly.
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

/// Intermediate config presets between default and fully manual.
///
/// These set multiple options at once — no p-dependent logic.
/// You can still override individual fields after applying a strategy.
///
/// Note: all presets use sequential execution by default. Rayon parallelism
/// must be opted-in explicitly via `with_parallelism(Parallelism::Rayon)`.
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
/// The config is used as-is — no auto-tuning, no fallback logic.
#[derive(Debug, Clone)]
pub struct RmtConfig {
    // === Core parameters ===
    /// Concentration ratio p / n
    pub c: f64,
    /// Regularization parameter (default: 0.1 / sqrt(p))
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
    /// The config is used as-is — no auto-tuning.
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
        if resolved.stieltjes_method == StieltjesMethod::Auto {
            resolved.stieltjes_method = StieltjesMethod::resolve(p, resolved.parallelism);
        }
        if resolved.stieltjes_method == StieltjesMethod::AccuracyAuto {
            let parallel_rayon = matches!(resolved.parallelism, Parallelism::Rayon);
            resolved.stieltjes_method = pareto_autogen::pareto_pick(false, parallel_rayon, p);
        }
        if resolved.stieltjes_method == StieltjesMethod::SpeedAuto {
            let parallel_rayon = matches!(resolved.parallelism, Parallelism::Rayon);
            resolved.stieltjes_method = pareto_autogen::pareto_pick(true, parallel_rayon, p);
        }
        resolved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_resolution_is_parallelism_aware() {
        // At large p, Auto should pick TreeCode when parallel, Fft2 when not.
        let seq = RmtConfig::new(0.5)
            .with_stieltjes(StieltjesMethod::Auto)
            .with_parallelism(Parallelism::Sequential)
            .resolve_auto(10000);
        assert_eq!(seq.stieltjes_method, StieltjesMethod::Fft2);

        let par = RmtConfig::new(0.5)
            .with_stieltjes(StieltjesMethod::Auto)
            .with_parallelism(Parallelism::Rayon)
            .resolve_auto(10000);
        assert_eq!(par.stieltjes_method, StieltjesMethod::ChebCode);

        // At small p, both pick AutoVectorized.
        let seq_small = RmtConfig::new(0.5)
            .with_stieltjes(StieltjesMethod::Auto)
            .with_parallelism(Parallelism::Sequential)
            .resolve_auto(100);
        assert_eq!(seq_small.stieltjes_method, StieltjesMethod::AutoVectorized);

        let par_small = RmtConfig::new(0.5)
            .with_stieltjes(StieltjesMethod::Auto)
            .with_parallelism(Parallelism::Rayon)
            .resolve_auto(100);
        assert_eq!(par_small.stieltjes_method, StieltjesMethod::AutoVectorized);
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
        assert_eq!(cfg.stieltjes_method, StieltjesMethod::Fft2);
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
        cfg.parallelism = Parallelism::Rayon;
        Strategy::Speed.apply(&mut cfg);
        assert_eq!(cfg.parallelism, Parallelism::Rayon);
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
            (StieltjesMethod::SpeedAuto, Parallelism::Rayon),
            (StieltjesMethod::AccuracyAuto, Parallelism::Sequential),
            (StieltjesMethod::AccuracyAuto, Parallelism::Rayon),
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
