# `shrinkers` Python API Reference

The `shrinkers` module is a PyO3 binding over the Rust RMT shrinkage kernel.
It exposes these functions:

| Function | Purpose |
|---|---|
| `deconvolve_spiked` | **Primary**: spiked + bulk cleaning via free-probability deconvolution |
| `clean_correlation_matrix` | Clean a full correlation matrix (RIE + eigenvector overlaps) |
| `clean_covariance_matrix` | Same estimator, covariance-named twin for a full covariance matrix |
| `direct_precision_shrinkage` | Direct precision-matrix eigenvalue shrinkage |
| `inverse_nonlinear_shrinkage` | Ledoit–Wolf inverse shrinkage (QIS/LIS/GIS) precision eigenvalues |
| `estimate_precision_matrix` | Estimate the full precision matrix Ω̂ = Σ̂⁻¹ from a covariance matrix |
| `stieltjes_transform` | Raw empirical Stieltjes transform |
| `stieltjes_transform_with_deriv` | S and its analytic derivative dS/dx in one pass |
| `detect_spikes_bema` | BEMA spike detection (K, bulk edge, σ²) |
| `detect_spikes_tracy_widom` | Tracy–Widom edge spike detection |
| `inverse_bbp` | Inverse BBP / DGJ spike debiasing (scalar or array) |
| `analyze_spikes` | Full spiked-model analysis (detection + debiasing + overlaps) |
| `estimate_population_eigenvalues` | Per-eigenvalue population estimates under a spiked model |
| `ledoit_wolf_shrinkage` | Raw Ledoit–Wolf non-linear shrinkage ξ(λᵢ) |
| `shrink_eigenvalues` | Trace-preserving RIE shrinkage |

- **Module name:** `shrinkers`
- **Version:** `shrinkers.__version__` (single-sourced from `Cargo.toml`)
- **Python requirement:** `>= 3.9` (CI builds and runs the suite on 3.9–3.15;
  the wheel is `cp39-abi3`, so one artifact serves all of them)
- **Array convention:** all 1-D inputs are `numpy.ndarray` of `float64`
  (contiguous).
- **Threading:** heavy computation releases the GIL (`py.detach`), so Python
  threads stay responsive during long calls.
- **Validation:** inputs are checked at the boundary; violations raise
  `ValueError` (never a Rust panic).

---

## Input contracts

- Eigenvalues must be **finite**. Spectra taken as covariance/correlation
  eigenvalues must be **non-negative**: tiny negative round-off
  (≥ −1e-10·scale, typical of centered-sample eigendecompositions) is
  clamped to zero automatically; meaningfully negative values raise
  `ValueError`.
- The concentration ratio `c = p/n` must satisfy **0 < c ≤ 1** (the spiked +
  bulk estimators do not apply in the overparameterized regime p > n).
- Arrays must be contiguous; use `np.ascontiguousarray(...)` for views.

---

## Primary entry point

### `deconvolve_spiked(eigenvalues, c, n_points=200, eta="inferred", margin=1.0, *, method="auto", parallel=False, cutoff=None)`

Given the sample eigenvalues, recover the cleaned population spectrum under a
spiked covariance model. It orchestrates the full pipeline:

1. **Spike detection** (BEMA) — find the sample eigenvalues that escape the
   bulk edge.
2. **Spike debiasing** (inverse BBP / DGJ) — recover the population spike
   eigenvalues $\ell_i$ from the biased sample spikes.
3. **Bulk deconvolution** (El Karoui 2008) — remove the spikes and invert the
   Marčenko–Pastur equation on the remaining bulk to recover the population
   spectral density.

> **Why El Karoui over RIE?** RIE / Ledoit–Wolf shrinkage is the *pointwise*
> special case of free-probability deconvolution: it maps each individual
> sample eigenvalue to a population estimate. El Karoui is the full inversion
> that recovers the entire population spectral density, which is the general
> approach this package uses for the bulk.

**Parameters**

- `eigenvalues` — `np.ndarray[float64]`, shape `(p,)`. Sample eigenvalues
  (any order; sorted internally). Must be contiguous, finite, positive.
- `c` — `float`. Concentration ratio $p/n$, in $(0, 1]$.
- `n_points` — `int`, default `200`. Grid resolution for the bulk deconvolution.
- `eta` — `float | "inferred"`. Regularization for the bulk deconvolution;
  default `0.1 / sqrt(p)`.
- `margin` — `float`, default `1.0`. Multiplicative margin above the fitted
  bulk edge for spike detection (slightly above 1.0 adds robustness).
- `method` — keyword-only, `str`, default `"auto"`. Stieltjes kernel for the
  bulk deconvolution: any of `"naive"`, `"autovec"`, `"blocked"`,
  `"blocked_autovec"`, `"blocked_tiled"`, `"blocked_windowed"`,
  `"blocked_hybrid"`, `"adaptive"`, `"fft5"`, `"fft3"`, `"fft2"`, `"fmm"`
  (alias `"treecode"`), `"chebcode"` (alias `"chebyshev"`),
  `"chebcode_fast"` (alias `"chebf"`; ~2e-5, fastest through p = 50 000),
  `"chebcode_balanced"` (alias `"chebb"`; ~5e-10),
  `"chebcode_xtreme"` (alias `"chebx"`; ~6e-13), `"nufft"` (alias
  `"fft_fast"`; coarse-grid FFT speed path, auto-selected above p = 50 000), `"hodlr"`, `"ewald"`,
  `"dst"`, `"speed_auto"` (alias `"speed"`), `"accuracy_auto"`
  (alias `"accuracy"`), or `"auto"`.
  `"auto"` is the **speed policy**, not an exact fallback: it resolves
  through the same measured Pareto table as `"speed_auto"` (identical pick)
  and is therefore approximate — ~2e-5 at the retuned `chebcode_fast`
  point through p = 50 000, then the coarse-grid `nufft` (~3e-5…4e-5). For a
  machine-precision
  result ask for `"blocked"` (the `stieltjes_transform` default) or
  `"blocked_tiled"`, or for `"accuracy_auto"`.

  `"nufft"` / `"fft_fast"` is the large-p speed path: a coarse-grid FFT
  convolution whose grid is capped at 65 536 cells, so its cost grows with
  `p` only through the O(p) splat/interpolation. Measured Python-level at
  η = 1/√p: 3.1 → 1.8 ms at p = 100 000, 6.7 → 2.8 ms at 200 000 and
  31.9 → 8.4 ms at 1 000 000 versus `chebcode_fast`, with rel. error
  ~2.7e-5…4.3e-5. At very small η the capped grid trades the last digit for
  speed (~1e-4).

  On the deconvolution grid a
  whole-grid FFT pick is additionally re-routed to the treecode, which
  serves exactly the `n_points` queries.
- `parallel` — keyword-only, `bool | None`, default `False`.
  `True` enables multi-core execution, `False` forces single-threaded,
  `None` lets the library decide from the problem size. The threading
  backend is an implementation detail and deliberately unnamed.
- `cutoff` — keyword-only, `float | None | "inferred"`, default disabled
  (`None` and `"inferred"` are synonyms). Far-field cutoff ratio
  (10 ≈ 1% max per-term error).

**Returns**

- `dict` with keys:
  - `"k"` — `int`, number of detected spikes.
  - `"spikes"` — `np.ndarray[float64]`, estimated **population** spike
    eigenvalues $\ell_i$ (descending), debiased via inverse BBP.
  - `"spike_sample"` — `np.ndarray[float64]`, the sample eigenvalues classified
    as spikes (descending).
  - `"bulk_edge"` — `float`, estimated bulk edge $\lambda_+ = \sigma^2(1+\sqrt\gamma)^2$.
  - `"sigma2"` — `float`, estimated noise variance.
  - `"bulk"` — `dict`, the bulk deconvolution with keys:
    - `"lambda_grid"` — `np.ndarray[float64]`, λ values where density is evaluated.
    - `"density"` — `np.ndarray[float64]`, population spectral density $\rho(\lambda)$.
    - `"w_re"` — `np.ndarray[float64]`, real part of $w = z/a(z)$.
    - `"sample_stieltjes_real"` — `np.ndarray[float64]`, $\Re[g(z)]$.
    - `"sample_stieltjes_imag"` — `np.ndarray[float64]`, $\Im[g(z)]$.
    - `"population_stieltjes_real"` — `np.ndarray[float64]`, $\Re[m_\Sigma(w)]$.
    - `"population_stieltjes_imag"` — `np.ndarray[float64]`, $\Im[m_\Sigma(w)]$.

**Example**

```python
from shrinkers import deconvolve_spiked

res = deconvolve_spiked(evals, c=0.25, n_points=300, eta=0.05)
print(res["k"])                 # number of spikes
print(res["spikes"])            # debiased population spikes ℓ_i
print(res["bulk_edge"])         # bulk edge λ₊
print(res["sigma2"])            # noise variance σ²
print(res["bulk"]["density"])   # deconvolved bulk density
```

---

## Cleaning a correlation matrix

### `clean_correlation_matrix(correlation, c, *, renormalize=False)`

Clean a sample **correlation matrix** via RIE eigenvalue shrinkage + eigenvector
angular overlap correction. The spectral decomposition is computed internally,
then the RIE-cleaned eigenvalues and the theoretical eigenvector alignment are
returned alongside the cleaned covariance matrix.

**Parameters**

- `correlation` — `np.ndarray[float64]`, shape `(p, p)`. Sample correlation
  matrix (symmetric, finite). Must be contiguous.
- `c` — `float`. Concentration ratio $p/n$, in $(0, 1]$.
- `renormalize` — `bool`, default `False`. Rescale the cleaned matrix back to
  unit diagonal, $R = D^{-1/2}\hat\Sigma D^{-1/2}$ with $D = \mathrm{diag}(\hat\Sigma)$.
  The RIE + overlap reconstruction preserves the *trace* but not the individual
  diagonal entries, which drift well away from one when spikes are present
  (≈ 0.2 on a $p = 60$ three-factor example). Renormalising restores the exact
  correlation contract — and, since the population correlation has a unit
  diagonal, it lowered the distance to the population in the examples tested —
  at the cost of the reconstruction's rotation-equivariance. If any diagonal
  entry is non-positive (not a valid covariance) the matrix is returned
  unchanged.

**Returns**

- `dict` with keys:
  - `"covariance"` — `np.ndarray[float64]`, shape `(p, p)`. Cleaned covariance
    matrix, symmetric & positive definite.
  - `"eigenvectors"` — `np.ndarray[float64]`, shape `(p, p)`. Sample
    eigenvectors, columns sorted descending by eigenvalue.
  - `"eigenvalues"` — `np.ndarray[float64]`, shape `(p,)`. RIE-cleaned
    eigenvalues, descending.
  - `"overlaps"` — `np.ndarray[float64]`, shape `(p,)`. Squared angular overlaps
    $\alpha_i^2 = \cos^2(\theta_i)$ between each sample eigenvector and its
    (unknown) population counterpart, parallel to `eigenvalues`.
  - `"sigma2"` — `float`. Estimated noise variance $\sigma^2$
    (Marchenko–Pastur-median-corrected).

**Example**

```python
import numpy as np
from shrinkers import clean_correlation_matrix

# Sample correlation matrix from data X (T, N)
C = np.corrcoef(X, rowvar=False)
res = clean_correlation_matrix(C, c=N / T)

cleaned_cov = res["covariance"]   # cleaned covariance matrix
evecs = res["eigenvectors"]       # sample eigenvectors (descending)
evals = res["eigenvalues"]        # RIE-cleaned eigenvalues (descending)
overlaps = res["overlaps"]        # theoretical alignment of each eigenvector
sigma2 = res["sigma2"]            # noise variance

# Exact unit diagonal, for a bona fide correlation matrix
corr = clean_correlation_matrix(C, c=N / T, renormalize=True)["covariance"]
np.fill_diagonal(corr, 1.0)       # already 1 to round-off
```

### `clean_covariance_matrix(covariance, c)`

The covariance-named twin of `clean_correlation_matrix`: the same estimator on a
full **covariance** matrix. The estimator only ever sees a spectrum and `c`, so
it is agnostic to whether the input is a covariance or a correlation matrix —
the two entry points differ only in the scale of the matrix they return. Use
this one when the input carries variances you want preserved in the output.

Note that cleaning a covariance and then standardising it is **not** the same as
cleaning the correlation matrix directly: RIE commutes with orthogonal
conjugation but not with the diagonal congruence $D^{-1/2} S D^{-1/2}$. If you
want a cleaned correlation matrix, build the sample correlation matrix and call
`clean_correlation_matrix` on it.

**Parameters**

- `covariance` — `np.ndarray[float64]`, shape `(p, p)`. Sample covariance
  matrix (symmetric, finite). Must be contiguous.
- `c` — `float`. Concentration ratio $p/n$, in $(0, 1]$.

**Returns**

- `dict` with the same keys as `clean_correlation_matrix`.

---

## Precision estimation

### `direct_precision_shrinkage(eigenvalues, c)`

Direct Nonlinear Shrinkage (Ledoit & Wolf 2020): estimates the eigenvalues of
the precision matrix $\Omega = \Sigma^{-1}$ directly, without inverting a
cleaned covariance. Asymptotically optimal for precision loss.

**Parameters**

- `eigenvalues` — `np.ndarray[float64]`, shape `(p,)`, finite, positive.
- `c` — `float`, in $(0, 1]$.

**Returns**

- `dict` with key `"precision_eigenvalues"` — `np.ndarray[float64]`, shape
  `(p,)`.

```python
from shrinkers import direct_precision_shrinkage

res = direct_precision_shrinkage(evals, c=0.25)
omega_evals = res["precision_eigenvalues"]
```

---

### `inverse_nonlinear_shrinkage(eigenvalues, c, *, method="qis", parallel=False)`

Ledoit–Wolf **inverse nonlinear shrinkage** (Bernoulli 2022): estimates the
eigenvalues of the precision matrix $\Omega = \Sigma^{-1}$ *directly*, so the
small inverse eigenvalues are not over-inflated by inverting a
covariance-optimal estimator. Three losses are available:

| `method` | Loss the eigenvalues are optimal for |
|---|---|
| `"qis"` (default) | Frobenius / inverse Stein / minimum variance |
| `"lis"` | Stein's loss |
| `"gis"` | symmetrized Kullback–Leibler |

Only the non-singular regime $c \le 1$ ($p \le n$) is supported, and the
eigenvalues must be **strictly positive** (the kernel divides by λ and
evaluates the transform on $\lambda(1+ih)$, so a zero is a genuine error here).

**Parameters**

- `eigenvalues` — `np.ndarray[float64]`, shape `(p,)`, finite, strictly
  positive.
- `c` — `float`, in $(0, 1]$.
- `method` — `"qis"` (default), `"lis"` or `"gis"`.
- `parallel` — `False` (default, single-threaded), `True` (multi-core), or
  `None` (library picks by problem size).

**Returns**

- `dict` with:
  - `"precision_eigenvalues"` — `np.ndarray[float64]`, shape `(p,)`, the Ω̂
    eigenvalues, parallel to the input eigenvalues;
  - `"covariance_eigenvalues"` — the matching Σ̂ eigenvalues (the reciprocals,
    trace-rescaled for QIS);
  - `"smoothing"` — the Ledoit–Wolf bandwidth $h$.

```python
from shrinkers import inverse_nonlinear_shrinkage

res = inverse_nonlinear_shrinkage(evals, c=0.25)
omega_evals = res["precision_eigenvalues"]
```

---

### `estimate_precision_matrix(covariance, c, *, method="qis", parallel=False)`

Estimate the full precision matrix
$\hat{\Omega} = \hat{\Sigma}^{-1}$ from a sample covariance (or correlation)
matrix by inverse nonlinear shrinkage — the symmetric counterpart of
`clean_correlation_matrix`. The estimator is rotation-equivariant,
$\hat{\Omega} = U\,\mathrm{diag}(\omega)\,U'$, with $U$ the sample
eigenvectors.

**Parameters**

- `covariance` — `np.ndarray[float64]`, shape `(p, p)`, symmetric, finite,
  positive definite.
- `c` — `float`, in $(0, 1]$.
- `method` — `"qis"` (default), `"lis"` or `"gis"`.
- `parallel` — `False` (default), `True`, or `None` (library picks).

**Returns**

- `dict` with:
  - `"precision"` — precision matrix estimate Ω̂, shape `(p, p)`;
  - `"eigenvectors"` — sample eigenvectors `(p, p)`, columns sorted by
    descending sample eigenvalue;
  - `"precision_eigenvalues"` — Ω̂ eigenvalues `(p,)`, paired column-by-column
    with `"eigenvectors"`;
  - `"covariance_eigenvalues"` — the matching Σ̂ eigenvalues `(p,)`;
  - `"smoothing"` — the Ledoit–Wolf bandwidth $h$.

```python
from shrinkers import estimate_precision_matrix

res = estimate_precision_matrix(sample_cov, c=0.25)
omega = res["precision"]              # (p, p) precision matrix
omega_evals = res["precision_eigenvalues"]  # paired with res["eigenvectors"]
```

The dense path eigendecomposes internally with Jacobi iteration, so for large
$p$ prefer extracting the eigensystem in NumPy/SciPy and calling the Rust
`precision_from_eigensystem` entry point directly.

---

## Raw Stieltjes transform

### `stieltjes_transform(eigenvalues, eta="inferred", method="blocked", precision="f64", cutoff="inferred", parallel=False)`

Compute the empirical Stieltjes transform
$S(\lambda_i) = \frac{1}{p}\sum_j \frac{1}{\lambda_i - \lambda_j - i\eta}$.

**Parameters**

- `eigenvalues` — `np.ndarray[float64]`, shape `(p,)`, finite, non-empty.
- `eta` — `float | "inferred"`, default `0.1 / sqrt(p)`. Must be positive.
- `method` — see the list under `deconvolve_spiked`; default `"blocked"`.
- `precision` — `"f64"` (default, machine precision) or `"f32"` (~2× faster,
  ~1e-2 relative error).
- `cutoff` — `float | None | "inferred"`, default disabled. Far-field cutoff;
  only affects methods that support it (e.g. `"blocked"`).
- `parallel` — `False` (default, single-threaded), `True`
    (multi-core), or `None` (library picks by problem size).

**Returns**

- `dict` with `"real"` and `"imag"` arrays, shape `(p,)`.

```python
from shrinkers import stieltjes_transform

res = stieltjes_transform(evals, method="blocked_tiled")
m_real, m_imag = res["real"], res["imag"]
```

### `stieltjes_transform_with_deriv(eigenvalues, eta="inferred")`

Compute $S$ at every sample eigenvalue together with its analytic
derivative $S'(\lambda_i) = -\frac{1}{p}\sum_j \frac{1}{(\lambda_i - \lambda_j - i\eta)^2}$,
in one exact O(p²) pass. Useful for root-finding on γ or η.

**Parameters**

- `eigenvalues` — `np.ndarray[float64]`, shape `(p,)`, finite, non-empty.
- `eta` — `float | "inferred"`, default `0.1 / sqrt(p)`.

**Returns**

- `dict` with `"real"`, `"imag"`, `"deriv_real"`, `"deriv_imag"` arrays.

## Spiked-model toolkit

### `detect_spikes_bema(eigenvalues, c, margin=1.0)` / `detect_spikes_tracy_widom(eigenvalues, c, sigma2=None, significance=0.05)`

Determine the number of spikes $K$ and the noise level. Both return a dict
with `"k"`, `"spike_indices"` (indices into the **ascending-sorted**
eigenvalue array), `"bulk_edge"`, and `"sigma2"`.

### `inverse_bbp(lambda_hat, c, sigma2=1.0)`

Recover the population spike $\ell$ from sample spike(s) $\hat\lambda$
(scalar or array). Values at or below the BBP threshold return the bulk edge.

### `analyze_spikes(eigenvalues, c, margin=1.0)`

Full spiked analysis: returns `"k"`, `"spikes"` (population, descending),
`"overlaps"` ($\alpha_i^2$ per spike), `"bulk_edge"`, `"sigma2"`, and
`"ledoit_wolf"` (raw population estimates for all p eigenvalues).

### `estimate_population_eigenvalues(eigenvalues, c, margin=1.0)`

Per-eigenvalue population estimates: everything `analyze_spikes` gives for
the spikes, plus `"bulk_population"` / `"bulk_sample"` (ascending) from
Ledoit–Wolf pointwise deconvolution of the bulk.

### `ledoit_wolf_shrinkage(eigenvalues, c)` → ndarray

Raw Ledoit–Wolf estimates ξ(λᵢ) — **not** trace-rescaled.

### `shrink_eigenvalues(eigenvalues, c, *, method="auto", parallel=False)` → ndarray

Trace-preserving RIE shrinkage: the sum of the shrunk eigenvalues equals the
original trace exactly.

---

## Module attributes

- `shrinkers.__version__` — `str`, the crate version (single-sourced from
  `Cargo.toml`; e.g. `"0.1.0"`).
- `shrinkers.__doc__` — `str`, short module description.

---

## Notes & caveats

- **Contiguity:** array inputs must be contiguous `float64`. A non-contiguous
  view raises `ValueError`. Use `np.ascontiguousarray(...)` if needed.
- **`eta` default:** when `eta="inferred"`, the value `0.1 / sqrt(p)` is used.
- **Type hints:** a `shrinkers.pyi` stub ships with the package, providing
  `TypedDict` types for every returned dict.
