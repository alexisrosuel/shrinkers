# `shrinkers` Python API Reference

The `shrinkers` module is a PyO3 binding over the Rust RMT shrinkage kernel.
It exposes these functions:

| Function | Purpose |
|---|---|
| `deconvolve_spiked` | **Primary**: spiked + bulk cleaning via free-probability deconvolution |
| `clean_correlation_matrix` | Clean a full correlation matrix (RIE + eigenvector overlaps) |
| `clean_correlation_matrix_complex` | Same, for a complex Hermitian correlation matrix (e.g. a spectral coherence matrix) |
| `deconvolve_correlation_matrix_complex` | Spiked decomposition (BEMA + inverse BBP + Ledoit–Wolf bulk) of a complex Hermitian correlation matrix |
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
  `"chebcode_fast"` (alias `"chebf"`; ~1e-5, fastest at every size),
  `"chebcode_balanced"` (alias `"chebb"`; ~5e-10),
  `"chebcode_xtreme"` (alias `"chebx"`; ~6e-13), `"hodlr"`, `"ewald"`,
  `"dst"`, `"speed_auto"` (alias `"speed"`), `"accuracy_auto"`
  (alias `"accuracy"`), or `"auto"`.
  `"auto"` is the **speed policy**, not an exact fallback: it resolves
  through the same measured Pareto table as `"speed_auto"` (identical pick)
  and is therefore approximate — ~1e-5 at the retuned `chebcode_fast`
  point, which the table now picks at every size (it used to fall back to
  the FFT bank, ~4e-5, above p ≈ 20 000). For a machine-precision
  result ask for `"blocked"` (the `stieltjes_transform` default) or
  `"blocked_tiled"`, or for `"accuracy_auto"`. On the deconvolution grid a
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

### `clean_correlation_matrix(correlation, c)`

Clean a sample **correlation matrix** via RIE eigenvalue shrinkage + eigenvector
angular overlap correction. The spectral decomposition is computed internally,
then the RIE-cleaned eigenvalues and the theoretical eigenvector alignment are
returned alongside the cleaned covariance matrix.

**Parameters**

- `correlation` — `np.ndarray[float64]`, shape `(p, p)`. Sample correlation
  matrix (symmetric, finite). Must be contiguous.
- `c` — `float`. Concentration ratio $p/n$, in $(0, 1]$.

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
```

### `clean_correlation_matrix_complex(correlation, c)`

The same estimator for a **complex Hermitian** correlation matrix. This is the
frequency-domain case: a spectral coherence matrix

$$C(\nu) = \mathrm{dg}(S(\nu))^{-1/2}\,S(\nu)\,\mathrm{dg}(S(\nu))^{-1/2}$$

built from Fourier coefficients has unit diagonal and a Marchenko–Pastur bulk
with aspect ratio $c = M/B$ ($M$ channels, $B$ smoothing span), so the RMT
machinery applies unchanged — only the eigendecomposition and the reconstruction
change (the conjugate transpose replaces the transpose).

**Parameters**

- `correlation` — `np.ndarray[complex128]`, shape `(p, p)`. Hermitian (to
  `1e-12` relative tolerance), finite. Must be contiguous.
- `c` — `float`. Concentration ratio $p/n$, in $(0, 1]$.

**Returns**

- `dict` with the same keys as `clean_correlation_matrix`. `"covariance"` is
  `np.ndarray[complex128]` of shape `(p, p)` (Hermitian, positive definite) and
  `"eigenvectors"` is `np.ndarray[complex128]` of shape `(p, p)`;
  `"eigenvalues"`, `"overlaps"` and `"sigma2"` stay real.

**Raises**

- `ValueError` if the matrix is not square, not finite, or not Hermitian.

**Example**

```python
import numpy as np
from shrinkers import clean_correlation_matrix_complex

# Smoothed periodogram coherence matrix at frequency nu (M, M), complex
C = dg(S) ** -0.5 @ S @ dg(S) ** -0.5
res = clean_correlation_matrix_complex(C, c=M / B)

cleaned = res["covariance"]       # Hermitian cleaned correlation matrix
```

**Note.** `clean_correlation_matrix_complex` answers the *cleaning* question
(RIE on every eigenvalue, spike directions reweighted by their angular overlap).
The *estimation* question — how many coherent modes, their debiased eigenvalues,
the bulk spectrum — is answered by the sibling below.

### `deconvolve_correlation_matrix_complex(correlation, c, margin=1.0)`

The matrix-level counterpart of `estimate_population_eigenvalues`, for a
**complex Hermitian** correlation matrix. One pass from the matrix to the whole
spiked split:

1. the sample eigensystem of the input matrix (`"eigenvalues"` ascending,
   `"eigenvectors"` as complex ascending columns);
2. BEMA spike detection, inverse-BBP debiasing of the spikes, and Ledoit–Wolf /
   RIE pointwise deconvolution of the bulk.

Use this rather than calling the cleaning entry point when you want the
population *spectrum*; the eigenvectors come back as a by-product of the same
eigendecomposition, so a caller that also needs the coherent directions does not
pay for a second decomposition — and the split is guaranteed to come from the
*same* eigenvalues that are returned.

**Parameters**

- `correlation` — `np.ndarray[complex128]`, shape `(p, p)`. Hermitian (to
  `1e-12` relative tolerance), finite. Must be contiguous.
- `c` — `float`. Concentration ratio $p/n$, in $(0, 1]$.
- `margin` — `float`, default `1.0`. Multiplicative margin above the fitted
  bulk edge for spike detection (slightly above `1.0` adds robustness).

**Returns**

- `dict` with keys:
  - `"eigenvalues"` — `np.ndarray[float64]`, shape `(p,)`. Sample eigenvalues
    of the input matrix, ascending.
  - `"eigenvectors"` — `np.ndarray[complex128]`, shape `(p, p)`. Sample
    eigenvectors as columns, ascending, matching `"eigenvalues"`.
  - `"k"`, `"spikes"`, `"spike_sample"`, `"bulk_edge"`, `"sigma2"`,
    `"bulk_population"`, `"bulk_sample"` — the same population split as
    `estimate_population_eigenvalues`.

**Raises**

- `ValueError` if the matrix is not square, not finite, or not Hermitian, or if
  `c` / `margin` are out of range.

**Example**

```python
import numpy as np
from shrinkers import deconvolve_correlation_matrix_complex

# Spectral coherence matrix at frequency nu, see above
res = deconvolve_correlation_matrix_complex(C, c=M / B, margin=1.05)

print(res["k"])                    # number of coherent modes
print(res["spikes"])               # debiased population eigenvalues of the modes
print(res["eigenvectors"][:, -res["k"]:])  # the coherent directions
```

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
