"""Python API tests for the shrinkers PyO3 extension.

These require the extension to be built and installed into the active
environment first:

    pixi run build      # maturin develop --release
    pixi run test-py    # pytest tests
"""

from __future__ import annotations

import numpy as np
import pytest

import shrinkers as rk

# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def sample_spectrum(p: int = 200, seed: int = 0) -> np.ndarray:
    """A smooth, strictly positive, ascending spectrum (bulk-like)."""
    rng = np.random.default_rng(seed)
    evals = np.sort(0.5 + rng.uniform(0.0, 1.5, size=p))
    return np.ascontiguousarray(evals)


def spiked_spectrum(p: int = 300, spikes=(5.0, 7.0, 10.0), seed: int = 1) -> np.ndarray:
    """A bulk spectrum with a few clear spikes above the bulk edge."""
    bulk = sample_spectrum(p, seed)
    return np.sort(np.concatenate([bulk, np.asarray(spikes, dtype=np.float64)]))


# ──────────────────────────────────────────────
#  deconvolve_spiked
# ──────────────────────────────────────────────

class TestDeconvolveSpiked:
    def test_result_shape_and_keys(self):
        res = rk.deconvolve_spiked(spiked_spectrum(), c=0.25, n_points=50)
        assert set(res) == {"k", "spikes", "spike_sample", "bulk_edge", "sigma2", "bulk"}
        assert isinstance(res["k"], int)
        bulk = res["bulk"]
        assert set(bulk) == {
            "lambda_grid",
            "density",
            "w_re",
            "sample_stieltjes_real",
            "sample_stieltjes_imag",
            "population_stieltjes_real",
            "population_stieltjes_imag",
        }
        n = len(bulk["lambda_grid"])
        assert n == 50
        for key in bulk:
            assert len(bulk[key]) == n

    def test_detects_clear_spikes(self):
        evals = spiked_spectrum()
        res = rk.deconvolve_spiked(evals, c=0.25, n_points=50)
        assert res["k"] == 3
        # Population spikes below their (BBP-biased upward) sample values.
        assert res["spikes"][0] < res["spike_sample"][0]
        # Descending order.
        assert res["spikes"][0] > res["spikes"][1] > res["spikes"][2]

    def test_no_spikes_on_pure_bulk(self):
        res = rk.deconvolve_spiked(sample_spectrum(), c=0.25, n_points=50)
        assert res["k"] == 0
        assert len(res["spikes"]) == 0

    def test_density_is_finite(self):
        res = rk.deconvolve_spiked(sample_spectrum(), c=0.3, n_points=80)
        assert np.all(np.isfinite(res["bulk"]["density"]))

    def test_method_kwarg_accepted(self):
        a = rk.deconvolve_spiked(
            sample_spectrum(120), c=0.25, n_points=40, method="blocked"
        )
        b = rk.deconvolve_spiked(
            sample_spectrum(120), c=0.25, n_points=40, method="blocked_tiled"
        )
        # Two exact kernels must agree closely on the density grid.
        np.testing.assert_allclose(a["bulk"]["density"], b["bulk"]["density"], atol=1e-10)

    def test_unsorted_input_ok(self):
        rng = np.random.default_rng(7)
        evals = spiked_spectrum()
        shuffled = rng.permutation(evals)
        a = rk.deconvolve_spiked(evals, c=0.25, n_points=40)
        b = rk.deconvolve_spiked(shuffled, c=0.25, n_points=40)
        assert a["k"] == b["k"]
        np.testing.assert_allclose(a["spikes"], b["spikes"], rtol=1e-12)

    def test_nan_raises_value_error(self):
        evals = sample_spectrum()
        evals[3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            rk.deconvolve_spiked(evals, c=0.25)

    def test_meaningfully_negative_eigenvalue_raises(self):
        evals = sample_spectrum()
        evals[2] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            rk.deconvolve_spiked(evals, c=0.25)

    def test_roundoff_negative_eigenvalues_clamped(self):
        # Centered covariances produce ~-1e-15 dust; the boundary clamps it
        # to zero instead of rejecting the call.
        evals = sample_spectrum()
        evals[2] = -3e-15
        res = rk.deconvolve_spiked(evals, c=0.25, n_points=40)
        assert "bulk" in res

    def test_bad_c_raises(self):
        evals = sample_spectrum()
        with pytest.raises(ValueError, match="concentration"):
            rk.deconvolve_spiked(evals, c=0.0)
        with pytest.raises(ValueError, match="concentration"):
            rk.deconvolve_spiked(evals, c=1.5)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            rk.deconvolve_spiked(sample_spectrum(), c=0.25, method="warp9")


# ──────────────────────────────────────────────
#  clean_correlation_matrix
# ──────────────────────────────────────────────

class TestCleanCorrelationMatrix:
    @pytest.fixture()
    def sample_corr(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((400, 60))
        corr = np.corrcoef(x, rowvar=False)
        return np.ascontiguousarray(corr)

    def test_shapes_and_symmetry(self, sample_corr):
        p = sample_corr.shape[0]
        res = rk.clean_correlation_matrix(sample_corr, c=60 / 400)
        cov = res["covariance"]
        assert cov.shape == (p, p)
        assert np.allclose(cov, cov.T, atol=1e-12)
        assert res["eigenvalues"].shape == (p,)
        assert res["overlaps"].shape == (p,)
        assert 0.0 <= res["sigma2"]

    def test_positive_diagonal(self, sample_corr):
        res = rk.clean_correlation_matrix(sample_corr, c=60 / 400)
        assert np.all(np.diag(res["covariance"]) > 0)

    def test_nonsquare_raises(self):
        with pytest.raises(ValueError, match="square"):
            rk.clean_correlation_matrix(np.ones((3, 4)), c=0.5)

    def test_nonfinite_raises(self, sample_corr):
        bad = sample_corr.copy()
        bad[0, 1] = np.inf
        with pytest.raises(ValueError, match="finite"):
            rk.clean_correlation_matrix(bad, c=0.15)


# ──────────────────────────────────────────────
#  direct_precision_shrinkage
# ──────────────────────────────────────────────

class TestDirectPrecisionShrinkage:
    def test_output_finite_positive(self):
        res = rk.direct_precision_shrinkage(sample_spectrum(150), c=0.3)
        out = res["precision_eigenvalues"]
        assert out.shape == (150,)
        assert np.all(np.isfinite(out))
        assert np.all(out > 0)

    def test_identity_population_near_one(self):
        evals = np.linspace(0.9, 1.1, 200)
        res = rk.direct_precision_shrinkage(np.ascontiguousarray(evals), c=0.3)
        mean = float(np.mean(res["precision_eigenvalues"]))
        assert abs(mean - 1.0) < 0.1

    def test_zero_eigenvalue_accepted(self):
        # Zeros (and round-off dust) are tolerated; only meaningful
        # negativity is rejected.
        evals = sample_spectrum()
        evals[0] = 0.0
        res = rk.direct_precision_shrinkage(evals, c=0.3)
        assert np.all(np.isfinite(res["precision_eigenvalues"]))

    def test_meaningfully_negative_raises(self):
        evals = sample_spectrum()
        evals[0] = -0.5
        with pytest.raises(ValueError, match="non-negative"):
            rk.direct_precision_shrinkage(evals, c=0.3)


# ──────────────────────────────────────────────
#  stieltjes_transform
# ──────────────────────────────────────────────

class TestStieltjesTransform:
    def test_keys_and_imag_positive_convention(self):
        res = rk.stieltjes_transform(sample_spectrum(100))
        assert set(res) == {"real", "imag"}
        # The kernel computes S(λ) = (1/p) Σ_j 1/((λ-λ_j) - iη), i.e.
        # convention B: Im[S] > 0 (see src/deconvolution/mod.rs).
        assert np.all(res["imag"] > 0)

    def test_exact_methods_agree(self):
        evals = sample_spectrum(80)
        a = rk.stieltjes_transform(evals, method="naive")
        b = rk.stieltjes_transform(evals, method="blocked_tiled")
        np.testing.assert_allclose(a["real"], b["real"], atol=1e-11)
        np.testing.assert_allclose(a["imag"], b["imag"], atol=1e-11)

    def test_f32_dtype_and_close(self):
        evals = sample_spectrum(100)
        f64 = rk.stieltjes_transform(evals, precision="f64")
        f32 = rk.stieltjes_transform(evals, precision="f32")
        assert f32["real"].dtype == np.float32
        np.testing.assert_allclose(f32["real"], f64["real"], rtol=5e-2)

    def test_rayon_matches_seq(self):
        evals = sample_spectrum(150)
        seq = rk.stieltjes_transform(evals, parallelism="seq")
        par = rk.stieltjes_transform(evals, parallelism="rayon")
        np.testing.assert_allclose(seq["real"], par["real"], atol=1e-12)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            rk.stieltjes_transform(np.array([], dtype=np.float64))

    def test_bad_eta_raises(self):
        with pytest.raises(ValueError, match="eta"):
            rk.stieltjes_transform(sample_spectrum(50), eta=-0.1)

    def test_bad_parallelism_raises(self):
        with pytest.raises(ValueError, match="parallelism"):
            rk.stieltjes_transform(sample_spectrum(50), parallelism="threads")

    def test_non_contiguous_raises(self):
        evals = sample_spectrum(20)[::2]  # strided view
        with pytest.raises(ValueError, match="contiguous"):
            rk.stieltjes_transform(evals)


# ──────────────────────────────────────────────
#  Spiked-model toolkit
# ──────────────────────────────────────────────

class TestSpikedToolkit:
    def test_detect_spikes_bema(self):
        res = rk.detect_spikes_bema(spiked_spectrum(), c=0.25)
        assert set(res) == {"k", "spike_indices", "bulk_edge", "sigma2"}
        assert res["k"] == 3
        assert res["sigma2"] > 0

    def test_detect_spikes_tracy_widom(self):
        res = rk.detect_spikes_tracy_widom(spiked_spectrum(), c=0.25)
        assert res["k"] >= 1

    def test_inverse_bbp_scalar_and_array(self):
        ell = rk.inverse_bbp(6.0, c=0.25, sigma2=1.0)
        assert isinstance(ell, float)
        out = rk.inverse_bbp(np.array([6.0, 8.0]), c=0.25, sigma2=1.0)
        assert out.shape == (2,)
        assert np.all(out < [6.0, 8.0])  # BBP bias is upward

    def test_analyze_spikes(self):
        res = rk.analyze_spikes(spiked_spectrum(), c=0.25)
        assert res["k"] == 3
        assert len(res["overlaps"]) == 3
        assert len(res["ledoit_wolf"]) == len(spiked_spectrum())

    def test_estimate_population_eigenvalues(self):
        evals = spiked_spectrum()
        res = rk.estimate_population_eigenvalues(evals, c=0.25)
        assert res["k"] == 3
        assert len(res["bulk_population"]) == len(evals) - 3
        assert len(res["bulk_sample"]) == len(evals) - 3

    def test_ledoit_wolf_and_shrink_eigenvalues_lengths(self):
        evals = sample_spectrum(120)
        lw = rk.ledoit_wolf_shrinkage(evals, c=0.3)
        shrunk = rk.shrink_eigenvalues(evals, c=0.3)
        assert lw.shape == (120,)
        assert shrunk.shape == (120,)
        # shrink_eigenvalues preserves the trace.
        np.testing.assert_allclose(shrunk.sum(), evals.sum(), rtol=1e-10)

    def test_shrink_eigenvalues_method_kwarg(self):
        evals = sample_spectrum(100)
        a = rk.shrink_eigenvalues(evals, c=0.3, method="autovec")
        b = rk.shrink_eigenvalues(
            evals, c=0.3, method="blocked_tiled", parallel="rayon"
        )
        np.testing.assert_allclose(a, b, atol=1e-10)


# ──────────────────────────────────────────────
#  Module metadata
# ──────────────────────────────────────────────

def test_version_present():
    assert isinstance(rk.__version__, str)
    major = int(rk.__version__.split(".")[0])
    assert major >= 0


def test_gil_released_during_compute():
    """Heavy compute must release the GIL.

    Discriminator: the worker thread enters the Rust kernel (which either
    holds the GIL for the whole computation or releases it via py.detach).
    The main thread spins on pure-Python bytecode; those spins can only
    interleave with the kernel if the GIL was released.
    """
    import threading

    evals = sample_spectrum(20000)
    done = threading.Event()

    def work():
        rk.stieltjes_transform(evals)
        done.set()

    t = threading.Thread(target=work)
    t.start()

    spins = 0
    while not done.is_set():
        sum(range(1000))  # pure-Python work; requires the GIL
        spins += 1

    t.join(timeout=10)
    assert not t.is_alive()
    assert spins > 50, (
        f"main thread managed only {spins} spins during the Rust call — "
        "the GIL appears to be held for the whole computation"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

# ──────────────────────────────────────────────
#  None / "inferred" sentinels
# ──────────────────────────────────────────────

class TestInferredSentinels:
    """Every kwarg documented as 'float or None/"inferred"' must accept all
    three spellings (regression: None used to raise ValueError on the
    InferredF64 extractor, and "inferred" used to TypeError on plain
    Option<f64> parameters)."""

    def test_stieltjes_transform_cutoff_none(self):
        evals = sample_spectrum(80)
        a = rk.stieltjes_transform(evals, cutoff=None)
        b = rk.stieltjes_transform(evals, cutoff="inferred")
        c = rk.stieltjes_transform(evals, cutoff=10.0)
        # None and "inferred" are synonyms: identical disabled-cutoff call.
        np.testing.assert_array_equal(a["real"], b["real"])
        np.testing.assert_array_equal(a["imag"], b["imag"])
        # Enabled cutoff stays finite and same-shaped.
        assert set(c) == {"real", "imag"}
        assert np.all(np.isfinite(c["real"]))

    def test_deconvolve_spiked_cutoff_none(self):
        res = rk.deconvolve_spiked(spiked_spectrum(), c=0.25, n_points=50,
                                   cutoff=None)
        assert res["k"] >= 0

    def test_tracy_widom_sigma2_sentinels_agree(self):
        evals = spiked_spectrum(200, spikes=(6.0,), seed=3)
        a = rk.detect_spikes_tracy_widom(evals, c=0.25, sigma2=None)
        b = rk.detect_spikes_tracy_widom(evals, c=0.25, sigma2="inferred")
        assert a["k"] == b["k"]
        assert a["sigma2"] == pytest.approx(b["sigma2"])
