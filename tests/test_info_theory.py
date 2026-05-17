"""Tests for the Information-Theory layer (`src/info_theory.py`).

Phase 3 of the mutable-candy rescue plan. The module summarises a
returns panel in mutual information, effective dimensionality, joint
Gaussian entropy, regime KL divergence, and sign-entropy rate.
These tests pin the canonical limit cases for each measure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.info_theory import (
    DEFAULT_CRISIS_DATES,
    LOG2_E,
    _compute_regime_kl,
    _resolve_crisis_dates,
    d_eff,
    gaussian_mi_from_corr,
    gaussian_mi_matrix,
    joint_diff_entropy,
    kl_gaussian_covariances,
    log_det_term,
    nonlinear_excess,
    pairwise_mi_matrix,
    pairwise_mi_value,
    rolling_d_eff_dh,
    sign_entropy_rate,
    top_nonlinear_pairs,
)


# ---------------------------------------------------------------------------
# pairwise_mi_value / pairwise_mi_matrix
# ---------------------------------------------------------------------------

class TestPairwiseMI:
    def test_independent_streams_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 2000)
        y = rng.normal(0, 1, 2000)
        mi = pairwise_mi_value(x, y, n_bins=4)
        # Finite-sample bias is ~0.02 bits at n=2000, 4 bins; allow 0.05.
        assert 0.0 <= mi < 0.05, f"independent MI {mi:.4f} too high"

    def test_identical_streams_near_log_n_bins(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 2000)
        mi = pairwise_mi_value(x, x, n_bins=4)
        assert mi == pytest.approx(math.log2(4), abs=0.05), (
            f"identical-stream MI {mi:.4f} not near log2(4)=2"
        )

    def test_units_switch_bits_vs_nats(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, 1000)
        y = x + 0.1 * rng.normal(0, 1, 1000)
        mi_bits = pairwise_mi_value(x, y, n_bins=4, units="bits")
        mi_nats = pairwise_mi_value(x, y, n_bins=4, units="nats")
        assert mi_bits == pytest.approx(mi_nats * LOG2_E, rel=1e-6)

    def test_short_series_returns_zero(self):
        x = np.arange(10)
        y = np.arange(10)
        assert pairwise_mi_value(x, y, n_bins=4) == 0.0

    def test_matrix_symmetric(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({k: rng.normal(0, 1, 500) for k in "ABC"})
        mi = pairwise_mi_matrix(df, n_bins=4)
        assert mi.shape == (3, 3)
        np.testing.assert_allclose(mi.values, mi.values.T)

    def test_diagonal_equals_marginal_entropy(self):
        rng = np.random.default_rng(4)
        df = pd.DataFrame({"X": rng.normal(0, 1, 1500)})
        mi = pairwise_mi_matrix(df, n_bins=4)
        # Equal-frequency binning + n=1500 → marginal H ≈ log2(4) = 2 bits
        assert mi.iloc[0, 0] == pytest.approx(2.0, abs=0.05)

    def test_unknown_units_raises(self):
        with pytest.raises(ValueError, match="units"):
            pairwise_mi_value(np.zeros(50), np.zeros(50), n_bins=4, units="dits")


# ---------------------------------------------------------------------------
# gaussian_mi_from_corr / gaussian_mi_matrix
# ---------------------------------------------------------------------------

class TestGaussianMI:
    def test_zero_correlation_zero_mi(self):
        assert gaussian_mi_from_corr(0.0) == 0.0

    def test_full_correlation_clipped(self):
        # log(1 − 1²) diverges; the implementation clips to 0.9999.
        mi = gaussian_mi_from_corr(1.0)
        # ½ log(1/(1 − 0.9999²)) ≈ ½ log(50001) → ~4.91 nats ≈ 7.08 bits
        assert mi > 5.0
        assert math.isfinite(mi)

    def test_symmetric_in_rho(self):
        assert gaussian_mi_from_corr(0.7) == pytest.approx(
            gaussian_mi_from_corr(-0.7), rel=1e-12
        )

    def test_matrix_diagonal_zero(self):
        corr = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]], index=list("AB"), columns=list("AB")
        )
        g = gaussian_mi_matrix(corr)
        assert g.iloc[0, 0] == 0.0
        assert g.iloc[1, 1] == 0.0

    def test_matrix_off_diagonal_matches_scalar(self):
        corr = pd.DataFrame(
            [[1.0, 0.3], [0.3, 1.0]], index=list("AB"), columns=list("AB")
        )
        g = gaussian_mi_matrix(corr)
        assert g.iloc[0, 1] == pytest.approx(gaussian_mi_from_corr(0.3), rel=1e-9)


# ---------------------------------------------------------------------------
# nonlinear_excess / top_nonlinear_pairs
# ---------------------------------------------------------------------------

class TestNonlinearExcess:
    def test_excess_zero_for_pure_gaussian_proxy(self):
        idx = list("AB")
        df = pd.DataFrame([[2.0, 0.4], [0.4, 2.0]], index=idx, columns=idx)
        excess = nonlinear_excess(df, df)
        np.testing.assert_allclose(excess.values, np.zeros((2, 2)))

    def test_top_pairs_sorted_descending(self):
        idx = list("ABC")
        excess = pd.DataFrame(
            [[0.0, 0.5, 0.1], [0.5, 0.0, 0.9], [0.1, 0.9, 0.0]],
            index=idx, columns=idx,
        )
        top = top_nonlinear_pairs(excess, top_k=2)
        assert list(top["nonlinear_excess"]) == pytest.approx([0.9, 0.5])
        assert {tuple(sorted([r.ticker_a, r.ticker_b])) for _, r in top.iterrows()} == {
            ("B", "C"), ("A", "B"),
        }


# ---------------------------------------------------------------------------
# d_eff / joint_diff_entropy / log_det_term
# ---------------------------------------------------------------------------

class TestSpectrumMeasures:
    def test_d_eff_diagonal_equals_n(self):
        eigs = np.ones(5)
        assert d_eff(eigs) == pytest.approx(5.0)

    def test_d_eff_one_dominant_eig_near_one(self):
        eigs = np.array([100.0] + [1e-6] * 4)
        assert d_eff(eigs) < 1.1

    def test_d_eff_handles_zero_spectrum(self):
        assert d_eff(np.zeros(4)) == 0.0

    def test_log_det_zero_for_identity(self):
        I = np.eye(4)
        assert log_det_term(I) == pytest.approx(0.0)

    def test_log_det_negative_for_correlated(self):
        cov = np.array([[1.0, 0.9], [0.9, 1.0]])
        # det = 1 − 0.81 = 0.19 → −½ log(0.19) > 0
        assert log_det_term(cov) > 0.5

    def test_joint_entropy_is_finite_for_positive_definite(self):
        cov = np.eye(3) * 2.0
        h = joint_diff_entropy(cov)
        assert math.isfinite(h)
        # Equals ½ (3 log(2πe·2)).
        expected = 0.5 * (3 * math.log(2 * math.pi * math.e * 2.0))
        assert h == pytest.approx(expected, rel=1e-9)

    def test_joint_entropy_nan_for_singular(self):
        cov = np.zeros((3, 3))
        assert math.isnan(joint_diff_entropy(cov))


# ---------------------------------------------------------------------------
# KL between Gaussian covariances
# ---------------------------------------------------------------------------

class TestKLGaussian:
    def test_kl_self_zero(self):
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        assert kl_gaussian_covariances(cov, cov) == pytest.approx(0.0, abs=1e-9)

    def test_kl_strictly_positive(self):
        a = np.array([[1.0, 0.1], [0.1, 1.0]])
        b = np.array([[1.0, 0.9], [0.9, 1.0]])
        kl = kl_gaussian_covariances(a, b)
        assert kl > 0.0

    def test_kl_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            kl_gaussian_covariances(np.eye(2), np.eye(3))

    def test_kl_singular_b_returns_nan(self):
        # KL is mathematically undefined when Σ_b is singular (N(0, Σ_b)
        # doesn't have a density on the full space). The implementation
        # signals this by returning NaN rather than raising.
        a = np.eye(2)
        b = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert math.isnan(kl_gaussian_covariances(a, b))


# ---------------------------------------------------------------------------
# rolling_d_eff_dh + sign_entropy_rate
# ---------------------------------------------------------------------------

class TestRollingAndSignEntropy:
    def test_rolling_window_count(self):
        rng = np.random.default_rng(5)
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        df = pd.DataFrame(rng.normal(0, 1, (300, 4)), index=idx, columns=list("ABCD"))
        rolling = rolling_d_eff_dh(df, window=60, step=20)
        # Expect (300 − 60)/20 + 1 = 13 windows.
        assert len(rolling) == 13
        assert (rolling["d_eff"] > 0).all()

    def test_rolling_handles_all_nan_window(self):
        idx = pd.date_range("2020-01-01", periods=120, freq="B")
        df = pd.DataFrame(
            np.full((120, 3), np.nan), index=idx, columns=list("ABC")
        )
        rolling = rolling_d_eff_dh(df, window=60, step=10)
        assert rolling["d_eff"].isna().all()

    def test_sign_entropy_rate_iid_near_one_bit(self):
        rng = np.random.default_rng(6)
        x = rng.normal(0, 1, 2000)
        rate = sign_entropy_rate(x)
        assert 0.95 <= rate <= 1.05, f"iid sign entropy rate {rate:.3f} not ~1 bit"

    def test_sign_entropy_rate_perfectly_predictable_low(self):
        # Sign alternates → past perfectly predicts current → rate ≈ 0.
        x = np.array([(-1) ** k for k in range(2000)], dtype=float)
        rate = sign_entropy_rate(x)
        assert rate < 0.05, f"deterministic sign entropy rate {rate:.3f} should be ~0"

    def test_sign_entropy_rate_short_series_nan(self):
        assert math.isnan(sign_entropy_rate(np.arange(10)))


# ---------------------------------------------------------------------------
# Crisis spec resolution
# ---------------------------------------------------------------------------

class TestCrisisResolution:
    def _idx(self, start="2019-01-01", end="2024-01-01"):
        return pd.date_range(start, end, freq="B")

    def test_skips_events_outside_panel(self):
        idx = self._idx()
        out_of_range = [{"label": "old", "date": "1990-01-01"}]
        assert _resolve_crisis_dates(idx, out_of_range) == []

    def test_keeps_events_inside_panel(self):
        idx = self._idx()
        specs = _resolve_crisis_dates(idx, DEFAULT_CRISIS_DATES)
        assert len(specs) == 3
        for s in specs:
            assert "calm_start" in s
            assert "crisis_end" in s

    def test_kl_table_runs(self):
        rng = np.random.default_rng(7)
        idx = pd.date_range("2019-01-01", periods=1500, freq="B")
        cols = list("ABCD")
        df = pd.DataFrame(rng.normal(0, 1, (1500, 4)), index=idx, columns=cols)
        specs = _resolve_crisis_dates(idx, DEFAULT_CRISIS_DATES)
        rows = _compute_regime_kl(df, specs)
        assert len(rows) == len(specs)
        for r in rows:
            # Random-Gaussian panel: KL should be finite and small but
            # non-negative.
            assert math.isfinite(r["kl"])
            assert r["kl"] >= 0
