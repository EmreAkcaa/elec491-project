"""Tests for analysis module using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    compute_correlation_matrix,
    compute_descriptive_stats,
    compute_distance_matrix,
    compute_market_summary,
    get_top_bottom_pairs,
)


@pytest.fixture
def synthetic_returns():
    """Create synthetic log returns for 5 tickers."""
    rng = np.random.default_rng(123)
    n_days = 500
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    # Correlated returns: A and B highly correlated, C and D moderately, E independent
    base = rng.normal(0, 0.02, n_days)
    returns = pd.DataFrame(
        {
            "A": base + rng.normal(0, 0.005, n_days),
            "B": base + rng.normal(0, 0.005, n_days),
            "C": 0.5 * base + rng.normal(0, 0.015, n_days),
            "D": 0.5 * base + rng.normal(0, 0.015, n_days),
            "E": rng.normal(0, 0.02, n_days),
        },
        index=dates,
    )
    return returns


@pytest.fixture
def synthetic_universe():
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E"],
            "company_name": ["Co A", "Co B", "Co C", "Co D", "Co E"],
            "sector": ["Banking", "Banking", "Industrial", "Industrial", "Tech"],
            "provider_symbol": ["A.IS", "B.IS", "C.IS", "D.IS", "E.IS"],
        }
    )


class TestCorrelationMatrix:
    def test_symmetric(self, synthetic_returns):
        corr = compute_correlation_matrix(synthetic_returns, min_periods=50)
        pd.testing.assert_frame_equal(corr, corr.T)

    def test_diagonal_ones(self, synthetic_returns):
        corr = compute_correlation_matrix(synthetic_returns, min_periods=50)
        diag = np.diag(corr.values)
        np.testing.assert_allclose(diag, 1.0, atol=1e-10)

    def test_values_in_range(self, synthetic_returns):
        corr = compute_correlation_matrix(synthetic_returns, min_periods=50)
        assert (corr.values[~np.isnan(corr.values)] >= -1.0).all()
        assert (corr.values[~np.isnan(corr.values)] <= 1.0).all()

    def test_min_periods_produces_nan(self):
        """Pairs with insufficient overlap should produce NaN."""
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        rng = np.random.default_rng(42)

        # X has full data; Y only has first 100 days
        x = pd.Series(rng.normal(0, 0.02, 300), index=dates, name="X")
        y_vals = np.full(300, np.nan)
        y_vals[:100] = rng.normal(0, 0.02, 100)
        y = pd.Series(y_vals, index=dates, name="Y")

        df = pd.concat([x, y], axis=1)
        corr = compute_correlation_matrix(df, min_periods=200)

        # X-Y correlation should be NaN (only 100 overlapping obs, need 200)
        assert np.isnan(corr.loc["X", "Y"])
        # X-X should still be 1.0
        assert corr.loc["X", "X"] == pytest.approx(1.0)

    def test_min_periods_valid_when_sufficient(self):
        """Pairs with enough overlap should produce valid correlation."""
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        rng = np.random.default_rng(42)

        x = pd.Series(rng.normal(0, 0.02, 300), index=dates, name="X")
        y = pd.Series(rng.normal(0, 0.02, 300), index=dates, name="Y")

        df = pd.concat([x, y], axis=1)
        corr = compute_correlation_matrix(df, min_periods=200)

        assert not np.isnan(corr.loc["X", "Y"])
        assert -1.0 <= corr.loc["X", "Y"] <= 1.0


class TestDistanceMatrix:
    def test_values_in_range(self, synthetic_returns):
        corr = compute_correlation_matrix(synthetic_returns, min_periods=50)
        dist = compute_distance_matrix(corr)
        valid = dist.values[~np.isnan(dist.values)]
        assert (valid >= 0.0).all()
        assert (valid <= 2.0).all()

    def test_zero_when_rho_one(self):
        """Distance should be 0 when correlation is 1."""
        corr = pd.DataFrame(
            [[1.0, 1.0], [1.0, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        dist = compute_distance_matrix(corr)
        np.testing.assert_allclose(dist.values, 0.0, atol=1e-10)

    def test_sqrt2_when_rho_zero(self):
        """Distance should be sqrt(2) when correlation is 0."""
        corr = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        dist = compute_distance_matrix(corr)
        assert dist.loc["A", "B"] == pytest.approx(np.sqrt(2), abs=1e-10)


class TestDescriptiveStats:
    def test_output_columns(self, synthetic_returns):
        stats = compute_descriptive_stats(synthetic_returns)
        expected_cols = {
            "ticker", "count", "mean_daily_return", "std_daily_return",
            "annualized_return", "annualized_vol", "min_return", "max_return",
            "skewness", "kurtosis",
        }
        assert set(stats.columns) == expected_cols

    def test_count_matches(self, synthetic_returns):
        stats = compute_descriptive_stats(synthetic_returns)
        for _, row in stats.iterrows():
            assert row["count"] == 500


class TestTopBottomPairs:
    def test_has_both_types(self, synthetic_returns, synthetic_universe):
        corr = compute_correlation_matrix(synthetic_returns, min_periods=50)
        pairs = get_top_bottom_pairs(corr, synthetic_universe, n=3)
        assert "top" in pairs["rank_type"].values
        assert "bottom" in pairs["rank_type"].values

    def test_top_pair_is_ab(self, synthetic_returns, synthetic_universe):
        """A and B should be the most correlated pair by construction."""
        corr = compute_correlation_matrix(synthetic_returns, min_periods=50)
        pairs = get_top_bottom_pairs(corr, synthetic_universe, n=1)
        top = pairs[pairs["rank_type"] == "top"].iloc[0]
        assert {top["ticker_1"], top["ticker_2"]} == {"A", "B"}


class TestMarketSummary:
    def test_keys(self, synthetic_returns):
        corr = compute_correlation_matrix(synthetic_returns, min_periods=50)
        summary = compute_market_summary(corr)
        expected_keys = {
            "avg_pairwise_corr", "median_pairwise_corr", "std_pairwise_corr",
            "min_pairwise_corr", "max_pairwise_corr", "n_pairs", "n_tickers",
        }
        assert set(summary.keys()) == expected_keys

    def test_n_pairs(self, synthetic_returns):
        corr = compute_correlation_matrix(synthetic_returns, min_periods=50)
        summary = compute_market_summary(corr)
        n = corr.shape[0]
        assert summary["n_pairs"] == n * (n - 1) // 2
