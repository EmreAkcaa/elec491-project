"""Tests for rolling correlation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rolling_correlation import (
    compute_rolling_market_stats,
    compute_rolling_pair_correlation,
    compute_rolling_sector_stats,
    compute_window_correlation,
)


@pytest.fixture
def synthetic_returns():
    """500-day returns for 5 tickers with known structure."""
    rng = np.random.default_rng(42)
    n_days = 500
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

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
def sector_map():
    return {"A": "Banking", "B": "Banking", "C": "Industrial", "D": "Industrial", "E": "Tech"}


class TestRollingMarketStats:
    def test_output_columns(self, synthetic_returns):
        stats = compute_rolling_market_stats(
            synthetic_returns, window=60, step=10, min_periods=30
        )
        expected = {
            "avg_corr", "median_corr", "std_corr", "min_corr", "max_corr",
            "q25_corr", "q75_corr", "n_valid_pairs", "n_tickers_in_window",
        }
        assert set(stats.columns) == expected

    def test_values_in_range(self, synthetic_returns):
        stats = compute_rolling_market_stats(
            synthetic_returns, window=60, step=10, min_periods=30
        )
        assert (stats["avg_corr"] >= -1).all()
        assert (stats["avg_corr"] <= 1).all()
        assert (stats["min_corr"] <= stats["avg_corr"]).all()
        assert (stats["max_corr"] >= stats["avg_corr"]).all()

    def test_step_reduces_points(self, synthetic_returns):
        stats_1 = compute_rolling_market_stats(
            synthetic_returns, window=60, step=1, min_periods=30
        )
        stats_10 = compute_rolling_market_stats(
            synthetic_returns, window=60, step=10, min_periods=30
        )
        assert len(stats_10) < len(stats_1)
        assert len(stats_10) == pytest.approx(len(stats_1) / 10, abs=2)

    def test_larger_window_fewer_points(self, synthetic_returns):
        stats_60 = compute_rolling_market_stats(
            synthetic_returns, window=60, step=5, min_periods=30
        )
        stats_200 = compute_rolling_market_stats(
            synthetic_returns, window=200, step=5, min_periods=100
        )
        assert len(stats_200) < len(stats_60)

    def test_expanding_mode(self, synthetic_returns):
        stats = compute_rolling_market_stats(
            synthetic_returns, window=60, step=10, min_periods=30, expanding=True
        )
        assert len(stats) > 0
        # Expanding window: n_tickers should be constant (all have full data)
        assert (stats["n_tickers_in_window"] == 5).all()

    def test_spearman_method(self, synthetic_returns):
        stats = compute_rolling_market_stats(
            synthetic_returns, window=60, step=50, min_periods=30, method="spearman"
        )
        assert len(stats) > 0
        assert (stats["avg_corr"] >= -1).all()
        assert (stats["avg_corr"] <= 1).all()

    def test_n_valid_pairs(self, synthetic_returns):
        stats = compute_rolling_market_stats(
            synthetic_returns, window=60, step=50, min_periods=30
        )
        # 5 tickers => 5*4/2 = 10 pairs
        assert (stats["n_valid_pairs"] == 10).all()

    def test_percentiles_ordered(self, synthetic_returns):
        stats = compute_rolling_market_stats(
            synthetic_returns, window=60, step=10, min_periods=30
        )
        assert (stats["q25_corr"] <= stats["median_corr"]).all()
        assert (stats["median_corr"] <= stats["q75_corr"]).all()

    def test_empty_returns(self):
        empty = pd.DataFrame()
        stats = compute_rolling_market_stats(empty, window=60)
        assert len(stats) == 0


class TestRollingMarketStatsVectorizedEquivalence:
    """The Pearson + non-expanding path now dispatches to an incremental-sum
    fast implementation (`_compute_rolling_market_stats_fast`). These tests
    pin the fast path's output to the legacy from-scratch loop's output so
    a regression in either implementation surfaces immediately.
    """

    @staticmethod
    def _legacy_loop(returns: pd.DataFrame, window: int, step: int, min_periods: int) -> pd.DataFrame:
        """Inlined copy of the legacy Pearson loop — used only for testing.

        Mirrors `compute_rolling_market_stats(method='pearson', expanding=False)`
        as it existed BEFORE the fast-path dispatch was added. Kept here so
        the test isn't sensitive to future internal refactors of the public
        function (e.g. dispatching to a different fast path).
        """
        n_days = len(returns)
        records = []
        for end_idx in range(window, n_days + 1, step):
            wr = returns.iloc[end_idx - window: end_idx]
            obs = wr.count()
            valid = obs[obs >= min_periods].index
            if len(valid) < 2:
                continue
            corr = wr[valid].corr(method="pearson")
            mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
            upper = corr.values[mask]
            upper = upper[np.isfinite(upper)]
            if len(upper) == 0:
                continue
            records.append({
                "date": wr.index[-1],
                "avg_corr": float(np.mean(upper)),
                "median_corr": float(np.median(upper)),
                "std_corr": float(np.std(upper)),
                "min_corr": float(np.min(upper)),
                "max_corr": float(np.max(upper)),
                "q25_corr": float(np.percentile(upper, 25)),
                "q75_corr": float(np.percentile(upper, 75)),
                "n_valid_pairs": len(upper),
                "n_tickers_in_window": len(valid),
            })
        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df

    def _check_equivalence(self, returns: pd.DataFrame, window: int, step: int, min_periods: int):
        legacy = self._legacy_loop(returns, window=window, step=step, min_periods=min_periods)
        fast = compute_rolling_market_stats(
            returns, window=window, step=step, method="pearson",
            expanding=False, min_periods=min_periods,
        )
        # Same number of windows
        assert len(fast) == len(legacy), f"row count mismatch: fast={len(fast)}, legacy={len(legacy)}"
        # Same index
        pd.testing.assert_index_equal(fast.index, legacy.index)
        # Numeric columns within float-precision tolerance (incremental
        # sums introduce ~1e-15 error per arithmetic op, accumulates over
        # ~T/step steps to ~1e-13 even on the longest universes)
        for col in ["avg_corr", "median_corr", "std_corr", "min_corr",
                    "max_corr", "q25_corr", "q75_corr"]:
            np.testing.assert_allclose(
                fast[col].values, legacy[col].values, atol=1e-10, rtol=0,
                err_msg=f"column {col} diverged",
            )
        # Integer columns exact
        assert (fast["n_valid_pairs"].values == legacy["n_valid_pairs"].values).all()
        assert (fast["n_tickers_in_window"].values == legacy["n_tickers_in_window"].values).all()

    def test_equivalence_no_nan(self, synthetic_returns):
        self._check_equivalence(synthetic_returns, window=60, step=5, min_periods=30)

    def test_equivalence_step_one(self, synthetic_returns):
        self._check_equivalence(synthetic_returns, window=60, step=1, min_periods=30)

    def test_equivalence_large_step(self, synthetic_returns):
        self._check_equivalence(synthetic_returns, window=120, step=21, min_periods=72)

    def test_equivalence_with_nan(self, synthetic_returns):
        """Sprinkle NaN into the input — pairwise-complete observations
        must be honoured by both paths identically."""
        returns_with_nan = synthetic_returns.copy()
        rng = np.random.default_rng(42)
        # Punch ~5% holes
        mask = rng.random(returns_with_nan.shape) < 0.05
        returns_with_nan = returns_with_nan.mask(mask)
        self._check_equivalence(returns_with_nan, window=60, step=5, min_periods=30)

    def test_equivalence_ticker_dropouts(self, synthetic_returns):
        """One ticker has long runs of NaN below the min_periods threshold —
        forces the univariate-filter path on some windows."""
        returns_dropouts = synthetic_returns.copy()
        # Blank out ticker E for the first 200 rows (so windows landing
        # entirely in [0, 200] will drop E from the valid-ticker set)
        returns_dropouts.iloc[:200, returns_dropouts.columns.get_loc("E")] = np.nan
        self._check_equivalence(returns_dropouts, window=60, step=10, min_periods=40)


class TestRollingPairCorrelation:
    def test_pearson_output(self, synthetic_returns):
        corr = compute_rolling_pair_correlation(
            synthetic_returns, "A", "B", window=60, min_periods=30
        )
        assert isinstance(corr, pd.Series)
        valid = corr.dropna()
        assert len(valid) > 0
        assert (valid >= -1).all()
        assert (valid <= 1).all()

    def test_ab_highly_correlated(self, synthetic_returns):
        corr = compute_rolling_pair_correlation(
            synthetic_returns, "A", "B", window=100, min_periods=50
        )
        # A and B are constructed to be highly correlated
        assert corr.dropna().mean() > 0.6

    def test_ae_weakly_correlated(self, synthetic_returns):
        corr = compute_rolling_pair_correlation(
            synthetic_returns, "A", "E", window=100, min_periods=50
        )
        # E is independent
        assert abs(corr.dropna().mean()) < 0.3

    def test_ewm_window_type(self, synthetic_returns):
        corr = compute_rolling_pair_correlation(
            synthetic_returns, "A", "B", window=60, min_periods=30,
            window_type="ewm",
        )
        valid = corr.dropna()
        assert len(valid) > 0
        assert (valid >= -1).all()
        assert (valid <= 1).all()

    def test_expanding_window_type(self, synthetic_returns):
        corr = compute_rolling_pair_correlation(
            synthetic_returns, "A", "B", window=60, min_periods=30,
            window_type="expanding",
        )
        valid = corr.dropna()
        assert len(valid) > 0

    def test_spearman_pair(self, synthetic_returns):
        corr = compute_rolling_pair_correlation(
            synthetic_returns, "A", "B", window=60, min_periods=30,
            method="spearman",
        )
        valid = corr.dropna()
        assert len(valid) > 0
        assert (valid >= -1).all()
        assert (valid <= 1).all()

    def test_ewm_custom_span(self, synthetic_returns):
        corr = compute_rolling_pair_correlation(
            synthetic_returns, "A", "B", window=60, min_periods=30,
            window_type="ewm", ewm_span=30,
        )
        valid = corr.dropna()
        assert len(valid) > 0


class TestRollingSectorStats:
    def test_output_columns(self, synthetic_returns, sector_map):
        stats = compute_rolling_sector_stats(
            synthetic_returns, sector_map, window=60, step=50, min_periods=30
        )
        assert "intra_sector_avg" in stats.columns
        assert "inter_sector_avg" in stats.columns

    def test_has_per_sector_columns(self, synthetic_returns, sector_map):
        stats = compute_rolling_sector_stats(
            synthetic_returns, sector_map, window=60, step=50, min_periods=30
        )
        assert "intra_Banking" in stats.columns
        assert "intra_Industrial" in stats.columns

    def test_intra_higher_for_banking(self, synthetic_returns, sector_map):
        """Banking stocks (A, B) are highly correlated, so intra_Banking should be high."""
        stats = compute_rolling_sector_stats(
            synthetic_returns, sector_map, window=100, step=50, min_periods=50
        )
        # intra_Banking avg should be higher than inter_sector_avg
        avg_intra = stats["intra_Banking"].mean()
        avg_inter = stats["inter_sector_avg"].mean()
        assert avg_intra > avg_inter

    def test_step_reduces_points(self, synthetic_returns, sector_map):
        stats_5 = compute_rolling_sector_stats(
            synthetic_returns, sector_map, window=60, step=5, min_periods=30
        )
        stats_50 = compute_rolling_sector_stats(
            synthetic_returns, sector_map, window=60, step=50, min_periods=30
        )
        assert len(stats_50) < len(stats_5)


class TestWindowCorrelation:
    def test_returns_valid_matrix(self, synthetic_returns):
        end = synthetic_returns.index[-1]
        corr = compute_window_correlation(
            synthetic_returns, end_date=end, window=100, min_periods=50
        )
        assert not corr.empty
        assert corr.shape[0] == corr.shape[1]
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_symmetric(self, synthetic_returns):
        end = synthetic_returns.index[-1]
        corr = compute_window_correlation(
            synthetic_returns, end_date=end, window=100, min_periods=50
        )
        pd.testing.assert_frame_equal(corr, corr.T)

    def test_midpoint_date(self, synthetic_returns):
        mid = synthetic_returns.index[250]
        corr = compute_window_correlation(
            synthetic_returns, end_date=mid, window=100, min_periods=50
        )
        assert not corr.empty

    def test_insufficient_data(self, synthetic_returns):
        early = synthetic_returns.index[5]
        corr = compute_window_correlation(
            synthetic_returns, end_date=early, window=100, min_periods=50
        )
        assert corr.empty

    def test_spearman(self, synthetic_returns):
        end = synthetic_returns.index[-1]
        corr = compute_window_correlation(
            synthetic_returns, end_date=end, window=100, min_periods=50,
            method="spearman",
        )
        assert not corr.empty
        vals = corr.values[~np.eye(corr.shape[0], dtype=bool)]
        assert (vals >= -1).all()
        assert (vals <= 1).all()
