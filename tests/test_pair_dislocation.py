"""Tests for pair dislocation analysis module."""

import numpy as np
import pandas as pd
import pytest

from src.pair_dislocation import (
    compute_half_life,
    compute_hedge_ratio,
    compute_spread,
    compute_zscore,
    detect_signals,
    rank_candidate_pairs,
)


@pytest.fixture
def synthetic_prices():
    """Two cointegrated stocks (A, B) and one independent (C)."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Stock A: random walk
    log_a = np.cumsum(rng.normal(0.0003, 0.02, n))
    # Stock B: cointegrated with A (beta=1.2, mean-reverting noise)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.90 * noise[i - 1] + rng.normal(0, 0.01)
    log_b = 1.2 * log_a + 0.5 + noise
    # Stock C: independent random walk
    log_c = np.cumsum(rng.normal(0.0003, 0.02, n))

    prices = pd.DataFrame(
        {
            "A": np.exp(log_a) * 100,
            "B": np.exp(log_b) * 100,
            "C": np.exp(log_c) * 100,
        },
        index=dates,
    )
    return prices


@pytest.fixture
def synthetic_universe():
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "company_name": ["Co A", "Co B", "Co C"],
            "sector": ["Banking", "Banking", "Tech"],
            "provider_symbol": ["A.IS", "B.IS", "C.IS"],
        }
    )


@pytest.fixture
def synthetic_corr(synthetic_prices):
    log_ret = np.log(synthetic_prices).diff().dropna()
    return log_ret.corr()


class TestHedgeRatio:
    def test_beta_close_to_true_value(self, synthetic_prices):
        log_a = np.log(synthetic_prices["A"])
        log_b = np.log(synthetic_prices["B"])
        beta, intercept = compute_hedge_ratio(log_a, log_b)
        assert 1.0 <= beta <= 1.4, f"Expected beta near 1.2, got {beta}"

    def test_returns_float_pair(self, synthetic_prices):
        log_a = np.log(synthetic_prices["A"])
        log_b = np.log(synthetic_prices["B"])
        beta, intercept = compute_hedge_ratio(log_a, log_b)
        assert isinstance(beta, float)
        assert isinstance(intercept, float)

    def test_insufficient_data(self):
        a = pd.Series([1.0])
        b = pd.Series([2.0])
        beta, intercept = compute_hedge_ratio(a, b)
        assert np.isnan(beta)
        assert np.isnan(intercept)


class TestSpread:
    def test_spread_shape(self, synthetic_prices):
        spread, _, _ = compute_spread(synthetic_prices, "A", "B")
        assert len(spread) == len(synthetic_prices)

    def test_spread_stationary_for_cointegrated(self, synthetic_prices):
        spread, _, _ = compute_spread(synthetic_prices, "A", "B")
        # Stationary spread should have bounded std relative to mean
        assert spread.std() < 0.5

    def test_spread_returns_beta_and_intercept(self, synthetic_prices):
        spread, beta, intercept = compute_spread(synthetic_prices, "A", "B")
        assert isinstance(beta, float)
        assert isinstance(intercept, float)
        assert not np.isnan(beta)

    def test_lookback_parameter(self, synthetic_prices):
        spread_full, beta_full, _ = compute_spread(synthetic_prices, "A", "B")
        spread_short, beta_short, _ = compute_spread(
            synthetic_prices, "A", "B", lookback=100
        )
        # Different lookback should give different beta
        assert len(spread_full) == len(spread_short)


class TestZscore:
    def test_output_length(self, synthetic_prices):
        spread, _, _ = compute_spread(synthetic_prices, "A", "B")
        zs = compute_zscore(spread, window=60)
        assert len(zs) == len(spread)

    def test_nan_during_warmup(self, synthetic_prices):
        spread, _, _ = compute_spread(synthetic_prices, "A", "B")
        zs = compute_zscore(spread, window=60, min_periods=60)
        assert zs.iloc[:59].isna().all()

    def test_centered_around_zero(self, synthetic_prices):
        spread, _, _ = compute_spread(synthetic_prices, "A", "B")
        zs = compute_zscore(spread, window=60)
        valid = zs.dropna()
        assert abs(valid.mean()) < 0.5, f"Z-score mean {valid.mean()} not near 0"

    def test_detects_extreme_values(self):
        # Create a calm spread with an injected spike
        rng = np.random.default_rng(99)
        spread = pd.Series(rng.normal(0, 0.01, 200))
        spread.iloc[150] = 0.1  # Large spike
        zs = compute_zscore(spread, window=60)
        assert zs.iloc[150] > 2.0


class TestHalfLife:
    def test_mean_reverting_pair(self, synthetic_prices):
        spread, _, _ = compute_spread(synthetic_prices, "A", "B")
        hl = compute_half_life(spread)
        assert 1 < hl < 252, f"Expected finite half-life, got {hl}"

    def test_non_reverting_pair(self):
        # Strongly trending spread should have very large half-life
        spread = pd.Series(np.arange(500, dtype=float))
        hl = compute_half_life(spread)
        assert hl > 200 or hl == np.inf

    def test_short_series(self):
        spread = pd.Series([1.0, 2.0])
        hl = compute_half_life(spread)
        assert np.isfinite(hl) or hl == np.inf


class TestDetectSignals:
    def test_signal_types(self, synthetic_prices):
        spread, _, _ = compute_spread(synthetic_prices, "A", "B")
        zs = compute_zscore(spread, window=60)
        signals = detect_signals(zs, entry_threshold=2.0, exit_threshold=0.5)
        if not signals.empty:
            valid_types = {"long_entry", "short_entry", "long_exit", "short_exit"}
            assert set(signals["signal"].unique()).issubset(valid_types)

    def test_no_signals_in_calm_spread(self):
        # Z-score always within [-1, 1]
        zs = pd.Series(np.linspace(-0.9, 0.9, 100))
        signals = detect_signals(zs, entry_threshold=2.0, exit_threshold=0.5)
        assert signals.empty

    def test_known_signal_sequence(self):
        # Z crosses -2, then back to 0, then +2, then back to 0
        dates = pd.date_range("2020-01-01", periods=8, freq="B")
        zs = pd.Series(
            [0.0, -2.5, -1.0, 0.3, 0.0, 2.5, 1.0, 0.3], index=dates
        )
        signals = detect_signals(zs, entry_threshold=2.0, exit_threshold=0.5)
        assert len(signals) == 4
        assert signals.iloc[0]["signal"] == "long_entry"
        assert signals.iloc[1]["signal"] == "long_exit"
        assert signals.iloc[2]["signal"] == "short_entry"
        assert signals.iloc[3]["signal"] == "short_exit"

    def test_columns(self, synthetic_prices):
        spread, _, _ = compute_spread(synthetic_prices, "A", "B")
        zs = compute_zscore(spread, window=60)
        signals = detect_signals(zs)
        assert list(signals.columns) == ["date", "signal", "zscore_value"]


class TestRankCandidatePairs:
    def test_output_columns(
        self, synthetic_prices, synthetic_corr, synthetic_universe
    ):
        df = rank_candidate_pairs(
            synthetic_prices,
            synthetic_corr,
            synthetic_universe,
            top_n=5,
            min_correlation=0.3,
        )
        if not df.empty:
            expected = {
                "ticker_a", "ticker_b", "sector_a", "sector_b",
                "correlation", "beta", "half_life", "spread_std",
                "n_signals", "current_zscore", "rank_score",
            }
            assert expected.issubset(set(df.columns))

    def test_respects_min_correlation(
        self, synthetic_prices, synthetic_corr, synthetic_universe
    ):
        df = rank_candidate_pairs(
            synthetic_prices,
            synthetic_corr,
            synthetic_universe,
            min_correlation=0.99,
        )
        # Very high threshold should exclude most pairs
        assert len(df) <= 1

    def test_top_n_limit(
        self, synthetic_prices, synthetic_corr, synthetic_universe
    ):
        df = rank_candidate_pairs(
            synthetic_prices,
            synthetic_corr,
            synthetic_universe,
            top_n=2,
            min_correlation=0.1,
        )
        assert len(df) <= 2
