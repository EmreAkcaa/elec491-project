"""Tests for preprocessing module using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    compute_coverage,
    compute_log_returns,
    filter_by_coverage,
    flag_anomalies,
)


@pytest.fixture
def synthetic_prices():
    """Create synthetic price data with known coverage patterns."""
    dates = pd.date_range("2020-01-01", periods=1000, freq="B")
    rng = np.random.default_rng(42)

    # Ticker A: full coverage
    a_prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 1000)))
    # Ticker B: 95% coverage (50 NaN days)
    b_prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 1000)))
    nan_idx = rng.choice(1000, size=50, replace=False)
    b_prices[nan_idx] = np.nan
    # Ticker C: 80% coverage (200 NaN days) — should be dropped at 90% threshold
    c_prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 1000)))
    nan_idx_c = rng.choice(1000, size=200, replace=False)
    c_prices[nan_idx_c] = np.nan

    return pd.DataFrame(
        {"A": a_prices, "B": b_prices, "C": c_prices}, index=dates
    )


def test_compute_coverage(synthetic_prices):
    coverage = compute_coverage(synthetic_prices)
    assert len(coverage) == 3
    assert set(coverage.columns) == {"ticker", "total_days", "available_days", "coverage_pct"}

    a_row = coverage[coverage["ticker"] == "A"].iloc[0]
    assert a_row["coverage_pct"] == 1.0

    c_row = coverage[coverage["ticker"] == "C"].iloc[0]
    assert c_row["coverage_pct"] == pytest.approx(0.80, abs=0.01)


def test_filter_by_coverage_90pct(synthetic_prices):
    filtered, _, coverage = filter_by_coverage(
        synthetic_prices, pd.DataFrame(), threshold=0.90
    )
    # A (100%) and B (95%) should pass; C (80%) should be dropped
    assert "A" in filtered.columns
    assert "B" in filtered.columns
    assert "C" not in filtered.columns


def test_filter_keeps_nans(synthetic_prices):
    """Verify that filter does NOT do an inner join — NaNs are preserved."""
    filtered, _, _ = filter_by_coverage(
        synthetic_prices, pd.DataFrame(), threshold=0.90
    )
    # B has NaN days; they should still be NaN after filtering
    assert filtered["B"].isna().sum() > 0


def test_compute_log_returns():
    prices = pd.DataFrame(
        {"X": [100.0, 105.0, 110.25]},
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )
    returns = compute_log_returns(prices)
    assert len(returns) == 2

    expected_r1 = np.log(105.0 / 100.0)
    expected_r2 = np.log(110.25 / 105.0)
    assert returns["X"].iloc[0] == pytest.approx(expected_r1, abs=1e-10)
    assert returns["X"].iloc[1] == pytest.approx(expected_r2, abs=1e-10)


def test_log_returns_with_nans():
    """Log returns propagate NaN correctly."""
    prices = pd.DataFrame(
        {"X": [100.0, np.nan, 110.0, 115.0]},
        index=pd.date_range("2020-01-01", periods=4, freq="B"),
    )
    returns = compute_log_returns(prices)
    # After NaN price, return should also be NaN
    assert returns["X"].isna().sum() >= 1


def test_flag_anomalies():
    returns = pd.DataFrame(
        {"X": [0.01, 0.02, 0.50, -0.40, 0.001]},
        index=pd.date_range("2020-01-01", periods=5, freq="B"),
    )
    anomalies = flag_anomalies(returns, threshold=0.30)
    assert len(anomalies) == 2
    assert set(anomalies["ticker"]) == {"X"}
    assert 0.50 in anomalies["return_value"].values
    assert -0.40 in anomalies["return_value"].values


def test_manual_anomaly_nulls_applied():
    """Mirror of the masking loop in run_preprocessing: each (ticker, date)
    in manual_anomaly_nulls must NaN-mask the corresponding log_returns cell
    before anomaly flagging. Missing tickers/dates skip cleanly."""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = pd.DataFrame(
        {"AAA": np.linspace(100, 110, 10), "BBB": np.linspace(50, 55, 10)},
        index=dates,
    )
    log_returns = compute_log_returns(prices)

    # Apply the same in-place mask logic that run_preprocessing uses
    overrides = [
        ("AAA", "2020-01-03"),
        ("BBB", "2020-01-07"),
        ("AAA", "2099-12-31"),   # missing date -> should skip
        ("ZZZ", "2020-01-06"),   # missing ticker -> should skip
    ]
    applied = skipped = 0
    for ticker, date_str in overrides:
        ts = pd.Timestamp(date_str)
        if ticker in log_returns.columns and ts in log_returns.index:
            log_returns.loc[ts, ticker] = np.nan
            applied += 1
        else:
            skipped += 1

    assert applied == 2
    assert skipped == 2
    assert np.isnan(log_returns.loc["2020-01-03", "AAA"])
    assert np.isnan(log_returns.loc["2020-01-07", "BBB"])
    # Untouched cells stay non-NaN
    assert not np.isnan(log_returns.loc["2020-01-03", "BBB"])
    assert not np.isnan(log_returns.loc["2020-01-07", "AAA"])

    # And the masked cells are correctly dropped by flag_anomalies' stack()
    # (which by default has dropna=True so NaN cells never appear)
    flagged = flag_anomalies(log_returns, threshold=0.001)
    assert not (
        (flagged["ticker"] == "AAA") & (flagged["date"] == pd.Timestamp("2020-01-03"))
    ).any()
    assert not (
        (flagged["ticker"] == "BBB") & (flagged["date"] == pd.Timestamp("2020-01-07"))
    ).any()
