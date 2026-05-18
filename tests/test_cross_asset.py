"""Tests for PHASE X — cross-asset sensitivity stage + loaders.

Coverage:
  1. Pipeline stage (`src/cross_asset.py`): gating, schema, alignment.
  2. Loaders (`app/utils.py:load_cross_asset_*`): read what the pipeline
     wrote, return empty cleanly on miss.

The stage is BIST-only by design (other markets' returns aren't TRY-base
so the cross-asset math premise doesn't hold). Tests run against:
  - Synthetic fake-config for non-BIST gating
  - Real on-disk BIST data for loader + schema verification (skip if
    pipeline hasn't been run locally)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _REPO_ROOT / "app"
for _p in (str(_REPO_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Pipeline stage unit tests
# ---------------------------------------------------------------------------


def _make_returns_with_factor(
    n_days: int = 600, n_tickers: int = 10, seed: int = 7,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build synthetic BIST-like returns + USD/TRY + Gold price series.

    Each ticker has a different sensitivity to USD/TRY (loading on the FX
    factor varies by ticker). Used for end-to-end tests of the rolling
    correlation math without touching real disk data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2020-01-02", periods=n_days, name="Date")
    fx_factor = rng.normal(0, 0.012, size=n_days)
    gold_factor = rng.normal(0, 0.010, size=n_days)
    # Per-ticker loadings on FX (linearly varying), gold loadings 0.
    loadings_fx = np.linspace(-0.6, 0.6, n_tickers)
    idio = rng.normal(0, 0.015, size=(n_days, n_tickers))
    ticker_returns = (
        fx_factor[:, None] * loadings_fx[None, :]
        + idio
    )
    cols = [f"T{i:03d}" for i in range(n_tickers)]
    returns_df = pd.DataFrame(ticker_returns, index=dates, columns=cols)

    # Prices = exp(cumsum(returns)) * initial
    usd_try_prices = pd.Series(
        20.0 * np.exp(np.cumsum(fx_factor)),
        index=dates, name="usd_try",
    )
    gold_prices = pd.Series(
        2000.0 * np.exp(np.cumsum(gold_factor)),
        index=dates, name="gold_usd",
    )
    return returns_df, usd_try_prices, gold_prices


def test_align_returns_with_asset_inner_join():
    """Index intersection drops rows present in only one of the inputs."""
    from src.cross_asset import _align_returns_with_asset
    dates_a = pd.bdate_range("2020-01-02", periods=100, name="Date")
    dates_b = pd.bdate_range("2020-01-15", periods=100, name="Date")
    returns = pd.DataFrame(np.random.randn(100, 5), index=dates_a)
    asset = pd.Series(np.random.randn(100), index=dates_b, name="usd_try")
    aligned_r, aligned_a = _align_returns_with_asset(returns, asset)
    # Intersection of two 100-day windows offset by 9 business days = ~91 days.
    assert len(aligned_r) == len(aligned_a)
    assert aligned_r.index.equals(aligned_a.index)


def test_full_period_corr_recovers_loadings():
    """Synthetic data: per-ticker correlation with the FX factor should
    monotonically increase with the ticker's loading."""
    from src.cross_asset import _full_period_corr
    returns, usd_try_prices, _ = _make_returns_with_factor(n_days=800, n_tickers=10)
    fx_returns = np.log(usd_try_prices).diff().dropna()
    aligned_r = returns.loc[fx_returns.index]
    corr = _full_period_corr(aligned_r, fx_returns)
    # Loadings were linspace(-0.6, 0.6), so corr[T009] > corr[T000].
    assert corr["T009"] > corr["T000"]
    # First and last tickers should have substantially different (close to
    # max ± min) correlations.
    assert corr["T009"] - corr["T000"] > 0.5


def test_rolling_corr_shape_and_window():
    """Rolling-correlation DataFrame should match input shape and be NaN
    in early rows before `min_periods` is satisfied."""
    from src.cross_asset import _rolling_corr
    returns, usd_try_prices, _ = _make_returns_with_factor(n_days=500, n_tickers=5)
    fx_returns = np.log(usd_try_prices).diff().dropna()
    aligned_r = returns.loc[fx_returns.index]
    rolling = _rolling_corr(aligned_r, fx_returns, window=252)
    assert rolling.shape == aligned_r.shape
    # First few rows are NaN (min_periods = 0.6*252 ≈ 152).
    assert rolling.iloc[0].isna().all()
    # Late rows (well past min_periods) are filled.
    assert not rolling.iloc[-1].isna().all()


def test_run_cross_asset_skipped_for_non_bist(tmp_path):
    """Non-BIST markets must skip silently — no files, no crash."""
    from src.cross_asset import run_cross_asset

    class _FakeMarket:
        market_id = "sp500"

    class _FakeConfig:
        market = _FakeMarket()
        data_processed = tmp_path / "processed"
        data_results = tmp_path / "results"
        universe = pd.DataFrame(columns=["ticker", "sector"])

    run_cross_asset(_FakeConfig())
    # No output files written.
    assert not (tmp_path / "results").exists() or not list((tmp_path / "results").iterdir())


def test_run_cross_asset_writes_expected_files(tmp_path, monkeypatch):
    """End-to-end: BIST run produces the 3 expected parquets with the
    correct schema."""
    import src.cross_asset as ca

    # Synthetic returns + base assets — write them to the tmp_path that
    # the FakeConfig will point to.
    returns, usd_try_prices, gold_prices = _make_returns_with_factor(
        n_days=500, n_tickers=8,
    )
    processed = tmp_path / "processed"
    results = tmp_path / "results"
    processed.mkdir()
    results.mkdir()
    returns.to_parquet(processed / "log_returns.parquet")

    # Stub the base-asset loader to read from a controlled location.
    # The loader constructs `_PROJECT_ROOT / "data" / "raw" / "base_assets"`,
    # so we write under tmp_path/data/raw/base_assets and patch
    # _PROJECT_ROOT to point at tmp_path.
    base_assets_dir = tmp_path / "data" / "raw" / "base_assets"
    base_assets_dir.mkdir(parents=True)
    pd.DataFrame({"usd_try": usd_try_prices}).to_parquet(
        base_assets_dir / "usd_try.parquet",
    )
    pd.DataFrame({"gold_usd": gold_prices}).to_parquet(
        base_assets_dir / "gold_usd.parquet",
    )
    monkeypatch.setattr(ca, "_PROJECT_ROOT", tmp_path)

    class _FakeMarket:
        market_id = "bist"

    class _FakeConfig:
        market = _FakeMarket()
        data_processed = processed
        data_results = results
        universe = pd.DataFrame({
            "ticker": list(returns.columns),
            "sector": ["Synthetic"] * len(returns.columns),
        })

    ca.run_cross_asset(_FakeConfig())

    # Three output files exist.
    summary_path = results / "cross_asset_summary.parquet"
    rolling_usd_path = results / "cross_asset_corr_rolling_usd_try.parquet"
    rolling_gold_path = results / "cross_asset_corr_rolling_gold_usd.parquet"
    assert summary_path.exists()
    assert rolling_usd_path.exists()
    assert rolling_gold_path.exists()

    # Summary schema.
    summary = pd.read_parquet(summary_path)
    expected_cols = {
        "ticker", "sector",
        "corr_usd_try", "corr_gold_usd",
        "n_obs_usd_try", "n_obs_gold_usd",
    }
    assert expected_cols.issubset(set(summary.columns))
    assert len(summary) == len(returns.columns)

    # Rolling schema.
    rolling_usd = pd.read_parquet(rolling_usd_path)
    assert set(rolling_usd.columns) == set(returns.columns)
    # Sanity: correlation values are in [-1, 1] (allow for NaN).
    vals = rolling_usd.values
    vals_clean = vals[np.isfinite(vals)]
    assert (vals_clean >= -1.0).all() and (vals_clean <= 1.0).all()


# ---------------------------------------------------------------------------
# Loader tests (use real on-disk BIST artifacts when present)
# ---------------------------------------------------------------------------


def _has_bist_cross_asset_data() -> bool:
    return (
        _REPO_ROOT / "data" / "bist" / "results" / "cross_asset_summary.parquet"
    ).exists()


needs_cross_asset_data = pytest.mark.skipif(
    not _has_bist_cross_asset_data(),
    reason="BIST cross-asset artifacts not on disk; run pipeline first.",
)


@needs_cross_asset_data
def test_load_cross_asset_summary_returns_expected_schema(monkeypatch):
    import streamlit as st
    from utils import load_cross_asset_summary
    import utils
    st.cache_data.clear()
    monkeypatch.setattr(utils, "current_universe", lambda: "bist")

    summary = load_cross_asset_summary()
    assert not summary.empty
    expected = {"ticker", "sector", "corr_usd_try", "corr_gold_usd"}
    assert expected.issubset(set(summary.columns))
    # BIST has 73 tickers post-coverage filter.
    assert len(summary) == 73


@needs_cross_asset_data
def test_load_cross_asset_rolling_returns_matrix(monkeypatch):
    import streamlit as st
    from utils import load_cross_asset_rolling
    import utils
    st.cache_data.clear()
    monkeypatch.setattr(utils, "current_universe", lambda: "bist")

    rolling = load_cross_asset_rolling("usd_try")
    assert not rolling.empty
    # Date-indexed.
    assert isinstance(rolling.index, pd.DatetimeIndex)
    # 73 BIST tickers.
    assert rolling.shape[1] == 73

    # Missing asset key → empty DataFrame, no crash.
    nothing = load_cross_asset_rolling("nonexistent_asset")
    assert nothing.empty


@needs_cross_asset_data
def test_load_cross_asset_summary_for_non_bist_returns_empty(monkeypatch):
    """For non-BIST universes the summary parquet doesn't exist, so the
    loader must return empty (not crash)."""
    import streamlit as st
    from utils import load_cross_asset_summary
    import utils
    st.cache_data.clear()
    monkeypatch.setattr(utils, "current_universe", lambda: "sp500")
    summary = load_cross_asset_summary()
    assert summary.empty
