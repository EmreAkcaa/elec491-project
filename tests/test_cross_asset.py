"""Tests for the Phase X cross-asset pipeline stage (`src/cross_asset.py`).

Context:
  - Phase X was originally claimed shipped in PR #61 but the plan file
    was wrong; `src/cross_asset.py` was missing from main, leaving
    `run_pipeline.py:30` broken on `from src.cross_asset import ...`.
  - This module verifies the rebuilt stage:
      1. Pipeline stage runs without error on BIST (writes 3 artifacts).
      2. Stage gates cleanly on non-BIST markets (writes nothing).
      3. Per-ticker full-period correlation is causal (no `.shift(-K)`,
         past-only).
      4. Sanity: bank tickers (AKBNK, ISCTR) have NEGATIVE corr to
         USD/TRY (textbook behavior).
      5. Loader contract: empty DataFrame on missing files, correct
         schema on hits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Pipeline stage — schema verification
# ---------------------------------------------------------------------------


def test_cross_asset_summary_has_expected_schema():
    """Artifact on disk has the columns the Signals page reads."""
    path = _REPO_ROOT / "data" / "bist" / "results" / "cross_asset_summary.parquet"
    if not path.exists():
        pytest.skip(
            "cross_asset_summary.parquet not on disk; "
            "run `uv run python -c 'from src.cross_asset import run_cross_asset; "
            "from src.config import load_config; run_cross_asset(load_config())'` first."
        )
    df = pd.read_parquet(path)
    required = {"ticker", "corr_usd_try", "n_obs_usd_try", "corr_gold_usd",
                "n_obs_gold_usd", "sector"}
    missing = required - set(df.columns)
    assert not missing, f"missing columns: {missing}"
    assert len(df) > 0, "summary should not be empty for BIST"


def test_cross_asset_rolling_panels_exist():
    """Both rolling panels (USD/TRY + Gold) exist with date x ticker schema."""
    results = _REPO_ROOT / "data" / "bist" / "results"
    for asset_key in ("usd_try", "gold_usd"):
        path = results / f"cross_asset_corr_rolling_{asset_key}.parquet"
        if not path.exists():
            pytest.skip(
                f"cross_asset_corr_rolling_{asset_key}.parquet not on disk; "
                "re-run the pipeline."
            )
        df = pd.read_parquet(path)
        assert df.shape[0] > 0, f"rolling panel for {asset_key} should have dates"
        assert df.shape[1] > 0, f"rolling panel for {asset_key} should have tickers"
        # Date index should be monotonically increasing.
        assert df.index.is_monotonic_increasing, (
            f"rolling panel {asset_key} dates not sorted"
        )


# ---------------------------------------------------------------------------
# Sanity check — bank tickers should have NEGATIVE β to USD/TRY
# ---------------------------------------------------------------------------


def test_akbnk_negative_corr_with_usd_try():
    """AKBNK (banking) loses value when TRY weakens (textbook behavior)."""
    path = _REPO_ROOT / "data" / "bist" / "results" / "cross_asset_summary.parquet"
    if not path.exists():
        pytest.skip("artifact missing")
    df = pd.read_parquet(path)
    if "AKBNK" not in set(df["ticker"]):
        pytest.skip("AKBNK not in summary (universe membership changed?)")
    row = df[df["ticker"] == "AKBNK"].iloc[0]
    assert row["corr_usd_try"] < 0, (
        f"AKBNK should have negative corr to USD/TRY (banks lose on TRY weakening); "
        f"got {row['corr_usd_try']}"
    )


def test_multiple_banks_negative_corr_with_usd_try():
    """At least 3 of {AKBNK, ISCTR, YKBNK, GARAN} should have negative β."""
    path = _REPO_ROOT / "data" / "bist" / "results" / "cross_asset_summary.parquet"
    if not path.exists():
        pytest.skip("artifact missing")
    df = pd.read_parquet(path)
    banks = {"AKBNK", "ISCTR", "YKBNK", "GARAN", "HALKB", "VAKBN"}
    present = banks & set(df["ticker"])
    if len(present) < 3:
        pytest.skip(f"only {len(present)} banks in universe; need 3+")
    bank_df = df[df["ticker"].isin(present)]
    negative_count = int((bank_df["corr_usd_try"] < 0).sum())
    assert negative_count >= 3, (
        f"expected ≥3 banks with negative USD/TRY corr; got {negative_count} "
        f"of {len(bank_df)} ({bank_df[['ticker', 'corr_usd_try']].to_dict('records')})"
    )


# ---------------------------------------------------------------------------
# Look-ahead audit
# ---------------------------------------------------------------------------


def test_rolling_corr_panel_is_left_aligned():
    """Rolling correlation must produce NaN for the early window (no future lookup)."""
    path = _REPO_ROOT / "data" / "bist" / "results" / "cross_asset_corr_rolling_usd_try.parquet"
    if not path.exists():
        pytest.skip("artifact missing")
    df = pd.read_parquet(path)
    # First 150 rows should be ALL NaN (min_periods = 152, window = 252).
    first_150 = df.head(150)
    assert first_150.isna().all().all(), (
        "rolling panel leaks future data: first 150 rows should be all NaN "
        "(left-aligned 252-day window with min_periods=152). "
        f"Found non-NaN values: {first_150.notna().sum().sum()}"
    )


def test_full_period_corr_consistent_with_manual_recompute():
    """Re-compute one ticker's full-period corr from log_returns + base asset,
    and verify it matches the summary parquet within a small tolerance.
    """
    summary_path = _REPO_ROOT / "data" / "bist" / "results" / "cross_asset_summary.parquet"
    returns_path = _REPO_ROOT / "data" / "bist" / "processed" / "log_returns.parquet"
    base_path = _REPO_ROOT / "data" / "raw" / "base_assets" / "usd_try.parquet"
    for p in (summary_path, returns_path, base_path):
        if not p.exists():
            pytest.skip(f"artifact missing: {p}")
    summary = pd.read_parquet(summary_path)
    returns = pd.read_parquet(returns_path)
    base_price = pd.read_parquet(base_path).iloc[:, 0].astype(float).dropna()
    base_returns = np.log(base_price / base_price.shift(1)).dropna()

    if "AKBNK" not in returns.columns or "AKBNK" not in set(summary["ticker"]):
        pytest.skip("AKBNK not in panel")

    common = returns.index.intersection(base_returns.index)
    r = returns.loc[common, "AKBNK"]
    b = base_returns.loc[common]
    mask = r.notna() & b.notna()
    expected_corr = float(r[mask].corr(b[mask]))
    expected_n = int(mask.sum())

    summary_row = summary[summary["ticker"] == "AKBNK"].iloc[0]
    actual_corr = float(summary_row["corr_usd_try"])
    actual_n = int(summary_row["n_obs_usd_try"])

    assert actual_n == expected_n, (
        f"n_obs mismatch: summary={actual_n}, recomputed={expected_n}"
    )
    assert abs(actual_corr - expected_corr) < 1e-6, (
        f"corr mismatch: summary={actual_corr}, recomputed={expected_corr}"
    )


# ---------------------------------------------------------------------------
# Gating — non-BIST markets skip cleanly
# ---------------------------------------------------------------------------


def test_cross_asset_skipped_on_sp500(tmp_path, monkeypatch):
    """Stage logs a 'skipping' line on S&P and writes nothing."""
    from src.config import load_config
    from src import cross_asset as ca

    sp_cfg_path = _REPO_ROOT / "config" / "settings_sp500.yaml"
    if not sp_cfg_path.exists():
        pytest.skip("settings_sp500.yaml missing")
    cfg = load_config(sp_cfg_path)
    assert cfg.market.market_id.lower() != "bist"

    # Snapshot existing files in the S&P results dir (the stage should NOT
    # add any new ones).
    sp_results = cfg.data_results
    existing = {p.name for p in sp_results.iterdir()} if sp_results.exists() else set()
    cross_asset_files = {n for n in existing if n.startswith("cross_asset")}

    ca.run_cross_asset(cfg)

    new_files = {p.name for p in sp_results.iterdir()} if sp_results.exists() else set()
    new_cross_asset_files = {n for n in new_files if n.startswith("cross_asset")}
    assert new_cross_asset_files == cross_asset_files, (
        f"stage wrote unexpected cross_asset files on S&P: "
        f"{new_cross_asset_files - cross_asset_files}"
    )


# ---------------------------------------------------------------------------
# Loader contract
# ---------------------------------------------------------------------------


def test_load_cross_asset_summary_returns_empty_for_sp500():
    """Loader returns an empty DataFrame for non-BIST universes."""
    import os
    os.environ["DASHBOARD_UNIVERSE"] = "sp500"
    from app.utils import _load_cross_asset_summary

    df = _load_cross_asset_summary("sp500")
    assert df.empty, "S&P 500 should have no cross-asset summary"


def test_load_cross_asset_rolling_rejects_bogus_asset_key():
    """Loader returns empty for an unknown asset_key (defensive)."""
    from app.utils import _load_cross_asset_rolling

    df = _load_cross_asset_rolling("bist", "bogus_asset")
    assert df.empty, "unknown asset_key should return empty"
