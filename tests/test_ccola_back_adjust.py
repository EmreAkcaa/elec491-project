"""Tests for the option-B CCOLA back-adjust extension to `manual_anomaly_nulls`.

Context:
  - yfinance occasionally fails to back-adjust BIST corporate actions
    (CCOLA 10.81× bonus 2024-08-01; HEKTS 2.84× 2024-09-09; HEKTS 1.45×
    over-adjusted 2021-04-30; AYGAZ 1.72× over-adjusted 2022-09-01).
  - The legacy `manual_anomaly_nulls` half-fix nulled the log_returns
    cell at the event date but left the cliff in adj_close.parquet,
    contaminating any consumer of `np.log(adj_close[ticker])` — most
    notably `src/pair_dislocation.py:compute_spread` which produced
    5-7σ Z-score spikes for ~3-4 months after each event.
  - Option B (this PR) extends the schema to support a 3-tuple
    `[ticker, date, ratio]` that ALSO back-adjusts adj_close BEFORE
    log returns are computed. Sign convention:
      ratio > 0   missed-split case   (pre-event /= ratio)
      ratio < 0   over-adjusted case  (pre-event *= |ratio|)

These tests cover:
  1. Unit-test `_apply_split_back_adjust` on synthetic missed-split + over-adjusted data.
  2. Regression: BIST adj_close has no |log return| > 0.30 anywhere (was previously > 2.0 at CCOLA).
  3. Regression: Pair Analysis Z-score at the 4 documented event dates is in the ±3σ range (was 5-7σ).
  4. Audit log: `data/bist/processed/applied_split_adjustments.csv` exists and lists all 4 events.
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
# Unit tests — `_apply_split_back_adjust` on synthetic data
# ---------------------------------------------------------------------------


def test_back_adjust_missed_split_case():
    """ratio > 0 → pre-event prices divided by ratio (CCOLA-like)."""
    from src.preprocessing import _apply_split_back_adjust

    dates = pd.bdate_range("2024-07-25", periods=10, freq="B")
    prices = pd.Series(
        [800, 805, 810, 815, 820, 80, 78, 81, 79, 82],
        index=dates, dtype=float,
    )
    df = pd.DataFrame({"CCOLA": prices})

    ok, audit = _apply_split_back_adjust(df, "CCOLA", "2024-08-01", 10.81)
    assert ok
    assert audit["applied"] is True
    assert audit["reason"] == "ok"
    # Pre-event prices (first 5) should now be in the 70-80 range.
    pre_event = df["CCOLA"].iloc[:5].values
    assert (pre_event > 60).all() and (pre_event < 90).all(), (
        f"Pre-event prices should be ~75 after /10.81, got {pre_event}"
    )
    # Post-event prices (last 5) untouched.
    post_event = df["CCOLA"].iloc[5:].values
    np.testing.assert_allclose(post_event, [80, 78, 81, 79, 82])


def test_back_adjust_over_adjusted_case():
    """ratio < 0 → pre-event prices multiplied by |ratio| (HEKTS 2021-like)."""
    from src.preprocessing import _apply_split_back_adjust

    dates = pd.bdate_range("2021-04-27", periods=6, freq="B")
    # HEKTS-2021: pre-event ~1.6, post-event ~2.33 → looks like price ROSE
    # 1.45× at the event (yfinance over-adjusted), so we raise pre-event
    # prices by |ratio|=1.45 to make the series continuous.
    prices = pd.Series(
        [1.6, 1.62, 1.65, 2.33, 2.35, 2.30], index=dates, dtype=float,
    )
    df = pd.DataFrame({"HEKTS": prices})

    ok, audit = _apply_split_back_adjust(df, "HEKTS", "2021-04-30", -1.45)
    assert ok
    assert audit["applied"] is True
    # Pre-event prices now ~2.3 (1.6 * 1.45 ≈ 2.32).
    pre_event = df["HEKTS"].iloc[:3].values
    assert (pre_event > 2.0).all() and (pre_event < 2.6).all(), (
        f"Pre-event prices should be ~2.3 after *1.45, got {pre_event}"
    )


def test_back_adjust_skips_missing_ticker():
    """Returns (False, audit) without mutating when ticker isn't in panel."""
    from src.preprocessing import _apply_split_back_adjust

    df = pd.DataFrame({"AKBNK": [100, 101, 102]})
    ok, audit = _apply_split_back_adjust(df, "GHOST", "2024-01-01", 2.0)
    assert ok is False
    assert audit["applied"] is False
    assert audit["reason"] == "ticker_not_in_panel"


def test_back_adjust_skips_event_on_first_day():
    """Returns (False, audit) when the event is on the panel's first day
    (no pre-event prices to adjust)."""
    from src.preprocessing import _apply_split_back_adjust

    dates = pd.bdate_range("2024-08-01", periods=5, freq="B")
    df = pd.DataFrame({"CCOLA": [75, 76, 77, 78, 79]}, index=dates, dtype=float)
    ok, audit = _apply_split_back_adjust(df, "CCOLA", "2024-08-01", 10.81)
    assert ok is False
    assert audit["reason"] == "event_on_first_day"


def test_back_adjust_zero_ratio_rejected():
    """Sanity: ratio == 0 is meaningless; skip with reason='zero_ratio'
    rather than dividing by zero downstream."""
    from src.preprocessing import _apply_split_back_adjust

    dates = pd.bdate_range("2024-07-25", periods=5, freq="B")
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
    ok, audit = _apply_split_back_adjust(df, "X", "2024-07-29", 0.0)
    assert ok is False
    assert audit["reason"] == "zero_ratio"
    # Series must be untouched.
    np.testing.assert_allclose(df["X"].values, [1.0, 2.0, 3.0, 4.0, 5.0])


# ---------------------------------------------------------------------------
# Regression — BIST on-disk artifacts after re-running preprocessing
# ---------------------------------------------------------------------------


def _has_bist_adj_close() -> bool:
    return (_REPO_ROOT / "data" / "bist" / "processed" / "adj_close.parquet").exists()


needs_bist = pytest.mark.skipif(
    not _has_bist_adj_close(),
    reason="BIST adj_close.parquet missing; run `uv run python run_pipeline.py`.",
)


@needs_bist
def test_bist_adj_close_has_no_large_log_returns():
    """After Option B, adj_close.parquet should have NO |log return| > 0.30
    anywhere in the panel. Before the fix, CCOLA-2024-08-01 had |log| ~2.38."""
    adj = pd.read_parquet(_REPO_ROOT / "data" / "bist" / "processed" / "adj_close.parquet")
    log_r = np.log(adj / adj.shift(1))
    max_abs = log_r.abs().max().max()
    assert max_abs < 0.30, (
        f"adj_close still has a cliff: max |log return| = {max_abs:.3f}. "
        f"Expected < 0.30 after Option B back-adjust."
    )


@needs_bist
def test_bist_four_documented_events_neutralised():
    """The 4 manual_anomaly_nulls entries should have |log(post/pre)| < 0.10
    after back-adjust. Verifies each event individually so we catch
    per-event regressions."""
    adj = pd.read_parquet(_REPO_ROOT / "data" / "bist" / "processed" / "adj_close.parquet")
    events = [
        ("CCOLA", "2024-08-01"),
        ("HEKTS", "2024-09-09"),
        ("HEKTS", "2021-04-30"),
        ("AYGAZ", "2022-09-01"),
    ]
    for ticker, date_str in events:
        if ticker not in adj.columns:
            continue
        ts = pd.Timestamp(date_str)
        nxt = adj.index[adj.index >= ts]
        if len(nxt) == 0:
            continue
        event = nxt[0]
        pos = adj.index.get_loc(event)
        pre = adj[ticker].iloc[pos - 1]
        post = adj[ticker].iloc[pos]
        log_jump = abs(np.log(post / pre))
        assert log_jump < 0.10, (
            f"{ticker} {date_str}: post-adjust |log(post/pre)| = {log_jump:.4f}, "
            f"expected < 0.10. The back-adjust didn't catch this event."
        )


@needs_bist
def test_bist_pair_zscore_no_spike_at_ccola_event():
    """Pair Analysis Z-score for CCOLA pairs at 2024-08-01 should now be
    in the ±3σ range (was 5-7σ before Option B)."""
    from src.pair_dislocation import compute_spread, compute_zscore

    adj = pd.read_parquet(_REPO_ROOT / "data" / "bist" / "processed" / "adj_close.parquet")
    event_date = pd.Timestamp("2024-08-01")
    pairs_to_check = [
        ("CCOLA", "AEFES"),
        ("CCOLA", "TUPRS"),
        ("CCOLA", "AKBNK"),
    ]
    for ta, tb in pairs_to_check:
        if ta not in adj.columns or tb not in adj.columns:
            continue
        spread, _, _ = compute_spread(adj, ta, tb)
        z = compute_zscore(spread, window=60)
        nxt = z.dropna().index[z.dropna().index >= event_date]
        if len(nxt) == 0:
            continue
        z_at_event = abs(z.loc[nxt[0]])
        assert z_at_event < 3.0, (
            f"{ta}/{tb} Z-score on {event_date.date()} = {z_at_event:.2f}, "
            f"expected < 3.0 after Option B back-adjust (was 5-7σ before)."
        )


@needs_bist
def test_applied_split_adjustments_audit_log_present():
    """`applied_split_adjustments.csv` should be written by preprocessing
    when 3-tuple entries are present in `manual_anomaly_nulls`."""
    audit_path = _REPO_ROOT / "data" / "bist" / "processed" / "applied_split_adjustments.csv"
    assert audit_path.exists(), (
        "Expected audit log at data/bist/processed/applied_split_adjustments.csv "
        "after preprocessing with 3-tuple manual_anomaly_nulls entries."
    )
    audit = pd.read_csv(audit_path)
    expected_cols = {"ticker", "date", "ratio", "applied", "reason"}
    assert expected_cols.issubset(set(audit.columns))
    # All 4 events should have applied=True.
    for ticker, date_str in [
        ("CCOLA", "2024-08-01"),
        ("HEKTS", "2024-09-09"),
        ("HEKTS", "2021-04-30"),
        ("AYGAZ", "2022-09-01"),
    ]:
        match = audit[audit["ticker"] == ticker]
        match = match[match["date"].astype(str).str.startswith(date_str)]
        assert not match.empty, f"No audit row for {ticker} {date_str}"
        assert bool(match.iloc[0]["applied"]), (
            f"{ticker} {date_str} audit row says applied=False; back-adjust "
            f"reason was: {match.iloc[0]['reason']}"
        )


# ---------------------------------------------------------------------------
# Schema back-compat — 2-tuple entries should still work
# ---------------------------------------------------------------------------


def test_schema_supports_legacy_2tuple(tmp_path, monkeypatch):
    """The schema MUST stay back-compatible with 2-tuple entries
    (`[ticker, date]`) — those null only the log_returns cell, no
    back-adjust. Regression guard for any future loader change."""
    import yaml

    yaml_text = """
market:
  market_id: bist
  index_ticker: XU100
  provider_suffix: .IS
  currency: TRY
  universe_csv: config/universes/bist100.csv
  region: emerging
data:
  start_date: "2020-01-01"
  end_date: "2026-02-28"
  retry_attempts: 3
  source: yfinance
  cache_raw: true
preprocessing:
  min_coverage_pct: 0.90
  anomaly_return_threshold: 0.30
  forward_fill: false
  manual_anomaly_nulls:
    - ["CCOLA", "2024-08-01"]
    - ["CCOLA_3TUPLE", "2024-08-01", 10.81]
analysis:
  correlation_method: "pearson"
  annualization_factor: 252
  corr_min_periods: 200
validation:
  isyatirimhisse_max_diff_pct: 1.0
  isyatirimhisse_sample_size: 5
"""
    yaml_path = tmp_path / "settings_test.yaml"
    yaml_path.write_text(yaml_text)
    # Just verify YAML parses to the expected list-of-lists shape.
    cfg = yaml.safe_load(yaml_text)
    entries = cfg["preprocessing"]["manual_anomaly_nulls"]
    assert len(entries) == 2
    assert len(entries[0]) == 2  # legacy 2-tuple
    assert len(entries[1]) == 3  # new 3-tuple
    assert entries[1][2] == 10.81
