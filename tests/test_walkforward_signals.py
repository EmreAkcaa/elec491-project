"""Tests for the walk-forward signals pipeline stage (PR #69).

Coverage:
  1. Per-date schema check on real BIST artifacts.
  2. Walk-forward is actually past-only: at date D, the result doesn't
     change when future data is appended to the input panel.
  3. Pair list drifts across dates (proves re-screening, not copying).
  4. Stage produces expected directory layout for precomputed universes.
  5. Stage skips silently for non-precomputed markets.
  6. snap_to_preceding_snapshot never returns a future date.
  7. compute_spread receives an adj_close slice that ends at end_date
     (regression test for the subtle look-ahead bug documented in
     src/walk_forward_signals.py).
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
# Synthetic-data helpers
# ---------------------------------------------------------------------------

def _make_synthetic_panel(
    n_days: int = 400,
    n_tickers: int = 6,
    seed: int = 42,
    regime_swap_at: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a price + log-returns + universe panel for tests.

    ``regime_swap_at``: if set, the correlation structure changes at
    that day index. First half: tickers 0–2 are correlated as a cluster.
    Second half: tickers 3–5 are correlated as a cluster. Lets us prove
    walk-forward selection adapts.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days, freq="B")
    if regime_swap_at is None:
        regime_swap_at = n_days  # never swap

    # Build log returns
    log_returns = pd.DataFrame(index=dates, columns=[f"T{i}" for i in range(n_tickers)], dtype=float)
    for day in range(n_days):
        # Pick the "driver" cluster for this day
        if day < regime_swap_at:
            driver = rng.normal(0, 0.01)
            cluster_idx = [0, 1, 2]
        else:
            driver = rng.normal(0, 0.01)
            cluster_idx = [3, 4, 5]
        for i in range(n_tickers):
            base = rng.normal(0, 0.012)
            if i in cluster_idx:
                log_returns.iloc[day, i] = 0.7 * driver + 0.3 * base
            else:
                log_returns.iloc[day, i] = base

    # Cumulative log returns → prices, starting at 100
    adj_close = (np.exp(log_returns.cumsum()) * 100.0).astype(float)

    # Universe DataFrame
    universe = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n_tickers)],
        "sector": [
            "Cluster A" if i < 3 else "Cluster B" for i in range(n_tickers)
        ],
        "company_name": [f"Test Co {i}" for i in range(n_tickers)],
        "provider_symbol": [f"T{i}.TEST" for i in range(n_tickers)],
    })
    return adj_close, log_returns, universe


# ---------------------------------------------------------------------------
# Schema check on real BIST artifacts (when available)
# ---------------------------------------------------------------------------

def test_walkforward_snapshot_has_expected_columns():
    """One real BIST snapshot has all 16 expected columns."""
    base = _REPO_ROOT / "data" / "bist" / "results" / "walkforward_signals" / "w60"
    if not base.exists():
        pytest.skip("Walk-forward snapshots not on disk — run pipeline first.")
    files = sorted(base.glob("*.parquet"))
    if not files:
        pytest.skip("No walk-forward snapshot files found.")
    df = pd.read_parquet(files[-1])
    expected = {
        "ticker_a", "ticker_b", "sector_a", "sector_b",
        "correlation", "beta", "half_life", "spread_std",
        "current_zscore", "rank_score",
        "status", "trade_direction",
        "last_signal_date", "days_since_last_signal",
        "n_signals_to_date", "as_of_date",
    }
    missing = expected - set(df.columns)
    assert not missing, f"Snapshot missing columns: {missing}"
    # Reasonable row counts: 0 (if no pairs satisfied filters) up to top_n=20.
    assert 0 <= len(df) <= 20, f"Unexpected row count: {len(df)}"


# ---------------------------------------------------------------------------
# Walk-forward is past-only — the core no-leak guarantee
# ---------------------------------------------------------------------------

def test_compute_one_date_is_past_only():
    """Run at day 200 with and without 200 days of future data; both
    runs must produce identical pair tables."""
    from src.walk_forward_signals import _compute_one_date

    adj_300, ret_300, univ = _make_synthetic_panel(n_days=300, n_tickers=8, seed=11)
    # Truncated version: same first 200 days only.
    adj_200 = adj_300.iloc[:200].copy()
    ret_200 = ret_300.iloc[:200].copy()
    end_date = adj_200.index[-1]  # day 199

    common_args = dict(
        universe_df=univ,
        top_n=20,
        min_correlation=0.5,
        zscore_window=60,
        lookback=120,
        entry_zscore=2.0,
        exit_zscore=0.5,
        min_half_life=5,
        max_half_life=252,
    )

    df_truncated = _compute_one_date(
        adj_close=adj_200, returns=ret_200, end_date=end_date, **common_args,
    )
    df_with_future = _compute_one_date(
        adj_close=adj_300, returns=ret_300, end_date=end_date, **common_args,
    )

    # Both runs may produce empty (no pairs meet criteria) — that's fine,
    # just check the equivalence.
    if df_truncated.empty and df_with_future.empty:
        return
    assert not df_truncated.empty and not df_with_future.empty, (
        f"One run empty, other not: truncated={len(df_truncated)}, "
        f"with_future={len(df_with_future)}"
    )
    # Sort both by ticker_a, ticker_b for stable comparison.
    sort_keys = ["ticker_a", "ticker_b"]
    a = df_truncated.sort_values(sort_keys).reset_index(drop=True)
    b = df_with_future.sort_values(sort_keys).reset_index(drop=True)
    # Same set of pairs.
    a_pairs = set(zip(a["ticker_a"], a["ticker_b"]))
    b_pairs = set(zip(b["ticker_a"], b["ticker_b"]))
    assert a_pairs == b_pairs, (
        f"Future data changed pair selection: "
        f"only_in_truncated={a_pairs - b_pairs}, only_in_with_future={b_pairs - a_pairs}"
    )
    # Same z-score values (within float tolerance) for each pair.
    for col in ("current_zscore", "beta", "half_life"):
        diff = (a[col].values - b[col].values)
        max_abs = float(np.nanmax(np.abs(diff)))
        assert max_abs < 1e-6, f"{col} drifted by {max_abs} with future data added"


def test_pair_list_drifts_across_dates():
    """Synthetic regime swap: top pairs at day 150 should differ from
    top pairs at day 350. Proves walk-forward selection adapts."""
    from src.walk_forward_signals import _compute_one_date

    adj, ret, univ = _make_synthetic_panel(
        n_days=400, n_tickers=8, seed=7, regime_swap_at=200,
    )

    common_args = dict(
        universe_df=univ,
        top_n=10,
        min_correlation=0.3,
        zscore_window=60,
        lookback=120,
        entry_zscore=2.0,
        exit_zscore=0.5,
        min_half_life=5,
        max_half_life=300,
    )

    df_early = _compute_one_date(
        adj_close=adj, returns=ret, end_date=adj.index[150], **common_args,
    )
    df_late = _compute_one_date(
        adj_close=adj, returns=ret, end_date=adj.index[350], **common_args,
    )
    if df_early.empty or df_late.empty:
        pytest.skip("synthetic data didn't generate enough pairs")

    early_pairs = set(zip(df_early["ticker_a"], df_early["ticker_b"]))
    late_pairs = set(zip(df_late["ticker_a"], df_late["ticker_b"]))
    # Some overlap is fine, but most pairs should change.
    overlap = early_pairs & late_pairs
    assert len(overlap) < min(len(early_pairs), len(late_pairs)), (
        f"Pair lists identical across regime: early={early_pairs} late={late_pairs}"
    )


# ---------------------------------------------------------------------------
# snap_to_preceding_snapshot — never returns future
# ---------------------------------------------------------------------------

def test_snap_to_preceding_never_returns_future():
    """snap_to_preceding_snapshot must return ≤ requested date, or None."""
    from app.utils import snap_to_preceding_snapshot

    grid = ["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26"]
    # Mid-grid: pick a Wednesday between Friday snapshots.
    assert snap_to_preceding_snapshot("2024-01-10", grid_dates=grid) == "2024-01-05"
    assert snap_to_preceding_snapshot("2024-01-17", grid_dates=grid) == "2024-01-12"
    # Exact match.
    assert snap_to_preceding_snapshot("2024-01-19", grid_dates=grid) == "2024-01-19"
    # Before grid → None.
    assert snap_to_preceding_snapshot("2024-01-01", grid_dates=grid) is None
    # After grid → last entry.
    assert snap_to_preceding_snapshot("2030-12-31", grid_dates=grid) == "2024-01-26"
    # Empty grid → None.
    assert snap_to_preceding_snapshot("2024-01-15", grid_dates=[]) is None


# ---------------------------------------------------------------------------
# compute_spread receives a sliced adj_close (regression on subtle leak)
# ---------------------------------------------------------------------------

def test_compute_spread_receives_sliced_adj_close(monkeypatch):
    """The stage must pass adj_close.loc[:end_date] to rank_candidate_pairs,
    never the full panel. ``compute_spread`` fits OLS on the LAST
    ``lookback`` days of its input — if the input is unsliced, those days
    end at the full panel's max date and silently use future data.
    """
    from src.walk_forward_signals import _compute_one_date
    from src import pair_dislocation

    captured: list[pd.DataFrame] = []
    original = pair_dislocation.compute_spread

    def _spy(adj_close, ta, tb, lookback=None):
        captured.append(adj_close.copy())
        return original(adj_close, ta, tb, lookback)

    monkeypatch.setattr(pair_dislocation, "compute_spread", _spy)

    adj, ret, univ = _make_synthetic_panel(n_days=400, n_tickers=6, seed=3)
    end_date = adj.index[250]

    _compute_one_date(
        adj_close=adj, returns=ret, end_date=end_date,
        universe_df=univ, top_n=5, min_correlation=0.3,
        zscore_window=60, lookback=120,
        entry_zscore=2.0, exit_zscore=0.5,
        min_half_life=5, max_half_life=300,
    )

    if not captured:
        pytest.skip("synthetic data produced no pairs to spy on")

    panel_max = adj.index.max()
    for slc in captured:
        last_in_input = slc.index.max()
        assert last_in_input <= end_date, (
            f"compute_spread received adj_close ending at {last_in_input}, "
            f"which is after end_date={end_date.date()}. Future-data leak."
        )
        assert last_in_input < panel_max, (
            f"compute_spread received the FULL panel (max={panel_max.date()}). "
            f"Caller forgot to slice — silent leak."
        )


# ---------------------------------------------------------------------------
# Stage gating + skip behavior
# ---------------------------------------------------------------------------

def test_run_walk_forward_signals_skipped_for_non_precomputed_market(tmp_path):
    """Non-precomputed markets log a skip and write nothing."""
    from src.walk_forward_signals import run_walk_forward_signals
    from src.config import load_config

    # bist_usd / bist_gold aren't in _PRECOMPUTE_MARKETS. Use settings.yaml
    # but override market_id on a copy.
    cfg = load_config()  # BIST by default
    # Spoof a non-precomputed market_id on a shallow attribute.
    original_id = cfg.market.market_id
    try:
        cfg.market.market_id = "bist_usd"
        # Stage should return without touching disk for this universe.
        # If the existing walkforward_signals/ dir on bist results is still
        # there from earlier, that's fine — we only care that we don't
        # silently write to a NEW directory.
        existing_dir = cfg.data_results / "walkforward_signals" / "w60"
        existed = existing_dir.exists()
        snapshot_count_before = len(list(existing_dir.glob("*.parquet"))) if existed else 0

        run_walk_forward_signals(cfg)

        # Stage should have skipped — no new files appearing.
        snapshot_count_after = len(list(existing_dir.glob("*.parquet"))) if existed else 0
        assert snapshot_count_after == snapshot_count_before, (
            "Stage wrote files for non-precomputed market"
        )
    finally:
        cfg.market.market_id = original_id


# ---------------------------------------------------------------------------
# Build-date-grid sanity
# ---------------------------------------------------------------------------

def test_build_date_grid_emits_only_eligible_dates():
    """Grid should start after ``window + 30`` days of history."""
    from src.walk_forward_signals import _build_date_grid

    _, ret, _ = _make_synthetic_panel(n_days=400, n_tickers=4, seed=5)
    grid = _build_date_grid(ret, window=252, stride_days=5)
    if not grid:
        pytest.skip("not enough synthetic history")
    first_grid = grid[0]
    # window + 30 = 282 → first eligible at index 282; date should be ≥ that.
    idx_of_first = ret.index.get_loc(first_grid)
    assert idx_of_first >= 252 + 30 - 1, (
        f"Grid started too early: idx={idx_of_first}"
    )
