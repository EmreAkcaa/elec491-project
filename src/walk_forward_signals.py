"""Walk-forward signal snapshots.

For each grid date D, re-screen the candidate pair list using only data
up to D, compute the state-machine status of each top pair at D, and
write a per-date parquet. The Signals page reads these snapshots so that
scrubbing the date picker shows trades that an honest as-of-D observer
would have proposed — no hindsight in selection, no future leakage in
state.

Storage budget (~5-10 MB total):
  - BIST: 5-business-day stride × ~263 dates × 20 pairs × 13 cols ≈ ~3 MB
  - S&P:  21-business-day stride × ~62 dates × 20 pairs × 13 cols  ≈ ~1 MB

Gating mirrors src/pit_snapshots.py:
  - bist_usd / bist_gold: skipped (live compute on the Signals page
    falls through to the legacy dislocation_candidates.csv path).
  - eeg_motor_left_right: skipped (no pair trading semantics).

Output layout::

    data/<universe>/results/
        walkforward_signals/w60/
            2021-02-16.parquet   # 20-row pair table with full state info
            2021-02-23.parquet
            ...

Each snapshot row carries:
  ticker_a, ticker_b, sector_a, sector_b, correlation, beta, half_life,
  spread_std, current_zscore, rank_score, status, trade_direction,
  last_signal_date, days_since_last_signal, n_signals_to_date, as_of_date

Subtle look-ahead trap, documented here so future maintainers don't
re-introduce it:

  ``compute_spread(adj_close, ta, tb, lookback=252)`` fits OLS on the
  LAST 252 days of ``adj_close``. If you pass the full panel that's 252
  days ending at ``adj_close.index[-1]`` — which is the most recent
  observation in the file, NOT the grid date D. Always pass
  ``adj_close.loc[:D]`` so "last 252 days" means "252 days ending at D".
  ``rank_candidate_pairs`` calls ``compute_spread`` internally with the
  ``adj_close`` argument it receives — so the slicing must happen at the
  stage boundary.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.rolling_correlation import compute_window_correlation
from src.pair_dislocation import (
    compute_spread,
    compute_zscore,
    detect_signals,
    rank_candidate_pairs,
    state_at,
    trade_direction,
)

logger = logging.getLogger(__name__)


# Mirror src/pit_snapshots.py gates so the two precompute grids stay
# aligned by construction.
_PRECOMPUTE_MARKETS: set[str] = {"bist", "sp500"}

_STRIDE_BUSINESS_DAYS: dict[str, int] = {
    "bist": 5,    # weekly
    "sp500": 21,  # monthly
}

# Z-score window is the "w" subdirectory selector (mirrors w252 used by
# PIT snapshots). 60 trading days is the dislocation default in
# config/settings.yaml.
_WINDOW_Z = 60

# Correlation matrix window. 252 days = 1 year — same as the rest of the
# pipeline so the user's mental model stays coherent across pages.
_WINDOW_CORR = 252


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_date_grid(
    returns: pd.DataFrame,
    *,
    window: int,
    stride_days: int,
) -> list[pd.Timestamp]:
    """Build the grid of as-of dates to snapshot.

    Direct mirror of ``src/pit_snapshots.py:_build_date_grid`` — same
    stride semantics so the walk-forward grid aligns with the PIT MST
    grid. Pair Analysis loads PIT MSTs at the user's as-of date; aligned
    grids mean "no extra snap" required when both surfaces ask for the
    same date.
    """
    if returns.empty:
        return []
    first_eligible = returns.index[min(window + 30, len(returns) - 1)]
    last = returns.index[-1]
    if first_eligible >= last:
        return []
    grid = pd.date_range(
        start=first_eligible,
        end=last,
        freq=f"{stride_days}B",
    )
    out: list[pd.Timestamp] = []
    for g in grid:
        idx = returns.index.searchsorted(g, side="right") - 1
        if idx < 0:
            continue
        out.append(returns.index[idx])
    seen: set[pd.Timestamp] = set()
    unique: list[pd.Timestamp] = []
    for d in out:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def _compute_one_date(
    *,
    adj_close: pd.DataFrame,
    returns: pd.DataFrame,
    end_date: pd.Timestamp,
    universe_df: pd.DataFrame,
    top_n: int,
    min_correlation: float,
    zscore_window: int,
    lookback: int,
    entry_zscore: float,
    exit_zscore: float,
    min_half_life: int,
    max_half_life: int,
) -> pd.DataFrame:
    """Compute the walk-forward signal table for a single as-of date.

    All inputs sliced to ``[:end_date]`` before any computation — this is
    the only place that contract lives, so callers don't have to remember
    to pre-slice.
    """
    # Slice past-only. dropna(axis=1, how="all") drops tickers that
    # weren't listed yet at end_date (their column is all NaN in the
    # slice), which keeps rank_candidate_pairs from screening dead
    # columns.
    adj_slice = adj_close.loc[:end_date].dropna(axis=1, how="all")
    ret_slice = returns.loc[:end_date]

    # Correlation matrix from a 252-day window ending at end_date.
    # ``compute_window_correlation`` is left-aligned by construction —
    # it slices ``returns.loc[start:end_date]`` internally where start
    # is end_date − window trading days.
    corr = compute_window_correlation(
        ret_slice, end_date, window=_WINDOW_CORR, method="pearson",
    )
    if corr.empty:
        return pd.DataFrame()

    # Re-rank pairs using only what an observer at end_date would have.
    ranked = rank_candidate_pairs(
        adj_close=adj_slice,
        corr=corr,
        universe=universe_df,
        top_n=top_n,
        min_correlation=min_correlation,
        zscore_window=zscore_window,
        lookback=lookback,
        entry_zscore=entry_zscore,
        exit_zscore=exit_zscore,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
    )
    if ranked.empty:
        return pd.DataFrame()

    # rank_candidate_pairs already returns: ticker_a, ticker_b, sector_a,
    # sector_b, correlation, beta, half_life, spread_std, n_signals,
    # current_zscore, rank_score. We replay the state machine to also
    # populate status / trade_direction / last_signal_date /
    # days_since_last_signal / n_signals_to_date.
    rows: list[dict] = []
    as_of_ts = pd.Timestamp(end_date)
    for _, r in ranked.iterrows():
        ta = r["ticker_a"]
        tb = r["ticker_b"]
        # Re-run the cheap path past-only (compute_spread/zscore are
        # already past-only when fed adj_slice).
        spread, _beta, _intercept = compute_spread(
            adj_slice, ta, tb, lookback=lookback,
        )
        zscore = compute_zscore(spread, window=zscore_window)
        status, last_signal = state_at(
            zscore, as_of_ts, entry_z=entry_zscore, exit_z=exit_zscore,
        )
        signals_df = detect_signals(
            zscore, entry_threshold=entry_zscore, exit_threshold=exit_zscore,
        )
        # n_signals_to_date: count past-only, not the full-history count
        # that rank_candidate_pairs stored. The full-history count would
        # silently leak future signals into a historical snapshot.
        if signals_df.empty:
            n_signals_to_date = 0
        else:
            n_signals_to_date = int(
                (pd.to_datetime(signals_df["date"]) <= as_of_ts).sum()
            )

        last_signal_iso = (
            pd.Timestamp(last_signal).strftime("%Y-%m-%d")
            if last_signal is not None
            else ""
        )
        days_since = (
            int((as_of_ts - pd.Timestamp(last_signal)).days)
            if last_signal is not None
            else -1
        )

        rows.append({
            "ticker_a": ta,
            "ticker_b": tb,
            "sector_a": r.get("sector_a", ""),
            "sector_b": r.get("sector_b", ""),
            "correlation": float(r["correlation"]),
            "beta": float(r["beta"]),
            "half_life": float(r["half_life"]),
            "spread_std": float(r["spread_std"]),
            "current_zscore": float(r["current_zscore"]),
            "rank_score": float(r["rank_score"]),
            "status": status,
            "trade_direction": trade_direction(status, ta, tb),
            "last_signal_date": last_signal_iso,
            "days_since_last_signal": days_since,
            "n_signals_to_date": n_signals_to_date,
            "as_of_date": as_of_ts.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(rows)


def run_walk_forward_signals(config: PipelineConfig) -> None:
    """Pipeline stage: write walk-forward signal snapshots.

    Gates on ``config.market.market_id in {"bist", "sp500"}``. Mirrors
    the gating + logging cadence of ``src/pit_snapshots.py``.
    """
    market_id = config.market.market_id.lower()
    if market_id not in _PRECOMPUTE_MARKETS:
        logger.info(
            "Walk-forward signals SKIPPED for market_id=%r (precompute set: %s)",
            market_id, sorted(_PRECOMPUTE_MARKETS),
        )
        return

    logger.info("=== Walk-forward Signals — %s ===", market_id)

    returns_path = config.data_processed / "log_returns.parquet"
    adj_path = config.data_processed / "adj_close.parquet"
    if not (returns_path.exists() and adj_path.exists()):
        logger.warning(
            "Missing inputs (log_returns or adj_close) — run preprocessing first."
        )
        return

    returns = pd.read_parquet(returns_path)
    adj_close = pd.read_parquet(adj_path)
    logger.info(
        "Loaded log returns: %d days × %d tickers (%s → %s)",
        returns.shape[0], returns.shape[1],
        returns.index.min().date(), returns.index.max().date(),
    )

    stride = _STRIDE_BUSINESS_DAYS[market_id]
    date_grid = _build_date_grid(
        returns, window=_WINDOW_CORR, stride_days=stride,
    )
    if not date_grid:
        logger.warning(
            "Date grid empty — not enough history for window=%d", _WINDOW_CORR,
        )
        return
    logger.info(
        "Will write %d walk-forward snapshots (stride=%d business days) for w%d.",
        len(date_grid), stride, _WINDOW_Z,
    )

    out_dir = config.data_results / "walkforward_signals" / f"w{_WINDOW_Z}"
    _ensure_dir(out_dir)

    dc = config.dislocation

    t0 = time.perf_counter()
    n_written = 0
    n_skipped = 0
    # Quiet the per-date "Screening %d pairs ..." log from
    # rank_candidate_pairs — 263 of them on BIST drowns the rest of the
    # pipeline log. The full path stays at INFO inside src.pair_dislocation
    # for one-off runs; we only mute it during the grid loop.
    pd_logger = logging.getLogger("src.pair_dislocation")
    prior_level = pd_logger.level
    pd_logger.setLevel(logging.WARNING)
    try:
        for i, end_date in enumerate(date_grid, start=1):
            try:
                snap = _compute_one_date(
                    adj_close=adj_close,
                    returns=returns,
                    end_date=end_date,
                    universe_df=config.universe,
                    top_n=dc.top_n_candidates,
                    min_correlation=dc.min_correlation,
                    zscore_window=dc.zscore_window,
                    lookback=dc.lookback_window,
                    entry_zscore=dc.entry_zscore,
                    exit_zscore=dc.exit_zscore,
                    min_half_life=dc.min_half_life,
                    max_half_life=dc.max_half_life,
                )
            except Exception as exc:
                logger.warning(
                    "  walk-forward snapshot failed at %s: %s",
                    end_date.date(), exc,
                )
                n_skipped += 1
                continue
            if snap.empty:
                n_skipped += 1
            else:
                iso = end_date.strftime("%Y-%m-%d")
                snap.to_parquet(
                    out_dir / f"{iso}.parquet", compression="snappy", index=False,
                )
                n_written += 1
            if i % 50 == 0 or i == len(date_grid):
                elapsed = time.perf_counter() - t0
                logger.info(
                    "  …%d/%d snapshots done in %.1fs (%.0f ms/snapshot avg)",
                    i, len(date_grid), elapsed, (elapsed / i) * 1000,
                )
    finally:
        pd_logger.setLevel(prior_level)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Walk-forward signals complete for %s: %d written, %d skipped, %.1fs total.",
        market_id, n_written, n_skipped, elapsed,
    )
