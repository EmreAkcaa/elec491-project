"""Precomputed Point-In-Time correlation / MST / dislocation snapshots.

PHASE 3 (slim) — see ``/Users/emre/.claude/plans/...spicy-finch.md``.

Generates a date-strided grid of PIT snapshots so the Time Machine slider
becomes near-instant on HF Spaces. Live compute is still possible (Time
Machine falls back when no snapshot file exists) — this stage is the
fast-path for the two flagship universes the demo audience sees most:
**BIST (TRY base) + S&P 500**, window=252 only.

Storage budget (~125 MB total):
  - BIST: 5-business-day stride × 73 tickers ≈ ~25 MB
    (correlation matrix is small; weekly snapshots give smooth scrubbing)
  - S&P: 21-business-day stride × 485 tickers (float32) ≈ ~95 MB
    (correlation matrix is 27× larger; monthly snapshots stay in budget)
  - MST + dislocation: ~5 MB combined

Gating:
  - bist_usd / bist_gold: skipped. These re-express the SAME 73 BIST
    tickers in a different base currency; the correlation MATRIX shifts
    modestly but the user is rarely scrubbing on these views. Live
    compute (50-200 ms) is acceptable when the user does flip basis.
  - eeg_motor_left_right: skipped. EEG doesn't have the same crisis-
    chronology demo use case driving the snapshot grid.

Output layout::

    data/<universe>/results/
        pit_corr/w252/
            2020-01-15.parquet   # PIT correlation matrix (tickers × tickers)
            2020-01-22.parquet
            ...
        pit_mst/w252/
            2020-01-15.csv       # MST edges (source, target, weight)
            ...
        pit_dislocation/w252/
            2020-01-15.parquet   # Top-20 most-negatively-correlated pairs
            ...

The Time Machine loader (``app/utils.py:load_pit_*_snapshot``) snaps any
user-picked date to the nearest grid date by scanning the directory.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from src.config import PipelineConfig
from src.analysis import compute_distance_matrix
from src.rolling_correlation import compute_window_correlation

logger = logging.getLogger(__name__)


# Universes we precompute for. Everything else lives on live compute.
_PRECOMPUTE_MARKETS: set[str] = {"bist", "sp500"}

# Per-universe stride knobs. The S&P matrix is ~27× larger than BIST
# (485×485 vs 73×73), so we use a coarser stride to stay in storage
# budget without sacrificing the BIST high-res experience.
_STRIDE_BUSINESS_DAYS: dict[str, int] = {
    "bist": 5,    # weekly; ~279 snapshots over 5 years
    "sp500": 21,  # monthly; ~70 snapshots over 5 years
}

# float32 storage for S&P. Correlation values rarely need more than 5–6
# significant figures, so float32 (7-decimal precision) costs nothing
# perceptually and halves disk + RAM footprint per snapshot.
_FLOAT_DTYPE: dict[str, str] = {
    "bist": "float64",
    "sp500": "float32",
}

# Fixed pipeline parameters for the snapshot grid. These match what Time
# Machine asks for by default; off-grid combinations still fall through
# to live compute.
_WINDOW = 252
_METHOD = "pearson"
_TOP_N_DISLOCATIONS = 20


def _ensure_dir(path: Path) -> None:
    """Create a directory tree silently if missing."""
    path.mkdir(parents=True, exist_ok=True)


def _build_date_grid(
    returns: pd.DataFrame,
    *,
    window: int,
    stride_days: int,
) -> list[pd.Timestamp]:
    """Return the list of end-dates to snapshot.

    Skips early dates with insufficient history (need at least
    ``window + 30`` trading days behind). Uses business-day stride so
    we don't waste snapshots on weekends.
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
    # Snap each grid date to the nearest preceding actual trading day
    # so the snapshot's end_date corresponds to a real bar.
    out: list[pd.Timestamp] = []
    for g in grid:
        idx = returns.index.searchsorted(g, side="right") - 1
        if idx < 0:
            continue
        out.append(returns.index[idx])
    # De-dupe in case the snap collapsed several grid points to the
    # same trading day (long holiday weeks).
    seen: set[pd.Timestamp] = set()
    unique: list[pd.Timestamp] = []
    for d in out:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def _build_mst_edges(corr: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Build an MST from a correlation matrix and return its edges.

    Output columns match ``mst_edges.csv``: source, target, distance,
    correlation. Returns None when NetworkX isn't available or the
    correlation matrix is too small to span an MST.
    """
    if not HAS_NETWORKX:
        return None
    if corr.empty or len(corr) < 2:
        return None
    dist = compute_distance_matrix(corr)
    G = nx.Graph()
    cols = list(dist.columns)
    for i, t1 in enumerate(cols):
        for j in range(i + 1, len(cols)):
            t2 = cols[j]
            d = float(dist.iloc[i, j])
            if np.isfinite(d):
                G.add_edge(t1, t2, weight=d)
    if G.number_of_edges() == 0:
        return None
    mst = nx.minimum_spanning_tree(G, algorithm="kruskal", weight="weight")
    rows = []
    for u, v in mst.edges():
        d = float(mst[u][v]["weight"])
        # Recover correlation from distance: d = sqrt(2 * (1 - rho))
        # → rho = 1 - d²/2
        rho = 1.0 - (d ** 2) / 2.0
        rows.append({
            "source": u,
            "target": v,
            "distance": d,
            "correlation": rho,
        })
    return pd.DataFrame(rows)


def _top_dislocations(corr: pd.DataFrame, n: int = _TOP_N_DISLOCATIONS) -> pd.DataFrame:
    """Return the N most-negatively-correlated pairs from the correlation matrix.

    This is a windowed analogue of the full ``rank_candidate_pairs``
    that requires running OLS + half-life + Z-score across all pairs
    PER DATE — far too slow to do 279 times. The "most negatively
    correlated" subset captures the demo intuition (mean-reversion
    candidates) at <10 ms per date.
    """
    if corr.empty or len(corr) < 2:
        return pd.DataFrame(columns=["ticker_a", "ticker_b", "correlation"])
    cols = list(corr.columns)
    records: list[tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iloc[i, j])
            if np.isfinite(r):
                records.append((cols[i], cols[j], r))
    if not records:
        return pd.DataFrame(columns=["ticker_a", "ticker_b", "correlation"])
    df = pd.DataFrame(records, columns=["ticker_a", "ticker_b", "correlation"])
    df = df.sort_values("correlation", ascending=True).head(n).reset_index(drop=True)
    return df


def _snapshot_one_date(
    *,
    returns: pd.DataFrame,
    end_date: pd.Timestamp,
    out_corr_dir: Path,
    out_mst_dir: Path,
    out_dis_dir: Path,
    dtype: str,
) -> bool:
    """Compute + write one date's PIT corr/MST/dislocation files.

    Returns True iff a snapshot was successfully written. Returns False
    when the PIT correlation matrix is empty (not enough history).
    """
    corr = compute_window_correlation(
        returns, end_date, window=_WINDOW, method=_METHOD,
    )
    if corr.empty:
        return False

    if dtype != "float64":
        corr = corr.astype(dtype)

    iso = end_date.strftime("%Y-%m-%d")
    corr.to_parquet(out_corr_dir / f"{iso}.parquet", compression="snappy")

    mst_edges = _build_mst_edges(corr.astype("float64"))  # MST math wants f64
    if mst_edges is not None:
        mst_edges.to_csv(out_mst_dir / f"{iso}.csv", index=False)

    dis = _top_dislocations(corr.astype("float64"))
    if not dis.empty:
        dis.to_parquet(out_dis_dir / f"{iso}.parquet", compression="snappy")

    return True


def run_pit_snapshots(config: PipelineConfig) -> None:
    """Pipeline stage: precompute Time Machine PIT snapshots.

    Gate: ``config.market.market_id`` must be in ``_PRECOMPUTE_MARKETS``.
    Skips silently with a log line otherwise (bist_usd, bist_gold, eeg).
    """
    market_id = config.market.market_id.lower()
    if market_id not in _PRECOMPUTE_MARKETS:
        logger.info(
            "PIT snapshots SKIPPED for market_id=%r (precompute set: %s)",
            market_id, sorted(_PRECOMPUTE_MARKETS),
        )
        return

    logger.info("=== PIT Snapshots — %s ===", market_id)

    returns_path = config.data_processed / "log_returns.parquet"
    if not returns_path.exists():
        logger.warning(
            "log_returns.parquet not found at %s — run preprocessing first.",
            returns_path,
        )
        return

    returns = pd.read_parquet(returns_path)
    logger.info(
        "Loaded log returns: %d days × %d tickers (%s → %s)",
        returns.shape[0], returns.shape[1],
        returns.index.min().date(), returns.index.max().date(),
    )

    stride = _STRIDE_BUSINESS_DAYS[market_id]
    dtype = _FLOAT_DTYPE[market_id]
    date_grid = _build_date_grid(returns, window=_WINDOW, stride_days=stride)
    if not date_grid:
        logger.warning("Date grid empty — not enough history for window=%d", _WINDOW)
        return
    logger.info(
        "Will write %d snapshots (stride=%d business days, dtype=%s) for w=%d.",
        len(date_grid), stride, dtype, _WINDOW,
    )

    out_corr_dir = config.data_results / "pit_corr" / f"w{_WINDOW}"
    out_mst_dir = config.data_results / "pit_mst" / f"w{_WINDOW}"
    out_dis_dir = config.data_results / "pit_dislocation" / f"w{_WINDOW}"
    for d in (out_corr_dir, out_mst_dir, out_dis_dir):
        _ensure_dir(d)

    t0 = time.perf_counter()
    n_written = 0
    n_skipped = 0
    for i, end_date in enumerate(date_grid, start=1):
        wrote = _snapshot_one_date(
            returns=returns,
            end_date=end_date,
            out_corr_dir=out_corr_dir,
            out_mst_dir=out_mst_dir,
            out_dis_dir=out_dis_dir,
            dtype=dtype,
        )
        if wrote:
            n_written += 1
        else:
            n_skipped += 1
        # Per-snapshot timing is noisy; log every 50 snapshots to keep
        # output digestible while still surfacing pipeline progress.
        if i % 50 == 0 or i == len(date_grid):
            elapsed = time.perf_counter() - t0
            logger.info(
                "  …%d/%d snapshots done in %.1fs (%.0f ms/snapshot avg)",
                i, len(date_grid), elapsed, (elapsed / i) * 1000,
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        "PIT snapshots complete for %s: %d written, %d skipped, %.1fs total.",
        market_id, n_written, n_skipped, elapsed,
    )
