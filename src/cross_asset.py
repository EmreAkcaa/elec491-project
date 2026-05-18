"""Cross-asset sensitivity — per-ticker correlation with USD/TRY + Gold.

PHASE X — see ``/Users/emre/.claude/plans/...spicy-finch.md``.

For each BIST ticker, computes:
  1. The full-period Pearson correlation of its TRY-base log returns
     with USD/TRY log returns and with Gold (USD/oz) log returns.
  2. A rolling 252-day Pearson correlation series for both base assets,
     so the dashboard can render "BIST ticker's sensitivity to FX/Gold
     over time" charts.

These power the Cross-Market page's "FX & Gold Sensitivity (BIST only)"
subsection: "Top-5 most-TRY-sensitive BIST stocks" + "Top-5 most-Gold-
sensitive BIST stocks" tables.

Gating
------
Only `market_id == "bist"` runs this stage. The math is meaningful
specifically on TRY-base BIST returns; re-expressing in USD or Gold
would zero out the corresponding cross-asset correlation by construction
(the base asset would be a constant factor in the returns formula).

Storage
-------
For BIST 73 tickers × 1543 trading days:
  - Each rolling parquet: ~73 cols × 1543 rows × float64 = ~900 KB
  - Summary parquet: 73 rows × 5 cols ≈ ~5 KB
Total per market: ~1.8 MB. Negligible compared to PIT snapshots.

Output
------
  data/bist/results/
      cross_asset_corr_rolling_usd_try.parquet   # dates × tickers (rolling rho)
      cross_asset_corr_rolling_gold_usd.parquet  # dates × tickers (rolling rho)
      cross_asset_summary.parquet                 # ticker × (sector, corr_usd_try, corr_gold_usd, n_obs)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


# Only BIST gets this stage. The math premise (TRY-base returns vs FX/Gold)
# only holds when the underlying returns are in TRY.
_CROSS_ASSET_MARKETS: set[str] = {"bist"}

# Base assets we compute against. Each key matches the parquet filename
# under ``data/raw/base_assets/`` and the price column inside it.
_BASE_ASSETS: tuple[str, ...] = ("usd_try", "gold_usd")

_ROLLING_WINDOW = 252
_MIN_PERIODS_RATIO = 0.6  # require 60% of window worth of observations

# Project root for locating base_assets/. Mirrors src/config.PROJECT_ROOT
# rather than importing it (avoids circular import in some tooling).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_base_asset_returns(asset_key: str) -> pd.Series:
    """Load a base-asset price series from disk and convert to log returns.

    Returns an empty Series if the file is missing (caller should check
    .empty before using). Index is `Date` (datetime64[ns]); name matches
    `asset_key`.
    """
    path = _PROJECT_ROOT / "data" / "raw" / "base_assets" / f"{asset_key}.parquet"
    if not path.exists():
        logger.warning("Base asset parquet not found: %s", path)
        return pd.Series(dtype=float, name=asset_key)
    df = pd.read_parquet(path)
    if asset_key not in df.columns:
        logger.warning(
            "Base asset parquet %s missing expected column %r", path, asset_key,
        )
        return pd.Series(dtype=float, name=asset_key)
    # Convert price to log returns. Drop the leading NaN from .diff().
    series = np.log(df[asset_key]).diff().dropna()
    series.name = asset_key
    if series.index.name is None:
        series.index.name = "Date"
    return series


def _align_returns_with_asset(
    returns: pd.DataFrame, asset_returns: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Inner-join returns DataFrame with a base-asset Series on date.

    Date calendars differ between BIST (Türkiye trading days) and FX/Gold
    (international markets). Inner-join is the cleanest treatment — keeps
    only days where BOTH a BIST bar and a base-asset bar exist.
    """
    aligned_index = returns.index.intersection(asset_returns.index)
    return returns.loc[aligned_index], asset_returns.loc[aligned_index]


def _full_period_corr(
    returns: pd.DataFrame, asset_returns: pd.Series,
) -> pd.Series:
    """Full-period Pearson correlation per ticker with the asset series.

    Result is a Series indexed by ticker, value = ρ(ticker_returns,
    asset_returns) over the full overlapping window.
    """
    # DataFrame.corrwith handles NaN per-pair correctly.
    return returns.corrwith(asset_returns, drop=False)


def _rolling_corr(
    returns: pd.DataFrame, asset_returns: pd.Series, window: int,
) -> pd.DataFrame:
    """Rolling Pearson correlation per ticker with the asset series.

    Output: DataFrame indexed by date, columns = tickers, values =
    rolling-window ρ. Returns DataFrame is the same shape as the input
    `returns` but with the leading `window-1` rows mostly NaN.
    """
    min_periods = max(30, int(window * _MIN_PERIODS_RATIO))
    out_cols: dict[str, pd.Series] = {}
    for ticker in returns.columns:
        s = returns[ticker]
        # Pairs of (ticker, asset) — pandas rolling.corr handles NaN cleanly.
        rolling = s.rolling(window=window, min_periods=min_periods).corr(asset_returns)
        out_cols[ticker] = rolling
    return pd.DataFrame(out_cols)


def run_cross_asset(config: PipelineConfig) -> None:
    """Pipeline stage: compute per-ticker cross-asset sensitivities.

    Gate: only ``market_id == "bist"`` runs. Skips silently otherwise.
    """
    market_id = config.market.market_id.lower()
    if market_id not in _CROSS_ASSET_MARKETS:
        logger.info(
            "Cross-asset SKIPPED for market_id=%r (only %s runs this stage)",
            market_id, sorted(_CROSS_ASSET_MARKETS),
        )
        return

    logger.info("=== Cross-Asset Sensitivity — %s ===", market_id)

    returns_path = config.data_processed / "log_returns.parquet"
    if not returns_path.exists():
        logger.warning("log_returns.parquet not found at %s", returns_path)
        return
    returns = pd.read_parquet(returns_path)
    logger.info(
        "Loaded BIST log returns: %d days × %d tickers", *returns.shape,
    )

    # Build the per-ticker summary table by accumulating one column per asset.
    summary_rows: dict[str, pd.Series] = {}
    n_obs_rows: dict[str, pd.Series] = {}

    out_dir = config.data_results
    out_dir.mkdir(parents=True, exist_ok=True)

    for asset_key in _BASE_ASSETS:
        asset_returns = _load_base_asset_returns(asset_key)
        if asset_returns.empty:
            logger.warning("Skipping %s — base asset data missing.", asset_key)
            continue

        aligned_returns, aligned_asset = _align_returns_with_asset(returns, asset_returns)
        logger.info(
            "  %s: aligned %d shared trading days (%s → %s)",
            asset_key,
            len(aligned_asset),
            aligned_asset.index.min().date(),
            aligned_asset.index.max().date(),
        )

        # Full-period summary.
        fp_corr = _full_period_corr(aligned_returns, aligned_asset)
        summary_rows[f"corr_{asset_key}"] = fp_corr
        # Per-ticker n_obs (non-NaN bars actually used).
        n_obs_rows[f"n_obs_{asset_key}"] = aligned_returns.count()

        # Rolling sensitivity series — write to its own parquet.
        rolling_df = _rolling_corr(aligned_returns, aligned_asset, _ROLLING_WINDOW)
        out_path = out_dir / f"cross_asset_corr_rolling_{asset_key}.parquet"
        rolling_df.to_parquet(out_path, compression="snappy")
        logger.info(
            "  wrote %s (%d × %d, ~%.0f KB)",
            out_path.name, rolling_df.shape[0], rolling_df.shape[1],
            out_path.stat().st_size / 1024,
        )

    # Combine summary columns with sector metadata.
    sector_map = dict(zip(config.universe["ticker"], config.universe["sector"]))
    tickers = list(returns.columns)
    rows = []
    for t in tickers:
        row: dict[str, object] = {
            "ticker": t,
            "sector": sector_map.get(t, ""),
        }
        for col_name, series in summary_rows.items():
            row[col_name] = float(series.get(t, np.nan))
        for col_name, series in n_obs_rows.items():
            row[col_name] = int(series.get(t, 0))
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / "cross_asset_summary.parquet"
    summary_df.to_parquet(summary_path, compression="snappy")
    logger.info(
        "  wrote %s (%d tickers × %d cols)",
        summary_path.name, summary_df.shape[0], summary_df.shape[1],
    )

    logger.info("Cross-asset analysis complete for %s.", market_id)
