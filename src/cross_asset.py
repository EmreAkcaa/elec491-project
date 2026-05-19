"""Cross-asset correlation analysis (Phase X, properly built 2026-05-19).

For BIST family only. Computes per-ticker correlation against base-asset
log returns (USD/TRY, Gold-USD) at two horizons:

  * Full-period Pearson: summarised in ``cross_asset_summary.parquet``.
    One row per ticker; columns ``corr_usd_try``, ``corr_gold_usd``,
    ``n_obs_usd_try``, ``n_obs_gold_usd`` plus sector metadata.

  * 252-day rolling Pearson: panels in
    ``cross_asset_corr_rolling_usd_try.parquet`` and
    ``cross_asset_corr_rolling_gold_usd.parquet``. Indexed by date,
    columns = tickers, values = rolling correlation. Used by the
    Signals page to surface stocks whose current rolling beta has
    deviated from their historical baseline.

Sign convention:

  * USD/TRY = TRY per 1 USD. Rising = TRY weakening.
  * Gold-USD = USD per oz of gold. Rising = gold strengthening (which
    typically tracks USD weakening).

So a negative ``corr_usd_try`` means the stock loses value when TRY
weakens — bank-style behaviour. A positive ``corr_usd_try`` means the
stock benefits from TRY weakness — exporter-style behaviour.

Look-ahead: none. All rolling computations are past-only (``min_periods``
bounded, ``rolling().corr()`` is left-aligned by date). Full-period
correlations are by definition a property of the historical window;
they are not used as predictive signals at training time.

Gating: ``config.market.market_id == "bist"`` — other markets skip with
a single info-log line. Output files are universe-keyed via
``config.data_results``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PipelineConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

# Base assets live outside any universe results dir — they're shared series.
# Same path the dashboard's `load_base_asset` reads from.
_BASE_ASSETS_DIR = PROJECT_ROOT / "data" / "raw" / "base_assets"

# Rolling window for the cross-asset beta panels. 252 trading days ≈ 1 year.
_ROLLING_WINDOW = 252
_ROLLING_MIN_PERIODS_RATIO = 0.6  # ≥152 obs required to emit a value.


def _load_base_asset_returns(asset_key: str) -> Optional[pd.Series]:
    """Load a base asset price series and convert to log returns.

    Returns ``None`` (and logs a warning) if the parquet is missing.
    """
    path = _BASE_ASSETS_DIR / f"{asset_key}.parquet"
    if not path.exists():
        logger.warning("cross_asset: base asset %s not found at %s", asset_key, path)
        return None
    df = pd.read_parquet(path)
    if df.empty or df.shape[1] == 0:
        logger.warning("cross_asset: base asset %s parquet is empty", asset_key)
        return None
    # The column is named after the asset (e.g. 'usd_try'); take the first
    # column defensively in case it's renamed downstream.
    price = df.iloc[:, 0].astype(float).dropna()
    log_returns = np.log(price / price.shift(1)).dropna()
    log_returns.name = asset_key
    return log_returns


def _rolling_corr_panel(
    returns: pd.DataFrame,
    base_returns: pd.Series,
    window: int = _ROLLING_WINDOW,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Per-ticker rolling Pearson correlation against a single base asset.

    Both inputs are inner-joined on the date index before rolling. The
    output is a (date × ticker) DataFrame; values before ``min_periods``
    are observed are NaN by construction.
    """
    if min_periods is None:
        min_periods = int(window * _ROLLING_MIN_PERIODS_RATIO)
    common = returns.index.intersection(base_returns.index)
    if len(common) < min_periods:
        logger.warning(
            "cross_asset: not enough overlap to compute rolling corr (%d < %d)",
            len(common),
            min_periods,
        )
        return pd.DataFrame()
    r = returns.loc[common]
    b = base_returns.loc[common]
    # pandas' Series.rolling().corr(other_series) operates column-wise when
    # the calling object is a DataFrame; the result is a (date × ticker)
    # frame.
    panel = r.rolling(window=window, min_periods=min_periods).corr(b)
    return panel


def _full_period_corr(
    returns: pd.DataFrame,
    base_returns: pd.Series,
) -> pd.DataFrame:
    """Per-ticker full-period Pearson correlation against a base asset.

    Returns a DataFrame indexed by ticker with columns ``corr`` and ``n_obs``.
    Each ticker's ``n_obs`` reflects the per-ticker non-NaN overlap with the
    base asset (so a ticker that listed late shows fewer observations).
    """
    common = returns.index.intersection(base_returns.index)
    if len(common) == 0:
        return pd.DataFrame(columns=["corr", "n_obs"])
    r = returns.loc[common]
    b = base_returns.loc[common]

    out: dict[str, dict[str, float]] = {}
    for ticker in r.columns:
        col = r[ticker]
        mask = col.notna() & b.notna()
        n = int(mask.sum())
        if n < 30:  # arbitrary minimum to avoid silly correlations on micro-samples
            out[ticker] = {"corr": float("nan"), "n_obs": n}
            continue
        corr = float(col[mask].corr(b[mask]))
        out[ticker] = {"corr": corr, "n_obs": n}
    return pd.DataFrame(out).T


def run_cross_asset(config: PipelineConfig) -> None:
    """Pipeline stage: write per-ticker cross-asset correlation panels.

    Gated on ``config.market.market_id == "bist"``. Other markets log
    a single 'skipped' line and return.

    Artifacts written under ``config.data_results``:

      * ``cross_asset_corr_rolling_usd_try.parquet``  (date × ticker)
      * ``cross_asset_corr_rolling_gold_usd.parquet`` (date × ticker)
      * ``cross_asset_summary.parquet``               (ticker × stats)

    The Signals page reads these via cached loaders in ``app.utils``. The
    summary parquet powers the cross-asset breakout leaderboard; the
    rolling panels power the time-series chart that contextualises the
    current beta against its historical band.
    """
    logger.info("=== Cross-asset correlation ===")
    market_id = config.market.market_id.lower()
    if market_id != "bist":
        logger.info(
            "cross_asset: skipping (market_id=%s, BIST family only)", market_id,
        )
        return

    # Load TRY-base log returns (the BIST stocks' native return panel).
    returns_path = config.data_processed / "log_returns.parquet"
    if not returns_path.exists():
        logger.warning("cross_asset: %s missing, skipping stage", returns_path)
        return
    returns = pd.read_parquet(returns_path)
    logger.info(
        "cross_asset: loaded log_returns %d days x %d tickers",
        returns.shape[0],
        returns.shape[1],
    )

    # Load base assets (price -> log return). usd_try and gold_usd are the
    # canonical keys, matching `app.utils._load_base_asset` and the schema
    # in Pair Analysis's Compare-against picker.
    base_assets: dict[str, pd.Series] = {}
    for key in ("usd_try", "gold_usd"):
        series = _load_base_asset_returns(key)
        if series is not None:
            base_assets[key] = series

    if not base_assets:
        logger.warning("cross_asset: no base assets available, stage produced nothing")
        return

    config.data_results.mkdir(parents=True, exist_ok=True)

    # ── Rolling panels ──────────────────────────────────────────────────
    for key, base_returns in base_assets.items():
        panel = _rolling_corr_panel(returns, base_returns, window=_ROLLING_WINDOW)
        if panel.empty:
            logger.warning("cross_asset: rolling panel for %s is empty", key)
            continue
        out_path = config.data_results / f"cross_asset_corr_rolling_{key}.parquet"
        panel.to_parquet(out_path)
        non_nan_dates = int(panel.dropna(how="all").shape[0])
        logger.info(
            "cross_asset: wrote rolling panel %s (%d dates with values, %d tickers)",
            out_path.name,
            non_nan_dates,
            panel.shape[1],
        )

    # ── Full-period summary ─────────────────────────────────────────────
    summary_parts: list[pd.DataFrame] = []
    for key, base_returns in base_assets.items():
        df = _full_period_corr(returns, base_returns)
        df = df.rename(columns={"corr": f"corr_{key}", "n_obs": f"n_obs_{key}"})
        summary_parts.append(df)

    summary = pd.concat(summary_parts, axis=1)
    summary.index.name = "ticker"

    # Attach sector metadata from the loaded universe CSV (already on
    # `config.universe`). If a ticker is in the panel but not in the
    # universe (rare — defensive) we just leave sector NaN; the Signals
    # page falls back to "Unknown".
    try:
        if "sector" in config.universe.columns and "ticker" in config.universe.columns:
            sector_map = (
                config.universe.set_index("ticker")["sector"].to_dict()
            )
            summary["sector"] = summary.index.map(sector_map)
    except Exception as exc:  # pragma: no cover — diagnostic only
        logger.warning("cross_asset: failed to attach sector metadata: %s", exc)

    summary = summary.reset_index()
    summary_path = config.data_results / "cross_asset_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    logger.info(
        "cross_asset: wrote summary %s (%d tickers, columns=%s)",
        summary_path.name,
        len(summary),
        list(summary.columns),
    )

    # Helpful sanity log: report the most-TRY-sensitive and most-Gold-
    # sensitive tickers. Banks should appear in the negative-TRY camp.
    try:
        if "corr_usd_try" in summary.columns:
            top_neg_try = summary.dropna(subset=["corr_usd_try"]).nsmallest(3, "corr_usd_try")
            logger.info(
                "cross_asset: most-negative USD/TRY-correlated tickers: %s",
                list(zip(top_neg_try["ticker"], top_neg_try["corr_usd_try"].round(3))),
            )
    except Exception:
        pass

    logger.info("Cross-asset correlation complete")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    from src.config import load_config

    cfg = load_config()
    run_cross_asset(cfg)
