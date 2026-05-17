"""Fetch numéraire base assets (FX, gold) for the basis-variant experiment.

Phase 4 of the mutable-candy rescue. The BIST log-returns panel is
re-expressed in USD and in gold by subtracting the log-return of the
chosen base asset:

    log r_TRY  = TRY-denominated daily log return
    log r_USD  = log r_TRY  − log r_USDTRY    (USD-denominated)
    log r_GOLD = log r_TRY  − log r_GOLD_TRY  (gold-denominated)

This module just downloads the base assets and aligns them to the BIST
trading calendar. The transform itself is in ``src/basis_transform.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaseAsset:
    """Specification for one base asset to download."""

    name: str         # logical name (e.g. "usd_try", "gold_usd")
    yf_symbol: str    # yfinance ticker (e.g. "USDTRY=X", "GC=F")
    description: str


BASE_ASSETS: tuple[BaseAsset, ...] = (
    BaseAsset(
        name="usd_try",
        yf_symbol="USDTRY=X",
        description="USD/TRY spot. Higher value = TRY weaker per USD.",
    ),
    BaseAsset(
        name="gold_usd",
        yf_symbol="GC=F",
        description="Gold futures, USD per ounce.",
    ),
)


def base_asset_path(name: str) -> Path:
    """Filesystem location for a fetched base-asset parquet."""
    return PROJECT_ROOT / "data" / "raw" / "base_assets" / f"{name}.parquet"


def fetch_base_asset(asset: BaseAsset, start: str, end: str) -> pd.Series:
    """Download one base asset's adjusted close series via yfinance."""
    logger.info(
        "Downloading base asset %s (%s) %s → %s",
        asset.name, asset.yf_symbol, start, end,
    )
    df = yf.download(
        tickers=asset.yf_symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(
            f"yfinance returned empty data for {asset.yf_symbol} ({asset.name})"
        )
    # yfinance >= 0.2.x returns MultiIndex columns when even a single ticker
    # is requested; squeeze it down.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Adj Close" in df.columns:
        series = df["Adj Close"].astype(float)
    elif "Close" in df.columns:
        series = df["Close"].astype(float)
    else:
        raise RuntimeError(
            f"Neither 'Adj Close' nor 'Close' present for {asset.yf_symbol}; "
            f"got columns {df.columns.tolist()}"
        )
    series.name = asset.name
    return series.dropna()


def fetch_all_base_assets(
    start: str,
    end: str,
    assets: tuple[BaseAsset, ...] = BASE_ASSETS,
) -> dict[str, pd.Series]:
    """Fetch every asset in `assets` and persist to ``data/raw/base_assets/``.

    Returns a dict {name: series}.
    """
    out_dir = PROJECT_ROOT / "data" / "raw" / "base_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.Series] = {}
    for a in assets:
        series = fetch_base_asset(a, start, end)
        path = base_asset_path(a.name)
        series.to_frame().to_parquet(path)
        logger.info(
            "Saved %s: %d observations, %s → %s → %s",
            a.name, len(series),
            series.index.min().date(), series.index.max().date(), path,
        )
        result[a.name] = series
    return result


def load_base_asset(name: str) -> pd.Series:
    """Load a cached base-asset series (raises FileNotFoundError if missing)."""
    path = base_asset_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Base asset {name!r} not fetched yet. Path: {path}. "
            "Run scripts/fetch_base_assets.py or call fetch_all_base_assets()."
        )
    df = pd.read_parquet(path)
    return df.iloc[:, 0].astype(float).dropna()


def gold_in_try(start: str, end: str) -> pd.Series:
    """Gold price denominated in TRY = gold_usd × USDTRY, aligned to common
    trading days. Auto-fetches both inputs if missing.
    """
    try:
        usd_try = load_base_asset("usd_try")
        gold_usd = load_base_asset("gold_usd")
    except FileNotFoundError:
        fetch_all_base_assets(start, end)
        usd_try = load_base_asset("usd_try")
        gold_usd = load_base_asset("gold_usd")

    common = usd_try.index.intersection(gold_usd.index)
    gold_try = (gold_usd.loc[common] * usd_try.loc[common]).astype(float)
    gold_try.name = "gold_try"
    return gold_try
