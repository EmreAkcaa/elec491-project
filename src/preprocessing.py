"""Preprocessing: coverage filter, log returns, anomaly detection."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_raw_prices(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw parquet and return (adj_close, raw_close) DataFrames."""
    raw_path = DATA_RAW / "prices_raw.parquet"
    df = pd.read_parquet(raw_path)

    if isinstance(df.columns, pd.MultiIndex):
        adj_close = df["Adj Close"] if "Adj Close" in df.columns.get_level_values(0) else pd.DataFrame()
        raw_close = df["Close"] if "Close" in df.columns.get_level_values(0) else pd.DataFrame()
    else:
        adj_close = df
        raw_close = pd.DataFrame()

    logger.info(
        "Loaded raw prices: %d days x %d tickers",
        len(adj_close),
        adj_close.shape[1] if not adj_close.empty else 0,
    )
    return adj_close, raw_close


def compute_coverage(adj_close: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker data coverage.

    Coverage = non-NaN trading days / total trading days in the date range.
    """
    total_days = len(adj_close)
    available = adj_close.notna().sum()

    coverage = pd.DataFrame(
        {
            "ticker": available.index,
            "total_days": total_days,
            "available_days": available.values,
            "coverage_pct": (available.values / total_days).round(4),
        }
    )
    return coverage.sort_values("coverage_pct", ascending=False).reset_index(drop=True)


def filter_by_coverage(
    adj_close: pd.DataFrame,
    raw_close: pd.DataFrame,
    threshold: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop tickers below coverage threshold.

    Does NOT do inner join — surviving tickers keep their original NaN pattern.
    Returns (filtered_adj_close, filtered_raw_close, coverage_report).
    """
    coverage = compute_coverage(adj_close)

    passing = coverage[coverage["coverage_pct"] >= threshold]["ticker"].tolist()
    dropped = coverage[coverage["coverage_pct"] < threshold]["ticker"].tolist()

    logger.info(
        "Coverage filter (%.0f%%): %d pass, %d dropped",
        threshold * 100,
        len(passing),
        len(dropped),
    )
    if dropped:
        logger.info("Dropped tickers: %s", dropped)

    filtered_adj = adj_close[passing]

    # Filter raw_close to same tickers (if available)
    if not raw_close.empty:
        common = [t for t in passing if t in raw_close.columns]
        filtered_raw = raw_close[common]
    else:
        filtered_raw = pd.DataFrame()

    return filtered_adj, filtered_raw, coverage


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns: ln(P_t / P_{t-1}).

    Drops the first row (NaN from shift).
    """
    returns = np.log(prices / prices.shift(1))
    returns = returns.iloc[1:]  # drop first NaN row
    logger.info(
        "Computed log returns: %d days x %d tickers",
        returns.shape[0],
        returns.shape[1],
    )
    return returns


def flag_anomalies(
    returns: pd.DataFrame, threshold: float = 0.30
) -> pd.DataFrame:
    """Flag return observations where |r| > threshold.

    Returns a DataFrame with columns: date, ticker, return_value.
    """
    mask = returns.abs() > threshold
    anomalies = returns.where(mask).stack().dropna().reset_index()
    anomalies.columns = ["date", "ticker", "return_value"]
    anomalies = anomalies.sort_values("return_value", key=abs, ascending=False)

    logger.info("Flagged %d anomalous return observations (|r| > %.2f)", len(anomalies), threshold)
    return anomalies.reset_index(drop=True)


def run_preprocessing(config: PipelineConfig) -> None:
    """Full preprocessing pipeline step."""
    logger.info("=== Preprocessing ===")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    adj_close, raw_close = load_raw_prices(config)

    # Coverage filter
    filtered_adj, filtered_raw, coverage = filter_by_coverage(
        adj_close, raw_close, threshold=config.preprocessing.min_coverage_pct
    )

    # Log returns
    log_returns = compute_log_returns(filtered_adj)

    # Anomaly detection
    anomalies = flag_anomalies(
        log_returns, threshold=config.preprocessing.anomaly_return_threshold
    )

    # Save artifacts
    filtered_adj.to_parquet(DATA_PROCESSED / "adj_close.parquet")
    logger.info("Saved filtered adj close")

    if not filtered_raw.empty:
        filtered_raw.to_parquet(DATA_PROCESSED / "raw_close.parquet")
        logger.info("Saved filtered raw close")

    log_returns.to_parquet(DATA_PROCESSED / "log_returns.parquet")
    logger.info("Saved log returns")

    coverage.to_csv(DATA_PROCESSED / "coverage_report.csv", index=False)
    logger.info("Saved coverage report")

    anomalies.to_csv(DATA_PROCESSED / "anomalies.csv", index=False)
    logger.info("Saved anomalies")

    logger.info("Preprocessing complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.config import load_config

    cfg = load_config()
    run_preprocessing(cfg)
