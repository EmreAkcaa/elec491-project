"""Cross-validation of yfinance data against isyatirimhisse."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def _select_sample_tickers(
    tickers: list[str], n: int = 10, seed: int = 42
) -> list[str]:
    """Deterministically select n sample tickers for validation."""
    rng = np.random.default_rng(seed)
    n = min(n, len(tickers))
    indices = rng.choice(len(tickers), size=n, replace=False)
    return [tickers[i] for i in sorted(indices)]


def _to_ddmmyyyy(iso_date: str) -> str:
    """Convert 'YYYY-MM-DD' to 'DD-MM-YYYY' for isyatirimhisse."""
    parts = iso_date.split("-")
    return f"{parts[2]}-{parts[1]}-{parts[0]}"


def _fetch_isyatirim_data(
    ticker: str, start_date: str, end_date: str
) -> pd.Series | None:
    """Fetch daily close prices from isyatirimhisse for a single ticker."""
    try:
        from isyatirimhisse import fetch_stock_data

        df = fetch_stock_data(
            symbols=ticker,
            start_date=_to_ddmmyyyy(start_date),
            end_date=_to_ddmmyyyy(end_date),
        )

        if df is None or df.empty:
            logger.warning("isyatirimhisse returned no data for %s", ticker)
            return None

        # v5 API returns HGDG_KAPANIS for closing price, HGDG_TARIH for date
        if "HGDG_KAPANIS" not in df.columns or "HGDG_TARIH" not in df.columns:
            logger.warning("Unexpected columns for %s: %s", ticker, df.columns.tolist())
            return None

        df["HGDG_TARIH"] = pd.to_datetime(df["HGDG_TARIH"])
        series = df.set_index("HGDG_TARIH")["HGDG_KAPANIS"].astype(float)
        return series.sort_index()

    except Exception as e:
        logger.warning("isyatirimhisse fetch failed for %s: %s", ticker, e)
        return None


def _compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a price series."""
    return np.log(prices / prices.shift(1)).dropna()


def validate_sample(config: PipelineConfig) -> pd.DataFrame:
    """Cross-validate a sample of tickers against isyatirimhisse.

    Compares daily log returns (not price levels) to be robust against
    differing adjustment conventions across data sources.
    """
    if not config.validation.enabled:
        logger.info("Validation disabled in config, skipping")
        return pd.DataFrame()

    logger.info("=== Data Validation ===")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Load yfinance data
    raw_path = DATA_RAW / "prices_raw.parquet"
    if not raw_path.exists():
        logger.warning("Raw prices not found, skipping validation")
        return pd.DataFrame()

    yf_prices = pd.read_parquet(raw_path)

    # Get adj close columns
    if isinstance(yf_prices.columns, pd.MultiIndex):
        adj_close = yf_prices["Adj Close"]
    else:
        adj_close = yf_prices

    # Select sample tickers
    available_tickers = [t for t in config.tickers if t in adj_close.columns]
    sample_tickers = _select_sample_tickers(
        available_tickers, n=config.validation.sample_size
    )
    logger.info("Validation sample tickers: %s", sample_tickers)

    results = []
    for ticker in sample_tickers:
        logger.info("Validating %s...", ticker)

        # yfinance log returns
        yf_series = adj_close[ticker].dropna()
        if yf_series.empty:
            results.append(
                {
                    "ticker": ticker,
                    "mean_abs_return_diff": np.nan,
                    "max_abs_return_diff": np.nan,
                    "n_days_compared": 0,
                    "status": "NO_YF_DATA",
                }
            )
            continue

        yf_returns = _compute_log_returns(yf_series)

        # isyatirimhisse log returns
        isy_series = _fetch_isyatirim_data(
            ticker, config.data.start_date, config.data.end_date
        )

        if isy_series is None or isy_series.empty:
            results.append(
                {
                    "ticker": ticker,
                    "mean_abs_return_diff": np.nan,
                    "max_abs_return_diff": np.nan,
                    "n_days_compared": 0,
                    "status": "NO_ISY_DATA",
                }
            )
            continue

        isy_returns = _compute_log_returns(isy_series)

        # Align on common dates
        common_dates = yf_returns.index.intersection(isy_returns.index)
        n_common = len(common_dates)

        if n_common < 10:
            results.append(
                {
                    "ticker": ticker,
                    "mean_abs_return_diff": np.nan,
                    "max_abs_return_diff": np.nan,
                    "n_days_compared": n_common,
                    "status": "INSUFFICIENT_OVERLAP",
                }
            )
            continue

        diff = (yf_returns.loc[common_dates] - isy_returns.loc[common_dates]).abs()
        mean_diff = diff.mean()
        max_diff = diff.max()

        status = "PASS" if mean_diff < 0.02 else "FLAG"
        if status == "FLAG":
            logger.warning(
                "%s flagged: mean abs return diff = %.4f", ticker, mean_diff
            )

        results.append(
            {
                "ticker": ticker,
                "mean_abs_return_diff": round(mean_diff, 6),
                "max_abs_return_diff": round(max_diff, 6),
                "n_days_compared": n_common,
                "status": status,
            }
        )

    report = pd.DataFrame(results)

    # Check failure rate
    if not report.empty:
        n_flagged = (report["status"] == "FLAG").sum()
        n_total = len(report)
        if n_flagged > n_total / 2:
            logger.warning(
                "WARNING: >50%% of validation sample failed (%d/%d). "
                "Data may have quality issues but pipeline continues.",
                n_flagged,
                n_total,
            )

    # Save report
    report.to_csv(DATA_PROCESSED / "validation_report.csv", index=False)
    logger.info("Validation report saved to %s", DATA_PROCESSED / "validation_report.csv")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.config import load_config

    cfg = load_config()
    validate_sample(cfg)
