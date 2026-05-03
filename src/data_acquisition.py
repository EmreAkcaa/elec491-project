"""Data acquisition from yfinance for BIST-100 equities."""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import PipelineConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

DATA_RAW = PROJECT_ROOT / "data" / "raw"


def fetch_all_tickers(config: PipelineConfig) -> tuple[pd.DataFrame, list[str]]:
    """Download price data for all universe tickers via yfinance.

    Downloads in chunks of 25 with a 1-second delay between chunks.

    Returns
    -------
    combined : pd.DataFrame
        Wide DataFrame with MultiIndex columns: (field, ticker).
        Fields include 'Adj Close' and 'Close'.
    failures : list[str]
        Tickers that failed to download or are missing from the result.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    symbols = config.provider_symbols
    start = config.data.start_date
    end = config.data.end_date
    interval = config.data.download_interval

    chunk_size = 25
    chunks = [symbols[i : i + chunk_size] for i in range(0, len(symbols), chunk_size)]

    all_frames = []
    failures = []

    for i, chunk in enumerate(chunks):
        logger.info(
            "Downloading chunk %d/%d (%d tickers)...",
            i + 1,
            len(chunks),
            len(chunk),
        )
        try:
            df = yf.download(
                tickers=chunk,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if df is not None and not df.empty:
                all_frames.append(df)
            else:
                logger.warning("Chunk %d returned empty data", i + 1)
                failures.extend(chunk)
        except Exception as e:
            logger.error("Chunk %d failed: %s", i + 1, e)
            failures.extend(chunk)

        if i < len(chunks) - 1:
            time.sleep(1)

    if not all_frames:
        raise RuntimeError("No data downloaded from any chunk")

    combined = pd.concat(all_frames, axis=1)

    # yfinance returns MultiIndex columns: (field, ticker)
    # Keep only Adj Close and Close
    fields_to_keep = ["Adj Close", "Close"]
    available_fields = [f for f in fields_to_keep if f in combined.columns.get_level_values(0)]
    combined = combined[available_fields]

    # Build a reverse map: provider_symbol -> ticker
    sym_to_ticker = dict(
        zip(config.universe["provider_symbol"], config.universe["ticker"])
    )

    # Rename provider_symbols to tickers in the column level
    new_cols = []
    for field_name, sym in combined.columns:
        ticker = sym_to_ticker.get(sym, sym)
        new_cols.append((field_name, ticker))
    combined.columns = pd.MultiIndex.from_tuples(new_cols, names=["field", "ticker"])

    # Check which tickers are missing
    downloaded_tickers = set(combined.columns.get_level_values("ticker"))
    expected_tickers = set(config.tickers)
    missing = expected_tickers - downloaded_tickers
    if missing:
        logger.warning("Missing tickers after download: %s", sorted(missing))
        failures.extend(sorted(missing))

    logger.info(
        "Downloaded %d tickers, %d failures",
        len(downloaded_tickers),
        len(set(failures)),
    )

    return combined, list(set(failures))


def fetch_index(config: PipelineConfig) -> pd.DataFrame:
    """Download the market index (XU100) separately."""
    index_sym = config.index_provider_symbol
    logger.info("Downloading index: %s", index_sym)

    df = yf.download(
        tickers=index_sym,
        start=config.data.start_date,
        end=config.data.end_date,
        interval=config.data.download_interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        logger.warning("Index %s returned no data", index_sym)
        return pd.DataFrame()

    return df


def save_raw_data(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    failures: list[str],
    config: PipelineConfig,
) -> None:
    """Persist raw price data and metadata."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    prices.to_parquet(DATA_RAW / "prices_raw.parquet")
    logger.info("Saved raw prices: %s", DATA_RAW / "prices_raw.parquet")

    if not index_df.empty:
        index_df.to_parquet(DATA_RAW / "xu100.parquet")
        logger.info("Saved index: %s", DATA_RAW / "xu100.parquet")

    metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "source": "yfinance",
        "start_date": config.data.start_date,
        "end_date": config.data.end_date,
        "ticker_count": len(
            set(prices.columns.get_level_values("ticker"))
        ),
        "failures": failures,
    }
    with open(DATA_RAW / "fetch_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved fetch metadata")


def run_acquisition(config: PipelineConfig) -> None:
    """Full acquisition pipeline step."""
    logger.info("=== Data Acquisition ===")
    prices, failures = fetch_all_tickers(config)
    index_df = fetch_index(config)
    save_raw_data(prices, index_df, failures, config)
    logger.info("Acquisition complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.config import load_config

    cfg = load_config()
    run_acquisition(cfg)
