"""Analysis: descriptive stats, correlation, distance matrix."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import PipelineConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)



def compute_descriptive_stats(
    returns: pd.DataFrame, ann_factor: int = 252
) -> pd.DataFrame:
    """Per-stock descriptive statistics.

    Includes: mean, std, annualized vol, min, max, skewness, kurtosis,
    count (non-NaN observations).
    """
    summary = pd.DataFrame(
        {
            "ticker": returns.columns,
            "count": returns.count().values,
            "mean_daily_return": returns.mean().values,
            "std_daily_return": returns.std().values,
            "annualized_return": (returns.mean().values * ann_factor),
            "annualized_vol": (returns.std().values * np.sqrt(ann_factor)),
            "min_return": returns.min().values,
            "max_return": returns.max().values,
            "skewness": returns.skew().values,
            "kurtosis": returns.kurtosis().values,
        }
    )
    return summary.sort_values("annualized_vol", ascending=False).reset_index(
        drop=True
    )


def compute_correlation_matrix(
    returns: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 200,
) -> pd.DataFrame:
    """Pairwise correlation using pairwise-complete observations.

    Uses min_periods to ensure sufficient overlap for meaningful correlations.
    """
    corr = returns.corr(method=method, min_periods=min_periods)
    n_total = corr.shape[0] * corr.shape[1]
    n_nan = corr.isna().sum().sum()
    n_valid = n_total - n_nan
    logger.info(
        "Correlation matrix: %dx%d, %d valid pairs (%.1f%%), %d NaN",
        corr.shape[0],
        corr.shape[1],
        n_valid,
        100 * n_valid / max(n_total, 1),
        n_nan,
    )
    return corr


def compute_distance_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """Distance matrix: d = sqrt(2 * (1 - rho))."""
    dist = np.sqrt(2 * (1 - corr))
    return dist


def get_top_bottom_pairs(
    corr: pd.DataFrame,
    universe: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Extract top-N most and bottom-N least correlated pairs.

    Uses upper triangle only (no diagonal, no duplicates).
    """
    # Build sector lookup
    sector_map = dict(zip(universe["ticker"], universe["sector"]))

    # Upper triangle mask
    # Reset index/column names to avoid conflicts during stack/reset_index
    corr_clean = corr.copy()
    corr_clean.index.name = "row"
    corr_clean.columns.name = "col"
    mask = np.triu(np.ones(corr_clean.shape, dtype=bool), k=1)
    pairs = corr_clean.where(mask).stack().reset_index()
    pairs.columns = ["ticker_1", "ticker_2", "correlation"]
    pairs = pairs.dropna(subset=["correlation"])

    pairs["sector_1"] = pairs["ticker_1"].map(sector_map)
    pairs["sector_2"] = pairs["ticker_2"].map(sector_map)

    top_n = pairs.nlargest(n, "correlation")
    bottom_n = pairs.nsmallest(n, "correlation")

    result = pd.concat(
        [top_n.assign(rank_type="top"), bottom_n.assign(rank_type="bottom")]
    )
    return result.reset_index(drop=True)


def compute_market_summary(corr: pd.DataFrame) -> dict:
    """Summary statistics of the pairwise correlation distribution."""
    # Upper triangle only (exclude diagonal)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper = corr.where(mask).stack().dropna()

    return {
        "avg_pairwise_corr": round(float(upper.mean()), 4),
        "median_pairwise_corr": round(float(upper.median()), 4),
        "std_pairwise_corr": round(float(upper.std()), 4),
        "min_pairwise_corr": round(float(upper.min()), 4),
        "max_pairwise_corr": round(float(upper.max()), 4),
        "n_pairs": int(len(upper)),
        "n_tickers": int(corr.shape[0]),
    }


def run_analysis(config: PipelineConfig) -> None:
    """Full analysis pipeline step."""
    logger.info("=== Analysis ===")
    config.data_results.mkdir(parents=True, exist_ok=True)

    # Load preprocessed returns
    returns = pd.read_parquet(config.data_processed / "log_returns.parquet")
    logger.info("Loaded log returns: %d days x %d tickers", *returns.shape)

    # Descriptive stats
    summary = compute_descriptive_stats(
        returns, ann_factor=config.analysis.annualization_factor
    )
    summary.to_parquet(config.data_results / "summary_stats.parquet")
    logger.info("Saved summary stats")

    # Correlation matrix
    corr = compute_correlation_matrix(
        returns,
        method=config.analysis.correlation_method,
        min_periods=config.analysis.corr_min_periods,
    )
    corr.to_parquet(config.data_results / "pearson_corr.parquet")
    logger.info("Saved Pearson correlation matrix")

    # Distance matrix
    dist = compute_distance_matrix(corr)
    dist.to_parquet(config.data_results / "distance_matrix.parquet")
    logger.info("Saved distance matrix")

    # Top/bottom pairs
    pairs = get_top_bottom_pairs(corr, config.universe, n=10)
    pairs.to_csv(config.data_results / "top_bottom_pairs.csv", index=False)
    logger.info("Saved top/bottom pairs")

    # Market summary
    market_summary = compute_market_summary(corr)
    logger.info("Market summary: %s", market_summary)

    # Pipeline metadata
    coverage = pd.read_csv(config.data_processed / "coverage_report.csv")
    metadata = {
        "run_timestamp": pd.Timestamp.now().isoformat(),
        "config": {
            "start_date": config.data.start_date,
            "end_date": config.data.end_date,
            "min_coverage_pct": config.preprocessing.min_coverage_pct,
            "correlation_method": config.analysis.correlation_method,
            "corr_min_periods": config.analysis.corr_min_periods,
        },
        "universe_count": len(config.tickers),
        "tickers_after_filter": int(returns.shape[1]),
        "trading_days": int(returns.shape[0]),
        "market_summary": market_summary,
    }
    with open(config.data_results / "pipeline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved pipeline metadata")

    logger.info("Analysis complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.config import load_config

    cfg = load_config()
    run_analysis(cfg)
