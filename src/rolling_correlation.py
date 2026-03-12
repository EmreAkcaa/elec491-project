"""Rolling / sliding-window correlation analysis with comprehensive options.

Supports:
- Rolling (fixed window), expanding, and EWMA windowing
- Pearson, Spearman, and Kendall correlation methods
- Market-level aggregate stats over time
- Pair-level rolling correlation
- Sector-level intra vs inter correlation breakdown
- Point-in-time correlation matrix snapshots
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PipelineConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RESULTS = PROJECT_ROOT / "data" / "results"

# Notable BIST-100 events for overlay on time-series charts
DEFAULT_EVENTS = [
    {"date": "2020-03-11", "label": "WHO Declares Pandemic"},
    {"date": "2022-02-24", "label": "Russia-Ukraine War"},
    {"date": "2023-02-06", "label": "Turkey Earthquakes"},
]


def compute_rolling_market_stats(
    returns: pd.DataFrame,
    window: int = 252,
    step: int = 1,
    min_periods: Optional[int] = None,
    method: str = "pearson",
    expanding: bool = False,
) -> pd.DataFrame:
    """Time series of market-level pairwise correlation statistics.

    For each window position, computes the full correlation matrix and
    extracts summary statistics from the upper triangle.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns (dates x tickers).
    window : int
        Rolling window size in trading days (ignored if expanding=True).
    step : int
        Step size between consecutive windows.
        1 = daily, 5 ~ weekly, 21 ~ monthly.
    min_periods : int or None
        Min non-NaN observations per pair within a window.
        Defaults to max(30, window * 0.6).
    method : str
        'pearson', 'spearman', or 'kendall'.
    expanding : bool
        If True, use expanding window from the start (ignores `window`).

    Returns
    -------
    pd.DataFrame
        Indexed by date with columns: avg_corr, median_corr, std_corr,
        min_corr, max_corr, q25_corr, q75_corr, n_valid_pairs,
        n_tickers_in_window.
    """
    if min_periods is None:
        min_periods = max(30, int(window * 0.6))

    n_days = len(returns)
    records = []

    start_from = min_periods if expanding else window

    for end_idx in range(start_from, n_days + 1, step):
        if expanding:
            window_returns = returns.iloc[:end_idx]
        else:
            window_returns = returns.iloc[end_idx - window : end_idx]

        end_date = window_returns.index[-1]

        # Keep only tickers with enough data in this window
        obs_counts = window_returns.count()
        valid_tickers = obs_counts[obs_counts >= min_periods].index
        if len(valid_tickers) < 2:
            continue

        corr = window_returns[valid_tickers].corr(method=method)

        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        upper = corr.values[mask]
        upper = upper[np.isfinite(upper)]

        if len(upper) == 0:
            continue

        records.append(
            {
                "date": end_date,
                "avg_corr": float(np.mean(upper)),
                "median_corr": float(np.median(upper)),
                "std_corr": float(np.std(upper)),
                "min_corr": float(np.min(upper)),
                "max_corr": float(np.max(upper)),
                "q25_corr": float(np.percentile(upper, 25)),
                "q75_corr": float(np.percentile(upper, 75)),
                "n_valid_pairs": len(upper),
                "n_tickers_in_window": len(valid_tickers),
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    logger.info(
        "Rolling market stats: window=%d, step=%d, method=%s, expanding=%s, %d points",
        window,
        step,
        method,
        expanding,
        len(df),
    )
    return df


def compute_rolling_pair_correlation(
    returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int = 252,
    min_periods: Optional[int] = None,
    method: str = "pearson",
    window_type: str = "rolling",
    ewm_span: Optional[int] = None,
) -> pd.Series:
    """Rolling correlation for a specific stock pair.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns (dates x tickers).
    ticker_a, ticker_b : str
        Ticker symbols.
    window : int
        Window size (ignored for expanding/ewm unless ewm_span is None).
    min_periods : int or None
        Defaults to max(30, window * 0.6).
    method : str
        'pearson', 'spearman', or 'kendall'.
    window_type : str
        'rolling' (fixed window), 'expanding', or 'ewm'.
    ewm_span : int or None
        Span for EWMA. Defaults to `window` if None.

    Returns
    -------
    pd.Series
        Rolling correlation indexed by date.
    """
    if min_periods is None:
        min_periods = max(30, int(window * 0.6))

    a = returns[ticker_a]
    b = returns[ticker_b]

    if window_type == "ewm":
        span = ewm_span if ewm_span is not None else window
        rolling_corr = a.ewm(span=span, min_periods=min_periods).corr(b)
    elif window_type == "expanding":
        rolling_corr = a.expanding(min_periods=min_periods).corr(b)
    elif method == "pearson":
        # Fast vectorized path for Pearson
        rolling_corr = a.rolling(window, min_periods=min_periods).corr(b)
    else:
        # Spearman / Kendall: iterate per window (correct rank computation)
        values = []
        dates = []
        for end_idx in range(window, len(returns) + 1):
            w = returns.iloc[end_idx - window : end_idx]
            wa = w[ticker_a].dropna()
            wb = w[ticker_b].dropna()
            common = wa.index.intersection(wb.index)
            if len(common) >= min_periods:
                values.append(float(wa[common].corr(wb[common], method=method)))
            else:
                values.append(np.nan)
            dates.append(returns.index[end_idx - 1])
        rolling_corr = pd.Series(values, index=pd.DatetimeIndex(dates))

    rolling_corr.name = f"{ticker_a}_{ticker_b}_{method}"
    return rolling_corr


def compute_rolling_sector_stats(
    returns: pd.DataFrame,
    sector_map: dict,
    window: int = 252,
    step: int = 5,
    min_periods: Optional[int] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Rolling intra-sector vs inter-sector average correlation.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns.
    sector_map : dict
        Mapping of ticker -> sector.
    window, step, min_periods, method :
        Same as compute_rolling_market_stats.

    Returns
    -------
    pd.DataFrame
        Indexed by date with columns: intra_sector_avg, inter_sector_avg,
        plus per-sector intra-correlation columns (intra_Banking, etc.).
    """
    if min_periods is None:
        min_periods = max(30, int(window * 0.6))

    # Only use tickers that have sector info
    tickers = [t for t in returns.columns if t in sector_map]
    returns_filtered = returns[tickers]
    sectors = {t: sector_map[t] for t in tickers}
    unique_sectors = sorted(set(sectors.values()))

    n_days = len(returns_filtered)
    records = []

    for end_idx in range(window, n_days + 1, step):
        window_returns = returns_filtered.iloc[end_idx - window : end_idx]
        end_date = window_returns.index[-1]

        obs_counts = window_returns.count()
        valid_tickers = obs_counts[obs_counts >= min_periods].index.tolist()
        if len(valid_tickers) < 2:
            continue

        corr = window_returns[valid_tickers].corr(method=method)

        intra_vals = []
        inter_vals = []
        sector_intra = {s: [] for s in unique_sectors}

        for i, t1 in enumerate(valid_tickers):
            for j, t2 in enumerate(valid_tickers):
                if i >= j:
                    continue
                val = corr.loc[t1, t2]
                if not np.isfinite(val):
                    continue
                s1 = sectors.get(t1)
                s2 = sectors.get(t2)
                if s1 == s2:
                    intra_vals.append(val)
                    if s1 in sector_intra:
                        sector_intra[s1].append(val)
                else:
                    inter_vals.append(val)

        record = {
            "date": end_date,
            "intra_sector_avg": float(np.mean(intra_vals)) if intra_vals else np.nan,
            "inter_sector_avg": float(np.mean(inter_vals)) if inter_vals else np.nan,
        }

        # Per-sector intra-correlation (only if sector has >= 2 stocks)
        for s in unique_sectors:
            vals = sector_intra[s]
            record[f"intra_{s}"] = float(np.mean(vals)) if len(vals) >= 1 else np.nan

        records.append(record)

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    logger.info(
        "Rolling sector stats: window=%d, step=%d, %d points, %d sectors",
        window,
        step,
        len(df),
        len(unique_sectors),
    )
    return df


def compute_window_correlation(
    returns: pd.DataFrame,
    end_date: pd.Timestamp,
    window: int = 252,
    min_periods: Optional[int] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Correlation matrix for a specific window ending at end_date.

    Useful for point-in-time snapshots (e.g., slider in dashboard).
    """
    if min_periods is None:
        min_periods = max(30, int(window * 0.6))

    mask = returns.index <= end_date
    if mask.sum() < min_periods:
        return pd.DataFrame()

    available = returns.loc[mask]
    window_returns = available.iloc[-min(window, len(available)) :]

    valid_tickers = window_returns.columns[window_returns.count() >= min_periods]
    if len(valid_tickers) < 2:
        return pd.DataFrame()

    return window_returns[valid_tickers].corr(method=method)


def run_rolling_analysis(config: PipelineConfig) -> None:
    """Precompute rolling stats for configured window sizes."""
    logger.info("=== Rolling Correlation Analysis ===")
    DATA_RESULTS.mkdir(parents=True, exist_ok=True)

    returns = pd.read_parquet(DATA_PROCESSED / "log_returns.parquet")
    logger.info("Loaded log returns: %d days x %d tickers", *returns.shape)

    rolling_cfg = config.rolling
    windows = rolling_cfg.windows
    step = rolling_cfg.step
    method = rolling_cfg.method

    # Market-level stats for each window size
    for w in windows:
        min_p = max(30, int(w * rolling_cfg.min_periods_ratio))
        stats = compute_rolling_market_stats(
            returns, window=w, step=step, min_periods=min_p, method=method
        )
        path = DATA_RESULTS / f"rolling_market_stats_w{w}.parquet"
        stats.to_parquet(path)
        logger.info("Saved rolling market stats (window=%d): %s", w, path.name)

    # Sector-level stats for the largest window
    sector_map = dict(zip(config.universe["ticker"], config.universe["sector"]))
    largest_window = max(windows)
    sector_stats = compute_rolling_sector_stats(
        returns,
        sector_map=sector_map,
        window=largest_window,
        step=step,
        min_periods=max(30, int(largest_window * rolling_cfg.min_periods_ratio)),
        method=method,
    )
    sector_stats.to_parquet(DATA_RESULTS / "rolling_sector_stats.parquet")
    logger.info("Saved rolling sector stats")

    logger.info("Rolling analysis complete")
