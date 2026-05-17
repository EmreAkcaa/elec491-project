"""Pair dislocation analysis: spread, Z-score, mean-reversion signals."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PipelineConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)



def compute_hedge_ratio(
    log_prices_a: pd.Series,
    log_prices_b: pd.Series,
) -> tuple[float, float]:
    """OLS hedge ratio: log(Pb) = beta * log(Pa) + intercept.

    Parameters
    ----------
    log_prices_a : pd.Series
        Log prices of ticker A (independent variable).
    log_prices_b : pd.Series
        Log prices of ticker B (dependent variable).

    Returns
    -------
    beta : float
        Hedge ratio (slope).
    intercept : float
        Regression intercept.
    """
    mask = log_prices_a.notna() & log_prices_b.notna()
    xa = log_prices_a[mask].values
    xb = log_prices_b[mask].values
    if len(xa) < 2:
        return np.nan, np.nan
    beta, intercept = np.polyfit(xa, xb, 1)
    return float(beta), float(intercept)


def compute_spread(
    adj_close: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    lookback: Optional[int] = None,
) -> tuple[pd.Series, float, float]:
    """Compute log-price spread: spread = log(Pb) - beta * log(Pa) - intercept.

    Parameters
    ----------
    adj_close : pd.DataFrame
        Adjusted close prices (dates x tickers).
    ticker_a, ticker_b : str
        Ticker symbols.
    lookback : int or None
        If given, OLS is fit on the last *lookback* days only.

    Returns
    -------
    spread : pd.Series
        Spread time series indexed by date.
    beta : float
        Hedge ratio used.
    intercept : float
        OLS intercept used.
    """
    pa = adj_close[ticker_a].dropna()
    pb = adj_close[ticker_b].dropna()
    common = pa.index.intersection(pb.index)
    pa, pb = pa.loc[common], pb.loc[common]

    log_a = np.log(pa)
    log_b = np.log(pb)

    if lookback is not None and len(common) > lookback:
        fit_a = log_a.iloc[-lookback:]
        fit_b = log_b.iloc[-lookback:]
    else:
        fit_a, fit_b = log_a, log_b

    beta, intercept = compute_hedge_ratio(fit_a, fit_b)
    spread = log_b - beta * log_a - intercept
    spread.name = f"{ticker_b}_vs_{ticker_a}_spread"
    return spread, beta, intercept


def compute_zscore(
    spread: pd.Series,
    window: int = 60,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Rolling Z-score of the spread.

    Z(t) = (spread(t) - rolling_mean(t)) / rolling_std(t)

    Parameters
    ----------
    spread : pd.Series
        Spread time series.
    window : int
        Rolling window size.
    min_periods : int or None
        Minimum observations. Defaults to max(10, window // 2).

    Returns
    -------
    pd.Series
        Rolling Z-score indexed by date.
    """
    if min_periods is None:
        min_periods = max(10, window // 2)
    roll_mean = spread.rolling(window, min_periods=min_periods).mean()
    roll_std = spread.rolling(window, min_periods=min_periods).std()
    zscore = (spread - roll_mean) / roll_std
    zscore.name = "zscore"
    return zscore


def compute_half_life(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life via AR(1) regression.

    Regresses delta_spread(t) on spread(t-1):
        delta_spread = phi * spread_lag + alpha
    Half-life = -ln(2) / ln(1 + phi)

    Returns np.inf if the spread is not mean-reverting (phi >= 0).
    """
    s = spread.dropna()
    if len(s) < 3:
        return np.inf
    spread_lag = s.iloc[:-1].values
    delta = s.diff().iloc[1:].values
    mask = np.isfinite(spread_lag) & np.isfinite(delta)
    spread_lag, delta = spread_lag[mask], delta[mask]
    if len(spread_lag) < 2:
        return np.inf
    phi, _ = np.polyfit(spread_lag, delta, 1)
    if phi >= 0:
        return np.inf
    return float(-np.log(2) / np.log(1 + phi))


def detect_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> pd.DataFrame:
    """Detect entry/exit dislocation signals from Z-score.

    Signal logic (state machine):
    - LONG_ENTRY: Z drops below -entry_threshold
    - SHORT_ENTRY: Z rises above +entry_threshold
    - LONG_EXIT / SHORT_EXIT: Z reverts within +/-exit_threshold

    Returns
    -------
    pd.DataFrame
        Columns: date, signal, zscore_value.
    """
    signals: list[dict] = []
    state = "flat"  # flat | long | short

    for date, z in zscore.dropna().items():
        if state == "flat":
            if z <= -entry_threshold:
                state = "long"
                signals.append(
                    {"date": date, "signal": "long_entry", "zscore_value": z}
                )
            elif z >= entry_threshold:
                state = "short"
                signals.append(
                    {"date": date, "signal": "short_entry", "zscore_value": z}
                )
        elif state == "long":
            if abs(z) <= exit_threshold:
                state = "flat"
                signals.append(
                    {"date": date, "signal": "long_exit", "zscore_value": z}
                )
            elif z >= entry_threshold:
                state = "short"
                signals.append(
                    {"date": date, "signal": "short_entry", "zscore_value": z}
                )
        elif state == "short":
            if abs(z) <= exit_threshold:
                state = "flat"
                signals.append(
                    {"date": date, "signal": "short_exit", "zscore_value": z}
                )
            elif z <= -entry_threshold:
                state = "long"
                signals.append(
                    {"date": date, "signal": "long_entry", "zscore_value": z}
                )

    return pd.DataFrame(signals)


def rank_candidate_pairs(
    adj_close: pd.DataFrame,
    corr: pd.DataFrame,
    universe: pd.DataFrame,
    top_n: int = 20,
    min_correlation: float = 0.5,
    zscore_window: int = 60,
    lookback: Optional[int] = 252,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    min_half_life: int = 5,
    max_half_life: int = 252,
) -> pd.DataFrame:
    """Screen and rank candidate pairs for dislocation analysis.

    Returns
    -------
    pd.DataFrame
        Ranked candidate pairs with columns: ticker_a, ticker_b,
        sector_a, sector_b, correlation, beta, half_life,
        spread_std, n_signals, current_zscore, rank_score.
    """
    sector_map = dict(zip(universe["ticker"], universe["sector"]))
    tickers = [t for t in corr.columns if t in adj_close.columns]

    # Extract upper-triangle pairs above min correlation
    pairs: list[tuple[str, str, float]] = []
    for i, ta in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            tb = tickers[j]
            c = corr.loc[ta, tb]
            if np.isfinite(c) and c >= min_correlation:
                pairs.append((ta, tb, c))

    logger.info(
        "Screening %d pairs with correlation >= %.2f", len(pairs), min_correlation
    )

    rows: list[dict] = []
    for ta, tb, c in pairs:
        spread, beta, intercept = compute_spread(adj_close, ta, tb, lookback)
        if spread.dropna().empty:
            continue
        hl = compute_half_life(spread)
        if hl < min_half_life or hl > max_half_life:
            continue
        zs = compute_zscore(spread, window=zscore_window)
        signals_df = detect_signals(zs, entry_zscore, exit_zscore)
        current_z = zs.dropna().iloc[-1] if not zs.dropna().empty else 0.0

        rows.append(
            {
                "ticker_a": ta,
                "ticker_b": tb,
                "sector_a": sector_map.get(ta, ""),
                "sector_b": sector_map.get(tb, ""),
                "correlation": round(c, 4),
                "beta": round(beta, 4),
                "half_life": round(hl, 1),
                "spread_std": round(float(spread.std()), 6),
                "n_signals": len(signals_df),
                "current_zscore": round(float(current_z), 4),
            }
        )

    if not rows:
        logger.warning("No candidate pairs found matching criteria.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Composite rank score: higher is better
    # Normalize each component to [0, 1] then weight
    def _norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.0, index=s.index)

    df["rank_score"] = (
        0.30 * _norm(df["correlation"])
        + 0.25 * _norm(1.0 / df["half_life"])
        + 0.20 * _norm(df["spread_std"])
        + 0.15 * _norm(df["current_zscore"].abs())
        + 0.10 * _norm(df["n_signals"].astype(float))
    )
    df["rank_score"] = df["rank_score"].round(4)
    df = df.sort_values("rank_score", ascending=False).head(top_n).reset_index(drop=True)

    logger.info("Ranked %d candidate pairs (top %d returned).", len(rows), len(df))
    return df


def run_pair_dislocation(config: PipelineConfig) -> None:
    """Pipeline Step 7: pair dislocation analysis."""
    logger.info("Step 7 — Pair dislocation analysis")

    adj_close = pd.read_parquet(config.data_processed / "adj_close.parquet")
    corr = pd.read_parquet(config.data_results / "pearson_corr.parquet")

    dc = config.dislocation
    candidates = rank_candidate_pairs(
        adj_close=adj_close,
        corr=corr,
        universe=config.universe,
        top_n=dc.top_n_candidates,
        min_correlation=dc.min_correlation,
        zscore_window=dc.zscore_window,
        lookback=dc.lookback_window,
        entry_zscore=dc.entry_zscore,
        exit_zscore=dc.exit_zscore,
        min_half_life=dc.min_half_life,
        max_half_life=dc.max_half_life,
    )

    if not candidates.empty:
        candidates.to_csv(config.data_results / "dislocation_candidates.csv", index=False)
        candidates.to_parquet(config.data_results / "dislocation_candidates.parquet", index=False)
        logger.info(
            "Saved %d dislocation candidates to data/results/", len(candidates)
        )
    else:
        logger.warning("No dislocation candidates to save.")
