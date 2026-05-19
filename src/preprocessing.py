"""Preprocessing: coverage filter, log returns, anomaly detection."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)


def load_raw_prices(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw parquet and return (adj_close, raw_close) DataFrames."""
    raw_path = config.data_raw / "prices_raw.parquet"
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
    long = returns.stack().rename("return_value").reset_index()
    long.columns = ["date", "ticker", "return_value"]
    flagged = long[long["return_value"].abs() > threshold].copy()
    flagged = flagged.sort_values("return_value", key=abs, ascending=False)

    logger.info("Flagged %d anomalous return observations (|r| > %.2f)", len(flagged), threshold)
    return flagged.reset_index(drop=True)


def _apply_split_back_adjust(
    adj_close: pd.DataFrame,
    ticker: str,
    date_str: str,
    ratio: float,
) -> tuple[bool, dict]:
    """Back-adjust pre-event prices for a single (ticker, date, ratio) entry.

    Sign convention:
      ratio > 0  →  missed-split case (yfinance failed to back-adjust).
                    Pre-event prices were too HIGH; multiply by 1/ratio to
                    lower them. E.g., CCOLA 2024-08-01 ratio=10.81 →
                    pre-event prices /= 10.81.
      ratio < 0  →  over-adjusted case (yfinance applied a phantom forward
                    split). Pre-event prices were too LOW; multiply by
                    |ratio| to raise them. E.g., HEKTS 2021-04-30
                    ratio=-1.45 → pre-event prices *= 1.45.

    Returns (applied: bool, audit_row: dict). On success, mutates `adj_close`
    in place via direct .loc assignment (caller already passed a copy).
    """
    audit = {
        "ticker": ticker,
        "date": date_str,
        "ratio": ratio,
        "applied": False,
        "pre_price": float("nan"),
        "post_price": float("nan"),
        "reason": "",
    }
    if ticker not in adj_close.columns:
        audit["reason"] = "ticker_not_in_panel"
        return False, audit
    ts = pd.Timestamp(date_str)
    if ts not in adj_close.index:
        # Snap to next trading day if the literal date isn't an index entry
        # (e.g., the user wrote a Saturday by accident).
        nxt = adj_close.index[adj_close.index >= ts]
        if len(nxt) == 0:
            audit["reason"] = "date_after_panel_end"
            return False, audit
        ts = nxt[0]
        audit["date"] = ts.strftime("%Y-%m-%d")
    pos = adj_close.index.get_loc(ts)
    if pos == 0:
        audit["reason"] = "event_on_first_day"
        return False, audit
    idx_prior = adj_close.index[pos - 1]
    audit["pre_price"] = float(adj_close.loc[idx_prior, ticker])
    audit["post_price"] = float(adj_close.loc[ts, ticker])

    if ratio > 0:
        adj_close.loc[:idx_prior, ticker] = adj_close.loc[:idx_prior, ticker] / ratio
    elif ratio < 0:
        adj_close.loc[:idx_prior, ticker] = adj_close.loc[:idx_prior, ticker] * abs(ratio)
    else:
        audit["reason"] = "zero_ratio"
        return False, audit

    audit["applied"] = True
    audit["reason"] = "ok"
    return True, audit


def run_preprocessing(config: PipelineConfig) -> None:
    """Full preprocessing pipeline step."""
    logger.info("=== Preprocessing ===")
    config.data_processed.mkdir(parents=True, exist_ok=True)

    adj_close, raw_close = load_raw_prices(config)

    # Coverage filter
    filtered_adj, filtered_raw, coverage = filter_by_coverage(
        adj_close, raw_close, threshold=config.preprocessing.min_coverage_pct
    )
    # `filter_by_coverage` returns a view/slice; promote to a true copy
    # so `_apply_split_back_adjust`'s in-place .loc assignments don't
    # trigger pandas' SettingWithCopyWarning.
    filtered_adj = filtered_adj.copy()

    # Apply manual anomaly nulls (mask known unhandled corporate actions that
    # yfinance Adj-Close failed to back-adjust). The schema supports two
    # shapes per entry:
    #
    #   2-tuple [ticker, "YYYY-MM-DD"]:
    #       null only the log_returns cell. Leaves the cliff in adj_close.
    #       (Legacy behavior — `compute_spread` and other consumers that
    #        read np.log(adj_close) directly still see the discontinuity.)
    #
    #   3-tuple [ticker, "YYYY-MM-DD", ratio]:
    #       ALSO back-adjust adj_close BEFORE log returns are computed.
    #       Sign convention:
    #         ratio > 0   missed-split case   (pre-event /= ratio)
    #         ratio < 0   over-adjusted case  (pre-event *= |ratio|)
    #       The log_returns cell is still nulled defensively in case the
    #       back-adjustment leaves a tiny residual that crosses the
    #       anomaly threshold.
    #
    # Runs BEFORE compute_log_returns so downstream consumers of
    # `adj_close.parquet` (pair_dislocation.compute_spread,
    # snn_signals.build_input_features, dashboard Pair Analysis Spread tab)
    # see a continuous price series.
    audit_rows: list[dict] = []
    if config.preprocessing.manual_anomaly_nulls:
        # Stage 1: back-adjust adj_close for any 3-tuple entry.
        adjusted_3tuples = 0
        for entry in config.preprocessing.manual_anomaly_nulls:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 3):
                continue
            try:
                ticker, date_str, ratio = entry
                ok, audit = _apply_split_back_adjust(
                    filtered_adj, ticker, str(date_str), float(ratio),
                )
                audit_rows.append(audit)
                if ok:
                    adjusted_3tuples += 1
                else:
                    logger.warning(
                        "manual_anomaly_nulls back-adjust failed (%s, %s, ratio=%s): %s",
                        ticker, date_str, ratio, audit["reason"],
                    )
            except Exception as exc:
                logger.warning(
                    "manual_anomaly_nulls back-adjust invalid entry %r: %s",
                    entry, exc,
                )
        if adjusted_3tuples:
            logger.info(
                "Back-adjusted adj_close for %d corporate-action event(s) "
                "(3-tuple manual_anomaly_nulls entries)", adjusted_3tuples,
            )

    # Log returns — now computed off the back-adjusted adj_close so
    # 3-tuple events produce a clean log return for the event date.
    log_returns = compute_log_returns(filtered_adj)

    # Stage 2: null the log_returns cell for every manual_anomaly_nulls
    # entry (defense-in-depth — back-adjusted entries should already produce
    # a clean small log return at the event date, but the null is a
    # belt-and-suspenders guarantee against floating-point residuals).
    if config.preprocessing.manual_anomaly_nulls:
        applied = 0
        skipped = 0
        for entry in config.preprocessing.manual_anomaly_nulls:
            try:
                if isinstance(entry, (list, tuple)) and len(entry) == 3:
                    ticker, date_str, _ratio = entry
                else:
                    ticker, date_str = entry
                ts = pd.Timestamp(date_str)
                if ticker in log_returns.columns and ts in log_returns.index:
                    log_returns.loc[ts, ticker] = np.nan
                    applied += 1
                else:
                    logger.warning(
                        "manual_anomaly_nulls: skipped (%s, %s) — not in panel",
                        ticker, date_str,
                    )
                    skipped += 1
            except Exception as exc:
                logger.warning(
                    "manual_anomaly_nulls: invalid entry %r: %s", entry, exc,
                )
                skipped += 1
        logger.info(
            "Applied %d manual anomaly nulls (%d skipped)", applied, skipped,
        )

    # Audit log for 3-tuple entries. One row per attempt with applied/reason.
    if audit_rows:
        audit_df = pd.DataFrame(audit_rows)
        audit_path = config.data_processed / "applied_split_adjustments.csv"
        audit_df.to_csv(audit_path, index=False)
        logger.info(
            "Wrote %d-row audit log → %s", len(audit_df), audit_path.name,
        )

    # Anomaly detection
    anomalies = flag_anomalies(
        log_returns, threshold=config.preprocessing.anomaly_return_threshold
    )

    # Save artifacts
    filtered_adj.to_parquet(config.data_processed / "adj_close.parquet")
    logger.info("Saved filtered adj close")

    if not filtered_raw.empty:
        filtered_raw.to_parquet(config.data_processed / "raw_close.parquet")
        logger.info("Saved filtered raw close")

    log_returns.to_parquet(config.data_processed / "log_returns.parquet")
    logger.info("Saved log returns")

    coverage.to_csv(config.data_processed / "coverage_report.csv", index=False)
    logger.info("Saved coverage report")

    anomalies.to_csv(config.data_processed / "anomalies.csv", index=False)
    logger.info("Saved anomalies")

    logger.info("Preprocessing complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.config import load_config

    cfg = load_config()
    run_preprocessing(cfg)
