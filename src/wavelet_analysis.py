"""Wavelet multi-scale correlation analysis.

Decomposes stock return series into frequency bands using DWT, then computes
correlation matrices and MSTs at each wavelet scale to reveal scale-dependent
market structure.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pywt
except ImportError:
    pywt = None

from src.config import PipelineConfig, PROJECT_ROOT
from src.analysis import compute_correlation_matrix, compute_distance_matrix
from src.clustering import build_mst, mst_to_edge_df, compute_mst_metrics

logger = logging.getLogger(__name__)


# Approximate trading-day interpretation of wavelet scales
SCALE_LABELS = {
    1: "2-4 day",
    2: "4-8 day",
    3: "8-16 day",
    4: "16-32 day",
    5: "32-64 day",
    6: "64-128 day",
    7: "128-256 day",
}


def wavelet_decompose(
    returns: pd.DataFrame,
    wavelet: str = "db4",
    max_level: Optional[int] = None,
) -> dict[int, pd.DataFrame]:
    """Decompose each stock's return series using DWT.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns (dates x tickers).
    wavelet : str
        Wavelet family (default: Daubechies-4).
    max_level : int or None
        Maximum decomposition level. If None, computed from data length.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping from scale level (1=finest) to DataFrame of detail coefficients
        (aligned to original dates, zero-padded where necessary).
    """
    if pywt is None:
        raise ImportError("PyWavelets (pywt) is required for wavelet analysis. Install with: pip install PyWavelets")

    tickers = returns.columns.tolist()
    T = len(returns)

    if max_level is None:
        max_level = min(pywt.dwt_max_level(T, wavelet), 7)

    logger.info(
        "Wavelet decomposition: wavelet=%s, max_level=%d, T=%d, N=%d",
        wavelet, max_level, T, len(tickers),
    )

    scale_data: dict[int, pd.DataFrame] = {}

    for level in range(1, max_level + 1):
        detail_coeffs = {}
        for ticker in tickers:
            series = returns[ticker].fillna(0.0).values.copy()
            coeffs = pywt.wavedec(series, wavelet, level=level)
            # Reconstruct detail at this level only
            # Zero out all coefficients except the detail at this level
            zeroed = [np.zeros_like(c) for c in coeffs]
            zeroed[-(level)] = coeffs[-(level)]  # detail coeffs are stored in reverse
            reconstructed = pywt.waverec(zeroed, wavelet)[:T]
            detail_coeffs[ticker] = reconstructed

        df = pd.DataFrame(detail_coeffs, index=returns.index)
        scale_data[level] = df
        label = SCALE_LABELS.get(level, f"scale {level}")
        logger.info("  Scale %d (%s): reconstructed %d x %d", level, label, *df.shape)

    return scale_data


def compute_wavelet_scale_mst(
    scale_returns: pd.DataFrame,
    min_periods: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute correlation, distance, and MST for a single wavelet scale.

    Returns
    -------
    corr_matrix : pd.DataFrame
    mst_edges : pd.DataFrame
    mst_metrics : pd.DataFrame
    """
    corr_matrix = compute_correlation_matrix(scale_returns, min_periods=min_periods)
    dist = compute_distance_matrix(corr_matrix)
    mst = build_mst(dist)
    edges = mst_to_edge_df(mst, corr_matrix)
    metrics = compute_mst_metrics(mst)
    return corr_matrix, edges, metrics


def run_wavelet_analysis(config: PipelineConfig) -> None:
    """Pipeline step: wavelet multi-scale correlation and MST analysis."""
    logger.info("=== Wavelet Multi-Scale Analysis ===")
    config.data_results.mkdir(parents=True, exist_ok=True)

    returns = pd.read_parquet(config.data_processed / "log_returns.parquet")
    logger.info("Loaded log returns: %d days x %d tickers", *returns.shape)

    sector_map = dict(zip(config.universe["ticker"], config.universe["sector"]))

    # Decompose
    scale_data = wavelet_decompose(returns, wavelet="db4")

    # Compute correlation + MST at each scale
    for level, scale_returns in scale_data.items():
        label = SCALE_LABELS.get(level, f"scale_{level}")
        logger.info("Computing correlation and MST at scale %d (%s)", level, label)

        corr_matrix, mst_edges, mst_metrics = compute_wavelet_scale_mst(scale_returns)

        corr_matrix.to_parquet(config.data_results / f"wavelet_corr_scale{level}.parquet")
        mst_edges.to_csv(config.data_results / f"wavelet_mst_edges_scale{level}.csv", index=False)

        mst_metrics["sector"] = mst_metrics["ticker"].map(sector_map)
        mst_metrics.to_csv(
            config.data_results / f"wavelet_mst_metrics_scale{level}.csv", index=False
        )

    # Save scale metadata
    import json
    meta = {
        "wavelet": "db4",
        "n_scales": len(scale_data),
        "scales": {
            level: SCALE_LABELS.get(level, f"scale {level}")
            for level in scale_data
        },
    }
    with open(config.data_results / "wavelet_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Wavelet analysis complete: %d scales", len(scale_data))
