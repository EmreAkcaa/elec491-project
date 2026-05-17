"""RMT-based correlation matrix denoising using Marchenko-Pastur distribution.

Separates signal eigenvalues from noise eigenvalues in the correlation matrix,
then reconstructs a denoised correlation matrix using only the signal components.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import PipelineConfig, PROJECT_ROOT
from src.analysis import compute_distance_matrix
from src.clustering import build_mst, mst_to_edge_df, compute_mst_metrics

logger = logging.getLogger(__name__)



def marchenko_pastur_bounds(T: int, N: int, sigma2: float = 1.0) -> tuple[float, float]:
    """Compute Marchenko-Pastur theoretical eigenvalue bounds.

    Parameters
    ----------
    T : int
        Number of observations (trading days).
    N : int
        Number of variables (stocks).
    sigma2 : float
        Variance of noise (1.0 for correlation matrix).

    Returns
    -------
    lambda_min, lambda_max : float
        Lower and upper bounds of the MP distribution.
    """
    q = T / N
    lambda_max = sigma2 * (1 + 1 / q + 2 * np.sqrt(1 / q))
    lambda_min = sigma2 * (1 + 1 / q - 2 * np.sqrt(1 / q))
    return max(0.0, lambda_min), lambda_max


def denoise_correlation(
    corr: pd.DataFrame,
    T: int,
    method: str = "constant",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Denoise a correlation matrix using Random Matrix Theory.

    Eigenvalues within the Marchenko-Pastur bounds are classified as noise.
    Signal eigenvalues (above the upper MP bound) are preserved.

    Parameters
    ----------
    corr : pd.DataFrame
        Raw Pearson correlation matrix (N x N).
    T : int
        Number of observations used to estimate the correlation matrix.
    method : str
        How to handle noise eigenvalues:
        - 'constant': replace noise eigenvalues with their average
        - 'zero': set noise eigenvalues to zero

    Returns
    -------
    denoised_corr : pd.DataFrame
        Denoised correlation matrix.
    spectrum : pd.DataFrame
        Eigenvalue spectrum with columns: eigenvalue, is_signal, mp_upper, mp_lower.
    """
    tickers = corr.columns.tolist()
    N = len(tickers)

    # Clean NaN: fill with 0 correlation (distance = sqrt(2))
    corr_clean = corr.fillna(0.0).values
    corr_clean = (corr_clean + corr_clean.T) / 2
    np.fill_diagonal(corr_clean, 1.0)

    # Eigenvalue decomposition (symmetric matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_clean)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # MP bounds
    mp_min, mp_max = marchenko_pastur_bounds(T, N)
    is_signal = eigenvalues > mp_max

    n_signal = int(is_signal.sum())
    n_noise = N - n_signal
    logger.info(
        "RMT: %d signal eigenvalues (above MP upper=%.3f), %d noise eigenvalues",
        n_signal, mp_max, n_noise,
    )

    # Replace noise eigenvalues
    denoised_eigenvalues = eigenvalues.copy()
    if method == "constant" and n_noise > 0:
        avg_noise = eigenvalues[~is_signal].mean()
        denoised_eigenvalues[~is_signal] = avg_noise
    elif method == "zero":
        denoised_eigenvalues[~is_signal] = 0.0

    # Reconstruct correlation matrix
    denoised = eigenvectors @ np.diag(denoised_eigenvalues) @ eigenvectors.T

    # Force valid correlation matrix: diagonal = 1, clip to [-1, 1]
    d = np.sqrt(np.diag(denoised))
    d[d == 0] = 1.0
    denoised = denoised / np.outer(d, d)
    np.fill_diagonal(denoised, 1.0)
    denoised = np.clip(denoised, -1.0, 1.0)

    denoised_corr = pd.DataFrame(denoised, index=tickers, columns=tickers)

    # Build spectrum dataframe
    spectrum = pd.DataFrame({
        "eigenvalue": eigenvalues,
        "is_signal": is_signal,
        "mp_upper": mp_max,
        "mp_lower": mp_min,
        "explained_variance_pct": 100 * eigenvalues / eigenvalues.sum(),
    })

    return denoised_corr, spectrum


def run_rmt_denoising(config: PipelineConfig) -> None:
    """Pipeline step: RMT denoising of correlation matrix."""
    logger.info("=== RMT Correlation Denoising ===")
    config.data_results.mkdir(parents=True, exist_ok=True)

    corr = pd.read_parquet(config.data_results / "pearson_corr.parquet")
    returns = pd.read_parquet(config.data_processed / "log_returns.parquet")
    T = len(returns)

    # Denoise
    denoised_corr, spectrum = denoise_correlation(corr, T, method="constant")
    denoised_corr.to_parquet(config.data_results / "denoised_corr.parquet")
    spectrum.to_csv(config.data_results / "eigenvalue_spectrum.csv", index=False)
    logger.info("Saved denoised correlation matrix and eigenvalue spectrum")

    # Denoised distance matrix and MST
    denoised_dist = compute_distance_matrix(denoised_corr)
    mst = build_mst(denoised_dist)
    edges = mst_to_edge_df(mst, denoised_corr)
    edges.to_csv(config.data_results / "denoised_mst_edges.csv", index=False)

    metrics = compute_mst_metrics(mst)
    sector_map = dict(zip(config.universe["ticker"], config.universe["sector"]))
    metrics["sector"] = metrics["ticker"].map(sector_map)
    metrics.to_csv(config.data_results / "denoised_mst_node_metrics.csv", index=False)
    logger.info("Saved denoised MST edges and node metrics")

    logger.info("RMT denoising complete")
