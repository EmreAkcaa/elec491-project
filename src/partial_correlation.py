"""Graphical LASSO: sparse precision matrix and partial correlation network.

Uses L1-penalized maximum likelihood to estimate a sparse inverse covariance
(precision) matrix. Non-zero entries represent direct (conditional) dependencies,
filtering out indirect correlations mediated by third variables.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso

from src.config import PipelineConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RESULTS = PROJECT_ROOT / "data" / "results"


def fit_graphical_lasso(
    returns: pd.DataFrame,
    alpha: float | None = None,
    max_iter: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Fit Graphical LASSO to estimate sparse precision matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns (dates x tickers). NaN rows are dropped.
    alpha : float or None
        L1 regularization strength. If None, uses cross-validation to select.
    max_iter : int
        Maximum iterations for the solver.

    Returns
    -------
    precision : pd.DataFrame
        Sparse precision (inverse covariance) matrix.
    partial_corr : pd.DataFrame
        Partial correlation matrix derived from precision matrix.
    alpha_used : float
        The regularization parameter used.
    """
    tickers = returns.columns.tolist()

    # Drop rows with any NaN, then standardize
    clean = returns.dropna()
    if len(clean) < len(tickers):
        logger.warning(
            "Only %d clean observations for %d tickers; results may be unstable",
            len(clean), len(tickers),
        )

    if alpha is None:
        model = GraphicalLassoCV(max_iter=max_iter, cv=5)
    else:
        model = GraphicalLasso(alpha=alpha, max_iter=max_iter)

    model.fit(clean.values)
    alpha_used = model.alpha_ if hasattr(model, "alpha_") else alpha

    prec = model.precision_
    logger.info(
        "Graphical LASSO: alpha=%.4f, %d non-zero off-diagonal entries (of %d)",
        alpha_used,
        int((np.abs(prec) > 1e-10).sum() - len(tickers)),
        len(tickers) * (len(tickers) - 1),
    )

    precision_df = pd.DataFrame(prec, index=tickers, columns=tickers)

    # Convert precision to partial correlation
    # pcorr_ij = -P_ij / sqrt(P_ii * P_jj)
    d = np.sqrt(np.diag(prec))
    d[d == 0] = 1.0
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    partial_corr_df = pd.DataFrame(pcorr, index=tickers, columns=tickers)

    return precision_df, partial_corr_df, alpha_used


def extract_partial_corr_edges(
    partial_corr: pd.DataFrame,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """Extract non-zero edges from the partial correlation matrix.

    Parameters
    ----------
    partial_corr : pd.DataFrame
        Partial correlation matrix.
    threshold : float
        Minimum absolute partial correlation to include an edge.

    Returns
    -------
    pd.DataFrame
        Edge list with columns: source, target, partial_correlation, abs_partial_corr.
    """
    tickers = partial_corr.columns.tolist()
    rows = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            val = partial_corr.iloc[i, j]
            if abs(val) > threshold:
                rows.append({
                    "source": tickers[i],
                    "target": tickers[j],
                    "partial_correlation": round(float(val), 6),
                    "abs_partial_corr": round(abs(float(val)), 6),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("abs_partial_corr", ascending=False).reset_index(drop=True)
    logger.info("Extracted %d partial correlation edges (threshold=%.3f)", len(df), threshold)
    return df


def run_partial_correlation(config: PipelineConfig) -> None:
    """Pipeline step: Graphical LASSO partial correlation network."""
    logger.info("=== Graphical LASSO Partial Correlation ===")
    DATA_RESULTS.mkdir(parents=True, exist_ok=True)

    returns = pd.read_parquet(DATA_PROCESSED / "log_returns.parquet")
    logger.info("Loaded log returns: %d days x %d tickers", *returns.shape)

    # Fit model with cross-validated alpha
    precision, partial_corr, alpha_used = fit_graphical_lasso(returns)

    partial_corr.to_parquet(DATA_RESULTS / "partial_corr.parquet")
    logger.info("Saved partial correlation matrix (alpha=%.4f)", alpha_used)

    precision.to_parquet(DATA_RESULTS / "precision_matrix.parquet")
    logger.info("Saved precision matrix")

    # Extract edge list
    edges = extract_partial_corr_edges(partial_corr)
    edges.to_csv(DATA_RESULTS / "partial_corr_edges.csv", index=False)

    # Save metadata
    meta = {
        "alpha": float(alpha_used),
        "n_edges": len(edges),
        "n_tickers": len(partial_corr),
        "sparsity_pct": round(
            100 * (1 - len(edges) / (len(partial_corr) * (len(partial_corr) - 1) / 2)), 2
        ),
    }
    import json
    with open(DATA_RESULTS / "glasso_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Graphical LASSO complete: %d edges, %.1f%% sparsity",
        meta["n_edges"], meta["sparsity_pct"],
    )
