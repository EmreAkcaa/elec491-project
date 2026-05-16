"""Transfer entropy: directed information flow between stock pairs.

Measures asymmetric, nonlinear information transfer using Shannon entropy.
Produces a directed network showing which stocks lead and which follow.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PipelineConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)



def _discretize(series: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """Discretize a continuous series into equal-frequency bins."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(series[~np.isnan(series)], percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    return np.digitize(series, bin_edges[1:-1])


def _shannon_entropy(x: np.ndarray) -> float:
    """Shannon entropy H(X) in nats."""
    _, counts = np.unique(x, return_counts=True)
    probs = counts / counts.sum()
    return -float(np.sum(probs * np.log(probs + 1e-12)))


def _joint_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """Joint entropy H(X, Y) in nats."""
    combined = np.column_stack([x, y])
    _, counts = np.unique(combined, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return -float(np.sum(probs * np.log(probs + 1e-12)))


def _conditional_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """Conditional entropy H(X|Y) = H(X,Y) - H(Y)."""
    return _joint_entropy(x, y) - _shannon_entropy(y)


def _triple_joint_entropy(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Joint entropy H(X, Y, Z)."""
    combined = np.column_stack([x, y, z])
    _, counts = np.unique(combined, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return -float(np.sum(probs * np.log(probs + 1e-12)))


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    n_bins: int = 3,
) -> float:
    """Compute transfer entropy TE(source -> target).

    TE(X->Y) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})

    This measures how much knowing the past of X reduces uncertainty about
    the future of Y, beyond what Y's own past already provides.

    Parameters
    ----------
    source : np.ndarray
        Source time series.
    target : np.ndarray
        Target time series.
    lag : int
        Time lag for conditioning.
    n_bins : int
        Number of bins for discretization.

    Returns
    -------
    float
        Transfer entropy in nats. Non-negative; higher = more information flow.
    """
    # Align and discretize
    y_t = _discretize(target[lag:], n_bins)
    y_lag = _discretize(target[:-lag], n_bins)
    x_lag = _discretize(source[:-lag], n_bins)

    # Remove any positions with NaN in original data
    mask = np.isfinite(target[lag:]) & np.isfinite(target[:-lag]) & np.isfinite(source[:-lag])
    y_t, y_lag, x_lag = y_t[mask], y_lag[mask], x_lag[mask]

    if len(y_t) < 30:
        return 0.0

    # TE = H(Y_t, Y_lag) - H(Y_t, Y_lag, X_lag) - H(Y_lag) + H(Y_lag, X_lag)
    te = (
        _joint_entropy(y_t, y_lag)
        - _triple_joint_entropy(y_t, y_lag, x_lag)
        - _shannon_entropy(y_lag)
        + _joint_entropy(y_lag, x_lag)
    )
    return max(0.0, te)  # TE is theoretically non-negative


def compute_transfer_entropy_matrix(
    returns: pd.DataFrame,
    lag: int = 1,
    n_bins: int = 3,
    significance_shuffles: int = 100,
    significance_level: float = 0.05,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise transfer entropy for all stock pairs.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns (dates x tickers).
    lag : int
        Time lag.
    n_bins : int
        Discretization bins.
    significance_shuffles : int
        Number of shuffle permutations for significance testing.
    significance_level : float
        p-value threshold.
    seed : int or None
        Seed for the shuffle RNG. If None, results are non-deterministic.

    Returns
    -------
    te_matrix : pd.DataFrame
        Asymmetric matrix where entry [i,j] = TE(i -> j).
    net_te_matrix : pd.DataFrame
        Net transfer entropy: net[i,j] = TE(i->j) - TE(j->i).
        Positive = i leads j.
    """
    tickers = returns.columns.tolist()
    N = len(tickers)
    te = np.zeros((N, N))
    significant = np.ones((N, N), dtype=bool)
    rng = np.random.default_rng(seed)

    total_pairs = N * (N - 1)
    completed = 0

    for i in range(N):
        x = returns.iloc[:, i].values
        for j in range(N):
            if i == j:
                continue
            y = returns.iloc[:, j].values
            te[i, j] = transfer_entropy(x, y, lag=lag, n_bins=n_bins)

            # Significance test: shuffle source and recompute
            if significance_shuffles > 0:
                null_dist = np.zeros(significance_shuffles)
                for s in range(significance_shuffles):
                    x_shuffled = rng.permutation(x)
                    null_dist[s] = transfer_entropy(x_shuffled, y, lag=lag, n_bins=n_bins)
                p_value = (null_dist >= te[i, j]).mean()
                significant[i, j] = p_value < significance_level

            completed += 1
            if completed % 500 == 0:
                logger.info("  TE computation: %d/%d pairs (%.0f%%)", completed, total_pairs, 100 * completed / total_pairs)

    # Zero out insignificant entries
    te_filtered = te * significant

    te_matrix = pd.DataFrame(te_filtered, index=tickers, columns=tickers)
    net_te = te_filtered - te_filtered.T
    net_te_matrix = pd.DataFrame(net_te, index=tickers, columns=tickers)

    n_sig = int(significant.sum()) - N  # exclude diagonal
    logger.info(
        "Transfer entropy: %d significant directed edges (of %d), p<%.2f",
        n_sig, total_pairs, significance_level,
    )

    return te_matrix, net_te_matrix


def extract_te_edges(
    te_matrix: pd.DataFrame,
    net_te_matrix: pd.DataFrame,
    min_te: float = 0.001,
) -> pd.DataFrame:
    """Extract directed edges from the TE matrix.

    Returns
    -------
    pd.DataFrame
        Columns: source, target, te_forward, te_backward, net_te, direction.
    """
    tickers = te_matrix.columns.tolist()
    rows = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            te_ij = te_matrix.iloc[i, j]  # i -> j
            te_ji = te_matrix.iloc[j, i]  # j -> i
            net = te_ij - te_ji
            if te_ij > min_te or te_ji > min_te:
                rows.append({
                    "source": tickers[i],
                    "target": tickers[j],
                    "te_forward": round(float(te_ij), 6),
                    "te_backward": round(float(te_ji), 6),
                    "net_te": round(float(net), 6),
                    "dominant_direction": f"{tickers[i]}->{tickers[j]}" if net > 0 else f"{tickers[j]}->{tickers[i]}",
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("net_te", key=abs, ascending=False).reset_index(drop=True)
    return df


def compute_node_roles(
    te_matrix: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Classify stocks as information sources or sinks.

    Parameters
    ----------
    te_matrix : pd.DataFrame
        TE matrix where [i,j] = TE(i->j).
    universe : pd.DataFrame
        Universe with ticker and sector columns.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, sector, te_out, te_in, net_te_flow, role.
    """
    sector_map = dict(zip(universe["ticker"], universe["sector"]))
    tickers = te_matrix.columns.tolist()

    rows = []
    for ticker in tickers:
        te_out = float(te_matrix.loc[ticker].sum())  # total info exported
        te_in = float(te_matrix[ticker].sum())  # total info received
        net = te_out - te_in
        rows.append({
            "ticker": ticker,
            "sector": sector_map.get(ticker, ""),
            "te_out": round(te_out, 6),
            "te_in": round(te_in, 6),
            "net_te_flow": round(net, 6),
            "role": "source" if net > 0 else "sink",
        })

    df = pd.DataFrame(rows).sort_values("net_te_flow", ascending=False).reset_index(drop=True)
    n_sources = (df["role"] == "source").sum()
    logger.info("Node roles: %d sources, %d sinks", n_sources, len(df) - n_sources)
    return df


def run_transfer_entropy(config: PipelineConfig) -> None:
    """Pipeline step: transfer entropy directed information flow network."""
    logger.info("=== Transfer Entropy Analysis ===")
    config.data_results.mkdir(parents=True, exist_ok=True)

    returns = pd.read_parquet(config.data_processed / "log_returns.parquet")
    logger.info("Loaded log returns: %d days x %d tickers", *returns.shape)

    te_cfg = config.transfer_entropy

    # Compute TE matrix
    te_matrix, net_te_matrix = compute_transfer_entropy_matrix(
        returns,
        lag=te_cfg.lag,
        n_bins=te_cfg.n_bins,
        significance_shuffles=te_cfg.significance_shuffles,
        significance_level=te_cfg.significance_level,
        seed=te_cfg.seed,
    )
    te_matrix.to_parquet(config.data_results / "transfer_entropy_matrix.parquet")
    net_te_matrix.to_parquet(config.data_results / "net_transfer_entropy_matrix.parquet")
    logger.info("Saved transfer entropy matrices")

    # Extract edges
    edges = extract_te_edges(te_matrix, net_te_matrix)
    edges.to_csv(config.data_results / "te_network_edges.csv", index=False)
    logger.info("Saved %d TE network edges", len(edges))

    # Node roles
    roles = compute_node_roles(te_matrix, config.universe)
    roles.to_csv(config.data_results / "te_node_roles.csv", index=False)
    logger.info("Saved TE node roles")

    logger.info("Transfer entropy analysis complete")
