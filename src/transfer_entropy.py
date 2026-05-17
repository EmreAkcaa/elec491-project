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


def _circular_block_bootstrap(
    x: np.ndarray, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Single circular-block-bootstrap resample of `x`.

    Preserves within-block autocorrelation (Politis & Romano 1992) by
    sampling contiguous blocks of length `block_length` from a circularly
    extended copy of x and concatenating them. With `block_length=1` this
    degenerates to a plain i.i.d. permutation, which is exactly the broken
    surrogate the BH-FDR fix replaced.
    """
    n = len(x)
    if block_length <= 1:
        return rng.permutation(x)
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    x_ext = np.concatenate([x, x[:block_length]])
    blocks = [x_ext[s:s + block_length] for s in starts]
    return np.concatenate(blocks)[:n]


def _te_one_pair(
    i: int,
    j: int,
    x: np.ndarray,
    y: np.ndarray,
    lag: int,
    n_bins: int,
    n_shuffles: int,
    block_length: int,
    pair_seed: int,
) -> tuple[int, int, float, float]:
    """Compute TE(i -> j) plus its surrogate-null p-value.

    Returns (i, j, te, p_value). p_value = 1.0 when no shuffles are
    requested (caller can treat that as "no significance test run").
    """
    te_val = transfer_entropy(x, y, lag=lag, n_bins=n_bins)
    if n_shuffles <= 0:
        return i, j, te_val, 1.0
    rng = np.random.default_rng(pair_seed)
    null = np.empty(n_shuffles)
    for s in range(n_shuffles):
        x_surrogate = _circular_block_bootstrap(x, block_length, rng)
        null[s] = transfer_entropy(x_surrogate, y, lag=lag, n_bins=n_bins)
    # +1 smoothing keeps p-values bounded away from 0, which BH-FDR needs.
    p_value = (1 + (null >= te_val).sum()) / (1 + n_shuffles)
    return i, j, te_val, float(p_value)


def compute_transfer_entropy_matrix_full(
    returns: pd.DataFrame,
    lag: int = 1,
    n_bins: int = 3,
    significance_shuffles: int = 100,
    significance_level: float = 0.05,
    seed: int | None = None,
    n_jobs: int = -1,
    surrogate_block_length: int = 5,
    multiple_testing: str = "fdr_bh",
) -> dict[str, pd.DataFrame | int]:
    """Compute pairwise transfer entropy for all directed pairs (parallelised).

    Returns a dict with:
    - ``raw`` (DataFrame): TE values pre-significance, useful for ranking by
      magnitude when the FDR-corrected mask is sparse / empty.
    - ``pvals`` (DataFrame): per-pair surrogate-null p-values.
    - ``significant`` (DataFrame): boolean mask after `multiple_testing`.
    - ``filtered`` (DataFrame): ``raw * significant`` — the legacy contract.
    - ``net_filtered`` (DataFrame): ``filtered - filtered.T`` (positive =
      net info flow from row to column).
    - ``n_significant_fdr`` (int), ``n_significant_uncorrected`` (int),
      ``total_pairs`` (int): scalar summaries.

    Notes on the configuration knobs:
    - ``surrogate_block_length`` controls the circular-block-bootstrap null.
      The previous i.i.d. permutation (block_length=1) destroyed source
      autocorrelation and inflated significance.
    - ``multiple_testing`` defaults to Benjamini–Hochberg FDR control over
      the N*(N-1) directed pairs. Without it, ~5% of pairs always appear
      significant at alpha=0.05 by construction.

    For consumers that only need the legacy ``(filtered, net_filtered)``
    pair, see :func:`compute_transfer_entropy_matrix`.
    """
    from joblib import Parallel, delayed

    tickers = returns.columns.tolist()
    N = len(tickers)
    te = np.zeros((N, N))
    pvals = np.ones((N, N))

    cols = [returns.iloc[:, i].values for i in range(N)]
    base_seed = int(seed) if seed is not None else 0
    tasks = [(i, j) for i in range(N) for j in range(N) if i != j]
    total_pairs = len(tasks)

    logger.info(
        "TE: dispatching %d directed pairs across n_jobs=%s "
        "(N=%d tickers, shuffles=%d, block=%d, mt=%s)",
        total_pairs, n_jobs, N,
        significance_shuffles, surrogate_block_length, multiple_testing,
    )

    results = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
        delayed(_te_one_pair)(
            i, j,
            cols[i], cols[j],
            lag, n_bins,
            significance_shuffles, surrogate_block_length,
            base_seed * N * N + i * N + j,
        )
        for i, j in tasks
    )

    for i, j, te_val, p_value in results:
        te[i, j] = te_val
        pvals[i, j] = p_value

    significant = _apply_multiple_testing(
        pvals, tasks, multiple_testing, significance_level
    )

    te_filtered = te * significant
    n_sig = int(significant.sum()) - N
    n_sig_raw = int((pvals[~np.eye(N, dtype=bool)] < significance_level).sum())
    logger.info(
        "Transfer entropy: %d significant directed edges (of %d) at FDR≤%.2f "
        "via %s (vs %d uncorrected at p<%.2f)",
        n_sig, total_pairs, significance_level, multiple_testing,
        n_sig_raw, significance_level,
    )

    raw_df = pd.DataFrame(te, index=tickers, columns=tickers)
    pvals_df = pd.DataFrame(pvals, index=tickers, columns=tickers)
    sig_df = pd.DataFrame(significant, index=tickers, columns=tickers)
    filtered_df = pd.DataFrame(te_filtered, index=tickers, columns=tickers)
    net_filtered = pd.DataFrame(
        te_filtered - te_filtered.T, index=tickers, columns=tickers
    )

    return {
        "raw": raw_df,
        "pvals": pvals_df,
        "significant": sig_df,
        "filtered": filtered_df,
        "net_filtered": net_filtered,
        "n_significant_fdr": n_sig,
        "n_significant_uncorrected": n_sig_raw,
        "total_pairs": total_pairs,
    }


def compute_transfer_entropy_matrix(
    returns: pd.DataFrame,
    lag: int = 1,
    n_bins: int = 3,
    significance_shuffles: int = 100,
    significance_level: float = 0.05,
    seed: int | None = None,
    n_jobs: int = -1,
    surrogate_block_length: int = 5,
    multiple_testing: str = "fdr_bh",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backwards-compatible wrapper around :func:`compute_transfer_entropy_matrix_full`.

    Returns just ``(filtered, net_filtered)`` for callers that don't need
    raw values / p-values / the significance mask.
    """
    out = compute_transfer_entropy_matrix_full(
        returns,
        lag=lag,
        n_bins=n_bins,
        significance_shuffles=significance_shuffles,
        significance_level=significance_level,
        seed=seed,
        n_jobs=n_jobs,
        surrogate_block_length=surrogate_block_length,
        multiple_testing=multiple_testing,
    )
    return out["filtered"], out["net_filtered"]


def _apply_multiple_testing(
    pvals: np.ndarray,
    tasks: list[tuple[int, int]],
    method: str,
    alpha: float,
) -> np.ndarray:
    """Convert the raw N*N p-value matrix into a boolean significance mask
    with the requested family-wise / FDR correction applied across the
    off-diagonal pairs.

    The diagonal stays True (a node has no self-edge to test); insignificant
    off-diagonal entries become False and their TE values get zeroed by the
    caller.
    """
    N = pvals.shape[0]
    significant = np.zeros_like(pvals, dtype=bool)
    np.fill_diagonal(significant, True)

    if not tasks:
        return significant

    off_pvals = np.array([pvals[i, j] for i, j in tasks])

    if method == "none":
        reject = off_pvals < alpha
    elif method == "bonferroni":
        reject = off_pvals < (alpha / len(off_pvals))
    elif method == "fdr_bh":
        reject = _benjamini_hochberg(off_pvals, alpha)
    else:
        raise ValueError(
            f"Unknown multiple_testing method: {method!r}. "
            "Use 'fdr_bh', 'bonferroni', or 'none'."
        )

    for (i, j), keep in zip(tasks, reject):
        significant[i, j] = bool(keep)

    return significant


def _benjamini_hochberg(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini–Hochberg FDR control (Benjamini & Hochberg 1995).

    Returns a boolean reject array of the same shape as `pvals`. Controls
    the expected proportion of false discoveries among rejected hypotheses
    at level `alpha` under independence or positive dependence.

    Implementation is the textbook step-up procedure: sort, find the
    largest k where p_(k) ≤ (k / m) * alpha, reject ranks 1..k.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = np.arange(1, m + 1) * (alpha / m)
    below = ranked <= thresholds
    if not below.any():
        return np.zeros_like(pvals, dtype=bool)
    k = np.where(below)[0].max() + 1  # number of rejections (1-indexed)
    reject = np.zeros_like(pvals, dtype=bool)
    reject[order[:k]] = True
    return reject


TE_EDGE_COLUMNS = [
    "source",
    "target",
    "te_forward",
    "te_backward",
    "net_te",
    "dominant_direction",
]


def extract_te_edges(
    te_matrix: pd.DataFrame,
    net_te_matrix: pd.DataFrame,
    min_te: float = 0.001,
) -> pd.DataFrame:
    """Extract directed edges from the TE matrix.

    Returns
    -------
    pd.DataFrame
        Columns: source, target, te_forward, te_backward, net_te,
        dominant_direction. Returns an empty DataFrame with the correct
        columns (rather than a column-less one) when no pair clears
        `min_te`, so the persisted CSV stays valid for downstream loaders.
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

    if not rows:
        return pd.DataFrame(columns=TE_EDGE_COLUMNS)

    df = pd.DataFrame(rows, columns=TE_EDGE_COLUMNS)
    return df.sort_values("net_te", key=abs, ascending=False).reset_index(drop=True)


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

    # Compute TE matrix (full output: raw, pvals, mask, FDR-filtered, net).
    results = compute_transfer_entropy_matrix_full(
        returns,
        lag=te_cfg.lag,
        n_bins=te_cfg.n_bins,
        significance_shuffles=te_cfg.significance_shuffles,
        significance_level=te_cfg.significance_level,
        seed=te_cfg.seed,
        surrogate_block_length=te_cfg.surrogate_block_length,
        multiple_testing=te_cfg.multiple_testing,
    )

    te_matrix = results["filtered"]
    net_te_matrix = results["net_filtered"]

    # Legacy filenames keep downstream consumers working unchanged.
    te_matrix.to_parquet(config.data_results / "transfer_entropy_matrix.parquet")
    net_te_matrix.to_parquet(
        config.data_results / "net_transfer_entropy_matrix.parquet"
    )

    # New artifacts: raw TE magnitudes + per-pair p-values + significance mask.
    # The dashboard uses these to rank pairs by magnitude even when no pair
    # survives BH-FDR at the configured shuffle resolution, and to surface
    # the "uncorrected vs FDR" comparison as an honest methodological finding.
    results["raw"].to_parquet(
        config.data_results / "transfer_entropy_raw.parquet"
    )
    results["pvals"].to_parquet(
        config.data_results / "transfer_entropy_pvalues.parquet"
    )
    results["significant"].to_parquet(
        config.data_results / "transfer_entropy_significance.parquet"
    )

    import json
    with open(config.data_results / "transfer_entropy_summary.json", "w") as f:
        json.dump(
            {
                "n_significant_fdr": results["n_significant_fdr"],
                "n_significant_uncorrected": results["n_significant_uncorrected"],
                "total_pairs": results["total_pairs"],
                "significance_level": te_cfg.significance_level,
                "multiple_testing": te_cfg.multiple_testing,
                "surrogate_block_length": te_cfg.surrogate_block_length,
                "significance_shuffles": te_cfg.significance_shuffles,
            },
            f,
            indent=2,
        )

    logger.info("Saved transfer entropy matrices (legacy filtered + raw + pvals + mask + summary)")

    # Extract edges
    edges = extract_te_edges(te_matrix, net_te_matrix)
    edges.to_csv(config.data_results / "te_network_edges.csv", index=False)
    logger.info("Saved %d TE network edges", len(edges))

    # Node roles
    roles = compute_node_roles(te_matrix, config.universe)
    roles.to_csv(config.data_results / "te_node_roles.csv", index=False)
    logger.info("Saved TE node roles")

    logger.info("Transfer entropy analysis complete")
