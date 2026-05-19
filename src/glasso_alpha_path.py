"""GLASSO α-path: precompute sparsity snapshots across a log-spaced α grid.

Powers the interactive precision-sparsity timeline slider in the methods lab.
For each α in a log-spaced grid we fit GraphicalLasso once, threshold |Θ_ij|
to a binary sparsity pattern, and deduplicate — keeping only the smallest α
at which each unique pattern first appears. Slider in the dashboard then
scrubs through these snapshots in constant time (no recompute).

Gated to BIST only. S&P-500 has ~485 tickers vs BIST ~73, which would push
this stage from ~3-5 min to ~30-60 min — not worth the pipeline time today.

Output artifact: `data/<market>/results/glasso_alpha_path.npz` with arrays
    alphas:       (K,)       α thresholds, one per unique sparsity pattern
    n_edges:      (K,)       direct edge count at each snapshot
    sparsity_pct: (K,)       sparsity percentage at each snapshot
    patterns:     (K, N, N)  uint8 sparsity matrices (1 = direct edge)
    tickers:      (N,)       ticker symbols (object dtype)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


# |Θ_ij| above this floor counts as a direct conditional dependency. Mirrors
# the dashboard's precision sparsity heatmap threshold so the snapshots line
# up with what `render_glasso` already renders.
PRECISION_FLOOR = 1e-3


def _fit_pattern(
    returns: pd.DataFrame, alpha: float, max_iter: int = 200,
) -> tuple[np.ndarray | None, int]:
    """Fit GLASSO at a fixed α and return (binary_pattern, n_edges).

    `pattern` is a uint8 N×N matrix with 1 where |Θ_ij| > PRECISION_FLOOR
    (and 0 on the diagonal). `n_edges` counts unique pairs (upper-triangle).
    Returns (None, 0) when the solver fails — caller decides whether to skip.
    """
    try:
        model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
        model.fit(returns.values)
    except Exception as exc:  # pragma: no cover - sklearn convergence failures
        logger.warning("GLASSO fit failed at α=%.6f: %s", alpha, exc)
        return None, 0

    prec = np.abs(model.precision_)
    np.fill_diagonal(prec, 0.0)
    pattern = (prec > PRECISION_FLOOR).astype(np.uint8)
    n_edges = int(pattern.sum() // 2)  # symmetric matrix → halve
    return pattern, n_edges


def run_glasso_alpha_path(
    config: PipelineConfig,
    n_grid: int = 80,
    alpha_lo: float = 1e-5,
    alpha_hi: float = 5.0,
    max_iter: int = 200,
) -> None:
    """Sweep GLASSO over a log-spaced α grid; save unique sparsity snapshots.

    BIST-only — other markets exit silently with a log line. The dashboard
    surfaces a "run the pipeline" hint when the artifact is missing.
    """
    if config.market.market_id.lower() != "bist":
        logger.info(
            "GLASSO α-path skipped (BIST-only, current market: %s)",
            config.market.market_id,
        )
        return

    logger.info("=== GLASSO α-path sweep ===")
    config.data_results.mkdir(parents=True, exist_ok=True)

    returns_path = config.data_processed / "log_returns.parquet"
    returns = pd.read_parquet(returns_path).dropna()
    tickers = returns.columns.tolist()
    n_tickers = len(tickers)
    total_offdiag = n_tickers * (n_tickers - 1) / 2.0
    logger.info(
        "Sweeping %d α values from %.6f to %.6f over %d tickers (%d clean obs)",
        n_grid, alpha_lo, alpha_hi, n_tickers, len(returns),
    )

    alphas_grid = np.logspace(np.log10(alpha_lo), np.log10(alpha_hi), n_grid)

    # Dedupe by sparsity-pattern hash. Iterate in ascending α; the first time
    # we encounter each pattern records the α threshold at which the network
    # first reduced to that pattern.
    snapshots: list[tuple[float, np.ndarray, int]] = []
    seen_hashes: set[int] = set()

    for alpha in alphas_grid:
        pattern, n_edges = _fit_pattern(returns, float(alpha), max_iter=max_iter)
        if pattern is None:
            continue
        h = hash(pattern.tobytes())
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        snapshots.append((float(alpha), pattern, n_edges))
        sparsity_pct = 100.0 * (1.0 - n_edges / total_offdiag)
        logger.info(
            "  α=%.6f → %d edges, %.2f%% sparsity (new pattern #%d)",
            alpha, n_edges, sparsity_pct, len(snapshots),
        )

    if not snapshots:
        logger.warning("No successful GLASSO fits in α-path; nothing to save")
        return

    snapshots.sort(key=lambda t: t[0])
    alphas_arr = np.array([s[0] for s in snapshots], dtype=np.float64)
    patterns_arr = np.stack([s[1] for s in snapshots], axis=0)
    n_edges_arr = np.array([s[2] for s in snapshots], dtype=np.int32)
    sparsity_pct_arr = 100.0 * (1.0 - n_edges_arr / total_offdiag)

    out_path = config.data_results / "glasso_alpha_path.npz"
    np.savez_compressed(
        out_path,
        alphas=alphas_arr,
        n_edges=n_edges_arr,
        sparsity_pct=sparsity_pct_arr,
        patterns=patterns_arr,
        tickers=np.array(tickers, dtype=object),
    )
    logger.info(
        "Saved %d unique sparsity snapshots → %s (α %.6f..%.6f, sparsity %.1f%%..%.1f%%)",
        len(snapshots), out_path,
        float(alphas_arr.min()), float(alphas_arr.max()),
        float(sparsity_pct_arr.min()), float(sparsity_pct_arr.max()),
    )


if __name__ == "__main__":
    # Standalone entry point: skip the full pipeline and just (re)generate
    # the α-path artifact. Run with `python -m src.glasso_alpha_path`.
    import argparse

    from src.config import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="GLASSO α-path sweep (BIST only).")
    parser.add_argument("--config", default=None, help="Path to settings YAML.")
    parser.add_argument("--n-grid", type=int, default=80)
    parser.add_argument("--alpha-lo", type=float, default=1e-5)
    parser.add_argument("--alpha-hi", type=float, default=5.0)
    cli_args = parser.parse_args()

    run_glasso_alpha_path(
        load_config(cli_args.config),
        n_grid=cli_args.n_grid,
        alpha_lo=cli_args.alpha_lo,
        alpha_hi=cli_args.alpha_hi,
    )
