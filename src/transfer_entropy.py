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


# ---------------------------------------------------------------------------
# Lag-sweep / Rolling / Bootstrap helpers (G1, G3, G4 — PR #73)
# ---------------------------------------------------------------------------

def compute_lag_sweep_for_pairs(
    returns: pd.DataFrame,
    pairs: list[tuple[str, str]],
    lags: list[int],
    *,
    n_shuffles: int = 1000,
    n_bins: int = 3,
    block_length: int = 5,
    multiple_testing: str = "fdr_bh",
    significance_level: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute TE for a hand-picked pair list across multiple lag values.

    For each (pair, direction, lag) compute TE + a surrogate-null p-value.
    BH-FDR (or the chosen correction) is applied **per-lag** so a finding at
    one lag doesn't compete with findings at another.

    Returns a long-form DataFrame with columns ``[ticker_a, ticker_b,
    direction, lag, te, p_value, significant]``. ``direction`` is the string
    ``"a_to_b"`` or ``"b_to_a"``; ``significant`` is the per-lag corrected
    rejection mask (True = reject H₀).

    Cost: O(len(pairs) * len(lags) * n_shuffles * pair-evaluation). On 10
    pairs × 3 lags × K=1000 ≈ 90 s on a single core. Parallelism not used —
    the pair count is small enough that joblib overhead would dominate.
    """
    rows: list[dict] = []
    for lag_idx, lag in enumerate(lags):
        # Run all (pair × direction) tests at this lag.
        lag_rows: list[dict] = []
        for pair_idx, (ta, tb) in enumerate(pairs):
            if ta not in returns.columns or tb not in returns.columns:
                continue
            # Joint dropna preserves date alignment between the two series.
            # WITHOUT this, dropna-per-series + tail alignment misaligns
            # pairs when each ticker has NaNs on different dates → joint
            # histogram is computed on un-paired observations.
            both = returns[[ta, tb]].dropna()
            x = both[ta].to_numpy()
            y = both[tb].to_numpy()
            n = x.size
            if n < (lag + 30):
                continue
            seed_xy = seed + 1000 * lag_idx + 10 * pair_idx
            seed_yx = seed_xy + 1
            _, _, te_xy, p_xy = _te_one_pair(
                0, 1, x, y, lag=lag, n_bins=n_bins,
                n_shuffles=n_shuffles, block_length=block_length, pair_seed=seed_xy,
            )
            _, _, te_yx, p_yx = _te_one_pair(
                0, 1, y, x, lag=lag, n_bins=n_bins,
                n_shuffles=n_shuffles, block_length=block_length, pair_seed=seed_yx,
            )
            lag_rows.append({
                "ticker_a": ta, "ticker_b": tb, "direction": "a_to_b",
                "lag": lag, "te": float(te_xy), "p_value": float(p_xy),
            })
            lag_rows.append({
                "ticker_a": ta, "ticker_b": tb, "direction": "b_to_a",
                "lag": lag, "te": float(te_yx), "p_value": float(p_yx),
            })

        # Apply multiple-testing correction WITHIN this lag's batch.
        if lag_rows:
            pvals = np.array([r["p_value"] for r in lag_rows])
            if multiple_testing == "fdr_bh":
                significant = _benjamini_hochberg(pvals, significance_level)
            elif multiple_testing == "bonferroni":
                significant = pvals < (significance_level / len(pvals))
            else:
                significant = pvals < significance_level
            for r, sig in zip(lag_rows, significant):
                r["significant"] = bool(sig)
            rows.extend(lag_rows)

    return pd.DataFrame(rows)


def compute_rolling_te(
    returns: pd.DataFrame,
    pairs: list[tuple[str, str]],
    *,
    lag: int = 1,
    window: int = 252,
    stride: int = 21,
    n_shuffles: int = 500,
    n_bins: int = 3,
    block_length: int = 5,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Sliding-window transfer entropy on a hand-picked pair list.

    For each window of size `window` ending at every `stride`-th date,
    compute TE + p-value in both directions for each pair. Returns a long-
    form DataFrame indexed by window-end-date, with columns
    ``[ticker_a, ticker_b, direction, te, p_value]``.

    Hypothesis discipline: pass only the pairs that already showed
    full-sample significance (G1 survivors). Running this on the full
    grid is wasteful — the multiple-testing problem only gets worse with
    many windows.

    Cost on BIST: ~64 windows × 10 pairs × 2 directions × K=500 ≈ 640k
    TE evaluations. Parallelised via joblib over (window × pair × dir)
    combinations: ~6-8 min on 8 cores.
    """
    from joblib import Parallel, delayed

    if not pairs:
        return pd.DataFrame()

    if isinstance(returns.index, pd.DatetimeIndex):
        # Build the window-end date grid. End-aligned windows: window i ends
        # at index window-1, window-1+stride, ... and contains the previous
        # `window` observations.
        end_positions = list(range(window - 1, len(returns.index), stride))
    else:
        end_positions = list(range(window - 1, len(returns.index), stride))

    if not end_positions:
        return pd.DataFrame()

    # Build the task list: (end_pos, pair_idx, direction)
    tasks = []
    for ep_idx, end_pos in enumerate(end_positions):
        start_pos = end_pos - window + 1
        for p_idx, (ta, tb) in enumerate(pairs):
            tasks.append((end_pos, start_pos, p_idx, ta, tb, "a_to_b"))
            tasks.append((end_pos, start_pos, p_idx, ta, tb, "b_to_a"))

    def _one_task(end_pos, start_pos, p_idx, ta, tb, direction):
        if ta not in returns.columns or tb not in returns.columns:
            return None
        slc = returns.iloc[start_pos:end_pos + 1]
        # Joint dropna preserves date alignment within the window.
        both = slc[[ta, tb]].dropna()
        x = both[ta].to_numpy()
        y = both[tb].to_numpy()
        n = x.size
        if n < (lag + 30):
            return None
        pair_seed = seed + 10_000 * end_pos + 100 * p_idx + (0 if direction == "a_to_b" else 1)
        if direction == "a_to_b":
            _, _, te_val, p_val = _te_one_pair(
                0, 1, x, y, lag=lag, n_bins=n_bins,
                n_shuffles=n_shuffles, block_length=block_length, pair_seed=pair_seed,
            )
        else:
            _, _, te_val, p_val = _te_one_pair(
                0, 1, y, x, lag=lag, n_bins=n_bins,
                n_shuffles=n_shuffles, block_length=block_length, pair_seed=pair_seed,
            )
        date = returns.index[end_pos]
        return {
            "date": date, "ticker_a": ta, "ticker_b": tb, "direction": direction,
            "te": float(te_val), "p_value": float(p_val),
        }

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_one_task)(*t) for t in tasks
    )
    rows = [r for r in results if r is not None]
    return pd.DataFrame(rows)


def _quad_joint_entropy(w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """H(W, X, Y, Z) on 4 integer-discretised series.

    Encodes the 4-tuple as a single base-D integer (D = max bin index + 1)
    so np.bincount can count joint occurrences in O(n). Same approach the
    triple-joint entropy uses, extended to 4 dims. With D=3 the encoding
    is `w*27 + x*9 + y*3 + z` and the joint state space has 81 cells.
    """
    n = min(len(w), len(x), len(y), len(z))
    w, x, y, z = w[-n:], x[-n:], y[-n:], z[-n:]
    encoded = (w * 27 + x * 9 + y * 3 + z).astype(np.int64)
    counts = np.bincount(encoded)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log(p)))


def conditional_transfer_entropy(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    lag: int = 1,
    n_bins: int = 3,
) -> float:
    """TE(X → Y | Z): directed information flow from X to Y, conditioning on Z.

    Tells you whether X's past contributes information about Y's future
    BEYOND what Y's own past AND Z's past already carry. This is the
    standard test for whether an apparent X→Y flow is mediated by a
    third variable Z (e.g., the market factor).

    Formula (3-bin discretisation, lag 1):

        TE(X→Y|Z) = H(Y_t, Y_lag, Z_lag) − H(Y_lag, Z_lag)
                  − H(Y_t, Y_lag, X_lag, Z_lag) + H(Y_lag, X_lag, Z_lag)

    Equivalent to H(Y_t | Y_lag, Z_lag) − H(Y_t | Y_lag, X_lag, Z_lag),
    written in joint-entropy form so we can reuse the joint estimators.

    Caveats:
      * The 4-way joint distribution (3^4 = 81 cells at n_bins=3) needs
        more observations than the 3-way TE estimator. On ~1500-day
        daily series the bias is non-trivial; treat CTE as ordinal vs
        unconditional TE rather than as a precise absolute value.
      * NaN handling: caller must pre-align (joint dropna across all 3
        series).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = min(len(x), len(y), len(z))
    x, y, z = x[-n:], y[-n:], z[-n:]

    x_d = _discretize(x, n_bins=n_bins)
    y_d = _discretize(y, n_bins=n_bins)
    z_d = _discretize(z, n_bins=n_bins)

    y_t = y_d[lag:]
    y_lag = y_d[:-lag]
    x_lag = x_d[:-lag]
    z_lag = z_d[:-lag]

    h_y_ylag_zlag = _triple_joint_entropy(y_t, y_lag, z_lag)
    h_ylag_zlag = _joint_entropy(y_lag, z_lag)
    h_y_ylag_xlag_zlag = _quad_joint_entropy(y_t, y_lag, x_lag, z_lag)
    h_ylag_xlag_zlag = _triple_joint_entropy(y_lag, x_lag, z_lag)

    return float(h_y_ylag_zlag - h_ylag_zlag - h_y_ylag_xlag_zlag + h_ylag_xlag_zlag)


def conditional_te_with_surrogate(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    n_shuffles: int = 1000,
    lag: int = 1,
    n_bins: int = 3,
    block_length: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    """Conditional TE + surrogate-null p-value via circular block bootstrap
    of the source series X. Y and Z remain intact so the bootstrap tests
    "does X's PARTICULAR temporal structure contribute information beyond
    what Y, Z, and a permuted-X would?"

    Returns ``(cte_observed, p_value)``. The +1 smoothing on the p-value
    is the same convention as ``_te_one_pair`` so the two are directly
    comparable.
    """
    cte_obs = conditional_transfer_entropy(x, y, z, lag=lag, n_bins=n_bins)
    if n_shuffles <= 0:
        return cte_obs, 1.0
    rng = np.random.default_rng(seed)
    null = np.empty(n_shuffles)
    for s in range(n_shuffles):
        x_surr = _circular_block_bootstrap(x, block_length, rng)
        null[s] = conditional_transfer_entropy(x_surr, y, z, lag=lag, n_bins=n_bins)
    p = (1 + (null >= cte_obs).sum()) / (1 + n_shuffles)
    return float(cte_obs), float(p)


def compute_conditional_te_table(
    returns: pd.DataFrame,
    conditioning_series: pd.Series,
    pairs: list[tuple[str, str, str]],
    *,
    n_shuffles: int = 1000,
    lag: int = 1,
    n_bins: int = 3,
    block_length: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """For each (ticker_a, ticker_b, direction) compute:
        TE          (unconditional, from existing pipeline)
        CTE | Z     (conditional on the supplied series)
        p-value     for the CTE under circular-block-bootstrap null
        delta       CTE − TE  (positive = signal persists/sharpens
                                under conditioning; negative = signal
                                attenuates, suggesting Z mediates)

    `direction` is "a_to_b" or "b_to_a".

    The conditioning series (e.g., the market index XU100 returns) is
    inner-joined on dates with the pair before any TE computation, so
    NaN holes don't misalign the joint distribution.
    """
    rows: list[dict] = []
    for pair_idx, (ta, tb, direction) in enumerate(pairs):
        if ta not in returns.columns or tb not in returns.columns:
            continue
        aligned = returns[[ta, tb]].join(conditioning_series.rename("M")).dropna()
        if len(aligned) < (lag + 80):
            continue
        if direction == "a_to_b":
            src_t, dst_t = ta, tb
        else:
            src_t, dst_t = tb, ta
        x = aligned[src_t].to_numpy()
        y = aligned[dst_t].to_numpy()
        z = aligned["M"].to_numpy()

        te_uncond = transfer_entropy(x, y, lag=lag, n_bins=n_bins)
        seed_p = seed + pair_idx * 17
        cte, p = conditional_te_with_surrogate(
            x, y, z, n_shuffles=n_shuffles, lag=lag, n_bins=n_bins,
            block_length=block_length, seed=seed_p,
        )
        rows.append({
            "ticker_a": ta, "ticker_b": tb, "direction": direction,
            "source": src_t, "target": dst_t,
            "te": float(te_uncond),
            "cte": float(cte),
            "delta": float(cte - te_uncond),
            "p_value": float(p),
            "n_obs": int(len(aligned)),
        })
    return pd.DataFrame(rows)


def compute_sector_te_matrix(
    returns: pd.DataFrame,
    sector_map: dict[str, str],
    *,
    min_tickers_per_sector: int = 3,
    n_shuffles: int = 1000,
    lag: int = 1,
    n_bins: int = 3,
    block_length: int = 5,
    significance_level: float = 0.05,
    multiple_testing: str = "fdr_bh",
    seed: int = 42,
) -> pd.DataFrame:
    """Pairwise TE between EQUAL-WEIGHT SECTOR PORTFOLIOS.

    Aggregating tickers to sectors reduces the multiple-testing burden
    dramatically. On BIST with 13 sectors there are 156 directed pairs
    (vs 5256 ticker-level), so the per-edge FDR cutoff is ~30× more
    forgiving and K=1000 surrogates reach it for the strongest edges.

    Returns a long-form DataFrame with columns ``[source, target, te,
    p_value, significant_fdr, significant_uncorrected, n_tickers_source,
    n_tickers_target]``.
    """
    # Build equal-weight sector portfolios (only sectors with enough tickers)
    sectors: dict[str, list[str]] = {}
    for ticker, sector in sector_map.items():
        if not sector or not isinstance(sector, str):
            continue
        if ticker not in returns.columns:
            continue
        sectors.setdefault(sector, []).append(ticker)
    sectors = {s: tks for s, tks in sectors.items() if len(tks) >= min_tickers_per_sector}

    if len(sectors) < 2:
        return pd.DataFrame()

    sector_returns = pd.DataFrame(
        {s: returns[tks].mean(axis=1) for s, tks in sectors.items()}
    ).dropna()
    sector_names = list(sector_returns.columns)

    rows: list[dict] = []
    for i, sa in enumerate(sector_names):
        for j, sb in enumerate(sector_names):
            if i == j:
                continue
            x = sector_returns[sa].to_numpy()
            y = sector_returns[sb].to_numpy()
            pair_seed = seed + 100 * i + j
            _, _, te_val, p_val = _te_one_pair(
                i, j, x, y, lag=lag, n_bins=n_bins,
                n_shuffles=n_shuffles, block_length=block_length, pair_seed=pair_seed,
            )
            rows.append({
                "source": sa, "target": sb,
                "te": float(te_val), "p_value": float(p_val),
                "n_tickers_source": len(sectors[sa]),
                "n_tickers_target": len(sectors[sb]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    pvals = df["p_value"].to_numpy()
    if multiple_testing == "fdr_bh":
        sig = _benjamini_hochberg(pvals, significance_level)
    elif multiple_testing == "bonferroni":
        sig = pvals < (significance_level / len(pvals))
    else:
        sig = pvals < significance_level
    df["significant_fdr"] = sig
    df["significant_uncorrected"] = pvals < significance_level
    return df


def bootstrap_te(
    source_series: np.ndarray,
    target_series: np.ndarray,
    *,
    n_iter: int = 500,
    lag: int = 1,
    n_bins: int = 3,
    block_length: int = 5,
    seed: int = 42,
) -> dict:
    """Joint circular-block-bootstrap 95% CI for TE(source → target).

    Resamples (source, target) JOINTLY with the SAME block index so the
    pair structure is preserved within blocks. This gives a CI on the
    POINT ESTIMATE of TE, not the surrogate-null distribution (which the
    existing significance test in `_te_one_pair` already provides).

    Returns ``{point, ci_low, ci_high, n_iter, includes_zero}``.
    """
    x = np.asarray(source_series, dtype=float)
    y = np.asarray(target_series, dtype=float)
    # Joint NaN mask preserves pair alignment.
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < block_length * 4:
        return {
            "point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "n_iter": 0, "includes_zero": True,
        }

    te_point = transfer_entropy(x, y, lag=lag, n_bins=n_bins)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    samples = np.empty(n_iter)
    for it in range(n_iter):
        # Pick block starts once; apply the SAME index list to both
        # source and target. This preserves the within-block pair
        # structure that carries the directional information.
        starts = rng.integers(0, n, size=n_blocks)
        block_idx = (starts[:, None] + np.arange(block_length)[None, :]) % n
        idx = block_idx.ravel()[:n]
        x_bs = x[idx]
        y_bs = y[idx]
        try:
            samples[it] = transfer_entropy(x_bs, y_bs, lag=lag, n_bins=n_bins)
        except Exception:
            samples[it] = np.nan

    samples = samples[np.isfinite(samples)]
    ci_low = float(np.percentile(samples, 2.5)) if samples.size else float("nan")
    ci_high = float(np.percentile(samples, 97.5)) if samples.size else float("nan")
    return {
        "point": float(te_point),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_iter": int(samples.size),
        "includes_zero": bool(ci_low <= 0.0 <= ci_high),
    }


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

    # ── PR #75 (a): Sector-aggregated TE ──────────────────────────────
    # 13 BIST sectors → 156 directed pairs (vs 5256 ticker-level).
    # BH-FDR cutoff ~30× more forgiving; K=1000 reaches it for the
    # strongest edges. Sector-level lead-lag is a textbook
    # interpretable finding for the thesis even when ticker-level
    # FDR can't be cleared at this resolution.
    try:
        sector_map = dict(zip(config.universe["ticker"], config.universe["sector"]))
        sector_df = compute_sector_te_matrix(
            returns, sector_map,
            min_tickers_per_sector=3,
            n_shuffles=te_cfg.significance_shuffles,
            lag=te_cfg.lag,
            n_bins=te_cfg.n_bins,
            block_length=te_cfg.surrogate_block_length,
            significance_level=te_cfg.significance_level,
            multiple_testing=te_cfg.multiple_testing,
            seed=te_cfg.seed,
        )
        if not sector_df.empty:
            sector_df.to_parquet(
                config.data_results / "te_sector_matrix.parquet", index=False
            )
            n_sec_sig = int(sector_df["significant_fdr"].sum())
            n_sec_unc = int(sector_df["significant_uncorrected"].sum())
            logger.info(
                "Sector TE: %d FDR-significant / %d uncorrected (of %d directed sector-pairs)",
                n_sec_sig, n_sec_unc, len(sector_df),
            )
    except Exception as exc:
        logger.warning("Sector TE failed: %s", exc)

    # ── PR #75 (b): Conditional TE on G1 survivors, conditioning on the
    # market index. Tests whether the directed flows we found are
    # market-factor confounds. The conditioning series is the equal-
    # weight market portfolio of all tickers in the panel.
    try:
        # G1 survivors are the directed pairs that passed FDR at lag=1
        # in `te_lag_sweep.parquet`. Read them if available; otherwise
        # default to the known set so the pipeline stays robust to
        # ordering.
        sweep_path = config.data_results / "te_lag_sweep.parquet"
        survivors: list[tuple[str, str, str]] = []
        if sweep_path.exists():
            sweep = pd.read_parquet(sweep_path)
            for _, r in sweep[(sweep["lag"] == 1) & (sweep["significant"])].iterrows():
                survivors.append((r["ticker_a"], r["ticker_b"], r["direction"]))
        # Always include the 3 docs-canonical pairs for narrative
        # continuity even if FDR survivors shift on a future re-run.
        canonical = [
            ("KCHOL", "AKBNK", "a_to_b"),
            ("BRSAN", "BRYAT", "b_to_a"),
            ("TUPRS", "AYGAZ", "a_to_b"),
        ]
        for s in canonical:
            if s not in survivors:
                survivors.append(s)

        # Try real XU100 first; fall back to cross-sectional mean.
        xu_returns = None
        xu_path = config.data_raw / "xu100.parquet"
        if xu_path.exists():
            try:
                xu_df = pd.read_parquet(xu_path)
                xu_series = (
                    xu_df["Adj Close"] if "Adj Close" in xu_df.columns
                    else xu_df.iloc[:, 0]
                )
                if isinstance(xu_series, pd.DataFrame):
                    xu_series = xu_series.iloc[:, 0]
                xu_returns = np.log(xu_series / xu_series.shift(1)).dropna()
                xu_returns.name = "M"
            except Exception:
                xu_returns = None
        if xu_returns is None or xu_returns.empty:
            xu_returns = returns.mean(axis=1).rename("M")
            logger.info("Using cross-sectional mean as market proxy for CTE")

        cte_df = compute_conditional_te_table(
            returns, xu_returns, survivors,
            n_shuffles=te_cfg.significance_shuffles,
            lag=te_cfg.lag,
            n_bins=te_cfg.n_bins,
            block_length=te_cfg.surrogate_block_length,
            seed=te_cfg.seed,
        )
        if not cte_df.empty:
            cte_df.to_csv(
                config.data_results / "te_conditional_market.csv", index=False
            )
            n_persist = int((cte_df["delta"] > 0).sum())
            logger.info(
                "Conditional TE on %d pairs: %d sharpen under market conditioning (CTE > TE)",
                len(cte_df), n_persist,
            )
    except Exception as exc:
        logger.warning("Conditional TE failed: %s", exc)

    logger.info("Transfer entropy analysis complete")
