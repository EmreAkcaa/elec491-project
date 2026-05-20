"""In-memory pipeline runner for the Live page (PR #78).

The standard pipeline reads parquets from `data/<universe>/processed/`,
computes, writes parquets to `data/<universe>/results/`. This module
short-circuits the disk path: takes a user-uploaded price matrix as a
DataFrame, runs the cheap-and-medium stages in memory, returns a
results dict that the Live page renders.

Scope (per user direction 2026-05-20):
    Fast:        returns, correlation, distance, linkage, MST, dislocation
                 candidates, basic IT KPIs.
    Medium:      wavelet (3 scales only), GLASSO (single α), walk-forward
                 signals (single end-date snapshot).
    Excluded:    full TE, SNN, GLASSO α-path, walk-forward grid, PIT
                 snapshots, regime KL (no crisis dates without metadata).

Each stage is wrapped in try/except so a single failure on user data
doesn't kill the whole run. Progress is communicated via a callback so
Streamlit can render an `st.status` widget.

No persistence. Results live in `st.session_state` only.
"""

from __future__ import annotations

import io
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


# CSV size caps — protect HF Space CPU/memory from pathological inputs.
MAX_TICKERS = 250
MAX_ROWS = 2520  # ~10 years of trading days
MIN_TICKERS = 5
MIN_ROWS = 60


@dataclass
class LiveResult:
    """Container for all in-memory pipeline outputs.

    Each field is the corresponding artifact the dashboard would normally
    read from `data/<universe>/results/` — except they live in memory and
    will be discarded at session-end.

    `stage_status` is a dict mapping stage name → ("ok" | "skipped" |
    "error", optional message). The Live page renders this as a live
    status block during the run.
    """
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    distance: pd.DataFrame = field(default_factory=pd.DataFrame)
    linkage_Z: Optional[np.ndarray] = None
    cluster_assignments: pd.DataFrame = field(default_factory=pd.DataFrame)
    mst_edges: pd.DataFrame = field(default_factory=pd.DataFrame)
    mst_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    dislocation_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    it_kpis: dict = field(default_factory=dict)
    wavelet_corrs: dict[int, pd.DataFrame] = field(default_factory=dict)
    partial_corr: pd.DataFrame = field(default_factory=pd.DataFrame)
    walk_forward_signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    stage_status: dict[str, tuple[str, str]] = field(default_factory=dict)
    wall_time_sec: float = 0.0
    universe_label: str = "uploaded data"


# ---------------------------------------------------------------------------
# CSV validation
# ---------------------------------------------------------------------------

def validate_uploaded_csv(file_bytes: bytes) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Parse + sanity-check a user-uploaded CSV.

    Returns ``(prices_df, errors, warnings)``. `prices_df` is the parsed
    Date-indexed DataFrame on success; empty on error. `errors` is a
    list of fatal validation failures; if non-empty, do NOT proceed to
    pipeline. `warnings` are non-fatal (e.g., heuristic-detected returns
    instead of prices).

    Format: first column = date (parseable by pandas), remaining columns =
    ticker symbols with adjusted close prices (numeric).
    """
    errors: list[str] = []
    warnings: list[str] = []
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Try latin-1 as a common fallback for European exports.
        try:
            text = file_bytes.decode("latin-1")
            warnings.append("File decoded as latin-1; please prefer UTF-8 next time.")
        except Exception as exc:
            errors.append(f"Could not decode file as text: {exc}")
            return pd.DataFrame(), errors, warnings

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        errors.append(f"pandas couldn't parse the CSV: {exc}")
        return pd.DataFrame(), errors, warnings

    if df.empty or df.shape[1] < 2:
        errors.append("CSV has fewer than 2 columns (need Date + ≥1 ticker).")
        return pd.DataFrame(), errors, warnings

    # First column is the date.
    date_col = df.columns[0]
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    except Exception:
        # Try common European formats.
        try:
            df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors="raise")
        except Exception:
            errors.append(
                f"Could not parse the first column ('{date_col}') as dates. "
                "Use ISO format (YYYY-MM-DD) or a common European format."
            )
            return pd.DataFrame(), errors, warnings

    df = df.set_index(date_col).sort_index()

    # All other columns should be numeric.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop columns that are entirely NaN (failed coercion).
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)
        warnings.append(
            f"Dropped {len(all_nan_cols)} columns with all non-numeric values: "
            f"{', '.join(all_nan_cols[:8])}"
            + (f" (and {len(all_nan_cols)-8} more)" if len(all_nan_cols) > 8 else "")
        )

    # Size caps.
    if df.shape[0] > MAX_ROWS:
        errors.append(
            f"Too many rows ({df.shape[0]}); cap is {MAX_ROWS} (≈ 10 years). "
            "Trim the date range and re-upload."
        )
    if df.shape[1] > MAX_TICKERS:
        errors.append(
            f"Too many tickers ({df.shape[1]}); cap is {MAX_TICKERS}. "
            "HF Space memory cannot handle larger universes in real time."
        )
    if df.shape[0] < MIN_ROWS:
        errors.append(
            f"Too few rows ({df.shape[0]}); need ≥ {MIN_ROWS} days of data."
        )
    if df.shape[1] < MIN_TICKERS:
        errors.append(
            f"Too few tickers ({df.shape[1]}); need ≥ {MIN_TICKERS}."
        )

    if errors:
        return pd.DataFrame(), errors, warnings

    # Heuristic: if median absolute value < 0.1, likely log/raw returns,
    # not prices. Warn but proceed (compute_log_returns will produce
    # nonsense but at least the page doesn't crash).
    try:
        med_abs = float(df.abs().stack().median())
        if med_abs < 0.1:
            warnings.append(
                f"Heuristic check: median |value| = {med_abs:.4f} suggests the "
                "CSV contains RETURNS, not PRICES. Expected adjusted close "
                "prices in the typical 1–1000 range. Proceeding anyway, but "
                "results may be off."
            )
    except Exception:
        pass

    return df, errors, warnings


# ---------------------------------------------------------------------------
# In-memory pipeline orchestrator
# ---------------------------------------------------------------------------

def _step(
    result: LiveResult,
    name: str,
    fn: Callable[[], Any],
    progress_cb: Optional[Callable[[str, str, str], None]] = None,
) -> Any:
    """Run a pipeline stage, capture status, return its output or None.

    `progress_cb(name, status, message)` is called twice per stage:
    once with status="running", once with the terminal status.
    """
    if progress_cb:
        progress_cb(name, "running", "")
    try:
        out = fn()
        result.stage_status[name] = ("ok", "")
        if progress_cb:
            progress_cb(name, "ok", "")
        return out
    except Exception as exc:  # noqa: BLE001 — per-stage isolation is the point
        msg = f"{type(exc).__name__}: {exc}"
        result.stage_status[name] = ("error", msg)
        if progress_cb:
            progress_cb(name, "error", msg)
        return None


def run_live_pipeline(
    prices: pd.DataFrame,
    *,
    sector_map: Optional[dict[str, str]] = None,
    universe_label: str = "uploaded data",
    progress_cb: Optional[Callable[[str, str, str], None]] = None,
) -> LiveResult:
    """Run the in-memory pipeline on a user-uploaded price matrix.

    `prices` is a Date-indexed DataFrame with ticker columns. `sector_map`
    is optional — if provided (e.g., from a second uploader), sector-aware
    stages produce richer output; otherwise we substitute "Unknown" so the
    underlying functions don't crash.

    `progress_cb(stage_name, status, message)` is called as each stage
    starts and completes. `status ∈ {"running", "ok", "error"}`.

    Returns a `LiveResult` with whichever stages succeeded.
    """
    from src.preprocessing import compute_log_returns
    from src.analysis import compute_correlation_matrix, compute_distance_matrix
    from src.clustering import (
        compute_linkage, get_cluster_assignments, build_mst,
        compute_mst_metrics, mst_to_edge_df,
    )
    from src.pair_dislocation import rank_candidate_pairs
    from src.info_theory import (
        d_eff, log_det_term, sign_entropy_rate, predictability_diagnostics_per_ticker,
    )
    from src.wavelet_analysis import wavelet_decompose
    from src.partial_correlation import fit_graphical_lasso

    t0 = time.perf_counter()
    result = LiveResult(prices=prices, universe_label=universe_label)

    # Universe metadata stub for stages that want it (rank_candidate_pairs etc.)
    universe_df = pd.DataFrame({
        "ticker": list(prices.columns),
        "sector": [
            (sector_map or {}).get(t, "Unknown")
            for t in prices.columns
        ],
    })

    # ── Returns ──────────────────────────────────────────────────────
    returns = _step(
        result, "returns",
        lambda: compute_log_returns(prices),
        progress_cb,
    )
    if returns is None or returns.empty:
        result.wall_time_sec = time.perf_counter() - t0
        return result
    result.returns = returns

    # ── Correlation ──────────────────────────────────────────────────
    corr = _step(
        result, "correlation",
        lambda: compute_correlation_matrix(
            returns, method="pearson", min_periods=int(0.6 * len(returns)),
        ),
        progress_cb,
    )
    if corr is None or corr.empty:
        result.wall_time_sec = time.perf_counter() - t0
        return result
    # Fill any remaining NaNs with 0 so downstream linear-algebra doesn't blow up.
    result.correlation = corr.fillna(0.0)

    # ── Distance + linkage + clusters ────────────────────────────────
    dist = _step(
        result, "distance",
        lambda: compute_distance_matrix(result.correlation),
        progress_cb,
    )
    if dist is not None:
        result.distance = dist

    # compute_linkage returns (Z, labels) — unpack.
    _linkage_out = _step(
        result, "linkage",
        lambda: compute_linkage(result.distance, method="ward"),
        progress_cb,
    )
    _link_labels: Optional[list[str]] = None
    if _linkage_out is not None:
        result.linkage_Z, _link_labels = _linkage_out

    if result.linkage_Z is not None and _link_labels is not None:
        clusters = _step(
            result, "clusters",
            lambda: get_cluster_assignments(
                result.linkage_Z,
                labels=_link_labels,
                n_clusters=min(8, max(2, len(_link_labels) // 5)),
            ),
            progress_cb,
        )
        if clusters is not None:
            # Attach sector column from the universe stub for downstream rendering.
            clusters = clusters.merge(universe_df, on="ticker", how="left")
            result.cluster_assignments = clusters

    # ── MST ──────────────────────────────────────────────────────────
    mst_graph = _step(
        result, "mst",
        lambda: build_mst(result.distance),
        progress_cb,
    )
    if mst_graph is not None:
        result.mst_edges = mst_to_edge_df(mst_graph, result.correlation)
        result.mst_metrics = compute_mst_metrics(mst_graph)

    # ── Dislocation candidates ───────────────────────────────────────
    _step(
        result, "dislocation_candidates",
        lambda: _set_dislocation(result, prices, universe_df),
        progress_cb,
    )

    # ── Basic IT KPIs ────────────────────────────────────────────────
    _step(
        result, "it_kpis",
        lambda: _set_it_kpis(result),
        progress_cb,
    )

    # ── Predictability diagnostics ───────────────────────────────────
    _step(
        result, "predictability_diagnostics",
        lambda: _set_predictability(result, predictability_diagnostics_per_ticker),
        progress_cb,
    )

    # ── Wavelet (3 scales only for speed) ────────────────────────────
    _step(
        result, "wavelet",
        lambda: _set_wavelet(result, wavelet_decompose, scales=(1, 2, 3)),
        progress_cb,
    )

    # ── GLASSO (single α) ────────────────────────────────────────────
    _step(
        result, "partial_correlation",
        lambda: _set_partial_corr(result, fit_graphical_lasso),
        progress_cb,
    )

    # ── Walk-forward signals (single end-date snapshot) ──────────────
    _step(
        result, "walk_forward_signals",
        lambda: _set_walk_forward(result, prices, universe_df),
        progress_cb,
    )

    result.wall_time_sec = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Per-stage helpers (kept out of the main flow for readability)
# ---------------------------------------------------------------------------

def _set_dislocation(result: LiveResult, prices: pd.DataFrame, universe_df: pd.DataFrame) -> None:
    from src.pair_dislocation import rank_candidate_pairs
    candidates = rank_candidate_pairs(
        adj_close=prices,
        corr=result.correlation,
        universe=universe_df,
        top_n=20,
        min_correlation=0.5,
        zscore_window=60,
        lookback=252,
        entry_zscore=2.0,
        exit_zscore=0.5,
        min_half_life=5,
        max_half_life=252,
    )
    result.dislocation_candidates = candidates if candidates is not None else pd.DataFrame()


def _set_it_kpis(result: LiveResult) -> None:
    from src.info_theory import d_eff, log_det_term, sign_entropy_rate
    eigenvalues = np.linalg.eigvalsh(result.correlation.to_numpy())
    cov = result.returns.cov().to_numpy()
    sign_h_per_ticker = {
        t: sign_entropy_rate(result.returns[t].to_numpy())
        for t in result.returns.columns
    }
    mean_sign_h = float(
        pd.Series(sign_h_per_ticker).dropna().mean()
    ) if sign_h_per_ticker else float("nan")
    result.it_kpis = {
        "n_tickers": int(result.returns.shape[1]),
        "n_observations": int(result.returns.shape[0]),
        "d_eff": float(d_eff(eigenvalues)),
        "log_det_term": float(log_det_term(cov)),
        "mean_sign_entropy_rate_bits": mean_sign_h,
    }


def _set_predictability(result: LiveResult, fn: Callable) -> None:
    df = fn(result.returns)
    if df is not None and not df.empty:
        result.it_kpis["predictability_table"] = df


def _set_wavelet(result: LiveResult, decompose_fn: Callable, scales: tuple[int, ...]) -> None:
    """Decompose returns at the chosen scales and compute pairwise correlation
    of the detail coefficients at each scale.

    Signature alignment with `src.wavelet_analysis.wavelet_decompose`:
        wavelet_decompose(returns, wavelet="db4", max_level=N) -> dict[int, DataFrame]
    """
    decomposition = decompose_fn(result.returns, wavelet="db4", max_level=max(scales))
    # `decomposition` is a dict {scale: DataFrame of detail coefficients}.
    wavelet_corrs: dict[int, pd.DataFrame] = {}
    for s in scales:
        if s not in decomposition:
            continue
        details = decomposition[s]
        if details is None or details.empty:
            continue
        # Pearson correlation on the wavelet detail coefficients at this scale.
        wavelet_corrs[s] = details.corr(method="pearson")
    result.wavelet_corrs = wavelet_corrs


def _set_partial_corr(result: LiveResult, fit_fn: Callable) -> None:
    """Fit GLASSO at a single α (cross-validated within the function), keep
    the partial-correlation matrix."""
    out = fit_fn(result.returns)
    # fit_graphical_lasso returns (partial_corr, precision, alpha) per the
    # existing signature. Capture all three.
    if isinstance(out, tuple):
        partial = out[0]
    else:
        partial = out
    if partial is not None and hasattr(partial, "shape"):
        result.partial_corr = partial


def _set_walk_forward(result: LiveResult, prices: pd.DataFrame, universe_df: pd.DataFrame) -> None:
    """Run walk-forward signals at the LATEST date in the panel only.

    Live demo doesn't need a date scrubber — one snapshot is enough.
    """
    from src.walk_forward_signals import _compute_one_date
    end_date = result.returns.index[-1]
    snapshot = _compute_one_date(
        adj_close=prices,
        returns=result.returns,
        end_date=end_date,
        universe_df=universe_df,
        top_n=20,
        min_correlation=0.5,
        zscore_window=60,
        lookback=252,
        entry_zscore=2.0,
        exit_zscore=0.5,
        min_half_life=5,
        max_half_life=252,
    )
    if snapshot is not None and not snapshot.empty:
        result.walk_forward_signals = snapshot
