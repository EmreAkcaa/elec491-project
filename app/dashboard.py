"""StoNeCoAl — multi-universe correlation-network dashboard (BIST 100, S&P 500)."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import linkage, leaves_list  # noqa: F401
from scipy.spatial.distance import squareform  # noqa: F401

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

_APP_DIR      = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ──────────────────────────────────────────────────────────────────────────────
# EEG bulk-data materialisation
# ──────────────────────────────────────────────────────────────────────────────
# HF Spaces caps per-repo storage at 1 GB. Our 2 EEG processed parquets are
# 308 MB each — too big to ship in the Space repo. The canonical HF workaround
# is to put bulk data in a companion Dataset repo (50 GB per file) and have
# the Space download it on first launch.
#
# Local dev: parquets already on disk → no-op early-return.
# HF Spaces:  files absent → snapshot_download from EEG_DATASET_REPO once;
#             cached under ~/.cache/huggingface on subsequent reruns.
# Fallback:   if the download fails, EEG silently drops from the sidebar
#             selector (available_universes() detects the absence and filters).
def _materialise_eeg_data_if_needed() -> None:
    import os as _os
    # CI / smoke tests opt-out: setting STONECOAL_SKIP_EEG_DOWNLOAD=1 avoids
    # a slow / failing network call when EEG isn't part of the test surface.
    if _os.environ.get("STONECOAL_SKIP_EEG_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return

    eeg_dir = _PROJECT_ROOT / "data" / "eeg_motor_left_right" / "processed"
    sentinel = eeg_dir / "log_returns.parquet"
    if sentinel.exists() and sentinel.stat().st_size > 1_000_000:
        return  # local dev, or already-cached HF Spaces rebuild

    repo_id = _os.environ.get("EEG_DATASET_REPO", "FlyingSubmarine33/stonecoal-eeg")
    print(f"[EEG] Bulk parquets not on disk; fetching from dataset repo {repo_id} …")
    try:
        from huggingface_hub import snapshot_download
        eeg_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(eeg_dir),
            allow_patterns=["*.parquet", "*.csv"],
        )
        print(f"[EEG] Materialised bulk data from {repo_id} into {eeg_dir}")
    except Exception as exc:  # noqa: BLE001 — best-effort; failure is non-fatal
        # Don't crash the dashboard — available_universes() will see the
        # missing files and quietly omit EEG from the sidebar selector.
        print(f"[EEG] Could not fetch from {repo_id}: {exc}")
        print(f"[EEG] Dashboard will run with BIST + S&P only. To enable EEG, "
              f"upload the parquets with: uv run python scripts/upload_eeg_to_hf_dataset.py")


_materialise_eeg_data_if_needed()


from src.rolling_correlation import (  # noqa: E402
    compute_rolling_market_stats,
    compute_rolling_pair_correlation,
    compute_rolling_sector_stats,
    compute_window_correlation,
)

from utils import (  # noqa: E402
    PROJECT_ROOT, data_processed, current_universe,
    load_adj_close, load_log_returns, load_summary_stats, load_batch_corr,
    load_coverage, load_top_bottom, load_metadata, load_fetch_metadata,
    load_xu100, load_linkage, load_dendrogram_order, load_cluster_assignments,
    load_mst_edges, load_mst_metrics, load_dislocation_candidates, load_anomalies,
    load_rolling_market_stats_precomputed, load_rolling_sector_stats_precomputed,
    draw_event_markers, event_marker_manager_ui,
    get_colors, SECTOR_PALETTE, CHART_LAYOUT, apply_chart_style, inject_custom_css,
    section_header, render_chart, render_matrix_heatmap,
)
from chart_themes import render_theme_sidebar  # noqa: E402

# HF Spaces rebuilds the container on every deploy, so the stale-module cache
# problem that motivated importlib.reload on Streamlit Cloud no longer applies.
# Removing the reload eliminates Universe class identity churn across reruns —
# previously a contributor to "Tried to use SessionInfo before it was
# initialized" warnings in the server logs (PR #23).
from universe_registry import available_universes, get_universe  # noqa: E402


def _cap(u, attr, default):
    """Defensive capability lookup. If Streamlit Cloud has a stale `Universe`
    class cached without a Phase I field, fall back to ``default`` rather than
    crashing the dashboard with AttributeError. Treat finance defaults as the
    safe fallback for unknown universes since the original dashboard was
    finance-only."""
    return getattr(u, attr, default)


# ══════════════════════════════════════════════════════════════════════════════
# Cached computation helpers (module-level to avoid Streamlit re-registration)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _compute_corr(_returns: pd.DataFrame, cache_key: str, min_periods: int, method: str):
    # `_returns` underscore-prefix → Streamlit skips hashing the DataFrame;
    # the explicit `cache_key` (built cheaply by the caller from universe +
    # date endpoints + shape) drives cache identity. Replaces the previous
    # JSON ser/de pattern which cost 121 ms (38 ms to_json on every rerun
    # + 83 ms read_json on miss) per call on S&P-500 returns.
    return _returns.corr(method=method, min_periods=min_periods)


@st.cache_data(show_spinner=False)
def _pit_corr(_returns: pd.DataFrame, cache_key: str, end_date_str, window, method):
    return compute_window_correlation(
        _returns, pd.Timestamp(end_date_str), window=window, method=method,
    )


@st.cache_data(show_spinner=False)
def _mst_layout(_edges: pd.DataFrame, cache_key: str):
    _G = nx.Graph()
    for _, r in _edges.iterrows():
        _G.add_edge(r["source"], r["target"], weight=r["distance"])
    # Kamada-Kawai is O(N^3) and stalls for ~minute on the 485-node S&P MST.
    # spring_layout (Fruchterman-Reingold) with a fixed iteration count
    # gives a comparable-quality layout on the same MST in ~1 second; we
    # only fall back to it for large graphs so small graphs (BIST/EEG/<200
    # nodes) keep the cleaner Kamada-Kawai layout.
    if _G.number_of_nodes() > 200:
        return nx.spring_layout(_G, weight="weight", iterations=80, seed=42)
    return nx.kamada_kawai_layout(_G, weight="weight")


def _heatmap_axis_tickfont(n: int) -> int:
    """Tickfont size scaled to ticker count. Returns 0 to flag "hide the
    labels entirely" — callers translate that into `showticklabels=False`
    plus a sentinel font size (plotly rejects size=0)."""
    if n <= 80:
        return 7
    if n <= 200:
        return 5
    return 0


def _heatmap_axis_dtick(n: int) -> int:
    """Show every label up to ~80 tickers, then thin out so plotly
    doesn't try to render hundreds of axis annotations per heatmap."""
    if n <= 80:
        return 1
    if n <= 200:
        return 5
    return max(1, n // 30)  # ~30 visible labels max


def _heatmap_height(n: int, max_px: int = 1100) -> int:
    """Bound the heatmap pixel height so the browser doesn't have to
    paint a 5000px-tall canvas. ~12 px/cell up to the cap, then a
    flat plateau that lets plotly downsample-render."""
    return min(max_px, max(700, n * 12))


@st.cache_data(show_spinner=False)
def _compute_market_stats(_returns: pd.DataFrame, cache_key: str, window, step, method, expanding):
    return compute_rolling_market_stats(
        _returns, window=window, step=step, method=method, expanding=expanding,
    )


@st.cache_data(show_spinner=False)
def _compute_pair(_returns: pd.DataFrame, cache_key: str, a, b, window, method, wtype, ewm_span=None):
    # `ewm_span` is honoured only when wtype == "ewm"; the underlying
    # compute_rolling_pair_correlation ignores it for rolling/expanding.
    return compute_rolling_pair_correlation(
        _returns, a, b,
        window=window, method=method, window_type=wtype,
        ewm_span=ewm_span,
    )


@st.cache_data(show_spinner=False)
def _compute_sector(_returns: pd.DataFrame, cache_key: str, sec_map_items, window, step, method):
    return compute_rolling_sector_stats(
        _returns, dict(sec_map_items), window=window, step=step, method=method,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Fragment-scoped sub-tab renderers
# ══════════════════════════════════════════════════════════════════════════════
# Streamlit reruns the entire dashboard.py script on every widget interaction.
# Fragments (stable since Streamlit 1.37) scope re-execution to just their
# decorated function when an internal widget changes — other tabs/sub-tabs
# don't re-render, killing the perceived "gray screen" on slider drags etc.
#
# Each fragment reads its dependencies from module globals set by the linear
# script flow (returns, returns_cache_key, _active_universe, etc.). On fragment
# rerun those globals are NOT recomputed; that's correct because nothing inside
# the fragment changes them — they only change via outer widgets (which trigger
# a full script rerun anyway).
#
# We deliberately do NOT fragment:
#   - Rolling Market Stats → widgets are outer (rc_*), fragmenting just this
#     sub-tab gives marginal win.


def _open_pair_analysis_button(ticker_a: str, ticker_b: str, *, key: str) -> None:
    """Single canonical cross-page nav button to Pair Analysis with
    (ticker_a, ticker_b) preloaded.

    Used at every callsite (Rolling Pair sub-tab, Pairs & Dislocations
    Top/Bottom tabs, Dislocation Candidates) so the copy, icon, and
    session_state plumbing stay consistent. Audit item A3 — before this
    helper the same action shipped as 4 buttons with 3 different labels.

    `st.rerun(scope="app")` works both inside `@st.fragment` contexts
    (where the default fragment-scoped rerun would NOT switch pages)
    and outside them (where it's a plain full-script rerun).
    """
    if st.button(
        f":material/open_in_new:  Analyze {ticker_a} / {ticker_b} in Pair Analysis",
        key=key,
        type="secondary",
        use_container_width=True,
    ):
        st.session_state["pa_ticker_a"] = ticker_a
        st.session_state["pa_ticker_b"] = ticker_b
        st.session_state["_goto_pair_analysis"] = True
        st.rerun(scope="app")


@st.fragment
def _render_correlation_heatmap() -> None:
    """Full-period correlation heatmap. Owns heat_method + use_clustering_order
    widgets. Computes `corr` via @st.cache_data — the Pairs & Dislocations tab
    later recomputes the same `_compute_corr(...)` call using the heat_method
    value from session_state and gets a cache HIT, so cross-tab semantics are
    preserved without paying the compute twice.
    """
    _heat_c1, _heat_c2 = st.columns(2)
    with _heat_c1:
        heat_method = st.selectbox(
            "Correlation method", ["pearson", "spearman"],
            key="heat_method",
        )
    with _heat_c2:
        use_clustering_order = st.checkbox(
            "Reorder by hierarchical clustering", value=True, key="mo_corr_reorder",
        )

    with st.spinner("Computing correlation matrix..."):
        corr = _compute_corr(returns, returns_cache_key, dynamic_min_periods, heat_method)
    leaf_order = load_dendrogram_order()

    if use_clustering_order and leaf_order is not None:
        valid_order = [t for t in leaf_order if t in corr.columns]
        corr_display = corr.loc[valid_order, valid_order] if valid_order else corr
    else:
        corr_display = corr

    with st.container(border=True):
        _series_lower = _cap(_active_universe, 'series_label', 'Log return').lower()
        _samp_unit = (
            "trading days" if _cap(_active_universe, 'domain', 'finance') == "finance"
            else "samples"
        )
        st.caption(
            f"Pairwise correlation matrix of {_series_lower}s across all "
            f"{_samp_unit} in the window. Toggle method and clustering reorder above.",
        )

        render_matrix_heatmap(
            corr_display,
            chart_id="mo_heatmap",
            filename_base="correlation_heatmap",
            title_key="mo_heatmap",
            default_title="Correlation Matrix",
            zmin=-1.0, zmax=1.0, diverging=True,
            height=_heatmap_height(min(len(corr_display), 200)),
            hover_label="corr",
            colorbar_title="Corr",
        )
        with st.expander(
            f"Download full-resolution matrix as CSV "
            f"({corr_display.shape[0]}×{corr_display.shape[0]})"
        ):
            st.download_button(
                "Download CSV",
                data=corr_display.to_csv().encode("utf-8"),
                file_name=f"correlation_{_active_universe.key}.csv",
                mime="text/csv",
                key="mo_heatmap_csv_dl",
            )


@st.fragment
def _render_pit_correlation() -> None:
    """Point-in-time correlation snapshot. Owns pit_window/pit_method/pit_date.
    Dragging the date slider used to rerun the whole 1700-line script; now it
    reruns only this block."""
    _is_finance_pit = _cap(_active_universe, 'domain', 'finance') == "finance"
    st.caption(
        "Pick a date + window to see the correlation matrix at that point in time. "
        + ("Useful for comparing market structure during crises vs calm periods."
           if _is_finance_pit else
           "Useful for tracking how network structure shifts across the recording.")
    )

    trading_dates = returns.index.tolist()
    if not trading_dates:
        st.warning("No data available for this universe.")
        return
    _date_min = trading_dates[0].date()
    _date_max = trading_dates[-1].date()

    pit_c1, pit_c2, pit_c3 = st.columns(3)
    with pit_c1:
        _win_label = "Window (days)" if _is_finance_pit else "Window (samples)"
        # Number input replaces the 3-option selectbox. Bounded so the user
        # can't pick a window larger than the data window itself; default 252
        # matches the prior selectbox default.
        pit_window = int(st.number_input(
            _win_label,
            min_value=20,
            max_value=min(504, max(20, len(trading_dates))),
            value=252 if 252 <= len(trading_dates) else max(20, len(trading_dates) // 2),
            step=10,
            key="pit_window",
            help="Trading days in the rolling window used to compute the correlation snapshot.",
        ))
    with pit_c2:
        pit_method = st.selectbox(
            "Method", ["pearson", "spearman"], key="pit_method",
        )
    with pit_c3:
        # Date picker replaces the select_slider. The slider was good for
        # exploration but bad for known-date queries ("show me 2020-03-12");
        # number_input + date_input still re-runs only this fragment, so
        # interactive scrubbing is preserved at the cost of two clicks
        # instead of a drag.
        valid_start = max(0, pit_window - 1)
        _min_pick = trading_dates[valid_start].date() if valid_start < len(trading_dates) else _date_min
        _default_pick = _date_max
        pit_date_picked = st.date_input(
            "Snapshot date",
            value=_default_pick,
            min_value=_min_pick,
            max_value=_date_max,
            key="pit_date",
        )

    # date_input can return a date OR a tuple if range mode is on (we use
    # single-date mode, so it's always a single date). Coerce to Timestamp
    # for downstream _pit_corr (which serialises via isoformat()) — and
    # snap to the NEAREST trading day in case the user picks a weekend
    # / market holiday.
    if isinstance(pit_date_picked, tuple):
        pit_date_picked = pit_date_picked[0] if pit_date_picked else _default_pick
    _picked_ts = pd.Timestamp(pit_date_picked)
    _date_idx = returns.index.searchsorted(_picked_ts, side="right") - 1
    _date_idx = max(0, min(_date_idx, len(returns.index) - 1))
    pit_date = returns.index[_date_idx]
    if pit_date.date() != pit_date_picked:
        st.caption(
            f":material/info: Snapped to nearest trading day "
            f"{pit_date.strftime('%Y-%m-%d')} "
            f"(you picked {pit_date_picked.isoformat()} — weekend/holiday)."
        )

    with st.spinner("Computing snapshot correlation..."):
        pit_corr = _pit_corr(
            returns, returns_cache_key, pit_date.isoformat(), pit_window, pit_method,
        )
    if pit_corr.empty:
        st.warning("Not enough data for the selected date and window size.")
        return

    # use_clustering_order and leaf_order used to be script-level globals set
    # inside `with tab_corr:`. Now that the heatmap is also a fragment, they
    # aren't visible at module scope — read from session_state + load dendrogram
    # directly (load_dendrogram_order is @st.cache_data, so it's free).
    _use_clustering_order = bool(st.session_state.get("mo_corr_reorder", True))
    _leaf_order = load_dendrogram_order()
    if _use_clustering_order and _leaf_order is not None:
        pit_valid = [t for t in _leaf_order if t in pit_corr.columns]
        pit_display = pit_corr.loc[pit_valid, pit_valid] if pit_valid else pit_corr
    else:
        pit_display = pit_corr

    pit_mask = np.triu(np.ones(pit_display.shape, dtype=bool), k=1)
    pit_vals = pit_display.values[pit_mask]
    pit_vals = pit_vals[~np.isnan(pit_vals)]

    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric(
        f"{_cap(_active_universe, 'items_label', 'Tickers')} in Window",
        len(pit_display),
    )
    pm2.metric("Mean Corr", f"{np.mean(pit_vals):.4f}")
    pm3.metric("Median Corr", f"{np.median(pit_vals):.4f}")
    pm4.metric("Std Dev", f"{np.std(pit_vals):.4f}")

    render_matrix_heatmap(
        pit_display,
        chart_id="mo_pit_heatmap",
        filename_base="pit_correlation",
        title_key="mo_pit_heatmap",
        default_title="Point-in-Time Correlation",
        zmin=-1.0, zmax=1.0, diverging=True,
        height=_heatmap_height(min(len(pit_display), 200)),
        hover_label="corr",
        colorbar_title="Corr",
    )


@st.fragment
def _render_rolling_pair() -> None:
    """Rolling Analysis → Pair Correlation sub-tab. Owns pair_a/pair_b
    selectors + the cross-page "Open in Pair Analysis" button. The nav
    button uses `st.rerun(scope="app")` because the default `scope="fragment"`
    wouldn't actually leave this sub-tab."""
    # Use the same session_state keys as the Pair Analysis page
    # (pa_ticker_a / pa_ticker_b). Previously this sub-tab maintained its
    # own pair_a / pair_b state — two sources of truth for the same
    # concept, picking a pair here didn't carry over to Pair Analysis.
    # Audit item A5. Cross-page nav buttons (_open_pair_analysis_button)
    # also write to pa_ticker_a/b, so picks made in either view are now
    # always synced.
    ticker_list = sorted(returns.columns.tolist())
    if (
        "pa_ticker_a" not in st.session_state
        or st.session_state["pa_ticker_a"] not in ticker_list
    ):
        st.session_state["pa_ticker_a"] = ticker_list[0] if ticker_list else ""
    if (
        "pa_ticker_b" not in st.session_state
        or st.session_state["pa_ticker_b"] not in ticker_list
    ):
        st.session_state["pa_ticker_b"] = (
            ticker_list[1] if len(ticker_list) > 1 else (ticker_list[0] if ticker_list else "")
        )
    pc1, pc2 = st.columns(2)
    with pc1:
        pair_a = st.selectbox(f"{_item_rc} A", ticker_list, key="pa_ticker_a")
    with pc2:
        pair_b = st.selectbox(f"{_item_rc} B", ticker_list, key="pa_ticker_b")

    if pair_a and pair_b and pair_a != pair_b:
        # When window_type == "ewm", convert α → span (pandas formula:
        # span = 2/α - 1). Otherwise pass None and the underlying
        # compute_rolling_pair_correlation ignores ewm_span.
        _ewm_span = None
        if rc_window_type == "ewm":
            _alpha = float(st.session_state.get("rc_ewm_alpha", 0.05))
            _ewm_span = max(2, int((2.0 / _alpha) - 1))
        pair_corr = _compute_pair(
            returns, returns_cache_key, pair_a, pair_b,
            rc_window, rc_method, rc_window_type,
            ewm_span=_ewm_span,
        )

        fig_pair = go.Figure()
        fig_pair.add_hline(y=0, line_dash="dot", line_color=get_colors()["muted"], opacity=0.5)
        fig_pair.add_trace(go.Scatter(
            x=pair_corr.index, y=pair_corr.clip(lower=0),
            mode="lines", line=dict(width=0), showlegend=False,
            fill="tozeroy", fillcolor=get_colors()["positive"],
        ))
        fig_pair.add_trace(go.Scatter(
            x=pair_corr.index, y=pair_corr.clip(upper=0),
            mode="lines", line=dict(width=0), showlegend=False,
            fill="tozeroy", fillcolor=get_colors()["negative"],
        ))
        fig_pair.add_trace(go.Scatter(
            x=pair_corr.index, y=pair_corr.values,
            mode="lines", name=f"{pair_a} / {pair_b}",
            line=dict(color=get_colors()["primary"], width=1.8),
        ))

        _valid = pair_corr.dropna()
        if not _valid.empty:
            draw_event_markers(fig_pair, show_defaults, custom_events,
                               _valid.index.min(), _valid.index.max())
        apply_chart_style(fig_pair, height=420,
                          yaxis_title=f"{rc_method.title()} Correlation",
                          yaxis=dict(range=[-1.05, 1.05], gridcolor="rgba(141,153,174,0.15)"),
                          showlegend=False)
        render_chart(fig_pair, chart_id="mo_pair_corr", filename_base="pair_correlation",
                     title_key="mo_pair_corr", default_title="Pair Rolling Correlation")

        if _cap(_active_universe, 'has_pair_trading', True):
            _open_pair_analysis_button(pair_a, pair_b, key="pair_deep_dive")

        # Two normalized price lines (matches original behaviour).
        if pair_a in prices_window.columns and pair_b in prices_window.columns:
            pa = prices_window[pair_a] / prices_window[pair_a].iloc[0] * 100
            pb = prices_window[pair_b] / prices_window[pair_b].iloc[0] * 100
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(x=pa.index, y=pa, name=pair_a,
                                            line=dict(color=get_colors()["primary"], width=2)))
            fig_spread.add_trace(go.Scatter(x=pb.index, y=pb, name=pair_b,
                                            line=dict(color=get_colors()["secondary"], width=2)))
            apply_chart_style(fig_spread, height=300, yaxis_title="Normalized (100)")
            render_chart(fig_spread, chart_id="mo_pair_spread", filename_base="pair_spread",
                         title_key="mo_pair_spread", default_title="Pair Price Spread")
    elif pair_a == pair_b:
        st.info(f"Select two different {_items_rc.lower()}.")


# ══════════════════════════════════════════════════════════════════════════════
# Page config & global styling
# ══════════════════════════════════════════════════════════════════════════════

# ── Universe initialisation MUST run before st.set_page_config so the
# browser tab title reflects the active universe on first paint. We seed
# session_state from the DASHBOARD_UNIVERSE env var (falling back to bist),
# clamped to whichever universes have populated artifacts on disk.
_AVAIL_UNIVERSES = available_universes()
_AVAIL_KEYS      = [u.key for u in _AVAIL_UNIVERSES] or ["bist"]
_BOOT_KEY        = os.environ.get("DASHBOARD_UNIVERSE", "bist")
if _BOOT_KEY not in _AVAIL_KEYS:
    _BOOT_KEY = _AVAIL_KEYS[0]
# Defensive: HF Spaces health-probes and reconnecting browser tabs can invoke
# this script before Streamlit has a full session context. Touching
# session_state then raises "Tried to use SessionInfo before it was
# initialized". Fall back to the boot key without crashing — the user's real
# request will re-run the script with a proper session attached.
try:
    if "universe" not in st.session_state or st.session_state["universe"] not in _AVAIL_KEYS:
        st.session_state["universe"] = _BOOT_KEY
    _active_universe_key = st.session_state["universe"]
except Exception:  # noqa: BLE001 — SessionInfo not yet initialised
    _active_universe_key = _BOOT_KEY

_active_universe = get_universe(_active_universe_key)

st.set_page_config(
    page_title=f"StoNeCoAl — {_cap(_active_universe, 'short_label', 'BIST 100')}",
    page_icon="<svg xmlns='http://www.w3.org/2000/svg'/>",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

# ── Sidebar: dataset selector (top) + theme controls (existing)
with st.sidebar:
    if len(_AVAIL_UNIVERSES) > 1:
        st.markdown("**Dataset**")
        st.selectbox(
            "Universe",
            _AVAIL_KEYS,
            format_func=lambda k: get_universe(k).label,
            key="universe",  # bound directly to session_state; Streamlit reruns on change
            label_visibility="collapsed",
        )
        # Re-read after the selectbox in case the user just changed it.
        _active_universe = get_universe(st.session_state["universe"])
        st.caption(_cap(_active_universe, 'description', ''))
        st.markdown("---")
    elif _AVAIL_UNIVERSES:
        # Single universe present — show as a static caption, no selector clutter.
        st.markdown(f"**Dataset:** {_AVAIL_UNIVERSES[0].short_label}")
        st.caption(_AVAIL_UNIVERSES[0].description)
        st.markdown("---")
    render_theme_sidebar()

# ══════════════════════════════════════════════════════════════════════════════
# Top Header & Navigation
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"<div style='display:flex; align-items:center; gap:12px; padding:0; margin:0;'>"
    f"<span style='font-size:1.3rem; font-weight:800; letter-spacing:-0.02em; "
    f"color:#2B2D42;'>StoNeCoAl</span>"
    f"<span style='font-size:0.72rem; color:#8D99AE; letter-spacing:0.06em;'>"
    f"{_cap(_active_universe, 'short_label', 'BIST 100').upper()} NETWORK ANALYSIS</span></div>",
    unsafe_allow_html=True,
)

# Handle deferred navigation from cross-page jump buttons
if st.session_state.pop("_goto_pair_analysis", False):
    st.session_state["nav_page"] = "Pair Analysis"
if st.session_state.pop("_goto_cross_market", False):
    st.session_state["nav_page"] = "Cross-Market"

# Nav label for the overview page is domain-aware: "Market Overview" reads
# wrong when the active universe is EEG (no market), so non-finance domains
# get "Network Overview". Used for both the segmented-control label AND the
# session_state value, so the two stay in sync across universe switches.
_overview_label = (
    "Market Overview"
    if _cap(_active_universe, 'domain', 'finance') == "finance"
    else "Network Overview"
)

# Nav order (Phase 2 mutable-candy): foreground the project's strongest
# existing content — the cross-market BIST↔S&P comparison — as the FIRST
# nav option for finance universes. Demo and grading first-60-seconds land
# on this page rather than a coverage chart.
# EEG keeps its single-page Network Overview (no Cross-Market, no pair trading).
_eligible_for_cross_market = _cap(_active_universe, 'eligible_for_cross_market', True)
_nav_options: list[str] = []
if _eligible_for_cross_market:
    _nav_options.append("Cross-Market")
_nav_options.append(_overview_label)
if _cap(_active_universe, 'has_pair_trading', True):
    _nav_options.append("Pair Analysis")

# Default landing: Cross-Market for finance, the overview otherwise.
_default_nav = "Cross-Market" if _eligible_for_cross_market else _overview_label

# Clamp stored nav_page to options the current universe supports (otherwise
# Streamlit would render the segmented_control with an out-of-set default and
# raise StreamlitValueAssignmentNotAllowedError). Also catches the case where
# the user switches universe — their old nav_page (e.g. "Market Overview"
# under BIST) won't be in the new universe's options ("Network Overview"
# under EEG), so it resets to the domain-appropriate default.
if st.session_state.get("nav_page") not in _nav_options:
    st.session_state["nav_page"] = _default_nav

# Sprint 2 PR-P: dropped `default=_default_nav` from this call. Streamlit
# emits a warning when a widget has BOTH `key=` (binding session_state)
# AND `default=` (passing a default) — the two can collide. The guard
# block above already sets `st.session_state["nav_page"]` to a valid
# option on fresh-session AND when the active universe changes, so the
# `default=` here was redundant and tripped the warning banner on every
# first paint.
_nav = st.segmented_control(
    "Navigate",
    _nav_options,
    key="nav_page",
    label_visibility="collapsed",
)

# ── Cross-Market route ───────────────────────────────────────────────────────
# This page reads from BOTH universes directly (not via current_universe()),
# so route it BEFORE the per-universe data loads.
if _nav == "Cross-Market":
    from cross_market import render as _render_xmarket
    _render_xmarket()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Shared Data Loading  (per active universe)
# ══════════════════════════════════════════════════════════════════════════════

adj_close = load_adj_close()
full_returns = load_log_returns()
min_date = adj_close.index.min().date()
max_date = adj_close.index.max().date()

# ── Pair Analysis route ──────────────────────────────────────────────────────
if _nav == "Pair Analysis":
    coverage_df = load_coverage()
    from pair_analysis import render as _render_pair
    _render_pair(adj_close, full_returns, coverage_df, min_date, max_date)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Market Overview — Inline Settings & Key Metrics
# ══════════════════════════════════════════════════════════════════════════════

pipe_meta = load_metadata()
market_summary = pipe_meta.get("market_summary", {})

_settings_col, m1, m2, m3, m4, m5 = st.columns([1, 1, 1, 1, 1, 1.5])

with _settings_col:
    # NOTE: st.popover does NOT accept width= in Streamlit 1.41.1 (kwarg
    # landed in ~1.42+). use_container_width=True still works (with a
    # deprecation warning) and is the only valid spelling for this pin.
    with st.popover("Settings", icon=":material/settings:", use_container_width=True):
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        # Streamlit 1.41+ rejects popovers nested inside other popovers
        # (StreamlitAPIException). Use st.expander here — visually similar,
        # nestable inside popovers.
        with st.expander("Data Freshness", icon=":material/info:"):
            fetch_meta = load_fetch_metadata()
            if fetch_meta:
                st.write(f"**Fetch:** {fetch_meta.get('timestamp', 'N/A')[:16]}")
                st.write(f"**Source:** {fetch_meta.get('source', 'N/A')}")
                st.write(f"**{_cap(_active_universe, 'items_label', 'Tickers')}:** {fetch_meta.get('ticker_count', 'N/A')}")
                if fetch_meta.get("failures"):
                    st.write(f"**Failures:** {len(fetch_meta['failures'])}")
            if _cap(_active_universe, 'has_validation_report', True):
                val_path = data_processed() / "validation_report.csv"
                if val_path.exists():
                    val_df = pd.read_csv(val_path)
                    n_pass = (val_df["status"] == "PASS").sum()
                    st.write(f"**Validation:** {n_pass}/{len(val_df)} passed")

if len(date_range) == 2:
    start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_dt, end_dt = pd.Timestamp(min_date), pd.Timestamp(max_date)

returns = full_returns.loc[start_dt:end_dt]
prices_window = adj_close.loc[start_dt:end_dt]
window_length = len(returns)
dynamic_min_periods = max(30, int(window_length * 0.6))

m1.metric(_cap(_active_universe, 'items_label', 'Tickers'), f"{returns.shape[1]}")
m2.metric(
    "Samples" if _cap(_active_universe, 'domain', 'finance') == "neuroscience" else "Trading Days",
    f"{returns.shape[0]:,}",
)
m3.metric("Avg Correlation", f"{market_summary.get('avg_pairwise_corr', 0):.4f}")
m4.metric("Median Correlation", f"{market_summary.get('median_pairwise_corr', 0):.4f}")
m5.metric("Date Range", f"{start_dt.strftime('%Y-%m')} to {end_dt.strftime('%Y-%m')}")

# Cheap deterministic cache key for @st.cache_data helpers. The actual
# DataFrame is passed underscore-prefixed (Streamlit skips hashing it);
# identity comes from this string. Universe + date endpoints + shape
# uniquely identifies any returns slice we feed the helpers. Replaces the
# previous `_returns_json = returns.to_json(...)` pattern that cost ~38 ms
# per script rerun on S&P-500 (10 MB JSON ser/de roundtrip).
returns_cache_key = (
    f"{_active_universe.key}:{start_dt.date().isoformat()}:"
    f"{end_dt.date().isoformat()}:{returns.shape[0]}x{returns.shape[1]}"
)


# ══════════════════════════════════════════════════════════════════════════════
# Hero strip — sector-recovery validation (Phase 2.4 mutable-candy)
# ══════════════════════════════════════════════════════════════════════════════
# The proposal's primary validation criterion was "the MST recovers known
# economic sectors." Show that result UP FRONT (above the coverage charts),
# so the demo grader sees it in the first second on this page.

if _cap(_active_universe, 'has_pair_trading', True):
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        _clusters_df = load_cluster_assignments()
        _hero_caption_universe = (
            "Borsa Istanbul" if _active_universe.key == "bist"
            else "S&P 500" if _active_universe.key == "sp500"
            else _active_universe.label
        )
        if not _clusters_df.empty and "sector" in _clusters_df.columns:
            _clusters_clean = _clusters_df.dropna(subset=["sector", "cluster_id"])
            _ari = adjusted_rand_score(_clusters_clean["sector"], _clusters_clean["cluster_id"])
            _nmi = normalized_mutual_info_score(
                _clusters_clean["sector"], _clusters_clean["cluster_id"]
            )
            with st.container(border=True):
                hero_c1, hero_c2, hero_c3 = st.columns([1, 1, 3])
                hero_c1.metric("Sector ARI", f"{_ari:.2f}",
                               help="Adjusted Rand Index between Ward clusters and official sectors. 0=random, 1=perfect.")
                hero_c2.metric("Sector NMI", f"{_nmi:.2f}",
                               help="Normalized Mutual Information between Ward clusters and official sectors.")
                hero_c3.markdown(
                    f"**Statistical clusters extracted from raw price correlations recover the "
                    f"official {_hero_caption_universe} sector classification with "
                    f"**ARI = {_ari:.2f}, NMI = {_nmi:.2f}** (Ward linkage on Mantegna distance, "
                    f"n_clusters = {_clusters_clean['cluster_id'].nunique()}). "
                    "Open *Clustering & Network* below for the MST and dendrogram."
                )
    except Exception:
        # Hero strip is decorative — never block the page if it errors.
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Market Overview — Sub-Tab Layout (Pairs & Dislocations gated by capability)
# ══════════════════════════════════════════════════════════════════════════════

_tab_labels = ["Data & Stats", "Correlation", "Clustering & Network", "Rolling Analysis"]
if _cap(_active_universe, 'has_pair_trading', True):
    _tab_labels.append("Pairs & Dislocations")
_tab_labels.append("EEE Analysis")
_tabs = st.tabs(_tab_labels)
_tab_by_label = dict(zip(_tab_labels, _tabs))
tab_data    = _tab_by_label["Data & Stats"]
tab_corr    = _tab_by_label["Correlation"]
tab_cluster = _tab_by_label["Clustering & Network"]
tab_rolling = _tab_by_label["Rolling Analysis"]
tab_pairs   = _tab_by_label.get("Pairs & Dislocations")   # None when EEG
tab_eee     = _tab_by_label["EEE Analysis"]


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Data & Stats
# ══════════════════════════════════════════════════════════════════════════════

with tab_data:

    # ── Section 1: Coverage & Normalized Prices ─────────────────────────────
    with st.container(border=True):
        if _cap(_active_universe, 'has_index_series', True):
            section_header(
                "Data Coverage & Price Performance",
                f"Left: per-{_cap(_active_universe, 'item_label', 'Ticker').lower()} data availability "
                f"(90% threshold). Right: all prices rebased to 100 — the bold black "
                f"line is {_cap(_active_universe, 'index_ticker', 'XU100')}.",
            )
        else:
            section_header(
                f"Data Coverage & {_cap(_active_universe, 'series_label', 'Log return')} Performance",
                f"Left: per-{_cap(_active_universe, 'item_label', 'Ticker').lower()} data availability. "
                f"Right: {_cap(_active_universe, 'series_label', 'Log return').lower()} time-series for a "
                f"representative subset of channels (first 30 s).",
            )

        col_left, col_right = st.columns(2)

        with col_left:
            coverage = load_coverage()
            fig_cov = px.bar(
                coverage.sort_values("coverage_pct"),
                x="coverage_pct", y="ticker", orientation="h",
                labels={"coverage_pct": "Coverage %", "ticker": ""},
                color="coverage_pct",
                color_continuous_scale=["#E63946", "#FF9F1C", "#2EC4B6"],
                range_color=[0.7, 1.0],
            )
            fig_cov.add_vline(x=0.90, line_dash="dash", line_color="#E63946",
                              annotation_text="90% threshold", annotation_font_size=10)
            apply_chart_style(fig_cov, height=max(400, len(coverage) * 8),
                              coloraxis_showscale=False,
                              margin=dict(l=60, r=10, t=10, b=30),
                              yaxis=dict(dtick=1, tickfont=dict(size=7)))
            render_chart(fig_cov, chart_id="mo_coverage", filename_base="data_coverage",
                         title_key="mo_coverage",
                         default_title=f"Data Coverage by {_cap(_active_universe, 'item_label', 'Ticker')}")

        with col_right:
            if _cap(_active_universe, 'has_index_series', True):
                # Financial universe: rebased prices + bold market-index overlay.
                norm_prices = prices_window.divide(prices_window.iloc[0]) * 100
                xu100 = load_xu100()
                _index_label = _cap(_active_universe, 'index_ticker', 'XU100')  # "XU100" / "^GSPC"
                if not xu100.empty:
                    xu100_window = xu100.loc[start_dt:end_dt]
                    if not xu100_window.empty:
                        norm_prices[_index_label] = xu100_window / xu100_window.iloc[0] * 100

                # Per-ticker series count. For large universes (S&P-500 = 485)
                # the per-ticker spaghetti chart serialises to ~30 MB on the wire
                # (485 traces × ~1500 dates × Plotly metadata), blowing the
                # browser-side ~20 MB WebSocket frame cap and killing the whole
                # page render. The spaghetti adds zero analytical value at that
                # density anyway — collapse to a 10/50/90 percentile envelope
                # plus the bold market-index overlay. BIST (~73 tickers) keeps
                # the individual lines.
                _ticker_cols = [c for c in norm_prices.columns if c != _index_label]
                _SPAGHETTI_MAX = 80
                fig_prices = go.Figure()
                if len(_ticker_cols) > _SPAGHETTI_MAX:
                    _per_ticker = norm_prices[_ticker_cols]
                    q10 = _per_ticker.quantile(0.10, axis=1)
                    q50 = _per_ticker.quantile(0.50, axis=1)
                    q90 = _per_ticker.quantile(0.90, axis=1)
                    _band_color = "rgba(141,153,174,0.18)"
                    fig_prices.add_trace(go.Scatter(
                        x=q90.index, y=q90.values,
                        mode="lines", line=dict(width=0, color=_band_color),
                        showlegend=False, hoverinfo="skip",
                    ))
                    fig_prices.add_trace(go.Scatter(
                        x=q10.index, y=q10.values,
                        mode="lines", line=dict(width=0, color=_band_color),
                        fill="tonexty", fillcolor=_band_color,
                        name="10–90% band", hoverinfo="skip",
                    ))
                    fig_prices.add_trace(go.Scatter(
                        x=q50.index, y=q50.values,
                        mode="lines",
                        line=dict(width=1.2, color=get_colors()["muted"]),
                        name=f"Median ({len(_ticker_cols)} {_cap(_active_universe, 'items_label', 'tickers').lower()})",
                        hovertemplate="Median: %{y:.1f}<extra></extra>",
                    ))
                    if _index_label in norm_prices.columns:
                        fig_prices.add_trace(go.Scatter(
                            x=norm_prices.index, y=norm_prices[_index_label].values,
                            mode="lines",
                            line=dict(width=3.0, color="#2B2D42"),
                            name=_index_label,
                            hovertemplate=f"{_index_label}: %{{y:.1f}}<extra></extra>",
                        ))
                    apply_chart_style(
                        fig_prices, height=max(400, len(coverage) * 8),
                        showlegend=True,
                        yaxis_title="Normalized Price (base=100)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    font=dict(size=10)),
                    )
                else:
                    for col in norm_prices.columns:
                        is_index = col == _index_label
                        fig_prices.add_trace(go.Scatter(
                            x=norm_prices.index, y=norm_prices[col], name=col,
                            mode="lines",
                            line=dict(
                                width=3.0 if is_index else 0.6,
                                color="#2B2D42" if is_index else get_colors()["muted"],
                            ),
                            opacity=1.0 if is_index else 0.35,
                            hovertemplate=f"{col}: %{{y:.1f}}<extra></extra>" if is_index else None,
                            hoverinfo="skip" if not is_index else None,
                        ))
                    apply_chart_style(fig_prices, height=max(400, len(coverage) * 8),
                                      showlegend=False, yaxis_title="Normalized Price (base=100)")
                render_chart(fig_prices, chart_id="mo_prices", filename_base="normalized_prices",
                             title_key="mo_prices", default_title="Normalized Price Performance")
            else:
                # Non-financial universe (EEG): stacked voltage time-series of
                # 10 evenly-spaced channels for the first 30 seconds. Each trace
                # is vertically offset so overlaps stay readable.
                n_show = min(10, prices_window.shape[1])
                if n_show == 0:
                    st.info("No channels available for the active universe.")
                else:
                    sample_rate_hz = 160                              # PhysioNet EEG default
                    n_samples = min(int(30 * sample_rate_hz), len(prices_window))
                    channel_idx = np.linspace(0, prices_window.shape[1] - 1, n_show, dtype=int)

                    fig_volt = go.Figure()
                    cumulative_offset = 0.0
                    for i, col_idx in enumerate(channel_idx):
                        ch = prices_window.columns[col_idx]
                        series = prices_window.iloc[:n_samples, col_idx].values
                        std = float(np.nanstd(series)) if np.isfinite(series).any() else 1.0
                        spacing = max(std * 4.0, 1.0)
                        fig_volt.add_trace(go.Scatter(
                            x=np.arange(n_samples) / sample_rate_hz,
                            y=series + cumulative_offset,
                            mode="lines", name=ch,
                            line=dict(width=0.8, color=SECTOR_PALETTE[i % len(SECTOR_PALETTE)]),
                            hovertemplate=f"{ch}: %{{y:.2f}} {_cap(_active_universe, 'series_units', '')}<extra></extra>",
                        ))
                        cumulative_offset += spacing
                    apply_chart_style(
                        fig_volt, height=max(400, n_show * 38),
                        xaxis_title="Time (seconds)",
                        yaxis_title=(
                            f"{_cap(_active_universe, 'series_label', 'Log return')} "
                            f"({_cap(_active_universe, 'series_units', '')}) — stacked"
                        ),
                        showlegend=True,
                        legend=dict(orientation="v", yanchor="top", y=1.0,
                                    xanchor="left", x=1.02, font=dict(size=9)),
                    )
                    render_chart(
                        fig_volt, chart_id="mo_voltage",
                        filename_base="voltage_trace",
                        title_key="mo_voltage",
                        default_title=(
                            f"{_cap(_active_universe, 'series_label', 'Log return')} Time-Series "
                            f"(first {n_samples / sample_rate_hz:.0f}s, "
                            f"{n_show} sample channels)"
                        ),
                    )

    # ── Section 2: Descriptive Stats & Distribution ─────────────────────────
    with st.container(border=True):
        _is_finance     = _cap(_active_universe, 'domain', 'finance') == "finance"
        _item_label     = _cap(_active_universe, 'item_label', 'Ticker')
        _items_label    = _cap(_active_universe, 'items_label', 'Tickers')
        _series_label   = _cap(_active_universe, 'series_label', 'Log return')
        _series_units   = _cap(_active_universe, 'series_units', '')
        _series_axis    = f"{_series_label} ({_series_units})" if _series_units else _series_label

        if _is_finance:
            section_header(
                "Descriptive Statistics & Returns",
                f"Left: per-{_item_label.lower()} risk-return metrics from daily log returns. "
                f"Right: histogram for a selected {_item_label.lower()} — look for fat tails and skewness.",
            )
        else:
            section_header(
                f"Descriptive Statistics & {_series_label} Distribution",
                f"Left: per-{_item_label.lower()} distribution shape (skewness, kurtosis, extremes). "
                f"Annualised return / volatility columns are hidden — they're a financial-only construct. "
                f"Right: amplitude histogram for a selected {_item_label.lower()}.",
            )

        col_stats, col_hist = st.columns([3, 2])

        with col_stats:
            summary = load_summary_stats()
            if _is_finance:
                display_cols = [
                    "ticker", "count", "annualized_return", "annualized_vol",
                    "skewness", "kurtosis", "min_return", "max_return",
                ]
                float4_cols = ["annualized_return", "annualized_vol", "min_return", "max_return"]
            else:
                # Non-financial (EEG): drop "annualized_*" — meaningless on
                # 160 Hz sampled voltages. Keep distribution-shape metrics +
                # extremes.
                display_cols = [
                    "ticker", "count", "skewness", "kurtosis", "min_return", "max_return",
                ]
                float4_cols = ["min_return", "max_return"]
            display_df = summary[[c for c in display_cols if c in summary.columns]].copy()

            # Sort selector — user-controlled. Default mirrors the upstream
            # `compute_descriptive_stats` order (annualized_vol DESC for
            # finance / count DESC otherwise). Sort BEFORE float-formatting
            # so the ordering uses raw numeric values, not lexical strings.
            _sort_label_map = {
                "ticker":            _item_label,
                "count":             "Trading days",
                "annualized_return": "Annualized return",
                "annualized_vol":    "Annualized volatility",
                "skewness":          "Skewness",
                "kurtosis":          "Excess kurtosis",
                "min_return":        "Worst daily log return",
                "max_return":        "Best daily log return",
            }
            _sort_cols = list(display_df.columns)
            _default_sort_key = "annualized_vol" if "annualized_vol" in _sort_cols else "count"
            _sort_default_idx = _sort_cols.index(_default_sort_key) if _default_sort_key in _sort_cols else 0
            _sc1, _sc2 = st.columns([3, 1])
            with _sc1:
                _sort_col = st.selectbox(
                    "Sort by", _sort_cols,
                    index=_sort_default_idx,
                    format_func=lambda c: _sort_label_map.get(c, c),
                    key="mo_stats_sort_col",
                )
            with _sc2:
                _sort_dir = st.selectbox(
                    "Order", ["Descending", "Ascending"], index=0, key="mo_stats_sort_dir",
                )
            display_df = display_df.sort_values(
                _sort_col, ascending=(_sort_dir == "Ascending"),
            ).reset_index(drop=True)

            # Format floats AFTER sorting so the sort itself uses raw numerics.
            for c in float4_cols:
                if c in display_df.columns:
                    display_df[c] = display_df[c].map(lambda x: f"{x:.4f}")
            for c in ["skewness", "kurtosis"]:
                if c in display_df.columns:
                    display_df[c] = display_df[c].map(lambda x: f"{x:.2f}")

            # column_config: descriptive display names + help tooltips. Keeps
            # the internal raw column keys ("annualized_vol" etc.) so the
            # sort selector still works against the underlying DataFrame.
            _min_label = "Worst daily log return"
            _max_label = "Best daily log return"
            if not _is_finance and _series_units:
                _min_label += f" ({_series_units})"
                _max_label += f" ({_series_units})"
            st.dataframe(
                display_df, use_container_width=True, height=420, hide_index=True,
                column_config={
                    "ticker":            st.column_config.TextColumn(_item_label),
                    "count":             st.column_config.NumberColumn(
                        "Trading days",
                        help="Non-NaN observations in the date window",
                    ),
                    "annualized_return": st.column_config.TextColumn(
                        "Annualized return",
                        help="Daily mean × 252",
                    ),
                    "annualized_vol":    st.column_config.TextColumn(
                        "Annualized volatility",
                        help="Daily std × √252",
                    ),
                    "skewness":          st.column_config.TextColumn(
                        "Skewness",
                        help="Distribution asymmetry. >0 = right tail (occasional large positive returns)",
                    ),
                    "kurtosis":          st.column_config.TextColumn(
                        "Excess kurtosis",
                        help="Fat-tail measure. >0 = heavier tails than a normal distribution",
                    ),
                    "min_return":        st.column_config.TextColumn(
                        _min_label,
                        help="Largest single-day decline (log scale)",
                    ),
                    "max_return":        st.column_config.TextColumn(
                        _max_label,
                        help="Largest single-day gain (log scale)",
                    ),
                },
            )

        with col_hist:
            selected_ticker = st.selectbox(
                _item_label, sorted(returns.columns.tolist()), key="mo_hist_pick",
            )
            if selected_ticker:
                ticker_returns = returns[selected_ticker].dropna()
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=ticker_returns, nbinsx=80,
                    marker_color=get_colors()["primary"], opacity=0.75,
                    hovertemplate=f"{_series_label}: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>",
                ))
                _mean_r = ticker_returns.mean()
                fig_hist.add_vline(x=_mean_r, line_dash="dash", line_color=get_colors()["secondary"],
                                   annotation_text=f"Mean: {_mean_r:.4f}", annotation_font_size=10)
                apply_chart_style(
                    fig_hist, height=420,
                    xaxis_title=("Daily Log Return" if _is_finance else _series_axis),
                    yaxis_title="Frequency", showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                _hist_title = "Return Distribution" if _is_finance else f"{_series_label} Distribution"
                render_chart(fig_hist, chart_id="mo_hist", filename_base="distribution",
                             title_key="mo_hist", default_title=_hist_title)

    # ── Section 3: Return Anomalies (financial universes only) ──────────────
    if _cap(_active_universe, 'has_anomaly_detection', True):
      with st.container(border=True):
          section_header(
              "Return Anomalies",
              "Days where a ticker's daily log return exceeded the configured "
              "threshold (default ±30%) — usually corporate actions or data glitches.",
          )

          anomalies = load_anomalies()
          if anomalies.empty:
              st.success("No anomalies flagged in the current data window.")
          else:
              anom_view = anomalies.copy()
              if "date" in anom_view.columns:
                  anom_view["date"] = pd.to_datetime(anom_view["date"])
              anom_view["abs_return"] = anom_view["return_value"].abs()

              col_table, col_scatter = st.columns([2, 3])

              with col_table:
                  disp = anom_view.copy()
                  if "date" in disp.columns:
                      disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
                  disp = disp[["date", "ticker", "return_value", "abs_return"]]
                  st.dataframe(
                      disp.sort_values("abs_return", ascending=False),
                      use_container_width=True, hide_index=True, height=320,
                      column_config={
                          "return_value": st.column_config.NumberColumn(format="%.4f"),
                          "abs_return": st.column_config.NumberColumn("|return|", format="%.4f"),
                      },
                  )

              with col_scatter:
                  colors_now = get_colors()
                  fig_anom = go.Figure()
                  neg = anom_view[anom_view["return_value"] < 0]
                  pos = anom_view[anom_view["return_value"] >= 0]
                  if not neg.empty:
                      fig_anom.add_trace(go.Scatter(
                          x=neg["date"], y=neg["ticker"], mode="markers",
                          marker=dict(
                              size=8 + 30 * neg["abs_return"].clip(0, 1),
                              color=colors_now["secondary"], opacity=0.8,
                              symbol="triangle-down",
                              line=dict(width=0.5, color="#fff"),
                          ),
                          name="negative",
                          hovertemplate=(
                              "%{y} on %{x|%Y-%m-%d}<br>"
                              "return = %{customdata:.4f}<extra></extra>"
                          ),
                          customdata=neg["return_value"],
                      ))
                  if not pos.empty:
                      fig_anom.add_trace(go.Scatter(
                          x=pos["date"], y=pos["ticker"], mode="markers",
                          marker=dict(
                              size=8 + 30 * pos["abs_return"].clip(0, 1),
                              color=colors_now["primary"], opacity=0.8,
                              symbol="triangle-up",
                              line=dict(width=0.5, color="#fff"),
                          ),
                          name="positive",
                          hovertemplate=(
                              "%{y} on %{x|%Y-%m-%d}<br>"
                              "return = %{customdata:.4f}<extra></extra>"
                          ),
                          customdata=pos["return_value"],
                      ))
                  apply_chart_style(
                      fig_anom, height=320,
                      xaxis_title="Date", yaxis_title="Ticker",
                      yaxis=dict(tickfont=dict(size=9)),
                      showlegend=True,
                  )
                  render_chart(
                      fig_anom, chart_id="mo_anomalies",
                      filename_base="anomaly_timeline",
                      title_key="mo_anom",
                      default_title=f"Anomaly Timeline ({len(anom_view)} flagged events)",
                  )

    # ── Section 9: Universe-wide correlation summary ────────────────────────
    with st.container(border=True):
        section_header(
            "Market Summary" if _cap(_active_universe, 'domain', 'finance') == "finance"
            else "Network Summary"
        )
        if market_summary:
            cols = st.columns(5)
            cols[0].metric("Avg Pairwise Corr", f"{market_summary.get('avg_pairwise_corr', 0):.4f}")
            cols[1].metric("Median", f"{market_summary.get('median_pairwise_corr', 0):.4f}")
            cols[2].metric("Std Dev", f"{market_summary.get('std_pairwise_corr', 0):.4f}")
            cols[3].metric("Min", f"{market_summary.get('min_pairwise_corr', 0):.4f}")
            cols[4].metric("Max", f"{market_summary.get('max_pairwise_corr', 0):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Correlation
# ══════════════════════════════════════════════════════════════════════════════

with tab_corr:
    # Both sub-tabs are @st.fragment-scoped. Widget changes inside one
    # sub-tab (heat_method, pit_window, pit_date, etc.) only re-run that
    # fragment — the other sub-tab, the other top-level tabs, and the
    # rest of the script are skipped. Cross-tab state (heat_method,
    # use_clustering_order) flows via st.session_state.
    _corr_heatmap_tab, _corr_pit_tab = st.tabs(["Heatmap", "Point-in-Time Snapshot"])
    with _corr_heatmap_tab:
        _render_correlation_heatmap()
    with _corr_pit_tab:
        with st.container(border=True):
            _render_pit_correlation()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Clustering & Network
# ══════════════════════════════════════════════════════════════════════════════

with tab_cluster:

    # ── Section 4: Dendrogram & Cluster Assignments ─────────────────────────
    with st.container(border=True):
        _items_cl   = _cap(_active_universe, 'items_label', 'Tickers')
        _item_cl    = _cap(_active_universe, 'item_label', 'Ticker')
        _sector_cl  = _cap(_active_universe, 'sector_label', 'Sector')
        _series_cl  = _cap(_active_universe, 'series_label', 'log return').lower()
        section_header(
            f"Hierarchical Clustering & {_sector_cl} Validation",
            f"Dendrogram built from d = sqrt(2(1-rho)). {_items_cl} merging at lower heights "
            f"have more similar {_series_cl} dynamics. Validation metrics (ARI, NMI) measure "
            f"how well statistical clusters align with the universe's {_sector_cl.lower()} labels.",
        )

        col_dendro, col_clusters = st.columns([3, 2])

        with col_dendro:
            Z_loaded, labels_loaded = load_linkage()
            if Z_loaded is not None:
                n_leaves = len(labels_loaded)
                fig_dendro = ff.create_dendrogram(
                    np.eye(n_leaves),
                    orientation="bottom",
                    labels=labels_loaded,
                    linkagefun=lambda x: Z_loaded,
                )
                for trace in fig_dendro.data:
                    trace.update(line=dict(color=get_colors()["primary"], width=1.2))
                # Hide per-leaf labels when there are too many to read; the
                # rendering speed jump is dramatic on the 485-leaf S&P tree.
                _show_leaf_labels = n_leaves <= 100
                _leaf_tickfont = 7 if n_leaves <= 100 else 1  # plotly requires size>=1
                apply_chart_style(fig_dendro,
                    height=500,
                    margin=dict(l=10, r=10, t=10, b=100 if _show_leaf_labels else 30),
                    xaxis=dict(
                        tickfont=dict(size=_leaf_tickfont),
                        tickangle=-90,
                        showticklabels=_show_leaf_labels,
                    ),
                    yaxis_title="Distance",
                )
                if not _show_leaf_labels:
                    st.caption(
                        f":material/info: Per-leaf labels hidden on this "
                        f"{n_leaves}-leaf dendrogram for legibility; cluster "
                        "membership is in the table on the right."
                    )
                render_chart(fig_dendro, chart_id="mo_dendrogram", filename_base="dendrogram",
                             title_key="mo_dendrogram", default_title="Hierarchical Clustering")
            else:
                st.info("Run the clustering pipeline to generate the dendrogram.")

        with col_clusters:
            cluster_df = load_cluster_assignments()
            if not cluster_df.empty:
                n_clusters = cluster_df["cluster_id"].nunique()
                st.metric("Clusters Found", n_clusters)

                # Friend's UX feedback (this round): the raw cluster
                # assignments table (cluster_id × ticker × sector), the
                # cluster-vs-sector crosstab, and the ARI/NMI/Sectors
                # metric strip were all visual clutter. "Cluster Purity
                # is enough" — keep just (1) the universe-appropriate
                # sanity-check banners (one-line green/yellow status) and
                # (2) the per-cluster purity table below them.
                if "sector" in cluster_df.columns:
                    # Universe-appropriate sanity-check banners. The
                    # groups come from the Universe.sanity_check_groups
                    # dict in app/universe_registry.py — BIST checks
                    # the banking sector; S&P checks mega-cap tech;
                    # EEG checks central-motor / occipital / prefrontal
                    # electrode triples.
                    for group_label, members in (_cap(_active_universe, 'sanity_check_groups', None) or {}).items():
                        present = cluster_df[cluster_df["ticker"].isin(members)]
                        if present.empty:
                            continue
                        present_clusters = present["cluster_id"].unique()
                        names = present["ticker"].tolist()
                        if len(names) > 12:
                            names_str = ", ".join(names[:12]) + f", … (+{len(names) - 12} more)"
                        else:
                            names_str = ", ".join(names)
                        if len(present_clusters) == 1:
                            st.success(
                                f"**{group_label} sanity check passed.** All "
                                f"{len(present)} members ({names_str}) are in "
                                f"Cluster {present_clusters[0]}."
                            )
                        else:
                            st.warning(
                                f"**{group_label}:** {len(present)} members span "
                                f"{len(present_clusters)} clusters "
                                f"({', '.join(str(c) for c in sorted(present_clusters))}). "
                                f"Members: {names_str}"
                            )

                    st.markdown("**Cluster Purity**")
                    st.caption(
                        f"Each cluster's dominant {_sector_cl.lower()} and the fraction "
                        f"of members sharing that {_sector_cl.lower()}. Purity = 1.0 means "
                        f"every {_item_cl.lower()} in the cluster shares the same "
                        f"{_sector_cl.lower()}."
                    )
                    purity_rows = []
                    for cid, grp in cluster_df.groupby("cluster_id"):
                        sector_counts = grp["sector"].value_counts()
                        dominant = sector_counts.index[0]
                        purity = sector_counts.iloc[0] / len(grp)
                        purity_rows.append({
                            "Cluster": cid,
                            "Size": len(grp),
                            f"Dominant {_sector_cl}": dominant,
                            "Purity": f"{purity:.2f}",
                            "Members": ", ".join(sorted(grp["ticker"].tolist())),
                        })
                    purity_df = pd.DataFrame(purity_rows)
                    st.dataframe(purity_df, use_container_width=True, hide_index=True)
            else:
                st.info("Run the clustering pipeline to see cluster assignments.")

    # ── Section 5: MST Network ──────────────────────────────────────────────
    with st.container(border=True):
        _items_mst    = _cap(_active_universe, 'items_label', 'Tickers')
        _sector_mst   = _cap(_active_universe, 'sector_label', 'Sector')
        _domain_mst   = _cap(_active_universe, 'domain', 'finance')
        _bridge_scope = "across the market" if _domain_mst == "finance" else "across the network"
        section_header(
            "Minimum Spanning Tree",
            f"The MST reveals the backbone correlation structure. Nodes are colored by "
            f"{_sector_mst.lower()} and sized by degree. Hub {_items_mst.lower()} act as bridges "
            f"{_bridge_scope}.",
        )

        mst_edges = load_mst_edges()
        mst_metrics = load_mst_metrics()

        if not mst_edges.empty and not mst_metrics.empty and HAS_NETWORKX:
            # MST gets full page width (no col split). Hub table moves into
            # an expander below — the cluttered "Quick Jump to Pair Analysis"
            # mini-widget that used to live alongside the table is removed
            # entirely; the same nav is available from the Pair Analysis page
            # and the Pairs & Dislocations tab.
            G = nx.Graph()
            sector_map = dict(zip(mst_metrics["ticker"], mst_metrics["sector"]))
            degree_map = dict(zip(mst_metrics["ticker"], mst_metrics["degree"]))

            for _, row in mst_edges.iterrows():
                G.add_edge(row["source"], row["target"], weight=row["distance"])

            # `_mst_layout` is @st.cache_data — first compute is ~400 ms
            # on S&P, subsequent universe re-renders are instant.
            pos = _mst_layout(
                mst_edges,
                f"{_active_universe.key}:mst:{len(mst_edges)}",
            )

            edge_traces = []
            for u, v, d in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_traces.append(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode="lines",
                    line=dict(width=1.8, color="#A0A8B8"),
                    hoverinfo="text",
                    hovertext=f"{u} — {v}  (d = {d['weight']:.3f})",
                    showlegend=False,
                ))

            sectors = sorted(set(sector_map.values()) - {None, np.nan})
            color_map = {s: SECTOR_PALETTE[i % len(SECTOR_PALETTE)]
                         for i, s in enumerate(sectors)}

            node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                sec = sector_map.get(node, "Unknown")
                deg = degree_map.get(node, 1)
                node_text.append(f"<b>{node}</b><br>{_sector_mst}: {sec}<br>Degree: {deg}")
                node_color.append(color_map.get(sec, get_colors()["muted"]))
                node_size.append(14 + deg * 6)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=[n for n in G.nodes()],
                textposition="top center",
                textfont=dict(size=9, color="#2B2D42"),
                hovertext=node_text, hoverinfo="text",
                marker=dict(
                    size=node_size, color=node_color,
                    line=dict(width=2, color="white"),
                ),
                showlegend=False,
            )

            fig_mst = go.Figure(data=edge_traces + [node_trace])
            apply_chart_style(fig_mst,
                showlegend=True, height=750,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                           scaleanchor="x", scaleratio=1),
                legend=dict(font=dict(size=9), orientation="v",
                            yanchor="top", y=0.99, xanchor="left", x=0.01,
                            bgcolor="rgba(255,255,255,0.85)", borderwidth=1,
                            bordercolor="#e2e6ee"),
            )
            for sec in sectors:
                fig_mst.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=color_map[sec]),
                    name=sec, showlegend=True,
                ))
            render_chart(fig_mst, chart_id="mo_mst", filename_base="mst_network",
                         title_key="mo_mst", default_title="Minimum Spanning Tree")

            # Hub table behind an expander — was always-on in the right column
            # of a [3,2] split, now hidden by default to let the MST breathe.
            with st.expander(f"Hub {_items_mst} (by degree)", expanded=False):
                _item_mst = _cap(_active_universe, 'item_label', 'Ticker')
                display_metrics = mst_metrics.copy()
                display_metrics["betweenness_centrality"] = display_metrics[
                    "betweenness_centrality"
                ].map(lambda x: f"{x:.4f}")
                st.dataframe(
                    display_metrics, use_container_width=True, hide_index=True,
                    column_config={
                        "ticker":                 st.column_config.TextColumn(_item_mst),
                        "sector":                 st.column_config.TextColumn(_sector_mst),
                        "degree":                 st.column_config.NumberColumn("Degree"),
                        "betweenness_centrality": st.column_config.TextColumn("Betweenness"),
                    },
                )

        elif not HAS_NETWORKX:
            st.warning("Install `networkx` to display the MST network graph (`pip install networkx`).")
        else:
            st.info("Run the clustering pipeline to generate the MST network.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Rolling Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab_rolling:

    _is_finance_rc = _cap(_active_universe, 'domain', 'finance') == "finance"
    _item_rc       = _cap(_active_universe, 'item_label', 'Ticker')
    _items_rc      = _cap(_active_universe, 'items_label', 'Tickers')
    _sector_rc     = _cap(_active_universe, 'sector_label', 'Sector')

    with st.container(border=True):
        section_header(
            "Rolling Correlation Analysis",
            "Track how pairwise correlations evolve over time. " + (
                "Spikes during crises indicate correlation regime shifts."
                if _is_finance_rc
                else "Spikes can mark regime shifts or transient synchronization events."
            ),
        )

        # st.form gates the 4 outer rolling widgets behind an explicit
        # "Recompute" submit button. Off-grid params (e.g. step=1, spearman,
        # window=504) cost up to 12 s on S&P; without the form, the user
        # paid that cost on every intermediate selectbox change. With the
        # form, widget changes accumulate locally, then a single submit
        # triggers one script rerun. Precomputed combos (window ∈ {60,120,
        # 252}, step=5, pearson, rolling) still load instantly because the
        # downstream `_use_precomputed_market` check hits the parquet cache.
        st.caption(
            ":material/info: Configure window / step / method, then click "
            "**Recompute** to refresh charts. Precomputed combos "
            "(window ∈ {60, 120, 252}, step=5, pearson, rolling) load instantly. "
            "**EWM α** only applies when *Window type = ewm*; otherwise ignored."
        )

        with st.form("rolling_params", border=False):
            # All 5 widgets always render. Earlier version conditionally
            # showed EWM α only AFTER a Recompute with window_type=ewm —
            # a two-click trap, because Streamlit forms don't propagate
            # in-form widget changes until submit (so the disabled-trick
            # doesn't work either). Always-visible α with clear `help=`
            # copy is cleaner: harmless to touch when not in ewm mode
            # (value is just ignored by the rolling/expanding paths).
            # Audit item A4.
            _form_cols = st.columns([2, 2, 2, 2, 2, 1.2])
            with _form_cols[0]:
                rc_window = int(st.number_input(
                    "Window (days)" if _is_finance_rc else "Window (samples)",
                    min_value=20, max_value=504, value=252, step=10,
                    key="rc_win",
                    help="Trading days in each rolling window. {60, 120, 252} hit the precomputed parquet (instant); other values compute on the fly.",
                ))
            with _form_cols[1]:
                rc_step = st.selectbox(
                    "Step", [1, 5, 21], index=1, key="rc_step",
                    format_func=lambda x: {1: "1 (daily)", 5: "5 (weekly)", 21: "21 (monthly)"}.get(x, str(x)),
                )
            with _form_cols[2]:
                rc_method = st.selectbox("Method", ["pearson", "spearman"], key="rc_method")
            with _form_cols[3]:
                rc_window_type = st.selectbox("Window type", ["rolling", "expanding", "ewm"], key="rc_wtype")
            with _form_cols[4]:
                rc_ewm_alpha = float(st.number_input(
                    "EWM α", min_value=0.01, max_value=0.5,
                    value=0.05, step=0.01, key="rc_ewm_alpha",
                    help="Exponential weighting decay (Pair Correlation sub-tab only, when Window type = ewm). α=0.05 ≈ span 39 days; α=0.1 ≈ span 19 days. Ignored for rolling/expanding.",
                ))
            with _form_cols[5]:
                # Vertical alignment hack: empty markdown matches the label
                # height of the selectboxes so the button aligns to their
                # input row, not their label row.
                st.markdown("&nbsp;", unsafe_allow_html=True)
                st.form_submit_button(
                    "Recompute", type="primary", use_container_width=True,
                )

        rc_expanding = rc_window_type == "expanding"
        # Event Markers popover lives OUTSIDE the form — popovers + forms
        # in the same row break Streamlit's column layout, and event-marker
        # toggles are cheap (no compute) so they don't need recompute gating.
        show_defaults, custom_events = event_marker_manager_ui("rc", min_date, max_date)

        _ra_market_label = "Market Overview" if _is_finance_rc else "Network Overview"
        tab_market, tab_pair, tab_sector = st.tabs(
            [_ra_market_label, "Pair Correlation", f"{_sector_rc} Breakdown"]
        )

        # ── Sub-Tab 1: Market correlation over time ─────────────────────────
        with tab_market:
            # Try precomputed parquet first (matches windows the pipeline
            # bakes for the demo). Fall back to on-the-fly compute when the
            # user picks parameters outside the precomputed grid.
            _precomputed_windows = {60, 120, 252}
            _use_precomputed_market = (
                rc_window in _precomputed_windows
                and rc_step == 5
                and rc_method == "pearson"
                and not rc_expanding
            )

            if _use_precomputed_market:
                market_stats = load_rolling_market_stats_precomputed(rc_window)
                if not market_stats.empty:
                    st.caption(
                        f"Reading precomputed `rolling_market_stats_w{rc_window}.parquet` "
                        "(window/step/method match the pipeline; `step=5`, `pearson`)."
                    )
                else:
                    with st.status("Computing rolling stats...", expanded=False) as _ms_st:
                        market_stats = _compute_market_stats(
                            returns, returns_cache_key, rc_window, rc_step, rc_method, rc_expanding,
                        )
                        _ms_st.update(label="Rolling stats ready", state="complete")
            else:
                # Off-grid params take 10-15 seconds on S&P-500 (Python loop
                # over ~260 window positions, each running .corr() on a
                # 252×485 slice). Make the wait explicit so the user knows
                # it's working, not frozen. The same message is shown on
                # both cache miss and cache hit (the cache layer would
                # short-circuit before the spinner shows up — Streamlit's
                # `with st.status` block only renders if the body actually
                # runs, but @st.cache_data wraps from the outside, so this
                # path enters the status block first then hits the cache).
                with st.status(
                    "Computing rolling stats (off-grid params — first run "
                    "takes 10-15 s on S&P; cached on subsequent reruns)...",
                    expanded=False,
                ) as _ms_st:
                    market_stats = _compute_market_stats(
                        returns, returns_cache_key, rc_window, rc_step, rc_method, rc_expanding,
                    )
                    _ms_st.update(label="Rolling stats ready", state="complete")
                st.caption(
                    "Computed on-the-fly — parameters fall outside the precomputed "
                    "grid (`window∈{60,120,252}`, `step=5`, `pearson`, `rolling`)."
                )

            if not market_stats.empty:
                fig_rc = go.Figure()
                fig_rc.add_trace(go.Scatter(
                    x=market_stats.index, y=market_stats["q75_corr"],
                    mode="lines", line=dict(width=0), showlegend=False,
                ))
                fig_rc.add_trace(go.Scatter(
                    x=market_stats.index, y=market_stats["q25_corr"],
                    mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=get_colors()["positive"], name="IQR (Q25-Q75)",
                ))
                fig_rc.add_trace(go.Scatter(
                    x=market_stats.index, y=market_stats["avg_corr"],
                    mode="lines", name="Mean",
                    line=dict(color=get_colors()["primary"], width=2.2),
                ))
                fig_rc.add_trace(go.Scatter(
                    x=market_stats.index, y=market_stats["median_corr"],
                    mode="lines", name="Median",
                    line=dict(color="#FF9F1C", width=1.5, dash="dot"),
                ))
                draw_event_markers(fig_rc, show_defaults, custom_events,
                                   market_stats.index.min(), market_stats.index.max())
                apply_chart_style(fig_rc, height=450,
                                  yaxis_title=f"Pairwise {rc_method.title()} Correlation",
                                  xaxis_title="Date")
                render_chart(fig_rc, chart_id="mo_rolling_corr", filename_base="rolling_correlation",
                             title_key="mo_rolling_corr", default_title="Rolling Correlation Stats")

                if st.toggle("Show min/max envelope", key="mo_minmax_toggle"):
                    fig_mm = go.Figure()
                    fig_mm.add_trace(go.Scatter(
                        x=market_stats.index, y=market_stats["max_corr"],
                        mode="lines", line=dict(width=0), showlegend=False,
                    ))
                    fig_mm.add_trace(go.Scatter(
                        x=market_stats.index, y=market_stats["min_corr"],
                        mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(255,159,28,0.12)", name="Min-Max Range",
                    ))
                    fig_mm.add_trace(go.Scatter(
                        x=market_stats.index, y=market_stats["avg_corr"],
                        mode="lines", name="Mean",
                        line=dict(color=get_colors()["primary"], width=1.5),
                    ))
                    apply_chart_style(fig_mm, height=350, yaxis_title="Correlation")
                    render_chart(fig_mm, chart_id="mo_minmax", filename_base="minmax_range",
                                 title_key="mo_minmax", default_title="Min-Max Correlation Range")
            else:
                st.warning("Not enough data for the selected window size.")

        # ── Sub-Tab 2: Pair rolling correlation ─────────────────────────────
        # Body lives in `_render_rolling_pair` fragment (defined at module top).
        # Changing pair_a / pair_b only reruns the fragment, not the whole
        # script. The cross-page "Open in Pair Analysis" button uses
        # st.rerun(scope="app") to escape the fragment scope.
        with tab_pair:
            _render_rolling_pair()

        # ── Sub-Tab 3: Sector breakdown ─────────────────────────────────────
        with tab_sector:
            cluster_df_for_sectors = load_cluster_assignments()
            if not cluster_df_for_sectors.empty and "sector" in cluster_df_for_sectors.columns:
                sec_map = dict(zip(cluster_df_for_sectors["ticker"], cluster_df_for_sectors["sector"]))

                # Sector precompute is only for window=252, step=5, pearson.
                _use_precomputed_sector = (
                    rc_window == 252
                    and rc_step == 5
                    and rc_method == "pearson"
                    and not rc_expanding
                )
                if _use_precomputed_sector:
                    sector_stats = load_rolling_sector_stats_precomputed()
                    if not sector_stats.empty:
                        st.caption(
                            "Reading precomputed `rolling_sector_stats.parquet` "
                            "(`window=252`, `step=5`, `pearson`)."
                        )
                    else:
                        with st.status(f"Computing {_sector_rc.lower()} stats...", expanded=False) as _ss_st:
                            sector_stats = _compute_sector(
                                returns, returns_cache_key, tuple(sec_map.items()), rc_window, rc_step, rc_method,
                            )
                            _ss_st.update(label=f"{_sector_rc} stats ready", state="complete")
                else:
                    with st.status(f"Computing {_sector_rc.lower()} stats (custom params)...", expanded=False) as _ss_st:
                        sector_stats = _compute_sector(
                            returns, returns_cache_key, tuple(sec_map.items()), rc_window, rc_step, rc_method,
                        )
                        _ss_st.update(label=f"{_sector_rc} stats ready", state="complete")
                    st.caption(
                        f"Computed on-the-fly — {_sector_rc.lower()} precompute exists only for "
                        "`window=252`, `step=5`, `pearson`."
                    )

                if not sector_stats.empty:
                    fig_sec = go.Figure()
                    fig_sec.add_trace(go.Scatter(
                        x=sector_stats.index, y=sector_stats["intra_sector_avg"],
                        mode="lines", name=f"Intra-{_sector_rc} Avg",
                        line=dict(color=get_colors()["secondary"], width=2.2),
                    ))
                    fig_sec.add_trace(go.Scatter(
                        x=sector_stats.index, y=sector_stats["inter_sector_avg"],
                        mode="lines", name=f"Inter-{_sector_rc} Avg",
                        line=dict(color=get_colors()["primary"], width=2.2),
                    ))
                    draw_event_markers(fig_sec, show_defaults, custom_events,
                                       sector_stats.index.min(), sector_stats.index.max())
                    apply_chart_style(fig_sec, height=420, yaxis_title="Average Correlation")
                    render_chart(fig_sec, chart_id="mo_sector_corr", filename_base="sector_correlation",
                                 title_key="mo_sector_corr",
                                 default_title=f"{_sector_rc} Correlation")

                    intra_cols = [c for c in sector_stats.columns
                                  if c.startswith("intra_") and c != "intra_sector_avg"]
                    if intra_cols:
                        if st.toggle(f"Show per-{_sector_rc.lower()} breakdown", key="mo_per_sector_toggle"):
                            fig_per = go.Figure()
                            for i, col in enumerate(intra_cols):
                                sector_name = col.replace("intra_", "")
                                fig_per.add_trace(go.Scatter(
                                    x=sector_stats.index, y=sector_stats[col],
                                    mode="lines", name=sector_name,
                                    line=dict(color=SECTOR_PALETTE[i % len(SECTOR_PALETTE)], width=1.5),
                                ))
                            apply_chart_style(fig_per, height=420,
                                              yaxis_title=f"Intra-{_sector_rc} Correlation")
                            render_chart(fig_per, chart_id="mo_per_sector", filename_base="per_sector_corr",
                                         title_key="mo_per_sector",
                                         default_title=f"Per-{_sector_rc} Correlation")
                else:
                    st.warning(f"Not enough data for {_sector_rc.lower()} stats with this window.")
            else:
                st.info(f"Run the clustering pipeline to enable {_sector_rc.lower()} breakdown.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Pairs & Dislocations  (financial universes only)
# ══════════════════════════════════════════════════════════════════════════════

# Skip the whole block when the active universe has no pair-trading semantics.
# tab_pairs is None for EEG (we never added the tab in that case).
if tab_pairs is not None:
  with tab_pairs:

      # ── Section 7: Top/Bottom Pairs & Correlation Distribution ──────────────
      with st.container(border=True):
          section_header(
              "Top/Bottom Pairs & Correlation Distribution",
              "Most/least correlated pairs (left) and the full distribution of pairwise "
              "correlations (right). Click a pair to investigate in the Pair Analysis view.",
          )

          col_pairs, col_dist = st.columns([3, 2])

          with col_pairs:
              pairs = load_top_bottom()
              top_pairs = pairs[pairs["rank_type"] == "top"][
                  ["ticker_1", "ticker_2", "sector_1", "sector_2", "correlation"]
              ].reset_index(drop=True)
              bottom_pairs = pairs[pairs["rank_type"] == "bottom"][
                  ["ticker_1", "ticker_2", "sector_1", "sector_2", "correlation"]
              ].reset_index(drop=True)

              tab_top, tab_bottom = st.tabs(["Most Correlated", "Least Correlated"])
              with tab_top:
                  st.dataframe(top_pairs, use_container_width=True, hide_index=True)
                  _top_pair_idx = st.selectbox(
                      "Select pair to analyze",
                      range(len(top_pairs)),
                      format_func=lambda i: f"{top_pairs.iloc[i]['ticker_1']} / {top_pairs.iloc[i]['ticker_2']} ({top_pairs.iloc[i]['correlation']:.4f})",
                      key="top_pair_sel",
                  )
                  _open_pair_analysis_button(
                      top_pairs.iloc[_top_pair_idx]["ticker_1"],
                      top_pairs.iloc[_top_pair_idx]["ticker_2"],
                      key="top_pair_btn",
                  )

              with tab_bottom:
                  st.dataframe(bottom_pairs, use_container_width=True, hide_index=True)
                  _bot_pair_idx = st.selectbox(
                      "Select pair to analyze",
                      range(len(bottom_pairs)),
                      format_func=lambda i: f"{bottom_pairs.iloc[i]['ticker_1']} / {bottom_pairs.iloc[i]['ticker_2']} ({bottom_pairs.iloc[i]['correlation']:.4f})",
                      key="bot_pair_sel",
                  )
                  _open_pair_analysis_button(
                      bottom_pairs.iloc[_bot_pair_idx]["ticker_1"],
                      bottom_pairs.iloc[_bot_pair_idx]["ticker_2"],
                      key="bot_pair_btn",
                  )

          with col_dist:
              # `corr` used to be set as a script-level global inside
              # `with tab_corr:` (when the heatmap was not yet @st.fragment).
              # Now that the heatmap is fragment-scoped, `corr` is local to
              # `_render_correlation_heatmap()`. Recompute it here using the
              # current heat_method from session_state — `_compute_corr` is
              # @st.cache_data so we get a cache HIT (the heatmap fragment
              # already computed the same call with the same args).
              _heat_method_for_dist = st.session_state.get("heat_method", "pearson")
              corr = _compute_corr(
                  returns, returns_cache_key, dynamic_min_periods, _heat_method_for_dist,
              )
              mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
              upper_vals = corr.where(mask).stack().values
              upper_vals = upper_vals[~np.isnan(upper_vals)]

              fig_corr_dist = go.Figure()
              fig_corr_dist.add_trace(go.Histogram(
                  x=upper_vals, nbinsx=60,
                  marker_color=get_colors()["primary"], opacity=0.75,
                  hovertemplate="Corr: %{x:.3f}<br>Count: %{y}<extra></extra>",
              ))
              mean_val = np.mean(upper_vals)
              median_val = np.median(upper_vals)
              fig_corr_dist.add_vline(x=mean_val, line_dash="dash", line_color=get_colors()["secondary"],
                                       annotation_text=f"Mean: {mean_val:.3f}", annotation_font_size=10)
              fig_corr_dist.add_vline(x=median_val, line_dash="dot", line_color=get_colors()["tertiary"],
                                       annotation_text=f"Median: {median_val:.3f}", annotation_font_size=10)
              apply_chart_style(fig_corr_dist, height=420,
                                xaxis_title="Pairwise Correlation", yaxis_title="Frequency",
                                showlegend=False)
              render_chart(fig_corr_dist, chart_id="mo_corr_dist", filename_base="correlation_distribution",
                           title_key="mo_corr_dist", default_title="Correlation Distribution")

      # ── Section 8: Dislocation Candidates ───────────────────────────────────
      with st.container(border=True):
          section_header(
              "Dislocation Candidates",
              "Historically correlated pairs ranked by mean-reversion characteristics. "
              "Pairs with shorter half-lives and active Z-score dislocations are ranked higher.",
          )

          _candidates = load_dislocation_candidates()
          if not _candidates.empty:
              _display_cols = [
                  "ticker_a", "ticker_b", "sector_a", "sector_b",
                  "correlation", "beta", "half_life", "current_zscore",
                  "n_signals", "rank_score",
              ]
              _disp_cands = _candidates[[c for c in _display_cols if c in _candidates.columns]].copy()

              st.dataframe(
                  _disp_cands,
                  use_container_width=True,
                  hide_index=True,
                  column_config={
                      "correlation": st.column_config.NumberColumn(format="%.4f"),
                      "beta": st.column_config.NumberColumn(format="%.4f"),
                      "half_life": st.column_config.NumberColumn("Half-Life (days)", format="%.1f"),
                      "current_zscore": st.column_config.NumberColumn("Current Z", format="%.3f"),
                      "rank_score": st.column_config.NumberColumn("Score", format="%.4f"),
                  },
              )

              _cand_idx = st.selectbox(
                  "Select a candidate pair to analyze",
                  range(len(_disp_cands)),
                  format_func=lambda i: (
                      f"{_disp_cands.iloc[i]['ticker_a']} / {_disp_cands.iloc[i]['ticker_b']}  "
                      f"(Z={_disp_cands.iloc[i]['current_zscore']:.2f}, HL={_disp_cands.iloc[i]['half_life']:.0f}d)"
                  ),
                  key="cand_pair_sel",
              )
              _open_pair_analysis_button(
                  _disp_cands.iloc[_cand_idx]["ticker_a"],
                  _disp_cands.iloc[_cand_idx]["ticker_b"],
                  key="cand_pair_btn",
              )
          else:
              st.info(
                  "No dislocation candidates available. Run the pipeline "
                  "(`python run_pipeline.py`) to generate ranked candidate pairs."
              )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 — EEE Analysis (RMT, GLASSO, Wavelets, Transfer Entropy)
# ══════════════════════════════════════════════════════════════════════════════

with tab_eee:
    from eee_analysis import render as _render_eee
    _render_eee()
