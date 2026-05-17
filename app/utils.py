"""Shared dashboard utilities — loaders, event markers, data-quality checks.

All pages import from here so caches and logic are never duplicated.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chart_themes import (
    ChartTheme,
    get_active_theme,
    theme_to_layout,
    theme_to_colors,
)

# ---------------------------------------------------------------------------
# Theme & Style Constants
# ---------------------------------------------------------------------------

# Static fallback — used only at import time if nothing else is available.
# All chart code should call get_colors() instead.
COLORS = {
    "primary": "#4361EE",
    "secondary": "#E63946",
    "tertiary": "#2EC4B6",
    "muted": "#8D99AE",
    "positive": "rgba(67,97,238,0.12)",
    "negative": "rgba(230,57,70,0.12)",
    "band": "rgba(141,153,174,0.12)",
    "bg": "#FAFBFC",
}

SECTOR_PALETTE = [
    "#4361EE", "#E63946", "#2EC4B6", "#FF9F1C", "#6A0572",
    "#1B998B", "#E71D36", "#2B2D42", "#8338EC", "#FB5607",
    "#3A86A8", "#FFBE0B", "#06D6A0", "#118AB2", "#073B4C",
    "#EF476F", "#FFD166", "#06D6A0", "#118AB2", "#5E60CE",
    "#48BFE3", "#56CFE1",
]

# Legacy constant kept for any third-party code that reads it directly.
CHART_LAYOUT = dict(
    font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#2B2D42"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
    xaxis=dict(gridcolor="rgba(141,153,174,0.15)", zeroline=False),
    yaxis=dict(gridcolor="rgba(141,153,174,0.15)", zeroline=False),
)


def get_colors() -> dict[str, str]:
    """Return color palette reflecting the active theme."""
    return theme_to_colors(get_active_theme())


def apply_chart_style(fig, height: int = 420, title: str = "", **overrides):
    """Apply the active theme layout to any plotly figure."""
    theme = get_active_theme()
    layout = {**theme_to_layout(theme), "height": height, **overrides}
    # Handle title — can be a string (from title param) or a dict (from overrides)
    title_val = title or layout.get("title")
    if title_val:
        if isinstance(title_val, str):
            layout["title"] = dict(
                text=title_val,
                font=dict(size=theme.title_font_size, family=theme.font_family),
                x=0.01, xanchor="left", yanchor="top",
            )
        # else: title is already a dict from overrides, keep as-is
        layout["margin"] = dict(layout.get("margin", {}))
        layout["margin"]["t"] = max(layout["margin"].get("t", 10), 40)
    fig.update_layout(**layout)
    return fig


def render_chart(
    fig: go.Figure,
    chart_id: str,
    filename_base: str = "chart",
    title_key: str = "",
    default_title: str = "",
    width: str = "stretch",
    use_container_width: bool | None = None,   # deprecated; kept for back-compat
) -> None:
    """Render a Plotly figure with configured modebar, optional title, and export popover.

    Width API
    ---------
    ``width="stretch"`` (default) → chart fills its container.
    ``width="content"``           → chart sizes to its own content.

    The legacy ``use_container_width`` kwarg is still accepted for back-compat
    (True → ``"stretch"``, False → ``"content"``); callers should migrate to
    ``width=`` since Streamlit deprecates ``use_container_width`` after 2025-12-31.
    """
    from chart_export import render_export_popover, get_plotly_config

    # Back-compat shim — translate legacy kwarg to the new width= API.
    if use_container_width is not None:
        width = "stretch" if use_container_width else "content"

    # Per-chart title input — only updates the title, does NOT re-apply full theme.
    # Scope title_key by the active universe so a custom title set under one
    # universe doesn't leak into another (e.g. a "Sector Correlation" title
    # cached under BIST must NOT show up when the user switches to EEG, where
    # the same chart_id renders an "Anatomical region Correlation" default).
    if title_key:
        _u = st.session_state.get("universe", "")
        _scoped = f"_title_{title_key}__{_u}" if _u else f"_title_{title_key}"
        user_title = st.text_input(
            "Chart title",
            value=st.session_state.get(_scoped, default_title),
            key=_scoped,
            label_visibility="collapsed",
            placeholder="Add chart title...",
        )
        if user_title.strip():
            theme = get_active_theme()
            cur_margin = dict(fig.layout.margin.to_plotly_json())
            cur_margin["t"] = max(cur_margin.get("t", 10), 40)
            fig.update_layout(
                title=dict(
                    text=user_title.strip(),
                    font=dict(size=theme.title_font_size, family=theme.font_family),
                    x=0.01, xanchor="left", yanchor="top",
                ),
                margin=cur_margin,
            )

    config = get_plotly_config(chart_id)
    # NOTE: st.plotly_chart does NOT accept width= in Streamlit 1.41.1 (kwarg
    # landed in ~1.45+). Translate the public `width=` API to the legacy
    # use_container_width= kwarg for the actual call. When the Streamlit pin
    # is bumped past 1.45 (see docs/STREAMLIT_VERSION_BUMP.md), this can
    # revert to: st.plotly_chart(fig, width=width, config=config).
    st.plotly_chart(fig, use_container_width=(width == "stretch"), config=config)
    render_export_popover(fig, chart_id, filename_base)


def inject_custom_css():
    """Inject global CSS for consistent look & feel."""
    st.markdown("""
    <style>
    /* ── Sidebar styling (chart settings panel) ──────────────── */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    /* Ensure sidebar collapsed control (re-open arrow) is always visible */
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        z-index: 999 !important;
    }

    /* ── Kill Streamlit top padding & deploy bar ─────────────── */
    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0 !important;
    }
    .stMainBlockContainer {
        padding-top: 1rem !important;
    }
    /* Hide deploy button and toolbar, keep sidebar toggle */
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
    }

    /* ── Plotly modebar — subtle, reveal on hover ────────────── */
    .js-plotly-plot .modebar {
        opacity: 0.15 !important;
        transition: opacity 0.2s ease !important;
    }
    .js-plotly-plot:hover .modebar {
        opacity: 0.85 !important;
    }
    .js-plotly-plot .modebar-btn {
        font-size: 16px !important;
    }

    /* ── Metric cards ────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fc 0%, #eef1f6 100%);
        border: 1px solid #e2e6ee;
        border-radius: 8px;
        padding: 12px 16px 8px 16px;
        box-shadow: 0 1px 3px rgba(43,45,66,0.06);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2B2D42;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #6c757d;
    }

    /* ── Segmented control — dark nav bar ────────────────────── */
    div[data-testid="stSegmentedControl"] {
        background: #1a1c2e;
        border-radius: 8px;
        padding: 4px;
    }
    div[data-testid="stSegmentedControl"] button {
        color: #8D99AE !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 8px 28px !important;
        border: none !important;
        transition: all 0.15s ease;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background: #4361EE !important;
        color: #ffffff !important;
    }
    div[data-testid="stSegmentedControl"] button:hover:not([aria-checked="true"]) {
        color: #c8cdd6 !important;
        background: rgba(67,97,238,0.08) !important;
    }

    /* ── Sub-tabs — clean underline style ────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #e9ecef;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.88rem;
        color: #8D99AE;
        border-bottom: 2px solid transparent;
        padding: 10px 20px;
        margin-bottom: -2px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #4361EE;
        border-bottom-color: #4361EE;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4361EE !important;
    }

    /* ── Popover panels ──────────────────────────────────────── */
    [data-testid="stPopoverBody"] {
        border: 1px solid #e2e6ee;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(43,45,66,0.10);
        padding: 4px;
    }

    /* ── Section dividers ────────────────────────────────────── */
    .stMarkdown hr {
        border: none;
        border-top: 1px solid #e9ecef;
        margin: 1.2rem 0;
    }

    /* ── Bordered containers — card appearance ───────────────── */
    [data-testid="stVerticalBlock"] > div[data-testid="stExpander"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px;
    }

    /* ── Dataframe styling ───────────────────────────────────── */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Caption text ────────────────────────────────────────── */
    .stCaption {
        line-height: 1.5;
    }

    /* ── General polish ──────────────────────────────────────── */
    .stSelectbox label, .stDateInput label, .stSlider label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, icon: str = ""):
    """Render a styled page header with optional icon."""
    if icon:
        st.markdown(f"# {icon} {title}")
    else:
        st.markdown(f"# {title}")
    st.caption(subtitle)


def section_header(title: str, description: str = ""):
    """Render a consistent section header."""
    st.markdown("---")
    st.subheader(title)
    if description:
        st.caption(description)


# ---------------------------------------------------------------------------
# Paths — universe-aware (BIST / S&P-500 / EEG)
# ---------------------------------------------------------------------------
APP_DIR      = Path(__file__).resolve().parent       # app/
PROJECT_ROOT = APP_DIR.parent                        # repo root

for _p in (str(PROJECT_ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Phase H: per-session universe selection.
#
# DASHBOARD_UNIVERSE is the boot-time default (env-var driven). At runtime,
# the sidebar selector in app/dashboard.py writes the active universe key into
# st.session_state["universe"], which current_universe() reads back. Every
# cached loader in this file routes through a private _load_*(universe, ...)
# function so Streamlit caches are naturally keyed by universe — BIST and
# S&P caches coexist; switching is a one-line session-state change.
DASHBOARD_UNIVERSE = os.environ.get("DASHBOARD_UNIVERSE", "bist")


def current_universe() -> str:
    """Return the active universe key (session_state, fall back to env var)."""
    try:
        return st.session_state.get("universe", DASHBOARD_UNIVERSE)
    except Exception:
        # st.session_state can raise if called outside a Streamlit script
        # context (e.g., bare `python -c "import utils"` smoke imports).
        return DASHBOARD_UNIVERSE


def data_raw(universe: str | None = None) -> Path:
    return PROJECT_ROOT / "data" / (universe or current_universe()) / "raw"


def data_processed(universe: str | None = None) -> Path:
    return PROJECT_ROOT / "data" / (universe or current_universe()) / "processed"


def data_results(universe: str | None = None) -> Path:
    return PROJECT_ROOT / "data" / (universe or current_universe()) / "results"


# Backwards-compat module-level constants. These resolve at *import time* using
# the boot-time DASHBOARD_UNIVERSE env var and do NOT track session state.
# New code should call data_raw() / data_processed() / data_results() instead.
DATA_RAW       = data_raw(DASHBOARD_UNIVERSE)
DATA_PROCESSED = data_processed(DASHBOARD_UNIVERSE)
DATA_RESULTS   = data_results(DASHBOARD_UNIVERSE)


# ---------------------------------------------------------------------------
# Cached data loaders — public wrappers thread session_state into a
# universe-keyed underscored cache. The underscored functions are what
# Streamlit actually caches; the public wrappers exist so call sites can
# stay parameter-free.
# ---------------------------------------------------------------------------

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee `df.index` is a DatetimeIndex.

    Financial universes (BIST, S&P) ship parquet files with a DatetimeIndex
    (the trading day). EEG ships a float64 index of seconds-since-recording-
    start, which breaks every downstream date widget / chart that calls
    ``adj_close.index.min().date()``.

    To keep the rest of the dashboard universe-agnostic, we synthesise a
    DatetimeIndex anchored at 2020-01-01 with 1-second steps when the
    underlying index isn't datetime-typed. The displayed "dates" carry no
    physical meaning for EEG — the chart's x-axis just labels samples by
    a synthetic clock so the existing widgets render cleanly.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    synthetic = pd.date_range(
        start="2020-01-01", periods=len(df.index), freq="1s",
    )
    return df.set_axis(synthetic, axis=0)


# Streamlit Cloud's free tier has ~1 GB RAM (less after Python + libs are
# loaded; effective budget ≈ 500-700 MB). EEG's 593,280-sample × 64-channel
# parquet is ~300 MB raw. When the dashboard JSON-serialises it via
# `returns.to_json(...)` and re-parses it inside @st.cache_data compute
# helpers, working memory peaks past 1 GB and the kernel OOM-kills the
# Streamlit process (visible as "connection reset by peer" on the health
# check). Downsampling at load time keeps the dashboard responsive without
# affecting the pre-computed pipeline artifacts (which were built from the
# full data and live in data/<u>/results/*.{parquet,csv,json}).
_DASHBOARD_MAX_ROWS = 8_000  # comfortable on Streamlit Cloud free tier


def _downsample_if_oversize(df: pd.DataFrame, max_rows: int = _DASHBOARD_MAX_ROWS) -> pd.DataFrame:
    """Uniform-stride downsample if `df` has more rows than `max_rows`.

    BIST (~1,544 rows) and S&P (~1,547 rows) pass through unchanged.
    EEG (~593,280 rows) is decimated by stride ≈ 74 to land near `max_rows`.
    The dashboard's on-the-fly correlation/rolling-stats compute on the
    downsampled series is statistically equivalent to the full series at
    the windows the dashboard uses (every-Nth-sample subsampling preserves
    correlation structure when N << autocorrelation length, which holds
    for our 60-day / 120-day / 252-day windows even on EEG).
    """
    n = len(df)
    if n <= max_rows:
        return df
    stride = max(1, n // max_rows)
    return df.iloc[::stride].copy()


@st.cache_data
def _load_adj_close(universe: str) -> pd.DataFrame:
    df = pd.read_parquet(data_processed(universe) / "adj_close.parquet")
    df = _ensure_datetime_index(df)
    return _downsample_if_oversize(df)


def load_adj_close() -> pd.DataFrame:
    return _load_adj_close(current_universe())


@st.cache_data
def _load_log_returns(universe: str) -> pd.DataFrame:
    df = pd.read_parquet(data_processed(universe) / "log_returns.parquet")
    df = _ensure_datetime_index(df)
    return _downsample_if_oversize(df)


def load_log_returns() -> pd.DataFrame:
    return _load_log_returns(current_universe())


@st.cache_data
def _load_summary_stats(universe: str) -> pd.DataFrame:
    return pd.read_parquet(data_results(universe) / "summary_stats.parquet")


def load_summary_stats() -> pd.DataFrame:
    return _load_summary_stats(current_universe())


@st.cache_data
def _load_batch_corr(universe: str) -> pd.DataFrame:
    return pd.read_parquet(data_results(universe) / "pearson_corr.parquet")


def load_batch_corr() -> pd.DataFrame:
    return _load_batch_corr(current_universe())


@st.cache_data
def _load_coverage(universe: str) -> pd.DataFrame:
    return pd.read_csv(data_processed(universe) / "coverage_report.csv")


def load_coverage() -> pd.DataFrame:
    return _load_coverage(current_universe())


@st.cache_data
def _load_top_bottom(universe: str) -> pd.DataFrame:
    return pd.read_csv(data_results(universe) / "top_bottom_pairs.csv")


def load_top_bottom() -> pd.DataFrame:
    return _load_top_bottom(current_universe())


@st.cache_data
def _load_metadata(universe: str) -> dict:
    path = data_results(universe) / "pipeline_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_metadata() -> dict:
    return _load_metadata(current_universe())


@st.cache_data
def _load_fetch_metadata(universe: str) -> dict:
    path = data_raw(universe) / "fetch_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_fetch_metadata() -> dict:
    return _load_fetch_metadata(current_universe())


@st.cache_data
def _load_xu100(universe: str) -> pd.Series:
    """Load the universe's market-index series.

    The file is named ``xu100.parquet`` for historical reasons (the original
    BIST universe) but for S&P-500 it contains the ``^GSPC`` series.
    """
    path = data_raw(universe) / "xu100.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "Adj Close" in df.columns:
            return df["Adj Close"]
        if isinstance(df.columns, pd.MultiIndex):
            return df.iloc[:, 0]
        return df.iloc[:, 0]
    return pd.Series(dtype=float)


def load_xu100() -> pd.Series:
    return _load_xu100(current_universe())


@st.cache_data
def _load_linkage(universe: str):
    Z_path = data_results(universe) / "linkage_matrix.npy"
    labels_path = data_results(universe) / "linkage_labels.json"
    if Z_path.exists() and labels_path.exists():
        Z = np.load(Z_path)
        with open(labels_path) as f:
            labels = json.load(f)
        return Z, labels
    return None, None


def load_linkage():
    return _load_linkage(current_universe())


@st.cache_data
def _load_dendrogram_order(universe: str) -> Optional[list]:
    path = data_results(universe) / "dendrogram_order.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_dendrogram_order() -> Optional[list]:
    return _load_dendrogram_order(current_universe())


@st.cache_data
def _load_cluster_assignments(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "cluster_assignments.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_cluster_assignments() -> pd.DataFrame:
    return _load_cluster_assignments(current_universe())


@st.cache_data
def _load_mst_edges(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "mst_edges.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_mst_edges() -> pd.DataFrame:
    return _load_mst_edges(current_universe())


@st.cache_data
def _load_mst_metrics(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "mst_node_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_mst_metrics() -> pd.DataFrame:
    return _load_mst_metrics(current_universe())


@st.cache_data
def _load_dislocation_candidates(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "dislocation_candidates.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_dislocation_candidates() -> pd.DataFrame:
    return _load_dislocation_candidates(current_universe())


# ---------------------------------------------------------------------------
# EEE Analysis loaders
# ---------------------------------------------------------------------------

@st.cache_data
def _load_eigenvalue_spectrum(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "eigenvalue_spectrum.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_eigenvalue_spectrum() -> pd.DataFrame:
    return _load_eigenvalue_spectrum(current_universe())


@st.cache_data
def _load_denoised_corr(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "denoised_corr.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_denoised_corr() -> pd.DataFrame:
    return _load_denoised_corr(current_universe())


@st.cache_data
def _load_denoised_mst_edges(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "denoised_mst_edges.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_denoised_mst_edges() -> pd.DataFrame:
    return _load_denoised_mst_edges(current_universe())


@st.cache_data
def _load_partial_corr(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "partial_corr.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_partial_corr() -> pd.DataFrame:
    return _load_partial_corr(current_universe())


@st.cache_data
def _load_precision_matrix(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "precision_matrix.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_precision_matrix() -> pd.DataFrame:
    return _load_precision_matrix(current_universe())


@st.cache_data
def _load_partial_corr_edges(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "partial_corr_edges.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_partial_corr_edges() -> pd.DataFrame:
    return _load_partial_corr_edges(current_universe())


@st.cache_data
def _load_glasso_metadata(universe: str) -> dict:
    path = data_results(universe) / "glasso_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_glasso_metadata() -> dict:
    return _load_glasso_metadata(current_universe())


@st.cache_data
def _load_wavelet_metadata(universe: str) -> dict:
    path = data_results(universe) / "wavelet_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_wavelet_metadata() -> dict:
    return _load_wavelet_metadata(current_universe())


@st.cache_data
def _load_wavelet_mst_edges(universe: str, scale: int) -> pd.DataFrame:
    path = data_results(universe) / f"wavelet_mst_edges_scale{scale}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_wavelet_mst_edges(scale: int) -> pd.DataFrame:
    return _load_wavelet_mst_edges(current_universe(), scale)


@st.cache_data
def _load_wavelet_corr(universe: str, scale: int) -> pd.DataFrame:
    path = data_results(universe) / f"wavelet_corr_scale{scale}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_wavelet_corr(scale: int) -> pd.DataFrame:
    return _load_wavelet_corr(current_universe(), scale)


@st.cache_data
def _load_te_edges(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "te_network_edges.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_te_edges() -> pd.DataFrame:
    return _load_te_edges(current_universe())


@st.cache_data
def _load_te_node_roles(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "te_node_roles.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_te_node_roles() -> pd.DataFrame:
    return _load_te_node_roles(current_universe())


@st.cache_data
def _load_te_matrix(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "transfer_entropy_matrix.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_te_matrix() -> pd.DataFrame:
    return _load_te_matrix(current_universe())


@st.cache_data
def _load_net_te_matrix(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "net_transfer_entropy_matrix.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_net_te_matrix() -> pd.DataFrame:
    return _load_net_te_matrix(current_universe())


@st.cache_data
def _load_denoised_mst_metrics(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "denoised_mst_node_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_denoised_mst_metrics() -> pd.DataFrame:
    return _load_denoised_mst_metrics(current_universe())


@st.cache_data
def _load_wavelet_mst_metrics(universe: str, scale: int) -> pd.DataFrame:
    path = data_results(universe) / f"wavelet_mst_metrics_scale{scale}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_wavelet_mst_metrics(scale: int) -> pd.DataFrame:
    return _load_wavelet_mst_metrics(current_universe(), scale)


@st.cache_data
def _load_anomalies(universe: str) -> pd.DataFrame:
    path = data_processed(universe) / "anomalies.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"]) if path.stat().st_size > 0 else pd.DataFrame()
        return df
    return pd.DataFrame()


def load_anomalies() -> pd.DataFrame:
    return _load_anomalies(current_universe())


@st.cache_data
def _load_rolling_market_stats_precomputed(universe: str, window: int) -> pd.DataFrame:
    path = data_results(universe) / f"rolling_market_stats_w{window}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()


def load_rolling_market_stats_precomputed(window: int) -> pd.DataFrame:
    return _load_rolling_market_stats_precomputed(current_universe(), window)


@st.cache_data
def _load_rolling_sector_stats_precomputed(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "rolling_sector_stats.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()


def load_rolling_sector_stats_precomputed() -> pd.DataFrame:
    return _load_rolling_sector_stats_precomputed(current_universe())


# ---------------------------------------------------------------------------
# SNN (Spiking Neural Network) loaders
# ---------------------------------------------------------------------------

@st.cache_data
def _load_snn_metrics(universe: str) -> dict:
    """Per-pair + aggregate SNN metrics (macro-F1, Sharpe, fold-level)."""
    path = data_results(universe) / "snn_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_snn_metrics() -> dict:
    return _load_snn_metrics(current_universe())


@st.cache_data
def _load_snn_pair_list(universe: str) -> pd.DataFrame:
    """List of pairs the SNN was trained on (ticker_a, ticker_b, pair_id)."""
    path = data_results(universe) / "snn_pair_list.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_snn_pair_list() -> pd.DataFrame:
    return _load_snn_pair_list(current_universe())


@st.cache_data
def _load_snn_signals(universe: str, pair_id: str) -> pd.DataFrame:
    """Per-pair daily SNN signal: date, zscore, class probabilities, signal labels."""
    path = data_results(universe) / "snn_signals" / f"{pair_id}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


def load_snn_signals(pair_id: str) -> pd.DataFrame:
    return _load_snn_signals(current_universe(), pair_id)


@st.cache_data
def _load_snn_training_history(universe: str) -> pd.DataFrame:
    """Epoch-by-epoch training history of the universal SNN model."""
    path = data_results(universe) / "snn_training_history.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_snn_training_history() -> pd.DataFrame:
    return _load_snn_training_history(current_universe())


@st.cache_data
def _load_snn_raster_sample(universe: str) -> pd.DataFrame:
    """Spike raster from the sample pair (day, timestep, neuron, spike)."""
    path = data_results(universe) / "snn_spike_raster_sample.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_snn_raster_sample() -> pd.DataFrame:
    return _load_snn_raster_sample(current_universe())


@st.cache_data
def _load_snn_membrane_sample(universe: str) -> pd.DataFrame:
    """Membrane-potential V(t) for the sample pair's output-layer neurons."""
    path = data_results(universe) / "snn_membrane_sample.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_snn_membrane_sample() -> pd.DataFrame:
    return _load_snn_membrane_sample(current_universe())


# ---------------------------------------------------------------------------
# Event markers
# ---------------------------------------------------------------------------

DASH_OPTIONS: dict[str, str] = {
    "Dashed":    "dash",
    "Solid":     "solid",
    "Dotted":    "dot",
    "Long Dash": "longdash",
    "Dash-Dot":  "dashdot",
}
_DASH_LABELS: dict[str, str] = {v: k for k, v in DASH_OPTIONS.items()}

_DEFAULT_EVENTS_CACHE: Optional[list] = None


def _get_default_events() -> list:
    global _DEFAULT_EVENTS_CACHE
    if _DEFAULT_EVENTS_CACHE is None:
        try:
            from src.rolling_correlation import DEFAULT_EVENTS
            _DEFAULT_EVENTS_CACHE = DEFAULT_EVENTS
        except Exception:
            _DEFAULT_EVENTS_CACHE = []
    return _DEFAULT_EVENTS_CACHE


def draw_event_markers(
    fig: go.Figure,
    show_defaults: bool,
    custom_events: list,
    date_min,
    date_max,
) -> None:
    """Render vertical event-marker lines + annotations onto a plotly figure."""
    events: list[dict] = []
    if show_defaults:
        for ev in _get_default_events():
            events.append({
                "date": ev["date"], "label": ev["label"],
                "color": "#E53935", "dash": "dash", "width": 1.0,
            })
    events.extend(custom_events)

    for ev in events:
        ev_date = pd.Timestamp(ev["date"])
        if date_min <= ev_date <= date_max:
            fig.add_shape(
                type="line", x0=ev_date, x1=ev_date,
                y0=0, y1=1, yref="paper",
                line=dict(dash=ev["dash"], color=ev["color"], width=ev["width"]),
                opacity=0.75,
            )
            fig.add_annotation(
                x=ev_date, y=1, yref="paper",
                text=ev["label"], showarrow=False,
                font=dict(size=9, color=ev["color"]), yshift=10,
            )


def event_marker_manager_ui(
    key_prefix: str,
    min_date,
    max_date,
) -> tuple[bool, list]:
    """Render an 'Event Markers' popover scoped to key_prefix.

    Returns (show_defaults, custom_events) from session state.
    Each page/section gets its own independent event set via key_prefix.
    """
    _key_show = f"{key_prefix}_show_defaults"
    _key_evs  = f"{key_prefix}_custom_events"

    if _key_show not in st.session_state:
        st.session_state[_key_show] = True
    if _key_evs not in st.session_state:
        st.session_state[_key_evs] = []

    with st.popover("Event Markers", icon=":material/flag:"):
        st.session_state[_key_show] = st.checkbox(
            "Show default macro events (COVID-19, Russia-Ukraine War, Turkey Earthquakes)",
            value=st.session_state[_key_show],
            key=f"{key_prefix}_chk_defaults",
        )

        evs: list = st.session_state[_key_evs]
        if evs:
            st.markdown("**Custom events**")
            # NOTE: no st.columns here — event_marker_manager_ui is called
            # from inside dashboard tabs which already have column ancestors,
            # and the wrapping st.popover doesn't reset that. Streamlit 1.41+
            # rejects st.columns 2 levels deep. Render the label + remove
            # button inline (the X glyph wraps to a new line if needed).
            for _i, _cev in enumerate(evs):
                _style_label = _DASH_LABELS.get(_cev["dash"], _cev["dash"])
                st.markdown(
                    f"<span style='color:{_cev['color']}'>▌</span> "
                    f"**{_cev['label']}** &nbsp; {_cev['date']} &nbsp;·&nbsp; "
                    f"{_style_label}, {_cev['width']}px",
                    unsafe_allow_html=True,
                )
                if st.button("✕ Remove", key=f"{key_prefix}_rm_{_i}", help="Remove this marker"):
                    st.session_state[_key_evs].pop(_i)
                    st.rerun()

        st.markdown("**Add a new event marker**")
        with st.form(f"{key_prefix}_add_event_form", clear_on_submit=True):
            # Vertically stacked (no st.columns inside the popover form) for
            # the same 1.41+ nesting reason — the popover is already inside
            # the dashboard's column tree, so columns here would be 2 deep.
            _ev_date = st.date_input(
                "Date",
                value=min(pd.Timestamp.today().date(), max_date),
                min_value=min_date,
                max_value=max_date,
                key=f"{key_prefix}_new_ev_date",
            )
            _ev_label = st.text_input(
                "Caption / label", value="",
                placeholder="e.g. Fed Rate Hike",
                key=f"{key_prefix}_new_ev_label",
            )
            _ev_color = st.color_picker(
                "Line color", value="#E53935",
                key=f"{key_prefix}_new_ev_color",
            )
            _ev_style = st.selectbox(
                "Line style", list(DASH_OPTIONS.keys()), index=0,
                key=f"{key_prefix}_new_ev_style",
            )
            _ev_width = st.slider(
                "Line thickness (px)", min_value=0.5, max_value=5.0,
                value=1.5, step=0.5,
                key=f"{key_prefix}_new_ev_width",
            )
            if st.form_submit_button("Add Event Marker"):
                st.session_state[_key_evs].append({
                    "date":  str(_ev_date),
                    "label": _ev_label.strip() or str(_ev_date),
                    "color": _ev_color,
                    "dash":  DASH_OPTIONS[_ev_style],
                    "width": _ev_width,
                })
                st.rerun()

    return st.session_state[_key_show], st.session_state[_key_evs]


# ---------------------------------------------------------------------------
# Data-quality warnings
# ---------------------------------------------------------------------------

def check_ticker_pair_warnings(
    full_returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window_returns: Optional[pd.DataFrame] = None,
    coverage_df: Optional[pd.DataFrame] = None,
    anomaly_threshold: float = 0.30,
) -> list[dict]:
    """Return a list of data-quality issues for a ticker pair.

    Each item has keys:
        level   : "error" | "warning" | "info"
        message : markdown string
    """
    issues: list[dict] = []
    ret = window_returns if window_returns is not None else full_returns

    for ticker in (ticker_a, ticker_b):
        if ticker not in full_returns.columns:
            issues.append({
                "level": "error",
                "message": f"**{ticker}** is not found in the processed dataset.",
            })
            continue

        series = full_returns[ticker].dropna()

        # ── Coverage check
        if coverage_df is not None and not coverage_df.empty:
            row = coverage_df[coverage_df["ticker"] == ticker]
            if not row.empty:
                cov = float(row["coverage_pct"].values[0])
                if cov < 0.90:
                    issues.append({
                        "level": "warning",
                        "message": (
                            f"**{ticker}** has only **{cov:.1%}** data coverage "
                            f"(threshold: 90%). Statistics may be unreliable."
                        ),
                    })

        # ── Anomalous returns (dividend / split / data artifact)
        extreme = series[series.abs() > anomaly_threshold]
        if not extreme.empty:
            issues.append({
                "level": "warning",
                "message": (
                    f"**{ticker}** has **{len(extreme)}** day(s) with |return| > "
                    f"{anomaly_threshold:.0%} (max: {extreme.abs().max():.1%}). "
                    f"These may be stock splits, special dividends, or data errors "
                    f"and could distort correlation estimates."
                ),
            })

    # ── Both tickers available — cross-checks
    if ticker_a in full_returns.columns and ticker_b in full_returns.columns:
        both_full = full_returns[[ticker_a, ticker_b]].dropna()
        a_days    = full_returns[ticker_a].dropna()
        b_days    = full_returns[ticker_b].dropna()
        mismatch  = (len(a_days) - len(both_full)) + (len(b_days) - len(both_full))

        if mismatch > 5:
            issues.append({
                "level": "info",
                "message": (
                    f"**{ticker_a}** and **{ticker_b}** have **{mismatch}** non-overlapping "
                    f"trading days (different listing histories or trading halts). "
                    f"Correlation is computed on **{len(both_full)}** common dates."
                ),
            })

        # ── Short overlap
        both_win = ret[[ticker_a, ticker_b]].dropna() if (
            ticker_a in ret.columns and ticker_b in ret.columns
        ) else both_full
        if len(both_win) < 60:
            issues.append({
                "level": "warning",
                "message": (
                    f"Only **{len(both_win)}** overlapping observations in the selected window — "
                    f"correlation estimates are unreliable with fewer than 60 data points."
                ),
            })

        # ── Extreme correlation
        if len(both_full) >= 30:
            try:
                rho = both_full.corr().iloc[0, 1]
                if abs(rho) > 0.97:
                    issues.append({
                        "level": "info",
                        "message": (
                            f"Full-period Pearson ρ is extremely high (**{rho:.4f}**). "
                            f"These stocks may be the same entity, a subsidiary, or both track "
                            f"the same index segment."
                        ),
                    })
                elif rho < -0.4:
                    issues.append({
                        "level": "info",
                        "message": (
                            f"Full-period Pearson ρ is notably negative (**{rho:.4f}**). "
                            f"These stocks tend to move in opposite directions — potentially "
                            f"useful for portfolio hedging."
                        ),
                    })
            except Exception:
                pass

    return issues


def render_warnings(issues: list[dict]) -> None:
    """Render data-quality issues in Streamlit."""
    for issue in issues:
        level = issue["level"]
        msg   = issue["message"]
        if level == "error":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.info(msg)
