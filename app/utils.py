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
    use_container_width: bool = True,
) -> None:
    """Render a Plotly figure with configured modebar, optional title, and export popover."""
    from chart_export import render_export_popover, get_plotly_config

    # Per-chart title input — only updates the title, does NOT re-apply full theme
    if title_key:
        user_title = st.text_input(
            "Chart title",
            value=st.session_state.get(f"_title_{title_key}", default_title),
            key=f"_title_{title_key}",
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
    st.plotly_chart(fig, use_container_width=use_container_width, config=config)
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
# Paths
# ---------------------------------------------------------------------------
APP_DIR      = Path(__file__).resolve().parent       # app/
PROJECT_ROOT = APP_DIR.parent                        # repo root

for _p in (str(PROJECT_ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Phase D: the pipeline now writes per-market artifacts under data/<market>/...
# (BIST, sp500, eeg_*, etc.). The dashboard currently surfaces one universe at
# a time via the DASHBOARD_UNIVERSE env var (default: "bist"). Phase E will
# add a runtime sidebar selector that flips this.
DASHBOARD_UNIVERSE = os.environ.get("DASHBOARD_UNIVERSE", "bist")
DATA_RAW       = PROJECT_ROOT / "data" / DASHBOARD_UNIVERSE / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / DASHBOARD_UNIVERSE / "processed"
DATA_RESULTS   = PROJECT_ROOT / "data" / DASHBOARD_UNIVERSE / "results"


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_adj_close() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / "adj_close.parquet")


@st.cache_data
def load_log_returns() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / "log_returns.parquet")


@st.cache_data
def load_summary_stats() -> pd.DataFrame:
    return pd.read_parquet(DATA_RESULTS / "summary_stats.parquet")


@st.cache_data
def load_batch_corr() -> pd.DataFrame:
    return pd.read_parquet(DATA_RESULTS / "pearson_corr.parquet")


@st.cache_data
def load_coverage() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "coverage_report.csv")


@st.cache_data
def load_top_bottom() -> pd.DataFrame:
    return pd.read_csv(DATA_RESULTS / "top_bottom_pairs.csv")


@st.cache_data
def load_metadata() -> dict:
    path = DATA_RESULTS / "pipeline_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_fetch_metadata() -> dict:
    path = DATA_RAW / "fetch_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_xu100() -> pd.Series:
    path = DATA_RAW / "xu100.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "Adj Close" in df.columns:
            return df["Adj Close"]
        if isinstance(df.columns, pd.MultiIndex):
            return df.iloc[:, 0]
        return df.iloc[:, 0]
    return pd.Series(dtype=float)


@st.cache_data
def load_linkage():
    Z_path     = DATA_RESULTS / "linkage_matrix.npy"
    labels_path = DATA_RESULTS / "linkage_labels.json"
    if Z_path.exists() and labels_path.exists():
        Z = np.load(Z_path)
        with open(labels_path) as f:
            labels = json.load(f)
        return Z, labels
    return None, None


@st.cache_data
def load_dendrogram_order() -> Optional[list]:
    path = DATA_RESULTS / "dendrogram_order.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_cluster_assignments() -> pd.DataFrame:
    path = DATA_RESULTS / "cluster_assignments.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_mst_edges() -> pd.DataFrame:
    path = DATA_RESULTS / "mst_edges.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_mst_metrics() -> pd.DataFrame:
    path = DATA_RESULTS / "mst_node_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_dislocation_candidates() -> pd.DataFrame:
    path = DATA_RESULTS / "dislocation_candidates.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# EEE Analysis loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_eigenvalue_spectrum() -> pd.DataFrame:
    path = DATA_RESULTS / "eigenvalue_spectrum.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_denoised_corr() -> pd.DataFrame:
    path = DATA_RESULTS / "denoised_corr.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_denoised_mst_edges() -> pd.DataFrame:
    path = DATA_RESULTS / "denoised_mst_edges.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_partial_corr() -> pd.DataFrame:
    path = DATA_RESULTS / "partial_corr.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_precision_matrix() -> pd.DataFrame:
    path = DATA_RESULTS / "precision_matrix.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_partial_corr_edges() -> pd.DataFrame:
    path = DATA_RESULTS / "partial_corr_edges.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_glasso_metadata() -> dict:
    path = DATA_RESULTS / "glasso_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_wavelet_metadata() -> dict:
    path = DATA_RESULTS / "wavelet_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_wavelet_mst_edges(scale: int) -> pd.DataFrame:
    path = DATA_RESULTS / f"wavelet_mst_edges_scale{scale}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_wavelet_corr(scale: int) -> pd.DataFrame:
    path = DATA_RESULTS / f"wavelet_corr_scale{scale}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_te_edges() -> pd.DataFrame:
    path = DATA_RESULTS / "te_network_edges.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_te_node_roles() -> pd.DataFrame:
    path = DATA_RESULTS / "te_node_roles.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_te_matrix() -> pd.DataFrame:
    path = DATA_RESULTS / "transfer_entropy_matrix.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_net_te_matrix() -> pd.DataFrame:
    path = DATA_RESULTS / "net_transfer_entropy_matrix.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_denoised_mst_metrics() -> pd.DataFrame:
    path = DATA_RESULTS / "denoised_mst_node_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_wavelet_mst_metrics(scale: int) -> pd.DataFrame:
    path = DATA_RESULTS / f"wavelet_mst_metrics_scale{scale}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_anomalies() -> pd.DataFrame:
    path = DATA_PROCESSED / "anomalies.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"]) if path.stat().st_size > 0 else pd.DataFrame()
        return df
    return pd.DataFrame()


@st.cache_data
def load_rolling_market_stats_precomputed(window: int) -> pd.DataFrame:
    path = DATA_RESULTS / f"rolling_market_stats_w{window}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()


@st.cache_data
def load_rolling_sector_stats_precomputed() -> pd.DataFrame:
    path = DATA_RESULTS / "rolling_sector_stats.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# SNN (Spiking Neural Network) loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_snn_metrics() -> dict:
    """Per-pair + aggregate SNN metrics (macro-F1, Sharpe, fold-level)."""
    path = DATA_RESULTS / "snn_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_snn_pair_list() -> pd.DataFrame:
    """List of pairs the SNN was trained on (ticker_a, ticker_b, pair_id)."""
    path = DATA_RESULTS / "snn_pair_list.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_snn_signals(pair_id: str) -> pd.DataFrame:
    """Per-pair daily SNN signal: date, zscore, class probabilities, signal labels."""
    path = DATA_RESULTS / "snn_signals" / f"{pair_id}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


@st.cache_data
def load_snn_training_history() -> pd.DataFrame:
    """Epoch-by-epoch training history of the universal SNN model."""
    path = DATA_RESULTS / "snn_training_history.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_snn_raster_sample() -> pd.DataFrame:
    """Spike raster from the sample pair (day, timestep, neuron, spike)."""
    path = DATA_RESULTS / "snn_spike_raster_sample.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_snn_membrane_sample() -> pd.DataFrame:
    """Membrane-potential V(t) for the sample pair's output-layer neurons."""
    path = DATA_RESULTS / "snn_membrane_sample.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


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
            for _i, _cev in enumerate(evs):
                _style_label = _DASH_LABELS.get(_cev["dash"], _cev["dash"])
                _c1, _c2 = st.columns([6, 1])
                with _c1:
                    st.markdown(
                        f"<span style='color:{_cev['color']}'>▌</span> "
                        f"**{_cev['label']}** &nbsp; {_cev['date']} &nbsp;·&nbsp; "
                        f"{_style_label}, {_cev['width']}px",
                        unsafe_allow_html=True,
                    )
                with _c2:
                    if st.button("✕", key=f"{key_prefix}_rm_{_i}", help="Remove this marker"):
                        st.session_state[_key_evs].pop(_i)
                        st.rerun()

        st.markdown("**Add a new event marker**")
        with st.form(f"{key_prefix}_add_event_form", clear_on_submit=True):
            _fa, _fb = st.columns(2)
            with _fa:
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
            with _fb:
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
