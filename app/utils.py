"""Shared dashboard utilities — loaders, event markers, data-quality checks.

All pages import from here so caches and logic are never duplicated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR      = Path(__file__).resolve().parent       # app/
PROJECT_ROOT = APP_DIR.parent                        # repo root

for _p in (str(PROJECT_ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RESULTS   = PROJECT_ROOT / "data" / "results"
DATA_RAW       = PROJECT_ROOT / "data" / "raw"


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
    """Render an 'Event Markers' expander scoped to key_prefix.

    Returns (show_defaults, custom_events) from session state.
    Each page/section gets its own independent event set via key_prefix.
    """
    _key_show = f"{key_prefix}_show_defaults"
    _key_evs  = f"{key_prefix}_custom_events"

    if _key_show not in st.session_state:
        st.session_state[_key_show] = True
    if _key_evs not in st.session_state:
        st.session_state[_key_evs] = []

    with st.expander("Event Markers", expanded=False):
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
