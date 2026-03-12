"""Pair Analysis — deep-dive comparison of two BIST-100 stocks.

This page is intentionally self-contained so it can be developed
and extended independently from the main dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap — ensures app/ and project root are importable
# ---------------------------------------------------------------------------
_APP_DIR      = Path(__file__).resolve().parent.parent   # app/pages/../ = app/
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import (  # noqa: E402
    load_adj_close,
    load_log_returns,
    load_coverage,
    load_summary_stats,
    draw_event_markers,
    event_marker_manager_ui,
    check_ticker_pair_warnings,
    render_warnings,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Pair Analysis — StoNeCoAl", layout="wide")

# Back-navigation hint
st.markdown(
    "[← Back to Overview](/) &nbsp;&nbsp; *(use the sidebar to switch pages)*",
    unsafe_allow_html=True,
)
st.title("Pair Analysis")
st.caption(
    "Deep-dive comparison of two BIST-100 stocks: rolling correlation, return statistics, "
    "price performance, scatter, volatility, and drawdown analysis."
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
adj_close    = load_adj_close()
full_returns = load_log_returns()
coverage_df  = load_coverage()

min_date = adj_close.index.min().date()
max_date = adj_close.index.max().date()

# ---------------------------------------------------------------------------
# Sidebar — ticker & window selection
# ---------------------------------------------------------------------------
st.sidebar.header("Pair Selection")
ticker_list = sorted(full_returns.columns.tolist())

ticker_a = st.sidebar.selectbox("Ticker A", ticker_list, index=0, key="pa_ticker_a")
ticker_b = st.sidebar.selectbox(
    "Ticker B", ticker_list,
    index=min(1, len(ticker_list) - 1),
    key="pa_ticker_b",
)

st.sidebar.markdown("---")
st.sidebar.header("Date Range")
date_range = st.sidebar.date_input(
    "Window",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="pa_date_range",
)
if len(date_range) == 2:
    start_dt = pd.Timestamp(date_range[0])
    end_dt   = pd.Timestamp(date_range[1])
else:
    start_dt = pd.Timestamp(min_date)
    end_dt   = pd.Timestamp(max_date)

returns       = full_returns.loc[start_dt:end_dt]
prices_window = adj_close.loc[start_dt:end_dt]

st.sidebar.markdown("---")
st.sidebar.header("Rolling Window")
rc_window = st.sidebar.selectbox(
    "Window (days)", [30, 60, 120, 252, 504], index=2, key="pa_rc_win"
)
rc_step = st.sidebar.selectbox(
    "Step", [1, 5, 21], index=1, key="pa_rc_step",
    format_func=lambda v: {1: "1 (daily)", 5: "5 (weekly)", 21: "21 (monthly)"}[v],
)
rc_method      = st.sidebar.selectbox("Corr. method", ["pearson", "spearman"], key="pa_rc_method")
rc_window_type = st.sidebar.selectbox(
    "Window type", ["rolling", "expanding", "ewm"], key="pa_rc_wtype"
)

# ---------------------------------------------------------------------------
# Guard: same ticker selected
# ---------------------------------------------------------------------------
if ticker_a == ticker_b:
    st.warning("Please select two **different** tickers in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Data-quality warnings
# ---------------------------------------------------------------------------
issues = check_ticker_pair_warnings(full_returns, ticker_a, ticker_b, returns, coverage_df)
if issues:
    with st.expander(f"Data Quality Notices ({len(issues)})", expanded=True):
        render_warnings(issues)

# Guard: both tickers must exist in processed data
if ticker_a not in returns.columns or ticker_b not in returns.columns:
    st.error("One or both selected tickers are missing from the processed dataset. Run the pipeline first.")
    st.stop()

# ---------------------------------------------------------------------------
# Shared overlapping returns
# ---------------------------------------------------------------------------
both = returns[[ticker_a, ticker_b]].dropna()

# ---------------------------------------------------------------------------
# Event markers (independent per-page set)
# ---------------------------------------------------------------------------
show_defaults, custom_events = event_marker_manager_ui("pa", min_date, max_date)

# ---------------------------------------------------------------------------
# Section 1 — Summary statistics
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Summary Statistics")
st.caption(
    "Full-period (window) statistics. **Pearson ρ** measures linear co-movement; "
    "**Spearman ρ** is rank-based and more robust to outliers and fat-tailed return distributions."
)

if len(both) >= 2:
    rho_p = both.corr(method="pearson").iloc[0, 1]
    rho_s = both.corr(method="spearman").iloc[0, 1]

    _m1, _m2, _m3 = st.columns(3)
    _m1.metric("Pearson ρ", f"{rho_p:.4f}")
    _m2.metric("Spearman ρ", f"{rho_s:.4f}")
    _m3.metric("Overlapping Trading Days", f"{len(both):,}")

# Per-ticker stats table
try:
    _summary_all = load_summary_stats()
    _show_cols   = [
        "ticker", "annualized_return", "annualized_vol",
        "skewness", "kurtosis", "min_return", "max_return",
    ]
    _subset = _summary_all[_summary_all["ticker"].isin([ticker_a, ticker_b])][_show_cols].copy()
    for _c in ["annualized_return", "annualized_vol", "min_return", "max_return"]:
        _subset[_c] = _subset[_c].map(lambda v: f"{v:.4f}")
    for _c in ["skewness", "kurtosis"]:
        _subset[_c] = _subset[_c].map(lambda v: f"{v:.2f}")
    st.dataframe(_subset.set_index("ticker"), use_container_width=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Section 2 — Rolling correlation
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Rolling Correlation")
st.caption(
    f"How the **{rc_method.title()}** correlation between the two stocks changes over time "
    f"using a **{rc_window}-day {rc_window_type}** window. "
    "Values near +1 = stocks move almost identically; near -1 = move in opposite directions."
)

try:
    from src.rolling_correlation import compute_rolling_pair_correlation

    @st.cache_data
    def _pair_corr(ret_json: str, a: str, b: str, win: int, method: str, wtype: str) -> pd.Series:
        _ret = pd.read_json(ret_json, orient="split")
        return compute_rolling_pair_correlation(_ret, a, b, window=win, method=method, window_type=wtype)

    with st.spinner("Computing rolling correlation…"):
        pair_corr = _pair_corr(
            returns.to_json(orient="split", date_format="iso"),
            ticker_a, ticker_b, rc_window, rc_method, rc_window_type,
        )

    pair_valid = pair_corr.dropna()

    if not pair_valid.empty:
        # Rolling correlation stats bar
        _s1, _s2, _s3, _s4, _s5 = st.columns(5)
        _s1.metric("Mean ρ",           f"{pair_valid.mean():.4f}")
        _s2.metric("Median ρ",         f"{pair_valid.median():.4f}")
        _s3.metric("Max ρ",            f"{pair_valid.max():.4f}")
        _s4.metric("Min ρ",            f"{pair_valid.min():.4f}")
        _s5.metric("% Time Positive",  f"{(pair_valid > 0).mean():.1%}")

        fig_rc = go.Figure()
        # Zero line reference
        fig_rc.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
        # Positive / negative shading
        fig_rc.add_trace(go.Scatter(
            x=pair_corr.index, y=pair_corr.clip(lower=0),
            mode="lines", line=dict(width=0), showlegend=False,
            fill="tozeroy", fillcolor="rgba(65,105,225,0.10)",
        ))
        fig_rc.add_trace(go.Scatter(
            x=pair_corr.index, y=pair_corr.clip(upper=0),
            mode="lines", line=dict(width=0), showlegend=False,
            fill="tozeroy", fillcolor="rgba(220,20,60,0.10)",
        ))
        # Main line
        fig_rc.add_trace(go.Scatter(
            x=pair_corr.index, y=pair_corr.values,
            mode="lines", name=f"{ticker_a} / {ticker_b}",
            line=dict(color="royalblue", width=1.8),
        ))
        draw_event_markers(fig_rc, show_defaults, custom_events,
                           pair_valid.index.min(), pair_valid.index.max())
        fig_rc.update_layout(
            height=420,
            yaxis=dict(range=[-1.05, 1.05], title=f"{rc_method.title()} Correlation"),
            xaxis_title="Date",
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_rc, use_container_width=True)
    else:
        st.warning("Not enough overlapping data for the selected window size.")

except Exception as exc:
    st.error(f"Could not compute rolling correlation: {exc}")

# ---------------------------------------------------------------------------
# Section 3 — Normalized price comparison
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Price Performance")
st.caption(
    "Both stocks rebased to **100** at the start of the selected window. "
    "The **shaded area between the two lines** highlights divergence — wider gaps indicate "
    "the stocks are decoupling; convergence suggests correlated movement."
)

if ticker_a in prices_window.columns and ticker_b in prices_window.columns:
    pa = prices_window[ticker_a] / prices_window[ticker_a].iloc[0] * 100
    pb = prices_window[ticker_b] / prices_window[ticker_b].iloc[0] * 100

    fig_price = go.Figure()
    # Divergence fill between the two lines
    _aligned_p = pd.concat([pa, pb], axis=1).dropna()
    if not _aligned_p.empty:
        _pa_al = _aligned_p.iloc[:, 0]
        _pb_al = _aligned_p.iloc[:, 1]
        fig_price.add_trace(go.Scatter(
            x=list(_pa_al.index) + list(_pb_al.index[::-1]),
            y=list(_pa_al.values) + list(_pb_al.values[::-1]),
            fill="toself", fillcolor="rgba(99,110,250,0.10)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

    fig_price.add_trace(go.Scatter(
        x=pa.index, y=pa, name=ticker_a,
        mode="lines", line=dict(color="royalblue", width=2.2),
    ))
    fig_price.add_trace(go.Scatter(
        x=pb.index, y=pb, name=ticker_b,
        mode="lines", line=dict(color="crimson", width=2.2),
    ))
    draw_event_markers(fig_price, show_defaults, custom_events, pa.index.min(), pa.index.max())
    fig_price.update_layout(
        height=420, yaxis_title="Normalized Price (base = 100)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_price, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4 — Return scatter + regression line
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Daily Return Scatter")
st.caption(
    "Each point is one trading day plotted as **(return of A, return of B)**. "
    "A tight diagonal cluster indicates high positive correlation. "
    "The **red regression line** slope approximates the beta of B relative to A — "
    "a slope > 1 means B amplifies A's moves."
)

if len(both) >= 10:
    _ra = both[ticker_a].values
    _rb = both[ticker_b].values

    # Manual OLS regression (no statsmodels dependency)
    _m, _b_int = np.polyfit(_ra, _rb, 1)
    _x_line = np.linspace(_ra.min(), _ra.max(), 200)
    _y_line = _m * _x_line + _b_int

    _rho_label = both.corr().iloc[0, 1]

    fig_scat = go.Figure()
    fig_scat.add_trace(go.Scatter(
        x=_ra, y=_rb,
        mode="markers",
        name="Daily returns",
        marker=dict(color="royalblue", size=5, opacity=0.45),
        hovertemplate=f"{ticker_a}: %{{x:.4f}}<br>{ticker_b}: %{{y:.4f}}<extra></extra>",
    ))
    fig_scat.add_trace(go.Scatter(
        x=_x_line, y=_y_line,
        mode="lines", name=f"OLS (β={_m:.3f})",
        line=dict(color="crimson", width=2),
    ))
    fig_scat.update_layout(
        height=440,
        xaxis_title=f"{ticker_a} Daily Log Return",
        yaxis_title=f"{ticker_b} Daily Log Return",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        title=dict(text=f"Pearson ρ = {_rho_label:.4f}", font=dict(size=12), x=0.5),
    )
    st.plotly_chart(fig_scat, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5 — Overlaid return distributions
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Return Distributions")
st.caption(
    "Overlaid histograms of daily log returns with **transparency** so both distributions are visible. "
    "A wider spread = higher volatility. Heavier left tails = more severe drawdown days. "
    "Ideal diversification partners have non-overlapping distribution shapes."
)

if not both.empty:
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=both[ticker_a], name=ticker_a, nbinsx=70,
        marker_color="royalblue", opacity=0.60,
    ))
    fig_dist.add_trace(go.Histogram(
        x=both[ticker_b], name=ticker_b, nbinsx=70,
        marker_color="crimson", opacity=0.60,
    ))
    fig_dist.update_layout(
        barmode="overlay", height=380,
        xaxis_title="Daily Log Return", yaxis_title="Count",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 6 — Rolling volatility
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader(f"Rolling Volatility (Annualized, {rc_window}-day)")
st.caption(
    f"Rolling {rc_window}-day standard deviation of daily returns scaled by √252. "
    "The **shaded band between the two lines** shows when their volatility levels diverge — "
    "simultaneous spikes indicate shared systematic risk (e.g., market-wide shock)."
)

if not both.empty:
    _vol_a = both[ticker_a].rolling(rc_window).std() * np.sqrt(252)
    _vol_b = both[ticker_b].rolling(rc_window).std() * np.sqrt(252)

    fig_vol = go.Figure()

    # Spread fill
    _vol_aligned = pd.concat([_vol_a, _vol_b], axis=1).dropna()
    if not _vol_aligned.empty:
        _va, _vb = _vol_aligned.iloc[:, 0], _vol_aligned.iloc[:, 1]
        fig_vol.add_trace(go.Scatter(
            x=list(_va.index) + list(_vb.index[::-1]),
            y=list(_va.values) + list(_vb.values[::-1]),
            fill="toself", fillcolor="rgba(128,128,128,0.12)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

    fig_vol.add_trace(go.Scatter(
        x=_vol_a.index, y=_vol_a, name=ticker_a,
        mode="lines", line=dict(color="royalblue", width=1.8),
    ))
    fig_vol.add_trace(go.Scatter(
        x=_vol_b.index, y=_vol_b, name=ticker_b,
        mode="lines", line=dict(color="crimson", width=1.8),
    ))
    draw_event_markers(fig_vol, show_defaults, custom_events,
                       both.index.min(), both.index.max())
    fig_vol.update_layout(
        height=380, yaxis_title="Annualized Volatility",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 7 — Drawdown
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Drawdown")
st.caption(
    "Percentage decline from each stock's **running peak** price. "
    "Overlapping red/blue areas reveal periods of **simultaneous drawdown** — "
    "indicating correlated downside risk. Stocks that drawdown together offer less diversification benefit."
)

if ticker_a in prices_window.columns and ticker_b in prices_window.columns:
    def _drawdown(prices: pd.Series) -> pd.Series:
        cum_max = prices.cummax()
        return (prices - cum_max) / cum_max * 100   # in percent

    dd_a = _drawdown(prices_window[ticker_a])
    dd_b = _drawdown(prices_window[ticker_b])

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd_a.index, y=dd_a, name=ticker_a,
        mode="lines", fill="tozeroy",
        line=dict(color="royalblue", width=1.5),
        fillcolor="rgba(65,105,225,0.18)",
    ))
    fig_dd.add_trace(go.Scatter(
        x=dd_b.index, y=dd_b, name=ticker_b,
        mode="lines", fill="tozeroy",
        line=dict(color="crimson", width=1.5),
        fillcolor="rgba(220,20,60,0.18)",
    ))
    draw_event_markers(fig_dd, show_defaults, custom_events,
                       dd_a.index.min(), dd_a.index.max())
    fig_dd.update_layout(
        height=380, yaxis_title="Drawdown (%)",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 8 — MST position (placeholder)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("MST Network Position")
st.info(
    "**Coming soon.** This section will display where **{a}** and **{b}** sit in the "
    "Minimum Spanning Tree — including their nearest MST neighbors, the MST distance (path length) "
    "between them, and whether they belong to the same cluster. "
    "The `networkx` Cloud dependency is being resolved before this section is activated.".format(
        a=ticker_a, b=ticker_b
    )
)
