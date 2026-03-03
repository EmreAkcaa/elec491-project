"""StoNeCoAl — Single-page Streamlit dashboard for BIST-100 analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RESULTS = PROJECT_ROOT / "data" / "results"
DATA_RAW = PROJECT_ROOT / "data" / "raw"

st.set_page_config(page_title="StoNeCoAl — BIST-100", layout="wide")
st.title("StoNeCoAl — BIST-100 Correlation Analysis")


# ---------- helpers ----------

@st.cache_data
def load_adj_close():
    return pd.read_parquet(DATA_PROCESSED / "adj_close.parquet")


@st.cache_data
def load_log_returns():
    return pd.read_parquet(DATA_PROCESSED / "log_returns.parquet")


@st.cache_data
def load_summary_stats():
    return pd.read_parquet(DATA_RESULTS / "summary_stats.parquet")


@st.cache_data
def load_batch_corr():
    return pd.read_parquet(DATA_RESULTS / "pearson_corr.parquet")


@st.cache_data
def load_coverage():
    return pd.read_csv(DATA_PROCESSED / "coverage_report.csv")


@st.cache_data
def load_top_bottom():
    return pd.read_csv(DATA_RESULTS / "top_bottom_pairs.csv")


@st.cache_data
def load_metadata():
    meta_path = DATA_RESULTS / "pipeline_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_fetch_metadata():
    meta_path = DATA_RAW / "fetch_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_xu100():
    path = DATA_RAW / "xu100.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "Adj Close" in df.columns:
            return df["Adj Close"]
        elif isinstance(df.columns, pd.MultiIndex):
            return df.iloc[:, 0]
        return df.iloc[:, 0]
    return pd.Series(dtype=float)


@st.cache_data
def compute_corr_for_window(returns_json: str, min_periods: int):
    """Recompute correlation for a date-filtered window."""
    returns = pd.read_json(returns_json, orient="split")
    return returns.corr(method="pearson", min_periods=min_periods)


# ---------- sidebar ----------

st.sidebar.header("Settings")

adj_close = load_adj_close()
full_returns = load_log_returns()

min_date = adj_close.index.min().date()
max_date = adj_close.index.max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) == 2:
    start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_dt, end_dt = pd.Timestamp(min_date), pd.Timestamp(max_date)

# Filter data to selected window
returns = full_returns.loc[start_dt:end_dt]
prices_window = adj_close.loc[start_dt:end_dt]

# Dynamic min_periods for dashboard
window_length = len(returns)
dynamic_min_periods = max(30, int(window_length * 0.6))

# ---------- data freshness ----------

with st.sidebar.expander("Data Freshness"):
    fetch_meta = load_fetch_metadata()
    pipe_meta = load_metadata()
    if fetch_meta:
        st.write(f"**Fetch timestamp:** {fetch_meta.get('timestamp', 'N/A')}")
        st.write(f"**Source:** {fetch_meta.get('source', 'N/A')}")
        st.write(f"**Tickers:** {fetch_meta.get('ticker_count', 'N/A')}")
        if fetch_meta.get("failures"):
            st.write(f"**Failures:** {len(fetch_meta['failures'])}")

    # Validation status
    val_path = DATA_PROCESSED / "validation_report.csv"
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        n_pass = (val_df["status"] == "PASS").sum()
        n_total = len(val_df)
        st.write(f"**Validation:** {n_pass}/{n_total} passed")


# ---------- pipeline info ----------

st.markdown("---")
pipe_meta = load_metadata()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tickers (after filter)", f"{returns.shape[1]}/{pipe_meta.get('universe_count', '?')}")
with col2:
    st.metric("Trading Days (window)", f"{returns.shape[0]:,}")
with col3:
    st.metric("Last Run", pipe_meta.get("run_timestamp", "N/A")[:19] if pipe_meta.get("run_timestamp") else "N/A")


# ---------- data coverage ----------

st.markdown("---")
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Data Coverage")
    coverage = load_coverage()
    fig_cov = px.bar(
        coverage.sort_values("coverage_pct"),
        x="coverage_pct",
        y="ticker",
        orientation="h",
        labels={"coverage_pct": "Coverage %", "ticker": ""},
        height=max(400, len(coverage) * 8),
    )
    fig_cov.add_vline(x=0.90, line_dash="dash", line_color="red", annotation_text="90% threshold")
    fig_cov.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(dtick=1, tickfont=dict(size=7)))
    st.plotly_chart(fig_cov, use_container_width=True)

with col_right:
    st.subheader("Normalized Prices")
    # Normalize to 100 at start
    norm_prices = prices_window.divide(prices_window.iloc[0]) * 100
    # Add XU100 if available
    xu100 = load_xu100()
    if not xu100.empty:
        xu100_window = xu100.loc[start_dt:end_dt]
        if not xu100_window.empty:
            norm_xu100 = xu100_window / xu100_window.iloc[0] * 100
            norm_prices["XU100"] = norm_xu100

    fig_prices = go.Figure()
    for col in norm_prices.columns:
        line_width = 2.5 if col == "XU100" else 0.5
        opacity = 1.0 if col == "XU100" else 0.4
        color = "black" if col == "XU100" else None
        fig_prices.add_trace(
            go.Scatter(
                x=norm_prices.index,
                y=norm_prices[col],
                name=col,
                mode="lines",
                line=dict(width=line_width, color=color),
                opacity=opacity,
            )
        )
    fig_prices.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        yaxis_title="Normalized Price (base=100)",
    )
    st.plotly_chart(fig_prices, use_container_width=True)


# ---------- descriptive stats & return distribution ----------

st.markdown("---")
col_stats, col_hist = st.columns(2)

with col_stats:
    st.subheader("Descriptive Statistics")
    summary = load_summary_stats()
    display_cols = [
        "ticker", "count", "annualized_return", "annualized_vol",
        "skewness", "kurtosis", "min_return", "max_return",
    ]
    display_df = summary[display_cols].copy()
    for c in ["annualized_return", "annualized_vol", "min_return", "max_return"]:
        display_df[c] = display_df[c].map(lambda x: f"{x:.4f}")
    for c in ["skewness", "kurtosis"]:
        display_df[c] = display_df[c].map(lambda x: f"{x:.2f}")
    st.dataframe(display_df, use_container_width=True, height=400)

with col_hist:
    st.subheader("Return Distribution")
    selected_ticker = st.selectbox("Ticker", sorted(returns.columns.tolist()))
    if selected_ticker:
        ticker_returns = returns[selected_ticker].dropna()
        fig_hist = px.histogram(
            ticker_returns,
            nbins=80,
            labels={"value": "Log Return", "count": "Frequency"},
            title=f"{selected_ticker} Daily Log Returns",
        )
        fig_hist.update_layout(showlegend=False, margin=dict(l=0, r=20, t=40, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)


# ---------- correlation heatmap ----------

st.markdown("---")
st.subheader("Pearson Correlation Heatmap")

# Recompute for the selected window
corr = compute_corr_for_window(returns.to_json(orient="split", date_format="iso"), dynamic_min_periods)

fig_heat = go.Figure(
    data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
    )
)
n_tickers = len(corr)
fig_heat.update_layout(
    height=max(600, n_tickers * 8),
    width=max(600, n_tickers * 8),
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(tickfont=dict(size=7), dtick=1),
    yaxis=dict(tickfont=dict(size=7), dtick=1),
)
st.plotly_chart(fig_heat, use_container_width=True)


# ---------- top/bottom pairs & corr distribution ----------

st.markdown("---")
col_pairs, col_dist = st.columns(2)

with col_pairs:
    st.subheader("Top 10 / Bottom 10 Correlated Pairs")
    pairs = load_top_bottom()
    top_pairs = pairs[pairs["rank_type"] == "top"][
        ["ticker_1", "ticker_2", "sector_1", "sector_2", "correlation"]
    ]
    bottom_pairs = pairs[pairs["rank_type"] == "bottom"][
        ["ticker_1", "ticker_2", "sector_1", "sector_2", "correlation"]
    ]
    st.markdown("**Most Correlated**")
    st.dataframe(top_pairs, use_container_width=True, hide_index=True)
    st.markdown("**Least Correlated**")
    st.dataframe(bottom_pairs, use_container_width=True, hide_index=True)

with col_dist:
    st.subheader("Correlation Distribution")
    # Upper triangle
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper_vals = corr.where(mask).stack().values
    upper_vals = upper_vals[~np.isnan(upper_vals)]

    fig_corr_dist = px.histogram(
        x=upper_vals,
        nbins=60,
        labels={"x": "Pairwise Correlation", "count": "Frequency"},
    )
    mean_val = np.mean(upper_vals)
    median_val = np.median(upper_vals)
    fig_corr_dist.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean_val:.3f}")
    fig_corr_dist.add_vline(x=median_val, line_dash="dot", line_color="blue", annotation_text=f"Median: {median_val:.3f}")
    fig_corr_dist.update_layout(showlegend=False, margin=dict(l=0, r=20, t=0, b=0))
    st.plotly_chart(fig_corr_dist, use_container_width=True)


# ---------- market summary metric ----------

st.markdown("---")
market_summary = pipe_meta.get("market_summary", {})
if market_summary:
    cols = st.columns(5)
    cols[0].metric("Avg Pairwise Corr", f"{market_summary.get('avg_pairwise_corr', 0):.4f}")
    cols[1].metric("Median", f"{market_summary.get('median_pairwise_corr', 0):.4f}")
    cols[2].metric("Std Dev", f"{market_summary.get('std_pairwise_corr', 0):.4f}")
    cols[3].metric("Min", f"{market_summary.get('min_pairwise_corr', 0):.4f}")
    cols[4].metric("Max", f"{market_summary.get('max_pairwise_corr', 0):.4f}")
