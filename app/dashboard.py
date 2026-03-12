"""StoNeCoAl — BIST-100 Correlation Analysis dashboard (Overview page)."""

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

_APP_DIR      = Path(__file__).resolve().parent   # app/
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# All loaders and shared utilities live in utils.py (single source of truth)
from utils import (  # noqa: E402
    PROJECT_ROOT, DATA_PROCESSED, DATA_RESULTS, DATA_RAW,
    load_adj_close, load_log_returns, load_summary_stats, load_batch_corr,
    load_coverage, load_top_bottom, load_metadata, load_fetch_metadata,
    load_xu100, load_linkage, load_dendrogram_order, load_cluster_assignments,
    load_mst_edges, load_mst_metrics,
    draw_event_markers, event_marker_manager_ui,
)

st.set_page_config(page_title="StoNeCoAl — BIST-100", layout="wide")
st.title("StoNeCoAl — BIST-100 Correlation Analysis")


@st.cache_data
def compute_corr_for_window(returns_json: str, min_periods: int):
    """Recompute correlation for a date-filtered window."""
    _ret = pd.read_json(returns_json, orient="split")
    return _ret.corr(method="pearson", min_periods=min_periods)


# ---------- sidebar navigation ----------

st.sidebar.markdown("## Pages")
st.sidebar.page_link("dashboard.py", label="📊 Overview", icon=None)
st.sidebar.page_link("pages/1_Pair_Analysis.py", label="🔗 Pair Analysis")
st.sidebar.markdown("---")
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
    st.caption(
        "Each bar shows the percentage of trading days with available price data for a given ticker. "
        "Stocks below the **90% threshold** (red dashed line) were excluded from the analysis to ensure "
        "statistical reliability. Longer bars indicate more complete price histories."
    )
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
    st.caption(
        "All stock prices are rebased to **100** at the start of the selected window, allowing direct "
        "comparison of relative performance. The bold **black line** is the XU100 (BIST-100 index). "
        "Stocks above 100 outperformed their starting price; stocks below 100 lost value. "
        "Convergence/divergence of lines hints at correlation or decoupling among stocks."
    )
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
    st.caption(
        "Key risk-return metrics for each stock computed from daily **log returns**. "
        "**Annualized return** and **volatility** are scaled by √252 trading days. "
        "**Skewness** < 0 means more frequent large drops; **kurtosis** > 3 (excess) signals fat tails "
        "— i.e., extreme daily moves occur more often than a normal distribution would predict."
    )
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
    st.caption(
        "Histogram of daily **log returns** for the selected ticker. A bell-shaped curve centered near zero "
        "indicates typical market behavior. Look for **fat tails** (bars far from center) signaling extreme "
        "moves, and **skewness** (asymmetry) showing whether the stock tends to have larger up or down days."
    )
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


# ---------- correlation heatmap (cluster-reordered) ----------

st.markdown("---")
st.subheader("Pearson Correlation Heatmap")
st.caption(
    "A pairwise **Pearson correlation matrix** of daily log returns. Values range from **-1** (perfect inverse movement) "
    "to **+1** (perfect co-movement). Red/warm colors indicate positive correlation; blue/cool colors indicate negative. "
    "When **'Reorder by hierarchical clustering'** is enabled, tickers are rearranged so that highly correlated groups "
    "appear as block-diagonal clusters along the main diagonal — making sector-driven patterns easy to spot."
)

# Recompute correlation for the selected window
corr = compute_corr_for_window(returns.to_json(orient="split", date_format="iso"), dynamic_min_periods)

# Try to reorder by dendrogram leaf order for better block-diagonal structure
leaf_order = load_dendrogram_order()
use_clustering_order = st.checkbox("Reorder by hierarchical clustering", value=True)

if use_clustering_order and leaf_order is not None:
    # Only use tickers that exist in both the correlation matrix and the leaf order
    valid_order = [t for t in leaf_order if t in corr.columns]
    if valid_order:
        corr_display = corr.loc[valid_order, valid_order]
    else:
        corr_display = corr
else:
    corr_display = corr

fig_heat = go.Figure(
    data=go.Heatmap(
        z=corr_display.values,
        x=corr_display.columns.tolist(),
        y=corr_display.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
    )
)
n_tickers = len(corr_display)
fig_heat.update_layout(
    height=max(600, n_tickers * 8),
    width=max(600, n_tickers * 8),
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(tickfont=dict(size=7), dtick=1),
    yaxis=dict(tickfont=dict(size=7), dtick=1, autorange="reversed"),
)
st.plotly_chart(fig_heat, use_container_width=True)


# ---------- dendrogram & cluster view ----------

st.markdown("---")
col_dendro, col_clusters = st.columns([3, 2])

with col_dendro:
    st.subheader("Dendrogram")
    st.caption(
        "A **hierarchical clustering dendrogram** built from the correlation-based distance matrix "
        "(d = √(2(1-ρ))). Stocks that merge at **lower heights** (bottom of the tree) are more similar in their "
        "return behavior. Vertical lines show the merge distance — a large jump indicates two distinct groups being joined. "
        "This reveals the natural grouping structure of the market, often aligning with BIST industry sectors."
    )
    Z_loaded, labels_loaded = load_linkage()
    if Z_loaded is not None:
        # Build dendrogram using plotly figure_factory
        fig_dendro = ff.create_dendrogram(
            np.eye(len(labels_loaded)),
            orientation="bottom",
            labels=labels_loaded,
            linkagefun=lambda x: Z_loaded,
        )
        fig_dendro.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=10, b=100),
            xaxis=dict(tickfont=dict(size=7), tickangle=-90),
            yaxis_title="Distance",
        )
        st.plotly_chart(fig_dendro, use_container_width=True)
    else:
        st.info("Run the clustering pipeline to generate the dendrogram.")

with col_clusters:
    st.subheader("Cluster Memberships")
    st.caption(
        "Stocks are grouped into clusters by cutting the dendrogram at an **automatic threshold** "
        "(70% of the maximum merge distance). Each row shows a stock's cluster assignment and its BIST sector. "
        "The **Cluster vs Sector** cross-tab below reveals whether correlation-driven clusters coincide with "
        "official sector classifications — a mismatch may indicate hidden economic linkages."
    )
    cluster_df = load_cluster_assignments()
    if not cluster_df.empty:
        n_clusters = cluster_df["cluster_id"].nunique()
        st.metric("Number of Clusters", n_clusters)

        # Display cluster table sorted by cluster then ticker
        display_clusters = cluster_df.sort_values(
            ["cluster_id", "ticker"]
        ).reset_index(drop=True)
        st.dataframe(display_clusters, use_container_width=True, height=400, hide_index=True)

        # Cluster-sector cross-tab
        st.markdown("**Cluster vs Sector**")
        if "sector" in cluster_df.columns:
            crosstab = pd.crosstab(
                cluster_df["cluster_id"], cluster_df["sector"]
            )
            st.dataframe(crosstab, use_container_width=True)
    else:
        st.info("Run the clustering pipeline to see cluster assignments.")


# ---------- MST network graph ----------

st.markdown("---")
st.subheader("Minimum Spanning Tree")
st.caption(
    "The **Minimum Spanning Tree (MST)** connects all stocks using the shortest total correlation-based distance, "
    "filtering out redundant links to reveal the backbone structure of the market. Each node is a stock, colored by "
    "**sector** and sized by **degree** (number of connections). Hub nodes with many connections are influential — "
    "they act as bridges transmitting correlation across the market. Edge lengths reflect correlation distance: "
    "shorter edges = higher correlation. The **betweenness centrality** in the table quantifies how often a stock "
    "lies on the shortest path between other stocks — high values indicate systemic importance."
)

mst_edges = load_mst_edges()
mst_metrics = load_mst_metrics()

if not mst_edges.empty and not mst_metrics.empty and HAS_NETWORKX:
    col_mst_graph, col_mst_table = st.columns([3, 2])

    with col_mst_graph:
        G = nx.Graph()
        sector_map = dict(zip(mst_metrics["ticker"], mst_metrics["sector"]))
        degree_map = dict(zip(mst_metrics["ticker"], mst_metrics["degree"]))

        for _, row in mst_edges.iterrows():
            G.add_edge(row["source"], row["target"], weight=row["distance"])

        # Use spring layout with distance-based weights
        pos = nx.spring_layout(G, seed=42, k=2.0, iterations=100)

        # Build edge traces
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # Build node traces — color by sector, size by degree
        sectors = list(set(sector_map.values()) - {None, np.nan})
        sectors.sort()
        color_map = {s: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
                     for i, s in enumerate(sectors)}

        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            sec = sector_map.get(node, "Unknown")
            deg = degree_map.get(node, 1)
            node_text.append(f"{node}<br>Sector: {sec}<br>Degree: {deg}")
            node_color.append(color_map.get(sec, "#999"))
            node_size.append(8 + deg * 5)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=[n for n in G.nodes()],
            textposition="top center",
            textfont=dict(size=7),
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=1, color="white"),
            ),
        )

        fig_mst = go.Figure(data=[edge_trace, node_trace])
        fig_mst.update_layout(
            showlegend=False,
            height=700,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
        )

        # Add sector color legend manually
        for sec in sectors:
            fig_mst.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=color_map[sec]),
                name=sec,
                showlegend=True,
            ))
        fig_mst.update_layout(showlegend=True, legend=dict(font=dict(size=9)))

        st.plotly_chart(fig_mst, use_container_width=True)

    with col_mst_table:
        st.markdown("**Hub Stocks (by degree)**")
        display_metrics = mst_metrics.copy()
        display_metrics["betweenness_centrality"] = display_metrics[
            "betweenness_centrality"
        ].map(lambda x: f"{x:.4f}")
        st.dataframe(display_metrics, use_container_width=True, height=500, hide_index=True)
elif not HAS_NETWORKX:
    st.warning("Install `networkx` to display the MST network graph (`pip install networkx`).")
else:
    st.info("Run the clustering pipeline to generate the MST network.")


# ---------- rolling correlation analysis ----------

st.markdown("---")
st.subheader("Rolling Correlation Analysis")
st.caption(
    "Tracks how **pairwise correlations evolve over time** using a sliding window. This reveals whether stocks "
    "move together more during crises (correlation spikes) or decouple during calm periods. "
    "**Window**: number of past trading days used for each calculation. "
    "**Step**: how many days to skip between calculations (higher = faster but coarser). "
    "**Method**: Pearson measures linear co-movement; Spearman captures monotonic (rank-based) relationships. "
    "**Window type**: *Rolling* = fixed-size lookback; *Expanding* = growing window from start; *EWM* = exponentially "
    "weighted, giving more weight to recent data."
)

from src.rolling_correlation import (
    compute_rolling_market_stats,
    compute_rolling_pair_correlation,
    compute_rolling_sector_stats,
)

rc_col1, rc_col2, rc_col3, rc_col4 = st.columns(4)
with rc_col1:
    rc_window = st.selectbox("Window (days)", [60, 120, 252, 504], index=2, key="rc_win")
with rc_col2:
    rc_step = st.selectbox("Step", [1, 5, 21], index=1, key="rc_step",
                           format_func=lambda x: {1: "1 (daily)", 5: "5 (weekly)", 21: "21 (monthly)"}[x])
with rc_col3:
    rc_method = st.selectbox("Method", ["pearson", "spearman"], key="rc_method")
with rc_col4:
    rc_window_type = st.selectbox("Window type", ["rolling", "expanding", "ewm"], key="rc_wtype")

rc_expanding = rc_window_type == "expanding"

# Event markers — shared utility, scoped to "rc" prefix
show_defaults, custom_events = event_marker_manager_ui("rc", min_date, max_date)

tab_market, tab_pair, tab_sector = st.tabs(["Market Overview", "Pair Analysis", "Sector Breakdown"])

# --- Tab 1: Market correlation over time ---
with tab_market:
    st.caption(
        "**Market Overview** — Aggregates all pairwise correlations into summary statistics at each time step. "
        "The **blue line** is the mean correlation, the **orange dotted line** is the median, and the **shaded band** "
        "shows the interquartile range (Q25–Q75). When the band narrows and the mean rises, it signals a "
        "**correlation regime shift** — typically during market stress when stocks move in lockstep. "
        "Red dashed vertical lines mark major macro events (COVID, earthquakes, etc.)."
    )

    @st.cache_data
    def _compute_market_stats(ret_json, window, step, method, expanding):
        ret = pd.read_json(ret_json, orient="split")
        return compute_rolling_market_stats(
            ret, window=window, step=step, method=method, expanding=expanding,
        )

    with st.spinner("Computing rolling market stats..."):
        market_stats = _compute_market_stats(
            returns.to_json(orient="split", date_format="iso"),
            rc_window, rc_step, rc_method, rc_expanding,
        )

    if not market_stats.empty:
        fig_rc = go.Figure()

        # IQR band (q25 - q75)
        fig_rc.add_trace(go.Scatter(
            x=market_stats.index, y=market_stats["q75_corr"],
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig_rc.add_trace(go.Scatter(
            x=market_stats.index, y=market_stats["q25_corr"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(99,110,250,0.15)", name="IQR (Q25-Q75)",
        ))

        # Average and median
        fig_rc.add_trace(go.Scatter(
            x=market_stats.index, y=market_stats["avg_corr"],
            mode="lines", name="Mean", line=dict(color="royalblue", width=2),
        ))
        fig_rc.add_trace(go.Scatter(
            x=market_stats.index, y=market_stats["median_corr"],
            mode="lines", name="Median", line=dict(color="orange", width=1.5, dash="dot"),
        ))

        draw_event_markers(
            fig_rc, show_defaults, custom_events,
            market_stats.index.min(), market_stats.index.max(),
        )

        fig_rc.update_layout(
            height=450,
            yaxis_title=f"Pairwise {rc_method.title()} Correlation",
            xaxis_title="Date",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_rc, use_container_width=True)

        # Min/max range chart
        with st.expander("Min / Max range"):
            fig_mm = go.Figure()
            fig_mm.add_trace(go.Scatter(
                x=market_stats.index, y=market_stats["max_corr"],
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig_mm.add_trace(go.Scatter(
                x=market_stats.index, y=market_stats["min_corr"],
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(255,127,14,0.15)", name="Min-Max Range",
            ))
            fig_mm.add_trace(go.Scatter(
                x=market_stats.index, y=market_stats["avg_corr"],
                mode="lines", name="Mean", line=dict(color="royalblue", width=1.5),
            ))
            fig_mm.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                                 yaxis_title="Correlation")
            st.plotly_chart(fig_mm, use_container_width=True)
    else:
        st.warning("Not enough data for the selected window size.")

# --- Tab 2: Pair rolling correlation ---
with tab_pair:
    st.caption(
        "**Pair Analysis** — Select two stocks to see how their correlation changes over time. "
        "A correlation near **+1** means they consistently move together; near **0** means no linear relationship; "
        "near **-1** means they move in opposite directions. The **normalized price comparison** (expandable below) "
        "overlays both stocks' rebased prices so you can visually verify convergence/divergence periods."
    )
    ticker_list = sorted(returns.columns.tolist())
    pc1, pc2 = st.columns(2)
    with pc1:
        pair_a = st.selectbox("Ticker A", ticker_list, index=0, key="pair_a")
    with pc2:
        default_b = min(1, len(ticker_list) - 1)
        pair_b = st.selectbox("Ticker B", ticker_list, index=default_b, key="pair_b")

    if pair_a and pair_b and pair_a != pair_b:
        @st.cache_data
        def _compute_pair(ret_json, a, b, window, method, wtype):
            ret = pd.read_json(ret_json, orient="split")
            return compute_rolling_pair_correlation(
                ret, a, b, window=window, method=method, window_type=wtype,
            )

        with st.spinner("Computing pair correlation..."):
            pair_corr = _compute_pair(
                returns.to_json(orient="split", date_format="iso"),
                pair_a, pair_b, rc_window, rc_method, rc_window_type,
            )

        fig_pair = go.Figure()
        fig_pair.add_trace(go.Scatter(
            x=pair_corr.index, y=pair_corr.values,
            mode="lines", name=f"{pair_a} — {pair_b}",
            line=dict(color="royalblue", width=1.5),
        ))
        fig_pair.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

        _valid = pair_corr.dropna()
        if not _valid.empty:
            draw_event_markers(
                fig_pair, show_defaults, custom_events,
                _valid.index.min(), _valid.index.max(),
            )

        fig_pair.update_layout(
            height=400,
            yaxis_title=f"{rc_method.title()} Correlation",
            yaxis=dict(range=[-1.05, 1.05]),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_pair, use_container_width=True)

        # Show normalized price spread below
        if pair_a in prices_window.columns and pair_b in prices_window.columns:
            with st.expander("Normalized price comparison"):
                pa = prices_window[pair_a] / prices_window[pair_a].iloc[0] * 100
                pb = prices_window[pair_b] / prices_window[pair_b].iloc[0] * 100
                fig_spread = go.Figure()
                fig_spread.add_trace(go.Scatter(x=pa.index, y=pa, name=pair_a, mode="lines"))
                fig_spread.add_trace(go.Scatter(x=pb.index, y=pb, name=pair_b, mode="lines"))
                fig_spread.update_layout(height=300, yaxis_title="Normalized (100)",
                                         margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_spread, use_container_width=True)
    elif pair_a == pair_b:
        st.info("Select two different tickers.")

# --- Tab 3: Sector breakdown ---
with tab_sector:
    st.caption(
        "**Sector Breakdown** — Compares average **intra-sector** correlation (stocks within the same BIST sector) "
        "vs. **inter-sector** correlation (stocks from different sectors). Intra-sector correlation is typically "
        "higher because companies in the same industry share common risk factors. When the two lines converge, "
        "it suggests a market-wide factor (macro shock, sentiment) is dominating sector-specific effects. "
        "Expand **'Per-sector intra-correlation'** to see which sectors are internally cohesive over time."
    )
    cluster_df_for_sectors = load_cluster_assignments()
    if not cluster_df_for_sectors.empty and "sector" in cluster_df_for_sectors.columns:
        sec_map = dict(zip(cluster_df_for_sectors["ticker"], cluster_df_for_sectors["sector"]))

        @st.cache_data
        def _compute_sector(ret_json, sec_map_items, window, step, method):
            ret = pd.read_json(ret_json, orient="split")
            return compute_rolling_sector_stats(
                ret, dict(sec_map_items), window=window, step=step, method=method,
            )

        with st.spinner("Computing sector stats..."):
            sector_stats = _compute_sector(
                returns.to_json(orient="split", date_format="iso"),
                tuple(sec_map.items()), rc_window, rc_step, rc_method,
            )

        if not sector_stats.empty:
            # Intra vs inter
            fig_sec = go.Figure()
            fig_sec.add_trace(go.Scatter(
                x=sector_stats.index, y=sector_stats["intra_sector_avg"],
                mode="lines", name="Intra-Sector Avg",
                line=dict(color="crimson", width=2),
            ))
            fig_sec.add_trace(go.Scatter(
                x=sector_stats.index, y=sector_stats["inter_sector_avg"],
                mode="lines", name="Inter-Sector Avg",
                line=dict(color="steelblue", width=2),
            ))

            draw_event_markers(
                fig_sec, show_defaults, custom_events,
                sector_stats.index.min(), sector_stats.index.max(),
            )

            fig_sec.update_layout(
                height=400,
                yaxis_title="Average Correlation",
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_sec, use_container_width=True)

            # Per-sector breakdown
            intra_cols = [c for c in sector_stats.columns if c.startswith("intra_") and c not in ("intra_sector_avg",)]
            if intra_cols:
                with st.expander("Per-sector intra-correlation"):
                    fig_per = go.Figure()
                    for col in intra_cols:
                        sector_name = col.replace("intra_", "")
                        fig_per.add_trace(go.Scatter(
                            x=sector_stats.index, y=sector_stats[col],
                            mode="lines", name=sector_name,
                        ))
                    fig_per.update_layout(height=400, yaxis_title="Intra-Sector Correlation",
                                          margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_per, use_container_width=True)
        else:
            st.warning("Not enough data for sector stats with this window.")
    else:
        st.info("Run the clustering pipeline to enable sector breakdown.")


# ---------- top/bottom pairs & corr distribution ----------

st.markdown("---")
col_pairs, col_dist = st.columns(2)

with col_pairs:
    st.subheader("Top 10 / Bottom 10 Correlated Pairs")
    st.caption(
        "The **most correlated** pairs move almost identically — often from the same sector or holding-subsidiary "
        "relationships. The **least correlated** (or most negatively correlated) pairs move independently or inversely, "
        "making them candidates for **portfolio diversification**. Sector columns help identify whether high correlation "
        "is sector-driven or reflects a deeper economic link."
    )
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
    st.caption(
        "Distribution of all unique pairwise correlations (upper triangle of the matrix). "
        "A right-skewed distribution centered above zero is typical for stocks in the same national market — "
        "most pairs have positive co-movement due to shared macro factors. The **red dashed line** is the mean "
        "and the **blue dotted line** is the median. A wide spread indicates a diverse market; "
        "a narrow, high-mean distribution signals high systemic correlation."
    )
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
st.subheader("Market Summary")
st.caption(
    "Aggregate statistics of the full-period pairwise correlation matrix. "
    "**Avg Pairwise Corr** is the market-wide average co-movement — higher values mean the market behaves "
    "more as a single unit. **Std Dev** measures dispersion of correlations: low std dev = uniform correlation, "
    "high std dev = mixture of tightly and loosely linked stocks. **Min/Max** show the extremes."
)
market_summary = pipe_meta.get("market_summary", {})
if market_summary:
    cols = st.columns(5)
    cols[0].metric("Avg Pairwise Corr", f"{market_summary.get('avg_pairwise_corr', 0):.4f}")
    cols[1].metric("Median", f"{market_summary.get('median_pairwise_corr', 0):.4f}")
    cols[2].metric("Std Dev", f"{market_summary.get('std_pairwise_corr', 0):.4f}")
    cols[3].metric("Min", f"{market_summary.get('min_pairwise_corr', 0):.4f}")
    cols[4].metric("Max", f"{market_summary.get('max_pairwise_corr', 0):.4f}")
