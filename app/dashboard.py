"""StoNeCoAl — BIST-100 Correlation Analysis dashboard."""

import io
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

from src.rolling_correlation import (  # noqa: E402
    compute_rolling_market_stats,
    compute_rolling_pair_correlation,
    compute_rolling_sector_stats,
    compute_window_correlation,
)

from utils import (  # noqa: E402
    PROJECT_ROOT, DATA_PROCESSED, DATA_RESULTS, DATA_RAW,
    load_adj_close, load_log_returns, load_summary_stats, load_batch_corr,
    load_coverage, load_top_bottom, load_metadata, load_fetch_metadata,
    load_xu100, load_linkage, load_dendrogram_order, load_cluster_assignments,
    load_mst_edges, load_mst_metrics, load_dislocation_candidates, load_anomalies,
    load_rolling_market_stats_precomputed, load_rolling_sector_stats_precomputed,
    draw_event_markers, event_marker_manager_ui,
    get_colors, SECTOR_PALETTE, CHART_LAYOUT, apply_chart_style, inject_custom_css,
    section_header, render_chart,
)
from chart_themes import render_theme_sidebar  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Cached computation helpers (module-level to avoid Streamlit re-registration)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def _compute_corr(returns_json: str, min_periods: int, method: str):
    _ret = pd.read_json(io.StringIO(returns_json), orient="split")
    return _ret.corr(method=method, min_periods=min_periods)


@st.cache_data
def _pit_corr(ret_json, end_date_str, window, method):
    ret = pd.read_json(io.StringIO(ret_json), orient="split")
    return compute_window_correlation(
        ret, pd.Timestamp(end_date_str), window=window, method=method,
    )


@st.cache_data
def _mst_layout(edges_json):
    _G = nx.Graph()
    for _, r in pd.read_json(io.StringIO(edges_json), orient="split").iterrows():
        _G.add_edge(r["source"], r["target"], weight=r["distance"])
    return nx.kamada_kawai_layout(_G, weight="weight")


@st.cache_data
def _compute_market_stats(ret_json, window, step, method, expanding):
    ret = pd.read_json(io.StringIO(ret_json), orient="split")
    return compute_rolling_market_stats(
        ret, window=window, step=step, method=method, expanding=expanding,
    )


@st.cache_data
def _compute_pair(ret_json, a, b, window, method, wtype):
    ret = pd.read_json(io.StringIO(ret_json), orient="split")
    return compute_rolling_pair_correlation(
        ret, a, b, window=window, method=method, window_type=wtype,
    )


@st.cache_data
def _compute_sector(ret_json, sec_map_items, window, step, method):
    ret = pd.read_json(io.StringIO(ret_json), orient="split")
    return compute_rolling_sector_stats(
        ret, dict(sec_map_items), window=window, step=step, method=method,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Page config & global styling
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="StoNeCoAl — BIST-100",
    page_icon="<svg xmlns='http://www.w3.org/2000/svg'/>",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

with st.sidebar:
    render_theme_sidebar()

# ══════════════════════════════════════════════════════════════════════════════
# Top Header & Navigation
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<div style='display:flex; align-items:center; gap:12px; padding:0; margin:0;'>"
    "<span style='font-size:1.3rem; font-weight:800; letter-spacing:-0.02em; "
    "color:#2B2D42;'>StoNeCoAl</span>"
    "<span style='font-size:0.72rem; color:#8D99AE; letter-spacing:0.06em;'>"
    "BIST-100 NETWORK ANALYSIS</span></div>",
    unsafe_allow_html=True,
)

# Handle deferred navigation from cross-page jump buttons
if st.session_state.pop("_goto_pair_analysis", False):
    st.session_state["nav_page"] = "Pair Analysis"

_nav = st.segmented_control(
    "Navigate",
    ["Market Overview", "Pair Analysis"],
    key="nav_page",
    default="Market Overview",
    label_visibility="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# Shared Data Loading
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
    with st.popover("Settings", icon=":material/settings:", use_container_width=True):
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        with st.popover("Data Freshness", icon=":material/info:"):
            fetch_meta = load_fetch_metadata()
            if fetch_meta:
                st.write(f"**Fetch:** {fetch_meta.get('timestamp', 'N/A')[:16]}")
                st.write(f"**Source:** {fetch_meta.get('source', 'N/A')}")
                st.write(f"**Tickers:** {fetch_meta.get('ticker_count', 'N/A')}")
                if fetch_meta.get("failures"):
                    st.write(f"**Failures:** {len(fetch_meta['failures'])}")
            val_path = DATA_PROCESSED / "validation_report.csv"
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

m1.metric("Tickers", f"{returns.shape[1]}")
m2.metric("Trading Days", f"{returns.shape[0]:,}")
m3.metric("Avg Correlation", f"{market_summary.get('avg_pairwise_corr', 0):.4f}")
m4.metric("Median Correlation", f"{market_summary.get('median_pairwise_corr', 0):.4f}")
m5.metric("Date Range", f"{start_dt.strftime('%Y-%m')} to {end_dt.strftime('%Y-%m')}")

# Pre-serialize returns once for all cached computations
_returns_json = returns.to_json(orient="split", date_format="iso")

# ══════════════════════════════════════════════════════════════════════════════
# Market Overview — Sub-Tab Layout
# ══════════════════════════════════════════════════════════════════════════════

tab_data, tab_corr, tab_cluster, tab_rolling, tab_pairs, tab_eee = st.tabs([
    "Data & Stats", "Correlation", "Clustering & Network",
    "Rolling Analysis", "Pairs & Dislocations", "EEE Analysis",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Data & Stats
# ══════════════════════════════════════════════════════════════════════════════

with tab_data:

    # ── Section 1: Coverage & Normalized Prices ─────────────────────────────
    with st.container(border=True):
        section_header(
            "Data Coverage & Price Performance",
            "Left: per-ticker data availability (90% threshold). "
            "Right: all prices rebased to 100 — the bold black line is XU100.",
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
                         title_key="mo_coverage", default_title="Data Coverage by Ticker")

        with col_right:
            norm_prices = prices_window.divide(prices_window.iloc[0]) * 100
            xu100 = load_xu100()
            if not xu100.empty:
                xu100_window = xu100.loc[start_dt:end_dt]
                if not xu100_window.empty:
                    norm_prices["XU100"] = xu100_window / xu100_window.iloc[0] * 100

            fig_prices = go.Figure()
            for col in norm_prices.columns:
                is_index = col == "XU100"
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

    # ── Section 2: Descriptive Stats & Return Distribution ──────────────────
    with st.container(border=True):
        section_header(
            "Descriptive Statistics & Returns",
            "Left: per-stock risk-return metrics from daily log returns. "
            "Right: histogram for a selected ticker — look for fat tails and skewness.",
        )

        col_stats, col_hist = st.columns([3, 2])

        with col_stats:
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
            st.dataframe(display_df, use_container_width=True, height=420)

        with col_hist:
            selected_ticker = st.selectbox("Ticker", sorted(returns.columns.tolist()))
            if selected_ticker:
                ticker_returns = returns[selected_ticker].dropna()
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=ticker_returns, nbinsx=80,
                    marker_color=get_colors()["primary"], opacity=0.75,
                    hovertemplate="Return: %{x:.4f}<br>Count: %{y}<extra></extra>",
                ))
                _mean_r = ticker_returns.mean()
                fig_hist.add_vline(x=_mean_r, line_dash="dash", line_color=get_colors()["secondary"],
                                   annotation_text=f"Mean: {_mean_r:.4f}", annotation_font_size=10)
                apply_chart_style(fig_hist, height=420, xaxis_title="Daily Log Return",
                                  yaxis_title="Frequency", showlegend=False,
                                  margin=dict(l=0, r=0, t=10, b=0))
                render_chart(fig_hist, chart_id="mo_hist", filename_base="return_distribution",
                             title_key="mo_hist", default_title="Return Distribution")

    # ── Section 3: Return Anomalies ─────────────────────────────────────────
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

    # ── Section 9: Market Summary ───────────────────────────────────────────
    with st.container(border=True):
        section_header("Market Summary")
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

    _heat_c1, _heat_c2 = st.columns(2)
    with _heat_c1:
        heat_method = st.selectbox(
            "Correlation method", ["pearson", "spearman"],
            key="heat_method",
        )
    with _heat_c2:
        use_clustering_order = st.checkbox("Reorder by hierarchical clustering", value=True)

    with st.status("Computing correlation matrix...", expanded=False) as _corr_st:
        corr = _compute_corr(_returns_json, dynamic_min_periods, heat_method)
        _corr_st.update(label="Correlation matrix ready", state="complete")

    leaf_order = load_dendrogram_order()

    if use_clustering_order and leaf_order is not None:
        valid_order = [t for t in leaf_order if t in corr.columns]
        corr_display = corr.loc[valid_order, valid_order] if valid_order else corr
    else:
        corr_display = corr

    _corr_heatmap_tab, _corr_pit_tab = st.tabs(["Heatmap", "Point-in-Time Snapshot"])

    # ── Sub-tab: Full-period heatmap ──────────────────────────────────────
    with _corr_heatmap_tab:
        with st.container(border=True):
            st.caption(
                "Pairwise correlation matrix of daily log returns. "
                "Toggle method and clustering reorder above.",
            )

            n_tickers = len(corr_display)
            fig_heat = go.Figure(data=go.Heatmap(
                z=corr_display.values,
                x=corr_display.columns.tolist(),
                y=corr_display.index.tolist(),
                colorscale=get_colors().get("heatmap_cs", "RdBu_r"), zmid=0, zmin=-1, zmax=1,
                hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
                colorbar=dict(title="Corr", thickness=15, len=0.6),
            ))
            apply_chart_style(fig_heat,
                height=max(700, n_tickers * 12),
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(tickfont=dict(size=7), dtick=1),
                yaxis=dict(tickfont=dict(size=7), dtick=1, autorange="reversed"),
            )
            render_chart(fig_heat, chart_id="mo_heatmap", filename_base="correlation_heatmap",
                         title_key="mo_heatmap", default_title="Correlation Matrix")

    # ── Sub-tab: Point-in-Time Correlation Snapshot ────────────────────────
    with _corr_pit_tab:
        with st.container(border=True):
            st.caption(
                "Slide to a specific date to see the correlation matrix for that rolling window. "
                "Useful for comparing market structure during crises vs calm periods."
            )

            pit_c1, pit_c2, pit_c3 = st.columns(3)
            with pit_c1:
                pit_window = st.selectbox(
                    "Window (days)", [60, 120, 252], index=2, key="pit_window",
                )
            with pit_c2:
                pit_method = st.selectbox(
                    "Method", ["pearson", "spearman"], key="pit_method",
                )
            with pit_c3:
                trading_dates = returns.index.tolist()
                valid_start = max(0, pit_window - 1)
                pit_date = st.select_slider(
                    "Snapshot date",
                    options=trading_dates[valid_start:] if len(trading_dates) > valid_start else trading_dates,
                    value=trading_dates[-1] if trading_dates else None,
                    format_func=lambda d: d.strftime("%Y-%m-%d"),
                    key="pit_date",
                )

            if pit_date is not None:
                pit_corr = _pit_corr(
                    _returns_json, pit_date.isoformat(), pit_window, pit_method,
                )

                if not pit_corr.empty:
                    if use_clustering_order and leaf_order is not None:
                        pit_valid = [t for t in leaf_order if t in pit_corr.columns]
                        pit_display = pit_corr.loc[pit_valid, pit_valid] if pit_valid else pit_corr
                    else:
                        pit_display = pit_corr

                    pit_mask = np.triu(np.ones(pit_display.shape, dtype=bool), k=1)
                    pit_vals = pit_display.values[pit_mask]
                    pit_vals = pit_vals[~np.isnan(pit_vals)]

                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric("Tickers in Window", len(pit_display))
                    pm2.metric("Mean Corr", f"{np.mean(pit_vals):.4f}")
                    pm3.metric("Median Corr", f"{np.median(pit_vals):.4f}")
                    pm4.metric("Std Dev", f"{np.std(pit_vals):.4f}")

                    n_pit = len(pit_display)
                    fig_pit = go.Figure(data=go.Heatmap(
                        z=pit_display.values,
                        x=pit_display.columns.tolist(),
                        y=pit_display.index.tolist(),
                        colorscale=get_colors().get("heatmap_cs", "RdBu_r"), zmid=0, zmin=-1, zmax=1,
                        hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
                        colorbar=dict(title="Corr", thickness=15, len=0.6),
                    ))
                    apply_chart_style(fig_pit,
                        height=max(700, n_pit * 12),
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(tickfont=dict(size=7), dtick=1),
                        yaxis=dict(tickfont=dict(size=7), dtick=1, autorange="reversed"),
                    )
                    render_chart(fig_pit, chart_id="mo_pit_heatmap", filename_base="pit_correlation",
                                 title_key="mo_pit_heatmap", default_title="Point-in-Time Correlation")
                else:
                    st.warning("Not enough data for the selected date and window size.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Clustering & Network
# ══════════════════════════════════════════════════════════════════════════════

with tab_cluster:

    # ── Section 4: Dendrogram & Cluster Assignments ─────────────────────────
    with st.container(border=True):
        section_header(
            "Hierarchical Clustering & Sector Validation",
            "Dendrogram built from d = sqrt(2(1-rho)). Stocks merging at lower heights "
            "have more similar return dynamics. Sector validation metrics (ARI, NMI) measure "
            "how well statistical clusters align with Borsa Istanbul's official sector classifications.",
        )

        col_dendro, col_clusters = st.columns([3, 2])

        with col_dendro:
            Z_loaded, labels_loaded = load_linkage()
            if Z_loaded is not None:
                fig_dendro = ff.create_dendrogram(
                    np.eye(len(labels_loaded)),
                    orientation="bottom",
                    labels=labels_loaded,
                    linkagefun=lambda x: Z_loaded,
                )
                for trace in fig_dendro.data:
                    trace.update(line=dict(color=get_colors()["primary"], width=1.2))
                apply_chart_style(fig_dendro,
                    height=500,
                    margin=dict(l=10, r=10, t=10, b=100),
                    xaxis=dict(tickfont=dict(size=7), tickangle=-90),
                    yaxis_title="Distance",
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

                display_clusters = cluster_df.sort_values(["cluster_id", "ticker"]).reset_index(drop=True)
                st.dataframe(display_clusters, use_container_width=True, height=350, hide_index=True)

                if "sector" in cluster_df.columns:
                    st.markdown("**Cluster vs Sector**")
                    crosstab = pd.crosstab(cluster_df["cluster_id"], cluster_df["sector"])
                    st.dataframe(crosstab, use_container_width=True)

                    try:
                        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                        _has_sklearn = True
                    except ImportError:
                        _has_sklearn = False

                    if _has_sklearn:
                        cluster_labels = cluster_df["cluster_id"].values
                        sector_labels = cluster_df["sector"].values
                        ari = adjusted_rand_score(sector_labels, cluster_labels)
                        nmi = normalized_mutual_info_score(sector_labels, cluster_labels)

                        st.markdown("**Sector Validation**")
                        st.caption(
                            "How well do statistical clusters match Borsa Istanbul's official "
                            "sector classifications? ARI and NMI range from 0 (random) to 1 (perfect match)."
                        )
                        sv1, sv2, sv3 = st.columns(3)
                        sv1.metric("Adjusted Rand Index", f"{ari:.3f}")
                        sv2.metric("Normalized Mutual Info", f"{nmi:.3f}")
                        sv3.metric("Sectors Represented", f"{cluster_df['sector'].nunique()}")

                        _bank_keywords = ["Bank", "Finans"]
                        bank_tickers = cluster_df[
                            cluster_df["sector"].str.contains("|".join(_bank_keywords), case=False, na=False)
                        ]
                        if not bank_tickers.empty:
                            bank_clusters = bank_tickers["cluster_id"].unique()
                            n_bank_clusters = len(bank_clusters)
                            bank_names = ", ".join(bank_tickers["ticker"].tolist())
                            if n_bank_clusters == 1:
                                st.success(
                                    f"**Bank sanity check passed.** All {len(bank_tickers)} financial tickers "
                                    f"({bank_names}) are in Cluster {bank_clusters[0]}."
                                )
                            else:
                                st.warning(
                                    f"**Bank sanity check:** {len(bank_tickers)} financial tickers span "
                                    f"{n_bank_clusters} clusters ({', '.join(str(c) for c in sorted(bank_clusters))}). "
                                    f"Tickers: {bank_names}"
                                )

                        st.markdown("**Cluster Purity**")
                        st.caption(
                            "Purity = fraction of the dominant sector within each cluster. "
                            "A purity of 1.0 means every stock in the cluster belongs to the same sector."
                        )
                        purity_rows = []
                        for cid, grp in cluster_df.groupby("cluster_id"):
                            sector_counts = grp["sector"].value_counts()
                            dominant = sector_counts.index[0]
                            purity = sector_counts.iloc[0] / len(grp)
                            purity_rows.append({
                                "Cluster": cid,
                                "Size": len(grp),
                                "Dominant Sector": dominant,
                                "Purity": f"{purity:.2f}",
                                "Members": ", ".join(sorted(grp["ticker"].tolist())),
                            })
                        purity_df = pd.DataFrame(purity_rows)
                        st.dataframe(purity_df, use_container_width=True, hide_index=True)
            else:
                st.info("Run the clustering pipeline to see cluster assignments.")

    # ── Section 5: MST Network ──────────────────────────────────────────────
    with st.container(border=True):
        section_header(
            "Minimum Spanning Tree",
            "The MST reveals the backbone correlation structure. Nodes are colored by "
            "sector and sized by degree. Hub stocks act as bridges across the market.",
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

                with st.status("Computing MST layout...", expanded=False) as _mst_st:
                    pos = _mst_layout(mst_edges.to_json(orient="split"))
                    _mst_st.update(label="MST layout ready", state="complete")

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
                    node_text.append(f"<b>{node}</b><br>Sector: {sec}<br>Degree: {deg}")
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

            with col_mst_table:
                st.markdown("**Hub Stocks (by degree)**")
                display_metrics = mst_metrics.copy()
                display_metrics["betweenness_centrality"] = display_metrics[
                    "betweenness_centrality"
                ].map(lambda x: f"{x:.4f}")
                st.dataframe(display_metrics, use_container_width=True, height=500, hide_index=True)

                st.markdown("---")
                st.markdown("**Quick Jump to Pair Analysis**")
                _hub_tickers = mst_metrics.head(10)["ticker"].tolist()
                _sel = st.selectbox("Select hub stock", _hub_tickers, key="mst_hub_jump")
                if st.button("Analyze this pair", key="mst_jump_btn"):
                    st.session_state["pa_ticker_a"] = _sel
                    # Ensure ticker_b is set AND differs from ticker_a,
                    # otherwise Pair Analysis lands on the same-ticker
                    # degraded state. (The collision-resolver in
                    # pair_analysis.render will also catch this, but
                    # picking a meaningful partner here avoids the
                    # auto-snap surprise for the user.)
                    _existing_b = st.session_state.get("pa_ticker_b")
                    if _existing_b is None or _existing_b == _sel:
                        st.session_state["pa_ticker_b"] = next(
                            (t for t in _hub_tickers if t != _sel),
                            _sel,
                        )
                    st.session_state["_goto_pair_analysis"] = True
                    st.rerun()

        elif not HAS_NETWORKX:
            st.warning("Install `networkx` to display the MST network graph (`pip install networkx`).")
        else:
            st.info("Run the clustering pipeline to generate the MST network.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Rolling Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab_rolling:

    with st.container(border=True):
        section_header(
            "Rolling Correlation Analysis",
            "Track how pairwise correlations evolve over time. Spikes during crises "
            "indicate correlation regime shifts.",
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
        show_defaults, custom_events = event_marker_manager_ui("rc", min_date, max_date)

        tab_market, tab_pair, tab_sector = st.tabs(["Market Overview", "Pair Correlation", "Sector Breakdown"])

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
                    with st.status("Computing rolling market stats...", expanded=False) as _ms_st:
                        market_stats = _compute_market_stats(
                            _returns_json, rc_window, rc_step, rc_method, rc_expanding,
                        )
                        _ms_st.update(label="Market stats ready", state="complete")
            else:
                with st.status("Computing rolling market stats (custom params)...", expanded=False) as _ms_st:
                    market_stats = _compute_market_stats(
                        _returns_json, rc_window, rc_step, rc_method, rc_expanding,
                    )
                    _ms_st.update(label="Market stats ready", state="complete")
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
        with tab_pair:
            ticker_list = sorted(returns.columns.tolist())
            # Init-once + key= only (no index=) to avoid Streamlit's
            # "default value but also had its value set via the Session
            # State API" warning banner. Same fix as pair_analysis.py.
            if (
                "pair_a" not in st.session_state
                or st.session_state["pair_a"] not in ticker_list
            ):
                st.session_state["pair_a"] = ticker_list[0] if ticker_list else ""
            if (
                "pair_b" not in st.session_state
                or st.session_state["pair_b"] not in ticker_list
            ):
                st.session_state["pair_b"] = (
                    ticker_list[1] if len(ticker_list) > 1 else (ticker_list[0] if ticker_list else "")
                )
            pc1, pc2 = st.columns(2)
            with pc1:
                pair_a = st.selectbox("Ticker A", ticker_list, key="pair_a")
            with pc2:
                pair_b = st.selectbox("Ticker B", ticker_list, key="pair_b")

            if pair_a and pair_b and pair_a != pair_b:
                with st.status("Computing pair correlation...", expanded=False) as _pc_st:
                    pair_corr = _compute_pair(
                        _returns_json, pair_a, pair_b, rc_window, rc_method, rc_window_type,
                    )
                    _pc_st.update(label="Pair correlation ready", state="complete")

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

                if st.button(f"Open full Pair Analysis for {pair_a} / {pair_b}", key="pair_deep_dive"):
                    st.session_state["pa_ticker_a"] = pair_a
                    st.session_state["pa_ticker_b"] = pair_b
                    st.session_state["_goto_pair_analysis"] = True
                    st.rerun()

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
                st.info("Select two different tickers.")

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
                        with st.status("Computing sector stats...", expanded=False) as _ss_st:
                            sector_stats = _compute_sector(
                                _returns_json, tuple(sec_map.items()), rc_window, rc_step, rc_method,
                            )
                            _ss_st.update(label="Sector stats ready", state="complete")
                else:
                    with st.status("Computing sector stats (custom params)...", expanded=False) as _ss_st:
                        sector_stats = _compute_sector(
                            _returns_json, tuple(sec_map.items()), rc_window, rc_step, rc_method,
                        )
                        _ss_st.update(label="Sector stats ready", state="complete")
                    st.caption(
                        "Computed on-the-fly — sector precompute exists only for "
                        "`window=252`, `step=5`, `pearson`."
                    )

                if not sector_stats.empty:
                    fig_sec = go.Figure()
                    fig_sec.add_trace(go.Scatter(
                        x=sector_stats.index, y=sector_stats["intra_sector_avg"],
                        mode="lines", name="Intra-Sector Avg",
                        line=dict(color=get_colors()["secondary"], width=2.2),
                    ))
                    fig_sec.add_trace(go.Scatter(
                        x=sector_stats.index, y=sector_stats["inter_sector_avg"],
                        mode="lines", name="Inter-Sector Avg",
                        line=dict(color=get_colors()["primary"], width=2.2),
                    ))
                    draw_event_markers(fig_sec, show_defaults, custom_events,
                                       sector_stats.index.min(), sector_stats.index.max())
                    apply_chart_style(fig_sec, height=420, yaxis_title="Average Correlation")
                    render_chart(fig_sec, chart_id="mo_sector_corr", filename_base="sector_correlation",
                                 title_key="mo_sector_corr", default_title="Sector Correlation")

                    intra_cols = [c for c in sector_stats.columns
                                  if c.startswith("intra_") and c != "intra_sector_avg"]
                    if intra_cols:
                        if st.toggle("Show per-sector breakdown", key="mo_per_sector_toggle"):
                            fig_per = go.Figure()
                            for i, col in enumerate(intra_cols):
                                sector_name = col.replace("intra_", "")
                                fig_per.add_trace(go.Scatter(
                                    x=sector_stats.index, y=sector_stats[col],
                                    mode="lines", name=sector_name,
                                    line=dict(color=SECTOR_PALETTE[i % len(SECTOR_PALETTE)], width=1.5),
                                ))
                            apply_chart_style(fig_per, height=420,
                                              yaxis_title="Intra-Sector Correlation")
                            render_chart(fig_per, chart_id="mo_per_sector", filename_base="per_sector_corr",
                                         title_key="mo_per_sector", default_title="Per-Sector Correlation")
                else:
                    st.warning("Not enough data for sector stats with this window.")
            else:
                st.info("Run the clustering pipeline to enable sector breakdown.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Pairs & Dislocations
# ══════════════════════════════════════════════════════════════════════════════

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
                if st.button("Open in Pair Analysis", key="top_pair_btn"):
                    st.session_state["pa_ticker_a"] = top_pairs.iloc[_top_pair_idx]["ticker_1"]
                    st.session_state["pa_ticker_b"] = top_pairs.iloc[_top_pair_idx]["ticker_2"]
                    st.session_state["_goto_pair_analysis"] = True
                    st.rerun()

            with tab_bottom:
                st.dataframe(bottom_pairs, use_container_width=True, hide_index=True)
                _bot_pair_idx = st.selectbox(
                    "Select pair to analyze",
                    range(len(bottom_pairs)),
                    format_func=lambda i: f"{bottom_pairs.iloc[i]['ticker_1']} / {bottom_pairs.iloc[i]['ticker_2']} ({bottom_pairs.iloc[i]['correlation']:.4f})",
                    key="bot_pair_sel",
                )
                if st.button("Open in Pair Analysis", key="bot_pair_btn"):
                    st.session_state["pa_ticker_a"] = bottom_pairs.iloc[_bot_pair_idx]["ticker_1"]
                    st.session_state["pa_ticker_b"] = bottom_pairs.iloc[_bot_pair_idx]["ticker_2"]
                    st.session_state["_goto_pair_analysis"] = True
                    st.rerun()

        with col_dist:
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
            if st.button("Analyze in Pair Analysis", key="cand_pair_btn"):
                st.session_state["pa_ticker_a"] = _disp_cands.iloc[_cand_idx]["ticker_a"]
                st.session_state["pa_ticker_b"] = _disp_cands.iloc[_cand_idx]["ticker_b"]
                st.session_state["_goto_pair_analysis"] = True
                st.rerun()
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
