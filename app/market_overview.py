"""Market Overview page content.

PHASE 2 — Stage 1 (multi-page migration). Extracted from
`app/dashboard.py` (formerly lines ~828–1977) into a standalone
module with a clean `render()` contract.

This page owns the 5 full-period static-analysis sub-tabs:
  - Data & Stats
  - Correlation (full-period heatmap)
  - Clustering & Network
  - Rolling Analysis (3 inner sub-tabs)
  - Pairs & Dislocations (finance-only)

Plus the page-header KPI strip with the date-range picker.
The page owns the `date_range` widget because the windowed
`returns` / `prices_window` slice it produces is consumed by
every sub-tab below.

Stage 2 (next) wraps this `render(...)` in a thin page script
(`app/pages/02_market_overview.py`) that the new
`app/main.py` router hooks into via `st.navigation(...)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from src.rolling_correlation import (
    compute_rolling_market_stats,
    compute_rolling_pair_correlation,
    compute_rolling_sector_stats,
)

from utils import (
    SECTOR_PALETTE, apply_chart_style,
    load_anomalies, load_batch_corr, load_cluster_assignments, load_coverage,
    load_dendrogram_order, load_dislocation_candidates, load_linkage,
    load_metadata, load_mst_edges, load_mst_metrics,
    load_rolling_market_stats_precomputed, load_rolling_sector_stats_precomputed,
    load_summary_stats, load_top_bottom, load_xu100,
    draw_event_markers, event_marker_manager_ui,
    get_colors, render_chart, render_matrix_heatmap, render_subtabs,
    section_header,
)
from chart_themes import render_theme_popover


def _cap(u, attr, default):
    """Defensive capability lookup. Re-imported here so market_overview.py
    is fully self-contained — same fallback semantic as dashboard.py."""
    return getattr(u, attr, default)


# ══════════════════════════════════════════════════════════════════════════════
# Cached computation helpers (module-level to avoid Streamlit re-registration)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _compute_corr(_returns: pd.DataFrame, cache_key: str, min_periods: int, method: str):
    """`_returns` underscore-prefix → Streamlit skips hashing the DataFrame;
    the explicit `cache_key` drives cache identity."""
    return _returns.corr(method=method, min_periods=min_periods)


@st.cache_data(show_spinner=False)
def _mst_layout(_edges: pd.DataFrame, cache_key: str):
    """Build a layout for the main MST. PHASE Y (Y2): try precomputed JSON
    layout from `data/<universe>/results/layouts/main_mst.json` FIRST;
    fall back to live `nx.spring_layout` / `nx.kamada_kawai_layout` when
    missing. Saves ~1–2 s on S&P 485-node MST after first paint."""
    from utils import load_mst_layout
    _G = nx.Graph()
    for _, r in _edges.iterrows():
        _G.add_edge(r["source"], r["target"], weight=r["distance"])

    precomputed = load_mst_layout("main_mst")
    if precomputed:
        graph_nodes = set(_G.nodes())
        pos_filtered = {n: precomputed[n] for n in graph_nodes if n in precomputed}
        if len(pos_filtered) == len(graph_nodes):
            return pos_filtered

    if _G.number_of_nodes() > 200:
        return nx.spring_layout(_G, weight="weight", iterations=80, seed=42)
    return nx.kamada_kawai_layout(_G, weight="weight")


def _heatmap_axis_tickfont(n: int) -> int:
    if n <= 80:
        return 7
    if n <= 200:
        return 5
    return 0


def _heatmap_axis_dtick(n: int) -> int:
    if n <= 80:
        return 1
    if n <= 200:
        return 5
    return max(1, n // 30)


def _heatmap_height(n: int, max_px: int = 1100) -> int:
    return min(max_px, max(700, n * 12))


@st.cache_data(show_spinner=False)
def _compute_market_stats(_returns: pd.DataFrame, cache_key: str, window, step, method, expanding):
    return compute_rolling_market_stats(
        _returns, window=window, step=step, method=method, expanding=expanding,
    )


@st.cache_data(show_spinner=False)
def _compute_pair(_returns: pd.DataFrame, cache_key: str, a, b, window, method, wtype, ewm_span=None):
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
# PHASE 2 / Stage 1: helpers parameterised so they don't close over module
# globals from the old single-script dashboard.py. Each accepts its
# data dependencies as explicit kwargs; widget state is still read from
# `st.session_state` directly (form-bound keys like `rc_*` survive across
# widget interactions naturally).


@st.fragment
def _render_correlation_heatmap(
    *,
    returns: pd.DataFrame,
    returns_cache_key: str,
    dynamic_min_periods: int,
    active_universe,
) -> None:
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

    corr = _compute_corr(returns, returns_cache_key, dynamic_min_periods, heat_method)
    leaf_order = load_dendrogram_order()

    if use_clustering_order and leaf_order is not None:
        valid_order = [t for t in leaf_order if t in corr.columns]
        corr_display = corr.loc[valid_order, valid_order] if valid_order else corr
    else:
        corr_display = corr

    with st.container(border=True):
        _series_lower = _cap(active_universe, 'series_label', 'Log return').lower()
        _samp_unit = (
            "trading days" if _cap(active_universe, 'domain', 'finance') == "finance"
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
                file_name=f"correlation_{active_universe.key}.csv",
                mime="text/csv",
                key="mo_heatmap_csv_dl",
            )


@st.fragment
def _render_rolling_pair(
    *,
    returns: pd.DataFrame,
    returns_cache_key: str,
    prices_window: pd.DataFrame,
    rc_window: int,
    rc_method: str,
    rc_window_type: str,
    show_defaults: bool,
    custom_events,
    item_label: str,
    items_label: str,
) -> None:
    """Rolling Analysis → Pair Correlation sub-tab. Owns pair_a/pair_b
    selectors. Shares session_state keys (pa_ticker_a / pa_ticker_b)
    with the Pair Analysis page so a pick made in either view carries
    over to the other."""
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
        pair_a = st.selectbox(f"{item_label} A", ticker_list, key="pa_ticker_a")
    with pc2:
        pair_b = st.selectbox(f"{item_label} B", ticker_list, key="pa_ticker_b")

    if pair_a and pair_b and pair_a != pair_b:
        # When window_type == "ewm", convert α → span (pandas: span = 2/α − 1).
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

        # Normalized price lines.
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
        st.info(f"Select two different {items_label.lower()}.")


@st.fragment
def _render_minmax_envelope_block(market_stats, rc_method_label: str) -> None:
    """Min/max envelope toggle + chart. Fragment-scoped so toggling doesn't
    cause a full script rerun (and therefore doesn't bump scroll position
    or trigger the cascade of side-effects that earlier produced sporadic
    nav-state weirdness)."""
    if not st.toggle("Show min/max envelope", key="mo_minmax_toggle"):
        return
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


@st.fragment
def _render_per_sector_breakdown_block(sector_stats, intra_cols, sector_label: str) -> None:
    """Per-sector breakdown toggle + chart. Fragment-scoped — same rationale
    as `_render_minmax_envelope_block`."""
    if not st.toggle(f"Show per-{sector_label.lower()} breakdown", key="mo_per_sector_toggle"):
        return
    fig_per = go.Figure()
    for i, col in enumerate(intra_cols):
        sector_name = col.replace("intra_", "")
        fig_per.add_trace(go.Scatter(
            x=sector_stats.index, y=sector_stats[col],
            mode="lines", name=sector_name,
            line=dict(color=SECTOR_PALETTE[i % len(SECTOR_PALETTE)], width=1.5),
        ))
    apply_chart_style(fig_per, height=420,
                      yaxis_title=f"Intra-{sector_label} Correlation")
    render_chart(fig_per, chart_id="mo_per_sector", filename_base="per_sector_corr",
                 title_key="mo_per_sector",
                 default_title=f"Per-{sector_label} Correlation")


# ══════════════════════════════════════════════════════════════════════════════
# Main page entry point
# ══════════════════════════════════════════════════════════════════════════════

def render(
    *,
    full_returns: pd.DataFrame,
    adj_close: pd.DataFrame,
    min_date,
    max_date,
    active_universe,
) -> None:
    """Render the Market Overview page.

    Args:
        full_returns: Universe-keyed log returns DataFrame, full history.
        adj_close: Universe-keyed adjusted close prices, full history.
        min_date: Earliest available date (date object).
        max_date: Latest available date (date object).
        active_universe: Universe dataclass for the active universe (used
            for capability flags, labels, sector_label, etc.).

    The page owns the date_range widget at the top of the header strip;
    the windowed `returns` / `prices_window` derived from it is what every
    sub-tab below consumes. `returns_cache_key` is built from the universe
    key + date endpoints + shape so all cached helpers (`_compute_corr`,
    `_compute_market_stats`, `_compute_pair`, `_compute_sector`) key on
    a cheap deterministic string.
    """
    pipe_meta = load_metadata()
    market_summary = pipe_meta.get("market_summary", {})

    # ── Header strip (PORT arda/ui-cleanup item 2: split into 2 rows) ────
    # Row 1: date range + theme popover (left side).
    # Row 2: 5 KPI cards across full width.
    # Same metrics, same order — only the DOM layout differs from the
    # prior 7-col single row.
    _date_col, _theme_col = st.columns([1.5, 0.9])

    with _theme_col:
        render_theme_popover()

    with _date_col:
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    if len(date_range) == 2:
        start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start_dt, end_dt = pd.Timestamp(min_date), pd.Timestamp(max_date)

    returns = full_returns.loc[start_dt:end_dt]
    prices_window = adj_close.loc[start_dt:end_dt]
    window_length = len(returns)
    dynamic_min_periods = max(30, int(window_length * 0.6))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(_cap(active_universe, 'items_label', 'Tickers'), f"{returns.shape[1]}")
    m2.metric(
        "Samples" if _cap(active_universe, 'domain', 'finance') == "neuroscience" else "Trading Days",
        f"{returns.shape[0]:,}",
    )
    m3.metric("Avg Correlation", f"{market_summary.get('avg_pairwise_corr', 0):.4f}")
    m4.metric("Median Correlation", f"{market_summary.get('median_pairwise_corr', 0):.4f}")
    m5.metric("Date Range", f"{start_dt.strftime('%Y-%m')} to {end_dt.strftime('%Y-%m')}")

    returns_cache_key = (
        f"{active_universe.key}:{start_dt.date().isoformat()}:"
        f"{end_dt.date().isoformat()}:{returns.shape[0]}x{returns.shape[1]}"
    )

    # ── Sub-tab layout (Pairs & Dislocations gated by capability) ───────
    _tab_labels = ["Data & Stats", "Correlation", "Clustering & Network", "Rolling Analysis"]
    if _cap(active_universe, 'has_pair_trading', True):
        _tab_labels.append("Pairs & Dislocations")
    _active_main_tab = render_subtabs("market_overview", tuple(_tab_labels))
    _show_tab_data    = _active_main_tab == "Data & Stats"
    _show_tab_corr    = _active_main_tab == "Correlation"
    _show_tab_cluster = _active_main_tab == "Clustering & Network"
    _show_tab_rolling = _active_main_tab == "Rolling Analysis"
    _show_tab_pairs   = (
        _active_main_tab == "Pairs & Dislocations"
        and "Pairs & Dislocations" in _tab_labels
    )

    # ── Tab 1 — Data & Stats ──────────────────────────────────────────────
    if _show_tab_data:
        _render_tab_data_stats(
            returns=returns, prices_window=prices_window,
            start_dt=start_dt, end_dt=end_dt,
            active_universe=active_universe,
            market_summary=market_summary,
        )

    # ── Tab 2 — Correlation ──────────────────────────────────────────────
    if _show_tab_corr:
        _render_correlation_heatmap(
            returns=returns,
            returns_cache_key=returns_cache_key,
            dynamic_min_periods=dynamic_min_periods,
            active_universe=active_universe,
        )

    # ── Tab 3 — Clustering & Network ─────────────────────────────────────
    if _show_tab_cluster:
        _render_tab_clustering(active_universe=active_universe)

    # ── Tab 4 — Rolling Analysis ─────────────────────────────────────────
    if _show_tab_rolling:
        _render_tab_rolling(
            returns=returns, returns_cache_key=returns_cache_key,
            prices_window=prices_window,
            active_universe=active_universe,
            min_date=min_date, max_date=max_date,
        )

    # ── Tab 5 — Pairs & Dislocations (finance only) ──────────────────────
    if _show_tab_pairs:
        _render_tab_pairs(
            returns=returns, returns_cache_key=returns_cache_key,
            dynamic_min_periods=dynamic_min_periods,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab-body helpers (auto-extracted from dashboard.py during PHASE 2 / Stage 1)
# ══════════════════════════════════════════════════════════════════════════════

def _render_tab_data_stats(
    *,
    returns: pd.DataFrame,
    prices_window: pd.DataFrame,
    start_dt,
    end_dt,
    active_universe,
    market_summary: dict,
) -> None:
    """Auto-extracted from dashboard.py during PHASE 2 / Stage 1.
    PORT arda/ui-cleanup item 1: removed three section_header subtitle
    paragraphs (Data Coverage, Descriptive Statistics, Return Anomalies)."""
    # ── Section 1: Coverage & Normalized Prices ─────────────────────────────
    with st.container(border=True):
        if _cap(active_universe, 'has_index_series', True):
            section_header("Data Coverage & Price Performance")
        else:
            section_header(
                f"Data Coverage & {_cap(active_universe, 'series_label', 'Log return')} Performance"
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
                         default_title=f"Data Coverage by {_cap(active_universe, 'item_label', 'Ticker')}")

        with col_right:
            if _cap(active_universe, 'has_index_series', True):
                # Financial universe: rebased prices + bold market-index overlay.
                norm_prices = prices_window.divide(prices_window.iloc[0]) * 100
                xu100 = load_xu100()
                _index_label = _cap(active_universe, 'index_ticker', 'XU100')  # "XU100" / "^GSPC"
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
                        name=f"Median ({len(_ticker_cols)} {_cap(active_universe, 'items_label', 'tickers').lower()})",
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
                            hovertemplate=f"{ch}: %{{y:.2f}} {_cap(active_universe, 'series_units', '')}<extra></extra>",
                        ))
                        cumulative_offset += spacing
                    apply_chart_style(
                        fig_volt, height=max(400, n_show * 38),
                        xaxis_title="Time (seconds)",
                        yaxis_title=(
                            f"{_cap(active_universe, 'series_label', 'Log return')} "
                            f"({_cap(active_universe, 'series_units', '')}) — stacked"
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
                            f"{_cap(active_universe, 'series_label', 'Log return')} Time-Series "
                            f"(first {n_samples / sample_rate_hz:.0f}s, "
                            f"{n_show} sample channels)"
                        ),
                    )

    # ── Section 2: Descriptive Stats & Distribution ─────────────────────────
    with st.container(border=True):
        _is_finance     = _cap(active_universe, 'domain', 'finance') == "finance"
        _item_label     = _cap(active_universe, 'item_label', 'Ticker')
        _items_label    = _cap(active_universe, 'items_label', 'Tickers')
        _series_label   = _cap(active_universe, 'series_label', 'Log return')
        _series_units   = _cap(active_universe, 'series_units', '')
        _series_axis    = f"{_series_label} ({_series_units})" if _series_units else _series_label

        # PORT arda/ui-cleanup item 1: descriptive-stats section_header subtitle removed.
        if _is_finance:
            section_header("Descriptive Statistics & Returns")
        else:
            section_header(f"Descriptive Statistics & {_series_label} Distribution")

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
    # PORT arda/ui-cleanup item 1: anomalies section_header subtitle removed.
    if _cap(active_universe, 'has_anomaly_detection', True):
      with st.container(border=True):
          section_header("Return Anomalies")

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
                          "date":         st.column_config.TextColumn("Date"),
                          "ticker":       st.column_config.TextColumn("Ticker"),
                          "return_value": st.column_config.NumberColumn("Return", format="%.4f"),
                          "abs_return":   st.column_config.NumberColumn("|Return|", format="%.4f"),
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
            "Market Summary" if _cap(active_universe, 'domain', 'finance') == "finance"
            else "Network Summary"
        )
        if market_summary:
            cols = st.columns(5)
            cols[0].metric("Avg Pairwise Corr", f"{market_summary.get('avg_pairwise_corr', 0):.4f}")
            cols[1].metric("Median", f"{market_summary.get('median_pairwise_corr', 0):.4f}")
            cols[2].metric("Std Dev", f"{market_summary.get('std_pairwise_corr', 0):.4f}")
            cols[3].metric("Min", f"{market_summary.get('min_pairwise_corr', 0):.4f}")
            cols[4].metric("Max", f"{market_summary.get('max_pairwise_corr', 0):.4f}")


def _render_tab_clustering(
    *,
    active_universe,
) -> None:
    """Auto-extracted from dashboard.py during PHASE 2 / Stage 1."""
    # ── Section 4: Dendrogram & Cluster Assignments ─────────────────────────
    with st.container(border=True):
        _items_cl   = _cap(active_universe, 'items_label', 'Tickers')
        _item_cl    = _cap(active_universe, 'item_label', 'Ticker')
        _sector_cl  = _cap(active_universe, 'sector_label', 'Sector')
        _series_cl  = _cap(active_universe, 'series_label', 'log return').lower()
        # PORT arda/ui-cleanup item 3: full-width multi-colored dendrogram
        # with cluster info BELOW (not right-column). Section_header subtitle
        # removed.
        section_header(f"Hierarchical Clustering & {_sector_cl} Validation")

        # Load cluster_df up front so we can derive color_threshold for
        # the dendrogram + still have it available for the cluster-info
        # block below.
        cluster_df = load_cluster_assignments()
        _n_clusters_hint = (
            cluster_df["cluster_id"].nunique() if not cluster_df.empty else 0
        )

        # Dendrogram — full-width row.
        Z_loaded, labels_loaded = load_linkage()
        if Z_loaded is not None:
            n_leaves = len(labels_loaded)
            # color_threshold tells ff.create_dendrogram to use its default
            # palette to color the top N clusters distinctly. We derive N
            # from cluster_df. If unknown, falls through to default (=auto).
            _color_threshold = None
            if _n_clusters_hint > 1 and Z_loaded.shape[0] >= _n_clusters_hint - 1:
                # Distance at which Ward forms exactly _n_clusters_hint clusters.
                # Pick the merge height just BELOW the (n - n_clusters)th merge
                # so that everything below threshold is a within-cluster branch
                # (gets a distinct color), and merges above are the cluster trunks.
                _cut_idx = max(0, Z_loaded.shape[0] - _n_clusters_hint)
                _color_threshold = float(Z_loaded[_cut_idx, 2])
            _dendro_kwargs = dict(
                orientation="bottom",
                labels=labels_loaded,
                linkagefun=lambda x: Z_loaded,
            )
            if _color_threshold is not None:
                _dendro_kwargs["color_threshold"] = _color_threshold
            fig_dendro = ff.create_dendrogram(np.eye(n_leaves), **_dendro_kwargs)
            # Keep the line-width bump but DON'T override colors — let
            # ff.create_dendrogram's default palette show through.
            for trace in fig_dendro.data:
                trace.line.width = 1.5
            # Hide per-leaf labels when there are too many to read.
            _show_leaf_labels = n_leaves <= 100
            _leaf_tickfont = 7 if n_leaves <= 100 else 1
            apply_chart_style(fig_dendro,
                height=750,
                margin=dict(l=10, r=10, t=10, b=100 if _show_leaf_labels else 30),
                xaxis=dict(
                    tickfont=dict(size=_leaf_tickfont),
                    tickangle=-90,
                    showticklabels=_show_leaf_labels,
                ),
                yaxis_title="Distance",
            )
            render_chart(fig_dendro, chart_id="mo_dendrogram", filename_base="dendrogram",
                         title_key="mo_dendrogram", default_title="Hierarchical Clustering")
        else:
            st.info("Run the clustering pipeline to generate the dendrogram.")

        # Cluster info block — full-width row BELOW the dendrogram.
        if not cluster_df.empty:
            n_clusters = cluster_df["cluster_id"].nunique()
            st.metric("Clusters Found", n_clusters)

            if "sector" in cluster_df.columns:
                # Universe-appropriate sanity-check banners.
                for group_label, members in (_cap(active_universe, 'sanity_check_groups', None) or {}).items():
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
        _items_mst    = _cap(active_universe, 'items_label', 'Tickers')
        _sector_mst   = _cap(active_universe, 'sector_label', 'Sector')
        _domain_mst   = _cap(active_universe, 'domain', 'finance')
        _bridge_scope = "across the market" if _domain_mst == "finance" else "across the network"
        # PORT arda/ui-cleanup item 3: MST section_header subtitle removed.
        section_header("Minimum Spanning Tree")

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
                f"{active_universe.key}:mst:{len(mst_edges)}",
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
            # PORT arda/ui-cleanup item 3: trailing MST caption removed.

            # Hub table behind an expander — was always-on in the right column
            # of a [3,2] split, now hidden by default to let the MST breathe.
            with st.expander(f"Hub {_items_mst} (by degree)", expanded=False):
                _item_mst = _cap(active_universe, 'item_label', 'Ticker')
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


def _render_tab_rolling(
    *,
    returns: pd.DataFrame,
    returns_cache_key: str,
    prices_window: pd.DataFrame,
    active_universe,
    min_date,
    max_date,
) -> None:
    """Auto-extracted from dashboard.py during PHASE 2 / Stage 1."""
    _is_finance_rc = _cap(active_universe, 'domain', 'finance') == "finance"
    _item_rc       = _cap(active_universe, 'item_label', 'Ticker')
    _items_rc      = _cap(active_universe, 'items_label', 'Tickers')
    _sector_rc     = _cap(active_universe, 'sector_label', 'Sector')

    with st.container(border=True):
        section_header(
            "Rolling Correlation Analysis",
            "Track how pairwise correlations evolve over time. " + (
                "Spikes during crises indicate correlation regime shifts."
                if _is_finance_rc
                else "Spikes can mark regime shifts or transient synchronization events."
            ),
        )

        # st.form gates the rolling widgets behind an explicit "Recompute"
        # submit button. Off-grid params (e.g. step=1, spearman, expanding)
        # cost up to 12 s on S&P; without the form, the user paid that cost
        # on every intermediate selectbox change. With the form, widget
        # changes accumulate locally, then a single submit triggers one
        # script rerun. Precomputed combos (window ∈ {60, 120, 252}, step=5,
        # pearson, rolling) still load instantly because the downstream
        # `_use_precomputed_market` check hits the parquet cache.
        # PHASE S (S2): window selector standardised to {60, 120, 252}
        # matching Time Machine. Removes the freeform 20–504 number_input
        # that was producing arbitrary cache-missing values.
        st.caption(
            ":material/touch_app: Configure window / step / method, then click "
            "**Recompute** to apply. Default combo (window=252, step=5, pearson, "
            "rolling) loads instantly; off-grid params take ~10–15 s on S&P. "
            "**EWM α** only applies when *Window type = ewm*."
        )

        with st.form("rolling_params", border=False):
            # PHASE S (S3): widgets in one row, Recompute button in its own
            # row below, horizontally centered. Previous layout crammed the
            # button into a 6th column and used a `&nbsp;` markdown hack to
            # force vertical alignment — brittle and ugly. New layout uses
            # natural form flow.
            _form_cols = st.columns(5)
            with _form_cols[0]:
                rc_window = int(st.selectbox(
                    "Window (days)" if _is_finance_rc else "Window (samples)",
                    [60, 120, 252],
                    index=2,
                    key="rc_win",
                    help="Trading days in each rolling window. All values hit the precomputed parquet (instant); standardised to match Time Machine.",
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

            # Recompute button in its own centered row.
            _btn_cols = st.columns([3, 2, 3])
            with _btn_cols[1]:
                st.form_submit_button(
                    "Recompute", use_container_width=True,
                )

        rc_expanding = rc_window_type == "expanding"
        # Event Markers popover lives OUTSIDE the form — popovers + forms
        # in the same row break Streamlit's column layout, and event-marker
        # toggles are cheap (no compute) so they don't need recompute gating.
        show_defaults, custom_events = event_marker_manager_ui("rc", min_date, max_date)

        _ra_market_label = "Market Overview" if _is_finance_rc else "Network Overview"
        # PHASE Y (Y1): render_subtabs replaces st.tabs so only the active
        # rolling-sub-tab body computes. Was the second-biggest perf cost
        # after Methods Lab (each tab's body ran every render even if hidden).
        _rolling_subtabs = (_ra_market_label, "Pair Correlation", f"{_sector_rc} Breakdown")
        _active_rolling_sub = render_subtabs("rolling_analysis", _rolling_subtabs)
        _show_rolling_market = _active_rolling_sub == _ra_market_label
        _show_rolling_pair   = _active_rolling_sub == "Pair Correlation"
        _show_rolling_sector = _active_rolling_sub == f"{_sector_rc} Breakdown"

        # ── Sub-Tab 1: Market correlation over time ─────────────────────────
        if _show_rolling_market:
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

                # PHASE S (S8): fragment-scoped toggle so flipping the envelope
                # doesn't trigger a full script rerun (which was causing scroll
                # jumps + sporadic nav weirdness).
                _render_minmax_envelope_block(market_stats, rc_method)
            else:
                st.warning("Not enough data for the selected window size.")

        # ── Sub-Tab 2: Pair rolling correlation ─────────────────────────────
        # Body lives in `_render_rolling_pair` fragment (defined at module
        # top). Changing pair_a / pair_b only reruns the fragment, not the
        # whole script.
        if _show_rolling_pair:
            _render_rolling_pair()

        # ── Sub-Tab 3: Sector breakdown ─────────────────────────────────────
        if _show_rolling_sector:
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
                        # PHASE S (S8): fragment-scoped toggle.
                        _render_per_sector_breakdown_block(sector_stats, intra_cols, _sector_rc)
                else:
                    st.warning(f"Not enough data for {_sector_rc.lower()} stats with this window.")
            else:
                st.info(f"Run the clustering pipeline to enable {_sector_rc.lower()} breakdown.")


def _render_tab_pairs(
    *,
    returns: pd.DataFrame,
    returns_cache_key: str,
    dynamic_min_periods: int,
) -> None:
    """Auto-extracted from dashboard.py during PHASE 2 / Stage 1."""

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

            # UX polish: dropped the "Analyze X/Y in Pair Analysis"
            # cross-page buttons + their selectbox companions. The button
            # was buggy (cross-page state plumbing through
            # `_goto_pair_analysis` was fragile across reruns) and the
            # selectbox added clicks. Users who want to deep-dive a pair
            # type its name directly in the Pair Analysis page.
            # Labels on every column so the table doesn't render raw
            # underscored names (`ticker_1`, `sector_2`, etc.).
            _pairs_column_config = {
                "ticker_1":    st.column_config.TextColumn("Ticker A"),
                "ticker_2":    st.column_config.TextColumn("Ticker B"),
                "sector_1":    st.column_config.TextColumn("Sector A"),
                "sector_2":    st.column_config.TextColumn("Sector B"),
                "correlation": st.column_config.NumberColumn("ρ", format="%.4f"),
            }
            tab_top, tab_bottom = st.tabs(["Most Correlated", "Least Correlated"])
            with tab_top:
                st.dataframe(
                    top_pairs, use_container_width=True, hide_index=True,
                    column_config=_pairs_column_config,
                )
            with tab_bottom:
                st.dataframe(
                    bottom_pairs, use_container_width=True, hide_index=True,
                    column_config=_pairs_column_config,
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

            # UX polish: dropped the candidate-pair selectbox + cross-page
            # "Analyze in Pair Analysis" button (buggy cross-page state).
            # The dataframe itself is sortable; users pick by sorting the
            # `rank_score` / `current_zscore` columns and typing the ticker
            # pair in Pair Analysis directly.
            # All columns labeled to avoid raw underscored headers
            # (`ticker_a`, `sector_b`, `n_signals` etc.) rendering as
            # demo-day eyesores. Labels mirror what the user sees in
            # Pair Analysis + top_bottom_pairs.
            st.dataframe(
                _disp_cands,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ticker_a":       st.column_config.TextColumn("Ticker A"),
                    "ticker_b":       st.column_config.TextColumn("Ticker B"),
                    "sector_a":       st.column_config.TextColumn("Sector A"),
                    "sector_b":       st.column_config.TextColumn("Sector B"),
                    "correlation":    st.column_config.NumberColumn("ρ", format="%.4f"),
                    "beta":           st.column_config.NumberColumn("β", format="%.4f"),
                    "half_life":      st.column_config.NumberColumn("Half-life (days)", format="%.1f"),
                    "current_zscore": st.column_config.NumberColumn("Current Z", format="%.3f"),
                    "n_signals":      st.column_config.NumberColumn("Signals"),
                    "rank_score":     st.column_config.NumberColumn("Score", format="%.4f"),
                },
            )
        else:
            st.info(
                "No dislocation candidates available. Run the pipeline "
                "(`python run_pipeline.py`) to generate ranked candidate pairs."
            )
