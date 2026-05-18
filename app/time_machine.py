"""Time Machine — date-driven correlation network analysis.

Single top-nav page that lets the user pick a date and see the
correlation matrix, the MST derived from it, and the most
"dislocated" pairs (lowest pairwise correlation) AT that point in
time.

Phase 1 implementation uses LIVE compute via
``compute_window_correlation`` (src/rolling_correlation.py:452) per
slider drag. Cost is ~50-500 ms per drag on S&P-500; wrapped in
``@st.fragment`` so the dashboard prologue + other top-nav pages do
NOT re-execute when the date changes — only this page does.

Phase 3 will replace the live computes with reads from a
precomputed snapshot grid (every 5 trading days), bringing per-drag
cost down to <200 ms on S&P-500. Until then, all loaders here live
in this file; nothing in the broader codebase needs to change when
that swap happens.

Per-universe gating:
  - Section 3 (Top dislocations) is hidden when the active universe
    has ``has_pair_trading=False`` (EEG). The dislocation framing is
    pair-trading-specific.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from src.analysis import compute_distance_matrix
from src.rolling_correlation import compute_window_correlation

from utils import (
    apply_chart_style,
    current_universe,
    get_colors,
    load_dendrogram_order,
    load_mst_edges,
    render_chart,
    render_matrix_heatmap,
    section_header,
)
from universe_registry import get_universe


# ── Cached helpers (scoped to this page) ────────────────────────────────

@st.cache_data(show_spinner="Computing snapshot correlation...")
def _pit_correlation_cached(
    _returns: pd.DataFrame,
    cache_key: str,
    end_date_iso: str,
    window: int,
    method: str,
) -> pd.DataFrame:
    """Wrap compute_window_correlation with the project's cache pattern.

    Underscored ``_returns`` is excluded from Streamlit's hash; the
    explicit ``cache_key`` string drives identity. Same pattern as
    ``app/dashboard.py:_pit_corr``; duplicated here so Time Machine
    is self-contained.
    """
    return compute_window_correlation(
        _returns, pd.Timestamp(end_date_iso), window=window, method=method,
    )


@st.cache_data(show_spinner=False)
def _pit_mst_cached(
    _corr: pd.DataFrame,
    cache_key: str,
) -> tuple[list[tuple[str, str, float]], dict[str, tuple[float, float]]]:
    """Build an MST + layout from a correlation matrix. Returns (edges, pos).

    Edges are (source, target, distance) tuples. ``pos`` is the layout
    dict. Caches both because the layout (kamada-kawai for small graphs,
    spring for large) is the expensive step.
    """
    if not HAS_NETWORKX or _corr.empty:
        return [], {}
    dist = compute_distance_matrix(_corr)
    G = nx.Graph()
    cols = list(dist.columns)
    for i, t1 in enumerate(cols):
        for j in range(i + 1, len(cols)):
            t2 = cols[j]
            d = float(dist.iloc[i, j])
            if np.isfinite(d):
                G.add_edge(t1, t2, weight=d)
    if G.number_of_edges() == 0:
        return [], {}
    mst = nx.minimum_spanning_tree(G, weight="weight")
    # Kamada-Kawai is O(N^3) — switch to spring for big graphs (S&P).
    if mst.number_of_nodes() > 200:
        pos = nx.spring_layout(mst, weight="weight", iterations=80, seed=42)
    else:
        pos = nx.kamada_kawai_layout(mst, weight="weight")
    edges = [(u, v, float(mst[u][v]["weight"])) for u, v in mst.edges()]
    return edges, pos


# ── MST plotter (small, shared between full-period + PIT branches) ──────

def _render_mst(
    edges: list[tuple[str, str, float]],
    pos: dict[str, tuple[float, float]],
    *,
    chart_id: str,
    default_title: str,
) -> None:
    """Render an MST given its edges + layout. Renders nothing for empty input."""
    if not edges or not pos:
        st.info("No MST data available for this snapshot.")
        return

    edge_x: list[float] = []
    edge_y: list[float] = []
    for u, v, _w in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.6, color="#A0A8B8"),
        hoverinfo="skip", showlegend=False,
    ))
    nodes = list(pos.keys())
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=nodes,
        textposition="top center",
        textfont=dict(size=7, color="#2B2D42"),
        marker=dict(
            size=9, color=get_colors()["primary"],
            line=dict(width=0.5, color="white"),
        ),
        hovertext=nodes, hoverinfo="text",
        showlegend=False,
    ))
    apply_chart_style(
        fig, height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
    )
    render_chart(
        fig, chart_id=chart_id, filename_base=chart_id,
        title_key=chart_id, default_title=default_title,
    )


# ── Main page entry ────────────────────────────────────────────────────

@st.fragment
def render(
    adj_close: pd.DataFrame,
    full_returns: pd.DataFrame,
    min_date,
    max_date,
) -> None:
    """Render the Time Machine page.

    ``@st.fragment`` scopes widget reruns to just this function — the
    dashboard prologue (sidebar, top-nav, etc.) is NOT re-executed when
    the user drags the date input or switches window/method. Each
    interaction takes ~50-500 ms (live PIT compute) instead of a full
    script rerun.

    Args mirror what other top-nav pages (Pair Analysis) receive: the
    universe-keyed data is already loaded upstream.
    """
    _u_key = current_universe()
    _active = get_universe(_u_key)

    trading_dates = full_returns.index
    if len(trading_dates) == 0:
        st.warning("No data available for this dataset.")
        return

    _date_min = trading_dates[0].date()
    _date_max = trading_dates[-1].date()

    # Header
    st.markdown(
        f"<div style='font-size:1.05rem; font-weight:700; color:#2B2D42; "
        f"margin-bottom:.25rem;'>Time Machine — {_active.short_label}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Pick a date to see the correlation network as it stood that day. "
        "Drag through crisis dates "
        "(2020-03-12 COVID, 2022-02-24 Ukraine, 2023-02-15 earthquake) "
        "and watch correlations spike + the MST collapse to a star. "
        "All compute is live in this phase (~50-500 ms per drag); "
        "Phase 3 will swap to precomputed snapshots for instant scrubbing."
    )

    # ── Master controls ─────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        picked_date = st.date_input(
            "Snapshot date",
            value=_date_max,
            min_value=_date_min,
            max_value=_date_max,
            key="tm_date",
        )
    with c2:
        window = int(st.selectbox(
            "Window (days)" if _active.domain == "finance" else "Window (samples)",
            [60, 120, 252], index=2, key="tm_window",
            help="Trading-day window for the rolling correlation snapshot.",
        ))
    with c3:
        method = st.selectbox(
            "Method", ["pearson", "spearman"], key="tm_method",
        )

    # date_input returns a date or (single-element) tuple in single-date mode.
    if isinstance(picked_date, tuple):
        picked_date = picked_date[0] if picked_date else _date_max

    # Snap to nearest preceding trading day so users picking
    # weekends/holidays don't see an empty snapshot.
    _picked_ts = pd.Timestamp(picked_date)
    _date_idx = trading_dates.searchsorted(_picked_ts, side="right") - 1
    _date_idx = max(0, min(_date_idx, len(trading_dates) - 1))
    snap_date = trading_dates[_date_idx]
    if snap_date.date() != picked_date:
        st.caption(
            f":material/info: Snapped to nearest trading day "
            f"**{snap_date.strftime('%Y-%m-%d')}** (you picked "
            f"{picked_date.isoformat()} — weekend or market holiday)."
        )

    # Cache key — same pattern as the rest of the codebase.
    _cache_key = (
        f"{_u_key}:{trading_dates[0].date().isoformat()}:"
        f"{trading_dates[-1].date().isoformat()}:"
        f"{full_returns.shape[0]}x{full_returns.shape[1]}"
    )

    pit_corr = _pit_correlation_cached(
        full_returns, _cache_key, snap_date.isoformat(), window, method,
    )
    if pit_corr.empty:
        st.warning(
            f"Not enough data at {snap_date.strftime('%Y-%m-%d')} for a "
            f"{window}-{'day' if _active.domain == 'finance' else 'sample'} "
            f"window. Pick a later date."
        )
        return

    # ── Section 1: Correlation snapshot ─────────────────────────────────
    with st.container(border=True):
        section_header(
            f"Correlation snapshot — {snap_date.strftime('%Y-%m-%d')}",
            f"Pairwise {method} correlation over the {window}-"
            f"{'day' if _active.domain == 'finance' else 'sample'} "
            f"window ending at this date.",
        )

        # KPI strip from upper triangle
        mask = np.triu(np.ones(pit_corr.shape, dtype=bool), k=1)
        vals = pit_corr.values[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric(
                f"{_active.items_label} in window", len(pit_corr),
            )
            m2.metric("Mean ρ", f"{np.mean(vals):.3f}")
            m3.metric("Median ρ", f"{np.median(vals):.3f}")
            m4.metric("Q75 ρ", f"{np.percentile(vals, 75):.3f}")
            m5.metric("Std ρ", f"{np.std(vals):.3f}")

        # Reorder by full-period dendrogram so the matrix structure
        # is visually stable across dates (otherwise rows/cols would
        # re-order on every snapshot).
        leaf_order = load_dendrogram_order()
        if leaf_order:
            present = [t for t in leaf_order if t in pit_corr.columns]
            if len(present) >= 2:
                pit_display = pit_corr.loc[present, present]
            else:
                pit_display = pit_corr
        else:
            pit_display = pit_corr

        render_matrix_heatmap(
            pit_display,
            chart_id="tm_pit_heatmap",
            filename_base="time_machine_pit_corr",
            title_key="tm_pit_heatmap",
            default_title=(
                f"PIT correlation @ {snap_date.strftime('%Y-%m-%d')} "
                f"({window}-{'d' if _active.domain == 'finance' else 'samp'}, {method})"
            ),
            zmin=-1.0, zmax=1.0, diverging=True,
            height=520,
            hover_label="corr",
            colorbar_title="Corr",
        )

    # ── Section 2: Network at this date ─────────────────────────────────
    with st.container(border=True):
        section_header(
            f"Network at this date — {snap_date.strftime('%Y-%m-%d')}",
            "Minimum Spanning Tree (MST) of the windowed correlation. "
            "Toggle to compare against the full-period MST.",
        )

        build_pit_mst = st.toggle(
            "MST from this snapshot (vs full-period MST)",
            value=True,
            key="tm_build_pit_mst",
            help=(
                "ON: derive an MST from the windowed correlation above. "
                "OFF: show the full-period MST baked at pipeline time. "
                "Compare to see how the network's backbone changes during stress."
            ),
        )

        if build_pit_mst:
            _mst_cache_key = (
                f"{_u_key}:pitmst:{snap_date.isoformat()}:{window}:{method}"
            )
            edges, pos = _pit_mst_cached(pit_corr, _mst_cache_key)
            _render_mst(
                edges, pos,
                chart_id="tm_pit_mst",
                default_title=(
                    f"PIT MST @ {snap_date.strftime('%Y-%m-%d')} "
                    f"({len(edges)} edges, {len(pos)} nodes)"
                ),
            )
        else:
            # Full-period MST from precomputed artifacts.
            mst_edges_df = load_mst_edges()
            if mst_edges_df.empty:
                st.info(
                    "Full-period MST not on disk — run the pipeline. "
                    "Toggle the switch above to build one from the PIT snapshot instead."
                )
            else:
                # Layout-build the full-period MST. Cached via cache_key.
                _fp_corr_proxy = pit_corr  # only used for cache_key shape
                _fp_mst_cache_key = f"{_u_key}:fpmst"

                @st.cache_data(show_spinner=False)
                def _fp_layout(edges_csv: str, cache_key: str):
                    G = nx.Graph()
                    edges_df = pd.read_csv(pd.io.common.StringIO(edges_csv))
                    for _, r in edges_df.iterrows():
                        G.add_edge(r["source"], r["target"], weight=float(r["distance"]))
                    if G.number_of_nodes() > 200:
                        return list(G.edges(data="weight")), nx.spring_layout(
                            G, weight="weight", iterations=80, seed=42,
                        )
                    return list(G.edges(data="weight")), nx.kamada_kawai_layout(
                        G, weight="weight",
                    )

                fp_edges_raw, fp_pos = _fp_layout(
                    mst_edges_df.to_csv(index=False), _fp_mst_cache_key,
                )
                _fp_edges = [(u, v, float(w)) for u, v, w in fp_edges_raw]
                _render_mst(
                    _fp_edges, fp_pos,
                    chart_id="tm_fp_mst",
                    default_title=(
                        f"Full-period MST ({len(_fp_edges)} edges, {len(fp_pos)} nodes)"
                    ),
                )

    # ── Section 3: Top dislocations at this date ────────────────────────
    if not _active.has_pair_trading:
        # EEG: no pair-trading framing.
        return

    with st.container(border=True):
        section_header(
            f"Top dislocations at {snap_date.strftime('%Y-%m-%d')}",
            "The 20 pairs with the lowest pairwise correlation in this "
            f"{window}-day window. Negatively-correlated pairs are mean-"
            "reversion candidates. Phase 3 will replace this list with "
            "PIT spread Z-score from precomputed snapshots.",
        )

        idx = list(pit_corr.columns)
        records = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                r = float(pit_corr.iloc[i, j])
                if np.isfinite(r):
                    records.append((idx[i], idx[j], r))
        if not records:
            st.info("No pairs available for dislocation ranking.")
            return

        df = (
            pd.DataFrame(records, columns=[
                _active.item_label + " A",
                _active.item_label + " B",
                "Correlation",
            ])
            .sort_values("Correlation", ascending=True)
            .head(20)
            .reset_index(drop=True)
        )
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Correlation": st.column_config.NumberColumn(
                    format="%.4f",
                    help="Pairwise correlation in the windowed snapshot",
                ),
            },
        )
