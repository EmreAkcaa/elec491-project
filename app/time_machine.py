"""Time Machine — date-driven correlation network analysis.

Single top-nav page that lets the user pick a date and see the
correlation matrix, the MST derived from it, and the most
"dislocated" pairs (lowest pairwise correlation) AT that point in
time.

Phase 1 implementation used LIVE compute via
``compute_window_correlation`` (src/rolling_correlation.py:452) per
slider drag. Cost was ~50-500 ms per drag on S&P-500; wrapped in
``@st.fragment`` so the dashboard prologue + other top-nav pages do
NOT re-execute when the date changes — only this page does.

PHASE 3 (slim) — flagship universes (BIST TRY + S&P) now read from
a precomputed snapshot grid via ``utils.load_pit_*_snapshot``. Slider
drag is now ~10-30 ms (file I/O only) at window=252. Universes /
windows outside the precomputed set (BIST USD/Gold, EEG, w∈{60,120})
fall back to live compute transparently — a caption indicates the
mode so the user understands the performance difference.

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
    SECTOR_PALETTE,
    apply_chart_style,
    current_universe,
    get_colors,
    load_cluster_assignments,
    load_dendrogram_order,
    load_mst_edges,
    load_pit_dislocation_snapshot,
    load_pit_mst_snapshot,
    load_pit_snapshot,
    render_chart,
    render_matrix_heatmap,
    section_header,
    snap_to_nearest_snapshot,
)
from universe_registry import get_universe


# Universes for which we precompute PIT snapshots in
# ``src/pit_snapshots.py``. Time Machine fast-paths through the loader
# for these + window=252; everything else falls back to live compute.
_PRECOMPUTED_MARKETS: set[str] = {"bist", "sp500"}
_PRECOMPUTED_WINDOW = 252
_PRECOMPUTED_METHOD = "pearson"


def _is_precomputed_path(universe_key: str, window: int, method: str) -> bool:
    """Return True iff the (universe, window, method) tuple has snapshots
    on disk per PHASE 3 (slim)."""
    return (
        universe_key in _PRECOMPUTED_MARKETS
        and window == _PRECOMPUTED_WINDOW
        and method == _PRECOMPUTED_METHOD
    )


# ── Cached helpers (scoped to this page) ────────────────────────────────

@st.cache_data(show_spinner=False)
def _pit_correlation_live(
    _returns: pd.DataFrame,
    cache_key: str,
    end_date_iso: str,
    window: int,
    method: str,
) -> pd.DataFrame:
    """Live compute via compute_window_correlation. Used as fallback when
    the (universe, window, method) tuple isn't in the precomputed grid
    (BIST USD/Gold, EEG, or non-default windows/methods).

    Underscored ``_returns`` is excluded from Streamlit's hash; the
    explicit ``cache_key`` string drives identity.
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

    Live-compute fallback when no precomputed PIT MST edges exist for the
    (universe, date, window) tuple. Edges are (source, target, distance)
    tuples; ``pos`` is the layout dict. Caches both because the layout
    (kamada-kawai for small graphs, spring for large) is the expensive step.
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


@st.cache_data(show_spinner=False)
def _pit_mst_from_edges_cached(
    _edges_df: pd.DataFrame,
    cache_key: str,
) -> tuple[list[tuple[str, str, float]], dict[str, tuple[float, float]]]:
    """Layout-only path: precomputed MST edges → live layout.

    PHASE 3 (slim) skips MST CONSTRUCTION (Kruskal over ~117K candidate
    edges for S&P) by reading the precomputed edges CSV. We still need to
    LAYOUT the MST live since position-only layout dicts don't compress
    well + bias future flexibility on the layout algorithm.
    """
    if not HAS_NETWORKX or _edges_df.empty:
        return [], {}
    G = nx.Graph()
    for _, r in _edges_df.iterrows():
        G.add_edge(str(r["source"]), str(r["target"]), weight=float(r["distance"]))
    if G.number_of_edges() == 0:
        return [], {}
    if G.number_of_nodes() > 200:
        pos = nx.spring_layout(G, weight="weight", iterations=80, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G, weight="weight")
    edges = [(u, v, float(G[u][v]["weight"])) for u, v in G.edges()]
    return edges, pos


# ── MST plotter (small, shared between full-period + PIT branches) ──────

def _render_mst(
    edges: list[tuple[str, str, float]],
    pos: dict[str, tuple[float, float]],
    *,
    chart_id: str,
    default_title: str,
    sector_map: dict[str, str] | None = None,
) -> None:
    """Render an MST given its edges + layout.

    Nodes are sized by degree (computed from ``edges``) and coloured by
    sector (looked up in ``sector_map``); a per-sector legend is rendered
    on the upper-left, matching the Market Overview MST and the Methods
    Lab RMT MST.
    """
    if not edges or not pos:
        st.info("No MST data available for this snapshot.")
        return

    sector_map = sector_map or {}

    # Degree per node = how many MST edges touch it.
    degree_map: dict[str, int] = {}
    for u, v, _w in edges:
        degree_map[u] = degree_map.get(u, 0) + 1
        degree_map[v] = degree_map.get(v, 0) + 1

    # Sector → colour: same SECTOR_PALETTE cycle as the other MST views.
    _seen_sectors: list[str] = []
    for n in pos.keys():
        sec = sector_map.get(n)
        if not sec or (isinstance(sec, float) and pd.isna(sec)):
            continue
        if sec not in _seen_sectors:
            _seen_sectors.append(sec)
    sector_colors = {
        s: SECTOR_PALETTE[i % len(SECTOR_PALETTE)] for i, s in enumerate(sorted(_seen_sectors))
    }

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
        line=dict(width=1.4, color="#A0A8B8"),
        hoverinfo="skip", showlegend=False,
    ))
    nodes = list(pos.keys())
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    node_color = [
        sector_colors.get(sector_map.get(n, ""), get_colors()["muted"])
        for n in nodes
    ]
    # Size scales with degree, matching Market Overview MST's `14 + degree*6`.
    node_size = [14 + degree_map.get(n, 1) * 6 for n in nodes]
    node_hover = [
        f"<b>{n}</b><br>Sector: {sector_map.get(n, 'Unknown')}<br>Degree: {degree_map.get(n, 0)}"
        for n in nodes
    ]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=nodes,
        textposition="top center",
        textfont=dict(size=8, color="#2B2D42"),
        marker=dict(
            size=node_size, color=node_color,
            line=dict(width=1.5, color="white"),
        ),
        hovertext=node_hover, hoverinfo="text",
        showlegend=False,
    ))
    # Legend entries — one invisible scatter per sector.
    for sec in sorted(_seen_sectors):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=sector_colors[sec]),
            name=sec, showlegend=True,
        ))

    apply_chart_style(
        fig, height=700,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        showlegend=bool(_seen_sectors),
        legend=dict(font=dict(size=9), orientation="v",
                    yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.85)", borderwidth=1,
                    bordercolor="#e2e6ee"),
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
        "and watch correlations spike + the MST collapse to a star."
    )

    # ── Master controls ─────────────────────────────────────────────────
    # Quick-jump presets — user can pick a crisis event and the date input
    # snaps to it. Setting the date via the preset is a one-time write to
    # session_state; subsequent manual edits of the date input override it.
    _CRISIS_PRESETS: dict[str, str] = {
        "— pick a date manually —": "",
        "COVID-19 selloff (2020-03-12)": "2020-03-12",
        "Russia–Ukraine war (2022-02-24)": "2022-02-24",
        "Türkiye earthquakes (2023-02-15)": "2023-02-15",
    }
    _preset_choice = st.selectbox(
        "Quick-jump to crisis event",
        list(_CRISIS_PRESETS.keys()),
        index=0,
        key="tm_crisis_preset",
        help="Pre-fill the snapshot date with a known stress event. "
             "Adjust the date input below freely afterwards.",
    )
    _preset_iso = _CRISIS_PRESETS[_preset_choice]
    if _preset_iso and st.session_state.get("tm_crisis_preset_applied") != _preset_iso:
        try:
            _preset_date = pd.Timestamp(_preset_iso).date()
            if _date_min <= _preset_date <= _date_max:
                st.session_state["tm_date"] = _preset_date
                st.session_state["tm_crisis_preset_applied"] = _preset_iso
        except (ValueError, TypeError):
            pass

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
        # Window picker — accepts any value in [30, 504] so users can tune
        # the rolling-correlation window beyond the precomputed grid. The
        # precomputed snapshot fast-path only triggers at window=252 +
        # pearson on BIST/S&P; off-grid values fall back to live compute
        # (a caption below tells the user which path ran).
        window = int(st.number_input(
            "Window (days)" if _active.domain == "finance" else "Window (samples)",
            min_value=30, max_value=504, value=252, step=1,
            key="tm_window_dyn",
            help="Trading-day window for the rolling correlation snapshot. "
                 "Window=252 + Pearson hits the precomputed grid (instant); "
                 "other values compute live (~50-500 ms on S&P).",
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

    # PHASE 3 (slim): try the precomputed snapshot grid FIRST when the
    # active universe + window + method match the pipeline output. On
    # cache hit (BIST TRY / S&P at w=252 pearson) the slider drag is
    # ~10-30 ms. Otherwise fall through to live compute.
    pit_corr = pd.DataFrame()
    snap_actual_iso: str | None = None
    used_precomputed = False
    if _is_precomputed_path(_u_key, window, method):
        snap_actual_iso = snap_to_nearest_snapshot(
            snap_date, window=window, kind="corr",
        )
        if snap_actual_iso is not None:
            pit_corr = load_pit_snapshot(window, snap_actual_iso)
            if not pit_corr.empty:
                used_precomputed = True
                # If the snapshot-grid date differs from the user's
                # snapped trading-day by more than 1 day, surface it.
                snap_ts = pd.Timestamp(snap_actual_iso)
                if abs((snap_ts - snap_date).days) >= 1:
                    st.caption(
                        f":material/bolt: Snapping to precomputed snapshot "
                        f"**{snap_actual_iso}** (closest in the "
                        f"{_PRECOMPUTED_WINDOW}-day grid). Live compute "
                        f"available for exact dates — toggle to use it."
                    )
                # Display date matches the precomputed snapshot.
                snap_date = snap_ts

    if pit_corr.empty:
        # Cache key — same pattern as the rest of the codebase.
        _cache_key = (
            f"{_u_key}:{trading_dates[0].date().isoformat()}:"
            f"{trading_dates[-1].date().isoformat()}:"
            f"{full_returns.shape[0]}x{full_returns.shape[1]}"
        )
        pit_corr = _pit_correlation_live(
            full_returns, _cache_key, snap_date.isoformat(), window, method,
        )
        if _u_key in _PRECOMPUTED_MARKETS and window != _PRECOMPUTED_WINDOW:
            # User chose w=60/120 — let them know why this is slower.
            st.caption(
                f":material/info: Computing live (precomputed grid only "
                f"covers window={_PRECOMPUTED_WINDOW}). Pick window=252 for "
                f"instant scrubbing."
            )
        elif _u_key not in _PRECOMPUTED_MARKETS:
            st.caption(
                f":material/info: Computing live (precomputed grid covers "
                f"BIST TRY + S&P at window=252 only)."
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

        # Sector map for node coloring — sourced from cluster_assignments so
        # we get the same ticker → sector mapping every other MST view uses.
        _cluster_df = load_cluster_assignments()
        if not _cluster_df.empty and "sector" in _cluster_df.columns:
            _sector_map = dict(zip(_cluster_df["ticker"], _cluster_df["sector"]))
        else:
            _sector_map = {}

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
            # PHASE 3 (slim): try precomputed MST edges from disk first;
            # we still need to LAYOUT the MST live (kamada_kawai / spring
            # don't precompute well because they're position-only and
            # don't compress meaningfully). So the precomputed path saves
            # only the MST CONSTRUCTION (Kruskal over 117K candidate
            # edges for S&P), not the layout.
            edges_df = pd.DataFrame()
            if used_precomputed and snap_actual_iso is not None:
                edges_df = load_pit_mst_snapshot(window, snap_actual_iso)

            _mst_cache_key = (
                f"{_u_key}:pitmst:{snap_date.isoformat()}:{window}:{method}"
            )
            if not edges_df.empty:
                edges, pos = _pit_mst_from_edges_cached(edges_df, _mst_cache_key)
            else:
                edges, pos = _pit_mst_cached(pit_corr, _mst_cache_key)
            _render_mst(
                edges, pos,
                chart_id="tm_pit_mst",
                default_title=(
                    f"PIT MST @ {snap_date.strftime('%Y-%m-%d')} "
                    f"({len(edges)} edges, {len(pos)} nodes)"
                ),
                sector_map=_sector_map,
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
                    sector_map=_sector_map,
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
            "reversion candidates.",
        )

        # PHASE 3 (slim): try precomputed top-20 dislocations table first.
        # Falls back to live ranking from the in-memory corr matrix.
        dis_df = pd.DataFrame()
        if used_precomputed and snap_actual_iso is not None:
            dis_df = load_pit_dislocation_snapshot(window, snap_actual_iso)
            if not dis_df.empty:
                # Relabel to match the live-compute output schema.
                dis_df = dis_df.rename(columns={
                    "ticker_a": _active.item_label + " A",
                    "ticker_b": _active.item_label + " B",
                    "correlation": "Correlation",
                })

        if dis_df.empty:
            # Live fallback — same logic as the prior Phase 1 implementation.
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
            dis_df = (
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
            dis_df, use_container_width=True, hide_index=True,
            column_config={
                "Correlation": st.column_config.NumberColumn(
                    format="%.4f",
                    help="Pairwise correlation in the windowed snapshot",
                ),
            },
        )
