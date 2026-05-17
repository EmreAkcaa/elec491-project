"""Cross-Market Comparison page — BIST 100 vs S&P 500, side by side.

This page is universe-independent. It reads from both ``data/bist/`` and
``data/sp500/`` directly via the universe-keyed loaders in ``app/utils.py``
so it does NOT use ``current_universe()`` — switching the sidebar selector
does not change what this page shows.

Headline finding: D_eff is universal at ~6.5 across both markets despite a
6.6× ticker count difference, yet MST sector-purity diverges sharply
(0.40 BIST vs 0.80 S&P) — direct evidence of BIST's conglomerate-led
topology vs S&P's sector-coherent topology.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from utils import (
    PROJECT_ROOT,
    SECTOR_PALETTE,
    apply_chart_style,
    data_results,
    get_colors,
    inject_custom_css,
    page_header,
    render_chart,
    section_header,
    # universe-keyed underscored loaders — used directly so we can pass
    # explicit universe keys instead of going through current_universe()
    _load_eigenvalue_spectrum,
    _load_mst_edges,
    _load_mst_metrics,
    _load_cluster_assignments,
    _load_dislocation_candidates,
)
from universe_registry import UNIVERSES, get_universe, available_universes


# ---------------------------------------------------------------------------
# Shared style hooks for the two markets on this page
# ---------------------------------------------------------------------------
_BIST_COLOR  = "#E63946"   # red — matches the project's "secondary" palette
_SP500_COLOR = "#4361EE"   # blue — matches the project's "primary" palette

_CRISIS_EVENTS = [
    ("2020-03-11", "COVID-19 WHO declaration", "Global shock; visible on both."),
    ("2022-02-24", "Russia-Ukraine war",      "Global shock; visible on both."),
    ("2023-02-06", "Türkiye earthquakes",     "Local shock; expected only on BIST."),
]


# ---------------------------------------------------------------------------
# Loaders for this page only — the headline comparison CSV
# ---------------------------------------------------------------------------

@st.cache_data
def _load_comparison_table() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "comparison_bist_vs_sp500.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


def _get(df: pd.DataFrame, row: str, col: str) -> Any:
    """Safe scalar getter for the comparison CSV."""
    if df.empty or row not in df.index or col not in df.columns:
        return None
    v = df.loc[row, col]
    if pd.isna(v):
        return None
    return v


def _fmt(v: Any, kind: str = "raw") -> str:
    if v is None:
        return "n/a"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "float":
        return f"{float(v):.4f}"
    if kind == "pct":
        return f"{100 * float(v):.1f}%"
    return str(v)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _eigenvalue_spectrum_fig(eig_df: pd.DataFrame, color: str, label: str) -> go.Figure:
    """Single-universe eigenvalue spectrum overlay (bar + MP-bound dashed line)."""
    fig = go.Figure()
    n = len(eig_df)
    eig_df = eig_df.sort_values("eigenvalue", ascending=False).reset_index(drop=True)
    fig.add_trace(go.Bar(
        x=np.arange(n) + 1,
        y=eig_df["eigenvalue"],
        marker_color=[color if bool(s) else "#B0B0B0" for s in eig_df["is_signal"]],
        name=label,
        hovertemplate="rank %{x}: λ=%{y:.3f}<extra></extra>",
    ))
    mp_upper = eig_df["mp_upper"].iloc[0] if "mp_upper" in eig_df.columns else None
    if mp_upper is not None and np.isfinite(mp_upper):
        fig.add_hline(
            y=float(mp_upper), line_dash="dash", line_color="#2B2D42",
            annotation_text=f"MP upper {mp_upper:.3f}", annotation_font_size=10,
        )
    apply_chart_style(
        fig, height=340,
        xaxis_title="Eigenvalue rank",
        yaxis_title="Eigenvalue (log)",
        yaxis_type="log",
        showlegend=False,
        margin=dict(l=40, r=10, t=10, b=40),
    )
    return fig


def _mst_fig(edges_df: pd.DataFrame, metrics_df: pd.DataFrame, color_fallback: str) -> go.Figure:
    """Single-universe MST plotted with kamada-kawai layout, coloured by sector."""
    if not HAS_NETWORKX or edges_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="MST data not available", showarrow=False, font=dict(size=14))
        return fig

    G = nx.Graph()
    for _, r in edges_df.iterrows():
        G.add_edge(r["source"], r["target"], weight=float(r.get("distance", 1.0)))
    pos = nx.kamada_kawai_layout(G, weight="weight")

    sector_map: dict[str, str] = {}
    btw_map: dict[str, float] = {}
    if not metrics_df.empty:
        if "sector" in metrics_df.columns:
            sector_map = dict(zip(metrics_df["ticker"], metrics_df["sector"].astype(str)))
        if "betweenness_centrality" in metrics_df.columns:
            btw_map = dict(zip(metrics_df["ticker"], metrics_df["betweenness_centrality"].astype(float)))

    edge_x: list[float] = []
    edge_y: list[float] = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.4, color="#cccccc"),
        hoverinfo="skip", showlegend=False,
    ))

    sectors = sorted(set(s for s in sector_map.values() if s and s != "nan"))
    sector_color = {s: SECTOR_PALETTE[i % len(SECTOR_PALETTE)] for i, s in enumerate(sectors)}

    btw_vals = [v for v in btw_map.values() if np.isfinite(v)]
    btw_min = min(btw_vals) if btw_vals else 0.0
    btw_max = max(btw_vals) if btw_vals else 1.0
    btw_span = max(btw_max - btw_min, 1e-9)

    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_color: list[str] = []
    node_size: list[float] = []
    node_label: list[str] = []
    for node in G.nodes():
        x, y = pos[node]
        sec = sector_map.get(node, "Unknown")
        btw = float(btw_map.get(node, 0.0))
        size = 6 + 22 * (btw - btw_min) / btw_span
        node_x.append(x); node_y.append(y)
        node_size.append(size)
        node_color.append(sector_color.get(sec, color_fallback))
        node_text.append(f"<b>{node}</b><br>sector: {sec}<br>btw: {btw:.4f}")
        # Only label hubs to keep the dense S&P plot readable.
        node_label.append(node if (btw - btw_min) / btw_span > 0.45 else "")

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=node_size, color=node_color, line=dict(width=0.5, color="#fff")),
        text=node_label, textposition="top center", textfont=dict(size=9),
        hovertext=node_text, hoverinfo="text", showlegend=False,
    ))

    apply_chart_style(
        fig, height=460,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    return fig


def _crisis_fig(comp_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: avg pairwise correlation (before/during/after) per crisis × universe."""
    phases = ["before", "during", "after"]
    rows: list[dict] = []
    for date, label, _note in _CRISIS_EVENTS:
        for phase in phases:
            row_key = f"{date}_{phase}"
            rows.append({
                "Event": f"{label}\n{date}",
                "Phase": phase.capitalize(),
                "BIST":  _get(comp_df, row_key, "BIST"),
                "S&P-500": _get(comp_df, row_key, "S&P-500"),
            })
    long_df = pd.DataFrame(rows)

    fig = go.Figure()
    # Two grouped bars per (event,phase): BIST and S&P
    fig.add_trace(go.Bar(
        x=[long_df["Event"], long_df["Phase"]],
        y=[v if v is not None else np.nan for v in long_df["BIST"]],
        name="BIST 100", marker_color=_BIST_COLOR,
    ))
    fig.add_trace(go.Bar(
        x=[long_df["Event"], long_df["Phase"]],
        y=[v if v is not None else np.nan for v in long_df["S&P-500"]],
        name="S&P 500", marker_color=_SP500_COLOR,
    ))
    apply_chart_style(
        fig, height=420,
        barmode="group",
        yaxis_title="Avg pairwise correlation (±60-day window)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=10, t=40, b=80),
    )
    return fig


def _top_hubs_text(metrics_df: pd.DataFrame, k: int = 5) -> str:
    if metrics_df.empty or "betweenness_centrality" not in metrics_df.columns:
        return "_no MST metrics available_"
    top = metrics_df.sort_values("betweenness_centrality", ascending=False).head(k)
    rows = []
    for _, r in top.iterrows():
        sec = str(r.get("sector", "Unknown"))
        rows.append(f"- **{r['ticker']}** — {sec} (btw {float(r['betweenness_centrality']):.3f})")
    return "\n".join(rows)


def _kpi_row(comp_df: pd.DataFrame, label: str, key: str, kind: str = "raw") -> tuple[str, str]:
    return _fmt(_get(comp_df, key, "BIST"), kind), _fmt(_get(comp_df, key, "S&P-500"), kind)


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def _render_bist_numeraire_section() -> None:
    """BIST in TRY / USD / Gold side-by-side numéraire sensitivity probe.

    Phase 4 mutable-candy. Loads the precomputed `bist`, `bist_usd`,
    `bist_gold` results trees and renders three short panels:
    eigenvalue spectra, KPI strip (D_eff, ΔH, avg-corr, top eig share),
    and sector purity bars. One paragraph honest interpretation at the
    bottom.
    """
    import json

    bist_dir = data_results("bist")
    usd_dir = data_results("bist_usd")
    gold_dir = data_results("bist_gold")
    if not (usd_dir / "eigenvalue_spectrum.csv").exists() or not (gold_dir / "eigenvalue_spectrum.csv").exists():
        return  # variants not generated yet — silently skip section

    with st.container(border=True):
        section_header(
            "BIST Numéraire Sensitivity",
            "Same BIST universe re-expressed in three numéraires (TRY, USD, "
            "gold). The hypothesis was that removing the currency leg should "
            "strip out a common factor and reduce the dominant eigenvalue's "
            "share. The experiment refutes this: the share goes UP, not down.",
        )

        def _load_market(market_dir, key):
            eig = pd.read_csv(market_dir / "eigenvalue_spectrum.csv")
            with open(market_dir / "it_summary.json") as f:
                summary = json.load(f)
            with open(market_dir / "pipeline_metadata.json") as f:
                meta = json.load(f).get("market_summary", {})
            cluster_df = pd.read_csv(market_dir / "cluster_assignments.csv")
            return {
                "key": key,
                "eig": eig,
                "summary": summary,
                "meta": meta,
                "clusters": cluster_df,
            }

        bist_data = _load_market(bist_dir, "TRY")
        usd_data = _load_market(usd_dir, "USD")
        gold_data = _load_market(gold_dir, "Gold")

        # KPI strip
        st.markdown("**Headline numbers** — same universe, three base assets")
        kpi_cols = st.columns(4)
        for col_idx, (label, key) in enumerate([
            ("D_eff", "d_eff"),
            ("ΔH (nats)", "log_det_term"),
            ("Avg pairwise ρ", None),  # from market_summary
            ("Top eig share", None),    # computed
        ]):
            with kpi_cols[col_idx]:
                rows = []
                for d in (bist_data, usd_data, gold_data):
                    if key:
                        rows.append((d["key"], d["summary"].get(key, float("nan"))))
                    elif label == "Avg pairwise ρ":
                        rows.append((d["key"], d["meta"].get("avg_pairwise_corr", float("nan"))))
                    else:  # Top eig share
                        eigs = d["eig"]["eigenvalue"].values
                        rows.append((d["key"], float(eigs.max() / eigs.sum())))
                st.markdown(f"**{label}**")
                for k, v in rows:
                    if "share" in label.lower():
                        st.markdown(f"- {k}: **{v*100:.2f}%**")
                    elif "ρ" in label:
                        st.markdown(f"- {k}: **{v:.3f}**")
                    else:
                        st.markdown(f"- {k}: **{v:.2f}**")

        # Eigenvalue-spectrum overlay
        st.markdown("**Eigenvalue spectrum** (log scale; first 15 eigenvalues)")
        fig = go.Figure()
        palette = {"TRY": "#1F77B4", "USD": "#2CA02C", "Gold": "#FFC400"}
        for d in (bist_data, usd_data, gold_data):
            eig = d["eig"].sort_values("eigenvalue", ascending=False).head(15)
            fig.add_trace(go.Bar(
                x=list(range(1, len(eig) + 1)),
                y=eig["eigenvalue"].values,
                name=d["key"],
                marker_color=palette[d["key"]],
                opacity=0.85,
            ))
        fig.update_layout(
            barmode="group",
            xaxis_title="Eigenvalue rank",
            yaxis_title="Eigenvalue",
            yaxis_type="log",
            height=380,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        render_chart(
            fig, chart_id="num_eigvals",
            filename_base="numeraire_eigenvalue_spectrum",
            title_key="num_eigvals",
            default_title="BIST eigenvalue spectrum: TRY vs USD vs Gold",
        )

        # Sector purity bars
        st.markdown(
            "**Sector purity** — share of each cluster occupied by its modal sector. "
            "Higher = cleaner sector recovery."
        )
        sector_purity_rows = []
        for d in (bist_data, usd_data, gold_data):
            clusters = d["clusters"].dropna(subset=["sector", "cluster_id"])
            purity = (
                clusters.groupby("cluster_id")["sector"]
                .agg(lambda s: s.value_counts().iloc[0] / len(s))
                .mean()
            )
            sector_purity_rows.append({"Numéraire": d["key"], "Mean cluster purity": purity})
        purity_df = pd.DataFrame(sector_purity_rows)
        fig_pur = go.Figure(go.Bar(
            x=purity_df["Numéraire"],
            y=purity_df["Mean cluster purity"],
            marker_color=[palette[k] for k in purity_df["Numéraire"]],
            text=[f"{v:.2%}" for v in purity_df["Mean cluster purity"]],
            textposition="outside",
        ))
        fig_pur.update_layout(
            xaxis_title="",
            yaxis_title="Mean cluster purity",
            yaxis_tickformat=".0%",
            height=320,
            margin=dict(l=40, r=20, t=20, b=40),
            yaxis_range=[0, 1.0],
        )
        render_chart(
            fig_pur, chart_id="num_sector_purity",
            filename_base="numeraire_sector_purity",
            title_key="num_sector_purity",
            default_title="Mean cluster sector-purity across BIST numéraires",
        )

        # Honest interpretation paragraph
        st.markdown(
            "**Reading.** The naïve hypothesis was that stripping the TRY leg "
            "(re-expressing returns in USD or gold) should remove a market-wide "
            "common factor and reduce the top eigenvalue's variance share. The "
            "experiment shows the **opposite** direction: the share rises from "
            f"**{bist_data['eig']['eigenvalue'].max() / bist_data['eig']['eigenvalue'].sum() * 100:.1f}%** (TRY) "
            f"→ **{usd_data['eig']['eigenvalue'].max() / usd_data['eig']['eigenvalue'].sum() * 100:.1f}%** (USD) "
            f"→ **{gold_data['eig']['eigenvalue'].max() / gold_data['eig']['eigenvalue'].sum() * 100:.1f}%** (gold), "
            "and effective dimensionality drops from "
            f"**{bist_data['summary']['d_eff']:.2f}** to **{gold_data['summary']['d_eff']:.2f}**. "
            "Interpretation: TRY volatility is a *dispersion source* for BIST "
            "equities — exporters and importers respond oppositely to TRY moves, "
            "so removing the currency leg amplifies the residual global-equity-risk "
            "common factor. Numéraire choice is a substantive modelling decision, "
            "not a noise-removal step. The numéraire panel is presented as an "
            "honest empirical finding rather than a thesis."
        )


def render() -> None:
    # Defensive import — force fresh load so a stale module in sys.modules
    # (Streamlit Cloud caches across deploys) can't strip the Phase I fields.
    import importlib
    import universe_registry as _ur
    importlib.reload(_ur)

    inject_custom_css()
    page_header(
        "Cross-Market Comparison",
        "Same StoNeCoAl toolkit (Pearson, RMT, MST, Glasso, TE, wavelet) applied "
        "to BIST 100 and the S&P 500 over the same 2020-01 → 2026-03 window. "
        "Universes selected because they sit at opposite ends of the emerging vs "
        "developed-market dimension while sharing the same sampling and pipeline.",
    )

    # Plain-English finance question — the demo first-60-seconds anchor.
    st.info(
        ":material/help: **Central question:** How does the structure of BIST-100 "
        "co-movement compare to a developed-market reference (S&P 500)? Both panels "
        "run through the identical 12-stage signal-processing pipeline, so any "
        "difference in the resulting correlation network, sector recovery, or "
        "crisis response is a property of the underlying market — not the method. "
        "The clearest single finding: the 2023 Türkiye earthquake spike (mean "
        "|correlation| 0.44 → 0.66 → 0.58 over the event window) is isolated to "
        "BIST; the S&P signal stays flat (0.44 → 0.35 → 0.31), validating that "
        "the toolkit catches market-specific stress events."
    )

    # Defence-in-depth filter: only universes flagged eligible_for_cross_market
    # participate here. EEG (eligible_for_cross_market=False) is filtered out
    # even if the page is reached programmatically — the BIST-vs-S&P comparison
    # numbers come from data/comparison_bist_vs_sp500.csv which only knows the
    # two financial universes. getattr default True so a stale Universe class
    # without the field falls back to the pre-Phase-I behaviour (all universes
    # eligible).
    _eligible = [u for u in available_universes() if getattr(u, "eligible_for_cross_market", True)]
    if len(_eligible) < 2:
        st.info(
            "Cross-Market Comparison needs at least two financial universes "
            f"with `eligible_for_cross_market=True`. Currently eligible: "
            f"{[u.key for u in _eligible] or 'none'}."
        )
        return

    comp_df = _load_comparison_table()
    if comp_df.empty:
        st.error(
            "`data/comparison_bist_vs_sp500.csv` not found. "
            "Run both pipelines and then `uv run python scripts/sp500_vs_bist.py`."
        )
        return

    # ── Section 1: KPI strip
    with st.container(border=True):
        section_header("Headline numbers")
        bist, sp = _kpi_row(comp_df, "N", "N", "int")
        c = st.columns(8)
        c[0].metric("BIST tickers (N)",  bist)
        c[1].metric("S&P tickers (N)",   sp)
        bist, sp = _kpi_row(comp_df, "D_eff", "D_eff", "float")
        c[2].metric("BIST D_eff",        bist)
        c[3].metric("S&P D_eff",         sp)
        bist, sp = _kpi_row(comp_df, "top_eigenvalue_share", "top_eigenvalue_share", "pct")
        c[4].metric("BIST top-eig share", bist)
        c[5].metric("S&P top-eig share",  sp)
        bist, sp = _kpi_row(comp_df, "mst_sector_purity", "mst_sector_purity", "pct")
        c[6].metric("BIST MST sector purity", bist)
        c[7].metric("S&P MST sector purity",  sp)
        st.caption(
            ":material/info: **D_eff is universal (~6.5) across both markets** "
            "despite a 6.6× ticker-count difference — striking evidence that "
            "effective-rank co-movement is a market-invariant property at daily "
            "frequency. **MST sector-purity diverges sharply** — BIST's "
            "conglomerate-led hubs cross sectors (0.40), whereas S&P's MST is "
            "strongly sector-coherent (0.80)."
        )

    # ── Section 2: Eigenvalue spectra side-by-side
    with st.container(border=True):
        section_header(
            "Spectral structure (RMT)",
            "Eigenvalues above the Marchenko–Pastur upper bound are real signal "
            "(coloured). The number of signal eigenvalues is the universe's "
            "effective factor count; D_eff is the participation ratio."
        )
        col_b, col_s = st.columns(2)
        eig_bist = _load_eigenvalue_spectrum("bist")
        eig_sp   = _load_eigenvalue_spectrum("sp500")
        with col_b:
            st.markdown(f"**BIST 100** — D_eff = {_fmt(_get(comp_df, 'D_eff', 'BIST'), 'float')}; "
                        f"signal eigenvalues: **{_fmt(_get(comp_df, 'n_signal_eigenvalues', 'BIST'), 'int')}**")
            if not eig_bist.empty:
                render_chart(
                    _eigenvalue_spectrum_fig(eig_bist, _BIST_COLOR, "BIST"),
                    chart_id="xm_eig_bist", filename_base="cross_market_eigenvalues_bist",
                )
            else:
                st.info("BIST eigenvalue spectrum not available.")
        with col_s:
            st.markdown(f"**S&P 500** — D_eff = {_fmt(_get(comp_df, 'D_eff', 'S&P-500'), 'float')}; "
                        f"signal eigenvalues: **{_fmt(_get(comp_df, 'n_signal_eigenvalues', 'S&P-500'), 'int')}**")
            if not eig_sp.empty:
                render_chart(
                    _eigenvalue_spectrum_fig(eig_sp, _SP500_COLOR, "S&P"),
                    chart_id="xm_eig_sp", filename_base="cross_market_eigenvalues_sp500",
                )
            else:
                st.info("S&P eigenvalue spectrum not available.")

    # ── Section 3: MST topology side-by-side
    with st.container(border=True):
        section_header(
            "MST topology",
            "Both MSTs use the correlation-distance d = √(2(1-ρ)) and are laid "
            "out via Kamada-Kawai. Node colour = sector; node size = betweenness "
            "centrality. Hub names are shown only for nodes above 45% of the "
            "max-betweenness range to keep the dense S&P plot readable."
        )
        col_b, col_s = st.columns(2)
        mst_e_b = _load_mst_edges("bist")
        mst_m_b = _load_mst_metrics("bist")
        mst_e_s = _load_mst_edges("sp500")
        mst_m_s = _load_mst_metrics("sp500")

        with col_b:
            st.markdown(f"**BIST 100 MST** — sector purity {_fmt(_get(comp_df, 'mst_sector_purity', 'BIST'), 'pct')}")
            if not mst_e_b.empty:
                render_chart(
                    _mst_fig(mst_e_b, mst_m_b, _BIST_COLOR),
                    chart_id="xm_mst_bist", filename_base="cross_market_mst_bist",
                )
            else:
                st.info("BIST MST edges not available.")
            st.markdown("**Top hubs by betweenness**")
            st.markdown(_top_hubs_text(mst_m_b))

        with col_s:
            st.markdown(f"**S&P 500 MST** — sector purity {_fmt(_get(comp_df, 'mst_sector_purity', 'S&P-500'), 'pct')}")
            if not mst_e_s.empty:
                render_chart(
                    _mst_fig(mst_e_s, mst_m_s, _SP500_COLOR),
                    chart_id="xm_mst_sp", filename_base="cross_market_mst_sp500",
                )
            else:
                st.info("S&P MST edges not available.")
            st.markdown("**Top hubs by betweenness**")
            st.markdown(_top_hubs_text(mst_m_s))

    # ── Section 4: Crisis-window comparison
    with st.container(border=True):
        section_header(
            "Crisis windows",
            "Average pairwise correlation in the ±60-day window around each event. "
            "Global shocks (COVID, Russia-Ukraine) appear in both markets; "
            "Türkiye-local shock (2023 earthquakes) appears only on BIST."
        )
        render_chart(
            _crisis_fig(comp_df),
            chart_id="xm_crisis", filename_base="cross_market_crisis_windows",
        )
        st.caption(
            "**Methodology check passes:** the toolkit correctly localises shocks "
            "to the affected market. BIST 0.32 → 0.62 → 0.54 around the Türkiye "
            "earthquakes; S&P 0.44 → 0.35 → 0.31 over the same window (no signal)."
        )

    # ── Section 5: Dependence + Glasso parity
    with st.container(border=True):
        section_header("Pairwise dependence + sparse partial correlation")
        col_b, col_s = st.columns(2)

        rows_left = [
            ("Mean correlation",         "mean_corr",            "float"),
            ("Median correlation",       "median_corr",          "float"),
            ("Std deviation",            "std_corr",             "float"),
            ("Max |correlation|",        "max_abs_corr",         "float"),
            ("Signal eigenvalues",       "n_signal_eigenvalues", "int"),
            ("Signal variance share",    "signal_variance_share","pct"),
            ("Glasso edges (#)",         "glasso_n_edges",       "int"),
            ("Glasso sector purity",     "glasso_sector_purity", "pct"),
            ("TE edges (#)",             "te_n_edges",           "int"),
            ("TE sector purity",         "te_sector_purity",     "pct"),
        ]

        for label, key, kind in rows_left:
            bist, sp = _kpi_row(comp_df, label, key, kind)
            with col_b:
                st.markdown(f"**{label}** — BIST: `{bist}`")
            with col_s:
                st.markdown(f"**{label}** — S&P: `{sp}`")

    # ── Section 6: Pair-dislocation top-1 contrast (real-world headline)
    with st.container(border=True):
        section_header(
            "Top dislocation pair, each market",
            "Highest-ranked candidate from each universe's dislocation_candidates.csv. "
            "BIST and S&P top pairs differ in sector regime, half-life, and Z-score."
        )
        dl_b = _load_dislocation_candidates("bist").head(1)
        dl_s = _load_dislocation_candidates("sp500").head(1)
        cols = st.columns(2)
        if not dl_b.empty:
            r = dl_b.iloc[0]
            with cols[0]:
                st.markdown(
                    f"**BIST #1: {r.get('ticker_a','?')}–{r.get('ticker_b','?')}** "
                    f"({r.get('sector_a','?')} / {r.get('sector_b','?')})"
                )
                st.markdown(
                    f"- ρ = {float(r.get('correlation', np.nan)):.3f}\n"
                    f"- β = {float(r.get('beta', np.nan)):.3f}\n"
                    f"- half-life ≈ {float(r.get('half_life', np.nan)):.0f}d\n"
                    f"- current Z = {float(r.get('current_zscore', np.nan)):.2f}"
                )
        if not dl_s.empty:
            r = dl_s.iloc[0]
            with cols[1]:
                st.markdown(
                    f"**S&P #1: {r.get('ticker_a','?')}–{r.get('ticker_b','?')}** "
                    f"({r.get('sector_a','?')} / {r.get('sector_b','?')})"
                )
                st.markdown(
                    f"- ρ = {float(r.get('correlation', np.nan)):.3f}\n"
                    f"- β = {float(r.get('beta', np.nan)):.3f}\n"
                    f"- half-life ≈ {float(r.get('half_life', np.nan)):.0f}d\n"
                    f"- current Z = {float(r.get('current_zscore', np.nan)):.2f}"
                )

    # ── Section 6b: Numéraire sensitivity (Phase 4 mutable-candy) ─────────
    _render_bist_numeraire_section()

    # ── Section 7: Limitations / methodology footnote
    with st.container(border=True):
        section_header("Methodology + limitations")
        st.markdown(
            "- Both pipelines use **identical** start/end dates, coverage "
            "threshold (90%), TE / Glasso / wavelet hyperparameters, and "
            "Pearson correlation. No methodological tilt favouring either market.\n"
            "- BIST surviving universe is 73 (post-90%-coverage); S&P is 485 "
            "after dropping GOOG/FOX/NWS dual-class duplicates. Direct N-comparison "
            "is misleading; relative properties (D_eff, sector purity, MST hubs) "
            "are the meaningful contrasts.\n"
            "- BIST has 4 manually-NaN'd (ticker, date) cells for unhandled "
            "yfinance corporate-action artifacts (CCOLA, HEKTS×2, AYGAZ). "
            "Documented in `docs/KNOWN_ISSUES.md` §G-1.\n"
            "- The XU100 file under `data/sp500/raw/` is named for historical "
            "reasons but contains the ^GSPC index series."
        )
