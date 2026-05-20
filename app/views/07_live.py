"""Live — upload your own price matrix, run the pipeline in-memory.

PR #78. CSV-only input (no yfinance fetch); results live in
`st.session_state` and disappear when the session ends — by design,
this is the demo-day feature for showing the pipeline on a non-
pre-staged market (DAX, FTSE, etc.) at presentation time.

Compute scope (per user direction 2026-05-20):
  Fast stages: returns, correlation, distance, MST, clustering,
  dislocation candidates, basic IT KPIs, predictability diagnostics.
  Medium stages: wavelet (3 scales), GLASSO (single α), walk-forward
  signals (single end-date snapshot).
  Skipped: full TE, SNN, GLASSO α-path, walk-forward grid.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from live_pipeline import (
    LiveResult, MAX_TICKERS, MAX_ROWS, MIN_TICKERS, MIN_ROWS,
    run_live_pipeline, validate_uploaded_csv,
)
from utils import (
    apply_chart_style, get_colors, inject_custom_css,
    page_header, render_chart, render_matrix_heatmap, section_header,
    SECTOR_PALETTE,
)


# ---------------------------------------------------------------------------
# Page header + input panel
# ---------------------------------------------------------------------------

inject_custom_css()
page_header(
    "Live pipeline",
    "Upload a price matrix CSV. Pipeline runs in-memory on this HF Space, "
    "results stay in your session.",
)

st.markdown(
    "**CSV format**: first column = `Date` (YYYY-MM-DD), remaining columns = "
    "ticker symbols with adjusted close prices. Need ≥ "
    f"{MIN_ROWS} rows and ≥ {MIN_TICKERS} tickers. Hard caps: {MAX_ROWS} rows, "
    f"{MAX_TICKERS} tickers. Nothing is persisted — by design."
)

_uploaded = st.file_uploader(
    "Price-matrix CSV",
    type=["csv"],
    accept_multiple_files=False,
    key="live_csv_uploader",
    help="Drag-and-drop or browse. Plain CSV only (no Excel, no zip).",
)

# Stash the parsed DataFrame across reruns. The uploader fires on every
# script run while a file is selected, so cache the validation output
# keyed by file content hash.
if _uploaded is not None:
    file_bytes = _uploaded.getvalue()
    content_key = f"live_csv_hash_{hash(file_bytes)}"
    if content_key not in st.session_state:
        prices_df, errors, warnings = validate_uploaded_csv(file_bytes)
        st.session_state[content_key] = {
            "prices": prices_df, "errors": errors, "warnings": warnings,
        }
    parsed = st.session_state[content_key]
    prices_df = parsed["prices"]
    errors = parsed["errors"]
    warnings = parsed["warnings"]

    if errors:
        for e in errors:
            st.error(e)
        st.stop()
    for w in warnings:
        st.warning(w)
    st.success(
        f"Loaded {prices_df.shape[0]} rows × {prices_df.shape[1]} tickers "
        f"({prices_df.index.min().date()} → {prices_df.index.max().date()})"
    )
else:
    st.info("Upload a CSV above to enable the **Run pipeline** button.")
    st.stop()


_run_clicked = st.button(
    "Run pipeline",
    type="primary",
    key="live_run_btn",
    disabled=prices_df.empty,
)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if _run_clicked:
    # Clear any prior results for this session.
    st.session_state.pop("_live_results", None)

    status = st.status("Running pipeline…", expanded=True)
    progress_lines: list[str] = []

    def _progress_cb(name: str, state: str, message: str) -> None:
        # Render one line per stage transition. Streamlit's st.status
        # accepts incremental write calls.
        if state == "running":
            line = f":material/hourglass_top: **{name}** — running…"
        elif state == "ok":
            line = f":material/check_circle: **{name}** — done"
        elif state == "error":
            line = f":material/error: **{name}** — skipped ({message})"
        else:
            line = f"**{name}** — {state}"
        progress_lines.append(line)
        # Re-render with all lines so far.
        with status:
            for ln in progress_lines:
                st.markdown(ln)

    t0 = time.perf_counter()
    result = run_live_pipeline(prices_df, progress_cb=_progress_cb)
    elapsed = time.perf_counter() - t0
    status.update(label=f"Pipeline complete in {elapsed:.1f} s", state="complete", expanded=False)
    st.session_state["_live_results"] = result

# ---------------------------------------------------------------------------
# Results panels (render whatever's in session_state)
# ---------------------------------------------------------------------------

result: LiveResult | None = st.session_state.get("_live_results")
if result is None:
    st.stop()

colors = get_colors()

# ── KPI strip ──
section_header(
    "Summary",
    f"In-memory run on {result.universe_label}. "
    f"{int(result.it_kpis.get('n_tickers', 0))} tickers × "
    f"{int(result.it_kpis.get('n_observations', 0))} days. "
    f"Wall time: {result.wall_time_sec:.1f} s.",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers", int(result.it_kpis.get("n_tickers", 0)))
c2.metric(
    "D_eff",
    f"{result.it_kpis.get('d_eff', float('nan')):.2f}",
    help="Participation ratio of the correlation eigenspectrum. Lower = stronger factor structure.",
)
c3.metric(
    "ΔH (nats)",
    f"{result.it_kpis.get('log_det_term', float('nan')):.2f}",
    help="Joint-Gaussian structure: −½ log det Σ.",
)
c4.metric(
    "Mean sign-H (bits)",
    f"{result.it_kpis.get('mean_sign_entropy_rate_bits', float('nan')):.3f}",
    help="1.0 = direction unpredictable from lag-1 sign.",
)

# ── Correlation heatmap ──
if not result.correlation.empty:
    section_header("Correlation matrix")
    order = None
    if not result.cluster_assignments.empty:
        # Order by cluster id for visual grouping.
        order = result.cluster_assignments.sort_values("cluster_id")["ticker"].tolist()
    render_matrix_heatmap(
        result.correlation,
        chart_id="live_corr_heatmap",
        filename_base="live_correlation",
        title_key="live_corr",
        default_title="Pearson correlation (clustered)",
        ordered_tickers=order,
        zmin=-1.0, zmax=1.0, diverging=True,
        height=520, hover_label="ρ",
    )

# ── MST plot ──
if not result.mst_edges.empty:
    section_header("Minimum spanning tree")
    try:
        import networkx as nx
    except ImportError:
        nx = None

    if nx is not None:
        G = nx.Graph()
        for _, r in result.mst_edges.iterrows():
            G.add_edge(r["source"], r["target"], weight=float(r.get("distance", 1.0)))
        pos = nx.kamada_kawai_layout(G)

        edge_x: list[float] = []
        edge_y: list[float] = []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        cluster_map: dict = {}
        if not result.cluster_assignments.empty:
            cluster_map = dict(zip(
                result.cluster_assignments["ticker"],
                result.cluster_assignments["cluster_id"],
            ))

        node_x: list[float] = []
        node_y: list[float] = []
        node_text: list[str] = []
        node_color: list = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x); node_y.append(y)
            cid = cluster_map.get(node, 0)
            node_color.append(SECTOR_PALETTE[int(cid) % len(SECTOR_PALETTE)])
            deg = G.degree(node)
            node_text.append(f"<b>{node}</b><br>degree: {deg}<br>cluster: {cid}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.6, color="#999999"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=10, color=node_color, line=dict(width=0.5, color="#fff")),
            text=list(G.nodes()), textposition="top center", textfont=dict(size=8),
            hovertext=node_text, hoverinfo="text", showlegend=False,
        ))
        apply_chart_style(
            fig, height=520,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       scaleanchor="x", scaleratio=1),
        )
        render_chart(
            fig, chart_id="live_mst",
            filename_base="live_mst",
            default_title=f"Minimum spanning tree ({G.number_of_nodes()} nodes, "
                          f"{G.number_of_edges()} edges) — node color = cluster",
        )
    else:
        st.info("Install `networkx` to render the MST plot.")

# ── Dislocation candidates ──
if not result.dislocation_candidates.empty:
    section_header("Top dislocation candidates")
    disp = result.dislocation_candidates.copy()
    for col in ["correlation", "beta", "spread_std", "current_zscore", "rank_score"]:
        if col in disp.columns:
            disp[col] = disp[col].round(4)
    if "half_life" in disp.columns:
        disp["half_life"] = disp["half_life"].round(1)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption(
        f"Ranked by composite score on {len(disp)} pairs. Half-life shown in trading days; "
        "current_zscore is the most-recent value of the rolling pair spread."
    )

# ── Walk-forward signals ──
if not result.walk_forward_signals.empty:
    section_header("Active pair signals (latest snapshot)")
    wf = result.walk_forward_signals.copy()
    for col in ["current_zscore", "correlation", "spread_std"]:
        if col in wf.columns:
            wf[col] = wf[col].round(4)
    if "half_life" in wf.columns:
        wf["half_life"] = wf["half_life"].round(1)
    st.dataframe(wf, use_container_width=True, hide_index=True)
    st.caption(
        f"Snapshot at {result.returns.index[-1].date()}. Each row's `status` "
        "tells you whether the pair is LONG/SHORT-entry, in-position, near, or flat."
    )

# ── Predictability diagnostics ──
predict_df = result.it_kpis.get("predictability_table")
if predict_df is not None and not predict_df.empty:
    section_header(
        "Predictability diagnostics",
        "Volatility clustering (ACF |r|), Hurst exponent, raw return autocorrelation.",
    )
    disp = predict_df.copy().head(15)
    for col in ["sign_entropy_bits", "acf_returns_lag1", "acf_abs_returns_lag1",
                "acf_abs_returns_lag5", "acf_abs_returns_lag22", "hurst_exponent"]:
        if col in disp.columns:
            disp[col] = disp[col].round(4)
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ── Wavelet correlation matrices (3 scales) ──
if result.wavelet_corrs:
    section_header(
        "Wavelet correlation (multi-scale)",
        f"Pairwise Pearson correlation of wavelet detail coefficients at "
        f"scales {sorted(result.wavelet_corrs.keys())}.",
    )
    scale_tabs = st.tabs([f"Scale {s}" for s in sorted(result.wavelet_corrs.keys())])
    for tab, s in zip(scale_tabs, sorted(result.wavelet_corrs.keys())):
        with tab:
            mat = result.wavelet_corrs[s]
            render_matrix_heatmap(
                mat,
                chart_id=f"live_wavelet_scale{s}",
                filename_base=f"live_wavelet_scale{s}",
                title_key=f"live_wave_{s}",
                default_title=f"Wavelet detail correlation (scale {s})",
                zmin=-1.0, zmax=1.0, diverging=True,
                height=440, hover_label="ρ",
            )

# ── Partial correlation (GLASSO) ──
if not result.partial_corr.empty:
    section_header(
        "Partial correlation (Graphical LASSO)",
        "Sparse precision-matrix-derived partial correlations: pair coupling after controlling for all others.",
    )
    render_matrix_heatmap(
        result.partial_corr,
        chart_id="live_partial_corr",
        filename_base="live_partial_correlation",
        title_key="live_partial",
        default_title="Partial correlation (CV-tuned α)",
        zmin=-1.0, zmax=1.0, diverging=True,
        height=520, hover_label="partial ρ",
    )

# ── Stage status block (always at the bottom for transparency) ──
section_header("Stage status")
status_rows = []
for stage, (state, msg) in result.stage_status.items():
    status_rows.append({
        "stage": stage,
        "status": state,
        "message": msg or "—",
    })
if status_rows:
    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True, hide_index=True,
    )
st.caption(f"Run completed in {result.wall_time_sec:.1f} s.")
