"""EEE Analysis dashboard tab — RMT, Graphical LASSO, Wavelets, Transfer Entropy, SNN."""

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

from utils import (
    get_colors, SECTOR_PALETTE, apply_chart_style, section_header, render_chart,
    render_matrix_heatmap,
    current_universe, _load_log_returns,
    load_eigenvalue_spectrum, load_denoised_corr, load_denoised_mst_edges,
    load_denoised_mst_metrics,
    load_mst_edges, load_mst_metrics, load_batch_corr,
    load_partial_corr, load_partial_corr_edges, load_glasso_metadata,
    load_precision_matrix, load_glasso_alpha_path,
    load_wavelet_metadata, load_wavelet_mst_edges, load_wavelet_corr,
    load_wavelet_mst_metrics,
    load_te_edges, load_te_node_roles, load_te_matrix, load_net_te_matrix,
    load_te_matrix_raw,
    load_cluster_assignments,
    load_snn_metrics, load_snn_pair_list, load_snn_signals,
    load_snn_training_history, load_snn_raster_sample, load_snn_membrane_sample,
    load_dendrogram_order,
    load_mi_matrix, load_mi_gaussian_matrix, load_mi_nonlinear_excess_top,
    load_mi_nonlinear_excess,
    load_rolling_info_theory, load_regime_kl, load_it_summary,
    load_entropy_rate_signs,
    load_te_summary, load_te_pvalues, load_te_significance,
    load_permutation_entropy,
    load_te_lag_sweep, load_rolling_te, load_te_with_ci,
    load_mi_excess_with_ci,
    load_predictability_diagnostics,
    load_te_conditional_market, load_te_sector_matrix,
    downsample_matrix_for_display,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_nx_graph(edges_df: pd.DataFrame, directed: bool = False) -> "nx.Graph":
    """Build a NetworkX graph from an edge DataFrame."""
    G = nx.DiGraph() if directed else nx.Graph()
    for _, r in edges_df.iterrows():
        w = r.get("distance", r.get("abs_partial_corr", r.get("net_te", 1.0)))
        G.add_edge(r["source"], r["target"], weight=abs(float(w)))
    return G


def _edges_from_raw_te(raw_te: pd.DataFrame, top_k: int = 200) -> pd.DataFrame:
    """Build the directed-edge table from the raw (pre-FDR) TE matrix.

    Used when the FDR-corrected edge list is empty so the dashboard still
    has something meaningful to display: the network of the top-k pairs
    ranked by raw TE magnitude.
    """
    tickers = raw_te.columns.tolist()
    rows = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            te_ij = float(raw_te.iloc[i, j])
            te_ji = float(raw_te.iloc[j, i])
            net = te_ij - te_ji
            rows.append({
                "source": tickers[i],
                "target": tickers[j],
                "te_forward": te_ij,
                "te_backward": te_ji,
                "net_te": net,
                "dominant_direction": (
                    f"{tickers[i]}->{tickers[j]}" if net > 0
                    else f"{tickers[j]}->{tickers[i]}"
                ),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "source", "target", "te_forward", "te_backward",
            "net_te", "dominant_direction",
        ])
    df = pd.DataFrame(rows)
    df = df.assign(abs_net_te=df["net_te"].abs())
    return (
        df.sort_values("abs_net_te", ascending=False)
        .head(top_k)
        .drop(columns="abs_net_te")
        .reset_index(drop=True)
    )


def _node_roles_from_raw_te(raw_te: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct source/sink role assignments from raw TE row/column sums."""
    tickers = raw_te.columns.tolist()
    rows = []
    for ticker in tickers:
        te_out = float(raw_te.loc[ticker].sum())
        te_in = float(raw_te[ticker].sum())
        net = te_out - te_in
        rows.append({
            "ticker": ticker,
            "te_out": te_out,
            "te_in": te_in,
            "net_te_flow": net,
            "role": "source" if net > 0 else "sink",
        })
    return (
        pd.DataFrame(rows)
        .sort_values("net_te_flow", ascending=False)
        .reset_index(drop=True)
    )


def _plot_network(
    edges_df: pd.DataFrame,
    sector_map: dict,
    title: str = "",
    directed: bool = False,
    edge_weight_col: str = "distance",
    height: int = 600,
    node_metrics: pd.DataFrame | None = None,
    size_metric: str = "betweenness_centrality",
    size_range: tuple[int, int] = (8, 28),
    sector_node_label: str = "Sector",
    *,
    pos: dict[str, tuple[float, float]] | None = None,
    layout_source: str | None = None,
) -> go.Figure:
    """Create a Plotly network graph from edges.

    If ``node_metrics`` is provided (with columns ``ticker`` and ``size_metric``),
    nodes are sized by that centrality measure mapped onto ``size_range``;
    otherwise size scales with degree (legacy behaviour).

    PHASE Y (Y2): callers can pre-supply layout positions via ``pos=`` OR
    name a precomputed source via ``layout_source=`` (e.g.,
    ``"denoised_mst"``, ``"wavelet_mst_scale3"``, ``"te_network"``). The
    function then tries the precomputed layout first and falls back to
    live nx.kamada_kawai_layout when neither is provided or the
    precomputed file is missing. Saves ~1-2 s per render on S&P.
    """
    if not HAS_NETWORKX or edges_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=16))
        return fig

    G = nx.DiGraph() if directed else nx.Graph()
    for _, r in edges_df.iterrows():
        w = abs(float(r.get(edge_weight_col, 1.0)))
        G.add_edge(r["source"], r["target"], weight=w if w > 0 else 0.01)

    # PHASE Y (Y2): try precomputed positions first.
    if pos is None and layout_source is not None:
        from utils import load_mst_layout
        precomputed = load_mst_layout(layout_source)
        if precomputed:
            # Only use precomputed positions for nodes that exist in this
            # graph (defensive against schema drift between snapshot pipeline
            # output + the edges CSV the caller passed in).
            graph_nodes = set(G.nodes())
            pos_filtered = {n: precomputed[n] for n in graph_nodes if n in precomputed}
            # Need positions for ALL nodes; if any missing, fall back to live.
            if len(pos_filtered) == len(graph_nodes):
                pos = pos_filtered
    if pos is None:
        pos = nx.kamada_kawai_layout(G, weight="weight")

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#ccc"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Nodes colored by sector; sized by metric (if provided) else degree
    sectors = sorted(set(sector_map.values()))
    sector_color_map = {s: SECTOR_PALETTE[i % len(SECTOR_PALETTE)] for i, s in enumerate(sectors)}

    metric_lookup: dict[str, float] = {}
    metric_min = metric_max = None
    metric_label = ""
    if (
        node_metrics is not None
        and not node_metrics.empty
        and "ticker" in node_metrics.columns
        and size_metric in node_metrics.columns
    ):
        metric_label = size_metric.replace("_", " ")
        metric_lookup = dict(zip(node_metrics["ticker"], node_metrics[size_metric].astype(float)))
        finite = [v for v in metric_lookup.values() if np.isfinite(v)]
        if finite:
            metric_min = float(np.min(finite))
            metric_max = float(np.max(finite))

    lo, hi = size_range
    span = (metric_max - metric_min) if (metric_max is not None and metric_max > metric_min) else None

    for node in G.nodes():
        x, y = pos[node]
        sector = sector_map.get(node, "Unknown")
        color = sector_color_map.get(sector, "#999")
        deg = G.degree(node)

        if span is not None and node in metric_lookup:
            metric_val = float(metric_lookup[node])
            normalised = (metric_val - metric_min) / span
            size = lo + normalised * (hi - lo)
            metric_text = f"<br>{metric_label}: {metric_val:.4f}"
        else:
            # Clamp the legacy degree-based fallback to size_range so that
            # high-degree nodes (e.g. GLASSO partial-corr hubs with deg ≥ 20)
            # don't balloon to ~50px and overlap their neighbours.
            size = max(lo, min(hi, 8 + deg * 2))
            metric_text = ""

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=size, color=color, line=dict(width=0.5, color="#fff")),
            text=node, textposition="top center", textfont=dict(size=7),
            hovertemplate=(
                f"<b>{node}</b><br>{sector_node_label}: {sector}<br>Degree: {deg}"
                f"{metric_text}<extra></extra>"
            ),
            showlegend=False,
        ))

    # Sector legend
    for sector in sectors:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color=sector_color_map[sector]),
            name=sector,
        ))

    apply_chart_style(fig, height=height,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return fig


def _plot_matrix_heatmap(
    matrix: pd.DataFrame,
    ordered_tickers: list[str] | None,
    *,
    zmin: float = -1.0,
    zmax: float = 1.0,
    diverging: bool = True,
    height: int = 480,
    hover_label: str = "value",
    max_display_dim: int = 200,
) -> go.Figure:
    """Render a square matrix as a heatmap, dendrogram-reordered when possible.

    For matrices larger than ``max_display_dim`` × ``max_display_dim`` the
    values are block-averaged down to that resolution so the JSON payload
    stays under the streamlit websocket cap (a full S&P 485×485 matrix is
    ~4.5 MB serialized; a 97×97 block-averaged version is ~0.18 MB).
    """
    if matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=14))
        apply_chart_style(fig, height=height)
        return fig

    if ordered_tickers:
        present = [t for t in ordered_tickers if t in matrix.columns and t in matrix.index]
        if len(present) >= 2:
            matrix = matrix.loc[present, present]

    matrix, block_size = downsample_matrix_for_display(matrix, max_dim=max_display_dim)
    n_after = matrix.shape[0]

    colorscale = "RdBu" if diverging else "Blues"
    if block_size > 1:
        # Per-cell hover would name a single (i, j) pair, but the cell is a
        # block-mean over `block_size² ≈ {bs}` pairs. Show the block range
        # so the hover stays accurate.
        bs = block_size
        hovertemplate = (
            f"block ({bs}×{bs}) mean<br>row %{{y}} · col %{{x}}<br>"
            f"{hover_label}=%{{z:.4f}}<extra></extra>"
        )
    else:
        hovertemplate = f"%{{y}} ↔ %{{x}}<br>{hover_label}=%{{z:.4f}}<extra></extra>"

    # Suppress per-cell tick labels for medium-N matrices; render only when
    # they're legible.
    _show_labels = n_after <= 80
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=list(matrix.columns),
        y=list(matrix.index),
        zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        reversescale=diverging,
        zmid=0 if diverging else None,
        hovertemplate=hovertemplate,
        colorbar=dict(thickness=12, len=0.85),
    ))
    apply_chart_style(
        fig, height=height,
        xaxis=dict(tickfont=dict(size=8), tickangle=-90, showticklabels=_show_labels),
        yaxis=dict(tickfont=dict(size=8), autorange="reversed", showticklabels=_show_labels),
    )
    return fig


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _sector_label(u) -> str:
    """Anatomical-region for EEG, Sector for finance, etc. None-safe."""
    return getattr(u, "sector_label", "Sector") if u is not None else "Sector"


def _item_label(u) -> str:
    return getattr(u, "item_label", "Ticker") if u is not None else "Ticker"


def _items_label(u) -> str:
    return getattr(u, "items_label", "Tickers") if u is not None else "Tickers"


@st.fragment
def render_rmt(sector_map: dict, *, u=None):
    """Render RMT denoising section."""
    with st.container(border=True):
        section_header(
            "RMT Correlation Denoising",
            "Random Matrix Theory separates signal from noise in the correlation matrix "
            "using the Marchenko-Pastur distribution. Eigenvalues within the MP bounds "
            "are indistinguishable from random noise.",
        )

        spectrum = load_eigenvalue_spectrum()
        if spectrum.empty:
            st.info("Run the pipeline to generate RMT results.")
            return

        n_signal = int(spectrum["is_signal"].sum())
        n_noise = len(spectrum) - n_signal
        mp_upper = spectrum["mp_upper"].iloc[0]
        mp_lower = spectrum["mp_lower"].iloc[0]

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal Eigenvalues", n_signal)
        c2.metric("Noise Eigenvalues", n_noise)
        c3.metric("MP Upper Bound", f"{mp_upper:.3f}")
        c4.metric("Variance Explained (Signal)", f"{spectrum.loc[spectrum['is_signal'], 'explained_variance_pct'].sum():.1f}%")

        # PORT arda/ui-cleanup item 9: stack eigenvalue spectrum + MST
        # vertically (full-width each) instead of the prior 50/50 split.
        # Spectrum height 420 → 600; MST height overridden to 700 after
        # `_plot_network` returns (helper default is smaller).
        colors = get_colors()
        fig_spec = go.Figure()

        # MP noise band
        fig_spec.add_hrect(
            y0=mp_lower, y1=mp_upper,
            fillcolor="rgba(230,57,70,0.12)", line_width=0,
            annotation_text="MP Noise Band", annotation_position="top right",
            annotation_font_size=10,
        )

        # Eigenvalues as bar chart
        bar_colors = [colors["primary"] if s else colors["muted"]
                      for s in spectrum["is_signal"]]
        fig_spec.add_trace(go.Bar(
            x=list(range(1, len(spectrum) + 1)),
            y=spectrum["eigenvalue"].values,
            marker_color=bar_colors,
            hovertemplate="Eigenvalue #%{x}: %{y:.3f}<extra></extra>",
        ))

        fig_spec.add_hline(y=mp_upper, line_dash="dash", line_color="#E63946",
                           annotation_text=f"MP upper = {mp_upper:.3f}",
                           annotation_font_size=10)
        apply_chart_style(fig_spec, height=600,
                          xaxis_title="Eigenvalue Index",
                          yaxis_title="Eigenvalue",
                          yaxis_type="log",
                          showlegend=False)
        render_chart(fig_spec, chart_id="rmt_spectrum", filename_base="eigenvalue_spectrum",
                     title_key="rmt_spectrum", default_title="Eigenvalue Spectrum vs MP Bounds")

        # MST view — full-width below the spectrum.
        raw_edges = load_mst_edges()
        denoised_edges = load_denoised_mst_edges()

        mst_choice = st.radio("MST View", ["Raw", "Denoised", "Both (overlay)"],
                              horizontal=True, key="rmt_mst_view")

        if mst_choice == "Raw":
            metrics_df = load_mst_metrics()
            fig = _plot_network(
                raw_edges, sector_map,
                edge_weight_col="distance",
                node_metrics=metrics_df,
                sector_node_label=_sector_label(u),
                layout_source="main_mst",
            )
        else:
            metrics_df = load_denoised_mst_metrics()
            fig = _plot_network(
                denoised_edges, sector_map,
                edge_weight_col="distance",
                node_metrics=metrics_df,
                sector_node_label=_sector_label(u),
                layout_source="denoised_mst",
            )

        # Bump MST height after the helper builds the figure (helper default
        # was sized for the 50/50 split; full-width needs more vertical room).
        fig.update_layout(height=700)
        render_chart(fig, chart_id="rmt_mst", filename_base="rmt_mst",
                     title_key="rmt_mst",
                     default_title=f"MST Network ({mst_choice}, nodes sized by betweenness)")
        # Subtitle clarifies what this MST is vs Clustering & Network's
        # raw MST and the Wavelet per-scale MSTs.
        if mst_choice == "Raw":
            st.caption(
                "Raw MST on Pearson correlation distance — same metric as "
                "**Clustering & Network → MST**. Toggle 'Denoised' to see "
                "the RMT-cleaned version."
            )
        elif mst_choice == "Denoised":
            st.caption(
                "Built on the **denoised** correlation matrix — noise "
                "eigenvalues (inside the Marchenko–Pastur band) replaced "
                "by their mean before reconstruction. Signal-only network "
                "backbone."
            )
        else:  # "Both (overlay)"
            st.caption(
                "Raw and Denoised MSTs overlaid on the same Kamada-Kawai "
                "layout — edges that survive denoising are the structurally "
                "meaningful ones."
            )

        # Denoised correlation heatmap (full width)
        st.markdown("**Denoised Correlation Matrix** — eigenvalues outside the MP band reconstructed; noise eigenvalues replaced with their mean.")
        denoised = load_denoised_corr()
        order = load_dendrogram_order()
        render_matrix_heatmap(
            denoised,
            chart_id="rmt_denoised_heatmap",
            filename_base="rmt_denoised_corr_heatmap",
            title_key="rmt_den_hm",
            default_title="Denoised correlation heatmap (ordered by dendrogram leaves)",
            ordered_tickers=order,
            zmin=-1.0, zmax=1.0, diverging=True,
            height=520, hover_label="ρ (denoised)",
        )


@st.cache_data(show_spinner="Refitting Graphical LASSO with custom α…")
def _recompute_glasso(
    universe_key: str, alpha: float, max_iter: int = 200,
) -> tuple["pd.DataFrame", dict, "pd.DataFrame", "pd.DataFrame"]:
    """Refit GLASSO on the fly for an interactive α override.

    Cached by (universe_key, alpha) so repeat slider stops are free. Returns
    the four artifacts the pipeline produces (edges, meta, partial_corr,
    precision) with matching schemas so downstream rendering is invariant to
    whether the data came from disk or this recompute.

    Skips cross-validation (alpha is supplied), so cost is the single
    GraphicalLasso fit — typically 1-5 s on BIST (~73 tickers), longer on S&P.
    """
    from src.partial_correlation import (
        fit_graphical_lasso, extract_partial_corr_edges,
    )

    returns = _load_log_returns(universe_key)
    precision_df, partial_df, alpha_used = fit_graphical_lasso(
        returns, alpha=alpha, max_iter=max_iter,
    )
    edges_df = extract_partial_corr_edges(partial_df, threshold=0.01)
    n_tickers = len(partial_df)
    total_offdiag = (n_tickers * (n_tickers - 1) / 2) if n_tickers > 1 else 1.0
    meta = {
        "alpha": float(alpha_used),
        "n_edges": int(len(edges_df)),
        "n_tickers": int(n_tickers),
        "sparsity_pct": round(
            100 * (1 - len(edges_df) / total_offdiag), 2
        ),
    }
    return edges_df, meta, partial_df, precision_df


@st.fragment
def render_glasso(sector_map: dict, *, u=None):
    """Render Graphical LASSO section."""
    with st.container(border=True):
        section_header(
            "Graphical LASSO — Partial Correlation Network",
            "L1-regularized sparse precision matrix estimation. Non-zero entries represent "
            "direct (conditional) dependencies, filtering out indirect correlations. "
            "If A correlates with B only because both correlate with C, the GLASSO removes the A-B edge.",
        )

        # Permanent explainer for the L1 regularization knob — sits above the
        # source selector so the slider's effect is obvious before users touch
        # it. Bullet form keeps it scannable; the expander hides it after
        # first read but it persists across reloads via streamlit's widget key.
        with st.expander(
            ":material/help: How **α** (L1 regularization) shapes the network",
            expanded=False,
        ):
            st.markdown(
                "GLASSO penalises the L1 norm of off-diagonal precision entries. "
                "**α** sets how aggressively weak partial correlations get zeroed out:\n\n"
                "- **Higher α → stronger penalty → sparser network.** Only the most "
                "robust direct dependencies survive. Push α high enough and every "
                "edge is pruned (precision matrix becomes diagonal → everything "
                "conditionally independent).\n"
                "- **Lower α → weaker penalty → denser network.** More edges, "
                "including weaker / noisier ones. At α = 0 GLASSO collapses to the "
                "unpenalised inverse covariance, which is often ill-conditioned "
                "when N is close to T.\n"
                "- **Pipeline default uses 5-fold cross-validation** "
                "(`GraphicalLassoCV`) to pick α by held-out log-likelihood — a "
                "data-driven middle ground. The custom slider below skips CV and "
                "uses the α you choose directly."
            )

        # Manual α entry. Defaults to the pipeline's CV-chosen α so the
        # initial state mirrors the on-disk artifacts; type any value to refit
        # GLASSO live. Recomputes are cached by (universe, α) so values you
        # revisit are instant. Values within fp-noise of the pipeline α reuse
        # the disk artifacts directly — no spurious refit on first mount.
        _pipeline_meta = load_glasso_metadata() or {}
        _pipeline_alpha = float(_pipeline_meta.get("alpha", 0.05) or 0.05)
        alpha_value = st.number_input(
            "α (L1 regularization strength)",
            min_value=1e-6,
            max_value=10.0,
            value=_pipeline_alpha,
            step=0.0001,
            format="%.6f",
            key="glasso_alpha_input",
            help=(
                f"Pipeline CV picked α ≈ {_pipeline_alpha:.6f}. "
                "Lower α → denser network (more edges, including weaker ones). "
                "Higher α → sparser network (only the strongest direct links survive). "
                "Type a new value and the page refits GLASSO with that α."
            ),
        )

        if abs(alpha_value - _pipeline_alpha) < 1e-6:
            # Pipeline default — reuse disk artifacts (no refit cost).
            edges = load_partial_corr_edges()
            meta = load_glasso_metadata()
            partial = load_partial_corr()
            precision = load_precision_matrix()
        else:
            try:
                edges, meta, partial, precision = _recompute_glasso(
                    current_universe(), float(alpha_value),
                )
            except Exception as exc:
                st.error(
                    f"Refitting Graphical LASSO with α = {alpha_value:.4f} failed: {exc}"
                )
                st.caption(
                    "Falling back to the pipeline result. Try a different α or "
                    "check the logs."
                )
                edges = load_partial_corr_edges()
                meta = load_glasso_metadata()
                partial = load_partial_corr()
                precision = load_precision_matrix()

        if edges.empty:
            if abs(alpha_value - _pipeline_alpha) >= 1e-6:
                st.warning(
                    f"No edges survive at α = {meta.get('alpha', 0):.4f} — the L1 "
                    "penalty has pruned the entire network. Lower α to recover edges."
                )
                # Still render the matrices below; only the network/table sections
                # have nothing to show, so guard them but not the heatmaps.
            else:
                st.info("Run the pipeline to generate GLASSO results.")
                return

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Direct Edges", meta.get("n_edges", len(edges)))
        c2.metric("Sparsity", f"{meta.get('sparsity_pct', 0):.1f}%")
        c3.metric("Regularization (alpha)", f"{meta.get('alpha', 0):.4f}")

        # Row 1: partial correlation network (full width — needs the room so
        # the kamada-kawai layout can spread out and ticker labels stay legible).
        fig = _plot_network(edges, sector_map,
                            edge_weight_col="abs_partial_corr",
                            title="Partial Correlation Network",
                            height=720,
                            sector_node_label=_sector_label(u))
        render_chart(fig, chart_id="glasso_net", filename_base="glasso_network",
                     title_key="glasso_net",
                     default_title="Partial Correlation Network (Direct Dependencies)")

        # Row 2: per-ticker connection inspector. Native plotly click events
        # would require collapsing the one-trace-per-node structure of
        # _plot_network (each node is its own Scatter trace, so a click yields
        # a curve_number, not a ticker). Dropdown is the lightweight option
        # that lets users drill into a specific node's neighbours.
        _item = getattr(u, "item_label", "Ticker") if u is not None else "Ticker"
        _sec = _sector_label(u)

        # Skip the inspector + top-30 sections entirely when there are no
        # edges (custom α set high enough to prune everything). The matrices
        # below still render — they're informative even with zero edges.
        if not edges.empty:
            st.markdown(f"**Inspect a {_item}'s Direct Connections**")
            all_tickers = sorted(set(edges["source"]).union(set(edges["target"])))
            if all_tickers:
                selected = st.selectbox(
                    f"Select {_item.lower()}",
                    all_tickers,
                    key="glasso_inspect_ticker",
                    label_visibility="collapsed",
                )
                mask = (edges["source"] == selected) | (edges["target"] == selected)
                conn = edges.loc[mask].copy()
                if conn.empty:
                    st.info(
                        f"{selected} has no direct dependencies in the GLASSO network — "
                        f"isolated under L1 regularization at α = {meta.get('alpha', 0):.4f}."
                    )
                else:
                    conn["neighbor"] = np.where(
                        conn["source"] == selected, conn["target"], conn["source"]
                    )
                    conn["neighbor_sector"] = conn["neighbor"].map(sector_map)
                    conn = conn[["neighbor", "partial_correlation", "neighbor_sector"]]
                    conn = conn.sort_values(
                        "partial_correlation",
                        key=lambda s: s.abs(),
                        ascending=False,
                    )
                    st.caption(
                        f"**{selected}** ({sector_map.get(selected, 'Unknown')}) connects to "
                        f"**{len(conn)}** other {_item.lower()}{'s' if len(conn) != 1 else ''} "
                        f"under GLASSO."
                    )
                    st.dataframe(
                        conn,
                        use_container_width=True,
                        height=min(420, 70 + 35 * len(conn)),
                        column_config={
                            "neighbor": st.column_config.TextColumn(f"Connected {_item}"),
                            "partial_correlation": st.column_config.NumberColumn(
                                "Partial Corr.", format="%.4f"
                            ),
                            "neighbor_sector": st.column_config.TextColumn(_sec),
                        },
                        hide_index=True,
                    )

            # Row 3: strongest direct dependencies — top-30 globally (own row,
            # full width). `_sec` and `_item` were hoisted above for the inspector.
            st.markdown("---")
            st.markdown("**Strongest Direct Dependencies (top 30 globally)**")
            display_edges = edges.head(30).copy()
            display_edges["sector_1"] = display_edges["source"].map(sector_map)
            display_edges["sector_2"] = display_edges["target"].map(sector_map)
            st.dataframe(
                display_edges[["source", "target", "partial_correlation", "sector_1", "sector_2"]],
                use_container_width=True, height=420,
                column_config={
                    "source": st.column_config.TextColumn(f"Source {_item}"),
                    "target": st.column_config.TextColumn(f"Target {_item}"),
                    "partial_correlation": st.column_config.NumberColumn("Partial Corr.", format="%.4f"),
                    "sector_1": st.column_config.TextColumn(f"{_sec} (Source)"),
                    "sector_2": st.column_config.TextColumn(f"{_sec} (Target)"),
                },
            )

        # Row 4: partial correlation heatmap (full width). `partial` was
        # already populated above based on the α source toggle.
        st.markdown("---")
        order = load_dendrogram_order()
        st.markdown(
            f"**Partial Correlation Matrix** — direct dependencies after conditioning "
            f"on all other {_items_label(u).lower()} (clipped to ±0.3 for visibility)."
        )
        if not partial.empty:
            # Zero the diagonal so it doesn't dominate the colorscale.
            # Use to_numpy(copy=True) → fresh writable ndarray, then wrap back
            # into a DataFrame. partial.copy() alone yields a read-only ndarray
            # under pandas 2.x copy-on-write, so np.fill_diagonal(...values)
            # raises "underlying array is read-only".
            _pc_arr = partial.to_numpy(copy=True)
            np.fill_diagonal(_pc_arr, 0.0)
            pc_display = pd.DataFrame(_pc_arr, index=partial.index, columns=partial.columns)
            render_matrix_heatmap(
                pc_display,
                chart_id="glasso_partial_heatmap",
                filename_base="glasso_partial_corr_heatmap",
                title_key="glasso_pc_hm",
                default_title="Partial correlation heatmap",
                ordered_tickers=order,
                zmin=-0.3, zmax=0.3, diverging=True,
                height=520, hover_label="partial ρ",
            )
        else:
            st.info("Run the pipeline to generate the partial correlation matrix.")

        # Row 5: precision matrix sparsity (stacked below, not side-by-side).
        # `precision` was populated above based on the α source toggle.
        st.markdown("**Precision Matrix Sparsity** — non-zero off-diagonal entries are direct conditional dependencies. Zero entries imply conditional independence under Gaussianity.")
        if not precision.empty:
            # Build a binary sparsity pattern of |Θ_ij| above a small floor.
            # Same CoW concern as the partial-corr block above: copy through a
            # writable ndarray before fill_diagonal.
            _prec_arr = precision.abs().to_numpy(copy=True)
            np.fill_diagonal(_prec_arr, 0.0)
            prec_abs = pd.DataFrame(_prec_arr, index=precision.index, columns=precision.columns)
            threshold = 1e-3
            sparsity = (prec_abs > threshold).astype(float)
            n_offdiag = sparsity.values.sum() // 2  # symmetric
            total_offdiag = (sparsity.shape[0] * (sparsity.shape[0] - 1)) // 2
            density = (n_offdiag / total_offdiag * 100) if total_offdiag else 0.0
            render_matrix_heatmap(
                sparsity,
                chart_id="glasso_precision_heatmap",
                filename_base="glasso_precision_sparsity",
                title_key="glasso_prec_hm",
                default_title=f"Precision matrix sparsity ({int(n_offdiag)} edges, {density:.1f}% density)",
                ordered_tickers=order,
                zmin=0.0, zmax=1.0, diverging=False,
                height=520, hover_label="|Θ| > 1e-3",
                colorbar_tickvals=(0.0, 1.0),
                colorbar_ticktext=("zero", "non-zero"),
            )
        else:
            st.info("Run the pipeline to generate the precision matrix.")

        # Row 6: precision-sparsity timeline. Reads precomputed snapshots from
        # data/<market>/results/glasso_alpha_path.npz (BIST only) and lets the
        # user scrub through α thresholds at which the sparsity pattern
        # changes. Slider is independent of the α used for the rendering
        # above — this is a separate, view-only path so it doesn't trigger
        # any refit.
        path_data = load_glasso_alpha_path()
        if path_data is not None and len(path_data["alphas"]) > 1:
            st.markdown("---")
            st.markdown(
                "**Precision Sparsity Timeline** — scrub α to watch direct "
                "connections drop out as the L1 penalty grows. Each step is a "
                "precomputed snapshot from `run_pipeline.py`; sliding is "
                "instant (no recompute)."
            )

            alphas_path = path_data["alphas"]
            sparsity_pcts_path = path_data["sparsity_pct"]
            n_edges_path = path_data["n_edges"]
            patterns_path = path_data["patterns"]
            path_tickers = path_data["tickers"]
            n_snapshots = len(alphas_path)

            selected_alpha = st.select_slider(
                f"α (drag to scrub through {n_snapshots} snapshots)",
                options=[float(a) for a in alphas_path],
                value=float(alphas_path[0]),
                format_func=lambda x: f"{x:.6f}",
                key="glasso_path_slider",
            )
            sel_idx = int(np.argmin(np.abs(alphas_path - selected_alpha)))

            cur_alpha = float(alphas_path[sel_idx])
            cur_sparsity = float(sparsity_pcts_path[sel_idx])
            cur_edges = int(n_edges_path[sel_idx])

            # Delta: which tickers just became isolated at this step?
            # Isolation = row-sum (degree) dropped from >0 to 0 since the
            # previous snapshot (which corresponds to a smaller α).
            if sel_idx > 0:
                prev_pattern = patterns_path[sel_idx - 1]
                cur_pattern = patterns_path[sel_idx]
                prev_degree = prev_pattern.sum(axis=1)
                cur_degree = cur_pattern.sum(axis=1)
                newly_isolated = [
                    path_tickers[i]
                    for i in range(len(path_tickers))
                    if prev_degree[i] > 0 and cur_degree[i] == 0
                ]
                if newly_isolated:
                    delta_text = (
                        f"**{len(newly_isolated)} ticker(s)** newly isolated "
                        f"at this step: {', '.join(newly_isolated)}."
                    )
                else:
                    delta_text = (
                        "No tickers were fully isolated at this step "
                        "(some edges still pruned, but every node keeps "
                        "at least one neighbour)."
                    )
            else:
                delta_text = (
                    "First snapshot (smallest α with a unique sparsity "
                    "pattern). All subsequent snapshots are progressively "
                    "sparser."
                )

            m1, m2, m3 = st.columns(3)
            m1.metric("α (this snapshot)", f"{cur_alpha:.6f}")
            m2.metric("Sparsity", f"{cur_sparsity:.1f}%")
            m3.metric("Direct edges", cur_edges)
            st.caption(delta_text)

            # Render the snapshot's binary sparsity pattern. Reuse the
            # dendrogram order loaded above so the timeline lines up
            # visually with the static precision sparsity heatmap.
            pattern_df = pd.DataFrame(
                patterns_path[sel_idx].astype(float),
                index=path_tickers,
                columns=path_tickers,
            )
            render_matrix_heatmap(
                pattern_df,
                chart_id="glasso_path_heatmap",
                filename_base="glasso_path_sparsity",
                title_key="glasso_path_hm",
                default_title=(
                    f"Precision sparsity at α = {cur_alpha:.6f} "
                    f"({cur_edges} edges, {cur_sparsity:.1f}% sparsity)"
                ),
                ordered_tickers=order,
                zmin=0.0, zmax=1.0, diverging=False,
                height=520, hover_label=f"|Θ| > 1e-3 at α={cur_alpha:.4f}",
                colorbar_tickvals=(0.0, 1.0),
                colorbar_ticktext=("zero", "non-zero"),
            )
        elif current_universe() == "bist":
            st.markdown("---")
            st.info(
                "**Precision Sparsity Timeline** not available — "
                "`glasso_alpha_path.npz` is missing. Rerun "
                "`uv run python run_pipeline.py` to generate it "
                "(adds ~3-5 min, BIST only)."
            )


@st.fragment
def _render_wavelet_for_scale(
    sector_map: dict, scales: dict, sector_node_label: str,
) -> None:
    """Scale-selector + per-scale MST render for the Wavelet section.

    Owns the `wavelet_scale` selectbox. Wrapped in @st.fragment so changing
    the scale only re-runs this block — not the entire EEE Analysis tab.
    The cross-scale summary table downstream stays untouched.

    Per-scale `title_key=f"wav_mst_{scale_level}"` ensures user-typed custom
    titles don't bleed across scale switches (the original `title_key="wav_mst"`
    was shared, so a custom title set on Scale 3 would persist when switching
    to Scale 5).
    """
    n_scales = len(scales)
    if n_scales == 0:
        st.info("No wavelet scales available.")
        return

    # Selectbox with physical-interpretation labels instead of bare "Scale 4".
    scale_options = [
        f"Scale {n} — {scales.get(str(n), '')}" for n in range(1, n_scales + 1)
    ]
    _default_idx = min(3, n_scales - 1)  # mirrors old slider's value=4
    scale_choice = st.selectbox(
        "Wavelet Scale", scale_options, index=_default_idx, key="wavelet_scale",
    )
    scale_level = int(scale_choice.split(" ")[1])
    scale_label = scales.get(str(scale_level), f"Scale {scale_level}")

    edges = load_wavelet_mst_edges(scale_level)
    scale_metrics = load_wavelet_mst_metrics(scale_level)
    if edges.empty:
        st.info("No MST data for this scale.")
        return

    fig = _plot_network(
        edges, sector_map, edge_weight_col="distance",
        node_metrics=scale_metrics,
        sector_node_label=sector_node_label,
        layout_source=f"wavelet_mst_scale{scale_level}",
    )
    total_weight = edges["distance"].sum()
    render_chart(
        fig,
        chart_id=f"wav_mst_{scale_level}",
        filename_base="wavelet_mst",
        title_key=f"wav_mst_{scale_level}",
        default_title=(
            f"MST at {scale_label} "
            f"(Σdistance: {total_weight:.1f}, nodes sized by betweenness)"
        ),
    )
    # Sprint 2 PR-I: subtitle clarifies this MST is computed on a single
    # wavelet frequency band only, contrasting with the full-period MSTs
    # elsewhere in the app.
    st.caption(
        f"Built on DWT detail coefficients at scale {scale_level} "
        f"({scale_label}) — isolates co-movement at this frequency band only. "
        "Contrasts with the full-period MST in **Clustering & Network**."
    )


@st.fragment
def render_wavelets(sector_map: dict, *, u=None):
    """Render Wavelet multi-scale analysis section."""
    with st.container(border=True):
        _domain_wav = getattr(u, "domain", "finance") if u is not None else "finance"
        _series_wav = (
            getattr(u, "series_label", "returns").lower() + "s"
            if u is not None and getattr(u, "domain", "finance") != "finance"
            else "returns"
        )
        if _domain_wav == "finance":
            _wav_desc = (
                "DWT (Daubechies-4) decomposes returns into frequency bands. "
                "Each scale isolates a specific frequency — unlike rolling windows which mix all frequencies. "
                "Short scales capture noise/day-trading; long scales reveal institutional/macro structure."
            )
        else:
            _wav_desc = (
                f"DWT (Daubechies-4) decomposes the {_series_wav} into frequency bands. "
                "Each scale isolates a specific oscillation range — short scales capture "
                "fast rhythms (gamma/beta in EEG), long scales reveal slow oscillations "
                "(theta/delta)."
            )
        section_header("Wavelet Multi-Scale Correlation", _wav_desc)

        meta = load_wavelet_metadata()
        if not meta:
            st.info("Run the pipeline to generate wavelet results.")
            return

        scales = meta.get("scales", {})

        # Per-scale MST render lives in `_render_wavelet_for_scale` fragment
        # (defined at module top). Changing the scale selectbox only re-runs
        # that fragment, not the whole EEE Analysis tab. The per-scale corr
        # distribution histogram that used to live in `col_corr` is removed —
        # the full-period distribution on the Pairs & Dislocations tab
        # already covers that view.
        _render_wavelet_for_scale(sector_map, scales, _sector_label(u))

        # Scale comparison summary — wrapped in expander (collapsed by default)
        # so the per-scale MST above is the primary view. Grader / cross-scale
        # comparison can pop the expander to see all 7 rows side-by-side.
        # Build cost is the same either way (st.expander only hides children,
        # doesn't skip execution); this is pure visual decluttering.
        with st.expander("Cross-scale comparison (all 7 scales)", expanded=False):
            summary_rows = []
            for lvl_str, label in scales.items():
                lvl = int(lvl_str)
                c = load_wavelet_corr(lvl)
                if not c.empty:
                    m = np.triu(np.ones(c.shape, dtype=bool), k=1)
                    vals = c.values[m]
                    vals = vals[np.isfinite(vals)]
                    e = load_wavelet_mst_edges(lvl)
                    metrics = load_wavelet_mst_metrics(lvl)
                    row = {
                        "Scale": lvl,
                        "Label": label,
                        "Avg Correlation": round(float(np.mean(vals)), 4),
                        "Std Correlation": round(float(np.std(vals)), 4),
                        "MST Total Weight": round(float(e["distance"].sum()), 1) if not e.empty else None,
                        "MST Edges": int(len(e)) if not e.empty else None,
                    }
                    if not metrics.empty and "betweenness_centrality" in metrics.columns:
                        bc = metrics["betweenness_centrality"].astype(float)
                        deg = metrics["degree"].astype(float) if "degree" in metrics.columns else pd.Series(dtype=float)
                        row["Max Betweenness"] = round(float(bc.max()), 4)
                        row["Avg Degree"] = round(float(deg.mean()), 2) if not deg.empty else None
                    summary_rows.append(row)
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No wavelet scale results found.")


@st.fragment
def render_transfer_entropy(sector_map: dict, *, u=None):
    """Render Transfer Entropy section."""
    with st.container(border=True):
        _il = _item_label(u)
        _items_lower = _items_label(u).lower()
        section_header(
            "Transfer Entropy — Directed Information Flow",
            "Unlike correlation (symmetric), transfer entropy measures directed causality: "
            f"'Does {_il} A's past reduce uncertainty about {_il} B's future?' "
            f"Produces an asymmetric network revealing which {_items_lower} lead and which follow.",
        )

        roles = load_te_node_roles()
        edges = load_te_edges()
        raw_te = load_te_matrix_raw()

        # If the FDR-corrected edge list is empty (the typical case on
        # large N at 100 shuffles), rank pairs by raw TE magnitude
        # instead. The displayed network is the strongest information-
        # flow edges; recompute node roles from the raw matrix so the
        # sources / sinks counts are meaningful even without FDR.
        if (edges.empty or len(edges) == 0) and not raw_te.empty:
            edges = _edges_from_raw_te(raw_te, top_k=200)
            roles = _node_roles_from_raw_te(raw_te)
            # Annotate roles with sector so downstream display works.
            roles["sector"] = roles["ticker"].map(sector_map).fillna("")

        if (roles is None or roles.empty) and (raw_te is None or raw_te.empty):
            st.info("Run the pipeline to generate transfer entropy results.")
            return

        n_sources = (roles["role"] == "source").sum() if not roles.empty else 0
        n_sinks = (roles["role"] == "sink").sum() if not roles.empty else 0

        # PR #71: surface the surrogate-null + FDR statistics that the pipeline
        # produces but the page previously ignored. Honest framing of the
        # statistical bar lets the user understand WHY the magnitude-ranked
        # network below may have edges that don't all survive FDR.
        te_summary = load_te_summary()
        n_sig_fdr = int(te_summary.get("n_significant_fdr", 0))
        n_sig_unc = int(te_summary.get("n_significant_uncorrected", 0))
        total_pairs = int(te_summary.get("total_pairs", 0))
        shuffles = int(te_summary.get("significance_shuffles", 0))
        correction = te_summary.get("multiple_testing", "fdr_bh")
        alpha = float(te_summary.get("significance_level", 0.05))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Information Sources", n_sources)
        c2.metric("Information Sinks", n_sinks)
        c3.metric(
            "Magnitude-ranked edges shown",
            len(edges),
            help=(
                "Strongest pair-wise transfer-entropy values across the "
                "directed network. Always shown so the network structure "
                "is visible even when no edges survive FDR."
            ),
        )
        if total_pairs:
            c4.metric(
                f"Significant at FDR α={alpha:g}",
                f"{n_sig_fdr} / {total_pairs}",
                help=(
                    f"From {shuffles} surrogate shuffles + {correction} "
                    f"multiple-testing correction. "
                    f"{n_sig_unc} pairs pass at uncorrected α={alpha:g} — "
                    f"of those, {n_sig_fdr} survive correction across "
                    f"{total_pairs} simultaneous tests."
                ),
            )
        else:
            c4.metric("FDR result", "n/a")

        col_net, col_table = st.columns([3, 2])

        with col_net:
            if HAS_NETWORKX and not edges.empty:
                # Build directed network from top edges
                top_n = st.slider("Show top N edges", 20, min(200, len(edges)), 50, key="te_top_n")
                top_edges = edges.head(top_n)

                G = nx.DiGraph()
                for _, r in top_edges.iterrows():
                    net_te = float(r["net_te"])
                    if net_te > 0:
                        G.add_edge(r["source"], r["target"], weight=abs(net_te))
                    else:
                        G.add_edge(r["target"], r["source"], weight=abs(net_te))

                if G.number_of_nodes() > 0:
                    pos = nx.kamada_kawai_layout(G)

                    fig = go.Figure()

                    # Edges with arrows
                    for u, v, d in G.edges(data=True):
                        x0, y0 = pos[u]
                        x1, y1 = pos[v]
                        fig.add_annotation(
                            x=x1, y=y1, ax=x0, ay=y0,
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True,
                            arrowhead=2, arrowsize=1.2,
                            arrowwidth=max(0.5, d["weight"] * 300),
                            arrowcolor="rgba(67,97,238,0.4)",
                        )

                    # Nodes
                    role_colors = {"source": "#E63946", "sink": "#4361EE"}
                    role_map = dict(zip(roles["ticker"], roles["role"]))

                    for node in G.nodes():
                        x, y = pos[node]
                        role = role_map.get(node, "sink")
                        color = role_colors.get(role, "#999")
                        deg = G.degree(node)
                        sector = sector_map.get(node, "")
                        fig.add_trace(go.Scatter(
                            x=[x], y=[y], mode="markers+text",
                            marker=dict(size=8 + deg, color=color,
                                        line=dict(width=0.5, color="#fff")),
                            text=node, textposition="top center", textfont=dict(size=7),
                            hovertemplate=(
                                f"<b>{node}</b> ({sector})<br>"
                                f"Role: {role}<br>Connections: {deg}<extra></extra>"
                            ),
                            showlegend=False,
                        ))

                    # Legend for roles
                    for role, color in role_colors.items():
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None], mode="markers",
                            marker=dict(size=8, color=color),
                            name=f"Info {role.title()}",
                        ))

                    apply_chart_style(fig, height=600,
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    render_chart(fig, chart_id="te_network", filename_base="te_network",
                                 title_key="te_net",
                                 default_title="Information Flow Network (arrows = direction)")

        with col_table:
            _sec = _sector_label(u)
            _item = getattr(u, "item_label", "Ticker") if u is not None else "Ticker"
            _te_cols = {
                "ticker":      st.column_config.TextColumn(_item),
                "sector":      st.column_config.TextColumn(_sec),
                "net_te_flow": st.column_config.NumberColumn("Net TE Flow", format="%.4f"),
                "te_out":      st.column_config.NumberColumn("TE Out", format="%.4f"),
                "te_in":       st.column_config.NumberColumn("TE In", format="%.4f"),
            }

            st.markdown("**Top Information Sources (Leaders)**")
            sources = roles[roles["role"] == "source"].head(15)
            st.dataframe(
                sources[["ticker", "sector", "net_te_flow", "te_out", "te_in"]],
                use_container_width=True, hide_index=True,
                column_config=_te_cols,
            )

            st.markdown("**Top Information Sinks (Followers)**")
            sinks = roles[roles["role"] == "sink"].tail(15).sort_values("net_te_flow")
            st.dataframe(
                sinks[["ticker", "sector", "net_te_flow", "te_out", "te_in"]],
                use_container_width=True, hide_index=True,
                column_config=_te_cols,
            )

        # Net-TE flow heatmap (full width)
        # Built from the RAW TE matrix (raw − rawᵀ) rather than the
        # FDR-filtered matrix. On the full 5256-pair grid at K=1000, 0
        # pairs survive FDR — so the filtered net matrix is all zeros
        # and the heatmap looks dead. The raw net carries the actual
        # directional structure (red = i leads j; blue = i lags j) at
        # full magnitude resolution. Significance is reported separately
        # in the FDR-survivor table above + on each individual edge in
        # hover.
        st.markdown("---")
        st.markdown(
            "**Net Information Flow Heatmap** — `net[i,j] = TE(i→j) − TE(j→i)` "
            "from the raw TE matrix. Red cells in row `i` mean information flows "
            "from `i` to `j` (i leads j); blue cells mean `i` lags `j`. Significance "
            "of individual edges is reported in the FDR-survivor table above."
        )
        if not raw_te.empty:
            raw_arr = raw_te.to_numpy()
            net_arr = raw_arr - raw_arr.T
            np.fill_diagonal(net_arr, 0.0)
            net_raw = pd.DataFrame(net_arr, index=raw_te.index, columns=raw_te.columns)
            order = load_dendrogram_order()
            v = float(np.nanmax(np.abs(net_arr)))
            v = max(v, 1e-6)
            render_matrix_heatmap(
                net_raw,
                chart_id="te_net_heatmap",
                filename_base="te_net_flow_heatmap",
                title_key="te_net_hm",
                default_title="Net transfer-entropy flow (red = source, blue = sink)",
                ordered_tickers=order,
                zmin=-v, zmax=v, diverging=True,
                height=520, hover_label="net TE",
            )
        else:
            st.info("Run the pipeline to generate the transfer-entropy matrices.")

        # ---- FDR-significant directed edges (uses pvals + sig_mask) -----
        pvals = load_te_pvalues()
        sig_mask = load_te_significance()
        if not pvals.empty and not sig_mask.empty and not raw_te.empty:
            sig_arr = sig_mask.to_numpy()
            te_arr = raw_te.to_numpy()
            tickers = list(sig_mask.columns)
            sig_rows: list[dict] = []
            for i in range(sig_arr.shape[0]):
                for j in range(sig_arr.shape[1]):
                    if i == j or not bool(sig_arr[i, j]):
                        continue
                    sig_rows.append({
                        "source": tickers[i],
                        "target": tickers[j],
                        "te (nats)": float(te_arr[i, j]),
                        "p-value": float(pvals.iloc[i, j]),
                    })
            st.markdown("---")
            if sig_rows:
                sig_df = pd.DataFrame(sig_rows).sort_values("te (nats)", ascending=False)
                sig_df["te (nats)"] = sig_df["te (nats)"].round(5)
                sig_df["p-value"] = sig_df["p-value"].round(4)
                st.markdown(
                    f"**Directed flows surviving {correction} at α = {alpha:g}** "
                    f"({len(sig_df)} edges / {total_pairs} directed pairs tested)"
                )
                st.dataframe(sig_df, use_container_width=True, hide_index=True)
            elif total_pairs:
                st.caption(
                    f"No directed pairs survive {correction} at α = {alpha:g} on the "
                    f"full {total_pairs}-pair grid with K = {shuffles} surrogate shuffles "
                    f"(BH cutoff ~9.5e-6 for the top edge; K=1000 caps min p at 1e-3). "
                    f"The network plot above ranks by raw TE magnitude. Real edges surface "
                    f"on smaller hypothesis sets — see the lag-1 panel below."
                )

        # ---- PR #73 G3: rolling TE chart --------------------------------
        rolling_te = load_rolling_te()
        if not rolling_te.empty:
            st.markdown("---")
            st.markdown(
                "**Rolling transfer entropy on G1 survivors** — 252-day "
                "window, 21-day stride. Shows how directional flow strengthens "
                "and weakens over time on the pairs that passed the full-sample "
                "FDR test."
            )
            colors = get_colors()
            palette = SECTOR_PALETTE[:10]
            # Build a "directional pair" label so the legend reads naturally
            rolling_te_disp = rolling_te.copy()
            rolling_te_disp["date"] = pd.to_datetime(rolling_te_disp["date"])
            rolling_te_disp["pair_dir"] = rolling_te_disp.apply(
                lambda r: (
                    f"{r['ticker_a']} → {r['ticker_b']}"
                    if r["direction"] == "a_to_b"
                    else f"{r['ticker_b']} → {r['ticker_a']}"
                ),
                axis=1,
            )

            fig_rt = go.Figure()
            for i, pair_dir in enumerate(sorted(rolling_te_disp["pair_dir"].unique())):
                sub = rolling_te_disp[rolling_te_disp["pair_dir"] == pair_dir].sort_values("date")
                fig_rt.add_trace(go.Scatter(
                    x=sub["date"], y=sub["te"],
                    mode="lines+markers",
                    line=dict(color=palette[i % len(palette)], width=1.6),
                    marker=dict(
                        size=6,
                        # Hollow markers when p>0.05; filled when sig
                        symbol=["circle" if p < 0.05 else "circle-open" for p in sub["p_value"]],
                    ),
                    name=pair_dir,
                    hovertemplate=(
                        "<b>" + pair_dir + "</b><br>"
                        "%{x|%Y-%m-%d}<br>TE = %{y:.5f}<br>"
                        "p = %{customdata:.4f}<extra></extra>"
                    ),
                    customdata=sub["p_value"],
                ))
            # Crisis verticals matching rolling D_eff panel
            for date_iso, label in [
                ("2022-02-24", "Ukraine"),
                ("2023-02-06", "Earthquake"),
            ]:
                date = pd.Timestamp(date_iso)
                fig_rt.add_shape(
                    type="line", x0=date, x1=date, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"),
                )
                fig_rt.add_annotation(
                    x=date, y=1.02, xref="x", yref="paper",
                    text=label, showarrow=False,
                    font=dict(size=10),
                )
            apply_chart_style(
                fig_rt, height=380,
                xaxis_title="Window-end date",
                yaxis_title="TE (nats)",
            )
            render_chart(
                fig_rt, chart_id="te_rolling_pairs",
                filename_base="rolling_te_per_pair",
                default_title="Rolling TE per pair (filled marker = p<0.05 in that window)",
            )

        # ---- TE confidence intervals on the FDR-survivor pairs ----------
        te_ci = load_te_with_ci()
        if not te_ci.empty:
            st.markdown("---")
            st.markdown(
                "**Joint-bootstrap 95% CIs on TE point estimates** "
                "(K=500 circular-block bootstrap, pair structure preserved)"
            )
            te_ci_disp = te_ci.copy()
            te_ci_disp["pair_dir"] = te_ci_disp.apply(
                lambda r: (
                    f"{r['ticker_a']} → {r['ticker_b']}"
                    if r["direction"] == "a_to_b"
                    else f"{r['ticker_b']} → {r['ticker_a']}"
                ),
                axis=1,
            )
            te_ci_disp["CI"] = te_ci_disp.apply(
                lambda r: f"[{r['te_ci_low']:+.5f}, {r['te_ci_high']:+.5f}]", axis=1,
            )
            te_ci_disp["robust?"] = te_ci_disp["includes_zero"].map({True: "marginal", False: "yes"})
            st.dataframe(
                te_ci_disp[["pair_dir", "te_point", "CI", "robust?"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "pair_dir": st.column_config.TextColumn("Directional pair"),
                    "te_point": st.column_config.NumberColumn("TE point (nats)", format="%.5f"),
                    "CI": st.column_config.TextColumn("95% CI (nats)"),
                    "robust?": st.column_config.TextColumn("CI excludes 0?"),
                },
            )

        # ---- Conditional TE: directed flow after market-factor removal ---
        # Tests whether the surviving directed flows are market-mediated
        # artifacts. CTE(X→Y | XU100) > TE(X→Y) means the flow is genuinely
        # pair-specific and not just both following the market.
        cte_df = load_te_conditional_market()
        if not cte_df.empty:
            st.markdown("---")
            st.markdown(
                "**Are these flows market-mediated?** TE compared with "
                "conditional TE that controls for the BIST market index."
            )
            cte_disp = cte_df.copy()
            cte_disp["pair_dir"] = cte_disp["source"] + " → " + cte_disp["target"]
            for col in ("te", "cte", "delta"):
                cte_disp[col] = cte_disp[col].round(5)
            cte_disp["p_value"] = cte_disp["p_value"].round(4)
            cte_disp["verdict"] = cte_disp["delta"].apply(
                lambda d: "pair-specific" if d > 0 else "market-mediated"
            )
            st.dataframe(
                cte_disp[["pair_dir", "te", "cte", "delta", "p_value", "verdict"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "pair_dir": st.column_config.TextColumn("Directional pair"),
                    "te": st.column_config.NumberColumn("TE (nats)", format="%.5f"),
                    "cte": st.column_config.NumberColumn("CTE | XU100", format="%.5f"),
                    "delta": st.column_config.NumberColumn("Δ = CTE − TE", format="%+.5f"),
                    "p_value": st.column_config.NumberColumn(
                        "p-value", format="%.4f",
                        help="Surrogate-null p-value of the conditional TE. "
                             "4-way joint estimator has more bias than 3-way "
                             "TE; treat magnitudes as ordinal vs TE.",
                    ),
                    "verdict": st.column_config.TextColumn("Verdict"),
                },
            )

        # ---- Sector-aggregated TE: lead-lag at sector resolution ---------
        # 13 sectors → 156 directed pairs (vs 5256 ticker-level). FDR
        # cutoff ~30× more forgiving; uncorrected significance at K=1000
        # surfaces the underlying sectoral lead-lag structure.
        sec_te = load_te_sector_matrix()
        if not sec_te.empty:
            st.markdown("---")
            sec_n_fdr = int(sec_te["significant_fdr"].sum())
            sec_n_unc = int(sec_te["significant_uncorrected"].sum())
            st.markdown(
                "**Sector-aggregated TE** — equal-weight sector portfolios, "
                f"{len(sec_te)} directed sector pairs. "
                f"{sec_n_fdr} FDR-survivors / {sec_n_unc} uncorrected at α = 0.05."
            )
            sectors = sorted(set(sec_te["source"]) | set(sec_te["target"]))
            mat = pd.DataFrame(0.0, index=sectors, columns=sectors)
            for _, r in sec_te.iterrows():
                mat.loc[r["source"], r["target"]] = float(r["te"])
            vmax = float(np.nanmax(np.abs(mat.to_numpy()))) if mat.size else 1.0
            render_matrix_heatmap(
                mat,
                chart_id="te_sector_heatmap",
                filename_base="te_sector_flow_heatmap",
                title_key="te_sector_hm",
                default_title="Sector → sector transfer entropy (nats)",
                zmin=0.0, zmax=max(vmax, 1e-6),
                diverging=False, height=440,
                hover_label="TE",
            )
            st.markdown("**Top 15 uncorrected-significant edges**")
            top_sec = (
                sec_te[sec_te["significant_uncorrected"]]
                .sort_values("te", ascending=False).head(15).copy()
            )
            if top_sec.empty:
                st.caption("No uncorrected-significant sector edges.")
            else:
                top_sec["te"] = top_sec["te"].round(5)
                top_sec["p_value"] = top_sec["p_value"].round(4)
                top_sec["fdr"] = top_sec["significant_fdr"].map({True: "yes", False: ""})
                st.dataframe(
                    top_sec[["source", "target", "te", "p_value", "fdr"]],
                    use_container_width=True, hide_index=True,
                    column_config={
                        "source": st.column_config.TextColumn("Source sector"),
                        "target": st.column_config.TextColumn("Target sector"),
                        "te": st.column_config.NumberColumn("TE (nats)", format="%.5f"),
                        "p_value": st.column_config.NumberColumn("p-value", format="%.4f"),
                        "fdr": st.column_config.TextColumn("FDR"),
                    },
                )


# ---------------------------------------------------------------------------
# SNN — Spiking Neural Network (pair-signal classifier)
# ---------------------------------------------------------------------------

CLASS_COLORS = {"HOLD": "#9CA3AF", "BUY": "#06D6A0", "SELL": "#E63946"}


@st.fragment
def render_snn(sector_map: dict, *, u=None):
    """Render Spiking Neural Network (neuromorphic) section.

    Honest framing: the SNN achieves macro-F1 ≈ 0.66 (3-class baseline 0.27)
    but on the trading-Sharpe metric it underperforms the simple |Z|>2 rule
    on both markets: BIST Δ-Sharpe = −0.27 (wins 10 of 20 pairs); S&P
    Δ-Sharpe = −0.84 (wins 7 of 20). We report this as a documented exploration
    of spike-coded neural inference applied to pair-spread classification —
    complementary to the rate-coded methods elsewhere in the project.
    """
    with st.container(border=True):
        section_header(
            "Neuromorphic Inference — Spiking Neural Network classifier",
            "A recurrent leaky-integrate-and-fire classifier trained with "
            "surrogate-gradient backprop-through-time on the pair-dislocation "
            "Z-score. Inputs are delta-modulated (Σ-Δ ADC analogue) spike "
            "trains; outputs are 3-class BUY / SELL / HOLD decisions. The same "
            "algorithmic substrate runs on Intel Loihi 2, IBM TrueNorth, and "
            "SpiNNaker neuromorphic processors.",
        )

        metrics = load_snn_metrics()
        pair_list = load_snn_pair_list()

        if not metrics or pair_list.empty:
            st.info(
                "Run the SNN pipeline to generate results: "
                "`uv sync --extra snn && uv run python run_pipeline.py`."
            )
            return

        agg = metrics.get("aggregate", {})
        cfg = metrics.get("config", {})
        sample_pair = metrics.get("sample_pair", "BRYAT_BRSAN")

        # ---- Headline metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Pairs trained", metrics.get("n_pairs", len(pair_list)))
        c2.metric("Mean macro-F1", f"{agg.get('mean_macro_f1', 0):.3f}",
                  help="Classification quality. Random baseline ≈ 0.33; majority-class baseline ≈ 0.27.")
        c3.metric("Mean SNN Sharpe", f"{agg.get('mean_snn_sharpe', 0):+.2f}",
                  help="Annualised Sharpe on the SNN's BUY/SELL signals.")
        c4.metric("Mean Classical Sharpe", f"{agg.get('mean_classical_sharpe', 0):+.2f}",
                  help="Same metric on the simple |Z|>2 rule.")
        delta_sh = agg.get("mean_delta_sharpe", 0)
        c5.metric("Mean Δ-Sharpe", f"{delta_sh:+.2f}",
                  delta=f"{delta_sh:+.2f}",
                  delta_color="normal",
                  help="SNN − Classical. Positive = SNN wins; negative = SNN loses.")

        per_pair = metrics.get("per_pair", {})
        n_beats = sum(1 for v in per_pair.values() if v.get("delta_sharpe", 0) > 0)
        top_wins = sorted(
            ((pid, v.get("delta_sharpe", 0)) for pid, v in per_pair.items()),
            key=lambda kv: kv[1], reverse=True,
        )[:3]
        wins_text = ", ".join(f"`{p}` (+{d:.2f})" for p, d in top_wins if d > 0)
        st.caption(
            f"Macro-F1 = **{agg.get('mean_macro_f1', 0):.2f}** "
            f"(random baseline 0.33; majority-class 0.27) confirms the "
            f"dislocation features carry learnable structure. The SNN beats "
            f"the classical `|Z|>2` rule on **{n_beats} of {len(per_pair)}** "
            f"pairs by Sharpe; strongest wins: {wins_text}."
        )

        with st.expander("Methodological context"):
            st.markdown(
                "The aggregate Δ-Sharpe is negative because the predictive "
                "information about 20-day-ahead mean reversion is largely "
                "concentrated in the current Z-score itself — adding neural "
                "machinery on top extracts limited additional signal at daily "
                "frequency, consistent with weak-form EMH on liquid equity "
                "markets. We retain the SNN for the methodological-breadth "
                "claim (spike-coded counterpart to the rate-coded methods "
                "elsewhere in the pipeline). See `docs/SNN_Report.md` §11.3 "
                "for the full per-pair breakdown."
            )

        # ---- Per-pair leaderboard
        st.markdown("**Per-pair leaderboard** (sorted by Δ-Sharpe vs |Z|>2 baseline)")
        rows = []
        for pid, p in per_pair.items():
            rows.append({
                "pair": pid,
                "macro_f1": p.get("macro_f1"),
                "snn_sharpe": p.get("snn_sharpe"),
                "classical_sharpe": p.get("classical_sharpe"),
                "delta_sharpe": p.get("delta_sharpe"),
                "snn_hit_rate": p.get("snn_hit_rate"),
                "snn_n_trades": p.get("snn_n_trades"),
                "classical_n_trades": p.get("classical_n_trades"),
                "n_test": p.get("n_test"),
            })
        leaderboard = (
            pd.DataFrame(rows)
            .sort_values("delta_sharpe", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            leaderboard,
            use_container_width=True,
            height=320,
            column_config={
                "pair":               st.column_config.TextColumn("Pair"),
                "macro_f1":           st.column_config.NumberColumn("F1", format="%.3f"),
                "snn_sharpe":         st.column_config.NumberColumn("SNN Sharpe", format="%+.2f"),
                "classical_sharpe":   st.column_config.NumberColumn("Classical Sharpe", format="%+.2f"),
                "delta_sharpe":       st.column_config.NumberColumn("Δ Sharpe", format="%+.2f"),
                "snn_hit_rate":       st.column_config.NumberColumn("Hit %", format="%.2f"),
                "snn_n_trades":       st.column_config.NumberColumn("SNN trades"),
                "classical_n_trades": st.column_config.NumberColumn("Classical trades"),
                "n_test":             st.column_config.NumberColumn("Test obs"),
            },
        )

        # ---- Per-pair signal explorer
        st.markdown("**Per-pair signal explorer**")
        pair_ids = pair_list["pair_id"].astype(str).tolist() if "pair_id" in pair_list.columns else list(per_pair.keys())
        default_idx = pair_ids.index(sample_pair) if sample_pair in pair_ids else 0
        selected_pair = st.selectbox(
            "Select pair",
            pair_ids,
            index=default_idx,
            key="snn_pair_selector",
        )

        signals = load_snn_signals(selected_pair)
        if not signals.empty and {"date", "zscore", "signal"}.issubset(signals.columns):
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(
                x=signals["date"], y=signals["zscore"],
                name="Z-score", mode="lines",
                line=dict(color=get_colors().get("muted", "#888"), width=1.2),
            ))
            for cls in ("BUY", "SELL"):
                mask = signals["signal"] == cls
                if mask.any():
                    fig_sig.add_trace(go.Scatter(
                        x=signals.loc[mask, "date"],
                        y=signals.loc[mask, "zscore"],
                        name=f"SNN {cls}",
                        mode="markers",
                        marker=dict(
                            color=CLASS_COLORS[cls], size=8,
                            symbol="triangle-up" if cls == "BUY" else "triangle-down",
                            line=dict(width=0.5, color="#222"),
                        ),
                    ))
            fig_sig.add_hline(y=2.0, line=dict(color="rgba(150,150,150,0.4)", dash="dot"))
            fig_sig.add_hline(y=-2.0, line=dict(color="rgba(150,150,150,0.4)", dash="dot"))
            fig_sig.add_hline(y=0.0, line=dict(color="rgba(150,150,150,0.6)", dash="dash"))
            apply_chart_style(
                fig_sig, height=360,
                xaxis_title="Date", yaxis_title="Z-score (rolling 60d)",
            )
            render_chart(
                fig_sig, chart_id=f"snn_signal_{selected_pair}",
                filename_base=f"snn_signal_{selected_pair}",
                default_title=f"SNN signals on Z-score — {selected_pair}",
            )
        else:
            st.info("No signals for this pair.")

        # ---- Training history
        st.markdown("**Training history** (universal model, pooled across all pairs)")
        history = load_snn_training_history()
        if not history.empty and "epoch" in history.columns:
            fig_h = go.Figure()
            if "train_loss" in history.columns:
                fig_h.add_trace(go.Scatter(
                    x=history["epoch"], y=history["train_loss"],
                    name="Train loss", mode="lines+markers",
                    line=dict(color=get_colors().get("primary", "#3B82F6")),
                ))
            if "val_loss" in history.columns:
                fig_h.add_trace(go.Scatter(
                    x=history["epoch"], y=history["val_loss"],
                    name="Val loss", mode="lines+markers",
                    line=dict(color=get_colors().get("secondary", "#EF4444"), dash="dash"),
                ))
            if "val_macro_f1" in history.columns:
                fig_h.add_trace(go.Scatter(
                    x=history["epoch"], y=history["val_macro_f1"],
                    name="Val macro-F1", mode="lines+markers", yaxis="y2",
                    line=dict(color="#06D6A0"),
                ))
            fig_h.update_layout(
                yaxis=dict(title="Loss"),
                yaxis2=dict(title="Val macro-F1", overlaying="y", side="right", range=[0, 1]),
            )
            apply_chart_style(fig_h, height=300, xaxis_title="Epoch")
            render_chart(
                fig_h, chart_id="snn_training_history",
                filename_base="snn_training_history",
                default_title="SNN training convergence (early-stopped on val loss)",
            )

        # ---- Spike raster + membrane V(t) for the sample pair only
        st.markdown(
            f"**Sample pair internals — `{sample_pair}`** "
            "(spike output + membrane-potential readout)"
        )
        raster = load_snn_raster_sample()
        membrane = load_snn_membrane_sample()

        col_r, col_m = st.columns(2)

        with col_r:
            if not raster.empty and {"day_index", "timestep", "neuron_name"}.issubset(raster.columns):
                # Compose a continuous "tick" axis = day_index * n_timesteps + timestep
                ticks_per_day = int(cfg.get("n_timesteps", 20))
                raster = raster.assign(
                    global_tick=raster["day_index"].astype(int) * ticks_per_day + raster["timestep"].astype(int)
                )
                fig_r = go.Figure()
                for cls, c in CLASS_COLORS.items():
                    mask = raster["neuron_name"] == cls
                    if mask.any():
                        fig_r.add_trace(go.Scatter(
                            x=raster.loc[mask, "global_tick"],
                            y=[cls] * mask.sum(),
                            mode="markers",
                            name=cls,
                            marker=dict(color=c, size=6, symbol="line-ns-open"),
                        ))
                apply_chart_style(fig_r, height=240, xaxis_title="SNN tick (day × 20 timesteps + t)")
                render_chart(
                    fig_r, chart_id="snn_raster",
                    filename_base="snn_raster",
                    default_title=f"Output-neuron spike raster — {sample_pair} window",
                )
            else:
                st.info("Spike raster not available for the sample pair.")

        with col_m:
            if not membrane.empty and {"day_index", "timestep", "neuron_name", "membrane"}.issubset(membrane.columns):
                ticks_per_day = int(cfg.get("n_timesteps", 20))
                membrane = membrane.assign(
                    global_tick=membrane["day_index"].astype(int) * ticks_per_day + membrane["timestep"].astype(int)
                )
                fig_m = go.Figure()
                for cls, c in CLASS_COLORS.items():
                    sub = membrane[membrane["neuron_name"] == cls].sort_values("global_tick")
                    if not sub.empty:
                        fig_m.add_trace(go.Scatter(
                            x=sub["global_tick"], y=sub["membrane"],
                            name=cls, mode="lines",
                            line=dict(color=c, width=1.6),
                        ))
                fig_m.add_hline(
                    y=float(cfg.get("v_threshold", 0.5)),
                    line=dict(color="rgba(120,120,120,0.5)", dash="dot"),
                    annotation_text="V_th",
                )
                apply_chart_style(fig_m, height=240, xaxis_title="SNN tick", yaxis_title="Membrane V(t)")
                render_chart(
                    fig_m, chart_id="snn_membrane",
                    filename_base="snn_membrane",
                    default_title=f"Output-layer membrane V(t) — {sample_pair} window",
                )
            else:
                st.info("Membrane trace not available for the sample pair.")

        # ---- Architecture / hyperparameter summary
        with st.expander("Architecture and hyperparameters"):
            st.markdown(
                "- **Hidden layer:** recurrent LIF (`snn.RLeaky`), "
                f"`n_hidden = {cfg.get('n_hidden', '?')}`, "
                f"β = {cfg.get('beta', '?')}, V_th = {cfg.get('v_threshold', '?')}\n"
                f"- **Output layer:** non-resetting LIF (membrane-potential readout)\n"
                f"- **Input encoders:** delta modulation (Z, ΔZ) + population coding "
                f"(`n_population_fields = {cfg.get('n_population_fields', '?')}`)\n"
                f"- **Window:** {cfg.get('window_size', '?')} trading days × "
                f"{cfg.get('n_timesteps', '?')} SNN ticks per day = "
                f"{int(cfg.get('window_size', 1)) * int(cfg.get('n_timesteps', 1))} unrolled timesteps\n"
                f"- **Universal model:** one network across all pairs with 20-dim "
                "one-hot pair embedding (vs per-pair training)\n"
                f"- **Loss:** focal loss (γ = {cfg.get('focal_gamma', '?')}) with "
                f"`sqrt(inv_freq)` class weights\n"
                f"- **Training:** Adam(lr={cfg.get('learning_rate', '?')}, "
                f"wd={cfg.get('weight_decay', '?')}); early-stop patience "
                f"{cfg.get('early_stop_patience', '?')}; seed {cfg.get('seed', '?')}\n"
                f"- **Total inputs to fc1:** {metrics.get('n_inputs', '?')} channels "
                "(45 spike channels + 20 pair one-hot)"
            )


# ---------------------------------------------------------------------------
# Information Theory — MI / D_eff / ΔH / regime KL
# ---------------------------------------------------------------------------

def _is_domain_finance(u) -> bool:
    return getattr(u, "domain", "finance") == "finance"


@st.fragment
def render_info_theory(sector_map: dict, *, u=None):
    """Render the Information-Theory sub-tab (Phase 3 mutable-candy).

    Composes four short panels: summary KPIs, MI vs Pearson, rolling D_eff(t),
    and regime KL. Honest, not philosophical: the TA asked for an information-
    theory perspective and this is it, framed as "what the system looks like
    in nats and bits," not "a unifying overhead theorem."
    """
    with st.container(border=True):
        section_header(
            "Information Theory — joint distribution in bits and nats",
            "Pairwise mutual information catches non-linear coupling Pearson "
            "misses; effective dimensionality (D_eff) measures how much of the "
            "joint distribution is informationally independent; KL divergence "
            "between Gaussian covariances quantifies regime change.",
        )

        summary = load_it_summary()
        if not summary:
            st.info(
                "Run the pipeline (or just `run_info_theory(config)`) to "
                "generate the information-theory artifacts."
            )
            return

        n_tickers = int(summary.get("n_tickers", 0))
        d_eff_val = float(summary.get("d_eff", float("nan")))
        dh_val = float(summary.get("log_det_term", float("nan")))
        sign_h = float(summary.get("mean_sign_entropy_rate_bits", float("nan")))

        # ---- Panel A: summary KPIs ----
        c1, c2, c3, c4 = st.columns(4)
        if np.isfinite(d_eff_val):
            c1.metric(
                "Effective dimensionality D_eff",
                f"{d_eff_val:.2f}",
                help=(
                    f"Participation ratio of the correlation eigenspectrum: "
                    f"(Σλ)² / Σλ². {n_tickers} tickers collapse into ≈ "
                    f"{d_eff_val:.1f} informationally independent dimensions."
                ),
            )
        if np.isfinite(dh_val):
            c2.metric(
                "Joint structure ΔH",
                f"{dh_val:.2f} nats",
                help=(
                    "−½ log det Σ — the Gaussian-joint-structure piece. "
                    "Larger means more redundant joint information; "
                    "0 means independent."
                ),
            )
        else:
            c2.metric(
                "Joint structure ΔH",
                "n/a",
                help=(
                    "ΔH = −½ log det Σ is undefined when the correlation "
                    "matrix is exactly singular. This is expected on EEG "
                    "after common-average referencing (one channel becomes "
                    "a linear combination of the others). Pipeline applies "
                    "a small ridge in this case; if you're still seeing n/a "
                    "your build predates the shrinkage fix."
                ),
            )
        if np.isfinite(sign_h):
            c3.metric(
                "Mean sign-entropy rate",
                f"{sign_h:.3f} bits/day",
                help=(
                    "H(sign_t | sign_{t-1}) averaged across tickers. "
                    "≈ 1 bit/day means tomorrow's direction is independent "
                    "of today's — the canonical weak-form-EMH fingerprint."
                ),
            )
        c4.metric("Tickers / channels", n_tickers)

        st.caption(
            ":material/info: All four panels below are derived from the "
            "`mi_matrix`, `rolling_info_theory`, `regime_kl` and "
            "`it_summary` artifacts under `data/<market>/results/`."
        )

        # ---- Panel C: rolling D_eff(t) + ΔH(t) with crisis markers ----
        rolling = load_rolling_info_theory()
        if not rolling.empty:
            st.markdown("**Rolling D_eff and ΔH over time** — joint-structure dynamics")
            crisis_specs = load_regime_kl() or []
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=rolling.index, y=rolling["d_eff"],
                name="D_eff (LHS)", mode="lines",
                line=dict(color="#4361EE", width=1.6),
                hovertemplate="%{x|%Y-%m-%d}<br>D_eff=%{y:.2f}<extra></extra>",
            ))
            fig_roll.add_trace(go.Scatter(
                x=rolling.index, y=rolling["log_det_term"],
                name="ΔH = −½ log det Σ (RHS)", mode="lines", yaxis="y2",
                line=dict(color="#E63946", width=1.4, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>ΔH=%{y:.2f}<extra></extra>",
            ))
            # add_vline + annotation_text breaks on pandas Timestamp x-values
            # in current plotly (it tries to mean two Timestamps). Add the
            # line shape and the label separately.
            for spec in crisis_specs:
                date = pd.Timestamp(spec["date"])
                fig_roll.add_shape(
                    type="line",
                    x0=date, x1=date, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dot"),
                )
                fig_roll.add_annotation(
                    x=date, y=1.02, xref="x", yref="paper",
                    text=spec["label"], showarrow=False,
                    font=dict(size=10), align="center",
                )
            apply_chart_style(
                fig_roll, height=380,
                xaxis_title="Date",
                yaxis_title="D_eff",
            )
            fig_roll.update_layout(
                yaxis2=dict(title="ΔH (nats)", overlaying="y", side="right"),
            )
            render_chart(
                fig_roll, chart_id="it_rolling",
                filename_base="it_rolling_d_eff_dh",
                title_key="it_rolling",
                default_title="Rolling D_eff and ΔH(t) with crisis markers",
            )

        # ---- Panel D: regime KL table ----
        regime = load_regime_kl()
        if regime:
            st.markdown(
                "**Regime KL divergence** — `D_KL(N(0, Σ_calm) ‖ N(0, Σ_crisis))` "
                "in nats, with Ledoit–Wolf shrinkage so high-dim singularity "
                "doesn't blow up the inverse-trace term."
            )
            tbl = pd.DataFrame([
                {
                    "Crisis": r["label"],
                    "Date": r["date"][:10],
                    "Calm window": f"{r['calm_start'][:10]} → {r['calm_end'][:10]}",
                    "Crisis window": f"{r['crisis_start'][:10]} → {r['crisis_end'][:10]}",
                    "KL (nats)": (
                        f"{r['kl']:.1f}" if np.isfinite(r.get('kl', float('nan')))
                        else "n/a"
                    ),
                    "Tickers": r.get("n_tickers", "—"),
                }
                for r in regime
            ])
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        # ---- Panel E: Per-ticker predictability diagnostics -------------
        # Three classic financial stylised facts that sign-entropy (lag-1,
        # 2-state) cannot see: volatility clustering, Hurst exponent,
        # raw return autocorrelation.
        predict_df = load_predictability_diagnostics()
        if not predict_df.empty:
            st.markdown("---")
            section_header(
                "Per-ticker predictability diagnostics",
                "Volatility clustering (ACF of |returns|), Hurst exponent, "
                "and raw return autocorrelation — three concrete measures "
                "of what's predictable about each ticker's return process.",
            )

            # Headline KPIs
            n_total = len(predict_df)
            n_vol_cluster = int((predict_df["acf_abs_returns_lag1"].dropna() > 0.20).sum())
            n_persistent = int((predict_df["hurst_exponent"].dropna() > 0.55).sum())
            n_random_walk = int(
                ((predict_df["hurst_exponent"] >= 0.45)
                 & (predict_df["hurst_exponent"] <= 0.55)).sum()
            )
            n_mean_revert = int((predict_df["hurst_exponent"].dropna() < 0.45).sum())
            bartlett = 1.96 / np.sqrt(int(load_it_summary().get("n_observations", 1) or 1))
            n_acf_r = int((predict_df["acf_returns_lag1"].abs() > bartlett).sum())

            _k1, _k2, _k3, _k4 = st.columns(4)
            _k1.metric(
                "Volatility clustering",
                f"{n_vol_cluster}/{n_total}",
                help=(
                    "Tickers with ACF(|returns|, lag=1) > 0.20. Tomorrow's "
                    "volatility is strongly predictable from today's even "
                    "when tomorrow's direction isn't."
                ),
            )
            _k2.metric(
                "Persistent (Hurst > 0.55)",
                f"{n_persistent}/{n_total}",
                help=(
                    "Tickers with rescaled-range Hurst exponent > 0.55 — "
                    "positive long-range memory (trending regime). Hurst ≈ "
                    "0.5 is the random-walk null."
                ),
            )
            _k3.metric(
                "Random walk (0.45 ≤ H ≤ 0.55)",
                f"{n_random_walk}/{n_total}",
                help="Tickers with no detectable long-range memory.",
            )
            _k4.metric(
                f"|ACF(r, 1)| > {bartlett:.3f}",
                f"{n_acf_r}/{n_total}",
                help=(
                    "Tickers whose raw lag-1 return autocorrelation exceeds "
                    "the Bartlett 95% threshold — direct return predictability."
                ),
            )

            # Build the comparison table — most-misleading tickers first
            disp = predict_df.copy()
            disp["deviation_sign_h"] = (1.0 - disp["sign_entropy_bits"]).round(4)
            disp["abs_acf_r"] = disp["acf_returns_lag1"].abs().round(4)
            for col in ["sign_entropy_bits", "acf_returns_lag1",
                        "acf_abs_returns_lag1", "acf_abs_returns_lag5",
                        "acf_abs_returns_lag22", "hurst_exponent"]:
                disp[col] = disp[col].round(4)
            # Sort by the "predictable magnitude" measure — the one
            # sign-entropy is blind to.
            disp = disp.sort_values("acf_abs_returns_lag1", ascending=False).head(15)
            st.markdown(
                "**Top-15 tickers ranked by volatility clustering** "
                "(highest ACF of |returns|). Sign-entropy column shows how "
                "close each ticker is to 1.0 — a measure that calls them "
                "'unpredictable' despite the strong magnitude autocorrelation."
            )
            st.dataframe(
                disp[["ticker", "sign_entropy_bits", "acf_returns_lag1",
                      "acf_abs_returns_lag1", "acf_abs_returns_lag5",
                      "acf_abs_returns_lag22", "hurst_exponent"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker"),
                    "sign_entropy_bits": st.column_config.NumberColumn(
                        "Sign-entropy (bits)", format="%.4f",
                        help="1.0 = 'looks unpredictable' by the 2-state lag-1 view.",
                    ),
                    "acf_returns_lag1": st.column_config.NumberColumn(
                        "ACF(r, lag-1)", format="%+.4f",
                        help="Direct lag-1 autocorrelation of raw returns. "
                             "Small but often non-zero on BIST.",
                    ),
                    "acf_abs_returns_lag1": st.column_config.NumberColumn(
                        "ACF(|r|, lag-1)", format="%.4f",
                        help="VOLATILITY CLUSTERING — how strongly today's "
                             "|return| predicts tomorrow's |return|. The "
                             "classic financial stylised fact sign-entropy misses.",
                    ),
                    "acf_abs_returns_lag5": st.column_config.NumberColumn(
                        "ACF(|r|, lag-5)", format="%.4f",
                        help="Volatility clustering at one trading week.",
                    ),
                    "acf_abs_returns_lag22": st.column_config.NumberColumn(
                        "ACF(|r|, lag-22)", format="%.4f",
                        help="Volatility clustering at one trading month.",
                    ),
                    "hurst_exponent": st.column_config.NumberColumn(
                        "Hurst (R/S)", format="%.4f",
                        help="0.5 = random walk; >0.5 = persistent; <0.5 = "
                             "mean-reverting. Computed via simple rescaled-range.",
                    ),
                },
            )
            st.caption(
                f":material/info: BIST: **{n_vol_cluster} of {n_total}** "
                f"tickers have lag-1 |r| autocorrelation > 0.20 (volatility "
                f"clustering). **{n_persistent} of {n_total}** have Hurst > "
                f"0.55 (persistent / trending regime). Sign-entropy at lag-1 "
                f"with 2 states cannot detect either of these. The 'EMH "
                f"passes' read from sign-entropy alone is incomplete."
            )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    """Render the full EEE Analysis tab content.

    The Neuromorphic Signals (SNN) sub-tab is hidden when the active universe
    has ``has_snn=False`` in app/universe_registry.py — the SNN classifier is
    pair-trading-specific and doesn't apply to non-financial universes (EEG).
    """
    # Note: previous versions called importlib.reload(universe_registry) here
    # to defeat Streamlit Cloud's stale-module cache. HF Spaces rebuilds the
    # container on every deploy, so the reload is unnecessary and was
    # contributing to "SessionInfo before init" log noise via Universe class
    # identity churn. Removed in PR #23.
    from universe_registry import get_universe
    from utils import current_universe, render_subtabs
    _active = get_universe(current_universe())

    clusters = load_cluster_assignments()
    sector_map = dict(zip(clusters["ticker"], clusters["sector"])) if not clusters.empty else {}

    # PHASE Y (Y1): replaced `st.tabs(...)` with `render_subtabs(...)` so
    # only the active sub-tab body executes. Methods Lab previously paid
    # for all 6 render_*() bodies on every page load (5-15 s on S&P);
    # now it pays only for the visible sub-tab.
    _sub_labels = (
        "RMT Denoising",
        "Graphical LASSO",
        "Wavelet Multi-Scale",
        "Transfer Entropy",
        "Information Theory",
    )
    if getattr(_active, "has_snn", True):
        _sub_labels = _sub_labels + ("Neuromorphic Signals",)

    _active_sub = render_subtabs("methods_lab", _sub_labels, label="Method")

    if _active_sub == "RMT Denoising":
        render_rmt(sector_map, u=_active)
    elif _active_sub == "Graphical LASSO":
        render_glasso(sector_map, u=_active)
    elif _active_sub == "Wavelet Multi-Scale":
        render_wavelets(sector_map, u=_active)
        # Neuroscience caption: scales are time-bands at 160 Hz, not days.
        if getattr(_active, "domain", "finance") == "neuroscience":
            st.caption(
                ":material/info: For EEG (160 Hz sampling), wavelet scales 1-7 "
                "correspond to bands ranging from ~12.5 ms (gamma, scale 1) up to "
                "~1.6 s (slow oscillations, scale 7), not days as in the financial "
                "universes."
            )
    elif _active_sub == "Transfer Entropy":
        render_transfer_entropy(sector_map, u=_active)
    elif _active_sub == "Information Theory":
        render_info_theory(sector_map, u=_active)
    elif _active_sub == "Neuromorphic Signals":
        render_snn(sector_map, u=_active)
