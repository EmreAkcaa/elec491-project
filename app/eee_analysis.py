"""EEE Analysis dashboard tab — RMT, Graphical LASSO, Wavelets, Transfer Entropy."""

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
    load_eigenvalue_spectrum, load_denoised_corr, load_denoised_mst_edges,
    load_denoised_mst_metrics,
    load_mst_edges, load_mst_metrics, load_batch_corr,
    load_partial_corr, load_partial_corr_edges, load_glasso_metadata,
    load_precision_matrix,
    load_wavelet_metadata, load_wavelet_mst_edges, load_wavelet_corr,
    load_wavelet_mst_metrics,
    load_te_edges, load_te_node_roles, load_te_matrix, load_net_te_matrix,
    load_cluster_assignments,
    load_dendrogram_order,
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
) -> go.Figure:
    """Create a Plotly network graph from edges.

    If ``node_metrics`` is provided (with columns ``ticker`` and ``size_metric``),
    nodes are sized by that centrality measure mapped onto ``size_range``;
    otherwise size scales with degree (legacy behaviour).
    """
    if not HAS_NETWORKX or edges_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=16))
        return fig

    G = nx.DiGraph() if directed else nx.Graph()
    for _, r in edges_df.iterrows():
        w = abs(float(r.get(edge_weight_col, 1.0)))
        G.add_edge(r["source"], r["target"], weight=w if w > 0 else 0.01)

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
            size = 8 + deg * 2
            metric_text = ""

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=size, color=color, line=dict(width=0.5, color="#fff")),
            text=node, textposition="top center", textfont=dict(size=7),
            hovertemplate=(
                f"<b>{node}</b><br>Sector: {sector}<br>Degree: {deg}"
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
) -> go.Figure:
    """Render a square matrix as a heatmap, dendrogram-reordered when possible."""
    if matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=14))
        apply_chart_style(fig, height=height)
        return fig

    if ordered_tickers:
        present = [t for t in ordered_tickers if t in matrix.columns and t in matrix.index]
        if len(present) >= 2:
            matrix = matrix.loc[present, present]

    colorscale = "RdBu" if diverging else "Blues"
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=list(matrix.columns),
        y=list(matrix.index),
        zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        reversescale=diverging,
        zmid=0 if diverging else None,
        hovertemplate=f"%{{y}} ↔ %{{x}}<br>{hover_label}=%{{z:.4f}}<extra></extra>",
        colorbar=dict(thickness=12, len=0.85),
    ))
    apply_chart_style(fig, height=height,
                      xaxis=dict(tickfont=dict(size=8), tickangle=-90),
                      yaxis=dict(tickfont=dict(size=8), autorange="reversed"))
    return fig


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_rmt(sector_map: dict):
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

        col_spec, col_mst = st.columns(2)

        with col_spec:
            # Eigenvalue spectrum plot
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
            apply_chart_style(fig_spec, height=420,
                              xaxis_title="Eigenvalue Index",
                              yaxis_title="Eigenvalue",
                              yaxis_type="log",
                              showlegend=False)
            render_chart(fig_spec, chart_id="rmt_spectrum", filename_base="eigenvalue_spectrum",
                         title_key="rmt_spectrum", default_title="Eigenvalue Spectrum vs MP Bounds")

        with col_mst:
            # Side-by-side MST comparison
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
                )
            else:
                metrics_df = load_denoised_mst_metrics()
                fig = _plot_network(
                    denoised_edges, sector_map,
                    edge_weight_col="distance",
                    node_metrics=metrics_df,
                )

            render_chart(fig, chart_id="rmt_mst", filename_base="rmt_mst",
                         title_key="rmt_mst",
                         default_title=f"MST Network ({mst_choice}, nodes sized by betweenness)")

        # Denoised correlation heatmap (full width)
        st.markdown("**Denoised Correlation Matrix** — eigenvalues outside the MP band reconstructed; noise eigenvalues replaced with their mean.")
        denoised = load_denoised_corr()
        order = load_dendrogram_order()
        fig_den = _plot_matrix_heatmap(
            denoised, order,
            zmin=-1.0, zmax=1.0, diverging=True,
            height=520, hover_label="ρ (denoised)",
        )
        render_chart(
            fig_den, chart_id="rmt_denoised_heatmap",
            filename_base="rmt_denoised_corr_heatmap",
            title_key="rmt_den_hm",
            default_title="Denoised correlation heatmap (ordered by dendrogram leaves)",
        )


def render_glasso(sector_map: dict):
    """Render Graphical LASSO section."""
    with st.container(border=True):
        section_header(
            "Graphical LASSO — Partial Correlation Network",
            "L1-regularized sparse precision matrix estimation. Non-zero entries represent "
            "direct (conditional) dependencies, filtering out indirect correlations. "
            "If A correlates with B only because both correlate with C, the GLASSO removes the A-B edge.",
        )

        edges = load_partial_corr_edges()
        meta = load_glasso_metadata()
        if edges.empty:
            st.info("Run the pipeline to generate GLASSO results.")
            return

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Direct Edges", meta.get("n_edges", len(edges)))
        c2.metric("Sparsity", f"{meta.get('sparsity_pct', 0):.1f}%")
        c3.metric("Regularization (alpha)", f"{meta.get('alpha', 0):.4f}")

        col_net, col_table = st.columns([3, 2])

        with col_net:
            # Partial correlation network
            fig = _plot_network(edges, sector_map,
                                edge_weight_col="abs_partial_corr",
                                title="Partial Correlation Network")
            render_chart(fig, chart_id="glasso_net", filename_base="glasso_network",
                         title_key="glasso_net",
                         default_title="Partial Correlation Network (Direct Dependencies)")

        with col_table:
            st.markdown("**Strongest Direct Dependencies**")
            display_edges = edges.head(30).copy()
            display_edges["sector_1"] = display_edges["source"].map(sector_map)
            display_edges["sector_2"] = display_edges["target"].map(sector_map)
            st.dataframe(
                display_edges[["source", "target", "partial_correlation", "sector_1", "sector_2"]],
                use_container_width=True, height=500,
                column_config={
                    "partial_correlation": st.column_config.NumberColumn(format="%.4f"),
                },
            )

        # Partial correlation + precision matrix heatmaps
        st.markdown("---")
        order = load_dendrogram_order()
        col_pc, col_prec = st.columns(2)

        with col_pc:
            st.markdown("**Partial Correlation Matrix** — direct dependencies after conditioning on all other tickers (clipped to ±0.3 for visibility).")
            partial = load_partial_corr()
            if not partial.empty:
                # Zero diagonal so it doesn't dominate the colorscale
                pc_display = partial.copy()
                np.fill_diagonal(pc_display.values, 0.0)
                fig_pc = _plot_matrix_heatmap(
                    pc_display, order,
                    zmin=-0.3, zmax=0.3, diverging=True,
                    height=460, hover_label="partial ρ",
                )
                render_chart(
                    fig_pc, chart_id="glasso_partial_heatmap",
                    filename_base="glasso_partial_corr_heatmap",
                    title_key="glasso_pc_hm",
                    default_title="Partial correlation heatmap",
                )
            else:
                st.info("Run the pipeline to generate the partial correlation matrix.")

        with col_prec:
            st.markdown("**Precision Matrix Sparsity** — non-zero off-diagonal entries are direct conditional dependencies. Zero entries imply conditional independence under Gaussianity.")
            precision = load_precision_matrix()
            if not precision.empty:
                # Build a binary sparsity pattern of |Θ_ij| above a small floor
                prec_abs = precision.abs()
                # Zero the diagonal
                np.fill_diagonal(prec_abs.values, 0.0)
                threshold = 1e-3
                sparsity = (prec_abs > threshold).astype(float)
                fig_prec = _plot_matrix_heatmap(
                    sparsity, order,
                    zmin=0.0, zmax=1.0, diverging=False,
                    height=460, hover_label="|Θ| > 1e-3",
                )
                # Override colorbar tick labels to show binary semantics
                fig_prec.update_traces(colorbar=dict(
                    thickness=12, len=0.85,
                    tickvals=[0, 1],
                    ticktext=["zero", "non-zero"],
                ))
                n_offdiag = sparsity.values.sum() // 2  # symmetric
                total_offdiag = (sparsity.shape[0] * (sparsity.shape[0] - 1)) // 2
                density = (n_offdiag / total_offdiag * 100) if total_offdiag else 0.0
                render_chart(
                    fig_prec, chart_id="glasso_precision_heatmap",
                    filename_base="glasso_precision_sparsity",
                    title_key="glasso_prec_hm",
                    default_title=f"Precision matrix sparsity ({int(n_offdiag)} edges, {density:.1f}% density)",
                )
            else:
                st.info("Run the pipeline to generate the precision matrix.")


def render_wavelets(sector_map: dict):
    """Render Wavelet multi-scale analysis section."""
    with st.container(border=True):
        section_header(
            "Wavelet Multi-Scale Correlation",
            "DWT (Daubechies-4) decomposes returns into frequency bands. "
            "Each scale isolates a specific frequency — unlike rolling windows which mix all frequencies. "
            "Short scales capture noise/day-trading; long scales reveal institutional/macro structure.",
        )

        meta = load_wavelet_metadata()
        if not meta:
            st.info("Run the pipeline to generate wavelet results.")
            return

        scales = meta.get("scales", {})
        n_scales = len(scales)

        # Scale selector
        scale_level = st.slider(
            "Wavelet Scale",
            min_value=1, max_value=n_scales, value=4,
            format="Scale %d",
            key="wavelet_scale",
        )
        scale_label = scales.get(str(scale_level), f"Scale {scale_level}")
        st.caption(f"**Scale {scale_level}** ({scale_label})")

        col_mst, col_corr = st.columns([3, 2])

        with col_mst:
            edges = load_wavelet_mst_edges(scale_level)
            scale_metrics = load_wavelet_mst_metrics(scale_level)
            if not edges.empty:
                fig = _plot_network(
                    edges, sector_map, edge_weight_col="distance",
                    node_metrics=scale_metrics,
                )
                total_weight = edges["distance"].sum()
                render_chart(fig, chart_id=f"wav_mst_{scale_level}", filename_base="wavelet_mst",
                             title_key="wav_mst",
                             default_title=f"MST at {scale_label} (Σdistance: {total_weight:.1f}, nodes sized by betweenness)")
            else:
                st.info("No MST data for this scale.")

        with col_corr:
            corr = load_wavelet_corr(scale_level)
            if not corr.empty:
                # Correlation distribution at this scale
                mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
                upper_vals = corr.values[mask]
                upper_vals = upper_vals[np.isfinite(upper_vals)]

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=upper_vals, nbinsx=60,
                    marker_color=get_colors()["primary"], opacity=0.75,
                ))
                avg_corr = float(np.mean(upper_vals))
                fig_hist.add_vline(x=avg_corr, line_dash="dash", line_color=get_colors()["secondary"],
                                   annotation_text=f"Mean: {avg_corr:.3f}", annotation_font_size=10)
                apply_chart_style(fig_hist, height=420,
                                  xaxis_title="Pairwise Correlation",
                                  yaxis_title="Frequency", showlegend=False)
                render_chart(fig_hist, chart_id=f"wav_hist_{scale_level}",
                             filename_base="wavelet_corr_dist",
                             title_key="wav_hist",
                             default_title=f"Correlation Distribution at {scale_label}")

        # Scale comparison summary
        st.markdown("**Cross-Scale Summary**")
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


def render_transfer_entropy(sector_map: dict):
    """Render Transfer Entropy section."""
    with st.container(border=True):
        section_header(
            "Transfer Entropy — Directed Information Flow",
            "Unlike correlation (symmetric), transfer entropy measures directed causality: "
            "'Does Stock A's past reduce uncertainty about Stock B's future?' "
            "Produces an asymmetric network revealing which stocks lead and which follow.",
        )

        roles = load_te_node_roles()
        edges = load_te_edges()
        if roles.empty:
            st.info("Run the pipeline to generate transfer entropy results.")
            return

        n_sources = (roles["role"] == "source").sum()
        n_sinks = (roles["role"] == "sink").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Information Sources", n_sources)
        c2.metric("Information Sinks", n_sinks)
        c3.metric("Significant Directed Edges", len(edges))

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
            st.markdown("**Top Information Sources (Leaders)**")
            sources = roles[roles["role"] == "source"].head(15)
            st.dataframe(
                sources[["ticker", "sector", "net_te_flow", "te_out", "te_in"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "net_te_flow": st.column_config.NumberColumn("Net TE Flow", format="%.4f"),
                    "te_out": st.column_config.NumberColumn("TE Out", format="%.4f"),
                    "te_in": st.column_config.NumberColumn("TE In", format="%.4f"),
                },
            )

            st.markdown("**Top Information Sinks (Followers)**")
            sinks = roles[roles["role"] == "sink"].tail(15).sort_values("net_te_flow")
            st.dataframe(
                sinks[["ticker", "sector", "net_te_flow", "te_out", "te_in"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "net_te_flow": st.column_config.NumberColumn("Net TE Flow", format="%.4f"),
                    "te_out": st.column_config.NumberColumn("TE Out", format="%.4f"),
                    "te_in": st.column_config.NumberColumn("TE In", format="%.4f"),
                },
            )

        # Net-TE flow heatmap (full width)
        st.markdown("---")
        st.markdown(
            "**Net Information Flow Heatmap** — `net[i,j] = TE(i→j) − TE(j→i)`. "
            "Read row `i`: red cells in column `j` mean information flows from `i` to `j` "
            "(i leads j); blue cells mean `i` lags `j`."
        )
        net_te = load_net_te_matrix()
        if not net_te.empty:
            order = load_dendrogram_order()
            v = float(np.nanmax(np.abs(net_te.to_numpy()))) if net_te.size else 1.0
            v = max(v, 1e-6)
            fig_net = _plot_matrix_heatmap(
                net_te, order,
                zmin=-v, zmax=v, diverging=True,
                height=520, hover_label="net TE",
            )
            render_chart(
                fig_net, chart_id="te_net_heatmap",
                filename_base="te_net_flow_heatmap",
                title_key="te_net_hm",
                default_title="Net transfer-entropy flow (red = source, blue = sink)",
            )
        else:
            st.info("Run the pipeline to generate the net transfer-entropy matrix.")



# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    """Render the full EEE Analysis tab content."""
    clusters = load_cluster_assignments()
    sector_map = dict(zip(clusters["ticker"], clusters["sector"])) if not clusters.empty else {}

    sub_rmt, sub_glasso, sub_wavelet, sub_te = st.tabs([
        "RMT Denoising", "Graphical LASSO", "Wavelet Multi-Scale",
        "Transfer Entropy",
    ])

    with sub_rmt:
        render_rmt(sector_map)

    with sub_glasso:
        render_glasso(sector_map)

    with sub_wavelet:
        render_wavelets(sector_map)

    with sub_te:
        render_transfer_entropy(sector_map)
