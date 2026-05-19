"""Shared dashboard utilities — loaders, event markers, data-quality checks.

All pages import from here so caches and logic are never duplicated.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chart_themes import (
    ChartTheme,
    get_active_theme,
    theme_to_layout,
    theme_to_colors,
)

# ---------------------------------------------------------------------------
# Theme & Style Constants
# ---------------------------------------------------------------------------

# Static fallback — used only at import time if nothing else is available.
# All chart code should call get_colors() instead.
COLORS = {
    "primary": "#4361EE",
    "secondary": "#E63946",
    "tertiary": "#2EC4B6",
    "muted": "#8D99AE",
    "positive": "rgba(67,97,238,0.12)",
    "negative": "rgba(230,57,70,0.12)",
    "band": "rgba(141,153,174,0.12)",
    "bg": "#FAFBFC",
}

SECTOR_PALETTE = [
    "#4361EE", "#E63946", "#2EC4B6", "#FF9F1C", "#6A0572",
    "#1B998B", "#E71D36", "#2B2D42", "#8338EC", "#FB5607",
    "#3A86A8", "#FFBE0B", "#06D6A0", "#118AB2", "#073B4C",
    "#EF476F", "#FFD166", "#06D6A0", "#118AB2", "#5E60CE",
    "#48BFE3", "#56CFE1",
]

# Legacy constant kept for any third-party code that reads it directly.
CHART_LAYOUT = dict(
    font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#2B2D42"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
    xaxis=dict(gridcolor="rgba(141,153,174,0.15)", zeroline=False),
    yaxis=dict(gridcolor="rgba(141,153,174,0.15)", zeroline=False),
)


def get_colors() -> dict[str, str]:
    """Return color palette reflecting the active theme."""
    return theme_to_colors(get_active_theme())


def apply_chart_style(fig, height: int = 420, title: str = "", **overrides):
    """Apply the active theme layout to any plotly figure."""
    theme = get_active_theme()
    layout = {**theme_to_layout(theme), "height": height, **overrides}
    # Handle title — can be a string (from title param) or a dict (from overrides)
    title_val = title or layout.get("title")
    if title_val:
        if isinstance(title_val, str):
            layout["title"] = dict(
                text=title_val,
                font=dict(size=theme.title_font_size, family=theme.font_family),
                x=0.01, xanchor="left", yanchor="top",
            )
        # else: title is already a dict from overrides, keep as-is
        layout["margin"] = dict(layout.get("margin", {}))
        layout["margin"]["t"] = max(layout["margin"].get("t", 10), 40)
    fig.update_layout(**layout)
    return fig


# Matrices wider than this render as a server-rasterized PNG (kaleido) shipped
# via st.image() instead of an interactive Plotly heatmap. Upstream's
# downsample_matrix_for_display already block-averages to ~97×97 (485-ticker
# S&P), which gets each heatmap to ~0.5 MB — but the EEE tab stacks 5+
# heatmaps plus IT artifacts, MI scatter, MST traces, and rolling info-theory
# panels, easily summing past the ~21 MB browser-side WebSocket frame cap.
# PNG shrinks each matrix to ~150–500 KB regardless of N, and preserves full
# per-ticker resolution (no block-averaging on this path).
N_INTERACTIVE_MATRIX_MAX = 160


@st.cache_data(show_spinner="Rendering matrix as PNG (large-N path)...")
def _rasterize_matrix_png(
    matrix_bytes: bytes,
    shape: tuple[int, int],
    columns: tuple[str, ...],
    index: tuple[str, ...],
    title: str,
    zmin: float,
    zmax: float,
    diverging: bool,
    colorbar_tickvals: tuple[float, ...] | None,
    colorbar_ticktext: tuple[str, ...] | None,
    colorbar_title: str,
    width: int,
    height: int,
) -> bytes:
    """Build a Plotly heatmap server-side and rasterize to PNG via kaleido.

    Cached on the matrix bytes + render params so re-renders (widget changes,
    tab switches) don't re-spawn the kaleido subprocess.
    """
    z = np.frombuffer(matrix_bytes, dtype=np.float64).reshape(shape)
    colorscale = "RdBu" if diverging else "Blues"
    colorbar = dict(thickness=12, len=0.85)
    if colorbar_title:
        colorbar["title"] = colorbar_title
    if colorbar_tickvals is not None and colorbar_ticktext is not None:
        colorbar["tickvals"] = list(colorbar_tickvals)
        colorbar["ticktext"] = list(colorbar_ticktext)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(columns),
        y=list(index),
        zmin=zmin, zmax=zmax,
        zmid=0 if diverging else None,
        colorscale=colorscale,
        reversescale=diverging,
        showscale=True,
        colorbar=colorbar,
        hoverinfo="skip",
    ))
    apply_chart_style(
        fig, height=height,
        title=title,
        margin=dict(l=80, r=20, t=40 if title else 10, b=80),
        xaxis=dict(tickfont=dict(size=6), tickangle=-90, automargin=True,
                   showticklabels=(shape[0] <= 80)),
        yaxis=dict(tickfont=dict(size=6), autorange="reversed", automargin=True,
                   showticklabels=(shape[0] <= 80)),
        width=width,
    )
    return fig.to_image(format="png", width=width, height=height,
                        scale=2, engine="kaleido")


def render_matrix_heatmap(
    matrix: "pd.DataFrame",
    *,
    chart_id: str,
    filename_base: str = "matrix_heatmap",
    title_key: str = "",
    default_title: str = "Matrix heatmap",
    ordered_tickers: list | None = None,
    zmin: float = -1.0,
    zmax: float = 1.0,
    diverging: bool = True,
    height: int = 520,
    hover_label: str = "value",
    n_interactive_max: int = N_INTERACTIVE_MATRIX_MAX,
    colorbar_tickvals: tuple[float, ...] | None = None,
    colorbar_ticktext: tuple[str, ...] | None = None,
    colorbar_title: str = "",
) -> None:
    """Render a square correlation / precision / TE matrix as a heatmap.

    For N <= ``n_interactive_max`` (BIST, EEG) renders an interactive Plotly
    heatmap with hover, downsampling via ``downsample_matrix_for_display``
    only when needed. For larger matrices (S&P-500) rasterizes to PNG
    server-side via kaleido and ships the bytes via ``st.image()`` to keep
    the WebSocket frame well under the browser's ~21 MB JS-buffer limit.
    """
    if matrix.empty:
        st.info("No data available.")
        return

    if ordered_tickers:
        present = [t for t in ordered_tickers if t in matrix.columns and t in matrix.index]
        if len(present) >= 2:
            matrix = matrix.loc[present, present]

    n_original = len(matrix)

    if n_original <= n_interactive_max:
        # Interactive Plotly path — BIST/EEG behavior unchanged. Apply
        # upstream's downsample helper defensively (no-op below max_dim).
        display_matrix, block_size = downsample_matrix_for_display(matrix, max_dim=200)
        n_after = display_matrix.shape[0]
        colorscale = "RdBu" if diverging else "Blues"
        if block_size > 1:
            hovertemplate = (
                f"block ({block_size}×{block_size}) mean<br>"
                f"row %{{y}} · col %{{x}}<br>"
                f"{hover_label}=%{{z:.4f}}<extra></extra>"
            )
        else:
            hovertemplate = f"%{{y}} ↔ %{{x}}<br>{hover_label}=%{{z:.4f}}<extra></extra>"
        colorbar = dict(thickness=12, len=0.85)
        if colorbar_title:
            colorbar["title"] = colorbar_title
        if colorbar_tickvals is not None and colorbar_ticktext is not None:
            colorbar["tickvals"] = list(colorbar_tickvals)
            colorbar["ticktext"] = list(colorbar_ticktext)
        _show_labels = n_after <= 80
        fig = go.Figure(go.Heatmap(
            z=display_matrix.values,
            x=list(display_matrix.columns),
            y=list(display_matrix.index),
            zmin=zmin, zmax=zmax,
            zmid=0 if diverging else None,
            colorscale=colorscale,
            reversescale=diverging,
            hovertemplate=hovertemplate,
            colorbar=colorbar,
        ))
        apply_chart_style(
            fig, height=height,
            xaxis=dict(tickfont=dict(size=8), tickangle=-90, showticklabels=_show_labels),
            yaxis=dict(tickfont=dict(size=8), autorange="reversed", showticklabels=_show_labels),
        )
        render_chart(fig, chart_id=chart_id, filename_base=filename_base,
                     title_key=title_key, default_title=default_title)
        return

    # Large-N static PNG path — preserves full N×N resolution. No client-side
    # plotly bytes; the entire heatmap is server-rendered and shipped as a
    # ~150–500 KB image. Title baked into the PNG via _rasterize_matrix_png.
    # (The editable title input was removed per audit A1 — see render_chart
    # docstring; `title_key` arg kept as no-op for back-compat.)
    title_text = default_title

    st.caption(
        f":material/info: {n_original}×{n_original} matrix rendered as a static "
        "PNG to stay under the WebSocket payload limit. Hover/zoom are disabled "
        "on this path — per-pair lookups live in the Pair Analysis view."
    )

    png_width = 1400
    png_height = 1400

    z_arr = np.ascontiguousarray(matrix.values, dtype=np.float64)
    try:
        png_bytes = _rasterize_matrix_png(
            z_arr.tobytes(),
            z_arr.shape,
            tuple(map(str, matrix.columns)),
            tuple(map(str, matrix.index)),
            title_text,
            float(zmin), float(zmax),
            bool(diverging),
            colorbar_tickvals,
            colorbar_ticktext,
            colorbar_title,
            png_width, png_height,
        )
    except Exception as exc:
        st.error(f"Failed to rasterize heatmap: {exc}")
        st.caption("kaleido may be missing — `uv sync` to reinstall.")
        return

    st.image(png_bytes, use_container_width=True)

    _, dl_col = st.columns([8, 1])
    with dl_col:
        st.download_button(
            ":material/download: PNG",
            data=png_bytes,
            file_name=f"{filename_base}.png",
            mime="image/png",
            key=f"_dl_matrix_{chart_id}",
        )


def render_chart(
    fig: go.Figure,
    chart_id: str,
    filename_base: str = "chart",
    title_key: str = "",
    default_title: str = "",
    width: str = "stretch",
    use_container_width: bool | None = None,   # deprecated; kept for back-compat
) -> None:
    """Render a Plotly figure with configured modebar, optional title, and export popover.

    Width API
    ---------
    ``width="stretch"`` (default) → chart fills its container.
    ``width="content"``           → chart sizes to its own content.

    The legacy ``use_container_width`` kwarg is still accepted for back-compat
    (True → ``"stretch"``, False → ``"content"``); callers should migrate to
    ``width=`` since Streamlit deprecates ``use_container_width`` after 2025-12-31.
    """
    from chart_export import render_export_popover, get_plotly_config

    # Back-compat shim — translate legacy kwarg to the new width= API.
    if use_container_width is not None:
        width = "stretch" if use_container_width else "content"

    # PHASE S (S7): chart titles render as bold markdown ABOVE the chart,
    # NOT inside the Plotly figure. The previous in-figure title path
    # (`fig.update_layout(title=...)`) painted text in the top-left corner
    # of the chart canvas where it overlapped axis labels, legends, and MST
    # node labels on dense plots — user complaint "graph names inside graphs
    # are almost always blocking something behind". Single-place fix that
    # cleans up ~28 callsites across dashboard / pair_analysis / time_machine /
    # cross_market / eee_analysis without touching them.
    if default_title:
        st.markdown(f"**{default_title}**")
    try:
        # Defensive: shrink top margin + clear any pre-existing layout title.
        # Plotly subtlety (verified on 5.24.1, 2026-05-19): calling
        # `update_layout(title=None)` does NOT remove the title attribute —
        # it leaves an empty Title() object that serializes to
        # `"title": {}` in JSON. Plotly.js then reads `title.text` as
        # `undefined` and renders the literal string "undefined" inside the
        # chart canvas. Setting `title=dict(text="")` instead serializes to
        # `"title": {"text": ""}` which Plotly.js renders as no title at all.
        cur_margin = dict(fig.layout.margin.to_plotly_json())
        cur_margin["t"] = 10
        fig.update_layout(title=dict(text=""), margin=cur_margin)
    except Exception:
        # Plotly version drift safety — never block a chart on margin fiddling.
        pass

    config = get_plotly_config(chart_id)
    # NOTE: st.plotly_chart does NOT accept width= in Streamlit 1.41.1 (kwarg
    # landed in ~1.45+). Translate the public `width=` API to the legacy
    # use_container_width= kwarg for the actual call. When the Streamlit pin
    # is bumped past 1.45 (see docs/STREAMLIT_VERSION_BUMP.md), this can
    # revert to: st.plotly_chart(fig, width=width, config=config).
    st.plotly_chart(fig, use_container_width=(width == "stretch"), config=config)
    render_export_popover(fig, chart_id, filename_base)


def inject_custom_css():
    """Inject global CSS for consistent look & feel."""
    st.markdown("""
    <style>
    /* ── Sidebar styling ─────────────────────────────────────────
       HOTFIX 2026-05-19: removed `[data-testid="stSidebarNav"]
       { display: none !important; }` here. That rule was added during
       the single-script `dashboard.py` era to hide Streamlit's
       auto-discovery of a `pages/` directory (which we didn't have, but
       the testid was the catch-all). Phase 2 (PR #64) moved us to
       `st.navigation([...], position="sidebar")` which RENDERS into the
       same `stSidebarNav` testid — so the rule was hiding our own page
       list. Users could only see the default page (Cross-Market) with
       no way to navigate. We use `app/views/` (not `pages/`), so
       auto-discovery never fires and there's no longer a conflict to
       hide. */

    /* Ensure sidebar collapsed control (re-open arrow) is always visible */
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        z-index: 999 !important;
    }

    /* ── Kill Streamlit top padding & deploy bar ─────────────── */
    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0 !important;
    }
    .stMainBlockContainer {
        padding-top: 1rem !important;
    }
    /* Hide deploy button and toolbar, keep sidebar toggle */
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
    }

    /* ── Plotly modebar — subtle, reveal on hover ────────────── */
    .js-plotly-plot .modebar {
        opacity: 0.15 !important;
        transition: opacity 0.2s ease !important;
    }
    .js-plotly-plot:hover .modebar {
        opacity: 0.85 !important;
    }
    .js-plotly-plot .modebar-btn {
        font-size: 16px !important;
    }

    /* ── Metric cards ────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fc 0%, #eef1f6 100%);
        border: 1px solid #e2e6ee;
        border-radius: 8px;
        padding: 12px 16px 8px 16px;
        box-shadow: 0 1px 3px rgba(43,45,66,0.06);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2B2D42;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #6c757d;
    }

    /* ── Segmented control — dark nav bar ────────────────────── */
    div[data-testid="stSegmentedControl"] {
        background: #1a1c2e;
        border-radius: 8px;
        padding: 4px;
    }
    div[data-testid="stSegmentedControl"] button {
        color: #8D99AE !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 8px 28px !important;
        border: none !important;
        transition: all 0.15s ease;
    }
    /* Sprint 2 PR-N: amplify active-state visual weight on top-nav
       (segmented_control). Streamlit 1.41 dropped the aria-checked attr
       and ships per-state data-testids instead:
          stBaseButton-segmented_control        → inactive
          stBaseButton-segmented_controlActive  → active
       Box-shadow + bold + solid bg make the active state pop; hover bg
       cues clickability on the inactive ones. */
    button[data-testid="stBaseButton-segmented_controlActive"] {
        background: #4361EE !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(67, 97, 238, 0.30) !important;
        border-color: #4361EE !important;
    }
    button[data-testid="stBaseButton-segmented_control"]:hover {
        background: rgba(67, 97, 238, 0.15) !important;
        color: #4361EE !important;
        cursor: pointer;
    }

    /* ── Sub-tabs — clean underline style ──────────────────────
       Sprint 2 PR-N: thicker underline (3px) + subtle background tint
       on the active tab so first-time graders can spot the active
       sub-tab without squinting. */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #e9ecef;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.88rem;
        color: #8D99AE;
        border-bottom: 2px solid transparent;
        padding: 10px 20px;
        margin-bottom: -2px;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        color: #4361EE;
        background: rgba(67, 97, 238, 0.04);
        cursor: pointer;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #4361EE;
        border-bottom: 3px solid #4361EE;
        background: rgba(67, 97, 238, 0.06);
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4361EE !important;
        height: 3px !important;
    }

    /* ── PHASE Y (Y1) — Sub-tabs as st.segmented_control styled like st.tabs
       Streamlit's `st.tabs` doesn't expose the active tab to Python, so we
       can't lazy-render hidden bodies — Methods Lab paid for all 6 sub-tabs
       on every render. PHASE Y replaces sub-tab st.tabs with st.segmented_
       control wrapped in a `.subtab-as-tabs` div, then gates the body
       rendering on the active selection. This CSS keeps the EXISTING
       underlined-tab visual so the demo audience doesn't see a visual
       break — user explicit choice (2026-05-19).

       Selectors verified on Streamlit 1.41.1 (same selectors Phase S used
       for the top-nav segmented_control). When the Streamlit pin moves
       past 1.45, retest — data-testid names sometimes churn across
       major bumps. */
    .subtab-as-tabs {
        margin-bottom: 0.4rem;
        border-bottom: 2px solid #e9ecef;
    }
    .subtab-as-tabs div[data-testid="stSegmentedControl"] {
        background: transparent !important;
        padding: 0 !important;
        gap: 0 !important;
    }
    .subtab-as-tabs button[data-testid="stBaseButton-segmented_control"],
    .subtab-as-tabs button[data-testid="stBaseButton-segmented_controlActive"] {
        background: transparent !important;
        color: #8D99AE !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 10px 20px !important;
        margin-bottom: -2px !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }
    .subtab-as-tabs button[data-testid="stBaseButton-segmented_control"]:hover {
        color: #4361EE !important;
        background: rgba(67, 97, 238, 0.04) !important;
        cursor: pointer;
    }
    .subtab-as-tabs button[data-testid="stBaseButton-segmented_controlActive"] {
        color: #4361EE !important;
        border-bottom: 3px solid #4361EE !important;
        background: rgba(67, 97, 238, 0.06) !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }

    /* ── Popover panels ──────────────────────────────────────── */
    [data-testid="stPopoverBody"] {
        border: 1px solid #e2e6ee;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(43,45,66,0.10);
        padding: 4px;
    }

    /* ── Section dividers ────────────────────────────────────── */
    .stMarkdown hr {
        border: none;
        border-top: 1px solid #e9ecef;
        margin: 1.2rem 0;
    }

    /* ── Bordered containers — card appearance ───────────────── */
    [data-testid="stVerticalBlock"] > div[data-testid="stExpander"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px;
    }

    /* ── Dataframe styling ───────────────────────────────────── */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Caption text ────────────────────────────────────────── */
    .stCaption {
        line-height: 1.5;
    }

    /* ── General polish ──────────────────────────────────────── */
    .stSelectbox label, .stDateInput label, .stSlider label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #555;
    }

    /* ── UX polish — Loading-state indicator (single top progress bar)
       Phase S painted ONE spinner per `.element-container.stale`, which
       meant a page with 20 widgets showed 20 spinners during a rerun —
       user reaction: "loading icons are shown like 20 of them looks very
       weird". Replaced with a single TOP-OF-PAGE thin progress bar that
       activates whenever ANY element on the page is recomputing. Uses
       CSS `:has()` to detect stale descendants without JavaScript.

       Two effects layered:
        1. Subtle opacity dimming (0.78) on stale elements — tells the
           user WHICH elements are recomputing, secondary cue only.
        2. Single 2px progress bar at the top of the viewport with a
           sliding gradient — primary "page is computing" cue.

       `:has()` selector support is universal in modern browsers
       (Chrome 105+ / Safari 15.4+ / Firefox 121+, all shipping in
       early-to-mid 2022 onwards). HF Spaces' demo audience is on
       up-to-date browsers. */
    .element-container.stale {
        opacity: 0.78 !important;
        transition: opacity 0.2s ease-in-out;
    }
    body:has(.element-container.stale)::after,
    [data-testid="stAppViewContainer"]:has(.element-container.stale)::after {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(
            90deg,
            rgba(67,97,238,0) 0%,
            rgba(67,97,238,1.0) 50%,
            rgba(67,97,238,0) 100%
        );
        background-size: 35% 100%;
        background-repeat: no-repeat;
        background-position: -35% 0;
        animation: stonecoal-progress 1.2s linear infinite;
        z-index: 99999;
        pointer-events: none;
    }
    @keyframes stonecoal-progress {
        0%   { background-position: -35% 0; }
        100% { background-position: 135% 0; }
    }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, icon: str = ""):
    """Render a styled page header with optional icon."""
    if icon:
        st.markdown(f"# {icon} {title}")
    else:
        st.markdown(f"# {title}")
    st.caption(subtitle)


def section_header(title: str, description: str = ""):
    """Render a consistent section header."""
    st.markdown("---")
    st.subheader(title)
    if description:
        st.caption(description)


# ---------------------------------------------------------------------------
# PHASE Y (Y1) — Sub-tab helper
# ---------------------------------------------------------------------------

def render_subtabs(
    page_name: str,
    options: tuple[str, ...],
    *,
    default: str | None = None,
    label: str = "Sub-tab",
) -> str:
    """Render a sub-tab selector styled as `st.tabs` and return the active option.

    Replaces `st.tabs` at callsites where we want to LAZY-render the active
    sub-tab body. `st.tabs` always executes every body (Streamlit doesn't
    expose active-tab to Python); this helper exposes the active value so
    callers can do::

        active = render_subtabs("methods_lab", ("RMT", "GLASSO", "Wavelet"))
        if active == "RMT":
            render_rmt()
        elif active == "GLASSO":
            render_glasso()
        # ... only the selected body runs

    Visual: the CSS block `.subtab-as-tabs` (in `inject_custom_css`) styles
    the underlying `st.segmented_control` to look like `st.tabs` — same
    underlined-tab vocabulary the demo audience already recognises.

    State key: ``f"{page_name}_subtab_{dataset_key}"`` — namespaced per
    dataset so switching BIST → S&P → BIST restores each dataset's last
    active sub-tab independently (matches Phase S #1 pattern for top-nav).

    Pending-stash: when the active option disappears on universe switch
    (e.g., SNN sub-tab hidden on EEG), the value is stashed on a
    ``__pending`` key so a flip back restores it.
    """
    if not options:
        raise ValueError("render_subtabs requires at least one option")

    # Per-dataset namespacing. Streamlit's session_state is shared across
    # widget instances, so we MUST namespace by dataset to keep BIST's
    # sub-tab choice independent of S&P's.
    try:
        _dataset_key = st.session_state.get("dataset", DASHBOARD_UNIVERSE)
    except Exception:
        _dataset_key = "bist"
    state_key = f"{page_name}_subtab_{_dataset_key}"
    pending_key = f"{state_key}__pending"

    options_tuple = tuple(options)
    default_value = default if default in options_tuple else options_tuple[0]

    # Clamp + pending restore (mirrors Phase S #1 top-nav logic).
    stored = st.session_state.get(state_key)
    if stored not in options_tuple:
        if stored is not None and stored != default_value:
            st.session_state[pending_key] = stored
        st.session_state.pop(state_key, None)
        st.session_state[state_key] = default_value
    elif pending_key in st.session_state:
        pending_value = st.session_state[pending_key]
        if pending_value in options_tuple and stored == default_value:
            st.session_state.pop(state_key, None)
            st.session_state[state_key] = pending_value
            st.session_state.pop(pending_key, None)
        elif stored != default_value:
            st.session_state.pop(pending_key, None)

    # Wrap segmented_control in a div so CSS .subtab-as-tabs targets only
    # THIS instance, not the top-nav segmented_control (which keeps its
    # dark pill-button look).
    st.markdown('<div class="subtab-as-tabs">', unsafe_allow_html=True)
    active = st.segmented_control(
        label,
        options_tuple,
        key=state_key,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    return active or default_value


# ---------------------------------------------------------------------------
# Paths — universe-aware (BIST / S&P-500 / EEG)
# ---------------------------------------------------------------------------
APP_DIR      = Path(__file__).resolve().parent       # app/
PROJECT_ROOT = APP_DIR.parent                        # repo root

for _p in (str(PROJECT_ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Phase H: per-session universe selection.
#
# DASHBOARD_UNIVERSE is the boot-time default (env-var driven). At runtime,
# the sidebar selector in app/dashboard.py writes the active universe key into
# st.session_state["universe"], which current_universe() reads back. Every
# cached loader in this file routes through a private _load_*(universe, ...)
# function so Streamlit caches are naturally keyed by universe — BIST and
# S&P caches coexist; switching is a one-line session-state change.
DASHBOARD_UNIVERSE = os.environ.get("DASHBOARD_UNIVERSE", "bist")


# BIST family numéraire mapping. The sidebar exposes (dataset, bist_basis)
# as two controls; this dict resolves the (dataset="bist", bist_basis=…)
# pair to the on-disk universe key. Other datasets (sp500, eeg_*) are used
# verbatim and ignore bist_basis.
_BIST_BASIS_TO_KEY = {
    "try":  "bist",       # default — TRY-denominated original
    "usd":  "bist_usd",   # numéraire-converted via USD/TRY
    "gold": "bist_gold",  # numéraire-converted via gold spot
}


def current_universe() -> str:
    """Resolve (dataset, bist_basis) from session_state → universe key.

    The Phase 1 sidebar exposes two controls instead of one:
      * Dataset:       BIST 100 / S&P 500 / EEG Motor Imagery
      * Base currency: TRY / USD / Gold     (BIST only)

    This function is the SINGLE chokepoint that maps the (dataset,
    bist_basis) pair to one of the five on-disk universe keys:
    ``bist`` / ``bist_usd`` / ``bist_gold`` / ``sp500`` /
    ``eeg_motor_left_right``. Every ``@st.cache_data`` loader takes the
    universe key as a positional argument, so changing the resolution
    here auto-rekeys all dependent caches — callers don't change.

    Backward-compat: if Phase-1 keys aren't in session_state yet (e.g.,
    a smoke import outside Streamlit), falls back to the legacy
    ``st.session_state["universe"]`` and ultimately ``DASHBOARD_UNIVERSE``
    env var so nothing pre-Phase-1 breaks during the migration.
    """
    try:
        dataset = st.session_state.get("dataset")
        bist_basis = st.session_state.get("bist_basis", "try")
    except Exception:
        # Outside Streamlit script context (bare import / smoke test).
        dataset = None
        bist_basis = "try"

    if dataset is None:
        # Legacy fallback — pre-Phase-1 session or non-Streamlit context.
        try:
            return st.session_state.get("universe", DASHBOARD_UNIVERSE)
        except Exception:
            return DASHBOARD_UNIVERSE

    if dataset == "bist":
        return _BIST_BASIS_TO_KEY.get(bist_basis, "bist")
    return dataset


def data_raw(universe: str | None = None) -> Path:
    return PROJECT_ROOT / "data" / (universe or current_universe()) / "raw"


def data_processed(universe: str | None = None) -> Path:
    return PROJECT_ROOT / "data" / (universe or current_universe()) / "processed"


# Phase 4: base-asset loader for Pair Analysis "Compare against" FX/Gold.
# These series live OUTSIDE any universe subtree (they're global financial
# series, not universe-specific) at `data/raw/base_assets/{asset_key}.parquet`.
# Schema: Date-indexed parquet with a single price column named after
# `asset_key` (e.g., `usd_try`, `gold_usd`).
@st.cache_data(show_spinner="Loading base-asset price series...")
def _load_base_asset(asset_key: str) -> pd.Series:
    """Load a base-asset close-price series for cross-asset comparison.

    Returns an empty Series when the file is missing — callers should
    check ``series.empty`` before using. Cached by ``asset_key`` so the
    same series is loaded once per session for any number of Pair
    Analysis sessions that compare against it.
    """
    path = PROJECT_ROOT / "data" / "raw" / "base_assets" / f"{asset_key}.parquet"
    if not path.exists():
        return pd.Series(dtype=float, name=asset_key)
    df = pd.read_parquet(path)
    if asset_key not in df.columns:
        return pd.Series(dtype=float, name=asset_key)
    series = df[asset_key].copy()
    series.name = asset_key
    if series.index.name is None:
        series.index.name = "Date"
    return series


def load_base_asset(asset_key: str) -> pd.Series:
    """Public loader for base-asset price series. Currently supports
    ``"usd_try"`` and ``"gold_usd"``. Returns an empty Series if the
    file is missing (graceful degrade — UI shows a friendly notice).
    """
    return _load_base_asset(asset_key)


def data_results(universe: str | None = None) -> Path:
    return PROJECT_ROOT / "data" / (universe or current_universe()) / "results"


# Backwards-compat module-level constants. These resolve at *import time* using
# the boot-time DASHBOARD_UNIVERSE env var and do NOT track session state.
# New code should call data_raw() / data_processed() / data_results() instead.
DATA_RAW       = data_raw(DASHBOARD_UNIVERSE)
DATA_PROCESSED = data_processed(DASHBOARD_UNIVERSE)
DATA_RESULTS   = data_results(DASHBOARD_UNIVERSE)


# ---------------------------------------------------------------------------
# Cached data loaders — public wrappers thread session_state into a
# universe-keyed underscored cache. The underscored functions are what
# Streamlit actually caches; the public wrappers exist so call sites can
# stay parameter-free.
# ---------------------------------------------------------------------------

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee `df.index` is a DatetimeIndex.

    Financial universes (BIST, S&P) ship parquet files with a DatetimeIndex
    (the trading day). EEG ships a float64 index of seconds-since-recording-
    start, which breaks every downstream date widget / chart that calls
    ``adj_close.index.min().date()``.

    To keep the rest of the dashboard universe-agnostic, we synthesise a
    DatetimeIndex anchored at 2020-01-01 with 1-second steps when the
    underlying index isn't datetime-typed. The displayed "dates" carry no
    physical meaning for EEG — the chart's x-axis just labels samples by
    a synthetic clock so the existing widgets render cleanly.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    synthetic = pd.date_range(
        start="2020-01-01", periods=len(df.index), freq="1s",
    )
    return df.set_axis(synthetic, axis=0)


# Streamlit Cloud's free tier has ~1 GB RAM (less after Python + libs are
# loaded; effective budget ≈ 500-700 MB). EEG's 593,280-sample × 64-channel
# parquet is ~300 MB raw. When the dashboard JSON-serialises it via
# `returns.to_json(...)` and re-parses it inside @st.cache_data compute
# helpers, working memory peaks past 1 GB and the kernel OOM-kills the
# Streamlit process (visible as "connection reset by peer" on the health
# check). Downsampling at load time keeps the dashboard responsive without
# affecting the pre-computed pipeline artifacts (which were built from the
# full data and live in data/<u>/results/*.{parquet,csv,json}).
_DASHBOARD_MAX_ROWS = 8_000  # comfortable on Streamlit Cloud free tier


def _downsample_if_oversize(df: pd.DataFrame, max_rows: int = _DASHBOARD_MAX_ROWS) -> pd.DataFrame:
    """Uniform-stride downsample if `df` has more rows than `max_rows`.

    BIST (~1,544 rows) and S&P (~1,547 rows) pass through unchanged.
    EEG (~593,280 rows) is decimated by stride ≈ 74 to land near `max_rows`.
    The dashboard's on-the-fly correlation/rolling-stats compute on the
    downsampled series is statistically equivalent to the full series at
    the windows the dashboard uses (every-Nth-sample subsampling preserves
    correlation structure when N << autocorrelation length, which holds
    for our 60-day / 120-day / 252-day windows even on EEG).
    """
    n = len(df)
    if n <= max_rows:
        return df
    stride = max(1, n // max_rows)
    return df.iloc[::stride].copy()


@st.cache_data(show_spinner="Loading prices (first time only)…")
def _load_adj_close(universe: str) -> pd.DataFrame:
    df = pd.read_parquet(data_processed(universe) / "adj_close.parquet")
    df = _ensure_datetime_index(df)
    return _downsample_if_oversize(df)


def load_adj_close() -> pd.DataFrame:
    return _load_adj_close(current_universe())


@st.cache_data(show_spinner="Loading log returns (first time only)…")
def _load_log_returns(universe: str) -> pd.DataFrame:
    df = pd.read_parquet(data_processed(universe) / "log_returns.parquet")
    df = _ensure_datetime_index(df)
    return _downsample_if_oversize(df)


def load_log_returns() -> pd.DataFrame:
    return _load_log_returns(current_universe())


@st.cache_data
def _load_summary_stats(universe: str) -> pd.DataFrame:
    return pd.read_parquet(data_results(universe) / "summary_stats.parquet")


def load_summary_stats() -> pd.DataFrame:
    return _load_summary_stats(current_universe())


@st.cache_data
def _load_batch_corr(universe: str) -> pd.DataFrame:
    return pd.read_parquet(data_results(universe) / "pearson_corr.parquet")


def load_batch_corr() -> pd.DataFrame:
    return _load_batch_corr(current_universe())


@st.cache_data
def _load_coverage(universe: str) -> pd.DataFrame:
    return pd.read_csv(data_processed(universe) / "coverage_report.csv")


def load_coverage() -> pd.DataFrame:
    return _load_coverage(current_universe())


@st.cache_data
def _load_top_bottom(universe: str) -> pd.DataFrame:
    return pd.read_csv(data_results(universe) / "top_bottom_pairs.csv")


def load_top_bottom() -> pd.DataFrame:
    return _load_top_bottom(current_universe())


@st.cache_data
def _load_metadata(universe: str) -> dict:
    path = data_results(universe) / "pipeline_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_metadata() -> dict:
    return _load_metadata(current_universe())


@st.cache_data
def _load_fetch_metadata(universe: str) -> dict:
    path = data_raw(universe) / "fetch_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_fetch_metadata() -> dict:
    return _load_fetch_metadata(current_universe())


@st.cache_data
def _load_xu100(universe: str) -> pd.Series:
    """Load the universe's market-index series.

    The file is named ``xu100.parquet`` for historical reasons (the original
    BIST universe) but for S&P-500 it contains the ``^GSPC`` series.
    """
    path = data_raw(universe) / "xu100.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "Adj Close" in df.columns:
            return df["Adj Close"]
        if isinstance(df.columns, pd.MultiIndex):
            return df.iloc[:, 0]
        return df.iloc[:, 0]
    return pd.Series(dtype=float)


def load_xu100() -> pd.Series:
    return _load_xu100(current_universe())


@st.cache_data
def _load_linkage(universe: str):
    Z_path = data_results(universe) / "linkage_matrix.npy"
    labels_path = data_results(universe) / "linkage_labels.json"
    if Z_path.exists() and labels_path.exists():
        Z = np.load(Z_path)
        with open(labels_path) as f:
            labels = json.load(f)
        return Z, labels
    return None, None


def load_linkage():
    return _load_linkage(current_universe())


@st.cache_data
def _load_dendrogram_order(universe: str) -> Optional[list]:
    path = data_results(universe) / "dendrogram_order.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_dendrogram_order() -> Optional[list]:
    return _load_dendrogram_order(current_universe())


@st.cache_data
def _load_cluster_assignments(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "cluster_assignments.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_cluster_assignments() -> pd.DataFrame:
    return _load_cluster_assignments(current_universe())


@st.cache_data
def _load_mst_edges(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "mst_edges.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_mst_edges() -> pd.DataFrame:
    return _load_mst_edges(current_universe())


@st.cache_data
def _load_mst_metrics(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "mst_node_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_mst_metrics() -> pd.DataFrame:
    return _load_mst_metrics(current_universe())


@st.cache_data
def _load_dislocation_candidates(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "dislocation_candidates.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_dislocation_candidates() -> pd.DataFrame:
    return _load_dislocation_candidates(current_universe())


# ---------------------------------------------------------------------------
# Phase X — Cross-asset (FX / Gold) sensitivity loaders
# ---------------------------------------------------------------------------
# Written by ``src/cross_asset.py:run_cross_asset`` for BIST only. Other
# universes return empty frames; the Signals page hides the cross-asset
# section in that case. Schema:
#
#   cross_asset_summary.parquet        -- columns: ticker, sector,
#                                         corr_usd_try, n_obs_usd_try,
#                                         corr_gold_usd, n_obs_gold_usd
#   cross_asset_corr_rolling_<key>     -- date-indexed panel, columns =
#                                         tickers, values = 252d rolling
#                                         Pearson correlation with the
#                                         base asset's log returns.
#
# Look-ahead: none. All inputs are past-only and the rolling computation
# is left-aligned by date.

@st.cache_data
def _load_cross_asset_summary(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "cross_asset_summary.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_cross_asset_summary() -> pd.DataFrame:
    """Per-ticker full-period correlation with USD/TRY + Gold.

    Returns an empty DataFrame for non-BIST universes (the stage gates
    itself on ``market_id == "bist"``). The Signals page checks
    ``df.empty`` and hides the cross-asset breakout section accordingly.
    """
    return _load_cross_asset_summary(current_universe())


@st.cache_data
def _load_cross_asset_rolling(universe: str, asset_key: str) -> pd.DataFrame:
    if asset_key not in ("usd_try", "gold_usd"):
        return pd.DataFrame()
    path = data_results(universe) / f"cross_asset_corr_rolling_{asset_key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_cross_asset_rolling(asset_key: str) -> pd.DataFrame:
    """252-day rolling Pearson correlation panel (date x ticker) vs a base asset.

    ``asset_key`` must be ``"usd_try"`` or ``"gold_usd"``. Returns an
    empty DataFrame for non-BIST universes or when the artifact is missing.
    """
    return _load_cross_asset_rolling(current_universe(), asset_key)


# ---------------------------------------------------------------------------
# Phase 3 (slim) — PIT snapshot loaders
# ---------------------------------------------------------------------------
# These read precomputed snapshot files written by
# ``src/pit_snapshots.py:run_pit_snapshots``. Snapshots only exist for
# the universes in ``_PRECOMPUTE_MARKETS`` (bist + sp500) at window=252;
# loaders return empty DataFrames otherwise, letting Time Machine fall
# back to live compute transparently.
#
# Date semantics: user-picked dates are snapped to the NEAREST stored
# snapshot by scanning the directory once (cached per session). The
# Time Machine page shows a caption when the snap differs from the
# user's pick by more than 1 trading day, so the demo audience always
# knows which date the visualization is for.

_PIT_DIR_MAP = {
    "corr": "pit_corr",
    "mst": "pit_mst",
    "dislocation": "pit_dislocation",
}


@st.cache_data
def _pit_snapshot_dates(universe: str, window: int, kind: str) -> list[str]:
    """List the ISO-formatted dates available in a snapshot directory.

    Cached because the directory enumeration costs ~10 ms but is hit on
    every Time Machine slider drag (via _snap_to_nearest_snapshot).
    Cache key includes (universe, window, kind) so the cache invalidates
    correctly when the user flips dataset or window.
    """
    subdir = _PIT_DIR_MAP[kind]
    d = data_results(universe) / subdir / f"w{window}"
    if not d.exists():
        return []
    # File names are "YYYY-MM-DD.parquet" or "YYYY-MM-DD.csv" — strip
    # the extension to recover the date string.
    out: list[str] = []
    for p in sorted(d.iterdir()):
        stem = p.stem  # filename without extension
        # Defensive: skip non-date filenames if someone drops trash here
        if len(stem) == 10 and stem[4] == "-" and stem[7] == "-":
            out.append(stem)
    return out


def pit_snapshot_dates(window: int = 252, kind: str = "corr") -> list[str]:
    """Public wrapper: dates available in the current universe's grid."""
    return _pit_snapshot_dates(current_universe(), window, kind)


def snap_to_nearest_snapshot(
    requested_date: "pd.Timestamp", *, window: int = 252, kind: str = "corr",
) -> Optional[str]:
    """Return the ISO date string of the snapshot nearest the requested date.

    Returns None if no snapshots exist for the current (universe, window, kind).
    Snap distance is signed (we want the nearest, regardless of past/future)
    so the user gets the best-matching snapshot for whatever date they pick.
    """
    dates_iso = pit_snapshot_dates(window=window, kind=kind)
    if not dates_iso:
        return None
    requested_ts = pd.Timestamp(requested_date)
    # Convert all snapshot dates to timestamps once; pick min |diff|.
    diffs = [(abs((pd.Timestamp(d) - requested_ts).days), d) for d in dates_iso]
    diffs.sort(key=lambda x: x[0])
    return diffs[0][1]


@st.cache_data(show_spinner="Loading PIT correlation snapshot…")
def _load_pit_snapshot(universe: str, window: int, date_iso: str) -> pd.DataFrame:
    """Read one PIT correlation matrix from disk. Returns empty on miss.

    Universe + window + date together form the unique cache key, so
    repeat drags to the same date hit memory.
    """
    path = data_results(universe) / "pit_corr" / f"w{window}" / f"{date_iso}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_pit_snapshot(window: int, date_iso: str) -> pd.DataFrame:
    """Public wrapper: load PIT corr for the current universe + given date."""
    return _load_pit_snapshot(current_universe(), window, date_iso)


@st.cache_data(show_spinner=False)
def _load_pit_mst_snapshot(universe: str, window: int, date_iso: str) -> pd.DataFrame:
    """Read one PIT MST edges CSV. Returns empty on miss."""
    path = data_results(universe) / "pit_mst" / f"w{window}" / f"{date_iso}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_pit_mst_snapshot(window: int, date_iso: str) -> pd.DataFrame:
    """Public wrapper: load PIT MST edges for the current universe + date."""
    return _load_pit_mst_snapshot(current_universe(), window, date_iso)


@st.cache_data(show_spinner=False)
def _load_pit_dislocation_snapshot(
    universe: str, window: int, date_iso: str,
) -> pd.DataFrame:
    """Read one PIT top-dislocation table. Returns empty on miss."""
    path = data_results(universe) / "pit_dislocation" / f"w{window}" / f"{date_iso}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_pit_dislocation_snapshot(window: int, date_iso: str) -> pd.DataFrame:
    """Public wrapper: load PIT top dislocations for current universe + date."""
    return _load_pit_dislocation_snapshot(current_universe(), window, date_iso)


# ---------------------------------------------------------------------------
# Walk-forward signals loaders (PR #69)
# ---------------------------------------------------------------------------
# Per-date snapshots written by ``src/walk_forward_signals.py``. Each
# snapshot is a 20-row pair table re-screened past-only at the as-of
# date — so scrubbing the Signals page date picker shows trades an
# honest as-of-D observer would have proposed (no hindsight in pair
# selection, no future leak in state).
#
# Grid stride: 5B BIST, 21B S&P. ``snap_to_preceding_snapshot`` below
# returns the largest grid date ≤ user's pick — never a future date,
# unlike the existing ``snap_to_nearest_snapshot`` which is correct for
# Time Machine "around this date" semantic but would silently undo the
# walk-forward guarantee if reused here.

@st.cache_data(show_spinner=False)
def _walkforward_signals_dates(universe: str, window: int) -> list[str]:
    """List the ISO-formatted dates available in the walk-forward grid.

    Returns ``[]`` when the directory doesn't exist (universe not in the
    precompute set, or pipeline hasn't been run).
    """
    d = data_results(universe) / "walkforward_signals" / f"w{window}"
    if not d.exists():
        return []
    out: list[str] = []
    for f in d.iterdir():
        if not f.suffix == ".parquet":
            continue
        stem = f.stem
        if (
            len(stem) == 10 and stem[4] == "-" and stem[7] == "-"
            and stem[:4].isdigit() and stem[5:7].isdigit() and stem[8:].isdigit()
        ):
            out.append(stem)
    return sorted(out)


def walkforward_signals_dates(window: int = 60) -> list[str]:
    """Public wrapper: walk-forward grid dates for the active universe."""
    return _walkforward_signals_dates(current_universe(), window)


def snap_to_preceding_snapshot(
    requested_date,
    *,
    grid_dates: list[str],
) -> Optional[str]:
    """Return the largest grid date ≤ ``requested_date``, or None.

    Unlike ``snap_to_nearest_snapshot`` (which uses absolute distance),
    this never returns a future date — preserves the no-look-ahead
    guarantee when used for walk-forward signal lookup.

    Caller passes the grid date list explicitly so this stays cheap and
    testable in isolation (the standard pattern is
    ``snap_to_preceding_snapshot(date, grid_dates=walkforward_signals_dates())``).
    """
    if not grid_dates:
        return None
    requested_ts = pd.Timestamp(requested_date)
    # grid_dates is sorted ascending; find the rightmost date ≤ requested.
    last: Optional[str] = None
    for d in grid_dates:
        if pd.Timestamp(d) <= requested_ts:
            last = d
        else:
            break
    return last


@st.cache_data(show_spinner=False)
def _load_walkforward_signals_snapshot(
    universe: str, window: int, date_iso: str,
) -> pd.DataFrame:
    """Read one walk-forward signal snapshot. Returns empty on miss."""
    path = (
        data_results(universe) / "walkforward_signals" / f"w{window}"
        / f"{date_iso}.parquet"
    )
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_walkforward_signals_snapshot(date_iso: str, window: int = 60) -> pd.DataFrame:
    """Public wrapper: load a walk-forward signal snapshot for the active universe."""
    return _load_walkforward_signals_snapshot(current_universe(), window, date_iso)


# ---------------------------------------------------------------------------
# PHASE Y (Y2) — MST layout loaders
# ---------------------------------------------------------------------------
# Read precomputed NetworkX layout positions (written by src/mst_layouts.py)
# so the dashboard skips the live nx.spring_layout call on every render.
#
# Source names match the JSON filenames under data/<universe>/results/layouts/:
#   - "main_mst"           (from mst_edges.csv — Clustering & Network MST)
#   - "denoised_mst"       (from denoised_mst_edges.csv — RMT sub-tab)
#   - "wavelet_mst_scale1" .. "wavelet_mst_scale7" (Wavelet sub-tab)
#   - "te_network"         (from te_network_edges.csv — TE sub-tab)
#
# Returns empty dict when the layout file is missing (renderer should
# fall back to live `nx.spring_layout`/`nx.kamada_kawai_layout`).


@st.cache_data(show_spinner=False)
def _load_mst_layout(universe: str, source: str) -> dict[str, tuple[float, float]]:
    """Read one precomputed MST layout JSON. Returns positions dict.

    Cache key: (universe, source). The JSON file is ~25-50 KB so the
    read itself is ~10 ms; @st.cache_data caps it to one disk hit per
    (universe, source) per session.
    """
    path = data_results(universe) / "layouts" / f"{source}.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001 — log + return empty on malformed JSON
        return {}
    positions = payload.get("positions", {})
    # Coerce list-of-floats → tuple for renderer's expected type.
    return {str(node): (float(xy[0]), float(xy[1])) for node, xy in positions.items()}


def load_mst_layout(source: str) -> dict[str, tuple[float, float]]:
    """Public wrapper: read the precomputed MST layout for the active
    universe + given source. Returns empty dict on miss (renderer falls
    back to live compute)."""
    return _load_mst_layout(current_universe(), source)


# ---------------------------------------------------------------------------
# EEE Analysis loaders
# ---------------------------------------------------------------------------

@st.cache_data
def _load_eigenvalue_spectrum(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "eigenvalue_spectrum.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_eigenvalue_spectrum() -> pd.DataFrame:
    return _load_eigenvalue_spectrum(current_universe())


@st.cache_data(show_spinner="Loading RMT-denoised correlation matrix…")
def _load_denoised_corr(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "denoised_corr.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_denoised_corr() -> pd.DataFrame:
    return _load_denoised_corr(current_universe())


@st.cache_data
def _load_denoised_mst_edges(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "denoised_mst_edges.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_denoised_mst_edges() -> pd.DataFrame:
    return _load_denoised_mst_edges(current_universe())


@st.cache_data(show_spinner="Loading partial correlation matrix…")
def _load_partial_corr(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "partial_corr.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_partial_corr() -> pd.DataFrame:
    return _load_partial_corr(current_universe())


@st.cache_data(show_spinner="Loading precision matrix…")
def _load_precision_matrix(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "precision_matrix.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_precision_matrix() -> pd.DataFrame:
    return _load_precision_matrix(current_universe())


@st.cache_data
def _load_partial_corr_edges(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "partial_corr_edges.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_partial_corr_edges() -> pd.DataFrame:
    return _load_partial_corr_edges(current_universe())


@st.cache_data
def _load_glasso_metadata(universe: str) -> dict:
    path = data_results(universe) / "glasso_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_glasso_metadata() -> dict:
    return _load_glasso_metadata(current_universe())


@st.cache_data
def _load_wavelet_metadata(universe: str) -> dict:
    path = data_results(universe) / "wavelet_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_wavelet_metadata() -> dict:
    return _load_wavelet_metadata(current_universe())


@st.cache_data
def _load_wavelet_mst_edges(universe: str, scale: int) -> pd.DataFrame:
    path = data_results(universe) / f"wavelet_mst_edges_scale{scale}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_wavelet_mst_edges(scale: int) -> pd.DataFrame:
    return _load_wavelet_mst_edges(current_universe(), scale)


@st.cache_data
def _load_wavelet_corr(universe: str, scale: int) -> pd.DataFrame:
    path = data_results(universe) / f"wavelet_corr_scale{scale}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_wavelet_corr(scale: int) -> pd.DataFrame:
    return _load_wavelet_corr(current_universe(), scale)


_TE_EDGE_COLUMNS = [
    "source", "target", "te_forward", "te_backward", "net_te", "dominant_direction",
]


def downsample_matrix_for_display(
    df: pd.DataFrame, max_dim: int = 200, agg: str = "mean"
) -> tuple[pd.DataFrame, int]:
    """Block-downsample a square correlation/MI matrix when it's larger than
    ``max_dim`` × ``max_dim`` so the JSON payload stays under the streamlit
    websocket cap. Returns ``(matrix, block_size)`` — ``block_size == 1`` means
    no downsampling was applied.

    A 485-ticker S&P matrix at full resolution is ~4.5 MB serialized; with a
    block size of 5 it collapses to a 97×97 matrix at ~0.18 MB while keeping
    the visual block structure (you can't read 485 axis labels anyway).
    """
    n = df.shape[0]
    if n <= max_dim:
        return df, 1
    block = max(2, math.ceil(n / max_dim))
    new_n = n // block
    arr = df.values
    out = np.zeros((new_n, new_n))
    for i in range(new_n):
        for j in range(new_n):
            block_view = arr[i * block:(i + 1) * block, j * block:(j + 1) * block]
            out[i, j] = block_view.mean() if agg == "mean" else block_view.max()
    # Use the first label from each block so axis labels stay informative
    labels = [df.columns[i * block] for i in range(new_n)]
    return (
        pd.DataFrame(out, index=labels, columns=labels),
        block,
    )


@st.cache_data
def _load_te_edges(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "te_network_edges.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=_TE_EDGE_COLUMNS)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # Post-fix BH-FDR can leave zero significant edges → empty CSV.
        return pd.DataFrame(columns=_TE_EDGE_COLUMNS)


def load_te_edges() -> pd.DataFrame:
    return _load_te_edges(current_universe())


@st.cache_data
def _load_te_node_roles(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "te_node_roles.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_te_node_roles() -> pd.DataFrame:
    return _load_te_node_roles(current_universe())


@st.cache_data(show_spinner="Loading transfer entropy matrix…")
def _load_te_matrix(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "transfer_entropy_matrix.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_te_matrix() -> pd.DataFrame:
    return _load_te_matrix(current_universe())


@st.cache_data(show_spinner="Loading raw TE matrix…")
def _load_te_matrix_raw(universe: str) -> pd.DataFrame:
    """Pre-FDR TE values — useful for ranking edges by magnitude even when
    the FDR-corrected mask is sparse (the common case at 100-shuffle
    resolution on N>50 ticker grids)."""
    path = data_results(universe) / "transfer_entropy_raw.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # Fall back to the legacy filtered matrix when the raw file doesn't exist
    # (pre-Phase-1.3 pipeline runs).
    return _load_te_matrix(universe)


def load_te_matrix_raw() -> pd.DataFrame:
    return _load_te_matrix_raw(current_universe())


@st.cache_data(show_spinner="Loading net TE matrix…")
def _load_net_te_matrix(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "net_transfer_entropy_matrix.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_net_te_matrix() -> pd.DataFrame:
    return _load_net_te_matrix(current_universe())


@st.cache_data
def _load_denoised_mst_metrics(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "denoised_mst_node_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_denoised_mst_metrics() -> pd.DataFrame:
    return _load_denoised_mst_metrics(current_universe())


@st.cache_data
def _load_wavelet_mst_metrics(universe: str, scale: int) -> pd.DataFrame:
    path = data_results(universe) / f"wavelet_mst_metrics_scale{scale}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_wavelet_mst_metrics(scale: int) -> pd.DataFrame:
    return _load_wavelet_mst_metrics(current_universe(), scale)


@st.cache_data
def _load_anomalies(universe: str) -> pd.DataFrame:
    path = data_processed(universe) / "anomalies.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"]) if path.stat().st_size > 0 else pd.DataFrame()
        return df
    return pd.DataFrame()


def load_anomalies() -> pd.DataFrame:
    return _load_anomalies(current_universe())


@st.cache_data
def _load_rolling_market_stats_precomputed(universe: str, window: int) -> pd.DataFrame:
    path = data_results(universe) / f"rolling_market_stats_w{window}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()


def load_rolling_market_stats_precomputed(window: int) -> pd.DataFrame:
    return _load_rolling_market_stats_precomputed(current_universe(), window)


@st.cache_data
def _load_rolling_sector_stats_precomputed(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "rolling_sector_stats.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()


def load_rolling_sector_stats_precomputed() -> pd.DataFrame:
    return _load_rolling_sector_stats_precomputed(current_universe())


# ---------------------------------------------------------------------------
# SNN (Spiking Neural Network) loaders
# ---------------------------------------------------------------------------

@st.cache_data
def _load_snn_metrics(universe: str) -> dict:
    """Per-pair + aggregate SNN metrics (macro-F1, Sharpe, fold-level)."""
    path = data_results(universe) / "snn_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_snn_metrics() -> dict:
    return _load_snn_metrics(current_universe())


@st.cache_data
def _load_snn_pair_list(universe: str) -> pd.DataFrame:
    """List of pairs the SNN was trained on (ticker_a, ticker_b, pair_id)."""
    path = data_results(universe) / "snn_pair_list.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_snn_pair_list() -> pd.DataFrame:
    return _load_snn_pair_list(current_universe())


@st.cache_data
def _load_snn_signals(universe: str, pair_id: str) -> pd.DataFrame:
    """Per-pair daily SNN signal: date, zscore, class probabilities, signal labels."""
    path = data_results(universe) / "snn_signals" / f"{pair_id}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


def load_snn_signals(pair_id: str) -> pd.DataFrame:
    return _load_snn_signals(current_universe(), pair_id)


@st.cache_data
def _load_snn_training_history(universe: str) -> pd.DataFrame:
    """Epoch-by-epoch training history of the universal SNN model."""
    path = data_results(universe) / "snn_training_history.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_snn_training_history() -> pd.DataFrame:
    return _load_snn_training_history(current_universe())


@st.cache_data
def _load_snn_raster_sample(universe: str) -> pd.DataFrame:
    """Spike raster from the sample pair (day, timestep, neuron, spike)."""
    path = data_results(universe) / "snn_spike_raster_sample.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_snn_raster_sample() -> pd.DataFrame:
    return _load_snn_raster_sample(current_universe())


@st.cache_data
def _load_snn_membrane_sample(universe: str) -> pd.DataFrame:
    """Membrane-potential V(t) for the sample pair's output-layer neurons."""
    path = data_results(universe) / "snn_membrane_sample.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_snn_membrane_sample() -> pd.DataFrame:
    return _load_snn_membrane_sample(current_universe())


# ---------------------------------------------------------------------------
# Information-theory layer (Phase 3 mutable-candy)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading mutual information matrix…")
def _load_mi_matrix(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "mi_matrix.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_mi_matrix() -> pd.DataFrame:
    return _load_mi_matrix(current_universe())


@st.cache_data
def _load_mi_gaussian_matrix(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "mi_gaussian_matrix.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_mi_gaussian_matrix() -> pd.DataFrame:
    return _load_mi_gaussian_matrix(current_universe())


@st.cache_data
def _load_mi_nonlinear_excess(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "mi_nonlinear_excess.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_mi_nonlinear_excess() -> pd.DataFrame:
    return _load_mi_nonlinear_excess(current_universe())


@st.cache_data
def _load_mi_nonlinear_excess_top(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "mi_nonlinear_excess_top.csv"
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["ticker_a", "ticker_b", "nonlinear_excess"])
    return pd.DataFrame(columns=["ticker_a", "ticker_b", "nonlinear_excess"])


def load_mi_nonlinear_excess_top() -> pd.DataFrame:
    return _load_mi_nonlinear_excess_top(current_universe())


@st.cache_data
def _load_rolling_info_theory(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "rolling_info_theory.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_rolling_info_theory() -> pd.DataFrame:
    return _load_rolling_info_theory(current_universe())


@st.cache_data
def _load_regime_kl(universe: str) -> list:
    path = data_results(universe) / "regime_kl.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def load_regime_kl() -> list:
    return _load_regime_kl(current_universe())


@st.cache_data
def _load_it_summary(universe: str) -> dict:
    path = data_results(universe) / "it_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_it_summary() -> dict:
    return _load_it_summary(current_universe())


@st.cache_data
def _load_entropy_rate_signs(universe: str) -> pd.DataFrame:
    path = data_results(universe) / "entropy_rate_signs.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_entropy_rate_signs() -> pd.DataFrame:
    return _load_entropy_rate_signs(current_universe())


# ---------------------------------------------------------------------------
# Event markers
# ---------------------------------------------------------------------------

DASH_OPTIONS: dict[str, str] = {
    "Dashed":    "dash",
    "Solid":     "solid",
    "Dotted":    "dot",
    "Long Dash": "longdash",
    "Dash-Dot":  "dashdot",
}
_DASH_LABELS: dict[str, str] = {v: k for k, v in DASH_OPTIONS.items()}

_DEFAULT_EVENTS_CACHE: Optional[list] = None


def _get_default_events() -> list:
    global _DEFAULT_EVENTS_CACHE
    if _DEFAULT_EVENTS_CACHE is None:
        try:
            from src.rolling_correlation import DEFAULT_EVENTS
            _DEFAULT_EVENTS_CACHE = DEFAULT_EVENTS
        except Exception:
            _DEFAULT_EVENTS_CACHE = []
    return _DEFAULT_EVENTS_CACHE


def draw_event_markers(
    fig: go.Figure,
    show_defaults: bool,
    custom_events: list,
    date_min,
    date_max,
) -> None:
    """Render vertical event-marker lines + annotations onto a plotly figure."""
    events: list[dict] = []
    if show_defaults:
        for ev in _get_default_events():
            events.append({
                "date": ev["date"], "label": ev["label"],
                "color": "#E53935", "dash": "dash", "width": 1.0,
            })
    events.extend(custom_events)

    for ev in events:
        ev_date = pd.Timestamp(ev["date"])
        if date_min <= ev_date <= date_max:
            fig.add_shape(
                type="line", x0=ev_date, x1=ev_date,
                y0=0, y1=1, yref="paper",
                line=dict(dash=ev["dash"], color=ev["color"], width=ev["width"]),
                opacity=0.75,
            )
            fig.add_annotation(
                x=ev_date, y=1, yref="paper",
                text=ev["label"], showarrow=False,
                font=dict(size=9, color=ev["color"]), yshift=10,
            )


def event_marker_manager_ui(
    key_prefix: str,
    min_date,
    max_date,
) -> tuple[bool, list]:
    """Render an 'Event Markers' popover scoped to key_prefix.

    Returns (show_defaults, custom_events) from session state.
    Each page/section gets its own independent event set via key_prefix.
    """
    _key_show = f"{key_prefix}_show_defaults"
    _key_evs  = f"{key_prefix}_custom_events"

    if _key_show not in st.session_state:
        st.session_state[_key_show] = True
    if _key_evs not in st.session_state:
        st.session_state[_key_evs] = []

    with st.popover("Event Markers", icon=":material/flag:"):
        st.session_state[_key_show] = st.checkbox(
            "Show default macro events (COVID-19, Russia-Ukraine War, Turkey Earthquakes)",
            value=st.session_state[_key_show],
            key=f"{key_prefix}_chk_defaults",
        )

        evs: list = st.session_state[_key_evs]
        if evs:
            st.markdown("**Custom events**")
            # NOTE: no st.columns here — event_marker_manager_ui is called
            # from inside dashboard tabs which already have column ancestors,
            # and the wrapping st.popover doesn't reset that. Streamlit 1.41+
            # rejects st.columns 2 levels deep. Render the label + remove
            # button inline (the X glyph wraps to a new line if needed).
            for _i, _cev in enumerate(evs):
                _style_label = _DASH_LABELS.get(_cev["dash"], _cev["dash"])
                st.markdown(
                    f"<span style='color:{_cev['color']}'>▌</span> "
                    f"**{_cev['label']}** &nbsp; {_cev['date']} &nbsp;·&nbsp; "
                    f"{_style_label}, {_cev['width']}px",
                    unsafe_allow_html=True,
                )
                if st.button("✕ Remove", key=f"{key_prefix}_rm_{_i}", help="Remove this marker"):
                    st.session_state[_key_evs].pop(_i)
                    st.rerun()

        st.markdown("**Add a new event marker**")
        with st.form(f"{key_prefix}_add_event_form", clear_on_submit=True):
            # Vertically stacked (no st.columns inside the popover form) for
            # the same 1.41+ nesting reason — the popover is already inside
            # the dashboard's column tree, so columns here would be 2 deep.
            _ev_date = st.date_input(
                "Date",
                value=min(pd.Timestamp.today().date(), max_date),
                min_value=min_date,
                max_value=max_date,
                key=f"{key_prefix}_new_ev_date",
            )
            _ev_label = st.text_input(
                "Caption / label", value="",
                placeholder="e.g. Fed Rate Hike",
                key=f"{key_prefix}_new_ev_label",
            )
            _ev_color = st.color_picker(
                "Line color", value="#E53935",
                key=f"{key_prefix}_new_ev_color",
            )
            _ev_style = st.selectbox(
                "Line style", list(DASH_OPTIONS.keys()), index=0,
                key=f"{key_prefix}_new_ev_style",
            )
            _ev_width = st.slider(
                "Line thickness (px)", min_value=0.5, max_value=5.0,
                value=1.5, step=0.5,
                key=f"{key_prefix}_new_ev_width",
            )
            if st.form_submit_button("Add Event Marker"):
                st.session_state[_key_evs].append({
                    "date":  str(_ev_date),
                    "label": _ev_label.strip() or str(_ev_date),
                    "color": _ev_color,
                    "dash":  DASH_OPTIONS[_ev_style],
                    "width": _ev_width,
                })
                st.rerun()

    return st.session_state[_key_show], st.session_state[_key_evs]


# ---------------------------------------------------------------------------
# Data-quality warnings
# ---------------------------------------------------------------------------

def check_ticker_pair_warnings(
    full_returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window_returns: Optional[pd.DataFrame] = None,
    coverage_df: Optional[pd.DataFrame] = None,
    anomaly_threshold: float = 0.30,
) -> list[dict]:
    """Return a list of data-quality issues for a ticker pair.

    Each item has keys:
        level   : "error" | "warning" | "info"
        message : markdown string
    """
    issues: list[dict] = []
    ret = window_returns if window_returns is not None else full_returns

    for ticker in (ticker_a, ticker_b):
        if ticker not in full_returns.columns:
            issues.append({
                "level": "error",
                "message": f"**{ticker}** is not found in the processed dataset.",
            })
            continue

        series = full_returns[ticker].dropna()

        # ── Coverage check
        if coverage_df is not None and not coverage_df.empty:
            row = coverage_df[coverage_df["ticker"] == ticker]
            if not row.empty:
                cov = float(row["coverage_pct"].values[0])
                if cov < 0.90:
                    issues.append({
                        "level": "warning",
                        "message": (
                            f"**{ticker}** has only **{cov:.1%}** data coverage "
                            f"(threshold: 90%). Statistics may be unreliable."
                        ),
                    })

        # ── Anomalous returns (dividend / split / data artifact)
        extreme = series[series.abs() > anomaly_threshold]
        if not extreme.empty:
            issues.append({
                "level": "warning",
                "message": (
                    f"**{ticker}** has **{len(extreme)}** day(s) with |return| > "
                    f"{anomaly_threshold:.0%} (max: {extreme.abs().max():.1%}). "
                    f"These may be stock splits, special dividends, or data errors "
                    f"and could distort correlation estimates."
                ),
            })

    # ── Both tickers available — cross-checks
    if ticker_a in full_returns.columns and ticker_b in full_returns.columns:
        both_full = full_returns[[ticker_a, ticker_b]].dropna()
        a_days    = full_returns[ticker_a].dropna()
        b_days    = full_returns[ticker_b].dropna()
        mismatch  = (len(a_days) - len(both_full)) + (len(b_days) - len(both_full))

        if mismatch > 5:
            issues.append({
                "level": "info",
                "message": (
                    f"**{ticker_a}** and **{ticker_b}** have **{mismatch}** non-overlapping "
                    f"trading days (different listing histories or trading halts). "
                    f"Correlation is computed on **{len(both_full)}** common dates."
                ),
            })

        # ── Short overlap
        both_win = ret[[ticker_a, ticker_b]].dropna() if (
            ticker_a in ret.columns and ticker_b in ret.columns
        ) else both_full
        if len(both_win) < 60:
            issues.append({
                "level": "warning",
                "message": (
                    f"Only **{len(both_win)}** overlapping observations in the selected window — "
                    f"correlation estimates are unreliable with fewer than 60 data points."
                ),
            })

        # ── Extreme correlation
        if len(both_full) >= 30:
            try:
                rho = both_full.corr().iloc[0, 1]
                if abs(rho) > 0.97:
                    issues.append({
                        "level": "info",
                        "message": (
                            f"Full-period Pearson ρ is extremely high (**{rho:.4f}**). "
                            f"These stocks may be the same entity, a subsidiary, or both track "
                            f"the same index segment."
                        ),
                    })
                elif rho < -0.4:
                    issues.append({
                        "level": "info",
                        "message": (
                            f"Full-period Pearson ρ is notably negative (**{rho:.4f}**). "
                            f"These stocks tend to move in opposite directions — potentially "
                            f"useful for portfolio hedging."
                        ),
                    })
            except Exception:
                pass

    return issues


def render_warnings(issues: list[dict]) -> None:
    """Render data-quality issues in Streamlit."""
    for issue in issues:
        level = issue["level"]
        msg   = issue["message"]
        if level == "error":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.info(msg)
