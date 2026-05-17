"""Chart theme system — dataclass, presets, session-state helpers, sidebar UI.

Provides researcher-grade customization for all 23 Plotly charts in the dashboard.
Every chart reads its style from the active theme via get_active_theme().
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, asdict
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' + alpha -> 'rgba(R,G,B,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# ChartTheme dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChartTheme:
    """Immutable snapshot of all visual settings for charts."""

    # -- Typography --------------------------------------------------------
    font_family: str = "Inter, -apple-system, sans-serif"
    font_size: int = 12
    font_color: str = "#2B2D42"
    title_font_size: int = 14

    # -- Primary palette ---------------------------------------------------
    color_primary: str = "#4361EE"
    color_secondary: str = "#E63946"
    color_tertiary: str = "#2EC4B6"
    color_muted: str = "#8D99AE"
    color_bg: str = "#FAFBFC"

    # -- Lines -------------------------------------------------------------
    line_width: float = 2.2
    line_width_secondary: float = 1.8
    line_width_thin: float = 1.2

    # -- Axes & Grid -------------------------------------------------------
    axis_linewidth: float = 1.0
    show_gridlines: bool = True
    show_zeroline: bool = False

    # -- Layout ------------------------------------------------------------
    paper_bgcolor: str = "rgba(0,0,0,0)"
    plot_bgcolor: str = "rgba(0,0,0,0)"
    legend_orientation: str = "h"          # "h" | "v" | "hidden"
    legend_font_size: int = 11
    margin: dict = field(default_factory=lambda: dict(l=0, r=0, t=10, b=0))

    # -- Heatmap -----------------------------------------------------------
    heatmap_colorscale: str = "RdBu_r"

    # -- Export defaults ---------------------------------------------------
    export_width: int = 1200
    export_height: int = 800
    export_scale: float = 2.0

    # -- Derived colors (auto-computed from primary palette) ----------------
    @property
    def color_positive_fill(self) -> str:
        return hex_to_rgba(self.color_primary, 0.12)

    @property
    def color_negative_fill(self) -> str:
        return hex_to_rgba(self.color_secondary, 0.12)

    @property
    def color_band_fill(self) -> str:
        return hex_to_rgba(self.color_muted, 0.12)

    @property
    def axis_gridcolor(self) -> str:
        return hex_to_rgba(self.color_muted, 0.15)

    @property
    def axis_linecolor(self) -> str:
        return hex_to_rgba(self.color_muted, 0.30)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

DEFAULT_THEME = ChartTheme()

PUBLICATION_THEME = ChartTheme(
    font_family="Times New Roman, serif",
    font_size=13,
    font_color="#000000",
    title_font_size=15,
    color_primary="#1B1B1B",
    color_secondary="#555555",
    color_tertiary="#888888",
    color_muted="#AAAAAA",
    color_bg="#FFFFFF",
    line_width=2.0,
    line_width_secondary=1.6,
    line_width_thin=1.0,
    axis_linewidth=1.5,
    show_gridlines=True,
    show_zeroline=False,
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    legend_orientation="h",
    legend_font_size=11,
    heatmap_colorscale="RdBu_r",
    export_scale=3.0,
)

PRESENTATION_THEME = ChartTheme(
    font_family="Helvetica Neue, Helvetica, sans-serif",
    font_size=16,
    font_color="#1A1A2E",
    title_font_size=20,
    color_primary="#4361EE",
    color_secondary="#E63946",
    color_tertiary="#2EC4B6",
    color_muted="#8D99AE",
    color_bg="#F8F9FC",
    line_width=3.0,
    line_width_secondary=2.5,
    line_width_thin=1.8,
    axis_linewidth=1.2,
    legend_font_size=14,
    margin=field(default_factory=lambda: dict(l=10, r=10, t=50, b=10)),
    export_width=1920,
    export_height=1080,
    export_scale=2.0,
)
# Fix: dataclass default_factory doesn't work in constructor calls, set margin after
PRESENTATION_THEME.margin = dict(l=10, r=10, t=50, b=10)

HIGH_CONTRAST_THEME = ChartTheme(
    font_family="Arial, sans-serif",
    font_size=13,
    font_color="#000000",
    title_font_size=16,
    # Okabe-Ito palette (colorblind-safe)
    color_primary="#0072B2",
    color_secondary="#D55E00",
    color_tertiary="#009E73",
    color_muted="#666666",
    color_bg="#FFFFFF",
    line_width=2.5,
    line_width_secondary=2.0,
    line_width_thin=1.5,
    axis_linewidth=1.5,
    show_gridlines=True,
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    legend_font_size=12,
    export_scale=3.0,
)

PRESETS: dict[str, ChartTheme] = {
    "Default": DEFAULT_THEME,
    "Publication": PUBLICATION_THEME,
    "Presentation": PRESENTATION_THEME,
    "High Contrast": HIGH_CONTRAST_THEME,
}


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

_SESSION_KEY = "_chart_theme"


def get_active_theme() -> ChartTheme:
    """Return the currently active theme from session state."""
    return st.session_state.get(_SESSION_KEY, DEFAULT_THEME)


def set_active_theme(theme: ChartTheme) -> None:
    """Store the given theme in session state."""
    st.session_state[_SESSION_KEY] = theme


# ---------------------------------------------------------------------------
# Conversion helpers (theme -> Plotly-compatible dicts)
# ---------------------------------------------------------------------------

def theme_to_layout(theme: ChartTheme) -> dict[str, Any]:
    """Convert a ChartTheme into a Plotly layout dict."""
    layout: dict[str, Any] = dict(
        font=dict(family=theme.font_family, size=theme.font_size, color=theme.font_color),
        paper_bgcolor=theme.paper_bgcolor,
        plot_bgcolor=theme.plot_bgcolor,
        margin=dict(theme.margin),  # copy
        xaxis=dict(
            gridcolor=theme.axis_gridcolor if theme.show_gridlines else "rgba(0,0,0,0)",
            zeroline=theme.show_zeroline,
            linecolor=theme.axis_linecolor,
            linewidth=theme.axis_linewidth,
        ),
        yaxis=dict(
            gridcolor=theme.axis_gridcolor if theme.show_gridlines else "rgba(0,0,0,0)",
            zeroline=theme.show_zeroline,
            linecolor=theme.axis_linecolor,
            linewidth=theme.axis_linewidth,
        ),
    )
    if theme.legend_orientation == "hidden":
        layout["showlegend"] = False
    else:
        layout["legend"] = dict(
            orientation=theme.legend_orientation,
            yanchor="bottom",
            y=1.02,
            font=dict(size=theme.legend_font_size),
        )
    return layout


def theme_to_colors(theme: ChartTheme) -> dict[str, str]:
    """Convert theme palette into the COLORS dict format used by chart-building code."""
    return {
        "primary": theme.color_primary,
        "secondary": theme.color_secondary,
        "tertiary": theme.color_tertiary,
        "muted": theme.color_muted,
        "positive": theme.color_positive_fill,
        "negative": theme.color_negative_fill,
        "band": theme.color_band_fill,
        "bg": theme.color_bg,
        "heatmap_cs": theme.heatmap_colorscale,
    }


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------

# Storable fields that the UI exposes (excludes derived properties)
_FONT_OPTIONS = [
    "Inter, -apple-system, sans-serif",
    "Times New Roman, serif",
    "Helvetica Neue, Helvetica, sans-serif",
    "Arial, sans-serif",
    "Courier New, monospace",
]

_HEATMAP_SCALES = [
    "RdBu_r", "RdBu", "RdYlBu_r", "Viridis", "Plasma", "Inferno",
    "Cividis", "Blues", "Reds", "Greys",
]

_BG_OPTIONS = {
    "Transparent": "rgba(0,0,0,0)",
    "White": "#FFFFFF",
    "Light Grey": "#F5F5F5",
    "Dark": "#1A1C2E",
}


def render_theme_sidebar() -> None:
    """Render the full chart-settings panel inside st.sidebar."""
    st.markdown("### Chart Settings")

    theme = get_active_theme()

    # ── Preset selector ──────────────────────────────────────────────────
    preset_names = list(PRESETS.keys()) + ["Custom"]
    # Detect if current theme matches a preset
    current_preset = "Custom"
    for name, preset in PRESETS.items():
        if _themes_match(theme, preset):
            current_preset = name
            break

    selected_preset = st.selectbox(
        "Theme preset",
        preset_names,
        index=preset_names.index(current_preset),
        key="_theme_preset",
    )

    if selected_preset != "Custom" and selected_preset != current_preset:
        theme = deepcopy(PRESETS[selected_preset])
        set_active_theme(theme)
        st.rerun()

    # ── Fine-tune controls ───────────────────────────────────────────────

    with st.expander("Colors", expanded=False):
        new_primary = st.color_picker("Primary", theme.color_primary, key="_cp_primary")
        new_secondary = st.color_picker("Secondary", theme.color_secondary, key="_cp_secondary")
        new_tertiary = st.color_picker("Tertiary", theme.color_tertiary, key="_cp_tertiary")
        new_muted = st.color_picker("Muted / Grid", theme.color_muted, key="_cp_muted")
        new_heatmap = st.selectbox(
            "Heatmap colorscale", _HEATMAP_SCALES,
            index=_HEATMAP_SCALES.index(theme.heatmap_colorscale)
            if theme.heatmap_colorscale in _HEATMAP_SCALES else 0,
            key="_cp_heatmap",
        )

    with st.expander("Typography", expanded=False):
        new_font = st.selectbox(
            "Font family", _FONT_OPTIONS,
            index=_FONT_OPTIONS.index(theme.font_family)
            if theme.font_family in _FONT_OPTIONS else 0,
            key="_cp_font",
        )
        new_font_size = st.slider("Base font size", 8, 20, theme.font_size, key="_cp_fsize")
        new_title_size = st.slider("Title font size", 10, 24, theme.title_font_size, key="_cp_tsize")
        new_font_color = st.color_picker("Font color", theme.font_color, key="_cp_fcolor")

    with st.expander("Lines & Axes", expanded=False):
        new_lw = st.slider("Primary line width", 0.5, 5.0, theme.line_width, 0.1, key="_cp_lw")
        new_lw2 = st.slider("Secondary line width", 0.5, 5.0, theme.line_width_secondary, 0.1, key="_cp_lw2")
        new_grid = st.checkbox("Show gridlines", theme.show_gridlines, key="_cp_grid")
        new_ax_lw = st.slider("Axis line width", 0.0, 3.0, theme.axis_linewidth, 0.1, key="_cp_axlw")
        new_zeroline = st.checkbox("Show zero line", theme.show_zeroline, key="_cp_zero")

    with st.expander("Layout", expanded=False):
        legend_opts = {"Horizontal": "h", "Vertical": "v", "Hidden": "hidden"}
        legend_labels = list(legend_opts.keys())
        current_legend = next(
            (k for k, v in legend_opts.items() if v == theme.legend_orientation), "Horizontal"
        )
        new_legend = legend_opts[st.radio(
            "Legend orientation", legend_labels,
            index=legend_labels.index(current_legend),
            key="_cp_legend", horizontal=True,
        )]

        bg_labels = list(_BG_OPTIONS.keys())
        current_bg = next(
            (k for k, v in _BG_OPTIONS.items() if v == theme.paper_bgcolor), "Transparent"
        )
        new_bg = _BG_OPTIONS[st.selectbox(
            "Background", bg_labels,
            index=bg_labels.index(current_bg),
            key="_cp_bg",
        )]

    with st.expander("Export Defaults", expanded=False):
        # Bounds mirror app/chart_export.py:_EXPORT_{W,H}_{MIN,MAX} so a theme
        # saved with a small export size cannot pre-fill the inputs below the
        # min and trigger StreamlitValueBelowMinError.
        _W_MIN, _W_MAX = 200, 4000
        _H_MIN, _H_MAX = 200, 3000
        _w_default = max(_W_MIN, min(_W_MAX, int(theme.export_width)))
        _h_default = max(_H_MIN, min(_H_MAX, int(theme.export_height)))
        new_exp_w = st.number_input("Width (px)", _W_MIN, _W_MAX, _w_default, 100, key="_cp_ew")
        new_exp_h = st.number_input("Height (px)", _H_MIN, _H_MAX, _h_default, 100, key="_cp_eh")
        new_exp_s = st.slider("Scale", 1.0, 4.0, theme.export_scale, 0.5, key="_cp_es")

    # ── Build updated theme from current widget values ────────────────────
    updated = ChartTheme(
        font_family=new_font,
        font_size=new_font_size,
        font_color=new_font_color,
        title_font_size=new_title_size,
        color_primary=new_primary,
        color_secondary=new_secondary,
        color_tertiary=new_tertiary,
        color_muted=new_muted,
        color_bg=theme.color_bg,        # derived from bg choice
        line_width=new_lw,
        line_width_secondary=new_lw2,
        line_width_thin=theme.line_width_thin,
        axis_linewidth=new_ax_lw,
        show_gridlines=new_grid,
        show_zeroline=new_zeroline,
        paper_bgcolor=new_bg,
        plot_bgcolor=new_bg,
        legend_orientation=new_legend,
        legend_font_size=theme.legend_font_size,
        margin=dict(theme.margin),
        heatmap_colorscale=new_heatmap,
        export_width=new_exp_w,
        export_height=new_exp_h,
        export_scale=new_exp_s,
    )

    if not _themes_match(updated, theme):
        set_active_theme(updated)

    # ── Reset button ─────────────────────────────────────────────────────
    if st.button("Reset to Default", key="_theme_reset", use_container_width=True):
        set_active_theme(deepcopy(DEFAULT_THEME))
        st.rerun()


def _themes_match(a: ChartTheme, b: ChartTheme) -> bool:
    """Compare two themes by their stored fields (ignores derived properties)."""
    for f in fields(ChartTheme):
        if getattr(a, f.name) != getattr(b, f.name):
            return False
    return True
