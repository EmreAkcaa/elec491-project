"""Chart export — per-chart download popover with PNG / SVG / PDF support.

Uses Kaleido as the static-image engine behind Plotly's fig.to_image().
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from chart_themes import get_active_theme


# ---------------------------------------------------------------------------
# Core export function
# ---------------------------------------------------------------------------

def export_figure(
    fig: go.Figure,
    fmt: str = "png",
    width: int | None = None,
    height: int | None = None,
    scale: float | None = None,
) -> bytes:
    """Render a Plotly figure to bytes.

    Parameters
    ----------
    fig : plotly Figure
    fmt : 'png', 'svg', or 'pdf'
    width, height, scale : override theme export defaults if provided
    """
    import math
    theme = get_active_theme()
    # Use figure's own dimensions if set and finite, then fall back to theme defaults
    fig_w = fig.layout.width
    fig_h = fig.layout.height
    w = width or (fig_w if fig_w and math.isfinite(fig_w) else None) or theme.export_width
    h = height or (fig_h if fig_h and math.isfinite(fig_h) else None) or theme.export_height
    s = scale or theme.export_scale
    return fig.to_image(format=fmt, width=w, height=h, scale=s, engine="kaleido")


# ---------------------------------------------------------------------------
# Per-chart export popover
# ---------------------------------------------------------------------------

_FORMATS = ["png", "svg", "pdf"]
_SCALES = {"1x": 1.0, "2x": 2.0, "3x": 3.0, "4x": 4.0}


def render_export_popover(
    fig: go.Figure,
    chart_id: str,
    filename_base: str = "chart",
) -> None:
    """Render a small 'Export' popover below a chart with format + resolution controls."""
    # Right-align the export button
    _, col_btn = st.columns([8, 1])
    with col_btn:
        with st.popover(":material/download:", help="Export this chart"):
            theme = get_active_theme()

            fmt = st.selectbox(
                "Format", _FORMATS,
                format_func=str.upper,
                key=f"_exp_fmt_{chart_id}",
            )
            scale_label = st.selectbox(
                "Resolution", list(_SCALES.keys()),
                index=1,  # default 2x
                key=f"_exp_scale_{chart_id}",
            )
            scale = _SCALES[scale_label]

            # Use figure dimensions if set and finite, else theme defaults
            import math
            fig_w = fig.layout.width
            fig_h = fig.layout.height
            default_w = fig_w if fig_w and math.isfinite(fig_w) else theme.export_width
            default_h = fig_h if fig_h and math.isfinite(fig_h) else theme.export_height

            c1, c2 = st.columns(2)
            with c1:
                w = st.number_input(
                    "Width", 400, 4000, int(default_w), 100,
                    key=f"_exp_w_{chart_id}",
                )
            with c2:
                h = st.number_input(
                    "Height", 300, 3000, int(default_h), 100,
                    key=f"_exp_h_{chart_id}",
                )

            filename = f"{filename_base}.{fmt}"

            try:
                img_bytes = export_figure(fig, fmt=fmt, width=w, height=h, scale=scale)
                mime = {
                    "png": "image/png",
                    "svg": "image/svg+xml",
                    "pdf": "application/pdf",
                }[fmt]

                st.download_button(
                    label=f"Download {fmt.upper()}",
                    data=img_bytes,
                    file_name=filename,
                    mime=mime,
                    use_container_width=True,
                    key=f"_exp_dl_{chart_id}_{fmt}",
                )
            except Exception as exc:
                st.error(f"Export failed: {exc}")
                st.caption("Make sure `kaleido` is installed: `pip install kaleido`")


# ---------------------------------------------------------------------------
# Plotly modebar config builder
# ---------------------------------------------------------------------------

def get_plotly_config(chart_id: str = "") -> dict:
    """Build a Plotly config dict with a clean, minimal modebar."""
    theme = get_active_theme()
    return {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "autoScale2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "toggleSpikelines",
        ],
        "toImageButtonOptions": {
            "format": "svg",
            "filename": chart_id or "chart",
            "width": theme.export_width,
            "height": theme.export_height,
            "scale": theme.export_scale,
        },
    }
