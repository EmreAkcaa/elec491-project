"""StoNeCoAl system schematic — sensor → DSP → IT → network → inference.

Phase 6.1 of the mutable-candy rescue. Produces a single SVG of the
engineering chain that answers the supervisor's midterm complaint
("not an EEE project") visually: real sensors feed real DSP feed real
information theory feed real network extraction feed real (spiking)
inference, all in one figure.

Implemented as a hand-written SVG generator so we don't pull matplotlib
into the runtime dep list (we already lean on plotly for the dashboard).

Usage:
    uv run python scripts/draw_system_schematic.py
    # writes docs/figures/system_schematic.svg
"""

from __future__ import annotations

from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "docs" / "figures"


PALETTE = {
    "sensor":   "#4361EE",
    "dsp":      "#3A86FF",
    "stats":    "#7209B7",
    "it":       "#F72585",
    "network":  "#06D6A0",
    "infer":    "#FFC400",
    "ui":       "#9CA3AF",
    "arrow":    "#374151",
    "text":     "#111827",
    "subtext":  "#4B5563",
    "bg_alpha": "22",  # hex alpha suffix for the box fill
}

STAGES = [
    ("SENSORS", "sensor", [
        "Borsa Istanbul order-book aggregates (.IS, daily, 5 yr)",
        "S&P 500 exchange aggregates (NYSE/NASDAQ, daily, 5 yr)",
        "64-channel EEG cap (10-10 montage, 160 Hz ADC, PhysioNet BCI2000)",
    ]),
    ("ACQUISITION & PREPROCESSING", "dsp", [
        "yfinance / İş Yatırım pulls (5 yr × ~500 tickers)",
        "MNE-Python EDF+ ingest, bandpass 1–50 Hz, 50 Hz notch, CAR ref (EEG)",
        "log-return · calendar align · 90% coverage filter · anomaly mask",
    ]),
    ("STATISTICAL SIGNAL PROCESSING", "stats", [
        "Pearson + Spearman correlation, rolling-window stats",
        "RMT denoising (Marchenko-Pastur, signal vs noise eigenvalues)",
        "Graphical LASSO (sparse inverse covariance, partial correlation)",
        "Wavelet decomposition (db4, scales 1–7)",
    ]),
    ("INFORMATION THEORY", "it", [
        "Pairwise mutual information (plug-in + closed-form Gaussian baseline)",
        "Effective dimensionality D_eff, joint entropy ΔH = −½ log det Σ",
        "Transfer entropy with circular-block-bootstrap null + BH-FDR",
        "Regime KL divergence between calm and crisis covariances",
    ]),
    ("NETWORK EXTRACTION & VALIDATION", "network", [
        "Mantegna distance d = √(2(1 − ρ))",
        "Minimum Spanning Tree (Kruskal)",
        "Hierarchical clustering (Ward linkage, n_clusters via maxclust)",
        "Sector / anatomical-region validation (ARI, NMI)",
    ]),
    ("NEUROMORPHIC INFERENCE (optional)", "infer", [
        "Σ-Δ-style delta-modulation encoder, population coding",
        "Recurrent LIF SNN (96 hidden, β = 0.92, surrogate gradient)",
        "Target deployment substrates: Intel Loihi 2, IBM TrueNorth, SpiNNaker",
    ]),
    ("VISUALISATION", "ui", [
        "Streamlit dashboard (3 pages × N tabs per universe)",
        "Multi-universe sidebar selector (BIST, S&P, BIST/USD, BIST/Gold, EEG)",
        "Cross-market comparison + base-currency sensitivity sub-tabs",
    ]),
]


# Layout constants (SVG user units = pixels).
PAGE_W = 1100
MARGIN_X = 40
BOX_W = PAGE_W - 2 * MARGIN_X
BOX_H_BASE = 70  # title + first item
LINE_H = 18
ARROW_GAP = 28
TITLE_HEADER_PAD = 50
FOOTER_PAD = 36


def _stage_height(items: list[str]) -> int:
    """Box height grows with number of items so nothing overlaps."""
    return BOX_H_BASE + LINE_H * len(items)


def _stage_block(x: float, y: float, color: str, title: str, items: list[str]) -> str:
    h = _stage_height(items)
    fill = f"{color}{PALETTE['bg_alpha']}"
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{BOX_W}" height="{h}" rx="14" ry="14" '
        f'fill="{fill}" stroke="{color}" stroke-width="1.8" />',
        f'<text x="{x + 18:.1f}" y="{y + 28:.1f}" font-family="Inter, Helvetica, Arial, sans-serif" '
        f'font-size="16" font-weight="700" fill="{color}">{escape(title)}</text>',
    ]
    for i, item in enumerate(items):
        ty = y + 50 + i * LINE_H
        parts.append(
            f'<text x="{x + 22:.1f}" y="{ty:.1f}" font-family="Inter, Helvetica, Arial, sans-serif" '
            f'font-size="12" fill="{PALETTE["text"]}">• {escape(item)}</text>'
        )
    return "\n".join(parts)


def _arrow(x_center: float, y_top: float, y_bot: float) -> str:
    # SVG <defs> registers an arrowhead marker once at the top of the SVG.
    return (
        f'<line x1="{x_center:.1f}" y1="{y_top:.1f}" x2="{x_center:.1f}" y2="{y_bot:.1f}" '
        f'stroke="{PALETTE["arrow"]}" stroke-width="2.4" marker-end="url(#arrowhead)" />'
    )


def build_svg() -> str:
    # Pass 1: total height
    total_h = TITLE_HEADER_PAD + FOOTER_PAD
    for _, _, items in STAGES:
        total_h += _stage_height(items)
    total_h += ARROW_GAP * (len(STAGES) - 1)

    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" '
        f'height="{int(total_h)}" viewBox="0 0 {PAGE_W} {int(total_h)}">'
    )
    defs = (
        '<defs>'
        '<marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE["arrow"]}" />'
        '</marker>'
        '</defs>'
    )
    title = (
        f'<text x="{PAGE_W / 2:.1f}" y="28" text-anchor="middle" '
        f'font-family="Inter, Helvetica, Arial, sans-serif" font-size="18" '
        f'font-weight="700" fill="{PALETTE["text"]}">'
        'StoNeCoAl signal-processing pipeline — sensor to network to inference'
        '</text>'
    )

    parts = [header, defs, title]
    y_cursor = TITLE_HEADER_PAD
    for i, (stage_title, color_key, items) in enumerate(STAGES):
        parts.append(_stage_block(MARGIN_X, y_cursor, PALETTE[color_key], stage_title, items))
        y_cursor += _stage_height(items)
        if i < len(STAGES) - 1:
            mid_x = MARGIN_X + BOX_W / 2
            parts.append(_arrow(mid_x, y_cursor + 4, y_cursor + ARROW_GAP - 4))
            y_cursor += ARROW_GAP

    footer = (
        f'<text x="{PAGE_W / 2:.1f}" y="{int(total_h) - 12}" text-anchor="middle" '
        f'font-family="Inter, Helvetica, Arial, sans-serif" font-size="11" '
        f'font-style="italic" fill="{PALETTE["subtext"]}">'
        'EEE methods on the chain: sampling (160 Hz ADC), DSP (bandpass, notch, CAR), '
        'random matrix theory, sparse precision, multi-resolution wavelet, Shannon/KL '
        'information theory, transfer entropy, spike-coded LIF inference.'
        '</text>'
    )
    parts.append(footer)
    parts.append('</svg>')
    return "\n".join(parts)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_DIR / "system_schematic.svg"
    svg_path.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {svg_path} ({svg_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
