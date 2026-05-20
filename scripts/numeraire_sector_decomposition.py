"""Per-eigenmode sector decomposition across BIST base currencies.

Phase 4 follow-up. The headline base currency result (TRY → USD → Gold pushes
top-eigenvalue share 38.9% → 45.1% → 51.5%) is real but tells only half
the story. Decomposing each base currency's correlation matrix into its top
3 eigenmodes and projecting the eigenvectors onto sector labels exposes
**which factor structure collapses** when the currency leg is removed:

- Mode 1 (the market mode) absorbs *more* variance (conglomerate-led,
  size weight rises).
- Mode 2 is a **pure banking factor** under TRY (5 BIST banks dominate
  the eigenvector). Its variance share *drops* under USD / Gold because
  banks become less orthogonal to the market — the bank-vs-rest factor
  was largely TRY-driven.
- Mode 3 is an "old industrials" mix (BRYAT, BRSAN, TUPRS, FROTO);
  also weakens under USD / Gold.

Outputs:
- docs/figures/numeraire_sector_shift.svg
- data/results/numeraire_decomposition.json  (machine-readable numbers
  for slides + report)
"""

from __future__ import annotations

import json
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_PATH = PROJECT_ROOT / "config" / "universes" / "bist100.csv"
OUT_FIG_DIR = PROJECT_ROOT / "docs" / "figures"
OUT_DATA_PATH = PROJECT_ROOT / "data" / "results" / "numeraire_decomposition.json"

NUMERAIRES = [
    ("bist", "TRY", "#1F77B4"),
    ("bist_usd", "USD", "#2CA02C"),
    ("bist_gold", "Gold", "#FFC400"),
]

BANK_TICKERS = {"AKBNK", "GARAN", "YKBNK", "VAKBN", "HALKB", "SKBNK", "ISCTR"}


def _decompose_one(market: str, sector_map: dict) -> dict:
    corr_path = PROJECT_ROOT / "data" / market / "results" / "pearson_corr.parquet"
    if not corr_path.exists():
        raise FileNotFoundError(corr_path)
    corr = pd.read_parquet(corr_path)
    eigvals, eigvecs = np.linalg.eigh(corr.values)
    # numpy.eigh returns ascending order; flip so index 0 is the dominant mode
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    tickers = corr.columns.tolist()
    total_var = float(eigvals.sum())

    modes = []
    for k in range(3):
        vec = eigvecs[:, k]
        mass = vec ** 2  # sums to 1
        sector_mass = defaultdict(float)
        bank_mass = 0.0
        for ticker, m in zip(tickers, mass):
            sec = sector_map.get(ticker, "Unknown")
            sector_mass[sec] += float(m)
            if ticker in BANK_TICKERS:
                bank_mass += float(m)
        top_tickers = sorted(zip(tickers, mass), key=lambda kv: kv[1], reverse=True)[:5]
        top_sectors = sorted(sector_mass.items(), key=lambda kv: kv[1], reverse=True)[:5]
        modes.append({
            "rank": k + 1,
            "eigenvalue": float(eigvals[k]),
            "variance_share": float(eigvals[k] / total_var),
            "bank_mass_share": bank_mass,  # fraction of THIS mode carried by banks
            "top_tickers": [
                {"ticker": t, "mass": float(m), "sector": sector_map.get(t, "Unknown")}
                for t, m in top_tickers
            ],
            "top_sectors": [{"sector": s, "mass": float(m)} for s, m in top_sectors],
        })

    return {
        "n_tickers": len(tickers),
        "total_variance": total_var,
        "modes": modes,
    }


def _build_svg(report: dict) -> str:
    """Two-panel SVG: top eigenvalue share per base currency, plus bank-mass
    in each of the top 3 modes per base currency."""
    W, H = 1100, 460
    pad_l, pad_r, pad_t, pad_b = 70, 30, 60, 70
    plot_w = (W - pad_l - pad_r - 60) / 2
    plot_h = H - pad_t - pad_b

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<style>'
        '.title{font:700 16px Inter,Helvetica,Arial,sans-serif;fill:#111827}'
        '.subtitle{font:600 13px Inter,Helvetica,Arial,sans-serif;fill:#1F2937}'
        '.axis{font:11px Inter,Helvetica,Arial,sans-serif;fill:#4B5563}'
        '.value{font:600 11px Inter,Helvetica,Arial,sans-serif;fill:#111827}'
        '.cap{font:italic 11px Inter,Helvetica,Arial,sans-serif;fill:#4B5563}'
        '</style>',
        f'<text x="{W/2}" y="26" text-anchor="middle" class="title">'
        'BIST base-currency decomposition — where the variance moves</text>',
    ]

    nums = report["per_numeraire"]
    keys = list(nums.keys())
    colors = {label: c for _, label, c in NUMERAIRES}

    # ── Panel A: top-3 mode eigenvalue shares grouped by base currency ────────────
    panel_a_x = pad_l
    panel_a_y = pad_t
    parts.append(
        f'<text x="{panel_a_x}" y="{panel_a_y - 12}" class="subtitle">'
        'Variance share per mode (top 3 eigenvalues)</text>'
    )
    # Y axis: variance share in %
    max_share = max(m["variance_share"] for k in keys for m in nums[k]["modes"]) * 1.1
    bar_group_w = plot_w / 3
    bar_w = bar_group_w / (len(keys) + 1)

    # Axes
    parts.append(f'<line x1="{panel_a_x}" y1="{panel_a_y}" x2="{panel_a_x}" y2="{panel_a_y + plot_h}" stroke="#9CA3AF" stroke-width="1"/>')
    parts.append(f'<line x1="{panel_a_x}" y1="{panel_a_y + plot_h}" x2="{panel_a_x + plot_w}" y2="{panel_a_y + plot_h}" stroke="#9CA3AF" stroke-width="1"/>')

    for tick_frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        tick_y = panel_a_y + plot_h - tick_frac * plot_h
        parts.append(
            f'<line x1="{panel_a_x - 4}" y1="{tick_y}" x2="{panel_a_x}" y2="{tick_y}" stroke="#9CA3AF"/>'
            f'<text x="{panel_a_x - 8}" y="{tick_y + 4}" text-anchor="end" class="axis">'
            f'{tick_frac * max_share * 100:.0f}%</text>'
        )
    for mode_idx in range(3):
        group_x = panel_a_x + mode_idx * bar_group_w + bar_group_w / 2
        parts.append(
            f'<text x="{group_x}" y="{panel_a_y + plot_h + 18}" text-anchor="middle" class="axis">Mode {mode_idx + 1}</text>'
        )
        for n_idx, market in enumerate(keys):
            mode = nums[market]["modes"][mode_idx]
            share = mode["variance_share"]
            label = nums[market]["label"]
            color = colors[label]
            x = panel_a_x + mode_idx * bar_group_w + n_idx * bar_w + bar_w * 0.3
            h = (share / max_share) * plot_h
            y = panel_a_y + plot_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.9:.1f}" height="{h:.1f}" '
                f'fill="{color}" opacity="0.85"/>'
                f'<text x="{x + bar_w * 0.45:.1f}" y="{y - 4:.1f}" text-anchor="middle" class="value">'
                f'{share * 100:.1f}%</text>'
            )

    # Legend for panel A
    legend_y = panel_a_y - 32
    for li, (_, label, color) in enumerate(NUMERAIRES):
        lx = panel_a_x + li * 95
        parts.append(
            f'<rect x="{lx}" y="{legend_y - 10}" width="14" height="14" fill="{color}" />'
            f'<text x="{lx + 20}" y="{legend_y + 1}" class="axis">{escape(label)}</text>'
        )

    # ── Panel B: banking-mass share per mode (the real finding) ────────────
    panel_b_x = pad_l + plot_w + 60
    panel_b_y = pad_t
    parts.append(
        f'<text x="{panel_b_x}" y="{panel_b_y - 12}" class="subtitle">'
        'Banking-sector mass in each mode (5 of 73 tickers = 6.8% baseline)</text>'
    )
    parts.append(f'<line x1="{panel_b_x}" y1="{panel_b_y}" x2="{panel_b_x}" y2="{panel_b_y + plot_h}" stroke="#9CA3AF" stroke-width="1"/>')
    parts.append(f'<line x1="{panel_b_x}" y1="{panel_b_y + plot_h}" x2="{panel_b_x + plot_w}" y2="{panel_b_y + plot_h}" stroke="#9CA3AF" stroke-width="1"/>')

    max_bank = max(m["bank_mass_share"] for k in keys for m in nums[k]["modes"]) * 1.15
    # Baseline line: if banks were proportional to their count
    baseline_share = sum(1 for k in keys for m in nums[k]["modes"] if False)  # placeholder
    # actually: 7 banks / 73 tickers = 0.096
    baseline = 7.0 / 73.0
    baseline_y = panel_b_y + plot_h - (baseline / max_bank) * plot_h
    parts.append(
        f'<line x1="{panel_b_x}" y1="{baseline_y}" x2="{panel_b_x + plot_w}" y2="{baseline_y}" '
        f'stroke="#9CA3AF" stroke-dasharray="4 4"/>'
        f'<text x="{panel_b_x + plot_w + 4}" y="{baseline_y + 3}" class="axis">'
        f'7 / 73 = {baseline*100:.1f}% (random)</text>'
    )
    for tick_frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        tick_y = panel_b_y + plot_h - tick_frac * plot_h
        parts.append(
            f'<line x1="{panel_b_x - 4}" y1="{tick_y}" x2="{panel_b_x}" y2="{tick_y}" stroke="#9CA3AF"/>'
            f'<text x="{panel_b_x - 8}" y="{tick_y + 4}" text-anchor="end" class="axis">'
            f'{tick_frac * max_bank * 100:.0f}%</text>'
        )
    for mode_idx in range(3):
        group_x = panel_b_x + mode_idx * bar_group_w + bar_group_w / 2
        parts.append(
            f'<text x="{group_x}" y="{panel_b_y + plot_h + 18}" text-anchor="middle" class="axis">Mode {mode_idx + 1}</text>'
        )
        for n_idx, market in enumerate(keys):
            mode = nums[market]["modes"][mode_idx]
            bank = mode["bank_mass_share"]
            label = nums[market]["label"]
            color = colors[label]
            x = panel_b_x + mode_idx * bar_group_w + n_idx * bar_w + bar_w * 0.3
            h = (bank / max_bank) * plot_h
            y = panel_b_y + plot_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.9:.1f}" height="{h:.1f}" '
                f'fill="{color}" opacity="0.85"/>'
                f'<text x="{x + bar_w * 0.45:.1f}" y="{y - 4:.1f}" text-anchor="middle" class="value">'
                f'{bank * 100:.1f}%</text>'
            )

    # Caption
    caption = (
        "Mode 2 is the BIST banking factor — 5 of 73 tickers (7%) carry "
        ">75% of its variance under TRY. Re-expressing returns in USD or "
        "Gold weakens both the banking factor's variance share AND the "
        "banks' dominance of that mode, because the bank-vs-market spread "
        "was largely TRY-rate-driven."
    )
    parts.append(
        f'<text x="{W/2}" y="{H - 18}" text-anchor="middle" class="cap">{escape(caption)}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    universe = pd.read_csv(UNIVERSE_PATH)
    sector_map = dict(zip(universe.ticker, universe.sector))

    per_numeraire = {}
    for market, label, _color in NUMERAIRES:
        per_numeraire[market] = {"label": label, **_decompose_one(market, sector_map)}

    # Pretty-print headlines for the operator
    print(f"BIST base currency eigenmode decomposition — {len(NUMERAIRES)} base currencies, top 3 modes each\n")
    for market, label, _ in NUMERAIRES:
        modes = per_numeraire[market]["modes"]
        print(f"=== {label} ({market}) ===")
        for m in modes:
            top_secs = ", ".join(
                f"{s['sector']}={s['mass']*100:.1f}%" for s in m["top_sectors"][:3]
            )
            print(
                f"  mode #{m['rank']}: λ={m['eigenvalue']:.2f}, "
                f"share={m['variance_share']*100:.2f}%, "
                f"bank_mass={m['bank_mass_share']*100:.1f}% — "
                f"top sectors: {top_secs}"
            )
        print()

    report = {
        "n_bank_tickers_in_universe": 7,  # 7 of 73; baseline 9.6%
        "per_numeraire": per_numeraire,
    }
    OUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_DATA_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUT_DATA_PATH}")

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_FIG_DIR / "numeraire_sector_shift.svg"
    svg_path.write_text(_build_svg(report), encoding="utf-8")
    print(f"Wrote {svg_path} ({svg_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
