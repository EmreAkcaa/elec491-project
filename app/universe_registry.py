"""Universe registry — declares which datasets the dashboard can switch between.

Each :class:`Universe` corresponds to a ``data/<key>/{raw,processed,results}/``
tree produced by ``run_pipeline.py --config <config_path>``. ``available_universes()``
filters the registry to only those that have a populated results tree on disk,
so the sidebar selector never offers a broken option.

EEG is deliberately omitted from the registry: its anatomical-region sectors
and non-stock semantics don't fit the dashboard's stock-centric assumptions,
and the Pair Analysis / SNN sub-tabs would be empty. Adding EEG later is a
one-entry change once those tabs learn to handle non-stock universes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Universe:
    key: str            # filesystem dir name under data/
    label: str          # display name in the sidebar selector
    short_label: str    # short form for the page-header chip
    config_path: str    # YAML used to (re)run the pipeline
    currency: str       # display only
    index_ticker: str   # display only
    description: str    # tooltip / caption


UNIVERSES: dict[str, Universe] = {
    "bist": Universe(
        key="bist",
        label="BIST 100 — Türkiye",
        short_label="BIST 100",
        config_path="config/settings.yaml",
        currency="TRY",
        index_ticker="XU100",
        description=(
            "Borsa İstanbul large-caps; 73 surviving tickers after the 90% "
            "coverage filter. Strongly conglomerate-led market topology."
        ),
    ),
    "sp500": Universe(
        key="sp500",
        label="S&P 500 — United States",
        short_label="S&P 500",
        config_path="config/settings_sp500.yaml",
        currency="USD",
        index_ticker="^GSPC",
        description=(
            "Full S&P 500 (dual-class duplicates dropped); 485 surviving "
            "tickers. Sector-coherent topology with diversified financials, "
            "industrials, and tech hubs."
        ),
    ),
}


def available_universes() -> list[Universe]:
    """Return only universes whose data/<key>/results/ tree is populated.

    Used by the sidebar selector so a half-installed clone never offers a
    universe whose loaders would all return empty.
    """
    out: list[Universe] = []
    for u in UNIVERSES.values():
        meta_path = PROJECT_ROOT / "data" / u.key / "results" / "pipeline_metadata.json"
        if meta_path.exists():
            out.append(u)
    return out


def get_universe(key: str) -> Universe:
    """Look up a Universe by key; fall back to BIST if unknown."""
    return UNIVERSES.get(key, UNIVERSES["bist"])
