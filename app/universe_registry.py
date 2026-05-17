"""Universe registry — declares which datasets the dashboard can switch between.

Each :class:`Universe` corresponds to a ``data/<key>/{raw,processed,results}/``
tree produced by ``run_pipeline.py --config <config_path>`` (or
``run_pipeline_eeg.py`` for the EEG universe). ``available_universes()``
filters the registry to only those that have a populated results tree on disk,
so the sidebar selector never offers a broken option.

Capability flags (Phase I): each Universe carries booleans like
``has_pair_trading`` / ``has_snn`` / ``eligible_for_cross_market`` that the
dashboard consults to gate financial-only sections. New non-financial
universes register with the appropriate flags off; downstream code does
``if active_universe.has_X:`` instead of ``if active_universe.key == "eeg":``
to keep the conditional logic centralised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Universe:
    """Description of one switchable dataset in the dashboard.

    The first 5 fields are required; everything else has a sensible
    finance-friendly default so existing BIST / S&P entries stay backward
    compatible without restating every field.
    """

    key: str               # filesystem dir name under data/
    label: str             # display name in the sidebar selector
    short_label: str       # short form for the page-header chip
    config_path: str       # YAML used to (re)run the pipeline
    description: str       # tooltip / caption shown under the selector

    # ── Display terminology (drives axis labels, captions, table headers) ──
    item_label: str = "Ticker"                  # singular noun
    items_label: str = "Tickers"                # plural noun
    sector_label: str = "Sector"                # what "sector" means here
    series_label: str = "Log return"            # cells of log_returns.parquet
    series_units: str = ""                      # "" / "µV"
    network_label: str = "Network Analysis"

    # ── Financial-only attributes (None for non-financial universes) ──
    currency: str | None = None
    index_ticker: str | None = None
    index_display_name: str | None = None

    # ── Capability flags — gate which UI sections render ──
    has_index_series: bool = True
    has_pair_trading: bool = True
    has_snn: bool = True
    has_anomaly_detection: bool = True
    has_validation_report: bool = True
    eligible_for_cross_market: bool = True

    # ── Domain category (rarely used directly; mostly for future logic) ──
    domain: str = "finance"  # "finance" | "neuroscience"

    # ── Optional per-universe sanity-check groups for the clustering tab ──
    # Maps a human label → list of tickers/channels expected to cluster
    # together. e.g. BIST: {"financial sector": ["AKBNK", "GARAN", ...]};
    # EEG: {"central motor (C3/Cz/C4)": ["C3", "Cz", "C4"]}.
    # When None or empty, the clustering tab skips the sanity badge.
    sanity_check_groups: dict[str, list[str]] | None = None


UNIVERSES: dict[str, Universe] = {
    "bist": Universe(
        key="bist",
        label="BIST 100 — Türkiye",
        short_label="BIST 100",
        config_path="config/settings.yaml",
        description=(
            "Borsa İstanbul large-caps; 73 surviving tickers after the 90% "
            "coverage filter. Strongly conglomerate-led market topology."
        ),
        currency="TRY",
        index_ticker="XU100",
        index_display_name="XU100 (Borsa İstanbul 100)",
        sanity_check_groups={
            "Banking sector": [
                "AKBNK", "GARAN", "YKBNK", "VAKBN", "HALKB", "SKBNK", "ISCTR",
            ],
        },
    ),
    "sp500": Universe(
        key="sp500",
        label="S&P 500 — United States",
        short_label="S&P 500",
        config_path="config/settings_sp500.yaml",
        description=(
            "Full S&P 500 (dual-class duplicates dropped); 485 surviving "
            "tickers. Sector-coherent topology with diversified financials, "
            "industrials, and tech hubs."
        ),
        currency="USD",
        index_ticker="^GSPC",
        index_display_name="^GSPC (S&P 500)",
        sanity_check_groups={
            "Mega-cap tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"],
        },
    ),
    "eeg_motor_left_right": Universe(
        key="eeg_motor_left_right",
        label="EEG Motor Imagery — PhysioNet",
        short_label="EEG MI",
        config_path="config/settings_eeg.yaml",
        description=(
            "PhysioNet Motor Movement / Imagery dataset; 64 EEG electrodes "
            "(10-10 montage), 10 subjects pooled, sampled at 160 Hz. "
            "Demonstrates that the MST / RMT / Glasso / wavelet / TE toolkit "
            "transfers cleanly from financial correlation to brain "
            "functional connectivity."
        ),
        # Terminology
        item_label="Channel",
        items_label="Channels",
        sector_label="Anatomical region",
        series_label="Bandpass voltage",
        series_units="µV",
        network_label="Functional Connectivity Network",
        # Capability flags — all financial features off
        has_index_series=False,
        has_pair_trading=False,
        has_snn=False,
        has_anomaly_detection=False,
        has_validation_report=False,
        eligible_for_cross_market=False,
        # Domain
        domain="neuroscience",
        # Sanity checks: classic 10-10 anatomical electrode groups that
        # should cluster together if the methodology is producing sensible
        # brain-network topology.
        sanity_check_groups={
            "Central motor (C3/Cz/C4)":   ["C3", "Cz", "C4"],
            "Occipital (O1/Oz/O2)":       ["O1", "Oz", "O2"],
            "Prefrontal (Fp1/Fpz/Fp2)":   ["Fp1", "Fpz", "Fp2"],
        },
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
