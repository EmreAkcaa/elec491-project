"""Tests for the capability-flag-driven dashboard gating (Phase I).

These tests don't import the Streamlit dashboard itself (it executes script
logic at import time); instead they verify the registry-level filters that
the dashboard uses to decide which sections to render.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _REPO_ROOT / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


def _eeg_bulk_parquets_present() -> bool:
    """True iff the EEG bulk parquets are real on-disk files (not LFS pointer
    stubs and not absent). On CI we skip the LFS pull → these are missing →
    any test that requires EEG locally must skip cleanly."""
    eeg_dir = _REPO_ROOT / "data" / "eeg_motor_left_right" / "processed"
    f = eeg_dir / "log_returns.parquet"
    if not f.exists():
        return False
    try:
        # Real parquets start with the 4-byte PAR1 magic. LFS pointer stubs
        # are tiny text files starting with "version https://git-lfs...".
        if f.stat().st_size < 100_000:  # heuristic — real EEG parquet is 300 MB
            return False
    except OSError:
        return False
    return True


needs_eeg_bulk = pytest.mark.skipif(
    not _eeg_bulk_parquets_present(),
    reason=(
        "EEG bulk parquets not present locally — skipped on CI by design. "
        "Run `uv run python run_pipeline_eeg.py` locally to enable EEG-specific tests."
    ),
)


@pytest.fixture(scope="module")
def registry():
    import universe_registry as ur
    return ur


FINANCIAL_UNIVERSE_KEYS = {"bist", "bist_usd", "bist_gold", "sp500"}


def test_pair_trading_filter_segregates_universes(registry):
    """app/dashboard.py's nav-level gate checks `has_pair_trading`; this test
    pins the expected split between financial and non-financial universes.
    BIST numéraire variants (bist_usd, bist_gold) inherit the finance flags
    so they participate in pair analysis like the source BIST universe."""
    pair_capable = {u.key for u in registry.UNIVERSES.values() if u.has_pair_trading}
    pair_incapable = {u.key for u in registry.UNIVERSES.values() if not u.has_pair_trading}
    assert pair_capable == FINANCIAL_UNIVERSE_KEYS, (
        f"Pair-Analysis-eligible universes drifted; expected {FINANCIAL_UNIVERSE_KEYS}, "
        f"got {pair_capable}"
    )
    assert pair_incapable == {"eeg_motor_left_right"}, (
        f"Pair-Analysis-ineligible universes drifted; expected {{eeg_motor_left_right}}, "
        f"got {pair_incapable}"
    )


def test_snn_filter_segregates_universes(registry):
    """app/eee_analysis.py hides the Neuromorphic Signals sub-tab via
    `has_snn`; only financial universes ship the SNN."""
    snn_capable = {u.key for u in registry.UNIVERSES.values() if u.has_snn}
    assert snn_capable == FINANCIAL_UNIVERSE_KEYS


def test_cross_market_eligibility_filter(registry):
    """app/cross_market.py's defence-in-depth filter requires at least 2
    universes with `eligible_for_cross_market=True`. The BIST numéraire
    variants are intentionally EXCLUDED from cross-market eligibility —
    the BIST↔S&P comparison page only knows the two source markets, not
    the FX/gold-denominated variants."""
    eligible = [u for u in registry.UNIVERSES.values() if u.eligible_for_cross_market]
    ineligible = [u for u in registry.UNIVERSES.values() if not u.eligible_for_cross_market]
    assert len(eligible) >= 2, "Need ≥2 eligible universes for Cross-Market to render"
    assert all(u.key in ("bist", "sp500") for u in eligible), (
        "Only source markets BIST and S&P are eligible for cross-market"
    )
    ineligible_keys = {u.key for u in ineligible}
    assert "eeg_motor_left_right" in ineligible_keys
    assert "bist_usd" in ineligible_keys
    assert "bist_gold" in ineligible_keys


def test_anomaly_detection_filter_segregates_universes(registry):
    """app/dashboard.py's Anomalies section reads `has_anomaly_detection`.
    EEG has no concept of |log return| > threshold so it's off; financial
    universes (including BIST numéraire variants) keep it on."""
    anom_capable = {u.key for u in registry.UNIVERSES.values() if u.has_anomaly_detection}
    assert anom_capable == FINANCIAL_UNIVERSE_KEYS


def test_index_series_filter_segregates_universes(registry):
    """The market-index overlay (XU100 / ^GSPC) is a financial-only concept.
    BIST variants inherit XU100 from the source universe."""
    idx_capable = {u.key for u in registry.UNIVERSES.values() if u.has_index_series}
    assert idx_capable == FINANCIAL_UNIVERSE_KEYS


def test_validation_report_filter_segregates_universes(registry):
    """İş Yatırım cross-check is BIST-specific data infrastructure; S&P
    keeps the capability on so its (absent) validation_report.csv still
    causes the popover to show "no report". EEG turns it off entirely.
    BIST variants inherit the flag (their processed/ tree is a copy of
    BIST's, including validation_report.csv)."""
    val_capable = {u.key for u in registry.UNIVERSES.values() if u.has_validation_report}
    assert val_capable == FINANCIAL_UNIVERSE_KEYS


def test_eeg_pipeline_metadata_present_on_disk(registry):
    """If this test fails locally, the EEG pipeline rerun didn't complete and
    the dashboard will hide EEG from the selector. CI on Streamlit Cloud will
    have the LFS-pulled bulk parquets but should still have the small
    pipeline_metadata.json blob available."""
    eeg_meta = registry.PROJECT_ROOT / "data" / "eeg_motor_left_right" / "results" / "pipeline_metadata.json"
    assert eeg_meta.exists(), (
        f"EEG pipeline_metadata.json missing at {eeg_meta} — "
        "did the EEG pipeline rerun complete? "
        "`uv sync --extra eeg && uv run python run_pipeline_eeg.py`"
    )


@needs_eeg_bulk
def test_eeg_appears_in_available_universes_when_on_disk(registry):
    """Regression for the dashboard sidebar selector.

    Skipped on CI where the EEG bulk parquets are excluded from the
    `actions/checkout` step (we skip LFS to keep CI under a minute).
    The deploy environment downloads them from the companion HF Dataset
    via the dashboard's `_materialise_eeg_data_if_needed()` preload —
    not the same code path as this test exercises.
    """
    avail = {u.key for u in registry.available_universes()}
    assert "eeg_motor_left_right" in avail, (
        "EEG should be selectable in the sidebar (its pipeline_metadata.json "
        "AND bulk parquets are both on disk)"
    )
