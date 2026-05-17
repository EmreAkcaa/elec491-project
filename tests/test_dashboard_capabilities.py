"""Tests for the capability-flag-driven dashboard gating (Phase I).

These tests don't import the Streamlit dashboard itself (it executes script
logic at import time); instead they verify the registry-level filters that
the dashboard uses to decide which sections to render.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


@pytest.fixture(scope="module")
def registry():
    import universe_registry as ur
    return ur


def test_pair_trading_filter_segregates_universes(registry):
    """app/dashboard.py's nav-level gate checks `has_pair_trading`; this test
    pins the expected split between financial and non-financial universes."""
    pair_capable = {u.key for u in registry.UNIVERSES.values() if u.has_pair_trading}
    pair_incapable = {u.key for u in registry.UNIVERSES.values() if not u.has_pair_trading}
    assert pair_capable == {"bist", "sp500"}, (
        f"Pair-Analysis-eligible universes drifted; expected {{bist, sp500}}, got {pair_capable}"
    )
    assert pair_incapable == {"eeg_motor_left_right"}, (
        f"Pair-Analysis-ineligible universes drifted; expected {{eeg_motor_left_right}}, "
        f"got {pair_incapable}"
    )


def test_snn_filter_segregates_universes(registry):
    """app/eee_analysis.py hides the Neuromorphic Signals sub-tab via
    `has_snn`; only financial universes ship the SNN."""
    snn_capable = {u.key for u in registry.UNIVERSES.values() if u.has_snn}
    assert snn_capable == {"bist", "sp500"}


def test_cross_market_eligibility_filter(registry):
    """app/cross_market.py's defence-in-depth filter requires at least 2
    universes with `eligible_for_cross_market=True`."""
    eligible = [u for u in registry.UNIVERSES.values() if u.eligible_for_cross_market]
    ineligible = [u for u in registry.UNIVERSES.values() if not u.eligible_for_cross_market]
    assert len(eligible) >= 2, "Need ≥2 eligible universes for Cross-Market to render"
    assert all(u.key in ("bist", "sp500") for u in eligible)
    assert "eeg_motor_left_right" in {u.key for u in ineligible}


def test_anomaly_detection_filter_segregates_universes(registry):
    """app/dashboard.py's Anomalies section reads `has_anomaly_detection`.
    EEG has no concept of |log return| > threshold so it's off."""
    anom_capable = {u.key for u in registry.UNIVERSES.values() if u.has_anomaly_detection}
    assert anom_capable == {"bist", "sp500"}


def test_index_series_filter_segregates_universes(registry):
    """The market-index overlay (XU100 / ^GSPC) is a financial-only concept."""
    idx_capable = {u.key for u in registry.UNIVERSES.values() if u.has_index_series}
    assert idx_capable == {"bist", "sp500"}


def test_validation_report_filter_segregates_universes(registry):
    """İş Yatırım cross-check is BIST-specific data infrastructure; S&P
    keeps the capability on so its (absent) validation_report.csv still
    causes the popover to show "no report". EEG turns it off entirely."""
    val_capable = {u.key for u in registry.UNIVERSES.values() if u.has_validation_report}
    assert val_capable == {"bist", "sp500"}


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


def test_eeg_appears_in_available_universes_when_on_disk(registry):
    """Regression for the dashboard sidebar selector."""
    avail = {u.key for u in registry.available_universes()}
    assert "eeg_motor_left_right" in avail, (
        "EEG should be selectable in the sidebar (its pipeline_metadata.json is on disk)"
    )
