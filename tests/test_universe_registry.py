"""Tests for app/universe_registry.py — the registry of dashboard-switchable
universes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The Streamlit app directory isn't on sys.path by default for tests.
_APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


@pytest.fixture(scope="module")
def registry():
    """Import lazily so the fixture also exercises the import path."""
    import universe_registry as ur
    return ur


def test_bist_config_path_exists(registry):
    """The BIST universe's declared config_path must resolve to a real file."""
    cfg = registry.PROJECT_ROOT / registry.UNIVERSES["bist"].config_path
    assert cfg.exists(), f"BIST config_path does not exist: {cfg}"


def test_sp500_config_path_exists(registry):
    """The S&P-500 universe's declared config_path must resolve to a real file."""
    cfg = registry.PROJECT_ROOT / registry.UNIVERSES["sp500"].config_path
    assert cfg.exists(), f"S&P config_path does not exist: {cfg}"


def test_available_universes_only_returns_populated_ones(registry):
    """available_universes() must filter to universes with results/
    pipeline_metadata.json on disk; nothing else."""
    avail = registry.available_universes()
    assert isinstance(avail, list)
    for u in avail:
        meta = registry.PROJECT_ROOT / "data" / u.key / "results" / "pipeline_metadata.json"
        assert meta.exists(), (
            f"available_universes() returned {u.key!r} but its "
            f"pipeline_metadata.json does not exist at {meta}"
        )


def test_get_universe_falls_back_to_bist(registry):
    """get_universe() must return BIST when handed an unknown key."""
    u = registry.get_universe("some_universe_that_does_not_exist")
    assert u.key == "bist", "Unknown keys must fall back to bist for safety"
    # And the BIST entry itself must be retrievable directly.
    assert registry.get_universe("bist").key == "bist"
    assert registry.get_universe("sp500").key == "sp500"


# ── Phase I: EEG universe + capability flags ───────────────────────────


def test_eeg_universe_registered(registry):
    """EEG should be in the static UNIVERSES dict regardless of disk state."""
    assert "eeg_motor_left_right" in registry.UNIVERSES
    eeg = registry.UNIVERSES["eeg_motor_left_right"]
    assert eeg.label.startswith("EEG")
    assert eeg.short_label == "EEG MI"


def test_eeg_capability_flags_off_for_financial_features(registry):
    """EEG must turn OFF every financial-only capability flag so the dashboard
    correctly hides pair trading, SNN, anomalies, the market-index overlay,
    the validation report popover, and the Cross-Market eligibility."""
    eeg = registry.UNIVERSES["eeg_motor_left_right"]
    assert eeg.has_pair_trading is False
    assert eeg.has_snn is False
    assert eeg.has_anomaly_detection is False
    assert eeg.has_index_series is False
    assert eeg.has_validation_report is False
    assert eeg.eligible_for_cross_market is False
    # Currency + index ticker should be None for non-financial universes
    assert eeg.currency is None
    assert eeg.index_ticker is None


def test_eeg_universe_has_neuroscience_domain_and_terminology(registry):
    """Display-terminology fields must reflect neuroscience vocab so axis
    labels and captions render as Channel/Voltage/etc., not Ticker/Return."""
    eeg = registry.UNIVERSES["eeg_motor_left_right"]
    assert eeg.domain == "neuroscience"
    assert eeg.item_label == "Channel"
    assert eeg.items_label == "Channels"
    assert eeg.sector_label == "Anatomical region"
    assert "voltage" in eeg.series_label.lower()
    assert eeg.series_units == "µV"


def test_eeg_sanity_check_groups_use_real_electrode_names(registry):
    """The sanity_check_groups should reference actual 10-10 electrode labels
    so the clustering tab's membership badge has a chance of matching."""
    eeg = registry.UNIVERSES["eeg_motor_left_right"]
    assert eeg.sanity_check_groups is not None
    all_members = [m for members in eeg.sanity_check_groups.values() for m in members]
    # Spot-check: the three groups should cover central, occipital, prefrontal
    for required in ("C3", "Cz", "C4", "O1", "Oz", "O2", "Fp1", "Fpz", "Fp2"):
        assert required in all_members, (
            f"EEG sanity_check_groups missing expected electrode {required!r}"
        )


def test_lfs_pointer_detection_recognises_real_parquet(registry, tmp_path):
    """A real parquet-shaped file (PAR1 magic + multi-KB size) must NOT be
    flagged as an LFS pointer stub."""
    fake_parquet = tmp_path / "real.parquet"
    fake_parquet.write_bytes(b"PAR1" + (b"\x00" * 5000))
    assert registry._is_lfs_pointer_stub(fake_parquet) is False


def test_lfs_pointer_detection_recognises_pointer_stub(registry, tmp_path):
    """An actual Git LFS pointer file must be flagged so available_universes()
    can skip the universe instead of letting pd.read_parquet() crash."""
    fake_pointer = tmp_path / "pointer.parquet"
    fake_pointer.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:55da3079b351132bab6277095219b42e9e7fd752\n"
        b"size 322712970\n"
    )
    assert registry._is_lfs_pointer_stub(fake_pointer) is True


def test_lfs_pointer_detection_missing_file(registry, tmp_path):
    """Missing file → treated as broken (returns True → universe skipped)."""
    missing = tmp_path / "does_not_exist.parquet"
    assert registry._is_lfs_pointer_stub(missing) is True


def test_bulk_data_materialised_for_locally_present_universes(registry):
    """Whatever universes are on this clone with a pipeline_metadata.json must
    also have real parquets — otherwise local development would be broken."""
    for u in registry.UNIVERSES.values():
        meta = registry.PROJECT_ROOT / "data" / u.key / "results" / "pipeline_metadata.json"
        if not meta.exists():
            continue
        assert registry._bulk_data_materialised(u), (
            f"{u.key}: bulk parquets are LFS pointer stubs locally — "
            "run `git lfs pull` to materialise them"
        )


def test_financial_defaults_unchanged_for_bist_and_sp500(registry):
    """BIST and S&P must keep all financial capabilities ON by default —
    the Phase I refactor must not have silently disabled any financial
    feature for the existing universes."""
    for k in ("bist", "sp500"):
        u = registry.UNIVERSES[k]
        assert u.has_pair_trading is True, f"{k} lost pair trading"
        assert u.has_snn is True, f"{k} lost SNN"
        assert u.has_anomaly_detection is True, f"{k} lost anomaly detection"
        assert u.has_index_series is True, f"{k} lost index series"
        assert u.eligible_for_cross_market is True, f"{k} lost cross-market eligibility"
        assert u.domain == "finance"
        assert u.item_label == "Ticker"
        assert u.sector_label == "Sector"
        assert u.currency is not None
        assert u.index_ticker is not None
