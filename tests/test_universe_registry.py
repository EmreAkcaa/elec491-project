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
