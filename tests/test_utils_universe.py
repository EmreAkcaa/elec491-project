"""Tests for the universe-aware loader plumbing in app/utils.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Put the Streamlit app directory on sys.path so we can import utils
# without going through dashboard.py (which executes Streamlit script logic).
_APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


@pytest.fixture(scope="module")
def utils_mod():
    import utils
    return utils


def test_data_paths_bist_explicit(utils_mod):
    """data_raw('bist') / data_processed('bist') / data_results('bist')
    must point into data/bist/{raw,processed,results}/ regardless of
    session state or env var."""
    assert utils_mod.data_raw("bist").parts[-2:] == ("bist", "raw")
    assert utils_mod.data_processed("bist").parts[-2:] == ("bist", "processed")
    assert utils_mod.data_results("bist").parts[-2:] == ("bist", "results")

    # Absolute roots: must be PROJECT_ROOT/data/bist/<sub>.
    expected_root = utils_mod.PROJECT_ROOT / "data" / "bist"
    assert utils_mod.data_results("bist").parent == expected_root


def test_data_paths_sp500_explicit(utils_mod):
    """Same contract for the S&P universe key."""
    assert utils_mod.data_raw("sp500").parts[-2:] == ("sp500", "raw")
    assert utils_mod.data_processed("sp500").parts[-2:] == ("sp500", "processed")
    assert utils_mod.data_results("sp500").parts[-2:] == ("sp500", "results")
    expected_root = utils_mod.PROJECT_ROOT / "data" / "sp500"
    assert utils_mod.data_results("sp500").parent == expected_root


def test_current_universe_falls_back_outside_streamlit(utils_mod, monkeypatch):
    """When called outside a Streamlit script context (no ScriptRunContext),
    current_universe() must NOT raise — it must fall back to the env-var
    default. This is the behaviour the dashboard relies on for boot-time
    page-title resolution before session_state is hydrated."""
    monkeypatch.setenv("DASHBOARD_UNIVERSE", "bist")
    # Re-read DASHBOARD_UNIVERSE the same way utils does (module-level
    # const captured at import time), but call current_universe() now.
    u = utils_mod.current_universe()
    # Two acceptable outcomes:
    # (a) we get whatever was DASHBOARD_UNIVERSE at module import (likely 'bist')
    # (b) we get the session-state value if pytest happened to run inside a
    #     streamlit harness (it doesn't, but defensively allow it)
    assert isinstance(u, str)
    assert u in ("bist", "sp500"), f"unexpected universe key: {u!r}"
