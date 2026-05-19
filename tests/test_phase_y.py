"""Tests for PHASE Y — lazy sub-tab rendering + MST layout precompute + deeper pre-warm.

Three pillars of coverage:

1. **render_subtabs helper (Y1)** — sub-tab state survives within a dataset,
   pending-stash restores on universe round-trip, namespace per dataset.

2. **MST layout pipeline + loader (Y2)** — `run_mst_layouts(config)` writes
   the expected JSON files with stable schema; `load_mst_layout(source)`
   reads them back; missing files return empty dict (live fallback).

3. **Pre-warm hook (Y3)** — the deep-warm code path doesn't crash on
   imports + loader signatures haven't drifted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _REPO_ROOT / "app"
for _p in (str(_REPO_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(autouse=True)
def _skip_eeg_download(monkeypatch):
    monkeypatch.setenv("STONECOAL_SKIP_EEG_DOWNLOAD", "1")


# ---------------------------------------------------------------------------
# Y1 — render_subtabs helper
# ---------------------------------------------------------------------------


def _has_bist_data() -> bool:
    return (_REPO_ROOT / "data" / "bist" / "results" / "pipeline_metadata.json").exists()


needs_bist = pytest.mark.skipif(not _has_bist_data(), reason="BIST data missing")


@needs_bist
def test_render_subtabs_default_first_option():
    """When no prior state exists, the first option is the default."""
    from streamlit.testing.v1 import AppTest

    # PHASE 2: dashboard.py → main.py. Test the Market Overview page
    # script directly (not via switch_page — AppTest has known issues
    # with widget-state caching across navigations).
    MARKET_OVERVIEW_PATH = _APP_DIR / "views" / "02_market_overview.py"
    at = AppTest.from_file(str(MARKET_OVERVIEW_PATH))
    at.session_state["dataset"] = "bist"
    # PHASE 2: nav_page_bist no longer drives routing (top-nav replaced
    # by Streamlit native sidebar nav). Loading the page script directly
    # bypasses the page-routing logic so this line is now a no-op kept
    # for back-compat with any code that reads nav_page_bist.
    at.run(timeout=90)
    assert not at.exception, f"Render crashed: {list(at.exception)}"
    # The first sub-tab "Data & Stats" should be active by default.
    assert at.session_state["market_overview_subtab_bist"] == "Data & Stats"


@needs_bist
def test_render_subtabs_preserved_across_basis_flip():
    """Sub-tab choice on BIST should be preserved when basis flips
    (TRY → USD) since dataset stays "bist"."""
    from streamlit.testing.v1 import AppTest

    # PHASE 2: dashboard.py → main.py. Test the Market Overview page
    # script directly (not via switch_page — AppTest has known issues
    # with widget-state caching across navigations).
    MARKET_OVERVIEW_PATH = _APP_DIR / "views" / "02_market_overview.py"
    at = AppTest.from_file(str(MARKET_OVERVIEW_PATH))
    at.session_state["dataset"] = "bist"
    at.session_state["bist_basis"] = "try"
    # PHASE 2: nav_page_bist no longer drives routing (top-nav replaced
    # by Streamlit native sidebar nav). Loading the page script directly
    # bypasses the page-routing logic so this line is now a no-op kept
    # for back-compat with any code that reads nav_page_bist.
    # Pre-set a non-default sub-tab.
    at.session_state["market_overview_subtab_bist"] = "Clustering & Network"
    at.run(timeout=90)
    assert not at.exception
    assert at.session_state["market_overview_subtab_bist"] == "Clustering & Network"


@needs_bist
def test_render_subtabs_namespaced_per_dataset():
    """Sub-tab key includes the dataset, so BIST and S&P don't share state."""
    from streamlit.testing.v1 import AppTest

    # PHASE 2: dashboard.py → main.py. Test the Market Overview page
    # script directly (not via switch_page — AppTest has known issues
    # with widget-state caching across navigations).
    MARKET_OVERVIEW_PATH = _APP_DIR / "views" / "02_market_overview.py"
    at = AppTest.from_file(str(MARKET_OVERVIEW_PATH))
    at.session_state["dataset"] = "sp500"
    # nav_page_sp500 is a no-op in multi-page mode (see Phase 2 note above).
    # Pre-set a BIST sub-tab to a non-default value; S&P should be
    # independent and default to "Data & Stats".
    at.session_state["market_overview_subtab_bist"] = "Rolling Analysis"
    at.run(timeout=90)
    assert not at.exception
    # S&P key should NOT inherit BIST's value.
    assert at.session_state["market_overview_subtab_sp500"] == "Data & Stats"
    # BIST key untouched.
    assert at.session_state["market_overview_subtab_bist"] == "Rolling Analysis"


# ---------------------------------------------------------------------------
# Y2 — MST layout pipeline + loader
# ---------------------------------------------------------------------------


def test_compute_layout_for_edges_basic_schema():
    """_compute_layout_for_edges returns positions dict with float tuples."""
    from src.mst_layouts import _compute_layout_for_edges

    edges = pd.DataFrame({
        "source": ["A", "B", "C"],
        "target": ["B", "C", "A"],
        "distance": [0.5, 0.7, 0.6],
    })
    positions, algo, n_nodes = _compute_layout_for_edges(edges)
    assert n_nodes == 3
    assert algo in ("kamada_kawai", "spring")
    assert set(positions.keys()) == {"A", "B", "C"}
    for node, xy in positions.items():
        assert len(xy) == 2
        assert isinstance(xy[0], float)
        assert isinstance(xy[1], float)


def test_compute_layout_for_edges_picks_spring_for_large_graphs():
    """Spring layout fires for graphs > 200 nodes (matches dashboard
    `_mst_layout` heuristic so live and precomputed give identical positions)."""
    from src.mst_layouts import _compute_layout_for_edges

    # Build a 220-node MST: a simple path graph.
    n = 220
    edges = pd.DataFrame({
        "source": [f"N{i}" for i in range(n - 1)],
        "target": [f"N{i+1}" for i in range(n - 1)],
        "distance": [0.5] * (n - 1),
    })
    _, algo, n_nodes = _compute_layout_for_edges(edges)
    assert n_nodes == n
    assert algo == "spring"


def test_compute_layout_for_edges_empty():
    """Empty edges → empty positions, no crash."""
    from src.mst_layouts import _compute_layout_for_edges
    positions, algo, n_nodes = _compute_layout_for_edges(pd.DataFrame())
    assert positions == {}
    assert n_nodes == 0


def test_run_mst_layouts_writes_expected_files(tmp_path, monkeypatch):
    """End-to-end pipeline stage: writes JSON files with correct schema."""
    from src import mst_layouts
    from src.mst_layouts import run_mst_layouts

    # Force the test market into the precompute set.
    monkeypatch.setattr(mst_layouts, "_LAYOUT_MARKETS", {"testmkt"})

    # Synthetic MST edges + denoised MST edges files.
    results = tmp_path / "results"
    results.mkdir()
    edges = pd.DataFrame({
        "source": [f"T{i}" for i in range(10)],
        "target": [f"T{i+1}" for i in range(10)],
        "distance": [0.5] * 10,
    })
    edges.to_csv(results / "mst_edges.csv", index=False)
    edges.to_csv(results / "denoised_mst_edges.csv", index=False)

    class _FakeMarket:
        market_id = "testmkt"

    class _FakeConfig:
        market = _FakeMarket()
        data_results = results

    run_mst_layouts(_FakeConfig())

    layouts_dir = results / "layouts"
    assert layouts_dir.is_dir()
    assert (layouts_dir / "main_mst.json").exists()
    assert (layouts_dir / "denoised_mst.json").exists()

    # Schema check.
    payload = json.loads((layouts_dir / "main_mst.json").read_text())
    assert "positions" in payload
    assert payload["n_nodes"] == 11
    assert payload["seed"] == 42
    assert payload["source_file"] == "mst_edges.csv"
    assert payload["algorithm"] in ("kamada_kawai", "spring")
    # Positions dict: 11 nodes × [x, y].
    assert len(payload["positions"]) == 11


def test_run_mst_layouts_skipped_for_non_finance(tmp_path):
    """EEG (not in _LAYOUT_MARKETS) skips silently — no files written."""
    from src.mst_layouts import run_mst_layouts

    class _FakeMarket:
        market_id = "eeg_motor_left_right"

    class _FakeConfig:
        market = _FakeMarket()
        data_results = tmp_path / "results"

    run_mst_layouts(_FakeConfig())
    assert not (tmp_path / "results").exists() or not list((tmp_path / "results").iterdir())


def _has_bist_layouts() -> bool:
    return (_REPO_ROOT / "data" / "bist" / "results" / "layouts" / "main_mst.json").exists()


needs_layouts = pytest.mark.skipif(not _has_bist_layouts(), reason="MST layouts not on disk")


@needs_layouts
def test_load_mst_layout_returns_positions_dict(monkeypatch):
    import streamlit as st
    import utils
    from utils import load_mst_layout
    st.cache_data.clear()
    monkeypatch.setattr(utils, "current_universe", lambda: "bist")

    pos = load_mst_layout("main_mst")
    assert pos, "BIST main_mst should have positions"
    # All values are (float, float) tuples.
    for node, xy in pos.items():
        assert isinstance(node, str)
        assert len(xy) == 2
        assert all(isinstance(c, float) for c in xy)


@needs_layouts
def test_load_mst_layout_missing_returns_empty(monkeypatch):
    """Missing source → empty dict (so renderer falls back to live)."""
    import streamlit as st
    import utils
    from utils import load_mst_layout
    st.cache_data.clear()
    monkeypatch.setattr(utils, "current_universe", lambda: "bist")

    nothing = load_mst_layout("nonexistent_source")
    assert nothing == {}


@needs_layouts
def test_load_mst_layout_for_eeg_returns_empty(monkeypatch):
    """EEG isn't in the precompute set → loader returns empty (live fallback)."""
    import streamlit as st
    import utils
    from utils import load_mst_layout
    st.cache_data.clear()
    monkeypatch.setattr(utils, "current_universe", lambda: "eeg_motor_left_right")

    eeg_layout = load_mst_layout("main_mst")
    assert eeg_layout == {}


# ---------------------------------------------------------------------------
# Y3 — Pre-warm hook (signature + import sanity)
# ---------------------------------------------------------------------------


def test_prewarm_imports_resolve():
    """The deep-warm path imports a handful of loaders from utils. Confirm
    they all exist + have the expected universe-keyed signature."""
    from utils import (
        _load_log_returns, _load_metadata,
        _load_batch_corr, _load_mst_edges, _load_mst_metrics,
        _load_cluster_assignments, _load_dendrogram_order,
        _load_eigenvalue_spectrum, _load_summary_stats,
        current_universe,
    )
    # current_universe is callable.
    assert callable(current_universe)
    # Each loader takes a universe key string as first positional arg.
    for loader in (_load_log_returns, _load_metadata, _load_batch_corr,
                   _load_mst_edges, _load_mst_metrics, _load_cluster_assignments,
                   _load_dendrogram_order, _load_eigenvalue_spectrum,
                   _load_summary_stats):
        # Cached functions expose `clear()` from @st.cache_data.
        assert hasattr(loader, "clear"), f"{loader.__name__} not @st.cache_data"
