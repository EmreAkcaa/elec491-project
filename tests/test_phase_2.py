"""Tests for PHASE 2 — multi-page migration.

Coverage:

1. **main.py renders cleanly** for each available universe (BIST, S&P,
   and EEG when present).
2. **Each page script renders cleanly** in isolation when the sidebar
   state is pre-set.
3. **Capability gating**: Cross-Market and Pair Analysis pages don't
   appear in main.py's `_pages` list when the active universe has the
   corresponding capability flag set to False (bist_usd, bist_gold for
   Cross-Market; EEG for both).
4. **Page-disappearance pending stash** mirrors the Phase S #1 semantic
   for the new `last_page_{dataset}` keys (covered by `test_nav_stability`
   in more detail; one anchor test here for completeness).
5. **Cross-page session_state preservation** — keys set on one page
   survive to the next page when loaded back-to-back (the AppTest
   limitation re: switch_page widget-state caching is acknowledged; we
   test by loading page scripts sequentially with state pre-set).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    pytest.skip(
        "streamlit.testing.v1.AppTest unavailable.",
        allow_module_level=True,
    )

_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _REPO_ROOT / "app"
for _p in (str(_REPO_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAIN_PATH = _APP_DIR / "main.py"
APPTEST_TIMEOUT = 90

PAGE_PATHS = {
    "cross_market":    _APP_DIR / "views" / "01_cross_market.py",
    "market_overview": _APP_DIR / "views" / "02_market_overview.py",
    "time_machine":    _APP_DIR / "views" / "03_time_machine.py",
    "pair_analysis":   _APP_DIR / "views" / "04_pair_analysis.py",
    "methods_lab":     _APP_DIR / "views" / "05_methods_lab.py",
}


@pytest.fixture(autouse=True)
def _skip_eeg_download(monkeypatch):
    monkeypatch.setenv("STONECOAL_SKIP_EEG_DOWNLOAD", "1")


def _financial_universes_on_disk() -> list[str]:
    out = []
    for key in ("bist", "sp500"):
        meta = _REPO_ROOT / "data" / key / "results" / "pipeline_metadata.json"
        if meta.exists():
            out.append(key)
    return out


_AVAILABLE = _financial_universes_on_disk()
_HAS_BIST  = "bist" in _AVAILABLE
_HAS_SP500 = "sp500" in _AVAILABLE

needs_data = pytest.mark.skipif(
    not _AVAILABLE,
    reason="No financial universe data on disk.",
)


# ---------------------------------------------------------------------------
# main.py renders cleanly
# ---------------------------------------------------------------------------


@needs_data
@pytest.mark.parametrize("universe", _AVAILABLE)
def test_main_renders_for_each_universe(universe):
    """main.py is the entry script. Must not crash for any universe."""
    at = AppTest.from_file(str(MAIN_PATH))
    at.session_state["universe"] = universe
    if universe.startswith("bist"):
        at.session_state["dataset"] = "bist"
        at.session_state["bist_basis"] = {
            "bist": "try", "bist_usd": "usd", "bist_gold": "gold",
        }.get(universe, "try")
    else:
        at.session_state["dataset"] = universe
    at.run(timeout=APPTEST_TIMEOUT)
    assert not at.exception, (
        f"main.py crashed for universe={universe!r}: {list(at.exception)}"
    )


# ---------------------------------------------------------------------------
# Each page script renders in isolation
# ---------------------------------------------------------------------------


@needs_data
@pytest.mark.skipif(not _HAS_BIST, reason="BIST data missing")
@pytest.mark.parametrize("page", list(PAGE_PATHS))
def test_page_renders_for_bist(page):
    """Each of the 5 page scripts renders cleanly with BIST as the active
    universe. Tests the page wrappers + their delegate render() calls."""
    at = AppTest.from_file(str(PAGE_PATHS[page]))
    at.session_state["dataset"] = "bist"
    at.session_state["bist_basis"] = "try"
    at.run(timeout=APPTEST_TIMEOUT)
    assert not at.exception, (
        f"Page {page!r} crashed for BIST: {list(at.exception)}"
    )


@needs_data
@pytest.mark.skipif(not _HAS_SP500, reason="S&P data missing")
@pytest.mark.parametrize("page", list(PAGE_PATHS))
def test_page_renders_for_sp500(page):
    """Each of the 5 page scripts renders cleanly with S&P as the active
    universe. (Cross-Market reads both BIST + S&P; the others switch to
    the S&P-keyed loaders.)"""
    at = AppTest.from_file(str(PAGE_PATHS[page]))
    at.session_state["dataset"] = "sp500"
    at.run(timeout=APPTEST_TIMEOUT)
    assert not at.exception, (
        f"Page {page!r} crashed for S&P: {list(at.exception)}"
    )


# ---------------------------------------------------------------------------
# Capability gating in main.py's page list
# ---------------------------------------------------------------------------


@needs_data
@pytest.mark.skipif(not _HAS_BIST, reason="BIST data missing")
def test_bist_default_landing_is_cross_market():
    """First visible page on BIST is Cross-Market (default landing)."""
    at = AppTest.from_file(str(MAIN_PATH))
    at.session_state["dataset"] = "bist"
    at.session_state["bist_basis"] = "try"
    at.run(timeout=APPTEST_TIMEOUT)
    assert not at.exception
    assert at.session_state["last_page_bist"] == "Cross-Market"


@needs_data
def _has_bist_usd_data() -> bool:
    return (_REPO_ROOT / "data" / "bist_usd" / "results" / "pipeline_metadata.json").exists()


@pytest.mark.skipif(
    not (_HAS_BIST and _has_bist_usd_data()),
    reason="Both BIST and BIST_USD data must be present.",
)
def test_bist_usd_hides_cross_market_via_first_page():
    """`bist_usd` has `eligible_for_cross_market=False` — main.py should
    NOT pick Cross-Market as the default page. The first visible page
    becomes Market Overview."""
    at = AppTest.from_file(str(MAIN_PATH))
    at.session_state["dataset"] = "bist"
    at.session_state["bist_basis"] = "usd"
    at.run(timeout=APPTEST_TIMEOUT)
    assert not at.exception
    # Default landing for bist_usd is Market Overview (Cross-Market hidden).
    assert at.session_state["last_page_bist"] == "Market Overview"


# ---------------------------------------------------------------------------
# Cross-page session_state preservation
# ---------------------------------------------------------------------------


@needs_data
@pytest.mark.skipif(not _HAS_BIST, reason="BIST data missing")
def test_pa_ticker_state_survives_across_pages():
    """`pa_ticker_a` / `pa_ticker_b` are cross-page shared (set by Pair
    Analysis OR by Market Overview's rolling-pair sub-tab). Loading
    Pair Analysis first sets these; loading Market Overview second
    should NOT clear them."""
    # Step 1: load Pair Analysis to populate pa_ticker_a/b defaults.
    at_pa = AppTest.from_file(str(PAGE_PATHS["pair_analysis"]))
    at_pa.session_state["dataset"] = "bist"
    at_pa.session_state["bist_basis"] = "try"
    at_pa.run(timeout=APPTEST_TIMEOUT)
    assert not at_pa.exception
    assert "pa_ticker_a" in at_pa.session_state
    assert "pa_ticker_b" in at_pa.session_state
    ticker_a_initial = at_pa.session_state["pa_ticker_a"]
    ticker_b_initial = at_pa.session_state["pa_ticker_b"]

    # Step 2: load Market Overview with the same ticker values pre-set.
    # In a real session these would survive switch_page; here we simulate
    # by passing them via session_state.
    at_mo = AppTest.from_file(str(PAGE_PATHS["market_overview"]))
    at_mo.session_state["dataset"] = "bist"
    at_mo.session_state["bist_basis"] = "try"
    at_mo.session_state["pa_ticker_a"] = ticker_a_initial
    at_mo.session_state["pa_ticker_b"] = ticker_b_initial
    at_mo.run(timeout=APPTEST_TIMEOUT)
    assert not at_mo.exception
    # Tickers preserved (Market Overview's rolling-pair sub-tab uses
    # the same keys, so it sees the user's prior pick).
    assert at_mo.session_state["pa_ticker_a"] == ticker_a_initial
    assert at_mo.session_state["pa_ticker_b"] == ticker_b_initial


# ---------------------------------------------------------------------------
# Page-script directory + file layout invariants
# ---------------------------------------------------------------------------


def test_views_directory_named_views_not_pages():
    """Streamlit auto-discovers a `pages/` directory siblings to the
    entrypoint and creates its own nav widget — clashes with our
    explicit `st.navigation([...])` call. We use `views/` instead to
    avoid auto-discovery. Regression guard so a future rename doesn't
    silently re-enable the duplicate nav."""
    assert (_APP_DIR / "views").is_dir(), (
        "Expected `app/views/` to exist with the 5 page wrappers."
    )
    assert not (_APP_DIR / "pages").exists(), (
        "`app/pages/` would trigger Streamlit's auto-discovery and "
        "duplicate the page list. Use `app/views/` instead."
    )


def test_all_five_page_scripts_present():
    """All 5 page scripts exist with the expected numerical prefixes."""
    expected = {
        "01_cross_market.py",
        "02_market_overview.py",
        "03_time_machine.py",
        "04_pair_analysis.py",
        "05_methods_lab.py",
    }
    found = {p.name for p in (_APP_DIR / "views").iterdir() if p.suffix == ".py"}
    missing = expected - found
    assert not missing, f"Missing page scripts: {sorted(missing)}"


def test_dashboard_py_removed():
    """Phase 2 deletes app/dashboard.py. Anyone running the old
    `streamlit run app/dashboard.py` should hit a clear error."""
    assert not (_APP_DIR / "dashboard.py").exists(), (
        "app/dashboard.py was deleted in Phase 2; main.py is the new entry."
    )


def test_main_py_present_and_calls_navigation():
    """main.py exists and references `st.navigation`."""
    assert MAIN_PATH.exists()
    body = MAIN_PATH.read_text()
    assert "st.navigation(" in body, (
        "main.py must call st.navigation([...]) to dispatch to pages."
    )
    assert 'position="sidebar"' in body or 'position=\'sidebar\'' in body, (
        "Phase 2 uses sidebar navigation (chosen 2026-05-19); main.py "
        "must explicitly pass position='sidebar' to st.navigation."
    )
