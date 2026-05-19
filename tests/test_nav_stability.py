"""Tests for top-nav-equivalent state preservation across sidebar changes.

History:
  - **Phase S #1** (PR #59): added per-dataset `nav_page_{dataset}` +
    `nav_page_{dataset}__pending` to the SINGLE-SCRIPT dashboard so the
    top-nav segmented_control survived basis flips that hid capability-
    gated tabs.
  - **Phase 2** (this PR): replaced the custom top-nav segmented_control
    with Streamlit's native multi-page navigation. Phase S's stash
    semantic is preserved but the keys change:
        nav_page_{dataset}            → last_page_{dataset}
        nav_page_{dataset}__pending   → last_page_{dataset}__pending
    The clamp logic now lives in `app/main.py` (the new entry script).

User-facing intent (unchanged across the migration):
  - On BIST, click Cross-Market.
  - Flip basis TRY → USD (Cross-Market disappears for bist_usd).
  - Streamlit auto-redirects to default page (Market Overview).
  - The previously-active page ("Cross-Market") is stashed on
    `last_page_bist__pending`.
  - Flip basis USD → TRY (Cross-Market reappears).
  - main.py detects the stash + current default-page state, calls
    `st.switch_page` to restore Cross-Market.

These tests verify the stash logic on a SINGLE render (no `switch_page`
across runs — AppTest has known widget-state caching quirks there). The
clamp logic in main.py reads/writes session_state synchronously during
the render, so single-render tests cover the state machine fully.
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


@pytest.fixture(autouse=True)
def _skip_eeg_download(monkeypatch):
    monkeypatch.setenv("STONECOAL_SKIP_EEG_DOWNLOAD", "1")


def _has_bist_variants() -> bool:
    """Need all three BIST variants on disk to exercise basis flips."""
    for key in ("bist", "bist_usd", "bist_gold"):
        meta = _REPO_ROOT / "data" / key / "results" / "pipeline_metadata.json"
        if not meta.exists():
            return False
    return True


needs_bist_family = pytest.mark.skipif(
    not _has_bist_variants(),
    reason="BIST USD/Gold variants not on disk; run their pipelines first.",
)


# ---------------------------------------------------------------------------
# Single-run scenarios that exercise main.py's last_page + pending stash.
# ---------------------------------------------------------------------------


@needs_bist_family
def test_main_renders_with_default_state():
    """No prior state — main.py picks default page (first visible)."""
    at = AppTest.from_file(str(MAIN_PATH))
    at.session_state["dataset"] = "bist"
    at.session_state["bist_basis"] = "try"
    at.run(timeout=APPTEST_TIMEOUT)
    assert not at.exception, f"Render crashed: {list(at.exception)}"
    # main.py writes `last_page_bist` to track active page; default is
    # Cross-Market (first visible page on BIST TRY).
    assert at.session_state["last_page_bist"] == "Cross-Market"


@needs_bist_family
def test_cross_market_stash_when_capability_drops():
    """When the user is on Cross-Market AND we render in a basis where it
    is hidden (`bist_usd` has `eligible_for_cross_market=False`), the
    Phase S semantic stashes Cross-Market on `__pending` for round-trip
    restore.

    Test pattern: simulate the after-flip state — user was on Cross-Market
    (i.e., `last_page_bist == "Cross-Market"` from a prior TRY render),
    but the current render is `bist_usd`. main.py should:
      1. Detect Cross-Market not in visible pages
      2. Stash it on `last_page_bist__pending`
      3. Streamlit redirects to the new default (Market Overview)
    """
    at = AppTest.from_file(str(MAIN_PATH))
    at.session_state["dataset"] = "bist"
    at.session_state["bist_basis"] = "usd"  # bist_usd hides Cross-Market
    # Pre-set state as if the user just flipped from TRY (Cross-Market).
    at.session_state["last_page_bist"] = "Cross-Market"
    at.run(timeout=APPTEST_TIMEOUT)
    assert not at.exception, f"Render crashed: {list(at.exception)}"
    # Pending stash should hold the lost page for later restore.
    assert "last_page_bist__pending" in at.session_state, (
        "Cross-Market was lost when capability dropped; main.py should "
        "stash it on `last_page_bist__pending` for round-trip restore."
    )
    assert at.session_state["last_page_bist__pending"] == "Cross-Market"


@needs_bist_family
def test_pending_stash_restored_on_round_trip():
    """User round-tripped: was on Cross-Market under TRY, flipped to USD
    (Cross-Market stashed), now flipping back to TRY. main.py should
    detect the stash + visible page + auto-restore via `st.switch_page`.

    AppTest can't easily verify the `switch_page` call (it raises a
    NoReturn-like exception that aborts the current render), but we CAN
    verify that the stash is CONSUMED (key removed from session_state)
    when the conditions for restore are met.
    """
    at = AppTest.from_file(str(MAIN_PATH))
    at.session_state["dataset"] = "bist"
    at.session_state["bist_basis"] = "try"  # back to TRY → Cross-Market visible
    # Simulate the post-USD-detour state: stash holds Cross-Market;
    # last_page is the default (since Cross-Market was hidden during USD).
    at.session_state["last_page_bist__pending"] = "Cross-Market"
    at.session_state["last_page_bist"] = "Cross-Market"  # restored value
    at.run(timeout=APPTEST_TIMEOUT)
    # main.py's logic: if stash exists AND it's visible AND we're on the
    # default, it `switch_page`s to the stash + pops the pending key.
    # If main.py succeeded in restoring, the pending key is gone.
    # (If `switch_page` actually ran, AppTest would show a switch in
    # `at.main` — but the exact verifiable side-effect is the pending
    # key consumption.)
    assert not at.exception, f"Render crashed: {list(at.exception)}"
    # Pending key should be cleared after a successful restore.
    assert "last_page_bist__pending" not in at.session_state, (
        "Pending stash should be consumed after restore."
    )


@needs_bist_family
def test_pending_stash_dropped_when_user_navigated_away():
    """If the user navigates AWAY from the default after a clamp, the
    pending stash should be dropped on the next render (the user's intent
    has changed; they don't want the restore)."""
    at = AppTest.from_file(str(MAIN_PATH))
    at.session_state["dataset"] = "bist"
    at.session_state["bist_basis"] = "try"
    # User is on Pair Analysis (not default Cross-Market). Stale pending
    # holds a prior USD detour's "Cross-Market" — main.py should drop it
    # since the user has deliberately moved elsewhere.
    at.session_state["last_page_bist"] = "Pair Analysis"
    at.session_state["last_page_bist__pending"] = "Cross-Market"
    at.run(timeout=APPTEST_TIMEOUT)
    assert not at.exception, f"Render crashed: {list(at.exception)}"
    # Stash should be dropped because the user is no longer on the default.
    assert "last_page_bist__pending" not in at.session_state, (
        "User navigated to Pair Analysis from the default → stale pending "
        "stash for Cross-Market must be dropped (intent changed)."
    )


@needs_bist_family
def test_last_page_namespaced_per_dataset():
    """`last_page_{dataset}` keys are independent across datasets so a
    BIST round-trip via S&P doesn't leak state."""
    at = AppTest.from_file(str(MAIN_PATH))
    at.session_state["dataset"] = "sp500"
    # Pre-set BIST's last-page to non-default; rendering with dataset=sp500
    # should NOT inherit BIST's value (different key).
    at.session_state["last_page_bist"] = "Pair Analysis"
    at.run(timeout=APPTEST_TIMEOUT)
    assert not at.exception, f"Render crashed: {list(at.exception)}"
    # S&P key should be set to S&P's first visible page (Cross-Market).
    assert at.session_state["last_page_sp500"] == "Cross-Market"
    # BIST key untouched.
    assert at.session_state["last_page_bist"] == "Pair Analysis"
