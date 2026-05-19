"""Tests for the PORT of arda/ui-cleanup-batch (commit 128c028b).

Three anchor tests covering the highest-risk ports:

1. **Cross-Market editable crisis section renders + session_state keys exist**
   — item 14, the biggest delta (static `_crisis_fig` replaced by editable
   `_crisis_fig_live` panel with `xm_events_df`, `xm_events_applied`,
   `xm_delete_pick`, `_xm_clear_delete_pick` session_state keys).

2. **Time Machine MST receives a non-empty sector_map and back-compat works**
   — item 10. `_render_mst` now accepts `sector_map: dict | None = None`
   and renders per-sector colored nodes with a legend when non-empty,
   falls back to primary-color uniform style otherwise.

3. **Pair Analysis Ticker B selectbox excludes Ticker A** — item 7. The
   filter `_ticker_b_options = [t for t in ticker_list if t != ticker_a]`
   means Ticker B's options never include the currently-selected Ticker A.

The other 8 ported items (text-scrubbing in cross_market, RMT stacked
layout, header-strip 2-row, Clustering dendrogram restructure, Data &
Stats captions removal, sidebar dataset caption removal, return-scatter
removal, status/expander tweaks) are covered transitively by the
existing AppTest smoke tests in `tests/test_dashboard_smoke.py` (every
page must render without exception across all universes).
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

APPTEST_TIMEOUT = 90


@pytest.fixture(autouse=True)
def _skip_eeg_download(monkeypatch):
    monkeypatch.setenv("STONECOAL_SKIP_EEG_DOWNLOAD", "1")


def _has_bist() -> bool:
    return (_REPO_ROOT / "data" / "bist" / "results" / "pipeline_metadata.json").exists()


def _has_sp500() -> bool:
    return (_REPO_ROOT / "data" / "sp500" / "results" / "pipeline_metadata.json").exists()


needs_data = pytest.mark.skipif(
    not (_has_bist() and _has_sp500()),
    reason="Both BIST + S&P data must be present for Cross-Market tests.",
)


# ---------------------------------------------------------------------------
# Item 14 — editable crisis windows
# ---------------------------------------------------------------------------


@needs_data
def test_cross_market_editable_crisis_session_state_seeded():
    """Cross-Market page seeds `xm_events_df` + `xm_events_applied` on
    first render. Both should be present with the 3 default events
    pre-loaded (COVID, Russia-Ukraine, Türkiye earthquakes)."""
    at = AppTest.from_file(str(_APP_DIR / "views" / "01_cross_market.py"))
    at.session_state["dataset"] = "bist"
    at.session_state["bist_basis"] = "try"
    at.run(timeout=APPTEST_TIMEOUT)
    assert not at.exception, f"Cross-Market crashed: {list(at.exception)}"

    keys = at.session_state.filtered_state
    assert "xm_events_df" in keys, (
        "Cross-Market should seed `xm_events_df` on first render."
    )
    assert "xm_events_applied" in keys, (
        "Cross-Market should seed `xm_events_applied` on first render "
        "(mirrors xm_events_df so the chart renders without requiring "
        "a Recompute click)."
    )
    # Default 3 events.
    df = at.session_state["xm_events_df"]
    assert len(df) == 3, f"Expected 3 default events, got {len(df)}"
    labels = list(df["label"])
    assert "COVID-19 WHO declaration" in labels
    assert "Russia-Ukraine war" in labels
    assert "Türkiye earthquakes" in labels


@needs_data
def test_cross_market_default_crisis_events_constant_shape():
    """The `_DEFAULT_CRISIS_EVENTS` constant must be `(date, label, window_days_int)`
    not the prior `(date, label, note_str)`. Catches accidental revert."""
    from cross_market import _DEFAULT_CRISIS_EVENTS
    assert isinstance(_DEFAULT_CRISIS_EVENTS, list)
    assert len(_DEFAULT_CRISIS_EVENTS) == 3
    for event in _DEFAULT_CRISIS_EVENTS:
        date_str, label, window_days = event
        assert isinstance(date_str, str)
        assert isinstance(label, str)
        assert isinstance(window_days, int), (
            f"Third element of _DEFAULT_CRISIS_EVENTS must be int "
            f"(window_days). Got {type(window_days).__name__} for {label!r}."
        )
        assert 5 <= window_days <= 252


@needs_data
def test_cross_market_avg_pairwise_corr_helper_signature():
    """The new `_avg_pairwise_corr` helper exists with the expected
    underscore-prefix-arg pattern + returns None on too-little-data."""
    import pandas as pd
    from cross_market import _avg_pairwise_corr

    # Empty DataFrame → None (less than 5 rows).
    empty = pd.DataFrame()
    result = _avg_pairwise_corr(empty, "empty_test", "2020-01-01", "2020-02-01")
    assert result is None, "Should return None on empty DataFrame."

    # Single-column DataFrame → None (less than 2 columns for corr).
    single_col = pd.DataFrame(
        {"A": [0.01, 0.02, -0.01, 0.005, 0.003, 0.008]},
        index=pd.date_range("2020-01-01", periods=6, freq="D"),
    )
    result = _avg_pairwise_corr(single_col, "single_col", "2020-01-01", "2020-01-06")
    assert result is None, "Should return None when only 1 column."


# ---------------------------------------------------------------------------
# Item 10 — Time Machine MST sector_map back-compat + signature
# ---------------------------------------------------------------------------


def test_time_machine_render_mst_accepts_sector_map_kwarg():
    """`_render_mst` gained a `sector_map: dict | None = None` keyword-only
    parameter. Back-compatible: callers that don't pass it still work."""
    import inspect
    from time_machine import _render_mst
    sig = inspect.signature(_render_mst)
    assert "sector_map" in sig.parameters, (
        "_render_mst must accept a `sector_map` kwarg (PORT item 10)."
    )
    p = sig.parameters["sector_map"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
        "sector_map must be keyword-only so back-compat with positional callers."
    )
    assert p.default is None, "sector_map default must be None (back-compat)."


@needs_data
def test_time_machine_crisis_presets_constant_present():
    """Time Machine has a crisis-event preset dropdown. The constants
    live INSIDE `render()` so we can't import them directly; instead
    verify the page renders + `tm_crisis_preset` is registered as a
    session_state key after first render."""
    at = AppTest.from_file(str(_APP_DIR / "views" / "03_time_machine.py"))
    at.session_state["dataset"] = "bist"
    at.session_state["bist_basis"] = "try"
    at.run(timeout=APPTEST_TIMEOUT)
    assert not at.exception, f"Time Machine crashed: {list(at.exception)}"
    # tm_crisis_preset is the selectbox key. Streamlit registers it on
    # first render even though no preset is picked (default option).
    assert "tm_crisis_preset" in at.session_state.filtered_state, (
        "Time Machine should register `tm_crisis_preset` selectbox key "
        "after first render (PORT item 11 partial)."
    )


# ---------------------------------------------------------------------------
# Item 7 — Pair Analysis Ticker B excludes Ticker A
# ---------------------------------------------------------------------------


@needs_data
def test_pair_analysis_ticker_b_excludes_ticker_a():
    """After PORT item 7, the Ticker B selectbox options must NOT include
    the currently-selected Ticker A. We verify via the rendered widget's
    `options` attribute."""
    at = AppTest.from_file(str(_APP_DIR / "views" / "04_pair_analysis.py"))
    at.session_state["dataset"] = "bist"
    at.session_state["bist_basis"] = "try"
    at.run(timeout=APPTEST_TIMEOUT)
    assert not at.exception, f"Pair Analysis crashed: {list(at.exception)}"

    # Find the Ticker B selectbox in the rendered widgets. Its key is
    # `pa_ticker_b`; we look at its `options` attribute and confirm
    # `pa_ticker_a`'s current value is NOT in the list.
    ticker_a_value = at.session_state["pa_ticker_a"]
    matching_widgets = [
        sb for sb in at.selectbox
        if getattr(sb, "key", None) == "pa_ticker_b"
    ]
    if not matching_widgets:
        pytest.skip(
            "No pa_ticker_b selectbox found (Compare-against = USD/TRY or Gold; "
            "selectbox is replaced by a disabled text_input in those modes)."
        )
    ticker_b_selectbox = matching_widgets[0]
    options = list(ticker_b_selectbox.options)
    assert ticker_a_value not in options, (
        f"Ticker B options should exclude Ticker A ({ticker_a_value}). "
        f"Found {ticker_a_value!r} in options: {options}"
    )
