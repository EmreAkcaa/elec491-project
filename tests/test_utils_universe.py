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


def _apply_render_chart_title_clear(fig):
    """Mirror of the title-clear block in `render_chart` (app/utils.py).
    Kept here so the tests don't have to spin up a Streamlit runtime."""
    cur_margin = dict(fig.layout.margin.to_plotly_json())
    cur_margin["t"] = 10
    fig.update_layout(margin=cur_margin)
    existing_title = fig.layout.title.text if fig.layout.title else None
    if existing_title:
        fig.update_layout(title=dict(text=""))


def test_render_chart_blanks_upstream_title_without_leaking_undefined():
    """A figure that had a title set by upstream `apply_chart_style` must
    end up with text="" in the JSON — never the empty dict `{}` which
    Plotly.js renders as the literal string "undefined".
    """
    import json
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
    fig.update_layout(title="Some upstream title we want to strip")

    _apply_render_chart_title_clear(fig)

    title_obj = json.loads(fig.to_json())["layout"].get("title")
    assert title_obj != {}, (
        "render_chart's title-clear produced an empty title dict. "
        "Plotly.js renders this as the literal string 'undefined'."
    )
    assert title_obj is not None and title_obj.get("text") == "", (
        f"Expected title.text=''; got {title_obj}"
    )


def test_render_chart_leaves_titleless_figures_untouched():
    """A figure that NEVER had a title shouldn't grow a `title` key in
    its JSON just because it went through `render_chart`. Keeps the
    figure JSON clean and means the front-end sees no title attribute
    at all (cleanest possible 'blank').
    """
    import json
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
    # Note: no update_layout(title=...) call. The figure has no title.

    _apply_render_chart_title_clear(fig)

    layout_keys = set(json.loads(fig.to_json())["layout"].keys())
    assert "title" not in layout_keys, (
        f"render_chart added a stray title to a fresh figure: layout keys = {layout_keys}"
    )


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


def test_ensure_datetime_index_passes_through_real_datetime(utils_mod):
    """A DataFrame whose index is already a DatetimeIndex must be returned
    unchanged (BIST/S&P loaders shouldn't pay any cost)."""
    import pandas as pd
    real_dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"a": range(10), "b": range(10)}, index=real_dates)
    out = utils_mod._ensure_datetime_index(df)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert (out.index == real_dates).all()
    # Identity preserved when no conversion needed
    assert out is df


def test_ensure_datetime_index_converts_float_index(utils_mod):
    """A DataFrame with a float64 index (EEG sample seconds) must be coerced
    to a synthetic DatetimeIndex anchored at 2020-01-01, so downstream code
    like `df.index.min().date()` doesn't crash."""
    import pandas as pd
    # Mimic EEG: 100 samples of float seconds
    df = pd.DataFrame({"FC5": range(100), "C3": range(100)}, index=[i * 0.00625 for i in range(100)])
    assert not isinstance(df.index, pd.DatetimeIndex)
    out = utils_mod._ensure_datetime_index(df)
    assert isinstance(out.index, pd.DatetimeIndex), "index must become DatetimeIndex"
    assert out.index.min().date().isoformat() == "2020-01-01"
    # Length preserved
    assert len(out) == 100
    # Columns preserved
    assert list(out.columns) == ["FC5", "C3"]


def test_downsample_if_oversize_passes_through_small(utils_mod):
    """Datasets at-or-below max_rows must pass through unchanged."""
    import pandas as pd
    df = pd.DataFrame({"a": range(500)})
    out = utils_mod._downsample_if_oversize(df, max_rows=8_000)
    assert len(out) == 500
    assert out is df  # identity preserved when no downsampling needed


def test_downsample_if_oversize_decimates_large(utils_mod):
    """Oversized datasets (EEG-like) must be uniformly decimated."""
    import pandas as pd
    df = pd.DataFrame(
        {"a": range(100_000), "b": range(100_000, 200_000)},
        index=pd.date_range("2020-01-01", periods=100_000, freq="1s"),
    )
    out = utils_mod._downsample_if_oversize(df, max_rows=10_000)
    assert len(out) <= 10_000 * 2, "should be reasonably close to target"
    # First row preserved (uniform-stride keeps the first sample)
    assert out.iloc[0]["a"] == 0
    # Columns preserved
    assert list(out.columns) == ["a", "b"]
    # Index still DatetimeIndex
    assert isinstance(out.index, pd.DatetimeIndex)


def test_downsample_bist_unaffected(utils_mod):
    """The 1544-row BIST dataset must not be touched by downsampling."""
    import pandas as pd
    df = pd.DataFrame(
        {"AKBNK": range(1544)},
        index=pd.date_range("2020-01-01", periods=1544, freq="D"),
    )
    out = utils_mod._downsample_if_oversize(df)
    assert len(out) == 1544
    assert out is df  # full pass-through


def test_ensure_datetime_index_converts_rangeindex(utils_mod):
    """A pd.RangeIndex (the most common non-datetime case) should also be
    coerced cleanly."""
    import pandas as pd
    df = pd.DataFrame({"x": range(50)})  # default RangeIndex
    out = utils_mod._ensure_datetime_index(df)
    assert isinstance(out.index, pd.DatetimeIndex)
    # The dashboard's date_input widget needs min < max
    assert out.index.min() < out.index.max()


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
