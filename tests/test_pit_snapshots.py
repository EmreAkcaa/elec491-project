"""Tests for PHASE 3 (slim) — PIT snapshot precompute + loaders.

Two layers of coverage:

1. **Pipeline stage (`src/pit_snapshots.py`)**: writes the expected
   directory layout, schema, and snapshot counts when run against a
   small synthetic returns matrix.

2. **Loaders (`app/utils.py:load_pit_*_snapshot` + `snap_to_nearest_snapshot`)**:
   read what the pipeline wrote, snap user-picked dates to the nearest
   grid date, return empty DataFrames cleanly on misses.

The tests use temporary directories + monkeypatched config so they
don't touch the real `data/<universe>/results/` tree.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Pipeline stage unit tests (no Streamlit)
# ---------------------------------------------------------------------------


def _make_synthetic_returns(
    n_days: int = 500, n_tickers: int = 20, seed: int = 42,
) -> pd.DataFrame:
    """Plausible log-returns matrix for testing PIT pipeline + loaders.

    Date index is business-day; correlations are induced by a single
    common factor + idiosyncratic noise so the resulting correlation
    matrix has structure (not pure noise → meaningless MST).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2020-01-02", periods=n_days)
    common = rng.normal(0, 0.01, size=n_days)
    idio = rng.normal(0, 0.015, size=(n_days, n_tickers))
    data = common[:, None] * 0.6 + idio * 0.8
    cols = [f"T{i:03d}" for i in range(n_tickers)]
    return pd.DataFrame(data, index=dates, columns=cols)


def test_build_date_grid_skips_short_history():
    """Date grid should skip the first `window + 30` days."""
    from src.pit_snapshots import _build_date_grid
    returns = _make_synthetic_returns(n_days=400)
    grid = _build_date_grid(returns, window=252, stride_days=5)
    # 400 - 282 = 118 eligible days; at 5-business-day stride ≈ 24 snapshots.
    assert 15 < len(grid) < 30
    # First grid date must be at index >= window+30.
    assert returns.index.get_loc(grid[0]) >= 282


def test_build_date_grid_empty_when_no_history():
    """Returns DataFrame too short → empty grid (no crash)."""
    from src.pit_snapshots import _build_date_grid
    returns = _make_synthetic_returns(n_days=100)  # < window + 30
    grid = _build_date_grid(returns, window=252, stride_days=5)
    assert grid == []


def test_build_mst_edges_schema():
    """MST edges DF must have source / target / distance / correlation."""
    from src.pit_snapshots import _build_mst_edges
    returns = _make_synthetic_returns(n_days=400, n_tickers=20)
    corr = returns.tail(252).corr()
    edges = _build_mst_edges(corr)
    assert edges is not None
    assert set(edges.columns) == {"source", "target", "distance", "correlation"}
    # MST on N nodes has N-1 edges.
    assert len(edges) == 20 - 1
    # Sanity: distance and correlation relate via d = sqrt(2*(1-rho)).
    for _, row in edges.iterrows():
        recovered_d = np.sqrt(2 * (1 - row["correlation"]))
        assert abs(row["distance"] - recovered_d) < 1e-6


def test_top_dislocations_count_and_sort():
    """Top dislocations must be sorted ascending by correlation, capped at N."""
    from src.pit_snapshots import _top_dislocations
    returns = _make_synthetic_returns(n_days=300, n_tickers=15)
    corr = returns.tail(252).corr()
    dis = _top_dislocations(corr, n=10)
    assert len(dis) == 10
    assert list(dis["correlation"]) == sorted(dis["correlation"])
    # Schema match.
    assert set(dis.columns) == {"ticker_a", "ticker_b", "correlation"}


def test_run_pit_snapshots_writes_expected_layout(tmp_path, monkeypatch):
    """End-to-end pipeline stage: writes pit_corr/, pit_mst/, pit_dislocation/
    subdirectories with the right file count + format."""
    from src import pit_snapshots
    from src.pit_snapshots import run_pit_snapshots

    # Force a small market into the precomputed set so the test exercises
    # the write path (otherwise the gate short-circuits).
    monkeypatch.setattr(pit_snapshots, "_PRECOMPUTE_MARKETS", {"testmkt"})
    monkeypatch.setitem(pit_snapshots._STRIDE_BUSINESS_DAYS, "testmkt", 10)
    monkeypatch.setitem(pit_snapshots._FLOAT_DTYPE, "testmkt", "float32")

    # Build a fake config-like object with the .market.market_id +
    # .data_processed + .data_results properties the stage uses.
    returns = _make_synthetic_returns(n_days=400, n_tickers=15)
    processed = tmp_path / "processed"
    results = tmp_path / "results"
    processed.mkdir()
    results.mkdir()
    returns.to_parquet(processed / "log_returns.parquet")

    class _FakeMarket:
        market_id = "testmkt"

    class _FakeConfig:
        market = _FakeMarket()
        data_processed = processed
        data_results = results

    run_pit_snapshots(_FakeConfig())

    # Directory structure.
    corr_dir = results / "pit_corr" / "w252"
    mst_dir = results / "pit_mst" / "w252"
    dis_dir = results / "pit_dislocation" / "w252"
    assert corr_dir.is_dir()
    assert mst_dir.is_dir()
    assert dis_dir.is_dir()

    corr_files = sorted(corr_dir.glob("*.parquet"))
    assert len(corr_files) > 0
    # Filename ISO-format sanity.
    for p in corr_files:
        stem = p.stem
        assert len(stem) == 10 and stem[4] == "-" and stem[7] == "-"

    # One sample file must be readable + have the right shape.
    sample = pd.read_parquet(corr_files[0])
    assert sample.shape == (15, 15)
    # float32 storage per the test universe's dtype override.
    assert sample.dtypes.iloc[0] == np.float32

    # MST + dislocation files match the corr count (1:1 unless an edge
    # case excludes one).
    assert abs(len(list(mst_dir.glob("*.csv"))) - len(corr_files)) <= 1
    assert abs(len(list(dis_dir.glob("*.parquet"))) - len(corr_files)) <= 1


def test_run_pit_snapshots_skipped_for_non_precomputed_market(tmp_path):
    """Non-precomputed markets (e.g., bist_usd) must skip silently —
    log message, no files written, no crash."""
    from src.pit_snapshots import run_pit_snapshots

    class _FakeMarket:
        market_id = "bist_usd"

    class _FakeConfig:
        market = _FakeMarket()
        data_processed = tmp_path / "processed"
        data_results = tmp_path / "results"

    # Should not raise. Output dirs aren't even created.
    run_pit_snapshots(_FakeConfig())
    assert not (tmp_path / "results" / "pit_corr").exists()


# ---------------------------------------------------------------------------
# Loader tests (Streamlit cache layer)
# ---------------------------------------------------------------------------


def _has_bist_pit_data() -> bool:
    """Quick gate: skip loader tests if the live BIST PIT data isn't present."""
    return (_REPO_ROOT / "data" / "bist" / "results" / "pit_corr" / "w252").is_dir()


needs_pit_data = pytest.mark.skipif(
    not _has_bist_pit_data(),
    reason="BIST PIT snapshot grid not on disk; run pipeline with PIT stage.",
)


@needs_pit_data
def test_pit_snapshot_dates_returns_sorted_list():
    """pit_snapshot_dates must enumerate the directory and return a sorted
    list of ISO date strings."""
    import streamlit as st
    from utils import _pit_snapshot_dates
    # Clear cache so the function actually reads disk each time.
    st.cache_data.clear()
    dates = _pit_snapshot_dates("bist", 252, "corr")
    assert isinstance(dates, list)
    assert len(dates) > 100  # we expect 263 from the pipeline run
    assert dates == sorted(dates)
    assert all(len(d) == 10 and d[4] == "-" for d in dates)


@needs_pit_data
def test_snap_to_nearest_snapshot_picks_closest(monkeypatch):
    """snap_to_nearest_snapshot must return the date with min |diff|."""
    import streamlit as st
    from utils import snap_to_nearest_snapshot
    st.cache_data.clear()

    # Force current_universe to "bist" for this test.
    import utils
    monkeypatch.setattr(utils, "current_universe", lambda: "bist")

    # Pick an arbitrary mid-history date; expect a snapshot within 5 days
    # (5-business-day stride).
    requested = pd.Timestamp("2023-03-15")
    snapped = snap_to_nearest_snapshot(requested, window=252, kind="corr")
    assert snapped is not None
    assert abs((pd.Timestamp(snapped) - requested).days) <= 7


@needs_pit_data
def test_load_pit_snapshot_returns_matrix(monkeypatch):
    """load_pit_snapshot must return the correlation matrix when the
    file exists and an empty DataFrame on miss."""
    import streamlit as st
    from utils import _pit_snapshot_dates, load_pit_snapshot
    import utils
    st.cache_data.clear()
    monkeypatch.setattr(utils, "current_universe", lambda: "bist")

    dates = _pit_snapshot_dates("bist", 252, "corr")
    assert dates, "fixture data missing"
    df = load_pit_snapshot(252, dates[0])
    # Square correlation matrix.
    assert df.shape[0] == df.shape[1]
    # Diagonal must be 1 (correlation of ticker with itself).
    diag = np.diag(df.values)
    np.testing.assert_allclose(diag, np.ones_like(diag), rtol=1e-5)

    # Missing file → empty DataFrame, no exception.
    miss = load_pit_snapshot(252, "1900-01-01")
    assert miss.empty


@needs_pit_data
def test_load_pit_mst_snapshot_schema(monkeypatch):
    """load_pit_mst_snapshot must return the expected edges schema."""
    import streamlit as st
    from utils import _pit_snapshot_dates, load_pit_mst_snapshot
    import utils
    st.cache_data.clear()
    monkeypatch.setattr(utils, "current_universe", lambda: "bist")

    dates = _pit_snapshot_dates("bist", 252, "mst")
    assert dates
    edges = load_pit_mst_snapshot(252, dates[0])
    assert set(edges.columns) == {"source", "target", "distance", "correlation"}
    # MST on 73 BIST tickers has 72 edges.
    assert len(edges) == 72


@needs_pit_data
def test_load_pit_dislocation_snapshot_schema(monkeypatch):
    """load_pit_dislocation_snapshot must return ranked top-20 pairs."""
    import streamlit as st
    from utils import _pit_snapshot_dates, load_pit_dislocation_snapshot
    import utils
    st.cache_data.clear()
    monkeypatch.setattr(utils, "current_universe", lambda: "bist")

    dates = _pit_snapshot_dates("bist", 252, "dislocation")
    assert dates
    dis = load_pit_dislocation_snapshot(252, dates[0])
    assert set(dis.columns) == {"ticker_a", "ticker_b", "correlation"}
    assert len(dis) == 20
    # Sorted ascending by correlation.
    assert list(dis["correlation"]) == sorted(dis["correlation"])
