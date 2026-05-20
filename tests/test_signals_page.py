"""Tests for the Signals page (`app/signals.py` + `app/views/06_signals.py`).

Coverage:
  1. Page renders on BIST + S&P 500 without exceptions.
  2. Page warns-out gracefully on EEG (no pair-trading capability).
  3. Cross-asset section appears on BIST only.
  4. Look-ahead audit: leaderboard at date D does NOT use data after D.
  5. Cross-page state survival: pa_ticker_a/b NOT clobbered by sig_ticker_a/b.
  6. Capability gating: when has_pair_trading=False the warning fires and
     no other content renders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# AppTest is only relevant when streamlit is installed (it is, but be
# defensive for environments that skip the streamlit extra).
try:
    from streamlit.testing.v1 import AppTest  # noqa: F401
    _HAS_APPTEST = True
except ImportError:  # pragma: no cover
    _HAS_APPTEST = False


pytestmark = pytest.mark.skipif(not _HAS_APPTEST, reason="streamlit AppTest unavailable")


_VIEW = "app/views/06_signals.py"


def _at(universe: str, *, basis: str = "try"):
    """Construct an AppTest for the Signals page on `universe`.

    Pre-seeds session_state with dataset/universe/basis so the page's
    `current_universe()` resolves correctly.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_VIEW, default_timeout=60)
    at.session_state["dataset"] = (
        "bist" if universe.startswith("bist") else universe
    )
    at.session_state["universe"] = universe
    at.session_state["bist_basis"] = basis
    return at


# ---------------------------------------------------------------------------
# Render smoke
# ---------------------------------------------------------------------------


def test_signals_renders_on_bist():
    at = _at("bist")
    at.run()
    assert not at.exception, (
        "Signals page raised on BIST: "
        f"{[str(e.value) for e in at.exception]}"
    )
    headers = [s.value for s in at.subheader]
    # The first header is dynamic ("Trades to put on as of YYYY-MM-DD").
    assert any(h.startswith("Trades to put on as of") for h in headers), headers
    assert "Cluster consensus" in headers, headers
    assert "All ranked pairs" in headers, headers
    assert "Pair explorer" in headers, headers
    assert "Cross-asset β shift" in headers, headers


def test_signals_renders_on_bist_usd():
    """Base-currency flip (TRY→USD) shouldn't break the page."""
    at = _at("bist_usd", basis="usd")
    at.run()
    assert not at.exception, (
        f"Signals page raised on BIST(USD): "
        f"{[str(e.value) for e in at.exception]}"
    )
    headers = [s.value for s in at.subheader]
    assert any(h.startswith("Trades to put on as of") for h in headers), headers


def test_signals_renders_on_sp500_without_cross_asset():
    """S&P 500 has pair trading but no cross-asset section (BIST-only)."""
    at = _at("sp500")
    at.run()
    assert not at.exception, (
        f"Signals page raised on S&P 500: "
        f"{[str(e.value) for e in at.exception]}"
    )
    headers = [s.value for s in at.subheader]
    assert any(h.startswith("Trades to put on as of") for h in headers), headers
    assert "Pair explorer" in headers, headers
    assert "Cross-asset β shift" not in headers, (
        f"Cross-asset section leaked into S&P render: {headers}"
    )


def test_signals_warns_out_on_eeg():
    """EEG has has_pair_trading=False → page bails out with a warning."""
    at = _at("eeg_motor_left_right")
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1, "Expected a capability-gate warning"
    msg = at.warning[0].value.lower()
    assert "pair-trading" in msg or "finance-only" in msg, msg
    # And no other content should render.
    assert len(at.subheader) == 0, (
        f"EEG warn-out path should produce no section headers; got {[s.value for s in at.subheader]}"
    )


# ---------------------------------------------------------------------------
# Look-ahead audit
# ---------------------------------------------------------------------------


def test_leaderboard_at_historical_date_doesnt_use_future_data():
    """Set as-of to an older date and verify the leaderboard's
    `current_z` values for one pair match a manually-computed past-only
    Z-score (no future leakage).
    """
    # Read BIST panel directly — bypassing `load_adj_close()`'s
    # `current_universe()` resolver, which can be polluted across tests by
    # other suite members that switch universe via session_state.
    from app.signals import _build_leaderboard, _DEFAULT_LOOKBACK, _DEFAULT_ZWINDOW
    from src.pair_dislocation import compute_spread, compute_zscore

    adj_path = _REPO_ROOT / "data" / "bist" / "processed" / "adj_close.parquet"
    candidates_path = _REPO_ROOT / "data" / "bist" / "results" / "dislocation_candidates.csv"
    if not adj_path.exists() or not candidates_path.exists():
        pytest.skip("BIST artifacts missing")
    adj = pd.read_parquet(adj_path)
    candidates = pd.read_csv(candidates_path)
    # Pick a historical date well inside the panel.
    as_of_iso = "2024-01-15"
    # Unique cache_key per test run to dodge @st.cache_data pollution.
    cache_key = f"signals_test_historical:{adj.shape}"

    # Clear the cache for `_build_leaderboard` so we get a fresh compute.
    try:
        _build_leaderboard.clear()
    except Exception:
        pass

    lb = _build_leaderboard(
        adj, cache_key, candidates, as_of_iso,
        _DEFAULT_LOOKBACK, _DEFAULT_ZWINDOW, 2.0, 0.5,
    )
    assert not lb.empty, "leaderboard empty at historical date"

    # Pick first row and manually recompute using past-only slice.
    row = lb.iloc[0]
    ta, tb = row["ticker_a"], row["ticker_b"]
    expected_z = (
        compute_zscore(
            compute_spread(
                adj.loc[:as_of_iso], ta, tb, lookback=_DEFAULT_LOOKBACK,
            )[0],
            window=_DEFAULT_ZWINDOW,
        )
        .dropna()
        .iloc[-1]
    )
    actual_z = row["current_z"]
    assert abs(float(actual_z) - float(expected_z)) < 1e-6, (
        f"current_z at {as_of_iso} differs from past-only recompute: "
        f"actual={actual_z}, expected={expected_z}"
    )


def test_leaderboard_doesnt_change_when_future_data_changes():
    """Verify the leaderboard at historical date D is identical whether
    we feed it the full panel or just the slice [:D].
    """
    from app.signals import _build_leaderboard, _DEFAULT_LOOKBACK, _DEFAULT_ZWINDOW

    adj_path = _REPO_ROOT / "data" / "bist" / "processed" / "adj_close.parquet"
    candidates_path = _REPO_ROOT / "data" / "bist" / "results" / "dislocation_candidates.csv"
    if not adj_path.exists() or not candidates_path.exists():
        pytest.skip("BIST artifacts missing")
    adj = pd.read_parquet(adj_path)
    candidates = pd.read_csv(candidates_path)
    as_of_iso = "2023-06-15"
    as_of_ts = pd.Timestamp(as_of_iso)
    cache_key_full = f"signals_test_noleak:full:{adj.shape}"
    cache_key_sliced = f"signals_test_noleak:sliced:{adj.shape}"

    try:
        _build_leaderboard.clear()
    except Exception:
        pass

    lb_full = _build_leaderboard(
        adj, cache_key_full, candidates, as_of_iso,
        _DEFAULT_LOOKBACK, _DEFAULT_ZWINDOW, 2.0, 0.5,
    )
    lb_sliced = _build_leaderboard(
        adj.loc[:as_of_ts], cache_key_sliced, candidates, as_of_iso,
        _DEFAULT_LOOKBACK, _DEFAULT_ZWINDOW, 2.0, 0.5,
    )
    if lb_full.empty or lb_sliced.empty:
        pytest.skip("leaderboard empty")
    # Compare the current_z columns for the common pair rows.
    pairs_full = lb_full.set_index(["ticker_a", "ticker_b"])["current_z"]
    pairs_sliced = lb_sliced.set_index(["ticker_a", "ticker_b"])["current_z"]
    common = pairs_full.index.intersection(pairs_sliced.index)
    assert len(common) > 0, "no overlapping pairs between full and sliced runs"
    for idx in common:
        z_full = float(pairs_full.loc[idx])
        z_sliced = float(pairs_sliced.loc[idx])
        assert abs(z_full - z_sliced) < 1e-6, (
            f"leaderboard leaked future data for pair {idx}: "
            f"full-panel z={z_full}, sliced-panel z={z_sliced}"
        )


def test_trade_direction_sign_convention():
    """When z > 0 (B overpriced vs A), the trade must say SHORT B / LONG A.
    When z < 0, it must say LONG B / SHORT A. This is the spec.

    Caught by manual review 2026-05-19: the cluster-pressure direction
    label was previously inverted. A user following the page would have
    taken the opposite side of every trade. Regression test from this PR
    onward.
    """
    from app.signals import _todays_actions
    # Synthetic leaderboard: 1 short-entry (z=+3) and 1 long-entry (z=-3).
    lb = pd.DataFrame([
        {"ticker_a": "AAA", "ticker_b": "BBB", "current_z": 3.0,
         "status": "irrelevant", "half_life": 50.0, "correlation": 0.7,
         "sector_a": "", "sector_b": "", "days_since_last_signal": 0},
        {"ticker_a": "CCC", "ticker_b": "DDD", "current_z": -3.0,
         "status": "irrelevant", "half_life": 50.0, "correlation": 0.7,
         "sector_a": "", "sector_b": "", "days_since_last_signal": 0},
    ])
    longs, shorts, trades = _todays_actions(lb, entry_z=2.0)
    # z=+3 on AAA/BBB → short B, long A → LONG AAA, SHORT BBB
    assert longs.get("AAA") == 1, f"AAA should be long ×1; got {longs}"
    assert shorts.get("BBB") == 1, f"BBB should be short ×1; got {shorts}"
    # z=-3 on CCC/DDD → long B, short A → LONG DDD, SHORT CCC
    assert longs.get("DDD") == 1, f"DDD should be long ×1; got {longs}"
    assert shorts.get("CCC") == 1, f"CCC should be short ×1; got {shorts}"
    # Trade strings must match the LONG/SHORT semantics for the spread
    actions = {t["pair"]: t["action"] for t in trades}
    assert actions["AAA/BBB"] == "LONG AAA  /  SHORT BBB", actions
    assert actions["CCC/DDD"] == "LONG DDD  /  SHORT CCC", actions


def test_cluster_consensus_sign_convention():
    """Cluster view's `action` column must agree with _todays_actions.

    If a ticker shows up in 2+ short-entries as ticker_a, it should be
    LONG (the buy leg). If it shows up as ticker_b in those same entries,
    it should be SHORT.
    """
    from app.signals import _ticker_pressure_view
    lb = pd.DataFrame([
        # KCHOL appears as ticker_a in 3 short-entries with z > 0:
        # so KCHOL is the LONG leg → consensus should be LONG.
        {"ticker_a": "KCHOL", "ticker_b": "KRDMD", "current_z": 3.0,
         "status": "", "half_life": 50.0, "correlation": 0.7,
         "sector_a": "Conglo", "sector_b": "Steel", "days_since_last_signal": 0},
        {"ticker_a": "KCHOL", "ticker_b": "AKBNK", "current_z": 2.5,
         "status": "", "half_life": 50.0, "correlation": 0.7,
         "sector_a": "Conglo", "sector_b": "Banking", "days_since_last_signal": 0},
        # KRDMD appears as ticker_b in 2 pairs with z > 0 → SHORT leg.
        {"ticker_a": "PETKM", "ticker_b": "KRDMD", "current_z": 2.8,
         "status": "", "half_life": 50.0, "correlation": 0.7,
         "sector_a": "Chem", "sector_b": "Steel", "days_since_last_signal": 0},
    ])
    p = _ticker_pressure_view(lb, threshold=1.5)
    # KCHOL in 2 hot pairs as buy leg → LONG
    kchol = p[p["ticker"] == "KCHOL"]
    assert not kchol.empty, p
    assert kchol.iloc[0]["action"] == "LONG", f"KCHOL action wrong: {kchol.iloc[0].to_dict()}"
    assert kchol.iloc[0]["consensus_z"] > 0, kchol.iloc[0].to_dict()
    # KRDMD in 2 hot pairs as sell leg → SHORT
    krdmd = p[p["ticker"] == "KRDMD"]
    assert not krdmd.empty, p
    assert krdmd.iloc[0]["action"] == "SHORT", f"KRDMD action wrong: {krdmd.iloc[0].to_dict()}"
    assert krdmd.iloc[0]["consensus_z"] < 0, krdmd.iloc[0].to_dict()


def test_state_at_replays_state_machine_purely_past_only():
    """Direct test of `_state_at`: walking the same z-history past a fixed
    as-of date with extra future values appended must give the same answer."""
    from app.signals import _state_at

    dates = pd.bdate_range("2024-01-01", periods=100, freq="B")
    z = pd.Series([0.0] * 30 + [2.5, 2.0, 1.5, 0.0, -0.3, -0.4] * 5 + [0.0] * 40, index=dates)
    # Pick a date 50 in (well into the structured portion).
    as_of = dates[50]
    status_a, last_a = _state_at(z, as_of, entry_z=2.0, exit_z=0.5)

    # Append future data (5σ shock) and re-run with the same as_of.
    extended = pd.concat([z, pd.Series([5.0] * 100, index=pd.bdate_range(z.index[-1] + pd.Timedelta(days=1), periods=100, freq="B"))])
    status_b, last_b = _state_at(extended, as_of, entry_z=2.0, exit_z=0.5)

    assert status_a == status_b, (
        f"_state_at leaked future data: {status_a} vs {status_b}"
    )
    assert last_a == last_b


# ---------------------------------------------------------------------------
# Cross-page state safety
# ---------------------------------------------------------------------------


def test_signals_page_uses_distinct_widget_keys_from_pair_analysis():
    """`pa_ticker_a` and `sig_ticker_a` are independent — picking a pair
    on Signals shouldn't move Pair Analysis's selection."""
    at = _at("bist")
    at.session_state["pa_ticker_a"] = "AKBNK"
    at.session_state["pa_ticker_b"] = "ISCTR"
    at.session_state["sig_ticker_a"] = "TUPRS"
    at.session_state["sig_ticker_b"] = "BIMAS"
    at.run()
    # Both sets of keys persist independently.
    assert at.session_state["pa_ticker_a"] == "AKBNK"
    assert at.session_state["pa_ticker_b"] == "ISCTR"
    assert at.session_state["sig_ticker_a"] == "TUPRS"


# ---------------------------------------------------------------------------
# Main page wiring
# ---------------------------------------------------------------------------


def test_main_includes_signals_in_visible_pages_for_bist():
    """`main.py` page list contains Signals between Time Machine and Pair
    Analysis on BIST (capability-gated)."""
    src = (_REPO_ROOT / "app" / "main.py").read_text()
    assert "views/06_signals.py" in src, "Signals not wired into main.py _PAGE_PATHS"
    assert '"Signals"' in src, "Signals not added to visible_titles"


def _bist_grid() -> list[str]:
    """Read the BIST walk-forward grid directly from disk.

    Bypasses ``walkforward_signals_dates()`` because that loader is
    universe-keyed via ``current_universe()`` — and the full test
    suite can leave a previous test's universe in session state, so
    the loader can return the S&P grid instead of BIST when run in
    aggregate. Direct-read is deterministic across test ordering.
    """
    base = _REPO_ROOT / "data" / "bist" / "results" / "walkforward_signals" / "w60"
    if not base.exists():
        return []
    return sorted(f.stem for f in base.glob("*.parquet"))


def test_leaderboard_loads_from_walkforward_snapshot_when_available():
    """On a known walk-forward grid date the leaderboard should match
    the snapshot row-for-row (renamed current_zscore → current_z)."""
    import datetime

    grid = _bist_grid()
    if not grid:
        pytest.skip("walk-forward snapshots not on disk — run pipeline first")
    # Pick a date well into the grid.
    target_iso = grid[len(grid) // 2]
    target_date = datetime.date.fromisoformat(target_iso)

    at = _at("bist")
    at.session_state["sig_date"] = target_date
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    # Read snapshot directly to avoid universe-cache pollution.
    snap_path = (
        _REPO_ROOT / "data" / "bist" / "results"
        / "walkforward_signals" / "w60" / f"{target_iso}.parquet"
    )
    if not snap_path.exists():
        pytest.skip(f"snapshot {target_iso}.parquet missing")
    snap = pd.read_parquet(snap_path)
    if snap.empty:
        pytest.skip(f"snapshot empty for {target_iso}")

    # The leaderboard is the 3rd dataframe rendered:
    #   [0] = today's trades table (subset of leaderboard, |z|≥entry)
    #   [1] = cluster consensus
    #   [2] = all ranked pairs (the leaderboard)
    # Verify the all-ranked-pairs table's pairs match the snapshot.
    if len(at.dataframe) < 3:
        pytest.skip(f"page rendered only {len(at.dataframe)} dataframes")
    rendered_lb = at.dataframe[2].value
    if not isinstance(rendered_lb, pd.DataFrame) or rendered_lb.empty:
        pytest.skip("leaderboard dataframe empty")

    snap_pairs = set(zip(snap["ticker_a"], snap["ticker_b"]))
    rendered_pairs = set(zip(rendered_lb["ticker_a"], rendered_lb["ticker_b"]))
    assert snap_pairs == rendered_pairs, (
        f"Leaderboard pair set differs from snapshot at {target_iso}: "
        f"snap-only={snap_pairs - rendered_pairs}, "
        f"rendered-only={rendered_pairs - snap_pairs}"
    )


def test_drilling_in_carries_as_of_date_to_pair_analysis():
    """After clicking the drill-in button, pa_ticker_a/b and
    pa_as_of_date are written to session_state with the snapped date."""
    import datetime

    grid = _bist_grid()
    if not grid:
        pytest.skip("walk-forward snapshots not on disk")
    target_iso = grid[len(grid) // 2]
    target_date = datetime.date.fromisoformat(target_iso)

    at = _at("bist")
    at.session_state["sig_date"] = target_date
    at.run()
    assert not at.exception

    # Find the drill-in button if rendered; if not (no active trades at
    # this date), this test is moot — skip.
    drill_buttons = [b for b in at.button if b.key == "sig_drill_btn"]
    if not drill_buttons:
        pytest.skip(f"No drill-in button at {target_iso} (no active trades)")

    # Find the pair selectbox; pick its current value.
    drill_select = [s for s in at.selectbox if s.key == "sig_drill_pair"]
    if not drill_select:
        pytest.skip("Drill selectbox not found")
    chosen_pair = drill_select[0].value
    assert chosen_pair and "/" in chosen_pair, chosen_pair
    expected_ta, expected_tb = chosen_pair.split("/")

    # Click the drill-in button. switch_page raises a special internal
    # exception in AppTest; we wrap and assert session_state regardless.
    try:
        drill_buttons[0].click().run()
    except Exception:
        pass

    assert at.session_state["pa_ticker_a"] == expected_ta, (
        f"pa_ticker_a = {at.session_state.get('pa_ticker_a')!r}, expected {expected_ta!r}"
    )
    assert at.session_state["pa_ticker_b"] == expected_tb
    pa_as_of = at.session_state["pa_as_of_date"]
    # Compare as ISO strings (pa_as_of may be a date object).
    pa_as_of_iso = (
        pa_as_of.isoformat() if hasattr(pa_as_of, "isoformat") else str(pa_as_of)
    )
    assert pa_as_of_iso == target_iso, (
        f"pa_as_of_date = {pa_as_of_iso!r}, expected {target_iso!r}"
    )


def test_main_hides_signals_on_eeg_capability():
    """The visible_titles build only appends Signals when has_pair_trading."""
    src = (_REPO_ROOT / "app" / "main.py").read_text()
    # The pattern should be: Signals appended inside an `if has_pair_trading:` block.
    # Verify by string presence: the visible_titles append for Signals must be
    # inside the same conditional block as Pair Analysis (gated on
    # has_pair_trading), so EEG (has_pair_trading=False) skips both.
    assert "has_pair_trading" in src
    # Crude but effective: check that Signals appears in a block that mentions
    # has_pair_trading nearby.
    sig_idx = src.find('visible_titles.append("Signals")')
    pair_idx = src.find('visible_titles.append("Pair Analysis")')
    cap_idx = src.find("has_pair_trading", 0, sig_idx)
    assert sig_idx > 0, "Signals not in visible_titles"
    assert pair_idx > 0, "Pair Analysis not in visible_titles"
    assert cap_idx > 0, "has_pair_trading guard missing before Signals append"
    # They should be adjacent (Signals right before Pair Analysis).
    assert abs(sig_idx - pair_idx) < 200, (
        "Signals and Pair Analysis appends should be in the same block"
    )
