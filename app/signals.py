"""Signals page — honest information feed.

Surfaces three causal lenses on what the correlation/dislocation network
is saying RIGHT NOW (or on any historical date):

  1. **Active pair signals leaderboard** — for every precomputed top-20
     dislocation pair, recompute the Z-score AS-OF the user-picked date
     and report the current state machine status (long-entry candidate,
     in-position, flat, etc.). Sortable, color-coded.

  2. **Pair signal explorer** — pick any 2 tickers from the FULL
     universe, recompute spread + Z-score + signal history live. Reuses
     the same math the Pair Analysis page does. Defaults to the top-1
     dislocation candidate at the picked date.

  3. **Cross-asset β breakout** (BIST only) — top-N BIST stocks whose
     252-day rolling correlation with USD/TRY or Gold has deviated most
     from its full-period historical baseline. Plain-English
     interpretation per row. Reads ``cross_asset_summary.parquet`` +
     ``cross_asset_corr_rolling_*.parquet`` from the Phase X stage.

A 4th section documents that transfer-entropy lead-lag signals are
precomputed but not yet surfaced as signals — that's the next
page-shaped piece of work.

Look-ahead guarantees:
  * Section 1 computes Z-score with `compute_zscore(spread.loc[:date], …)`
    — past-only by construction.
  * Section 2 calls the same Pair Analysis helpers with no `.shift(-K)`.
  * Section 3 reads precomputed rolling correlations which are
    left-aligned by date (no centered rolling, no future data).
  * Section 4 is just a link.

Capability gating:
  * Whole page hidden when ``has_pair_trading=False`` (EEG case). The
    nav item in main.py is also hidden in that case; this is
    defence-in-depth against deep-links.
  * Section 3 (cross-asset) shown only when ``current_universe() == "bist"``.

This page deliberately does NOT include a backtest equity curve or any
"strategy P&L" framing. It's an information feed, not an
auto-trading recommendation. The honest framing is in the page subtitle
and the per-section captions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from universe_registry import get_universe
from utils import (
    apply_chart_style,
    current_universe,
    get_colors,
    inject_custom_css,
    load_adj_close,
    load_cluster_assignments,
    load_cross_asset_rolling,
    load_cross_asset_summary,
    load_dislocation_candidates,
    load_log_returns,
    load_walkforward_signals_snapshot,
    page_header,
    render_chart,
    section_header,
    snap_to_preceding_snapshot,
    walkforward_signals_dates,
)
from src.pair_dislocation import (
    compute_half_life,
    compute_spread,
    compute_zscore,
    detect_signals,
    state_at as _state_at,
    trade_direction as _trade_direction,
    STATUS_FLAT as _STATUS_FLAT,
    STATUS_IN_LONG as _STATUS_IN_LONG,
    STATUS_IN_SHORT as _STATUS_IN_SHORT,
    STATUS_LONG_ENTRY as _STATUS_LONG_ENTRY,
    STATUS_NA as _STATUS_NA,
    STATUS_NEAR_ENTRY as _STATUS_NEAR_ENTRY,
    STATUS_SHORT_ENTRY as _STATUS_SHORT_ENTRY,
)


# Defaults pulled from config/settings.yaml:dislocation. The Signals page
# uses the same thresholds as the rest of the pipeline so the leaderboard
# and the precomputed `dislocation_candidates.csv:n_signals` column stay
# semantically aligned. Anyone reading the table on Pair Analysis vs the
# leaderboard here sees the same signal grammar.
_DEFAULT_ZWINDOW = 60
_DEFAULT_ENTRY_Z = 2.0
_DEFAULT_EXIT_Z = 0.5
_DEFAULT_LOOKBACK = 252

# Status constants + _state_at are imported from src.pair_dislocation so the
# pipeline stage (src/walk_forward_signals.py) and this page share one
# canonical implementation. Locally renamed with `_STATUS_*` aliases to
# minimise churn in the rest of this file.


# ── Cached helpers ──────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _pair_zscore_history(
    _adj_close: pd.DataFrame,
    cache_key: str,
    ticker_a: str,
    ticker_b: str,
    lookback: int,
    zwindow: int,
) -> tuple[pd.Series, pd.Series, float, float]:
    """Recompute the (spread, zscore, beta, half_life) tuple for one pair.

    Look-ahead: none. ``compute_spread`` fits OLS on the LAST ``lookback``
    days of the joined history; ``compute_zscore`` is a left-aligned
    rolling window.
    """
    spread, _beta, _intercept = compute_spread(
        _adj_close, ticker_a, ticker_b, lookback=lookback,
    )
    zscore = compute_zscore(spread, window=zwindow)
    half_life = compute_half_life(spread)
    return spread, zscore, float(_beta), float(half_life)


def _status_to_emoji(status: str) -> str:
    return {
        _STATUS_LONG_ENTRY: "🟢",
        _STATUS_SHORT_ENTRY: "🔴",
        _STATUS_IN_LONG: "🟢",
        _STATUS_IN_SHORT: "🔴",
        _STATUS_NEAR_ENTRY: "🟡",
        _STATUS_FLAT: "⚪",
        _STATUS_NA: "·",
    }.get(status, "·")


@st.cache_data(show_spinner=False)
def _build_leaderboard(
    _adj_close: pd.DataFrame,
    cache_key: str,
    candidates: pd.DataFrame,
    as_of_iso: str,
    lookback: int,
    zwindow: int,
    entry_z: float,
    exit_z: float,
) -> pd.DataFrame:
    """Compute the leaderboard for every pair in ``candidates``.

    Returns a DataFrame ready for display with columns:
      ticker_a, ticker_b, sector_a, sector_b, current_z, status,
      days_since_last_signal, half_life, correlation.

    Look-ahead: none. Each pair's z-score history is recomputed using
    only data up to ``as_of_iso``; the state machine replay never reads
    forward.
    """
    as_of = pd.Timestamp(as_of_iso)
    # Slice adj_close once so all pair computations honour the as-of date.
    if as_of not in _adj_close.index:
        # Snap to the nearest past trading day.
        prev = _adj_close.index[_adj_close.index <= as_of]
        if len(prev) == 0:
            return pd.DataFrame()
        as_of = prev[-1]
    sliced = _adj_close.loc[:as_of]

    rows: list[dict] = []
    for _, row in candidates.iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        if ta not in sliced.columns or tb not in sliced.columns:
            continue
        try:
            spread, _beta, _intercept = compute_spread(
                sliced, ta, tb, lookback=lookback,
            )
            zscore = compute_zscore(spread, window=zwindow)
        except Exception:
            continue
        z_last = float(zscore.dropna().iloc[-1]) if not zscore.dropna().empty else float("nan")
        status, last_signal_date = _state_at(zscore, as_of, entry_z, exit_z)
        if last_signal_date is None:
            days_since = None
        else:
            days_since = int((as_of - last_signal_date).days)

        rows.append(
            {
                "ticker_a": ta,
                "ticker_b": tb,
                "sector_a": row.get("sector_a", ""),
                "sector_b": row.get("sector_b", ""),
                "current_z": z_last,
                "status": status,
                "days_since_last_signal": days_since,
                "half_life": row.get("half_life", float("nan")),
                "correlation": row.get("correlation", float("nan")),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _ticker_pressure_view(
    leaderboard: pd.DataFrame,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Collapse the pair leaderboard into a per-ticker action summary.

    A ticker shows up here when it appears in ≥2 pairs with current
    |Z| > ``threshold``. Surfaces dislocations that span multiple
    counterparties (e.g., "X is the buy leg across its whole sector").

    Trade direction derivation:
      spread = log(B) − β · log(A)
      z > 0  → B overpriced relative to A → trade: SHORT B, LONG A
      z < 0  → B underpriced relative to A → trade: LONG B, SHORT A

    So for a ticker X aggregated across pairs:
      X as ticker_a in pair (X, Y) with z=+v: trade is LONG X — contribute +v
      X as ticker_a in pair (X, Y) with z=−v: trade is SHORT X — contribute −v
      X as ticker_b in pair (Y, X) with z=+v: trade is SHORT X — contribute −v
      X as ticker_b in pair (Y, X) with z=−v: trade is LONG X — contribute +v

    Then avg of contributions > 0 → BUY X (LONG signal), < 0 → SELL X (SHORT signal).
    The magnitude tells you how strong/consistent the consensus is across pairs.
    """
    if leaderboard.empty:
        return pd.DataFrame()
    hot = leaderboard[leaderboard["current_z"].abs() > threshold]
    if hot.empty:
        return pd.DataFrame()
    rows: dict[str, dict] = {}
    for _, r in hot.iterrows():
        ta, tb, z = r["ticker_a"], r["ticker_b"], float(r["current_z"])
        # ta contribution: +z (positive z → LONG ta)
        rows.setdefault(
            ta, {"ticker": ta, "sector": r.get("sector_a", ""), "partners": [], "_buy_pressure": []},
        )
        partner_action = "SHORT" if z > 0 else "LONG"
        rows[ta]["partners"].append(f"{partner_action} {tb} (z={z:+.1f})")
        rows[ta]["_buy_pressure"].append(z)
        # tb contribution: -z (positive z → SHORT tb)
        rows.setdefault(
            tb, {"ticker": tb, "sector": r.get("sector_b", ""), "partners": [], "_buy_pressure": []},
        )
        partner_action_ta = "LONG" if z > 0 else "SHORT"
        rows[tb]["partners"].append(f"{partner_action_ta} {ta} (z={z:+.1f})")
        rows[tb]["_buy_pressure"].append(-z)
    out = []
    for v in rows.values():
        if len(v["_buy_pressure"]) < 2:
            continue
        avg = float(np.mean(v["_buy_pressure"]))
        out.append(
            {
                "ticker": v["ticker"],
                "sector": v["sector"],
                "n_pairs": len(v["_buy_pressure"]),
                "consensus_z": round(avg, 2),
                "action": "LONG" if avg > 0 else "SHORT",
                "paired_with": ", ".join(v["partners"]),
            }
        )
    if not out:
        return pd.DataFrame()
    return (
        pd.DataFrame(out)
        .sort_values(
            ["n_pairs", "consensus_z"],
            key=lambda c: c.abs() if c.name == "consensus_z" else c,
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def _todays_actions(leaderboard: pd.DataFrame, entry_z: float) -> tuple[dict[str, int], dict[str, int], list[dict]]:
    """For each pair currently at |z| ≥ entry_z, derive the concrete trade.

    Returns:
      longs   — {ticker: count} aggregated across active entry pairs
      shorts  — {ticker: count}
      trades  — per-pair list of {pair, action_a, action_b, ticker_a, ticker_b, z}
    """
    longs: dict[str, int] = {}
    shorts: dict[str, int] = {}
    trades: list[dict] = []
    if leaderboard.empty:
        return longs, shorts, trades
    active = leaderboard[leaderboard["current_z"].abs() >= entry_z]
    for _, r in active.iterrows():
        ta, tb, z = r["ticker_a"], r["ticker_b"], float(r["current_z"])
        if z >= entry_z:
            # short the spread → SHORT B, LONG A
            shorts[tb] = shorts.get(tb, 0) + 1
            longs[ta] = longs.get(ta, 0) + 1
            trades.append({
                "pair": f"{ta}/{tb}", "z": round(z, 2),
                "action": f"LONG {ta}  /  SHORT {tb}",
                "half_life": r.get("half_life", float("nan")),
            })
        elif z <= -entry_z:
            # long the spread → LONG B, SHORT A
            longs[tb] = longs.get(tb, 0) + 1
            shorts[ta] = shorts.get(ta, 0) + 1
            trades.append({
                "pair": f"{ta}/{tb}", "z": round(z, 2),
                "action": f"LONG {tb}  /  SHORT {ta}",
                "half_life": r.get("half_life", float("nan")),
            })
    return longs, shorts, trades


@st.cache_data(show_spinner=False)
def _cross_asset_breakouts(
    _summary: pd.DataFrame,
    _rolling: pd.DataFrame,
    cache_key: str,
    as_of_iso: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Rank stocks by how far their current rolling β has drifted from baseline.

    Baseline = expanding-window mean of the rolling correlation panel
    UP TO ``as_of_iso`` (past-only, walk-forward). Current = the latest
    rolling value at or before ``as_of_iso``. Deviation = current −
    baseline. ``_summary`` is read only for the sector column now (the
    full-period baseline it carries would leak future data here).

    Output columns: ticker, sector, baseline_corr, current_corr,
    deviation, abs_deviation. Sorted by abs_deviation DESC, top ``top_n``.

    Look-ahead: none. Both inputs are precomputed; the rolling panel is
    left-aligned and we only ever read entries at or before ``as_of_iso``.
    """
    if _summary.empty or _rolling.empty:
        return pd.DataFrame()
    as_of = pd.Timestamp(as_of_iso)
    sliced = _rolling.loc[:as_of]
    if sliced.empty:
        return pd.DataFrame()
    current_row = sliced.iloc[-1]
    # Past-only baseline: per-ticker mean of rolling values up to as_of.
    # This is the "expected" correlation level given history; deviation
    # measures how far today is from that expectation.
    baseline_row = sliced.mean(axis=0, skipna=True)
    sector_map = (
        _summary.set_index("ticker")["sector"].to_dict()
        if "sector" in _summary.columns else {}
    )

    rows = []
    for ticker, current in current_row.dropna().items():
        if ticker not in baseline_row.index:
            continue
        baseline = float(baseline_row.loc[ticker])
        if not np.isfinite(baseline):
            continue
        deviation = float(current) - baseline
        sector = str(sector_map.get(ticker, ""))
        rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "baseline_corr": baseline,
                "current_corr": float(current),
                "deviation": deviation,
                "abs_deviation": abs(deviation),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.sort_values("abs_deviation", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ── Chart helpers ───────────────────────────────────────────────────────


def _signal_zscore_chart(
    zscore: pd.Series,
    signals: pd.DataFrame,
    *,
    entry_z: float,
    exit_z: float,
    pair_label: str,
) -> go.Figure:
    """Z-score line + entry/exit horizontal bands + signal markers."""
    colors = get_colors()
    fig = go.Figure()

    # Z-score line
    fig.add_trace(
        go.Scatter(
            x=zscore.index,
            y=zscore.values,
            mode="lines",
            line=dict(color=colors["primary"], width=1.6),
            name="Z-score",
            hovertemplate="%{x|%Y-%m-%d}<br>Z = %{y:.2f}<extra></extra>",
        )
    )

    # Threshold lines
    for y, dash, name in (
        (entry_z, "dash", f"Entry +{entry_z:.1f}σ"),
        (-entry_z, "dash", f"Entry −{entry_z:.1f}σ"),
        (exit_z, "dot", f"Exit ±{exit_z:.1f}σ"),
        (-exit_z, "dot", None),
    ):
        fig.add_hline(
            y=y, line_dash=dash, line_color=colors["muted"], line_width=1,
            annotation_text=name if name else None,
            annotation_position="right" if name else None,
            annotation_font_size=10,
        )

    # Signal markers
    if not signals.empty:
        long_entries = signals[signals["signal"] == "long_entry"]
        short_entries = signals[signals["signal"] == "short_entry"]
        exits = signals[signals["signal"].isin(["long_exit", "short_exit"])]

        if not long_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_entries["date"],
                    y=long_entries["zscore_value"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up", color=colors["tertiary"],
                        size=11, line=dict(color="white", width=1),
                    ),
                    name="Long entry",
                    hovertemplate="Long entry<br>%{x|%Y-%m-%d}<br>Z = %{y:.2f}<extra></extra>",
                )
            )
        if not short_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=short_entries["date"],
                    y=short_entries["zscore_value"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down", color=colors["secondary"],
                        size=11, line=dict(color="white", width=1),
                    ),
                    name="Short entry",
                    hovertemplate="Short entry<br>%{x|%Y-%m-%d}<br>Z = %{y:.2f}<extra></extra>",
                )
            )
        if not exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=exits["date"],
                    y=exits["zscore_value"],
                    mode="markers",
                    marker=dict(
                        symbol="circle", color=colors["muted"], size=8,
                        line=dict(color="white", width=1),
                    ),
                    name="Exit",
                    hovertemplate="Exit<br>%{x|%Y-%m-%d}<br>Z = %{y:.2f}<extra></extra>",
                )
            )

    fig = apply_chart_style(fig)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis_title=None,
        yaxis_title="Z-score (σ)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ── Main render ─────────────────────────────────────────────────────────


@st.fragment
def render() -> None:
    """Render the Signals page (capability-gated on `has_pair_trading`)."""
    _active = get_universe(current_universe())
    if not getattr(_active, "has_pair_trading", True):
        st.warning(
            f"Signals are pair-trading only — not applicable to **"
            f"{getattr(_active, 'label', 'current')}**."
        )
        return

    inject_custom_css()
    page_header("Signals", "")

    adj_close = load_adj_close()
    candidates = load_dislocation_candidates()
    cache_key = (
        f"{current_universe()}:{adj_close.index.min().date()}:"
        f"{adj_close.index.max().date()}:{adj_close.shape[0]}x{adj_close.shape[1]}"
    )

    if adj_close.empty:
        st.error("No price panel on disk for this universe.")
        return

    _candidates_missing = candidates.empty

    # ── Top control row ────────────────────────────────────────────────
    min_d = adj_close.index.min().date()
    max_d = adj_close.index.max().date()
    if "sig_date" not in st.session_state:
        st.session_state["sig_date"] = max_d

    _c1, _c2, _c3 = st.columns([2, 2, 3])
    with _c1:
        picked_date = st.date_input(
            "As-of date", min_value=min_d, max_value=max_d, key="sig_date",
        )
    with _c2:
        entry_z = st.number_input(
            "Entry |Z|", min_value=0.5, max_value=5.0,
            value=_DEFAULT_ENTRY_Z, step=0.1, key="sig_entry_z",
        )
    with _c3:
        exit_z = st.number_input(
            "Exit |Z|", min_value=0.0, max_value=2.0,
            value=_DEFAULT_EXIT_Z, step=0.1, key="sig_exit_z",
        )

    as_of_iso = pd.Timestamp(picked_date).strftime("%Y-%m-%d")

    # ── Snapshot lookup (walk-forward path) vs live fallback ──────────
    # bist + sp500 have precomputed walk-forward snapshots written by
    # src/walk_forward_signals.py. Other universes (bist_usd, bist_gold,
    # eeg) fall back to the legacy live-compute path against the
    # full-history dislocation_candidates.csv list.
    _grid_dates = walkforward_signals_dates(window=60)
    _has_walkforward = bool(_grid_dates)
    _snapped_iso: Optional[str] = None
    _snapshot_df = pd.DataFrame()
    _used_fallback = False
    if _has_walkforward:
        _snapped_iso = snap_to_preceding_snapshot(picked_date, grid_dates=_grid_dates)
        if _snapped_iso is not None:
            _snapshot_df = load_walkforward_signals_snapshot(_snapped_iso, window=60)

    if not _snapshot_df.empty:
        # Walk-forward path: rename current_zscore → current_z so the
        # display code that already exists keeps working unchanged.
        leaderboard = _snapshot_df.rename(columns={"current_zscore": "current_z"}).copy()
    elif _candidates_missing:
        leaderboard = pd.DataFrame()
    else:
        # Fallback: live recompute against the full-history pair list.
        _used_fallback = True
        leaderboard = _build_leaderboard(
            adj_close, cache_key, candidates, as_of_iso,
            _DEFAULT_LOOKBACK, _DEFAULT_ZWINDOW,
            float(entry_z), float(exit_z),
        )

    # Status badge below the controls.
    if not _snapshot_df.empty:
        if _snapped_iso == as_of_iso:
            st.caption(
                f":material/lock: **Walk-forward snapshot for {_snapped_iso}** — "
                "pairs re-screened past-only at this date (no hindsight in selection, "
                "no future data in state)."
            )
        else:
            st.caption(
                f":material/lock: **Walk-forward snapshot for {_snapped_iso}** "
                f"(you picked {as_of_iso}; snapped to nearest preceding grid date). "
                "Pairs re-screened past-only — no hindsight in selection."
            )
    elif _has_walkforward and _snapped_iso is None:
        st.caption(
            f"You picked {as_of_iso}, which is before the walk-forward grid starts "
            f"({_grid_dates[0]}). Pick a later date."
        )
    elif _used_fallback:
        st.caption(
            f":material/warning: **Fallback live-compute** for {as_of_iso} — this "
            f"universe doesn't have walk-forward snapshots. Pair list comes from "
            f"`dislocation_candidates.csv` which was screened on the full history "
            f"(2020–latest), so selection has hindsight bias. Switch to BIST/S&P TRY "
            f"basis for the walk-forward path."
        )

    # ── Section 1: Today's actions ─────────────────────────────────────
    _section_title = (
        f"Trades to put on as of {_snapped_iso}"
        if _snapped_iso else f"Trades to put on as of {as_of_iso}"
    )
    section_header(_section_title)

    if _candidates_missing and _snapshot_df.empty:
        st.info("Switch sidebar basis to **TRY** to see ranked pairs.")
    elif leaderboard.empty:
        st.info("No pairs computable at this date.")
    else:
        longs, shorts, trades = _todays_actions(leaderboard, float(entry_z))
        if not trades:
            st.caption(
                f"No pair is at |z| ≥ {entry_z:.1f} on {as_of_iso}. "
                "Nothing to enter today; existing positions stay on until exit."
            )
        else:
            # Net portfolio summary
            _col_long, _col_short = st.columns(2)
            with _col_long:
                if longs:
                    parts = [f"**{t}** ×{n}" if n > 1 else f"**{t}**" for t, n in sorted(longs.items(), key=lambda x: -x[1])]
                    st.markdown(f"🟢 **LONG**: {', '.join(parts)}")
                else:
                    st.markdown("🟢 **LONG**: —")
            with _col_short:
                if shorts:
                    parts = [f"**{t}** ×{n}" if n > 1 else f"**{t}**" for t, n in sorted(shorts.items(), key=lambda x: -x[1])]
                    st.markdown(f"🔴 **SHORT**: {', '.join(parts)}")
                else:
                    st.markdown("🔴 **SHORT**: —")

            # Per-pair trades
            trades_df = pd.DataFrame(trades)
            trades_df["half_life"] = trades_df["half_life"].round(1)
            trades_df = trades_df.rename(
                columns={
                    "pair": "Pair", "z": "Z", "action": "Trade",
                    "half_life": "Half-life (d)",
                }
            )
            st.dataframe(
                trades_df, use_container_width=True, hide_index=True,
                column_config={
                    "Z": st.column_config.NumberColumn("Z", format="%+.2f"),
                    "Half-life (d)": st.column_config.NumberColumn("Half-life (d)", format="%.1f"),
                },
            )
            st.caption(
                f"{len(trades)} pair{'s' if len(trades)!=1 else ''} at "
                f"|z| ≥ {entry_z:.1f}. Hedge ratio (β) not shown — use Pair "
                "Analysis for sizing."
            )

            # ── Drill-in to Pair Analysis at the same as-of date ──────
            # selectbox + button (not st.dataframe on_select) because the
            # latter is fragile in Streamlit 1.41 AppTest. Sets
            # pa_ticker_a/b + pa_as_of_date + pa_as_of_source, then
            # switch_page to Pair Analysis. The receiving page reads
            # those session_state keys on its next render.
            _pair_labels = [t["pair"] for t in trades]
            _drill_col_sel, _drill_col_btn = st.columns([3, 1])
            with _drill_col_sel:
                _drill_pair = st.selectbox(
                    "Open one of these in Pair Analysis",
                    _pair_labels,
                    key="sig_drill_pair",
                    label_visibility="visible",
                )
            with _drill_col_btn:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                if st.button(
                    ":material/open_in_new: Drill in",
                    key="sig_drill_btn",
                    use_container_width=True,
                ):
                    if _drill_pair:
                        _ta, _tb = _drill_pair.split("/")
                        st.session_state["pa_ticker_a"] = _ta
                        st.session_state["pa_ticker_b"] = _tb
                        # Carry the snapped as-of date over.
                        _carry_iso = _snapped_iso if _snapped_iso else as_of_iso
                        st.session_state["pa_as_of_date"] = pd.Timestamp(_carry_iso).date()
                        st.session_state["pa_as_of_source"] = "drill_in_from_signals"
                        st.switch_page("views/04_pair_analysis.py")

    # ── Section 2: Cluster consensus (which tickers show up in many hot pairs) ─
    section_header("Cluster consensus")

    if not leaderboard.empty:
        pressure = _ticker_pressure_view(leaderboard, threshold=1.5)
        if pressure.empty:
            st.caption(
                f"No ticker appears in 2+ pairs at |z| > 1.5 on {as_of_iso}. "
                "Cluster signals are weak by this measure."
            )
        else:
            st.dataframe(
                pressure, use_container_width=True, hide_index=True,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker"),
                    "sector": st.column_config.TextColumn("Sector"),
                    "n_pairs": st.column_config.NumberColumn("Pairs", format="%d"),
                    "consensus_z": st.column_config.NumberColumn(
                        "Consensus Z", format="%+.2f",
                        help="Avg buy-pressure across pairs. Positive → LONG, negative → SHORT.",
                    ),
                    "action": st.column_config.TextColumn("Action"),
                    "paired_with": st.column_config.TextColumn("Paired with"),
                },
            )

    # ── Section 3: All ranked pairs ────────────────────────────────────
    section_header("All ranked pairs")

    if leaderboard.empty and not _candidates_missing:
        st.info("No leaderboard rows at this date.")
    elif not leaderboard.empty:
        status_counts = leaderboard["status"].value_counts()
        active_entries = int(status_counts.get(_STATUS_LONG_ENTRY, 0)) + int(
            status_counts.get(_STATUS_SHORT_ENTRY, 0)
        )
        in_position = int(status_counts.get(_STATUS_IN_LONG, 0)) + int(
            status_counts.get(_STATUS_IN_SHORT, 0)
        )
        approaching = int(status_counts.get(_STATUS_NEAR_ENTRY, 0))
        flat = int(status_counts.get(_STATUS_FLAT, 0))
        _k1, _k2, _k3, _k4 = st.columns(4)
        _k1.metric("Entry candidates", active_entries)
        _k2.metric("In position", in_position)
        _k3.metric("Approaching", approaching)
        _k4.metric("Flat", flat)

        display = leaderboard.copy()
        display.insert(0, "·", display["status"].map(_status_to_emoji))
        display["current_z"] = display["current_z"].round(2)
        display["correlation"] = display["correlation"].round(3)
        display["half_life"] = display["half_life"].round(1)

        # Trade column: positive z → SHORT B, LONG A. Negative z → LONG B, SHORT A.
        # Only populated when |z| >= entry_z (entry candidate) or |status| is
        # in-position; otherwise blank to avoid pretending there's a trade.
        def _trade(row):
            z = row["current_z"]
            ta, tb = row["ticker_a"], row["ticker_b"]
            if pd.isna(z):
                return ""
            status = row["status"]
            if status in (_STATUS_LONG_ENTRY, _STATUS_IN_LONG):
                return f"LONG {tb} / SHORT {ta}"
            if status in (_STATUS_SHORT_ENTRY, _STATUS_IN_SHORT):
                return f"SHORT {tb} / LONG {ta}"
            return ""
        display["trade"] = display.apply(_trade, axis=1)

        priority = {
            _STATUS_LONG_ENTRY: 0,
            _STATUS_SHORT_ENTRY: 0,
            _STATUS_IN_LONG: 1,
            _STATUS_IN_SHORT: 1,
            _STATUS_NEAR_ENTRY: 2,
            _STATUS_FLAT: 3,
            _STATUS_NA: 4,
        }
        display["_p"] = display["status"].map(priority).fillna(5)
        display["_az"] = display["current_z"].abs()
        display = display.sort_values(["_p", "_az"], ascending=[True, False]).drop(
            columns=["_p", "_az"]
        )

        # Reorder so Trade is right after Status.
        cols = ["·", "ticker_a", "ticker_b", "sector_a", "sector_b",
                "current_z", "status", "trade", "days_since_last_signal",
                "half_life", "correlation"]
        display = display[[c for c in cols if c in display.columns]]

        st.dataframe(
            display, use_container_width=True, hide_index=True,
            column_config={
                "·": st.column_config.TextColumn("·", width="small"),
                "ticker_a": st.column_config.TextColumn("Ticker A"),
                "ticker_b": st.column_config.TextColumn("Ticker B"),
                "sector_a": st.column_config.TextColumn("Sector A"),
                "sector_b": st.column_config.TextColumn("Sector B"),
                "current_z": st.column_config.NumberColumn("Current Z", format="%.2f"),
                "status": st.column_config.TextColumn("Status"),
                "trade": st.column_config.TextColumn("Trade"),
                "days_since_last_signal": st.column_config.NumberColumn(
                    "Days since signal", format="%d",
                ),
                "half_life": st.column_config.NumberColumn("Half-life (d)", format="%.1f"),
                "correlation": st.column_config.NumberColumn("Pair ρ", format="%.3f"),
            },
        )

    # ── Section 3: Pair explorer ───────────────────────────────────────
    section_header("Pair explorer")

    ticker_list = sorted(adj_close.columns.tolist())
    if len(ticker_list) < 2:
        st.info("Not enough tickers in this universe to compose a pair.")
        return

    # Default Ticker A / B from leaderboard top row (the most-prioritised
    # by the sort above). If leaderboard is empty fall back to the first
    # two tickers in the panel.
    if not leaderboard.empty:
        # Re-sort by same priority to get the page's recommended top pair.
        priority = {
            _STATUS_LONG_ENTRY: 0,
            _STATUS_SHORT_ENTRY: 0,
            _STATUS_IN_LONG: 1,
            _STATUS_IN_SHORT: 1,
            _STATUS_NEAR_ENTRY: 2,
            _STATUS_FLAT: 3,
            _STATUS_NA: 4,
        }
        _lb = leaderboard.copy()
        _lb["_p"] = _lb["status"].map(priority).fillna(5)
        _lb["_abs_z"] = _lb["current_z"].abs()
        _lb = _lb.sort_values(["_p", "_abs_z"], ascending=[True, False])
        default_a, default_b = _lb.iloc[0]["ticker_a"], _lb.iloc[0]["ticker_b"]
    else:
        default_a, default_b = ticker_list[0], ticker_list[1]

    if (
        "sig_ticker_a" not in st.session_state
        or st.session_state["sig_ticker_a"] not in ticker_list
    ):
        st.session_state["sig_ticker_a"] = default_a if default_a in ticker_list else ticker_list[0]
    if (
        "sig_ticker_b" not in st.session_state
        or st.session_state["sig_ticker_b"] not in ticker_list
    ):
        st.session_state["sig_ticker_b"] = (
            default_b if default_b in ticker_list and default_b != st.session_state["sig_ticker_a"]
            else next((t for t in ticker_list if t != st.session_state["sig_ticker_a"]), ticker_list[0])
        )
    # Pre-render A==B collision guard.
    if st.session_state["sig_ticker_a"] == st.session_state["sig_ticker_b"]:
        st.session_state["sig_ticker_b"] = next(
            (t for t in ticker_list if t != st.session_state["sig_ticker_a"]),
            ticker_list[0],
        )

    _ca, _cb = st.columns(2)
    with _ca:
        ticker_a = st.selectbox("Ticker A", ticker_list, key="sig_ticker_a")
    with _cb:
        # Filter out Ticker A from B's options (PORT arda item 7 convention).
        _b_options = [t for t in ticker_list if t != ticker_a]
        ticker_b = st.selectbox("Ticker B", _b_options, key="sig_ticker_b")

    spread, zscore, beta, half_life = _pair_zscore_history(
        adj_close, cache_key, ticker_a, ticker_b,
        _DEFAULT_LOOKBACK, _DEFAULT_ZWINDOW,
    )
    signals_full = detect_signals(zscore, entry_threshold=float(entry_z), exit_threshold=float(exit_z))

    # Filter signals to those at-or-before the picked as-of date, so the
    # markers on the chart honour the date scrubber semantics.
    as_of_ts = pd.Timestamp(picked_date)
    if not signals_full.empty:
        signals = signals_full[pd.to_datetime(signals_full["date"]) <= as_of_ts].copy()
    else:
        signals = signals_full

    # KPI strip for the pair
    z_now = float(zscore.loc[:as_of_ts].dropna().iloc[-1]) if not zscore.loc[:as_of_ts].dropna().empty else float("nan")
    status, last_signal_date = _state_at(zscore, as_of_ts, float(entry_z), float(exit_z))
    days_since = (
        int((as_of_ts - last_signal_date).days) if last_signal_date is not None else None
    )
    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("Current Z", f"{z_now:+.2f}σ" if np.isfinite(z_now) else "—")
    _m2.metric("Status", status)
    _m3.metric("Half-life", f"{half_life:.1f} d" if np.isfinite(half_life) else "∞")
    _m4.metric(
        "Days since last signal",
        days_since if days_since is not None else "—",
    )

    # Chart
    z_slice = zscore.loc[:as_of_ts] if as_of_ts in zscore.index or not zscore.empty else zscore
    fig = _signal_zscore_chart(
        z_slice, signals,
        entry_z=float(entry_z), exit_z=float(exit_z),
        pair_label=f"{ticker_a} / {ticker_b}",
    )
    render_chart(
        fig,
        chart_id="sig_pair_zscore",
        filename_base=f"signals_{ticker_a}_{ticker_b}_zscore",
        default_title=f"{ticker_a} / {ticker_b} — Z-score & signals",
    )

    # Signal history table (collapsed by default — same pattern Pair Analysis uses)
    with st.expander(f"Signal History ({len(signals)} signals up to {as_of_iso})", expanded=False):
        if signals.empty:
            st.caption("No state-machine signals fired in the available history.")
        else:
            disp = signals.copy()
            disp["zscore_value"] = disp["zscore_value"].round(2)
            disp = disp.rename(
                columns={
                    "date": "Date",
                    "signal": "Signal",
                    "zscore_value": "Z-score",
                }
            )
            st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Section 4: Cross-asset β shift (BIST only) ─────────────────────
    if current_universe() == "bist":
        section_header("Cross-asset β shift")
        summary = load_cross_asset_summary()
        if summary.empty:
            st.info("Cross-asset artifacts missing — re-run the pipeline.")
        else:
            _col_try, _col_gold = st.columns(2)
            with _col_try:
                st.markdown("**vs USD / TRY**")
                rolling_try = load_cross_asset_rolling("usd_try")
                breakouts_try = _cross_asset_breakouts(
                    summary[["ticker", "sector", "corr_usd_try"]],
                    rolling_try, cache_key, as_of_iso, top_n=8,
                )
                if breakouts_try.empty:
                    st.caption("No data.")
                else:
                    disp = breakouts_try.copy()
                    disp["baseline_corr"] = disp["baseline_corr"].round(3)
                    disp["current_corr"] = disp["current_corr"].round(3)
                    disp["deviation"] = disp["deviation"].round(3)
                    disp = disp.drop(columns=["abs_deviation"])
                    st.dataframe(
                        disp, use_container_width=True, hide_index=True,
                        column_config={
                            "ticker": st.column_config.TextColumn("Ticker"),
                            "sector": st.column_config.TextColumn("Sector"),
                            "baseline_corr": st.column_config.NumberColumn(
                                "Baseline", format="%+.3f",
                            ),
                            "current_corr": st.column_config.NumberColumn(
                                "Current", format="%+.3f",
                            ),
                            "deviation": st.column_config.NumberColumn(
                                "Δ", format="%+.3f",
                            ),
                        },
                    )

            with _col_gold:
                st.markdown("**vs Gold (USD / oz)**")
                rolling_gold = load_cross_asset_rolling("gold_usd")
                breakouts_gold = _cross_asset_breakouts(
                    summary[["ticker", "sector", "corr_gold_usd"]],
                    rolling_gold, cache_key, as_of_iso, top_n=8,
                )
                if breakouts_gold.empty:
                    st.caption("No data.")
                else:
                    disp = breakouts_gold.copy()
                    disp["baseline_corr"] = disp["baseline_corr"].round(3)
                    disp["current_corr"] = disp["current_corr"].round(3)
                    disp["deviation"] = disp["deviation"].round(3)
                    disp = disp.drop(columns=["abs_deviation"])
                    st.dataframe(
                        disp, use_container_width=True, hide_index=True,
                        column_config={
                            "ticker": st.column_config.TextColumn("Ticker"),
                            "sector": st.column_config.TextColumn("Sector"),
                            "baseline_corr": st.column_config.NumberColumn(
                                "Baseline", format="%+.3f",
                            ),
                            "current_corr": st.column_config.NumberColumn(
                                "Current", format="%+.3f",
                            ),
                            "deviation": st.column_config.NumberColumn(
                                "Δ", format="%+.3f",
                            ),
                        },
                    )
