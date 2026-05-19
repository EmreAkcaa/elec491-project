"""Pair Analysis view — rendered inline from dashboard.py via render().

Full deep-dive into two equities of the active universe (BIST or S&P): correlation
dynamics, price comparison, risk metrics, return distributions, volatility,
drawdown, and MST position. Data is loaded via the universe-aware loaders in
``app/utils.py``, so this page works for any universe registered in
``app/universe_registry.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from utils import (
    current_universe,
    draw_event_markers,
    event_marker_manager_ui,
    check_ticker_pair_warnings,
    render_warnings,
    load_summary_stats,
    load_mst_edges,
    load_mst_metrics,
    load_cluster_assignments,
    load_base_asset,
    get_colors,
    SECTOR_PALETTE,
    CHART_LAYOUT,
    apply_chart_style,
    inject_custom_css,
    section_header,
    render_chart,
)
from universe_registry import get_universe
from src.rolling_correlation import compute_rolling_pair_correlation
from src.pair_dislocation import (
    compute_spread,
    compute_zscore,
    compute_half_life,
    detect_signals,
)


# ══════════════════════════════════════════════════════════════════════════════
# Cached computation helpers (module-level to avoid Streamlit re-registration)
# ══════════════════════════════════════════════════════════════════════════════

# ── Phase 4: Compare-against (FX/Gold) constants + helpers ───────────────
# When the user picks a non-stock comparison target, the second leg of the
# pair is a single price series loaded from `data/raw/base_assets/`.
# We inject it as a synthetic column into local copies of `full_returns`
# (log returns) and `adj_close` (close prices), so the downstream math
# (spread, β, Z-score, dislocation, rolling correlation, volatility,
# drawdown) treats it identically to a stock leg with NO branching.

_COMPARE_LABEL_TO_ASSET: dict[str, str | None] = {
    "Another stock":      None,
    "USD / TRY":          "usd_try",
    "Gold (USD / oz)":    "gold_usd",
}
_ASSET_DISPLAY_LABEL: dict[str, str] = {
    "usd_try":  "USD/TRY",
    "gold_usd": "Gold (USD/oz)",
}


def _inject_base_asset(
    full_returns: pd.DataFrame,
    adj_close: pd.DataFrame,
    asset_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Augment ``full_returns`` + ``adj_close`` with a base-asset synthetic
    column so the rest of pair_analysis can treat it as just another ticker.

    Reindexes the base asset to the universe's date axis with forward-fill
    for small calendar mismatches (BIST trading days differ slightly from
    FX/commodity trading days). The added column shares the same Date
    index as `full_returns`.

    Returns ``(returns_aug, adj_close_aug, ok)``. ``ok`` is False if the
    base-asset parquet was missing or empty — caller should fall back.
    """
    series = load_base_asset(asset_key)
    if series.empty:
        return full_returns, adj_close, False

    aligned = series.reindex(full_returns.index).ffill()
    log_returns = np.log(aligned / aligned.shift(1))
    aligned.name = asset_key
    log_returns.name = asset_key
    new_returns = full_returns.assign(**{asset_key: log_returns})
    new_adj_close = adj_close.assign(**{asset_key: aligned})
    return new_returns, new_adj_close, True


@st.cache_data(show_spinner=False)
def _pair_corr(_ret: pd.DataFrame, cache_key: str, a, b, win, method, wtype):
    # `_ret` underscore-prefix → Streamlit skips hashing the DataFrame; cache
    # is keyed by `cache_key` (universe + date endpoints + shape).
    return compute_rolling_pair_correlation(_ret, a, b, window=win, method=method, window_type=wtype)


@st.cache_data(show_spinner=False)
def _rolling_vol(_r: pd.DataFrame, cache_key: str, ta, tb, win):
    va = _r[ta].rolling(win).std() * np.sqrt(252)
    vb = _r[tb].rolling(win).std() * np.sqrt(252)
    return va, vb


@st.cache_data(show_spinner=False)
def _compute_dislocation(_adj: pd.DataFrame, cache_key: str, ta, tb, lookback, zwin, entry_th, exit_th):
    spread, beta, intercept = compute_spread(_adj, ta, tb, lookback=lookback)
    zscore = compute_zscore(spread, window=zwin)
    half_life = compute_half_life(spread)
    signals = detect_signals(zscore, entry_threshold=entry_th, exit_threshold=exit_th)
    return spread, beta, intercept, zscore, half_life, signals


@st.cache_data
def _subgraph_layout(nodes, edges):
    _H = nx.Graph()
    _H.add_nodes_from(nodes)
    _H.add_edges_from(edges)
    return nx.spring_layout(_H, seed=42, k=3.0, iterations=80)


@st.fragment
def render(
    adj_close: pd.DataFrame,
    full_returns: pd.DataFrame,
    coverage_df: pd.DataFrame,
    min_date,
    max_date,
) -> None:
    """Render the full Pair Analysis view with inline controls.

    @st.fragment scope: changing ticker_a/ticker_b/date_range/dislocation
    params re-runs ONLY this function — not the dashboard.py script
    prologue (universe init, page_config, sidebar, top-nav, etc.). On S&P
    that prologue is ~200-500 ms, so the perceived snappiness on every
    pair switch and slider drag improves measurably.

    Pair Analysis has no `st.rerun()` calls; all internal navigation is
    cross-page via session_state flags (`_goto_pair_analysis` etc.),
    which the outer dashboard.py reads on its NEXT full rerun (triggered
    by users clicking a top-nav button, not by anything inside this
    fragment). So `scope="app"` isn't needed.

    Defence-in-depth: the dashboard nav already hides this page when the
    active universe has ``has_pair_trading=False``, but a direct deep-link
    via ?nav_page=Pair+Analysis could bypass that. We re-check the capability
    flag here and bail out with a friendly notice instead of rendering a
    nonsensical pair-trading view on non-financial data.
    """

    # CLAUDE.md: importlib.reload(universe_registry) was removed from
    # dashboard.py (PR #23) because it churns Universe class identity across
    # reruns, causing "Tried to use SessionInfo before it was initialized"
    # warnings AND invalidating downstream @st.cache_data entries keyed by
    # Universe instances. pair_analysis.py was missed in that pass — fixed
    # here. HF Spaces rebuilds the container on every deploy, so the stale-
    # module cache problem the reload was guarding against no longer applies.
    _active = get_universe(current_universe())
    if not getattr(_active, "has_pair_trading", True):
        st.warning(
            f"Pair Analysis isn't applicable to the **{getattr(_active, 'label', 'current')}** universe "
            "(no pair-trading semantics). Switch back to a financial universe "
            "from the sidebar selector."
        )
        return

    inject_custom_css()

    # ══════════════════════════════════════════════════════════════════════
    # Inline Controls (replaces sidebar)
    # ══════════════════════════════════════════════════════════════════════

    ticker_list = sorted(full_returns.columns.tolist())

    # Initialise widget state ONCE; passing index=/value= alongside key= when
    # the key is in session_state triggers a Streamlit warning banner that
    # pops up every time the user changes a selection.
    if (
        "pa_ticker_a" not in st.session_state
        or st.session_state["pa_ticker_a"] not in ticker_list
    ):
        st.session_state["pa_ticker_a"] = ticker_list[0]
    if (
        "pa_ticker_b" not in st.session_state
        or st.session_state["pa_ticker_b"] not in ticker_list
    ):
        st.session_state["pa_ticker_b"] = (
            ticker_list[1] if len(ticker_list) > 1 else ticker_list[0]
        )

    # Auto-resolve A==B collision BEFORE widgets are instantiated.
    # Writing to a widget key after the widget renders raises
    # StreamlitAPIException; doing it here is safe and prevents the page
    # from collapsing to a single warning.
    if (
        st.session_state["pa_ticker_a"] == st.session_state["pa_ticker_b"]
        and len(ticker_list) > 1
    ):
        st.session_state["pa_ticker_b"] = next(
            (t for t in ticker_list if t != st.session_state["pa_ticker_a"]),
            ticker_list[0],
        )

    if "pa_date_range" not in st.session_state:
        st.session_state["pa_date_range"] = (min_date, max_date)

    # Row 1: Ticker A + Compare-against + Ticker B + date range
    # Phase 4: "Compare against" picker lets the second leg be a base
    # asset (USD/TRY or Gold) instead of another stock. When non-stock
    # is chosen the asset series is loaded from
    # data/raw/base_assets/{asset_key}.parquet, log-returned, and injected
    # as a synthetic column into local `full_returns` + `adj_close`
    # copies. All downstream math (spread, β, Z-score, dislocation,
    # rolling corr, vol, drawdown) treats it identically to a stock leg.
    _c_a, _c_compare, _c_b, _c_date = st.columns([2, 2, 2, 3])
    with _c_a:
        ticker_a = st.selectbox("Ticker A", ticker_list, key="pa_ticker_a")
    with _c_compare:
        _compare_label = st.selectbox(
            "Compare against",
            list(_COMPARE_LABEL_TO_ASSET.keys()),
            index=0,
            key="pa_compare_target",
            help=(
                "**Another stock** — classic pair trade.\n\n"
                "**USD/TRY** — FX sensitivity. Exporter stocks (banks excl.) "
                "tend to be positively correlated with TRY weakening; "
                "domestic plays negatively.\n\n"
                "**Gold (USD/oz)** — inflation/risk-off sensitivity. "
                "Gold-correlated stocks behave as inflation hedges.\n\n"
                "All downstream math (spread, β, Z-score, dislocation) "
                "works identically — the base asset is injected as a "
                "synthetic second-leg ticker."
            ),
        )
    _compare_asset_key = _COMPARE_LABEL_TO_ASSET[_compare_label]

    # _b_disp is the LABEL shown in headers/captions. ticker_b is the
    # COLUMN KEY used for DataFrame lookups. They diverge only when
    # ticker_b is a base-asset key (where "usd_try" is the column but
    # "USD/TRY" is the display).
    # PORT arda/ui-cleanup item 7: Ticker B selectbox excludes Ticker A
    # from its options. The session-state swap above (lines 208-215) ensures
    # the stored pa_ticker_b value is never equal to pa_ticker_a, so the
    # filtered list never strips the currently-stored value.
    _ticker_b_options = [t for t in ticker_list if t != ticker_a]

    if _compare_asset_key is None:
        # Stock-vs-stock — existing behavior, existing Ticker B selectbox.
        with _c_b:
            ticker_b = st.selectbox("Ticker B", _ticker_b_options, key="pa_ticker_b")
        _b_disp = ticker_b
        _b_is_base_asset = False
    else:
        # Stock-vs-base-asset. Inject the asset as a synthetic column
        # into our LOCAL copies of full_returns + adj_close. The original
        # DataFrames passed into render() are unchanged (assign() returns
        # a new frame).
        full_returns, adj_close, _ok = _inject_base_asset(
            full_returns, adj_close, _compare_asset_key,
        )
        if not _ok:
            with _c_b:
                ticker_b = st.selectbox("Ticker B", _ticker_b_options, key="pa_ticker_b")
            _b_disp = ticker_b
            _b_is_base_asset = False
            st.warning(
                f"Base-asset series for **{_compare_label}** is missing on disk "
                f"(expected `data/raw/base_assets/{_compare_asset_key}.parquet`). "
                "Falling back to stock-vs-stock."
            )
        else:
            ticker_b = _compare_asset_key
            _b_disp = _ASSET_DISPLAY_LABEL[_compare_asset_key]
            _b_is_base_asset = True
            with _c_b:
                # Display-only — Ticker B is fixed to the chosen base asset.
                # text_input keeps the column-row alignment without rendering
                # a confusing always-disabled selectbox of irrelevant stocks.
                st.text_input(
                    "Ticker B",
                    value=_b_disp,
                    disabled=True,
                    key="pa_ticker_b_baseAsset_display",
                    help="Auto-set from 'Compare against' choice.",
                )
    with _c_date:
        date_range = st.date_input(
            "Date range",
            min_value=min_date, max_value=max_date,
            key="pa_date_range",
        )

    if len(date_range) == 2:
        start_dt = pd.Timestamp(date_range[0])
        end_dt   = pd.Timestamp(date_range[1])
    else:
        start_dt, end_dt = pd.Timestamp(min_date), pd.Timestamp(max_date)

    returns       = full_returns.loc[start_dt:end_dt]
    prices_window = adj_close.loc[start_dt:end_dt]


    # ══════════════════════════════════════════════════════════════════════
    # Page Header
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        f"<h1 style='margin-bottom:0;'>Pair Analysis</h1>"
        f"<p style='color:#8D99AE; margin-top:0; font-size:0.95rem;'>"
        f"Deep-dive comparison of <b>{ticker_a}</b> and <b>{_b_disp}</b></p>",
        unsafe_allow_html=True,
    )

    if ticker_a == ticker_b:
        # The pre-render collision-resolver above auto-fixes this in
        # normal flows. Only reachable in the pathological "only one
        # surviving ticker" case. Show a friendlier message but keep the
        # early return because downstream code is not designed to handle
        # ticker_a == ticker_b.
        st.warning(
            "Only one ticker is available in the current universe. "
            "Pair Analysis needs two distinct tickers — check the "
            "coverage filter in `config/settings.yaml` or extend the "
            "date range."
        )
        return

    # ── Data-quality warnings ────────────────────────────────────────────
    issues = check_ticker_pair_warnings(full_returns, ticker_a, ticker_b, returns, coverage_df)
    if issues:
        render_warnings(issues)

    if ticker_a not in returns.columns or ticker_b not in returns.columns:
        st.error("One or both selected tickers are missing from the processed dataset.")
        return

    both = returns[[ticker_a, ticker_b]].dropna()
    show_defaults, custom_events = event_marker_manager_ui("pa", min_date, max_date)

    # Build cheap deterministic cache keys for the @st.cache_data helpers.
    # The DataFrames themselves are passed underscore-prefixed (skipped by
    # Streamlit's hasher); cache identity comes from these strings. Universe
    # + date endpoints + shape uniquely identifies each slice in this app.
    _u = current_universe()
    returns_cache_key = (
        f"{_u}:{returns.index[0].date().isoformat()}:"
        f"{returns.index[-1].date().isoformat()}:{returns.shape[0]}x{returns.shape[1]}"
    )
    adj_cache_key = (
        f"{_u}:{adj_close.index[0].date().isoformat()}:"
        f"{adj_close.index[-1].date().isoformat()}:{adj_close.shape[0]}x{adj_close.shape[1]}"
    )

    # ══════════════════════════════════════════════════════════════════════
    # Sub-Tab Layout
    # ══════════════════════════════════════════════════════════════════════

    # PHASE Y (Y1): render_subtabs replaces st.tabs so only the active
    # sub-tab body executes. Previously all 5 tabs computed on every load
    # even though only one was visible — overview's KPI compute + correlation's
    # rolling rho + risk's vol + dislocation's spread + network's MST layout
    # all ran in series. Now they're behind `if _active == "X":` gates.
    from utils import render_subtabs
    _pa_subtabs = (
        "Overview", "Correlation", "Risk & Volatility",
        "Spread & Dislocation", "Network",
    )
    _active_pa_sub = render_subtabs("pair_analysis", _pa_subtabs)
    _show_pa_ov     = _active_pa_sub == "Overview"
    _show_pa_corr   = _active_pa_sub == "Correlation"
    _show_pa_risk   = _active_pa_sub == "Risk & Volatility"
    _show_pa_disloc = _active_pa_sub == "Spread & Dislocation"
    _show_pa_net    = _active_pa_sub == "Network"

    # ══════════════════════════════════════════════════════════════════════
    # Tab 1 — Overview
    # ══════════════════════════════════════════════════════════════════════

    if _show_pa_ov:

        # Key metrics
        with st.container(border=True):
            if len(both) >= 2:
                rho_p = both.corr(method="pearson").iloc[0, 1]
                rho_s = both.corr(method="spearman").iloc[0, 1]
                _m1, _m2, _m3, _m4 = st.columns(4)
                _m1.metric("Pearson", f"{rho_p:.4f}")
                _m2.metric("Spearman", f"{rho_s:.4f}")
                _m3.metric("Trading Days", f"{len(both):,}")
                d = np.sqrt(2 * (1 - rho_p))
                _m4.metric("Distance d(i,j)", f"{d:.4f}")

            try:
                _summary = load_summary_stats()
                _cols = ["ticker", "annualized_return", "annualized_vol",
                         "skewness", "kurtosis", "min_return", "max_return"]
                _sub = _summary[_summary["ticker"].isin([ticker_a, ticker_b])][_cols].copy()
                # Rename to human-readable headers so the table doesn't
                # render raw underscored names like "annualized_return".
                _sub = _sub.rename(columns={
                    "ticker":            "Ticker",
                    "annualized_return": "Annualized return",
                    "annualized_vol":    "Annualized volatility",
                    "skewness":          "Skewness",
                    "kurtosis":          "Excess kurtosis",
                    "min_return":        "Worst daily log return",
                    "max_return":        "Best daily log return",
                })
                for _c in ["Annualized return", "Annualized volatility",
                           "Worst daily log return", "Best daily log return"]:
                    _sub[_c] = _sub[_c].map(lambda v: f"{v:.4f}")
                for _c in ["Skewness", "Excess kurtosis"]:
                    _sub[_c] = _sub[_c].map(lambda v: f"{v:.2f}")
                st.dataframe(_sub.set_index("Ticker"), use_container_width=True)
            except Exception:
                pass

        # Price Performance — full-width.
        # PORT arda/ui-cleanup item 6: removed the right-column return-scatter
        # chart (`pa_scatter`). Section title shortened from "Price Performance
        # & Return Relationship" to "Price Performance". Price chart now
        # full-width at height 480 (was 440 in a 50/50 split).
        with st.container(border=True):
            section_header("Price Performance")

            if ticker_a in prices_window.columns and ticker_b in prices_window.columns:
                pa = prices_window[ticker_a] / prices_window[ticker_a].iloc[0] * 100
                pb = prices_window[ticker_b] / prices_window[ticker_b].iloc[0] * 100

                fig_price = go.Figure()
                _al = pd.concat([pa, pb], axis=1).dropna()
                if not _al.empty:
                    _pa_al, _pb_al = _al.iloc[:, 0], _al.iloc[:, 1]
                    fig_price.add_trace(go.Scatter(
                        x=list(_pa_al.index) + list(_pb_al.index[::-1]),
                        y=list(_pa_al.values) + list(_pb_al.values[::-1]),
                        fill="toself", fillcolor=get_colors()["positive"],
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                    ))
                fig_price.add_trace(go.Scatter(
                    x=pa.index, y=pa, name=ticker_a,
                    mode="lines", line=dict(color=get_colors()["primary"], width=2.2),
                ))
                fig_price.add_trace(go.Scatter(
                    x=pb.index, y=pb, name=_b_disp,
                    mode="lines", line=dict(color=get_colors()["secondary"], width=2.2),
                ))
                draw_event_markers(fig_price, show_defaults, custom_events,
                                   pa.index.min(), pa.index.max())
                apply_chart_style(fig_price, height=480,
                                  yaxis_title="Normalized Price (base = 100)")
                render_chart(fig_price, chart_id="pa_prices", filename_base="pair_prices",
                             title_key="pa_prices", default_title="Price Performance")

    # ══════════════════════════════════════════════════════════════════════
    # Tab 2 — Correlation
    # ══════════════════════════════════════════════════════════════════════

    if _show_pa_corr:

        # Rolling Correlation
        with st.container(border=True):
            section_header("Rolling Correlation")

            _rc1, _rc2, _rc3, _rc4 = st.columns(4)
            with _rc1:
                rc_window = st.selectbox(
                    "Window (days)", [30, 60, 120, 252, 504], index=2, key="pa_rc_win",
                )
            with _rc2:
                rc_step = st.selectbox(
                    "Step", [1, 5, 21], index=1, key="pa_rc_step",
                    format_func=lambda v: {1: "1 (daily)", 5: "5 (weekly)", 21: "21 (monthly)"}.get(v, str(v)),
                )
            with _rc3:
                rc_method = st.selectbox("Method", ["pearson", "spearman"], key="pa_rc_method")
            with _rc4:
                rc_window_type = st.selectbox("Window type", ["rolling", "expanding", "ewm"], key="pa_rc_wtype")

            try:
                with st.status("Computing rolling correlation...", expanded=False) as _status:
                    pair_corr = _pair_corr(
                        returns, returns_cache_key,
                        ticker_a, ticker_b, rc_window, rc_method, rc_window_type,
                    )
                    _status.update(label="Rolling correlation ready", state="complete")

                pair_valid = pair_corr.dropna()
                if not pair_valid.empty:
                    _s1, _s2, _s3, _s4, _s5 = st.columns(5)
                    _s1.metric("Mean", f"{pair_valid.mean():.4f}")
                    _s2.metric("Median", f"{pair_valid.median():.4f}")
                    _s3.metric("Max", f"{pair_valid.max():.4f}")
                    _s4.metric("Min", f"{pair_valid.min():.4f}")
                    _s5.metric("% Positive", f"{(pair_valid > 0).mean():.1%}")

                    fig_rc = go.Figure()
                    fig_rc.add_hline(y=0, line_dash="dot", line_color=get_colors()["muted"], opacity=0.4)
                    fig_rc.add_trace(go.Scatter(
                        x=pair_corr.index, y=pair_corr.clip(lower=0),
                        mode="lines", line=dict(width=0), showlegend=False,
                        fill="tozeroy", fillcolor=get_colors()["positive"],
                    ))
                    fig_rc.add_trace(go.Scatter(
                        x=pair_corr.index, y=pair_corr.clip(upper=0),
                        mode="lines", line=dict(width=0), showlegend=False,
                        fill="tozeroy", fillcolor=get_colors()["negative"],
                    ))
                    fig_rc.add_trace(go.Scatter(
                        x=pair_corr.index, y=pair_corr.values,
                        mode="lines", name=f"{ticker_a} / {_b_disp}",
                        line=dict(color=get_colors()["primary"], width=1.8),
                    ))
                    draw_event_markers(fig_rc, show_defaults, custom_events,
                                       pair_valid.index.min(), pair_valid.index.max())
                    apply_chart_style(fig_rc, height=420,
                                      yaxis=dict(range=[-1.05, 1.05],
                                                 title=f"{rc_method.title()} Correlation",
                                                 gridcolor="rgba(141,153,174,0.15)"),
                                      xaxis_title="Date", showlegend=False)
                    render_chart(fig_rc, chart_id="pa_rolling_corr", filename_base="pair_rolling_corr",
                                 title_key="pa_rolling_corr", default_title="Rolling Correlation")
                else:
                    st.warning("Not enough overlapping data for the selected window size.")

            except Exception as exc:
                st.error(f"Could not compute rolling correlation: {exc}")


    # ══════════════════════════════════════════════════════════════════════
    # Tab 3 — Risk & Volatility
    # ══════════════════════════════════════════════════════════════════════

    if _show_pa_risk:

        # Distributions & Rolling Volatility
        with st.container(border=True):
            section_header(
                "Return Distributions & Rolling Volatility",
                "Left: overlaid return histograms. Right: annualized rolling volatility — "
                "simultaneous spikes indicate shared systematic risk.",
            )

            _vol_sel, _vol_spacer = st.columns([1, 5])
            with _vol_sel:
                vol_window = st.selectbox(
                    "Volatility window (days)", [30, 60, 120, 252, 504],
                    index=2, key="pa_vol_win",
                )

            col_dist, col_vol = st.columns(2)

            with col_dist:
                if not both.empty:
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=both[ticker_a], name=ticker_a, nbinsx=70,
                        marker_color=get_colors()["primary"], opacity=0.60,
                    ))
                    fig_dist.add_trace(go.Histogram(
                        x=both[ticker_b], name=_b_disp, nbinsx=70,
                        marker_color=get_colors()["secondary"], opacity=0.60,
                    ))
                    apply_chart_style(fig_dist, height=400,
                                      barmode="overlay",
                                      xaxis_title="Daily Log Return", yaxis_title="Count")
                    render_chart(fig_dist, chart_id="pa_distributions", filename_base="return_distributions",
                                 title_key="pa_distributions", default_title="Return Distributions")

            with col_vol:
                if not both.empty:
                    # `both` is returns[[a, b]].dropna(); cache key derives from
                    # the underlying returns key + the two tickers + post-dropna
                    # row count (handles ticker-specific missingness).
                    both_cache_key = f"{returns_cache_key}:{ticker_a}:{ticker_b}:{len(both)}"
                    _vol_a, _vol_b = _rolling_vol(
                        both, both_cache_key,
                        ticker_a, ticker_b, vol_window,
                    )

                    fig_vol = go.Figure()
                    _va_al = pd.concat([_vol_a, _vol_b], axis=1).dropna()
                    if not _va_al.empty:
                        _vaa, _vbb = _va_al.iloc[:, 0], _va_al.iloc[:, 1]
                        fig_vol.add_trace(go.Scatter(
                            x=list(_vaa.index) + list(_vbb.index[::-1]),
                            y=list(_vaa.values) + list(_vbb.values[::-1]),
                            fill="toself", fillcolor=get_colors()["band"],
                            line=dict(width=0), showlegend=False, hoverinfo="skip",
                        ))
                    fig_vol.add_trace(go.Scatter(
                        x=_vol_a.index, y=_vol_a, name=ticker_a,
                        mode="lines", line=dict(color=get_colors()["primary"], width=1.8),
                    ))
                    fig_vol.add_trace(go.Scatter(
                        x=_vol_b.index, y=_vol_b, name=_b_disp,
                        mode="lines", line=dict(color=get_colors()["secondary"], width=1.8),
                    ))
                    draw_event_markers(fig_vol, show_defaults, custom_events,
                                       both.index.min(), both.index.max())
                    apply_chart_style(fig_vol, height=400,
                                      yaxis_title=f"Annualized Volatility ({vol_window}d)")
                    render_chart(fig_vol, chart_id="pa_volatility", filename_base="rolling_volatility",
                                 title_key="pa_volatility", default_title="Rolling Volatility")

        # Drawdown
        with st.container(border=True):
            section_header(
                "Drawdown Analysis",
                "Percentage decline from running peak. Overlapping filled areas "
                "highlight joint drawdown periods.",
            )

            if ticker_a in prices_window.columns and ticker_b in prices_window.columns:
                def _dd(prices: pd.Series) -> pd.Series:
                    return (prices - prices.cummax()) / prices.cummax() * 100

                dd_a = _dd(prices_window[ticker_a])
                dd_b = _dd(prices_window[ticker_b])

                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=dd_a.index, y=dd_a, name=ticker_a,
                    mode="lines", fill="tozeroy",
                    line=dict(color=get_colors()["primary"], width=1.5),
                    fillcolor="rgba(67,97,238,0.15)",
                ))
                fig_dd.add_trace(go.Scatter(
                    x=dd_b.index, y=dd_b, name=_b_disp,
                    mode="lines", fill="tozeroy",
                    line=dict(color=get_colors()["secondary"], width=1.5),
                    fillcolor="rgba(230,57,70,0.15)",
                ))
                draw_event_markers(fig_dd, show_defaults, custom_events,
                                   dd_a.index.min(), dd_a.index.max())
                apply_chart_style(fig_dd, height=400, yaxis_title="Drawdown (%)")
                render_chart(fig_dd, chart_id="pa_drawdown", filename_base="drawdown",
                             title_key="pa_drawdown", default_title="Drawdown from Peak")

    # ══════════════════════════════════════════════════════════════════════
    # Tab 4 — Spread & Dislocation
    # ══════════════════════════════════════════════════════════════════════

    if _show_pa_disloc:

        # Inline dislocation settings
        _dc1, _dc2, _dc3, _dc4 = st.columns(4)
        with _dc1:
            zscore_window = st.selectbox(
                "Z-Score Window (days)", [30, 60, 120], index=1, key="pa_zscore_win",
            )
        with _dc2:
            ols_lookback = st.selectbox(
                "OLS Lookback (days)", [120, 252, 504], index=1, key="pa_ols_lookback",
                format_func=lambda v: f"{v} days",
            )
        with _dc3:
            entry_z = st.slider(
                "Entry |Z| Threshold", 1.0, 3.0, 2.0, 0.25, key="pa_entry_z",
            )
        with _dc4:
            exit_z = st.slider(
                "Exit |Z| Threshold", 0.0, 1.5, 0.5, 0.25, key="pa_exit_z",
            )

        # Shared computation for spread & Z-score.
        # PORT arda/ui-cleanup item 8: removed the `st.status("Computing
        # spread & Z-score...")` wrapper. `_compute_dislocation` is
        # @st.cache_data so 95%+ of calls hit cache; the status block was
        # flashing for no reason. First-compute (cold cache) still gets
        # the shimmer overlay from app/utils.py inject_custom_css.
        try:
            spread, beta, intercept, zscore, half_life, signals_df = _compute_dislocation(
                adj_close, adj_cache_key,
                ticker_a, ticker_b, ols_lookback, zscore_window, entry_z, exit_z,
            )

            spread_w = spread.loc[start_dt:end_dt]
            zscore_w = zscore.loc[start_dt:end_dt]

            # ── Log-Price Spread ────────────────────────────────────────
            with st.container(border=True):
                section_header(
                    "Log-Price Spread",
                    f"Hedge-ratio-adjusted spread: log({_b_disp}) - beta x log({ticker_a}). "
                    "A stationary spread supports mean-reversion analysis.",
                )

                if not spread_w.dropna().empty:
                    _sp1, _sp2, _sp3, _sp4 = st.columns(4)
                    _sp1.metric("Beta (hedge ratio)", f"{beta:.4f}")
                    _sp2.metric("Half-Life (days)", f"{half_life:.1f}" if half_life != float("inf") else "inf")
                    _sp3.metric("Current Spread", f"{spread_w.dropna().iloc[-1]:.6f}")
                    _sp4.metric("Spread Std", f"{spread_w.std():.6f}")

                    roll_mean = spread_w.rolling(zscore_window, min_periods=max(10, zscore_window // 2)).mean()
                    roll_std = spread_w.rolling(zscore_window, min_periods=max(10, zscore_window // 2)).std()
                    upper_band = roll_mean + roll_std
                    lower_band = roll_mean - roll_std

                    fig_spread = go.Figure()
                    fig_spread.add_trace(go.Scatter(
                        x=upper_band.index, y=upper_band, mode="lines",
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                    ))
                    fig_spread.add_trace(go.Scatter(
                        x=lower_band.index, y=lower_band, mode="lines",
                        line=dict(width=0), fill="tonexty", fillcolor=get_colors()["band"],
                        showlegend=False, hoverinfo="skip",
                    ))
                    fig_spread.add_trace(go.Scatter(
                        x=roll_mean.index, y=roll_mean, mode="lines",
                        name="Rolling Mean", line=dict(color=get_colors()["muted"], width=1.2, dash="dash"),
                    ))
                    fig_spread.add_trace(go.Scatter(
                        x=spread_w.index, y=spread_w, mode="lines",
                        name="Spread", line=dict(color=get_colors()["primary"], width=1.8),
                    ))
                    draw_event_markers(fig_spread, show_defaults, custom_events,
                                       spread_w.index.min(), spread_w.index.max())
                    apply_chart_style(fig_spread, height=420,
                                      yaxis_title="Spread",
                                      xaxis_title="Date")
                    render_chart(fig_spread, chart_id="pa_spread", filename_base="log_spread",
                                 title_key="pa_spread", default_title="Log-Price Spread")
                else:
                    st.warning("Not enough data to compute spread for the selected window.")

            # ── Z-Score & Dislocation Signals ───────────────────────────
            with st.container(border=True):
                section_header("Z-Score & Dislocation Signals")

                if not zscore_w.dropna().empty:
                    _zv = zscore_w.dropna()
                    _curr_z = _zv.iloc[-1]
                    _window_signals = signals_df[
                        (signals_df["date"] >= start_dt) & (signals_df["date"] <= end_dt)
                    ] if not signals_df.empty else pd.DataFrame()
                    _n_long = len(_window_signals[_window_signals["signal"] == "long_entry"]) if not _window_signals.empty else 0
                    _n_short = len(_window_signals[_window_signals["signal"] == "short_entry"]) if not _window_signals.empty else 0
                    _n_exits = len(_window_signals[_window_signals["signal"].str.contains("exit")]) if not _window_signals.empty else 0

                    _z1, _z2, _z3, _z4 = st.columns(4)
                    _z1.metric("Current Z-Score", f"{_curr_z:.4f}")
                    _z2.metric("Long Entries", _n_long)
                    _z3.metric("Short Entries", _n_short)
                    _z4.metric("Exits", _n_exits)

                    fig_z = go.Figure()

                    fig_z.add_hrect(y0=entry_z, y1=max(float(_zv.max()) + 0.5, entry_z + 1),
                                    fillcolor="rgba(230,57,70,0.08)", line_width=0)
                    fig_z.add_hrect(y0=min(float(_zv.min()) - 0.5, -entry_z - 1), y1=-entry_z,
                                    fillcolor="rgba(230,57,70,0.08)", line_width=0)

                    fig_z.add_hline(y=entry_z, line_dash="dash", line_color=get_colors()["secondary"],
                                    annotation_text=f"+{entry_z}", annotation_position="top right")
                    fig_z.add_hline(y=-entry_z, line_dash="dash", line_color=get_colors()["secondary"],
                                    annotation_text=f"-{entry_z}", annotation_position="bottom right")
                    fig_z.add_hline(y=exit_z, line_dash="dot", line_color=get_colors()["tertiary"], opacity=0.6)
                    fig_z.add_hline(y=-exit_z, line_dash="dot", line_color=get_colors()["tertiary"], opacity=0.6)
                    fig_z.add_hline(y=0, line_dash="dot", line_color=get_colors()["muted"], opacity=0.3)

                    fig_z.add_trace(go.Scatter(
                        x=zscore_w.index, y=zscore_w, mode="lines",
                        name="Z-Score", line=dict(color=get_colors()["primary"], width=1.8),
                    ))

                    if not _window_signals.empty:
                        for _, sig in _window_signals.iterrows():
                            if sig["signal"] == "long_entry":
                                sym, color, name = "triangle-up", "#2EC4B6", "Long Entry"
                            elif sig["signal"] == "short_entry":
                                sym, color, name = "triangle-down", get_colors()["secondary"], "Short Entry"
                            else:
                                sym, color, name = "circle", get_colors()["muted"], "Exit"
                            fig_z.add_trace(go.Scatter(
                                x=[sig["date"]], y=[sig["zscore_value"]],
                                mode="markers", marker=dict(symbol=sym, size=10, color=color, line=dict(width=1, color="white")),
                                name=name, showlegend=False,
                                hovertemplate=f"<b>{name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Z: %{{y:.3f}}<extra></extra>",
                            ))

                    draw_event_markers(fig_z, show_defaults, custom_events,
                                       _zv.index.min(), _zv.index.max())
                    apply_chart_style(fig_z, height=420,
                                      yaxis_title="Z-Score", xaxis_title="Date",
                                      showlegend=False)
                    render_chart(fig_z, chart_id="pa_zscore", filename_base="zscore",
                                 title_key="pa_zscore", default_title="Z-Score")

                    # PORT arda/ui-cleanup item 8: Signal History wrapped in
                    # a collapsed-by-default expander. Same table content;
                    # only the chrome around it changes.
                    if not _window_signals.empty:
                        with st.expander(
                            f"Signal History ({len(_window_signals)} signals)",
                            expanded=False,
                        ):
                            _disp = _window_signals.copy()
                            _disp["date"] = pd.to_datetime(_disp["date"]).dt.strftime("%Y-%m-%d")
                            _disp["zscore_value"] = _disp["zscore_value"].map(lambda v: f"{v:.4f}")
                            st.dataframe(
                                _disp, hide_index=True, use_container_width=True,
                                column_config={
                                    "date":         st.column_config.TextColumn("Date"),
                                    "signal":       st.column_config.TextColumn("Signal"),
                                    "zscore_value": st.column_config.TextColumn("Z-score"),
                                },
                            )
                else:
                    st.warning("Not enough data to compute Z-score for the selected window.")

        except Exception as exc:
            st.error(f"Could not compute dislocation analysis: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # Tab 5 — Network
    # ══════════════════════════════════════════════════════════════════════

    if _show_pa_net:

        with st.container(border=True):
            section_header(
                "MST Network Position",
                f"Where {ticker_a} and {_b_disp} sit in the Minimum Spanning Tree — "
                "their neighbors, shortest path, and cluster membership.",
            )

            if _b_is_base_asset:
                # Base assets (USD/TRY, Gold) are NOT in the universe's
                # MST — the MST is built from the cross-ticker
                # correlation matrix and these are exogenous series.
                # Showing "USD/TRY not found in the MST" wording (which
                # is what the fall-through path produces) would suggest
                # a bug; the friendlier explanation is that this view
                # doesn't apply to base-asset comparisons.
                st.info(
                    f":material/info: **{_b_disp}** is an exogenous base "
                    "asset and is not part of the universe's MST. "
                    "Switch the **Compare against** picker back to "
                    "*Another stock* to see network position."
                )
                return

            mst_edges_df = load_mst_edges()
            mst_metrics_df = load_mst_metrics()
            cluster_df = load_cluster_assignments()

            if not mst_edges_df.empty and not mst_metrics_df.empty and HAS_NETWORKX:
                G = nx.Graph()
                for _, row in mst_edges_df.iterrows():
                    G.add_edge(row["source"], row["target"], weight=row["distance"])

                both_in_mst = ticker_a in G.nodes() and ticker_b in G.nodes()

                if both_in_mst:
                    try:
                        path = nx.shortest_path(G, ticker_a, ticker_b, weight="weight")
                        path_length = nx.shortest_path_length(G, ticker_a, ticker_b, weight="weight")
                    except nx.NetworkXNoPath:
                        path = []
                        path_length = float("inf")

                    neighbors_a = list(G.neighbors(ticker_a))
                    neighbors_b = list(G.neighbors(ticker_b))

                    cluster_a = cluster_b = "N/A"
                    if not cluster_df.empty:
                        _ca = cluster_df[cluster_df["ticker"] == ticker_a]
                        _cb = cluster_df[cluster_df["ticker"] == ticker_b]
                        if not _ca.empty:
                            cluster_a = str(_ca.iloc[0]["cluster_id"])
                        if not _cb.empty:
                            cluster_b = str(_cb.iloc[0]["cluster_id"])
                    same_cluster = cluster_a == cluster_b and cluster_a != "N/A"

                    _p1, _p2, _p3, _p4 = st.columns(4)
                    _p1.metric(f"{ticker_a} Degree", len(neighbors_a))
                    _p2.metric(f"{_b_disp} Degree", len(neighbors_b))
                    _p3.metric("MST Path Length", f"{path_length:.4f}" if path else "No path")
                    _p4.metric("Same Cluster?", "Yes" if same_cluster else "No",
                                delta=f"Clusters {cluster_a} & {cluster_b}" if not same_cluster else f"Cluster {cluster_a}",
                                delta_color="normal" if same_cluster else "off")

                    if path:
                        st.markdown(f"**Shortest MST path:** {' -> '.join(path)} ({len(path)-1} hops)")

                    highlight_nodes = set(path) | set(neighbors_a) | set(neighbors_b)
                    highlight_nodes.add(ticker_a)
                    highlight_nodes.add(ticker_b)

                    sub_nodes = set()
                    for n in highlight_nodes:
                        if n in G.nodes():
                            sub_nodes.add(n)
                            for neighbor in G.neighbors(n):
                                sub_nodes.add(neighbor)

                    H = G.subgraph(sub_nodes)

                    pos = _subgraph_layout(
                        tuple(sorted(H.nodes())),
                        tuple(sorted((u, v) for u, v in H.edges())),
                    )

                    sector_map = dict(zip(mst_metrics_df["ticker"], mst_metrics_df["sector"]))
                    sectors = sorted(set(sector_map.get(n, "Unknown") for n in H.nodes()))
                    color_map = {s: SECTOR_PALETTE[i % len(SECTOR_PALETTE)] for i, s in enumerate(sectors)}

                    path_edges = set()
                    if path:
                        for i in range(len(path) - 1):
                            path_edges.add((path[i], path[i+1]))
                            path_edges.add((path[i+1], path[i]))

                    edge_x, edge_y = [], []
                    path_edge_x, path_edge_y = [], []
                    for u, v in H.edges():
                        x0, y0 = pos[u]
                        x1, y1 = pos[v]
                        if (u, v) in path_edges:
                            path_edge_x.extend([x0, x1, None])
                            path_edge_y.extend([y0, y1, None])
                        else:
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])

                    fig_sub = go.Figure()

                    fig_sub.add_trace(go.Scatter(
                        x=edge_x, y=edge_y, mode="lines",
                        line=dict(width=0.8, color="#D4D8E0"),
                        hoverinfo="none", showlegend=False,
                    ))
                    if path_edge_x:
                        fig_sub.add_trace(go.Scatter(
                            x=path_edge_x, y=path_edge_y, mode="lines",
                            line=dict(width=3, color=get_colors()["secondary"]),
                            hoverinfo="none", name="Shortest path",
                        ))

                    for node in H.nodes():
                        x, y = pos[node]
                        sec = sector_map.get(node, "Unknown")
                        is_selected = node in (ticker_a, ticker_b)
                        is_on_path = node in set(path) if path else False

                        if is_selected:
                            size = 22
                            border_w = 3
                            border_c = get_colors()["secondary"]
                        elif is_on_path:
                            size = 14
                            border_w = 2
                            border_c = "#FF9F1C"
                        else:
                            size = 10
                            border_w = 1
                            border_c = "white"

                        fig_sub.add_trace(go.Scatter(
                            x=[x], y=[y], mode="markers+text",
                            marker=dict(size=size, color=color_map.get(sec, get_colors()["muted"]),
                                        line=dict(width=border_w, color=border_c)),
                            text=node,
                            textposition="top center",
                            textfont=dict(size=9 if is_selected else 7,
                                          color="#2B2D42" if is_selected else get_colors()["muted"]),
                            hovertext=f"<b>{node}</b><br>Sector: {sec}<br>{'SELECTED' if is_selected else ''}",
                            hoverinfo="text",
                            showlegend=False,
                        ))

                    apply_chart_style(fig_sub,
                        height=500,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        showlegend=False,
                    )
                    render_chart(fig_sub, chart_id="pa_subgraph", filename_base="mst_subgraph",
                                 title_key="pa_subgraph", default_title="MST Subgraph")

                    col_na, col_nb = st.columns(2)
                    with col_na:
                        st.markdown(f"**{ticker_a} MST Neighbors** ({len(neighbors_a)})")
                        if neighbors_a:
                            _na_data = []
                            for n in neighbors_a:
                                d = G[ticker_a][n]["weight"]
                                s = sector_map.get(n, "")
                                _na_data.append({"Neighbor": n, "Distance": f"{d:.4f}", "Sector": s})
                            st.dataframe(pd.DataFrame(_na_data), hide_index=True, use_container_width=True)
                    with col_nb:
                        st.markdown(f"**{ticker_b} MST Neighbors** ({len(neighbors_b)})")
                        if neighbors_b:
                            _nb_data = []
                            for n in neighbors_b:
                                d = G[ticker_b][n]["weight"]
                                s = sector_map.get(n, "")
                                _nb_data.append({"Neighbor": n, "Distance": f"{d:.4f}", "Sector": s})
                            st.dataframe(pd.DataFrame(_nb_data), hide_index=True, use_container_width=True)

                else:
                    missing = []
                    if ticker_a not in G.nodes():
                        missing.append(ticker_a)
                    if ticker_b not in G.nodes():
                        missing.append(ticker_b)
                    st.warning(
                        f"{'  '.join(missing)} not found in the MST. "
                        "This may be because they were dropped during the coverage filter."
                    )
            elif not HAS_NETWORKX:
                st.warning("Install `networkx` to display MST network position.")
            else:
                st.info("Run the clustering pipeline to generate MST data.")
