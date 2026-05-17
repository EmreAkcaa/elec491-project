# UI_REFERENCE

The Streamlit app has three top-level pages selected via a `st.segmented_control`
in `app/dashboard.py`:

1. **Market Overview** — `app/dashboard.py` (entry; ~1240 lines).
2. **Pair Analysis** — `app/pair_analysis.py:render`.
3. **Cross-Market** — `app/cross_market.py:render` (BIST 100 vs S&P 500 side-by-side).

Inside Market Overview, the **EEE Analysis** sub-tab dispatches to
`app/eee_analysis.py:render`.

---

## Universe switcher (Phase H)

The sidebar shows a **Dataset** dropdown when more than one universe has a
populated `data/<key>/results/pipeline_metadata.json` on disk. The selector
binds `key="universe"` directly to `st.session_state`, so changing it triggers
a Streamlit auto-rerun. Every cached loader in `app/utils.py` routes through
a universe-keyed underscore-prefixed function (`_load_X(universe, ...)`) so
BIST and S&P caches coexist; switching back to the previous universe hits a
warm cache.

The boot-time default still comes from the `DASHBOARD_UNIVERSE` env var so
`DASHBOARD_UNIVERSE=sp500 uv run streamlit run app/dashboard.py` lands
directly on the S&P universe.

Universes are registered in `app/universe_registry.py` — currently `bist`,
`sp500`, and `eeg_motor_left_right` (Phase I).

Each registered universe carries **capability flags** that the dashboard
consults to gate financial-only sections:

| Flag | Effect when False (EEG) |
|---|---|
| `has_pair_trading` | Pair Analysis nav option + Pairs & Dislocations sub-tab hidden |
| `has_snn` | Neuromorphic Signals sub-tab hidden in EEE Analysis |
| `has_index_series` | Price chart replaced with stacked voltage time-series; XU100/^GSPC overlay omitted |
| `has_anomaly_detection` | Return Anomalies section hidden in Data & Stats |
| `has_validation_report` | İş Yatırım validation row hidden in Data Freshness popover |
| `eligible_for_cross_market` | Universe filtered out of Cross-Market comparison |

Plus terminology fields (`item_label` / `items_label` / `sector_label` /
`series_label` / `series_units` / `network_label`) that drive axis labels
and captions throughout the dashboard, so EEG sees "Channels" / "Bandpass
voltage (µV)" / "Functional Connectivity Network" instead of the financial
defaults.

EEG sanity checks: when the active universe declares `sanity_check_groups`,
the Clustering & Network tab renders a per-group "members co-cluster"
badge. EEG ships three: central-motor (C3/Cz/C4), occipital (O1/Oz/O2),
prefrontal (Fp1/Fpz/Fp2).

Page-header behaviour:
- Browser tab title (`st.set_page_config(page_title=...)`) is set once at
  boot from the env-var default. It lags by one page-load when the user
  switches universes; this is a Streamlit API limitation
  (`set_page_config` cannot be re-called).
- The in-page header chip ("BIST 100 NETWORK ANALYSIS" / "S&P 500 NETWORK
  ANALYSIS") updates immediately on switch.

## Navigation model

`dashboard.py` is the only Streamlit "page" registered (no `pages/`
directory). Page switching is done by re-rendering the same script with a
different `nav_page` session-state value:

```python
# Build the nav dynamically: Pair Analysis is hidden when the active
# universe has has_pair_trading=False (e.g. EEG).
_nav_options = ["Market Overview"]
if _active_universe.has_pair_trading:
    _nav_options.append("Pair Analysis")
_nav_options.append("Cross-Market")

st.segmented_control(
    "Navigate", _nav_options, key="nav_page", default="Market Overview",
)
```

The "Pair Analysis" branch calls `pair_analysis.render(...)` and then
`st.stop()`; the "Cross-Market" branch calls `cross_market.render()` and
stops — that route runs BEFORE per-universe data loads since the page reads
from both `data/bist/` and `data/sp500/` directly.

Any chart in Market Overview that wants to jump to Pair Analysis sets the
deferred flag `_goto_pair_analysis`; analogously `_goto_cross_market` jumps
to the Cross-Market page:

```python
if st.session_state.pop("_goto_pair_analysis", False):
    st.session_state["nav_page"] = "Pair Analysis"
if st.session_state.pop("_goto_cross_market", False):
    st.session_state["nav_page"] = "Cross-Market"
```

---

## Page × tab × chart inventory

### Page 1 — Market Overview (`dashboard.py`)

Settings popover and key metrics live above the tabs (date range, data
freshness, ticker count, day count, avg/median correlation).

#### Tab 1 — Data & Stats (`tab_data`)

| Section | Chart | Data file | Library | Gate |
|---|---|---|---|---|
| Coverage & Prices | Coverage bar (per-ticker %) | `data/<universe>/processed/coverage_report.csv` | px.bar | always |
| Coverage & Prices | Normalised price line (rebased to 100) + index overlay (XU100 / ^GSPC) | `adj_close.parquet`, `xu100.parquet` | go.Scatter | `has_index_series` |
| Coverage & Prices | **EEG voltage time-series** — stacked, 10 evenly-spaced channels, first 30 s, vertically offset by ~4σ each | `adj_close.parquet` | go.Scatter | `not has_index_series` |
| Stats & Returns | Per-ticker (or per-channel) descriptive stats table | `summary_stats.parquet` | st.dataframe | always |
| Stats & Returns | Return / voltage distribution histogram (selected item) | `log_returns.parquet` | go.Histogram | always |
| Anomalies | Sortable anomaly table (`date`, `ticker`, `return_value`, `\|return\|`) | `data/<universe>/processed/anomalies.csv` | st.dataframe | `has_anomaly_detection` |
| Anomalies | Anomaly timeline scatter (date × ticker; triangle direction by sign, size by `\|return\|`) | `data/<universe>/processed/anomalies.csv` | go.Scatter | `has_anomaly_detection` |
| Market Summary | 5 metric tiles (avg/median/std/min/max corr) | `pipeline_metadata.json` | st.metric | always |

#### Tab 2 — Correlation (`tab_corr`)

Two sub-tabs: `Heatmap` and `Point-in-Time Snapshot`.

| Sub-tab | Chart | Data file | Library |
|---|---|---|---|
| Heatmap | Correlation heatmap (optionally reordered by dendrogram) | computed from `log_returns.parquet`; reorder from `dendrogram_order.json` | px.imshow |
| Point-in-Time | Sliderable correlation snapshot ending at chosen date | `log_returns.parquet` (windowed) | px.imshow |

#### Tab 3 — Clustering & Network (`tab_cluster`)

| Section | Chart | Data file | Library |
|---|---|---|---|
| Dendrogram | Hierarchical-clustering dendrogram | `linkage_matrix.npy`, `linkage_labels.json` | scipy + go.Scatter |
| MST | MST network diagram (kamada-kawai layout) | `mst_edges.csv`, `mst_node_metrics.csv` | go.Scatter (nodes/edges) |

#### Tab 4 — Rolling Analysis (`tab_rolling`)

Three sub-tabs: `Market Overview`, `Pair Correlation`, `Sector Breakdown`.

| Sub-tab | Chart | Data file | Library |
|---|---|---|---|
| Market | Rolling avg/median/q25/q75 correlation time series | `log_returns.parquet` (computed) | go.Scatter |
| Market | Min/max correlation range fill | computed | go.Scatter |
| Pair | Rolling pair correlation (selected pair, configurable window) | computed | go.Scatter |
| Pair | Pair price-spread time series | computed | go.Scatter |
| Sector | Intra vs inter-sector correlation | computed | go.Scatter |
| Sector | Per-sector intra correlation (faceted) | computed | go.Scatter |

> **Precompute-first dispatch (Market and Sector sub-tabs).** Reads
> `rolling_market_stats_w{60,120,252}.parquet` when `window ∈ {60,120,252}
> ∧ step=5 ∧ method="pearson" ∧ not expanding`, and
> `rolling_sector_stats.parquet` when `window=252 ∧ step=5 ∧
> method="pearson"`. Off-grid parameters fall back to the on-the-fly
> `_compute_market_stats` / `_compute_sector` caches. A caption shows
> which path ran.

#### Tab 5 — Pairs & Dislocations (`tab_pairs`)

Two inner tabs (`tab_top`, `tab_bottom`) inside the "Top/Bottom Pairs" section.

| Section | Chart | Data file | Library |
|---|---|---|---|
| Top/Bottom | Top correlated pairs table | `top_bottom_pairs.csv` | st.dataframe |
| Top/Bottom | Bottom correlated pairs table | `top_bottom_pairs.csv` | st.dataframe |
| Distribution | Correlation distribution histogram | `pearson_corr.parquet` (upper triangle) | go.Histogram |
| Dislocation | Ranked candidate pairs table with rank score | `dislocation_candidates.csv` | st.dataframe |

#### Tab 6 — EEE Analysis (`tab_eee` → `app/eee_analysis.py:render`)

Five sub-tabs: `RMT Denoising`, `Graphical LASSO`, `Wavelet Multi-Scale`,
`Transfer Entropy`, `Neuromorphic Signals`.

| Sub-tab | Chart | Data file | Library |
|---|---|---|---|
| RMT | Eigenvalue spectrum vs MP bounds | `eigenvalue_spectrum.csv` | go.Scatter / go.Bar |
| RMT | Raw / Denoised MST network (toggle); nodes sized by `betweenness_centrality` | `mst_edges.csv` + `mst_node_metrics.csv` (Raw) or `denoised_mst_edges.csv` + `denoised_mst_node_metrics.csv` (Denoised) | go.Scatter |
| RMT | Denoised correlation heatmap, dendrogram-ordered (±1 diverging RdBu) | `denoised_corr.parquet`, `dendrogram_order.json` | go.Heatmap |
| Glasso | Sparse partial-correlation network | `partial_corr_edges.csv` | go.Scatter |
| Glasso | Partial-correlation heatmap (clipped ±0.3, diagonal zeroed) | `partial_corr.parquet`, `dendrogram_order.json` | go.Heatmap |
| Glasso | Precision-matrix sparsity heatmap (`\|Θ\|>1e-3`, binary, off-diagonal) | `precision_matrix.parquet`, `dendrogram_order.json` | go.Heatmap |
| Wavelet | Wavelet MST at selected scale; nodes sized by centrality | `wavelet_mst_edges_scale{1..7}.csv`, `wavelet_mst_metrics_scale{1..7}.csv` | go.Scatter |
| Wavelet | Wavelet correlation distribution at selected scale | `wavelet_corr_scale{1..7}.parquet` | go.Histogram |
| Wavelet | Cross-scale summary table (avg corr, std, MST total weight, edge count, max betweenness, avg degree) | `wavelet_corr_scale*.parquet` + `wavelet_mst_edges_scale*.csv` + `wavelet_mst_metrics_scale*.csv` | st.dataframe |
| TE | Directed TE network (top edges by net TE) | `te_network_edges.csv` | go.Scatter (with arrows) |
| TE | Net information-flow heatmap (symmetric ±max, RdBu reversed, dendrogram-ordered) | `net_transfer_entropy_matrix.parquet`, `dendrogram_order.json` | go.Heatmap |
| Neuromorphic | Headline 5-metric KPI row (pairs trained, mean macro-F1, mean SNN Sharpe, mean classical Sharpe, mean Δ-Sharpe) + honest-framing caption (Δ-Sharpe = −1.11, beats classical on 5/20 pairs) | `snn_metrics.json` (`aggregate`, `per_pair`) | st.metric + st.caption |
| Neuromorphic | Per-pair leaderboard sorted by Δ-Sharpe (F1 / SNN-Sh / Cls-Sh / Δ-Sh / hit rate / trade counts) | `snn_metrics.json` (`per_pair`) | st.dataframe |
| Neuromorphic | Pair selector → SNN signal timeline (Z-score + BUY/SELL markers + ±2 reference lines) | `snn_signals/{pair_id}.parquet` | go.Scatter |
| Neuromorphic | Training history: train loss + val loss (left axis) + val macro-F1 (right axis) across epochs | `snn_training_history.csv` | go.Scatter (dual y-axes) |
| Neuromorphic | Sample-pair output-neuron spike raster (HOLD/BUY/SELL bands on y, SNN ticks on x) | `snn_spike_raster_sample.parquet` | go.Scatter (line-ns-open markers) |
| Neuromorphic | Sample-pair output-layer membrane V(t) trace for the 3 output neurons, with horizontal V_th reference | `snn_membrane_sample.parquet` | go.Scatter |
| Neuromorphic | Architecture / hyperparameter expander (read from `snn_metrics.json:config` block) | `snn_metrics.json` (`config`, `n_inputs`) | st.markdown |

---

### Page 2 — Pair Analysis (`pair_analysis.py:render`)

Pair selector at top (two ticker dropdowns + warning banner via
`utils.check_ticker_pair_warnings`). Five sub-tabs:

| Sub-tab | Section | Chart | Data file | Library |
|---|---|---|---|---|
| Overview | Prices | Normalised price comparison (selected pair + XU100) | `adj_close.parquet`, `xu100.parquet` | go.Scatter |
| Overview | Returns | Daily return scatter (`x = a, y = b`, regression line) | `log_returns.parquet` | go.Scatter |
| Correlation | Rolling | Rolling pair correlation (window/method/type configurable) | `log_returns.parquet` | go.Scatter |
| Risk | Distribution | Return distribution histograms (per ticker) | `log_returns.parquet` | go.Histogram |
| Risk | Volatility | Rolling realised volatility (per ticker) | `log_returns.parquet` | go.Scatter |
| Risk | Drawdown | Drawdown time series (per ticker) | `adj_close.parquet` | go.Scatter |
| Spread | Log spread | Log-price spread `log(Pb) - β log(Pa)` | computed | go.Scatter |
| Spread | Z-score | Rolling Z-score with entry/exit thresholds and signals | computed | go.Scatter |
| Network | MST sub-graph | Local MST around the pair (k-hop neighbourhood) | `mst_edges.csv` | go.Scatter |

---

### Page 3 — Cross-Market Comparison (`cross_market.py:render`)

Universe-independent — reads from BOTH `data/bist/` and `data/sp500/` directly
via the underscore-prefixed loaders (`_load_X(universe, ...)`), so the sidebar
selector does NOT affect this page.

| Section | Chart / Block | Data file(s) | Library |
|---|---|---|---|
| Headline KPIs | 8 metric tiles: N, D_eff, top-eig share, MST sector purity per universe (BIST / S&P) | `data/comparison_bist_vs_sp500.csv` | st.metric |
| Spectral structure (RMT) | Eigenvalue spectrum side-by-side (log y, signal-mode colouring, MP-bound overlay) | `data/bist/results/eigenvalue_spectrum.csv` + `data/sp500/results/eigenvalue_spectrum.csv` | go.Bar |
| MST topology | Both MSTs side-by-side via kamada-kawai layout, sector-coloured, sized by betweenness; hub labels only above 45% of max-btw (keeps S&P 485-node plot readable) | `mst_edges.csv` + `mst_node_metrics.csv` (per universe) | go.Scatter |
| MST topology | Top-5 hubs per universe (markdown list with sector + btw) | `mst_node_metrics.csv` | st.markdown |
| Crisis windows | Grouped bar chart: avg pairwise corr in ±60-day before/during/after buckets, for COVID / Russia-Ukraine / Türkiye-earthquake, × {BIST, S&P} | `data/comparison_bist_vs_sp500.csv` (`<date>_<phase>` rows) | go.Bar (group mode) |
| Pairwise dependence + Glasso/TE parity | Side-by-side stats: mean/median/std/max-abs correlation, signal eigenvalues, signal variance share, Glasso edge count + sector purity, TE edge count + sector purity | `data/comparison_bist_vs_sp500.csv` | st.markdown |
| Top dislocation pair | One per universe: ρ, β, half-life, current Z | `dislocation_candidates.csv` (per universe) | st.markdown |
| Methodology footnote | Identical hyperparameter disclosure, N-disparity caveat, manual_anomaly_nulls reference, xu100.parquet-for-^GSPC quirk | — | st.markdown |

---

## `app/utils.py` API (40 cached loaders + helpers)

### Loaders (cached with `@st.cache_data`)

Grouped by purpose. All return empty `DataFrame`/`dict`/`Series`/`None` when
the file is missing — no exceptions.

| Loader | Reads | Used by |
|---|---|---|
| `load_adj_close()` | `data/processed/adj_close.parquet` | dashboard, pair_analysis |
| `load_log_returns()` | `data/processed/log_returns.parquet` | dashboard, pair_analysis |
| `load_summary_stats()` | `data/results/summary_stats.parquet` | dashboard tab 1 |
| `load_batch_corr()` | `data/results/pearson_corr.parquet` | dashboard tab 2/5 |
| `load_coverage()` | `data/processed/coverage_report.csv` | dashboard tab 1 |
| `load_top_bottom()` | `data/results/top_bottom_pairs.csv` | dashboard tab 5 |
| `load_metadata()` | `data/results/pipeline_metadata.json` | dashboard headline |
| `load_fetch_metadata()` | `data/raw/fetch_metadata.json` | settings popover |
| `load_xu100()` | `data/raw/xu100.parquet` | dashboard tab 1, pair_analysis |
| `load_linkage()` | `linkage_matrix.npy` + `linkage_labels.json` | dashboard tab 3 |
| `load_dendrogram_order()` | `dendrogram_order.json` | dashboard tab 2 |
| `load_cluster_assignments()` | `cluster_assignments.csv` | eee_analysis (sector map) |
| `load_mst_edges()` | `mst_edges.csv` | dashboard tab 3, pair_analysis |
| `load_mst_metrics()` | `mst_node_metrics.csv` | dashboard tab 3 |
| `load_dislocation_candidates()` | `dislocation_candidates.{parquet,csv}` | dashboard tab 5, pair_analysis |
| `load_eigenvalue_spectrum()` | `eigenvalue_spectrum.csv` | eee tab RMT |
| `load_denoised_corr()` | `denoised_corr.parquet` | eee tab RMT (denoised correlation heatmap) |
| `load_denoised_mst_edges()` | `denoised_mst_edges.csv` | eee tab RMT (denoised MST) |
| `load_partial_corr()` | `partial_corr.parquet` | eee tab Glasso (partial-correlation heatmap) |
| `load_precision_matrix()` | `precision_matrix.parquet` | eee tab Glasso (precision sparsity heatmap) |
| `load_partial_corr_edges()` | `partial_corr_edges.csv` | eee tab Glasso |
| `load_glasso_metadata()` | `glasso_metadata.json` | eee tab Glasso |
| `load_wavelet_metadata()` | `wavelet_metadata.json` | eee tab Wavelet |
| `load_wavelet_mst_edges(scale)` | `wavelet_mst_edges_scale{n}.csv` | eee tab Wavelet |
| `load_wavelet_corr(scale)` | `wavelet_corr_scale{n}.parquet` | eee tab Wavelet |
| `load_te_edges()` | `te_network_edges.csv` | eee tab TE |
| `load_te_node_roles()` | `te_node_roles.csv` | eee tab TE |
| `load_te_matrix()` | `transfer_entropy_matrix.parquet` | eee tab TE |
| `load_net_te_matrix()` | `net_transfer_entropy_matrix.parquet` | eee tab TE (net flow heatmap) |
| `load_denoised_mst_metrics()` | `denoised_mst_node_metrics.csv` | eee tab RMT (denoised MST node sizing) |
| `load_wavelet_mst_metrics(scale)` | `wavelet_mst_metrics_scale{n}.csv` | eee tab Wavelet (per-scale MST node sizing + summary table) |
| `load_anomalies()` | `data/processed/anomalies.csv` | dashboard tab 1 (Return Anomalies section) |
| `load_rolling_market_stats_precomputed(window)` | `rolling_market_stats_w{n}.parquet` | dashboard Rolling Analysis Market sub-tab (precompute-first path) |
| `load_rolling_sector_stats_precomputed()` | `rolling_sector_stats.parquet` | dashboard Rolling Analysis Sector sub-tab (precompute-first path) |
| `load_snn_metrics()` | `data/results/snn_metrics.json` | eee tab Neuromorphic Signals (KPI row, honest-framing caption, leaderboard, architecture expander) |
| `load_snn_pair_list()` | `data/results/snn_pair_list.csv` | eee tab Neuromorphic Signals (pair selector source list) |
| `load_snn_signals(pair_id)` | `data/results/snn_signals/{pair_id}.parquet` | eee tab Neuromorphic Signals (per-pair signal timeline) |
| `load_snn_training_history()` | `data/results/snn_training_history.csv` | eee tab Neuromorphic Signals (training convergence chart) |
| `load_snn_raster_sample()` | `data/results/snn_spike_raster_sample.parquet` | eee tab Neuromorphic Signals (sample-pair output-neuron raster) |
| `load_snn_membrane_sample()` | `data/results/snn_membrane_sample.parquet` | eee tab Neuromorphic Signals (sample-pair membrane V(t) trace) |

### UI helpers

| Symbol | Purpose | File:line |
|---|---|---|
| `get_colors()` | Returns active palette dict | `utils.py:62` |
| `apply_chart_style(fig, ...)` | Apply consistent layout (axes, fonts, margins) | `utils.py:67` |
| `render_chart(fig, chart_id, filename_base, ...)` | Theme-aware chart renderer + PNG export hook | `utils.py:87` |
| `inject_custom_css()` | Project CSS injected once per session | `utils.py:125` |
| `page_header(title, subtitle)` | Top-of-page banner | `utils.py:274` |
| `section_header(title, description)` | Card-like section divider used everywhere | `utils.py:283` |
| `draw_event_markers(fig, events)` | Add `add_shape`+`add_annotation` for BIST events | `utils.py:561` |
| `event_marker_manager_ui(...)` | Sidebar UI to toggle/edit event markers | `utils.py:594` |
| `check_ticker_pair_warnings(a, b, ...)` | Coverage/window warnings for pair page | `utils.py:684` |
| `render_warnings(issues)` | Surface warnings as `st.warning` blocks | `utils.py:796` |
| `_plot_matrix_heatmap(matrix, ordered_tickers, ...)` | Dendrogram-ordered square-matrix heatmap; reused by RMT denoised heatmap, Glasso partial-corr and precision heatmaps, and TE net-flow heatmap | `eee_analysis.py:_plot_matrix_heatmap` |
| `_plot_network(edges_df, sector_map, ..., node_metrics, size_metric)` | Network graph helper with optional centrality-based node sizing (falls back to degree if `node_metrics` is missing) | `eee_analysis.py:_plot_network` |
| `SECTOR_PALETTE`, `CHART_LAYOUT` | Theming constants | `utils.py` (module-level) |

### Theming (`app/chart_themes.py`)

`render_theme_sidebar()` renders a sidebar dropdown. Selected theme writes
to `st.session_state["chart_theme"]`; `get_colors()` and `render_chart`
resolve from that key. Themes include light/dark/colorblind palettes
defined as constant dicts at module top.

### Export (`app/chart_export.py`)

`render_chart` calls into `chart_export.py` to add a "Download PNG" button
under each plot. Uses `kaleido` if available; fallback to plotly's static
image API.

---

## Session-state keys used across pages

| Key | Set by | Read by |
|---|---|---|
| `nav_page` | `st.segmented_control` in dashboard | dashboard top dispatch |
| `_goto_pair_analysis` | jump buttons in dashboard tabs | dashboard top dispatch (deferred set) |
| `chart_theme` | sidebar theme switcher | every `render_chart` call |
| `events` | event-marker manager UI | rolling correlation chart overlays |
| `pair_a`, `pair_b` | pair_analysis ticker dropdowns | pair_analysis chart computations |
| `heat_method` | tab 2 method selector | `_compute_corr` cache key |

---

## When to compute vs read

The dashboard caches three classes of computations module-level (see
`dashboard.py:48-95`):

```python
_compute_corr(returns_json, min_periods, method)           # corr matrix
_pit_corr(ret_json, end_date_str, window, method)          # point-in-time corr
_mst_layout(_edges_json)                                   # kamada-kawai layout
_compute_market_stats(ret_json, window, step, method, expanding)
_compute_pair(ret_json, a, b, window, method, wtype)
_compute_sector(ret_json, sec_map_items, window, step, method)
```

These keys hash the JSON-serialised return series — adding a new computation
should follow the same pattern (don't pass DataFrames through `@st.cache_data`
directly because pandas isn't hashable cleanly).
