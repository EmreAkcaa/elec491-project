# UI_REFERENCE

The Streamlit app has two top-level pages selected via a `st.segmented_control`
in `app/dashboard.py:130`:

1. **Market Overview** — `app/dashboard.py` (entry; 1025 lines).
2. **Pair Analysis** — `app/pair_analysis.py:render` (called from
   `dashboard.py:148-152`).

Inside Market Overview, the **EEE Analysis** sub-tab dispatches to
`app/eee_analysis.py:render` (called from `dashboard.py:1023-1025`).

---

## Navigation model

`dashboard.py` is the only Streamlit "page" registered (no `pages/`
directory). Page switching is done by re-rendering the same script with a
different `nav_page` session-state value:

```
st.segmented_control(
    "Navigate",
    ["Market Overview", "Pair Analysis"],
    key="nav_page",
    default="Market Overview",
)
```

The "Pair Analysis" branch calls `pair_analysis.render(...)` and then
`st.stop()` to skip the rest of the script. Any chart in Market Overview
that wants to jump to Pair Analysis sets the deferred flag
`_goto_pair_analysis` and reruns:

```python
if st.session_state.pop("_goto_pair_analysis", False):
    st.session_state["nav_page"] = "Pair Analysis"
```

---

## Page × tab × chart inventory

### Page 1 — Market Overview (`dashboard.py`)

Settings popover and key metrics live above the tabs (date range, data
freshness, ticker count, day count, avg/median correlation).

#### Tab 1 — Data & Stats (`tab_data`)

| Section | Chart | Data file | Library |
|---|---|---|---|
| Coverage & Prices | Coverage bar (per-ticker %) | `data/processed/coverage_report.csv` | px.bar |
| Coverage & Prices | Normalised price line (rebased to 100) + XU100 | `adj_close.parquet`, `xu100.parquet` | go.Scatter |
| Stats & Returns | Per-ticker descriptive stats table | `summary_stats.parquet` | st.dataframe |
| Stats & Returns | Return distribution histogram (selected ticker) | `log_returns.parquet` | go.Histogram |
| Market Summary | 5 metric tiles (avg/median/std/min/max corr) | `pipeline_metadata.json` | st.metric |

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

#### Tab 5 — Pairs & Dislocations (`tab_pairs`)

Two inner tabs (`tab_top`, `tab_bottom`) inside the "Top/Bottom Pairs" section.

| Section | Chart | Data file | Library |
|---|---|---|---|
| Top/Bottom | Top correlated pairs table | `top_bottom_pairs.csv` | st.dataframe |
| Top/Bottom | Bottom correlated pairs table | `top_bottom_pairs.csv` | st.dataframe |
| Distribution | Correlation distribution histogram | `pearson_corr.parquet` (upper triangle) | go.Histogram |
| Dislocation | Ranked candidate pairs table with rank score | `dislocation_candidates.csv` | st.dataframe |

#### Tab 6 — EEE Analysis (`tab_eee` → `app/eee_analysis.py:render`)

Four sub-tabs: `RMT Denoising`, `Graphical LASSO`, `Wavelet Multi-Scale`,
`Transfer Entropy`.

| Sub-tab | Chart | Data file | Library |
|---|---|---|---|
| RMT | Eigenvalue spectrum vs MP bounds | `eigenvalue_spectrum.csv` | go.Scatter / go.Bar |
| RMT | Denoised MST network | `denoised_mst_edges.csv` | go.Scatter |
| Glasso | Sparse partial-correlation network | `partial_corr_edges.csv` | go.Scatter |
| Wavelet | Wavelet MST network at selected scale | `wavelet_mst_edges_scale{1..7}.csv` | go.Scatter |
| Wavelet | Wavelet correlation distribution at selected scale | `wavelet_corr_scale{1..7}.parquet` | go.Histogram |
| TE | Directed TE network (top-N edges by net TE) | `te_network_edges.csv` | go.Scatter (with arrows) |

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

## `app/utils.py` API (28 cached loaders + helpers)

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
| `load_denoised_corr()` | `denoised_corr.parquet` | (currently no UI consumer) |
| `load_denoised_mst_edges()` | `denoised_mst_edges.csv` | eee tab RMT |
| `load_partial_corr()` | `partial_corr.parquet` | (currently no UI consumer) |
| `load_precision_matrix()` | `precision_matrix.parquet` | **no consumer yet** (added this session — see FUTURE_WORK F-1) |
| `load_partial_corr_edges()` | `partial_corr_edges.csv` | eee tab Glasso |
| `load_glasso_metadata()` | `glasso_metadata.json` | eee tab Glasso |
| `load_wavelet_metadata()` | `wavelet_metadata.json` | eee tab Wavelet |
| `load_wavelet_mst_edges(scale)` | `wavelet_mst_edges_scale{n}.csv` | eee tab Wavelet |
| `load_wavelet_corr(scale)` | `wavelet_corr_scale{n}.parquet` | eee tab Wavelet |
| `load_te_edges()` | `te_network_edges.csv` | eee tab TE |
| `load_te_node_roles()` | `te_node_roles.csv` | eee tab TE |
| `load_te_matrix()` | `transfer_entropy_matrix.parquet` | eee tab TE |

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
