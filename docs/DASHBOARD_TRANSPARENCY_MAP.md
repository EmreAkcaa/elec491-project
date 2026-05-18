# StoNeCoAl Dashboard — Full Transparency Map

**Verified against `main` HEAD = `1733a3c`** (post PRs #30/#31/#32/#33/#34/#35/#36/#37/#38/#39/#40 — Sprint 1 landed). All file paths, line numbers, algorithm names, and hyperparameter values were read directly from source — not from docs, not from subagent reports.

**Conventions used throughout:**

- `path/like/this.py:LINE` is a code location verified in this read.
- `data/<u>/...` means `data/{bist, bist_usd, bist_gold, sp500, eeg_motor_left_right}/...` — the five universe-keyed roots (`app/universe_registry.py:UNIVERSES`).
- **U** = user-tunable widget · **C** = config-driven (`config/settings*.yaml`) · **H** = hardcoded in source · **B** = baked at pipeline-run time, frozen in artifact.

---

## 0 · Top-level architecture in 60 seconds

```
yfinance / PhysioNet
        │
        ▼
data/<u>/raw/prices_raw.parquet                       ← src/data_acquisition.py, src/eeg_acquisition.py
        │
        ▼  (Step 1 — Preprocessing)
data/<u>/processed/                                    ← src/preprocessing.py
   ├─ adj_close.parquet         (dates × tickers)
   ├─ log_returns.parquet       (dates × tickers, NaN-preserving)
   ├─ coverage_report.csv       (per-ticker availability %)
   └─ anomalies.csv             (|return| > 0.30 flags)
        │
        ├──► Step 2  src/analysis.py            → pearson_corr, distance_matrix, summary_stats, top_bottom_pairs
        ├──► Step 3  src/clustering.py          → linkage_matrix.npy, dendrogram_order, cluster_assignments, mst_edges, mst_node_metrics
        ├──► Step 4  src/rolling_correlation.py → rolling_market_stats_w{60,120,252}, rolling_sector_stats
        ├──► Step 5  src/rmt_denoising.py       → eigenvalue_spectrum, denoised_corr, denoised_mst_*
        ├──► Step 6  src/partial_correlation.py → partial_corr, precision_matrix, partial_corr_edges, glasso_metadata
        ├──► Step 7  src/pair_dislocation.py    → dislocation_candidates
        ├──► Step 8  src/wavelet_analysis.py    → wavelet_corr_scale{1..7}, wavelet_mst_*scale{1..7}, wavelet_metadata
        ├──► Step 9  src/transfer_entropy.py    → transfer_entropy_matrix, _raw, _pvalues, _significance, net_transfer_entropy_matrix, te_network_edges, te_node_roles
        ├──► Step 10 src/info_theory.py         → mi_matrix, mi_gaussian_matrix, mi_nonlinear_excess, rolling_info_theory, it_summary, regime_kl, entropy_rate_signs
        └──► Step 11 src/snn_signals.py [opt]   → snn_metrics, snn_pair_list, snn_signals_*, snn_training_history, snn_*_sample
        │
        ▼
data/<u>/results/*.parquet/csv/json/npy
        │
        ▼
app/utils.py — ~50 `@st.cache_data load_*` loaders (one per artifact)
        │
        ▼
app/dashboard.py / cross_market.py / pair_analysis.py / eee_analysis.py
```

**Important**: nothing on the dashboard is computed from raw prices live — only the **full-period correlation** (`_compute_corr`), the **windowed point-in-time correlation** (`_pit_corr`), **rolling stats for off-grid params** (`_compute_market_stats`, `_compute_pair`, `_compute_sector`), the **pair-dislocation Z-score** (`_compute_dislocation`), and **rolling volatility** (`_rolling_vol`) are live computes on top of cached returns. All heavyweight analytics (RMT, GLASSO, wavelet, TE, IT, SNN) are precomputed at pipeline time, written to disk, and read as artifacts.

---

## 1 · Sidebar + global session state

| Control | Where | Default | Affects |
|---|---|---|---|
| **Universe selector** (BIST 100 — Türkiye / BIST 100 — USD-denominated / BIST 100 — Gold-denominated / S&P 500 — United States / EEG Motor Imagery — PhysioNet) | sidebar selectbox `dashboard.py:239` | env var `DASHBOARD_UNIVERSE` or first available | Routes every loader; `current_universe()` returns the active key. Five universes total; BIST-USD and BIST-Gold are numéraire-converted views of BIST-TRY (see §2 Numéraire sensitivity). |
| **Theme controls** | sidebar (chart_themes.py) | DEFAULT_THEME | Color palette, fonts, export sizing |
| **Top-nav** (Cross-Market / Market Overview / Pair Analysis) | `dashboard.py:311 st.segmented_control` | Cross-Market for finance, Network Overview for EEG | Branches the entire script via `if _nav == ...` |
| **Date range** | Settings popover, `st.date_input` `dashboard.py:537` | (min, max) of `adj_close.parquet` | `start_dt`, `end_dt` slice `returns`/`prices_window`; flows into `returns_cache_key` (universe + dates + shape), invalidating cached computes |

**Universe capability flags** (`app/universe_registry.py:25`, frozen dataclass `Universe`) drive *what tabs and sub-tabs even render*:

| Flag | BIST/S&P (finance) | EEG (neuroscience) | Effect |
|---|---|---|---|
| `domain` | "finance" | "neuroscience" | Switches charts/labels (e.g. EEG voltage plot instead of normalized prices) |
| `has_index_series` | True | False | Hides XU100/^GSPC overlay |
| `has_pair_trading` | True | False | Hides Pair Analysis top-nav, Pairs & Dislocations sub-tab |
| `has_snn` | True | False | Hides EEE → Neuromorphic Signals sub-tab |
| `eligible_for_cross_market` | True | False | Hides Cross-Market top-nav option |
| `sanity_check_groups` | dict of `{label → ticker list}` | EEG triples | Drives the green/yellow banners in Clustering & Network |
| `item_label`, `items_label`, `sector_label`, `series_label`, `series_units` | "Ticker", "Tickers", "Sector", "Log return", "" | "Channel", "Channels", "Region", "Bandpass voltage", "μV" | Label substitution everywhere |

**Universe-survival reality** (verified by reading `data/<u>/processed/log_returns.parquet`): BIST 102 listed → **73 survive** (1,543 trading days); S&P 500 → **485 survive** (1,546 trading days) after 90% coverage filter + dual-class dedupe (GOOG, FOX, NWS).

---

## 2 · Cross-Market page (`app/cross_market.py`)

**Purpose**: BIST vs S&P side-by-side with identical pipeline. EEG is excluded by `eligible_for_cross_market=False`. Page is **universe-independent** — does NOT use `current_universe()`; reads from both universes directly via underscored `_load_*("bist")` / `_load_*("sp500")`.

**Special dependency**: `data/comparison_bist_vs_sp500.csv` — produced offline by `scripts/sp500_vs_bist.py` (not part of `run_pipeline.py`). If missing, the page hard-errors with "Run both pipelines and then `scripts/sp500_vs_bist.py`".

| Section | Code | Data sources | What you see | User knobs |
|---|---|---|---|---|
| **Headline KPIs** | `cross_market.py:518` | `comparison_bist_vs_sp500.csv` rows: `N`, `D_eff`, `top_eigenvalue_share`, `mst_sector_purity` | 8 KPI cards (BIST + S&P pairs) | None |
| **Eigenvalue spectra** | `cross_market.py:543` | `_load_eigenvalue_spectrum("bist"/"sp500")` ← `data/<u>/results/eigenvalue_spectrum.csv` (from `src/rmt_denoising.py`) | Two bar charts: eigenvalues on log y-axis, signal (>MP upper) coloured, noise grey, MP upper as dashed line | None |
| **MST topology** | `cross_market.py:574` | `_load_mst_edges("bist"/"sp500")` + `_load_mst_metrics(...)` | Two Kamada-Kawai network graphs, nodes coloured by sector, sized by betweenness; hub names shown only for nodes above 45% of max betweenness range (`cross_market.py:190`) | None |
| **Crisis windows** | `cross_market.py:613` | `comparison_bist_vs_sp500.csv` window cells around 3 events (`_CRISIS_EVENTS`: COVID 2020-03-11, Ukraine 2022-02-24, Türkiye-quakes 2023-02-06) | Bar chart of mean pairwise correlation in ±60-day window per market per event | None |
| **Pairwise dependence + Glasso/TE parity** | `cross_market.py:631` | `comparison_bist_vs_sp500.csv` rows: `mean_corr`, `median_corr`, `std_corr`, `max_abs_corr`, `n_signal_eigenvalues`, `signal_variance_share`, `glasso_n_edges/sector_purity`, `te_n_edges/sector_purity` | Side-by-side markdown blocks | None |
| **Top dislocation pair contrast** | `cross_market.py:656` | `_load_dislocation_candidates("bist"/"sp500").head(1)` | ρ/β/half-life/Z for #1 pair each market | None |
| **Numéraire sensitivity** | `cross_market.py:_render_bist_numeraire_section()` ~370+ | `data/bist_gold/` + `data/bist_usd/` + `data/results/numeraire_decomposition.json` + `docs/figures/numeraire_sector_shift.svg` | Three eigenvalue spectra (TRY/USD/Gold) + sector-decomposition SVG + interpretation paragraph | None |
| **Methodology footnote** | `cross_market.py:697` | Static markdown | Caveats about identical params, N=73 vs 485, manual NaNs | None |

**✅ Bug residue cleared (PR #37, Sprint 1)**: the `importlib.reload(_ur)` antipattern that lived at `cross_market.py:467-469` was removed in PR-B; an explanatory `NOTE:` comment at `cross_market.py:465` flags why it was excised. The full quartet (`dashboard.py` PR #23, `pair_analysis.py` PR #33, `eee_analysis.py` PR #23, `cross_market.py` PR #37) is now consistent — no more Universe-class churn or "Tried to use SessionInfo before it was initialized" warnings traceable to module reloads.

---

## 3 · Market Overview (or "Network Overview" for EEG)

Six sub-tabs declared at `dashboard.py:448-453`:

```python
_tab_labels = ["Data & Stats", "Correlation", "Clustering & Network", "Rolling Analysis"]
if has_pair_trading: _tab_labels.append("Pairs & Dislocations")
_tab_labels.append("EEE Analysis")
```

### 3.1 Data & Stats (`tab_data`)

| Section | Code | Reads | Algorithm | User knobs |
|---|---|---|---|---|
| **Hero strip** (sector-recovery validation) | `dashboard.py:405-...` | `cluster_assignments.csv` | Cluster purity (post PR #34, was ARI/NMI) | None |
| **Coverage heatmap** (left) | `dashboard.py:474-505` | `load_coverage()` ← `data/<u>/processed/coverage_report.csv` (from `src/preprocessing.py:compute_coverage`) | Per-ticker non-NaN-day count divided by total days; horizontal bar with 90% threshold line | None |
| **Normalized price performance** (right, finance) | `dashboard.py:507-773` | `prices_window` (= `adj_close.parquet` sliced by date range) + `load_xu100()` ← `data/<u>/processed/index_series.parquet` (only for finance) | N > 80 → 10/50/90 percentile envelope + median + bold index overlay. N ≤ 80 → per-ticker spaghetti (BIST). Index series rebased to 100. | None (post PR #32 envelope is automatic for large N) |
| **Voltage time-series** (right, EEG) | `dashboard.py:778-821` | `prices_window` (= bandpass voltages) | Top 10 evenly-spaced channels, first 30s, stacked with vertical offset = 4×std | None |
| **Descriptive stats table** (left) | `dashboard.py:848-960` | `load_summary_stats()` ← `data/<u>/results/summary_stats.parquet` (from `src/analysis.py:compute_descriptive_stats:17`) | Per-ticker: `count`, `mean_daily_return`, `std_daily_return`, `annualized_return` (×252), `annualized_vol` (×√252), `skewness`, `kurtosis`, `min_return`, `max_return` | **PR #34**: Sort-by selectbox + Asc/Desc order; sort on raw numerics |
| **Return histogram** (right) | `dashboard.py:881-...` | `returns[selected_ticker].dropna()` | 80-bin histogram, mean/median vlines | Ticker selectbox |
| **Return anomalies** (bottom, finance only) | `dashboard.py:680-...` | `load_anomalies()` ← `data/<u>/processed/anomalies.csv` (`flag_anomalies(returns, threshold=0.30)` — config: `preprocessing.anomaly_return_threshold`) | Scatter of (date, ticker) where `|log return| > threshold` | None |

**Hyperparameters baked here**:
- Coverage filter threshold: `0.90` (`settings.yaml preprocessing.min_coverage_pct`)
- Annualization factor: `252` (`settings.yaml analysis.annualization_factor`; 1 for EEG)
- Anomaly threshold: `|return| > 0.30` (`settings.yaml`; 1.0 for EEG = disabled)
- **Manual anomaly nulls** (BIST-only): `[CCOLA 2024-08-01, HEKTS 2024-09-09, HEKTS 2021-04-30, AYGAZ 2022-09-01]` — unhandled yfinance bonus issues, masked to NaN BEFORE `flag_anomalies` runs (settings.yaml `preprocessing.manual_anomaly_nulls`)

### 3.2 Correlation (`tab_corr`)

Two sub-sub-tabs declared at `dashboard.py:1201`:

```python
_corr_heatmap_tab, _corr_pit_tab = st.tabs(["Heatmap", "Point-in-Time Snapshot"])
```

**Both sub-tabs are `@st.fragment`-scoped** (post PR #35). Widget changes inside one sub-tab don't re-render the other.

#### 3.2.1 Heatmap (`@st.fragment _render_correlation_heatmap` — added PR #35, `dashboard.py:221`)

| Aspect | Detail |
|---|---|
| **Compute** | `_compute_corr(returns, returns_cache_key, dynamic_min_periods, heat_method)` `dashboard.py:122` → `returns.corr(method=heat_method, min_periods=dynamic_min_periods)`, wrapped in `st.spinner("Computing correlation matrix...")` |
| **`dynamic_min_periods`** | `max(30, int(window_length * 0.6))` `dashboard.py:389` — 60% of the current date-window length |
| **Reordering** | `load_dendrogram_order()` ← `data/<u>/results/dendrogram_order.json` (from `src/clustering.py:get_leaf_order`) — pre-computed Ward-linkage leaf order |
| **Rendering** | `render_matrix_heatmap` (PR #30) — interactive Plotly for N ≤ 160, server-PNG via kaleido for larger |
| **CSV download** | Expander below the heatmap; full-resolution CSV (not capped) |
| **User knobs** | `heat_method` selectbox `[pearson, spearman]`, `use_clustering_order` checkbox (key `mo_corr_reorder`) |
| **Cross-tab interaction** | The Pairs & Dislocations tab (`dashboard.py:1810-1818`) recomputes `corr` via `_compute_corr(..., session_state["heat_method"])` — gets a **cache HIT** because the same args were just used here. No double compute. |

#### 3.2.2 Point-in-Time Snapshot (`@st.fragment _render_pit_correlation`, PR #33 + #35)

| Aspect | Detail |
|---|---|
| **Compute** | `_pit_corr(returns, cache_key, end_date_iso, window, method)` → `compute_window_correlation(returns, pd.Timestamp(end_date), window, method)` in `src/rolling_correlation.py:296`, wrapped in `st.spinner("Computing snapshot correlation...")` |
| **What that does** | Takes the `window` days ending at `end_date`; runs `.corr(method=method, min_periods=window//2)` |
| **Stats strip** | 4 metrics: tickers-in-window, mean/median/std of upper-triangle |
| **Rendering** | `render_matrix_heatmap` (same as Heatmap) — PNG for S&P |
| **User knobs (PR #35)** | `pit_window`: `st.number_input("Window", min=20, max=min(504, len(trading_dates)), value=252, step=10)`; `pit_method`: selectbox `[pearson, spearman]`; `pit_date`: `st.date_input` (replaces `select_slider`) — snaps to nearest preceding trading day on weekend/holiday picks with a caption notice |
| **Fragment scope** | Dragging widgets only re-runs this block (post PR #33). Doesn't affect Heatmap or Pairs tabs. |
| **Cross-fragment state** | Reads `use_clustering_order` from `st.session_state["mo_corr_reorder"]` (set by the Heatmap fragment) and calls `load_dendrogram_order()` directly (PR #35) — no longer relies on outer-scope globals |

### 3.3 Clustering & Network (`tab_cluster`)

| Section | Code | Reads | Algorithm |
|---|---|---|---|
| **Dendrogram** | `dashboard.py:1100-1136` | `load_linkage()` ← `data/<u>/results/{linkage_matrix.npy, linkage_labels.json}` (from `src/clustering.py:compute_linkage`) | `scipy.cluster.hierarchy.linkage(condensed_distance, method=ward)` on `distance = sqrt(2*(1-corr))`. Diagonal forced to 0; NaN filled with 2.0 (max distance). Currently single-coloured. |
| **Cluster Purity strip + per-cluster table + sanity banners** (right, post PR #34) | `dashboard.py:1138-1207` | `load_cluster_assignments()` ← `data/<u>/results/cluster_assignments.csv` | `fcluster(Z, t=n_clusters, criterion=maxclust)`; one-line green/yellow banners check per-universe `sanity_check_groups` (BIST banks, S&P mega-cap tech, EEG motor/occipital/prefrontal triples); per-cluster purity = dominant-sector fraction |
| **MST** (full width, post PR #34) | `dashboard.py:1243-1336` | `load_mst_edges()` ← `mst_edges.csv` + `load_mst_metrics()` ← `mst_node_metrics.csv` (from `src/clustering.py:build_mst` — Kruskal) | Layout: `_mst_layout` cached. **Algorithm switches by size** (`dashboard.py:144`): N ≤ 200 → `nx.kamada_kawai_layout` (cleaner, O(N³)); N > 200 → `nx.spring_layout(iterations=80, seed=42)` (faster). Node colour by sector; node size = `14 + 6×degree`. |
| **Hub table** (inside expander, post PR #34) | `dashboard.py:1337-1356` | `mst_node_metrics.csv` | Sorted by degree DESC; betweenness centrality formatted to 4 d.p. |

**Pipeline hyperparameters** (`settings_<u>.yaml clustering`):
- `linkage_method = ward` (changed 2026-05-17 from `single` — see CLAUDE.md; single linkage chained 45/73 BIST tickers into one mega-cluster, breaking the "banks cluster" sanity check)
- `n_clusters`: 20 (BIST), 25 (S&P), 8 (EEG)
- `criterion = maxclust`
- `distance_threshold = 1.0` (only used if `criterion = distance`)

**Hardcoded in source** (not in config):
- Distance metric: `sqrt(2*(1-corr))` — `src/analysis.py:68`
- MST algorithm: Kruskal (`src/clustering.py:109`)
- NaN→2.0 fill in distances `src/clustering.py:45`

### 3.4 Rolling Analysis (`tab_rolling`)

Three sub-sub-tabs declared at `dashboard.py:1432`:

```python
tab_market, tab_pair, tab_sector = st.tabs([_ra_market_label, "Pair Correlation", f"{_sector_rc} Breakdown"])
```

**Outer widgets are wrapped in `st.form("rolling_params")` with a Recompute button** (PR #34). Changes accumulate locally; only Recompute triggers a script rerun.

Post PR #35:
- `rc_window`: `st.number_input(min=20, max=504, value=252, step=10)` (was selectbox `[60,120,252,504]`)
- `rc_step`: selectbox `[1, 5, 21]`
- `rc_method`: selectbox `[pearson, spearman]`
- `rc_window_type`: selectbox `[rolling, expanding, ewm]`
- `rc_ewm_alpha` (conditional): `st.number_input(0.01, 0.5, value=0.05, step=0.01)` — only shown when last-submitted `window_type == "ewm"` (standard form limitation: in-form value doesn't propagate until submit, so the visibility is controlled by `session_state["rc_wtype"]`)

Event Markers popover lives **outside** the form (column-layout constraint with popovers).

#### 3.4.1 Market / Network Overview sub-sub-tab

| Aspect | Detail |
|---|---|
| **Precomputed fast-path** | If (`rc_window ∈ {60, 120, 252}` AND `rc_step=5` AND `method=pearson` AND not expanding): load `data/<u>/results/rolling_market_stats_w{N}.parquet`. **Instant.** |
| **Slow-path compute** | `_compute_market_stats(returns, cache_key, window, step, method, expanding)` → `src/rolling_correlation.py:compute_rolling_market_stats:33`. **Inner Python loop**: for each end-of-window, slice `returns.iloc[end-window:end]`, filter tickers with ≥ `min_periods` non-NaN, run `.corr()`, extract upper-triangle stats. **Profiled at 12 seconds on S&P off-grid params.** Status message updated PR #35 to flag the 10-15 s expected wait. |
| **Min periods within window** | `max(30, int(window * 0.6))` (60% — `rolling.min_periods_ratio` in settings.yaml) |
| **Charts** | Filled IQR (Q25–Q75) + mean line + median dotted line + event markers (COVID/Ukraine/quakes via `DEFAULT_EVENTS` in `src/rolling_correlation.py:26`) |
| **Min/max envelope** | Toggle `st.toggle("Show min/max envelope")` — default off |

#### 3.4.2 Pair Correlation sub-sub-tab (`@st.fragment _render_rolling_pair`, PR #33)

| Aspect | Detail |
|---|---|
| **Compute** | `_compute_pair(returns, cache_key, a, b, window, method, wtype, ewm_span=None)` → `compute_rolling_pair_correlation` `src/rolling_correlation.py:132` — **fast**, ~1 ms on S&P. EWM span computed via `(2/α) - 1` (PR #35) |
| **Charts** | Pair rolling correlation with positive/negative fills, hline at 0, event markers, normalized price overlay (two stocks rebased to 100) |
| **Fragment scope** | Changing `pair_a`/`pair_b` only re-runs this fragment. The "Open in Pair Analysis" button uses `st.rerun(scope="app")` to escape. |
| **User knobs** | `pair_a`, `pair_b` selectboxes (initialised to first two tickers; collision-resolver fills `pair_b ≠ pair_a`) |

#### 3.4.3 Sector Breakdown sub-sub-tab

| Aspect | Detail |
|---|---|
| **Precomputed fast-path** | If (`window=252` AND `step=5` AND `pearson` AND not expanding): load `data/<u>/results/rolling_sector_stats.parquet` |
| **Slow-path compute** | `_compute_sector(returns, cache_key, sec_map_items, window, step, method)` → `compute_rolling_sector_stats` `src/rolling_correlation.py:200` |
| **Charts** | Intra-sector vs inter-sector mean correlation over time + optional per-sector lines (behind `st.toggle("Show per-sector breakdown")`) |
| **Sector map** | From `cluster_assignments.csv` `sector` column (originally from `universe.csv`) |

### 3.5 Pairs & Dislocations (`tab_pairs`, finance only)

Two sub-sub-tabs declared at `dashboard.py:1671`:

```python
tab_top, tab_bottom = st.tabs(["Most Correlated", "Least Correlated"])
```

| Section | Code | Reads | What you see | User knobs |
|---|---|---|---|---|
| **Top/Bottom pairs** | `dashboard.py:1653-1698` | `load_top_bottom()` ← `data/<u>/results/top_bottom_pairs.csv` (from `src/analysis.py:get_top_bottom_pairs`) | DataFrames of top-N most/least correlated pairs with `ticker_1`, `ticker_2`, `sector_1`, `sector_2`, `correlation`. Pair selectbox + "Open in Pair Analysis" button. | Selectbox + button |
| **Correlation distribution** | `dashboard.py:1700-1721` | Upper triangle of `corr` — recomputed inline via `_compute_corr(returns, returns_cache_key, dynamic_min_periods, session_state["heat_method"])` (cache HIT, post PR #35) | Histogram of pairwise correlations, mean + median vlines | None |
| **Dislocation candidates** | `dashboard.py:1723-1773` | `load_dislocation_candidates()` ← `data/<u>/results/dislocation_candidates.csv` (from `src/pair_dislocation.py:run_pair_dislocation`) | Ranked table: ticker_a, ticker_b, sector_a, sector_b, correlation, β, half_life, current_zscore, n_signals, rank_score. Pair selectbox + "Analyze in Pair Analysis" button. | Selectbox + button |

**Dislocation pipeline parameters** (`settings_<u>.yaml dislocation`, all **C**):
- `zscore_window = 60` (rolling Z window in trading days)
- `entry_zscore = 2.0`, `exit_zscore = 0.5` (state-machine thresholds in `detect_signals`)
- `min_half_life = 5`, `max_half_life = 252` (mean-reversion AR(1) half-life filter)
- `top_n_candidates = 20`
- `lookback_window = 252` (OLS hedge-ratio fit window)
- `min_correlation = 0.5` (pair screening floor)

**Rank score** (**H** — hardcoded weights in `src/pair_dislocation.py:283`):
```
rank_score = 0.30 × norm(correlation)
           + 0.25 × norm(1 / half_life)
           + 0.20 × norm(spread_std)
           + 0.15 × norm(|current_zscore|)
           + 0.10 × norm(n_signals)
```
(`norm` = min-max scaling to [0,1] across candidates.)

### 3.6 EEE Analysis (`tab_eee` → `eee_analysis.render()`)

Five or six sub-sub-tabs declared at `eee_analysis.py:1408-1416`:
```
RMT Denoising · Graphical LASSO · Wavelet Multi-Scale · Transfer Entropy · Information Theory · [Neuromorphic Signals (if has_snn)]
```

#### 3.6.1 RMT Denoising

| Aspect | Detail |
|---|---|
| **Loaders** | `load_eigenvalue_spectrum()`, `load_mst_edges()` (raw), `load_denoised_mst_edges()`, `load_denoised_mst_metrics()`, `load_denoised_corr()`, `load_dendrogram_order()` |
| **Source** | `src/rmt_denoising.py:run_rmt_denoising` |
| **Algorithm** | Eigendecomposition of Pearson correlation (`np.linalg.eigh`); compare each λ to Marchenko-Pastur upper bound `λ_max = (1 + 1/q + 2√(1/q))` with `q = T/N`; noise eigenvalues replaced with the noise-mean (**H** — `method="constant"` in `src/rmt_denoising.py:142`); reconstruct denoised correlation matrix; rebuild MST on `sqrt(2*(1-denoised_corr))` distance |
| **KPIs** | Signal eigenvalues, noise eigenvalues, MP upper bound, variance explained by signal |
| **Charts** | Eigenvalue spectrum (log-y bar chart with MP noise band shaded); Raw vs Denoised MST switch (`st.radio`); full-width denoised correlation heatmap (PNG for S&P) |
| **User knobs** | `mst_choice` radio `[Raw, Denoised, Both (overlay)]` — only "Raw" and "Denoised" actually rendered |
| **What's NOT exposed** | `method=zero` alternative; per-universe MP-band override |

#### 3.6.2 Graphical LASSO

| Aspect | Detail |
|---|---|
| **Loaders** | `load_partial_corr_edges()`, `load_glasso_metadata()` + heatmap loaders `load_partial_corr()` + `load_precision_matrix()` |
| **Source** | `src/partial_correlation.py:run_partial_correlation` |
| **Algorithm** | `sklearn.covariance.GraphicalLassoCV(max_iter=200, cv=5)` — L1-penalized maximum likelihood; **alpha chosen by 5-fold CV** at pipeline time. Partial correlation derived from precision: `pcorr_ij = -P_ij / sqrt(P_ii × P_jj)`. Edge threshold `0.01` on `|partial_corr|` (**H** — `src/partial_correlation.py:88`). |
| **KPIs** | Direct edges, sparsity %, regularization alpha used |
| **Charts** | Partial-correlation network (Kamada-Kawai layout, `_plot_network`), strongest direct dependencies table (top 30), partial correlation heatmap (PNG for S&P), precision-matrix sparsity heatmap (binary, "zero"/"non-zero" colorbar) |
| **User knobs** | None on this tab |
| **What's NOT exposed** | Alpha override (currently CV-only), edge threshold |

#### 3.6.3 Wavelet Multi-Scale (post PR #34: `@st.fragment` + selectbox with physical labels)

| Aspect | Detail |
|---|---|
| **Loaders** | `load_wavelet_metadata()`, `load_wavelet_mst_edges(scale)`, `load_wavelet_mst_metrics(scale)`, `load_wavelet_corr(scale)` |
| **Source** | `src/wavelet_analysis.py:run_wavelet_analysis` |
| **Algorithm** | DWT with `wavelet="db4"` (Daubechies-4, **H** — hardcoded `src/wavelet_analysis.py:129`). Max 7 scales (**H**). For each scale level L: reconstruct detail coefficients only at L (zero out all others), compute Pearson correlation on the per-scale reconstructed series, build MST on `sqrt(2*(1-corr))`. |
| **Scale labels** | `SCALE_LABELS` dict at `src/wavelet_analysis.py:29`: 1→"2-4 day", 2→"4-8 day", ..., 7→"128-256 day" (**H** — these are trading-day approximations; for EEG at 160 Hz they're 12.5 ms → 1.6 s) |
| **Charts** | MST at selected scale (Kamada-Kawai for small N, spring for large), labelled with scale + Σdistance; cross-scale summary table |
| **User knobs** | `wavelet_scale` selectbox (post PR #34) `[Scale 1 — XX day cycles, ..., Scale 7 — ...]` |
| **Fragment scope (PR #34)** | Scale change only re-runs `_render_wavelet_for_scale`, not the rest of EEE tab |
| **What's NOT exposed** | Wavelet family, max_level, threshold on per-scale correlations |

#### 3.6.4 Transfer Entropy

| Aspect | Detail |
|---|---|
| **Loaders** | `load_te_edges()`, `load_te_node_roles()`, `load_te_matrix_raw()`, `load_net_te_matrix()` |
| **Source** | `src/transfer_entropy.py:compute_transfer_entropy_matrix_full` (parallelised via joblib) |
| **Algorithm** | For each directed pair (i, j): equal-frequency discretization into `n_bins=3` bins; compute `TE(i→j) = H(Y_t, Y_lag) − H(Y_t, Y_lag, X_lag) − H(Y_lag) + H(Y_lag, X_lag)` in nats; circular-block-bootstrap surrogate null with `block_length=5` (preserves source autocorrelation; Politis-Romano 1992); BH-FDR multiple-testing correction across N×(N-1) pairs (`multiple_testing=fdr_bh`); FDR-significant edges go in `te_network_edges.csv`. |
| **Fallback** | When the FDR-filtered edge set is empty (typical at 100-shuffle resolution on N>50), `eee_analysis.py:669` falls back to ranking by raw TE magnitude. Top 200 used for the network display. |
| **KPIs** | Information sources (count), information sinks (count), top directed edges |
| **Charts** | Directed network with arrows (arrow width scales with `|net_te|`); top-15 sources + top-15 sinks tables; net-TE-flow heatmap (`net[i,j] = TE(i→j) − TE(j→i)`, PNG for S&P) |
| **User knobs** | `te_top_n` slider `[20, min(200, len(edges))]` |
| **Hyperparameters baked** (`settings_<u>.yaml transfer_entropy`, all **C**): `lag=1`, `n_bins=3`, `significance_shuffles=100` (0 for EEG), `significance_level=0.05`, `surrogate_block_length=5`, `multiple_testing=fdr_bh`, `seed=42` |

#### 3.6.5 Information Theory

| Aspect | Detail |
|---|---|
| **Loaders** | `load_it_summary()`, `load_mi_matrix()`, `load_mi_gaussian_matrix()`, `load_mi_nonlinear_excess_top()`, `load_rolling_info_theory()`, `load_regime_kl()` |
| **Source** | `src/info_theory.py` (Stage 13) |
| **KPIs** | `D_eff` = participation ratio of eigenspectrum = `(Σλ)² / Σλ²`; `ΔH = -½ log det Σ` (Gaussian-joint-structure); mean sign-entropy rate (weak-form-EMH fingerprint); ticker count |
| **Algorithm** | Plug-in MI estimator with `n_bins=4` (different from TE's 3); Gaussian-MI baseline `I_gauss = -½ log(1-ρ²)`; non-linear excess = `MI - I_gauss`; ridge fallback on `log_det_term` for singular Σ (EEG after CAR re-referencing) |
| **Charts** | MI heatmap (off-diagonal, PNG for S&P) + MI-vs-Pearson scatter (capped at 2,000 points); rolling `D_eff(t)` and `ΔH(t)` traces; regime-KL table (Gaussian KL divergence between calm-period and crisis-period covariances) |
| **User knobs** | None |

#### 3.6.6 Neuromorphic Signals (SNN, finance only)

| Aspect | Detail |
|---|---|
| **Loaders** | `load_snn_metrics()`, `load_snn_pair_list()`, `load_snn_signals(pair_id)`, `load_snn_training_history()`, `load_snn_raster_sample()`, `load_snn_membrane_sample()` |
| **Source** | `src/snn_signals.py` (lazy-imports `torch` + `snntorch`; pipeline wraps in try/except so missing extra doesn't crash) |
| **Algorithm** | Leaky integrate-and-fire SNN with surrogate-gradient BPTT on pair-spread classification (BUY/HOLD/SELL); focal loss with `sqrt(inv_freq)` class weights; Adam optimizer; rate coding of spread features into spike trains |
| **Honest framing** | BIST 20 pairs, **10 wins / 10 losses**, mean ΔSharpe **−0.270**, median **−0.002**. S&P 20 pairs, **7 wins / 13 losses**, mean ΔSharpe **−0.838**, median **−0.384**. SNN underperforms the simple `|Z|>2` rule on both markets. |
| **Charts** | Per-pair confusion matrix, raster plot (spike trains), membrane-potential traces, training history (loss + macro-F1 over epochs), signal timeline |
| **User knobs** | Pair selectbox |
| **Hyperparameters** | ALL in `src/snn_signals.py:SNNConfig` dataclass — **NOT** in YAML (FUTURE_WORK F-2): hidden layer sizes, time steps, decay constants, focal γ, learning rate, weight decay, early-stop patience, seed |

---

## 4 · Pair Analysis page (`app/pair_analysis.py`, finance only)

**The entire `render()` function is `@st.fragment` (post PR #35, `pair_analysis.py:86`)** — changing ticker A / B / date range / dislocation params re-runs only `render()`, not the `dashboard.py` script prologue (universe init, page_config, sidebar, top-nav, ~200-500 ms on S&P).

Five sub-tabs declared at `pair_analysis.py:228`:

```python
tab_ov, tab_corr, tab_risk, tab_disloc, tab_net = st.tabs([
    "Overview", "Correlation", "Risk & Volatility",
    "Spread & Dislocation", "Network",
])
```

| Sub-tab | What renders | Compute | User knobs |
|---|---|---|---|
| **Overview** | 4 KPIs (Pearson, Spearman, trading days, distance), 7-col risk-return mini-table for the two tickers, normalized price overlay with divergence shading, daily-return scatter with OLS regression line + β | `both.corr(method="pearson"/"spearman")`, `np.polyfit` for OLS β | None on this sub-tab (tickers chosen at page top) |
| **Correlation** | Rolling pair correlation (positive/negative fill, event markers) | `_pair_corr(returns, cache_key, a, b, win, method, wtype)` (post PR #33 hash-key cache) | Rolling window selectbox `[30, 60, 120, 252, 504]`, step `[1, 5, 21]`, method `[pearson, spearman]`, window_type `[rolling, expanding, ewm]` |
| **Risk & Volatility** | Distribution histograms (both tickers overlaid), rolling annualized volatility (`std × √252`), drawdown analysis (`(P − cummax) / cummax × 100`) | `_rolling_vol(both, cache_key, ta, tb, win)` | Volatility window selectbox `[30, 60, 120, 252, 504]` |
| **Spread & Dislocation** | OLS log-spread `log(Pb) - β·log(Pa) - α`; rolling Z-score; mean-reversion half-life (AR(1) regression); BUY/SELL signal markers from state machine | `_compute_dislocation(adj_close, cache_key, a, b, lookback, zwin, entry, exit)` calling `compute_spread`, `compute_zscore`, `compute_half_life`, `detect_signals` from `src/pair_dislocation.py` | Z-score window `[30, 60, 120]`, OLS lookback `[120, 252, 504]`, entry-Z slider `[1.0–3.0]`, exit-Z slider `[0.0–1.5]` |
| **Network** | Sub-graph of MST centred on (ticker_a, ticker_b) with their 1-hop neighbours | `_subgraph_layout(nodes, edges)` `pair_analysis.py:79` (cached spring layout); reads `load_mst_edges` + `load_cluster_assignments` | None |

**Top-of-page widgets** (`pair_analysis.py:174-181`): ticker A selectbox + ticker B selectbox, both populated from `sorted(full_returns.columns)`. Date range `st.date_input`. Cross-tab `pa_ticker_a` / `pa_ticker_b` session_state synced via `_goto_pair_analysis` flag from the other dashboard tabs' "Open in Pair Analysis" buttons.

**Defence-in-depth**: re-checks `has_pair_trading` flag in case of direct deep-link.

---

## 5 · Hyperparameter index — everything in one place

### 5.1 Config-driven (per-universe YAML, **C**)

| Stage | Param | BIST | S&P | EEG |
|---|---|---|---|---|
| Data | `start_date` / `end_date` | 2020-01-01 → 2026-03-01 | same | placeholder (no dates) |
| Preprocessing | `min_coverage_pct` | 0.90 | 0.90 | 0.0 (disabled) |
| Preprocessing | `anomaly_return_threshold` | 0.30 | 0.30 | 1.0 (disabled) |
| Preprocessing | `manual_anomaly_nulls` | 4 BIST cells | — | — |
| Analysis | `correlation_method` | pearson | pearson | pearson |
| Analysis | `annualization_factor` | 252 | 252 | 1 |
| Analysis | `corr_min_periods` | 200 | 200 | 5000 (~31 s @ 160 Hz) |
| Rolling | `windows` (precomputed) | [60, 120, 252] | [60, 120, 252] | [320, 800] |
| Rolling | `step` | 5 | 5 | 80 (~0.5 s) |
| Rolling | `method` | pearson | pearson | pearson |
| Rolling | `min_periods_ratio` | 0.6 | 0.6 | 0.6 |
| Dislocation | `zscore_window` | 60 | 60 | 320 |
| Dislocation | `entry_zscore` / `exit_zscore` | 2.0 / 0.5 | 2.0 / 0.5 | 2.0 / 0.5 |
| Dislocation | `min/max_half_life` | 5 / 252 | 5 / 252 | 5 / 252 |
| Dislocation | `top_n_candidates` | 20 | 20 | 20 |
| Dislocation | `lookback_window` | 252 | 252 | 800 |
| Dislocation | `min_correlation` | 0.5 | 0.5 | 0.5 |
| Clustering | `linkage_method` | ward | ward | ward |
| Clustering | `n_clusters` | 20 | 25 | 8 |
| Clustering | `criterion` | maxclust | maxclust | maxclust |
| TE | `lag` | 1 | 1 | 5 (~31 ms) |
| TE | `n_bins` | 3 | 3 | 3 |
| TE | `significance_shuffles` | 100 | 100 | 0 (skipped — too costly @ 593k samples) |
| TE | `significance_level` | 0.05 | 0.05 | 0.05 |
| TE | `surrogate_block_length` | 5 | 5 | 5 |
| TE | `multiple_testing` | fdr_bh | fdr_bh | fdr_bh |
| TE | `seed` | 42 | 42 | 42 |

### 5.2 Hardcoded in source (**H** — not configurable without editing code)

| Param | Where | Value |
|---|---|---|
| Distance formula | `src/analysis.py:68` | `sqrt(2*(1-corr))` |
| NaN→max distance fill in clustering | `src/clustering.py:45` | 2.0 |
| RMT noise replacement method | `src/rmt_denoising.py:142` | "constant" (replace with noise mean) |
| GLASSO partial-corr edge threshold | `src/partial_correlation.py:88` | 0.01 |
| GLASSO `max_iter` | `src/partial_correlation.py:25` | 200 |
| GLASSO alpha selection | `src/partial_correlation.py:58` | 5-fold CV (`GraphicalLassoCV(cv=5)`) |
| Wavelet family | `src/wavelet_analysis.py:129` | "db4" (Daubechies-4) |
| Wavelet max levels | `src/wavelet_analysis.py:69` | `min(pywt.dwt_max_level(T), 7)` |
| Wavelet scale labels (day approximations) | `src/wavelet_analysis.py:29 SCALE_LABELS` | 1:"2-4 day" ... 7:"128-256 day" |
| TE discretization | `src/transfer_entropy.py:21 _discretize` | Equal-frequency bins |
| TE minimum observations | `src/transfer_entropy.py:96` | 30 |
| TE bootstrap p-value smoothing | `src/transfer_entropy.py:155` | `+1` smoothing (Laplace) |
| IT MI bin count | `src/info_theory.py:74` | 4 (different from TE's 3) |
| Dislocation rank-score weights | `src/pair_dislocation.py:283` | 0.30 / 0.25 / 0.20 / 0.15 / 0.10 |
| MST layout switchover | `app/dashboard.py:144 _mst_layout` | N > 200 → spring_layout(iter=80, seed=42); else Kamada-Kawai |
| Cross-Market crisis events | `app/cross_market.py:57 _CRISIS_EVENTS` | 3 dates (COVID/Ukraine/quakes) |
| Default events for chart overlays | `src/rolling_correlation.py:26 DEFAULT_EVENTS` | Global macro events (timestamps in source) |
| Normalized-prices spaghetti cutoff | `app/dashboard.py:834 _SPAGHETTI_MAX` | 80 tickers (envelope for larger) |
| Matrix-heatmap PNG cutoff | `app/utils.py:97 N_INTERACTIVE_MATRIX_MAX` | 160 tickers (PR #30) |
| Heatmap axis tickfont scaling | `app/dashboard.py:148-166 _heatmap_*` | Tiered by N |
| SNN entire hyperparameter set | `src/snn_signals.py SNNConfig` | Hidden sizes, time steps, β decay, focal γ, learning rate, weight decay, patience, seed |
| EWM α → span conversion | `app/dashboard.py:434` (post PR #35) | `span = max(2, int((2/α) - 1))` |

### 5.3 User-tunable widgets summary

| Page / Tab | Widget | Effect on compute | Cost of change |
|---|---|---|---|
| Sidebar / Universe | selectbox | Triggers full reload of all `data/<u>/*` artifacts | ~500-800 ms cold |
| Settings popover / Date range | `st.date_input` | Re-slices `returns` / `prices_window` → invalidates `returns_cache_key` → all cached computes miss | Variable; rolling stats can be 12 s on S&P off-grid |
| Top nav | `st.segmented_control` | Branches `_nav` → loads different page module | ~1-2 s first paint per page |
| Data & Stats | Sort by + Order | Reorders `display_df` (pure UI) | Negligible |
| Data & Stats | Histogram ticker | Re-slices `returns[selected_ticker]` | Negligible |
| Correlation Heatmap **(fragment, PR #35)** | method (pearson/spearman), reorder checkbox | Re-runs `_compute_corr` on cache miss | ~265 ms cold on S&P; spinner shows on miss |
| Correlation PIT **(fragment, PR #35)** | `pit_window` number_input (20-504), `pit_method`, `pit_date` (date_input with weekend-snap) | Re-runs `_pit_corr` (fragment-scoped) | ~50–500 ms |
| Clustering & Network | None | — | — |
| Rolling Analysis (form, PR #34) | `rc_window` number_input (20-504, PR #35), step, method, window_type + Recompute | Triggers full rerun on submit | 0 ms (precomputed) to 12 s (off-grid S&P) |
| Rolling Analysis (form, PR #35) | `rc_ewm_alpha` number_input (0.01-0.5, conditional on last-submitted ewm) | Plumbs through `_compute_pair(ewm_span=span(α))` | Pair Correlation only |
| Rolling → Pair **(fragment)** | `pair_a` / `pair_b` | Re-runs `_compute_pair` fragment | ~1 ms |
| Rolling → Sector | per-sector breakdown toggle | Adds per-sector traces | Negligible |
| EEE → RMT | MST view (Raw / Denoised / Both) | Switches between two cached MST edge sets | Negligible |
| EEE → Wavelet **(fragment, PR #34)** | scale selectbox | Switches scale level → re-runs fragment | Per-scale MST is cached |
| EEE → TE | top-N edges slider | Re-builds DiGraph from top-N rows | ~100 ms |
| EEE → SNN | pair selectbox | Loads per-pair SNN signals CSV | Negligible |
| **Pair Analysis (entire page is a fragment, PR #35)** | ticker A / B, date range | Re-runs `render()` only (skip dashboard prologue) | ~200-500 ms saved per change |
| Pair Analysis → Correlation | rolling window/step/method/wtype | Re-runs `_pair_corr` | ~1 ms |
| Pair Analysis → Risk | vol window | Re-runs `_rolling_vol` | ~5 ms |
| Pair Analysis → Spread | zwin / lookback / entry_z / exit_z | Re-runs `_compute_dislocation` | ~100 ms |

---

## 6 · `@st.fragment` inventory (post PR #35)

Five fragments live in the dashboard layer. Each scopes widget reruns to a single block instead of forcing a full script rerun:

| Fragment | File:line | Owns widgets | Why it matters |
|---|---|---|---|
| `_render_correlation_heatmap` | `dashboard.py:221` | `heat_method`, `mo_corr_reorder` | Switching correlation method no longer re-renders other tabs; Pairs & Dislocations tab reads `corr` via cache HIT |
| `_render_pit_correlation` | `dashboard.py:285` | `pit_window`, `pit_method`, `pit_date` | Date scrub doesn't blink other tabs |
| `_render_rolling_pair` | `dashboard.py:405` | `pair_a`, `pair_b` | Ticker change doesn't rerun heatmap / clustering / etc. Cross-page button uses `st.rerun(scope="app")` |
| `_render_wavelet_for_scale` | `eee_analysis.py:524` | `wavelet_scale` | Scale change doesn't blink RMT / Glasso / TE blocks |
| `pair_analysis.render` | `pair_analysis.py:86` | Entire page (ticker A/B, date range, all sub-tab widgets) | Skips dashboard.py prologue (universe init, page_config, sidebar, top-nav) on every interaction |

**Spinner discipline** (PR #35 adds explicit `st.spinner` on heavy paths):
- `_compute_corr` in heatmap fragment → "Computing correlation matrix..."
- `_pit_corr` in PIT fragment → "Computing snapshot correlation..."
- `_rasterize_matrix_png` decorator (`utils.py:97`) → "Rendering matrix as PNG (large-N path)..."
- Off-grid rolling stats → "Computing rolling stats (off-grid params — first run takes 10-15 s on S&P; cached on subsequent reruns)..."

---

## 7 · Inheritance map — who feeds whom

```
log_returns.parquet
  ├──► src/analysis.py
  │      ├── compute_descriptive_stats        → summary_stats.parquet         (Data & Stats table)
  │      ├── compute_correlation_matrix       → pearson_corr.parquet          (Correlation heatmap, Pairs distribution, RMT input, Cluster input, etc.)
  │      ├── compute_distance_matrix          → distance_matrix.parquet       (Cluster input, MST input)
  │      └── get_top_bottom_pairs             → top_bottom_pairs.csv          (Pairs & Dislocations tab)
  │
  ├──► src/clustering.py (consumes distance_matrix + pearson_corr)
  │      ├── compute_linkage                  → linkage_matrix.npy + linkage_labels.json   (Dendrogram)
  │      ├── get_leaf_order                   → dendrogram_order.json         (Heatmap reordering EVERYWHERE)
  │      ├── get_cluster_assignments          → cluster_assignments.csv       (Clustering tab, Cross-Market, sector_map feeding TE/Glasso/Wavelet)
  │      ├── build_mst + mst_to_edge_df       → mst_edges.csv                 (MST + Cross-Market MST)
  │      └── compute_mst_metrics              → mst_node_metrics.csv          (Hub table, node sizing)
  │
  ├──► src/rolling_correlation.py
  │      ├── compute_rolling_market_stats     → rolling_market_stats_w{60,120,252}.parquet  (Rolling Market sub-tab fast path)
  │      └── compute_rolling_sector_stats     → rolling_sector_stats.parquet  (Rolling Sector sub-tab fast path)
  │
  ├──► src/rmt_denoising.py (consumes pearson_corr)
  │      ├── eigenvalue_spectrum.csv          (EEE → RMT charts + Cross-Market eigenvalue spectra)
  │      ├── denoised_corr.parquet            (RMT heatmap)
  │      └── denoised_mst_edges.csv + denoised_mst_node_metrics.csv  (RMT MST)
  │
  ├──► src/partial_correlation.py (consumes log_returns directly)
  │      ├── partial_corr.parquet             (Glasso heatmap)
  │      ├── precision_matrix.parquet         (Glasso precision sparsity heatmap)
  │      ├── partial_corr_edges.csv           (Glasso network + table)
  │      └── glasso_metadata.json             (Glasso KPIs: alpha, sparsity, n_edges)
  │
  ├──► src/pair_dislocation.py (consumes adj_close + pearson_corr)
  │      └── dislocation_candidates.csv       (Pairs & Dislocations + Cross-Market top pair)
  │
  ├──► src/wavelet_analysis.py
  │      ├── wavelet_corr_scale{1..7}.parquet (Wavelet cross-scale summary table; per-scale histogram removed PR #34)
  │      ├── wavelet_mst_edges_scale{1..7}.csv (Wavelet MST)
  │      ├── wavelet_mst_metrics_scale{1..7}.csv  (Wavelet MST node sizing)
  │      └── wavelet_metadata.json            (Scale labels)
  │
  ├──► src/transfer_entropy.py
  │      ├── transfer_entropy_matrix.parquet      (FDR-filtered TE)
  │      ├── transfer_entropy_raw.parquet         (pre-FDR fallback)
  │      ├── transfer_entropy_pvalues.parquet     (BH-FDR input)
  │      ├── transfer_entropy_significance.parquet (BH-FDR mask)
  │      ├── net_transfer_entropy_matrix.parquet  (Net flow heatmap)
  │      ├── te_network_edges.csv                 (TE network + table)
  │      └── te_node_roles.csv                    (Sources / sinks classification)
  │
  ├──► src/info_theory.py (consumes log_returns + pearson_corr)
  │      ├── mi_matrix.parquet                (IT MI heatmap)
  │      ├── mi_gaussian_matrix.parquet       (IT MI baseline)
  │      ├── mi_nonlinear_excess.parquet      (IT non-linear excess matrix)
  │      ├── mi_nonlinear_excess_top.csv      (Top non-linear pairs)
  │      ├── rolling_info_theory.parquet      (D_eff(t), ΔH(t) traces)
  │      ├── it_summary.json                  (IT KPIs)
  │      ├── regime_kl.json                   (KL divergence between calm/crisis covariances)
  │      └── entropy_rate_signs.csv           (Sign-sequence entropy)
  │
  └──► src/snn_signals.py [optional]
         └── snn_metrics.json + snn_pair_list.csv + snn_signals_*.csv + snn_training_history.csv + snn_raster/membrane_sample.parquet
```

**Cross-cutting**:
- `universe.csv` — ticker list with sector labels. Feeds `sector_map` everywhere (consumed in cluster assignments, MST node colouring, TE roles annotation, dislocation candidate display).
- `pipeline_metadata.json` — high-level summary (`market_summary` dict with `avg_pairwise_corr`, `median_pairwise_corr`, etc.) → header KPI strip in Market Overview.
- `fetch_metadata.json` — data-freshness display in Settings popover.

---

## 8 · What's assumed (and what would break if the assumption fails)

| Assumption | Where | Failure mode |
|---|---|---|
| **`adj_close.parquet` and `log_returns.parquet` have the same index** | All cross-tab code uses both via the same date slice | Date misalignment → mismatched `prices_window` vs `returns` |
| **Universe ticker list survives 90% coverage filter** | `compute_coverage` → `filter_by_coverage` | BIST: 102 listed → 73 survive; S&P: 500 → 485 (after dropping GOOG/FOX/NWS dual-class duplicates) |
| **`sector` column is present in `universe.csv` AND `cluster_assignments.csv`** | Sector colouring + sanity checks + sector-aware rolling stats + Glasso edge table + TE node-role annotation | Missing sector → "Unknown" everywhere, sanity banners skip |
| **`distance_matrix` is symmetric with diag=0** | All MST + clustering routines force this via `(M + M.T)/2` and `np.fill_diagonal(M, 0)` | Slight numerical asymmetry would otherwise crash scipy's `squareform` |
| **`pearson_corr.parquet` exists before RMT / clustering / wavelet / dislocation / IT runs** | All those modules read it as input | Pipeline must run Step 2 (Analysis) before downstream steps; `run_pipeline.py` enforces order |
| **`adj_close.parquet` exists before dislocation runs** | `src/pair_dislocation.py:run_pair_dislocation` reads it | Same — Step 1 before Step 7 |
| **TE pipeline parallelisation uses joblib + 12 cores** | S&P runtime 60-90 min vs 9-10 hours single-threaded | Without joblib, S&P TE wouldn't fit in any HF Spaces deploy window |
| **`data/comparison_bist_vs_sp500.csv` exists** | Cross-Market page only | Page hard-errors with "Run both pipelines and then `scripts/sp500_vs_bist.py`" |
| **No look-ahead in features** | CLAUDE.md convention | Only the SNN's mean-reversion label generator uses forward-look (K=20 days, explicitly documented) |
| **Dendrogram leaf order is consistent across universe switches** | `dendrogram_order.json` written once per universe | If clustering re-runs with different `n_clusters`, leaf order changes; existing widget state (tickers) keeps working but heatmap reorder shifts |
| **First column of edges files is named `source`/`target`** | All network rendering iterates `for u, v, d in G.edges(data=True)` | Column rename in pipeline would break the dashboard silently |
| **GLASSO converges within 200 iters at the CV-selected alpha** | `src/partial_correlation.py:25` | Non-convergence → sklearn raises; pipeline step fails |
| **HF Spaces serves *.hf.space directly, not iframed huggingface.co** | iframe → XSRF/CORS broken (PR #31 disabled both) | If user accesses via huggingface.co iframe, the localStorage `getAnonymousId` parse error + SessionInfo failures appear |

---

## 9 · Things that look configurable but aren't — actual blast-radius items

Per CLAUDE.md FUTURE_WORK F-2 (verified by re-reading code):

| What looks tunable | Reality |
|---|---|
| Wavelet family in UI | Hardcoded to `db4`; no user control |
| RMT noise-replacement method | Hardcoded to `"constant"`; "zero" alternative exists but isn't exposed |
| GLASSO α | Always CV-chosen; no user override path |
| GLASSO edge threshold (0.01) | Hardcoded in `extract_partial_corr_edges` |
| Dislocation rank weights | Hardcoded; changing the demo story means editing source |
| MST layout algorithm | Auto-switches at N=200; no user override |
| All SNN training hyperparameters | In `SNNConfig` dataclass, never lifted to YAML |
| Default event markers | In `src/rolling_correlation.py:DEFAULT_EVENTS` — fixed dates, no per-universe override |
| TE significance threshold for "is this edge real" | After BH-FDR, hard 0.05 → fallback to raw-magnitude top-200 |
| `corr_min_periods` in dashboard live compute | Recomputed as `max(30, int(window_length * 0.6))`, doesn't read settings.yaml's `corr_min_periods` |

---

## 10 · The one-paragraph mental model

The pipeline runs **once locally** (or via the HF Spaces build) and freezes everything heavy as parquet/CSV/JSON artifacts under `data/<u>/results/`. The dashboard is a **thin presentation layer** — it caches loaders, applies a date-range slice + sort + colour, and renders. Only five user actions trigger live recompute on top of cached returns: the **full-period correlation matrix** (~265 ms on S&P), the **point-in-time windowed correlation** (~50-500 ms), the **rolling-stats market chart on off-grid params** (~12 s — gated by Recompute in PR #34), **rolling pair correlation** (~1 ms), and the **pair-dislocation spread + Z** (~100 ms). Everything else — clustering, MST topology, RMT denoising, GLASSO sparse partial correlation, wavelet multi-scale MSTs, transfer entropy network, mutual information matrices, SNN signals, dislocation candidate ranking — is precomputed and just rendered. Configuration lives in `config/settings_<u>.yaml` per universe with one master `config/settings.yaml` documenting defaults; per-universe overrides change `n_clusters`, `corr_min_periods`, and rolling-window units (days vs samples). The dashboard layer reads capability flags (`has_pair_trading`, `has_snn`, `has_index_series`, `eligible_for_cross_market`, `domain`) from `app/universe_registry.py` to decide which tabs to render and what labels to use — that's how the same code serves stock markets and EEG channels. As of PR #35, **five `@st.fragment` boundaries** (Correlation Heatmap, PIT, Rolling Pair, Wavelet, Pair Analysis page) scope widget reruns to single blocks instead of forcing full script reruns — that's the structural answer to the "old graphs greyed semi-transparent in new tab" feeling.
