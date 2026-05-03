# DATA_ARTIFACTS

Every file written by `run_pipeline.py`, with schema, producer, and consumer.
"Consumer" lists the dashboard sub-tab(s) that read the file — `[ORPHAN]`
means the pipeline writes it but no app code currently reads it (see
[`FUTURE_WORK.md`](FUTURE_WORK.md) F-1).

Schemas describe a single artifact instance after a successful pipeline run.
Shapes use `T = trading days, N = surviving tickers (~73 for BIST)`.

---

## `data/raw/`

### `prices_raw.parquet`

| | |
|---|---|
| Columns | MultiIndex `(field, ticker)` where `field ∈ {Adj Close, Close}` |
| Shape | `(T_raw, 2 * N_raw)` — `T_raw ≈ 1500` for the default date range, `N_raw = 102` |
| Index | DatetimeIndex (business days) |
| Producer | `src/data_acquisition.py:run_acquisition` |
| Consumer | `src/preprocessing.py`, `src/data_validation.py` |

### `xu100.parquet`

| | |
|---|---|
| Columns | yfinance default (`Open, High, Low, Close, Adj Close, Volume`) |
| Shape | `(T_raw, 6)` |
| Index | DatetimeIndex |
| Producer | `src/data_acquisition.py:fetch_index` |
| Consumer | `app/utils.py:load_xu100` (Market Overview tab 1, Pair Analysis Overview) |

### `fetch_metadata.json`

```json
{
  "timestamp": "2026-05-03T12:30:00",
  "source": "yfinance",
  "ticker_count": 102,
  "failures": ["..."]
}
```

| Producer | `src/data_acquisition.py` (writes JSON) |
|---|---|
| Consumer | `app/utils.py:load_fetch_metadata` (settings popover) |

---

## `data/processed/`

### `adj_close.parquet`

| | |
|---|---|
| Columns | tickers (one level, surviving after coverage filter) |
| Shape | `(T_raw, N)` — N ≈ 73 |
| Index | DatetimeIndex |
| Producer | `src/preprocessing.py:run_preprocessing` |
| Consumer | dashboard tab 1 (prices), pair_analysis (Overview, Risk, Spread) |

### `raw_close.parquet`

Same shape as `adj_close.parquet`, only present if the source had a `Close`
field and `data.store_raw_close=true`. Currently **no consumer** in the app.

### `log_returns.parquet`

| | |
|---|---|
| Columns | tickers (same as adj_close) |
| Shape | `(T_raw - 1, N)` |
| Index | DatetimeIndex |
| Producer | `src/preprocessing.py:compute_log_returns` |
| Consumer | the workhorse input: every `src/*` analysis module reads this; dashboard tabs 2–6 and all of pair_analysis. |

### `coverage_report.csv`

| Column | dtype | Description |
|---|---|---|
| `ticker` | string | |
| `total_days` | int | |
| `available_days` | int | |
| `coverage_pct` | float (0–1) | |

| Producer | `src/preprocessing.py:compute_coverage` |
|---|---|
| Consumer | `app/utils.py:load_coverage` (dashboard tab 1) |

### `anomalies.csv`

| Column | dtype | Description |
|---|---|---|
| `date` | datetime | |
| `ticker` | string | |
| `return_value` | float | log return for that day |

Sorted by `|return_value|` descending. **Tens of rows** post-fix (was
112,640 buggy rows pre-fix; see KNOWN_ISSUES H-1).

| Producer | `src/preprocessing.py:flag_anomalies` |
|---|---|
| Consumer | **no current UI consumer.** [ORPHAN — see FUTURE_WORK F-1] |

### `validation_report.csv`

| Column | dtype |
|---|---|
| `ticker` | string |
| `mean_abs_return_diff` | float |
| `max_abs_return_diff` | float |
| `n_days_compared` | int |
| `status` | string (`PASS`, `FLAG`, `NO_YF_DATA`, `NO_ISY_DATA`, `INSUFFICIENT_OVERLAP`) |

| Producer | `src/data_validation.py:validate_sample` |
|---|---|
| Consumer | dashboard settings popover (Data Freshness) — reads file directly, not via `utils.py`. |

---

## `data/results/`

### `summary_stats.parquet`

| Column | Description |
|---|---|
| `ticker` | |
| `count` | non-NaN observations |
| `mean_daily_return`, `std_daily_return` | |
| `annualized_return`, `annualized_vol` | × 252, × √252 |
| `min_return`, `max_return`, `skewness`, `kurtosis` | |

Sorted by `annualized_vol` descending.

| Producer | `src/analysis.py:compute_descriptive_stats` |
|---|---|
| Consumer | dashboard tab 1 (table) |

### `pearson_corr.parquet`

| | |
|---|---|
| Shape | `(N, N)`, symmetric |
| Diagonal | 1.0 |
| Producer | `src/analysis.py:compute_correlation_matrix` |
| Consumer | dashboard tabs 2/3/5; `pair_dislocation`; `rmt_denoising`; pair_analysis. |

### `distance_matrix.parquet`

| | |
|---|---|
| Shape | `(N, N)`, symmetric |
| Formula | `d = sqrt(2 * (1 - rho))` |
| Producer | `src/analysis.py:compute_distance_matrix` |
| Consumer | `src/clustering.py` (linkage + MST). **No UI consumer.** [ORPHAN] |

### `top_bottom_pairs.csv`

| Column | Description |
|---|---|
| `ticker_1`, `ticker_2` | |
| `correlation` | |
| `sector_1`, `sector_2` | |
| `rank_type` | `top` or `bottom` |

20 rows total (top 10 + bottom 10).

| Producer | `src/analysis.py:get_top_bottom_pairs` |
|---|---|
| Consumer | dashboard tab 5 (two tables). |

### `pipeline_metadata.json`

```json
{
  "run_timestamp": "...",
  "config": { "start_date": "...", "min_coverage_pct": 0.9, ... },
  "universe_count": 102,
  "tickers_after_filter": 73,
  "trading_days": 1497,
  "market_summary": { "avg_pairwise_corr": ..., ... }
}
```

| Producer | `src/analysis.py:run_analysis` |
|---|---|
| Consumer | `app/utils.py:load_metadata` (dashboard headline + tab 1 Market Summary). |

### `linkage_matrix.npy`, `linkage_labels.json`

Scipy linkage `Z` (shape `(N-1, 4)`) plus the matching ticker order.

| Producer | `src/clustering.py:compute_linkage` |
|---|---|
| Consumer | `app/utils.py:load_linkage` (dashboard tab 3 dendrogram). |

### `dendrogram_order.json`

```json
["AKBNK", "GARAN", "ISCTR", ...]
```

Optimal leaf order from `scipy.cluster.hierarchy.leaves_list`.

| Producer | `src/clustering.py:get_leaf_order` |
|---|---|
| Consumer | dashboard tab 2 heatmap reordering. |

### `cluster_assignments.csv`

| Column |
|---|
| `ticker` |
| `cluster_id` |
| `sector` |

| Producer | `src/clustering.py:get_cluster_assignments` |
|---|---|
| Consumer | `eee_analysis.render` builds sector_map from this file. |

### `mst_edges.csv`

| Column |
|---|
| `source`, `target` |
| `distance` |
| `correlation` |

`N-1` rows.

| Producer | `src/clustering.py:mst_to_edge_df` |
|---|---|
| Consumer | dashboard tab 3, pair_analysis Network sub-tab. |

### `mst_node_metrics.csv`

| Column |
|---|
| `ticker`, `degree`, `betweenness_centrality`, `sector` |

| Producer | `src/clustering.py:compute_mst_metrics` |
|---|---|
| Consumer | dashboard tab 3 (node sizing). |

### `rolling_market_stats_w60.parquet`, `_w120.parquet`, `_w252.parquet`

Indexed by date. Columns: `avg_corr, median_corr, std_corr, min_corr,
max_corr, q25_corr, q75_corr, n_valid_pairs, n_tickers_in_window`.

| Producer | `src/rolling_correlation.py:compute_rolling_market_stats` |
|---|---|
| Consumer | **none.** Dashboard recomputes rolling stats on the fly via `_compute_market_stats`. [ORPHAN — see FUTURE_WORK F-1] |

### `rolling_sector_stats.parquet`

Indexed by date. Columns: `intra_sector_avg, inter_sector_avg, intra_<Sector>` (one column per sector).

| Producer | `src/rolling_correlation.py:compute_rolling_sector_stats` |
|---|---|
| Consumer | **none.** [ORPHAN — see FUTURE_WORK F-1] |

### `dislocation_candidates.csv`, `dislocation_candidates.parquet`

Same content, two formats. CSV is human-friendly; parquet preserves dtypes.

| Column | Description |
|---|---|
| `ticker_a`, `ticker_b` | |
| `sector_a`, `sector_b` | |
| `correlation` | |
| `beta` | hedge ratio |
| `half_life` | mean-reversion days |
| `spread_std` | |
| `n_signals` | |
| `current_zscore` | |
| `rank_score` | composite (0–1) |

Up to `top_n_candidates` rows (default 20).

| Producer | `src/pair_dislocation.py:run_pair_dislocation` |
|---|---|
| Consumer | dashboard tab 5 (table); pair_analysis Spread sub-tab; `reservoir_computing` (top-3 for pair task). |

### `eigenvalue_spectrum.csv`

| Column |
|---|
| `eigenvalue` |
| `is_signal` (bool) |
| `mp_upper`, `mp_lower` |
| `explained_variance_pct` |

Sorted descending; `N` rows.

| Producer | `src/rmt_denoising.py:denoise_correlation` |
|---|---|
| Consumer | EEE tab RMT (spectrum chart). |

### `denoised_corr.parquet`

Symmetric `(N, N)` matrix; diagonal 1, clipped to `[-1, 1]`.

| Producer | `src/rmt_denoising.py:denoise_correlation` |
|---|---|
| Consumer | **`load_denoised_corr` exists but no UI consumer.** [ORPHAN — see FUTURE_WORK F-1] |

### `denoised_mst_edges.csv`

Same schema as `mst_edges.csv`.

| Producer | `src/rmt_denoising.py:run_rmt_denoising` |
|---|---|
| Consumer | EEE tab RMT (denoised MST chart). |

### `denoised_mst_node_metrics.csv`

Same schema as `mst_node_metrics.csv`.

| Producer | `src/rmt_denoising.py:run_rmt_denoising` |
|---|---|
| Consumer | **none.** [ORPHAN — see FUTURE_WORK F-1] |

### `partial_corr.parquet`

`(N, N)` partial correlation matrix.

| Producer | `src/partial_correlation.py:run_partial_correlation` |
|---|---|
| Consumer | `load_partial_corr` exists; **no UI consumer**. [ORPHAN] |

### `precision_matrix.parquet`

`(N, N)` sparse precision matrix Θ. Added this session — was previously
discarded.

| Producer | `src/partial_correlation.py:run_partial_correlation` |
|---|---|
| Consumer | `load_precision_matrix` exists; **no UI consumer yet**. [ORPHAN — see FUTURE_WORK F-1, recommended for Glasso heatmap.] |

### `partial_corr_edges.csv`

| Column |
|---|
| `source`, `target` |
| `partial_correlation` |
| `abs_partial_corr` |

Edges with `|pcorr| > 0.01`.

| Producer | `src/partial_correlation.py:extract_partial_corr_edges` |
|---|---|
| Consumer | EEE tab Glasso (network chart). |

### `glasso_metadata.json`

```json
{
  "alpha": 0.0125,
  "n_edges": 215,
  "n_tickers": 73,
  "sparsity_pct": 91.8
}
```

| Producer | `src/partial_correlation.py:run_partial_correlation` |
|---|---|
| Consumer | EEE tab Glasso (header stats). |

### `wavelet_corr_scale{1..7}.parquet`

`(N, N)` correlation matrix at wavelet scale ℓ.

| Producer | `src/wavelet_analysis.py:run_wavelet_analysis` |
|---|---|
| Consumer | EEE tab Wavelet (correlation distribution histogram). |

### `wavelet_mst_edges_scale{1..7}.csv`

Same schema as `mst_edges.csv`. One file per scale.

| Producer | `src/wavelet_analysis.py:run_wavelet_analysis` |
|---|---|
| Consumer | EEE tab Wavelet (per-scale MST chart). |

### `wavelet_mst_metrics_scale{1..7}.csv`

Same schema as `mst_node_metrics.csv`. One file per scale.

| Producer | `src/wavelet_analysis.py:run_wavelet_analysis` |
|---|---|
| Consumer | **none** (loaders only have edges, corr, metadata). [ORPHAN × 7 — see FUTURE_WORK F-1] |

### `wavelet_metadata.json`

```json
{
  "wavelet": "db4",
  "n_scales": 7,
  "scales": { "1": "2-4 day", "2": "4-8 day", ... }
}
```

| Producer | `src/wavelet_analysis.py:run_wavelet_analysis` |
|---|---|
| Consumer | EEE tab Wavelet (label dropdown). |

### `transfer_entropy_matrix.parquet`

`(N, N)` asymmetric matrix; `te[i,j] = TE(i→j)` after significance filtering.

| Producer | `src/transfer_entropy.py:compute_transfer_entropy_matrix` |
|---|---|
| Consumer | EEE tab TE (heatmap). |

### `net_transfer_entropy_matrix.parquet`

`(N, N)` antisymmetric: `net[i,j] = TE(i→j) - TE(j→i)`.

| Producer | `src/transfer_entropy.py:compute_transfer_entropy_matrix` |
|---|---|
| Consumer | **none directly** (the network chart uses edges; no `load_net_te_matrix`). [ORPHAN — see FUTURE_WORK F-1] |

### `te_network_edges.csv`

| Column |
|---|
| `source`, `target` |
| `te_forward`, `te_backward` |
| `net_te` |
| `dominant_direction` |

| Producer | `src/transfer_entropy.py:extract_te_edges` |
|---|---|
| Consumer | EEE tab TE (directed network chart). |

### `te_node_roles.csv`

| Column |
|---|
| `ticker`, `sector` |
| `te_out`, `te_in`, `net_te_flow` |
| `role` (`source` or `sink`) |

| Producer | `src/transfer_entropy.py:compute_node_roles` |
|---|---|
| Consumer | EEE tab TE (role table / coloured nodes). |

### `rc_dispersion_predictions.parquet`

| Column |
|---|
| `date`, `actual_dispersion`, `predicted_dispersion` |

| Producer | `src/reservoir_computing.py:run_reservoir_computing` |
|---|---|
| Consumer | **none.** [ORPHAN — see FUTURE_WORK F-1, recommended for new "Forecasting" sub-tab.] |

### `rc_feature_importance.csv`

| Column |
|---|
| `feature`, `weight_magnitude` |

| Producer | `src/reservoir_computing.py:run_reservoir_computing` |
|---|---|
| Consumer | **none.** [ORPHAN — see FUTURE_WORK F-1] |

### `rc_metrics.json`

```json
{
  "dispersion_prediction": { "rmse": ..., "r2": ..., "direction_of_change_accuracy": ... },
  "dispersion_fold_metrics": [ ... ],
  "baselines": { "persistence": {...}, "mean": {...} },
  "pair_spread_prediction": { "TICKERA-TICKERB": {...} },
  "esn_config": { "reservoir_size": 300, "spectral_radius": 0.9, ... },
  "feature_columns": [ ... ],
  "n_features": ..., "n_samples": ..., "train_size": ..., "test_size": ...
}
```

| Producer | `src/reservoir_computing.py:run_reservoir_computing` |
|---|---|
| Consumer | **none.** [ORPHAN — see FUTURE_WORK F-1] |

---

## Orphan summary (18 files)

These are written by the pipeline but never read by `app/`:

1. `data/processed/anomalies.csv`
2. `data/results/distance_matrix.parquet`
3. `data/results/rolling_market_stats_w60.parquet`
4. `data/results/rolling_market_stats_w120.parquet`
5. `data/results/rolling_market_stats_w252.parquet`
6. `data/results/rolling_sector_stats.parquet`
7. `data/results/denoised_corr.parquet` *(loader exists, no chart)*
8. `data/results/denoised_mst_node_metrics.csv`
9. `data/results/partial_corr.parquet` *(loader exists, no chart)*
10. `data/results/precision_matrix.parquet` *(new this session — loader exists, no chart)*
11. `data/results/net_transfer_entropy_matrix.parquet`
12. `data/results/wavelet_mst_metrics_scale1.csv` … `_scale7.csv` *(7 files)*
13. `data/results/rc_dispersion_predictions.parquet`
14. `data/results/rc_feature_importance.csv`
15. `data/results/rc_metrics.json`

Wiring proposals are in [`FUTURE_WORK.md`](FUTURE_WORK.md) F-1.
