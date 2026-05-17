# PIPELINE_REFERENCE

Per-module reference for everything under `src/`. Each entry: file:line of the
entry point, what it reads, what it writes, the method, hardcoded params, and
known issues.

---

## `src/config.py`

**Entry point:** `load_config(settings_path=None) -> PipelineConfig` (`config.py:133`).

**Reads:** `config/settings.yaml` and `config/universes/<file>.csv`.

**Returns:** `PipelineConfig` containing nested dataclasses for each stage.

**Hardcoded:** `PROJECT_ROOT` is the repo root computed from `__file__`.
`config.universe` is read directly into a DataFrame; required columns are
`ticker, company_name, sector, provider_symbol`. Universe must have unique
tickers; warns (not errors) if count ≠ 100. Currently 102 in the BIST
universe.

**Dataclasses:** `MarketConfig`, `DataConfig`, `PreprocessingConfig`,
`AnalysisConfig`, `ValidationConfig`, `RollingConfig`, `DislocationConfig`,
`TransferEntropyConfig` (added in this session for the seeded RNG).

**Tests:** none.

---

## `src/data_acquisition.py`

**Entry point:** `run_acquisition(config) -> None` (`data_acquisition.py:159`).

**Reads:** `config.provider_symbols`, `config.data.{start_date, end_date,
download_interval}`.

**Writes:**
- `data/raw/prices_raw.parquet` — wide MultiIndex `(field, ticker)` with
  fields `Adj Close` and `Close`.
- `data/raw/xu100.parquet` — XU100 series.
- `data/raw/fetch_metadata.json` — timestamp, source, ticker count, failures.

**Method:** `yf.download` in chunks of 25 with a 1s sleep between chunks.
Symbols are fetched as e.g. `AKBNK.IS`; columns are renamed back to bare
tickers (`AKBNK`).

**Hardcoded:** chunk size = 25, inter-chunk sleep = 1.0s, `auto_adjust=False`.

**Known issues (LOW):** `data/raw/raw_close.parquet` is only saved if the
config flag `data.store_raw_close` is true *and* yfinance returned a `Close`
column — see `src/preprocessing.py` for how it's split out.

**Tests:** none.

---

## `src/data_validation.py`

**Entry point:** `validate_sample(config) -> pd.DataFrame` (`data_validation.py:71`).

**Reads:** `data/raw/prices_raw.parquet`, `config.validation.{enabled, sample_size}`.

**Writes:** `data/processed/validation_report.csv` with one row per sampled
ticker (status PASS/FLAG/NO_YF_DATA/NO_ISY_DATA/INSUFFICIENT_OVERLAP).

**Method:** Pulls daily close prices from `isyatirimhisse` for a deterministic
sample of `sample_size` tickers (seed=42), aligns on common dates, compares
log returns. PASS iff `mean_abs_return_diff < 0.02`.

**Hardcoded:** RNG seed = 42 (`_select_sample_tickers`), threshold 0.02 for
PASS, requires ≥10 overlapping days, `_to_ddmmyyyy` date conversion for
isyatirimhisse v5 (uses `HGDG_KAPANIS` / `HGDG_TARIH`).

**Known issues (MED):** broad `except Exception` in `_fetch_isyatirim_data`
swallows network and parse errors alike; would benefit from narrower handling
plus a retry. Listed in FUTURE_WORK F-6.

**Tests:** none.

---

## `src/preprocessing.py`

**Entry point:** `run_preprocessing(config) -> None` (`preprocessing.py:123`).

**Reads:** `data/raw/prices_raw.parquet`.

**Writes:**
- `data/processed/adj_close.parquet` (filtered).
- `data/processed/raw_close.parquet` (filtered, only if non-empty).
- `data/processed/log_returns.parquet`.
- `data/processed/coverage_report.csv`.
- `data/processed/anomalies.csv`.

**Method:**
1. Compute coverage = `notna().sum() / len(adj_close)`.
2. Drop tickers with coverage `< min_coverage_pct` (preserves NaN pattern; no
   inner join).
3. Log returns: `np.log(P / P.shift(1))`, drop first row.
4. Anomaly flag: stack to long form, filter `abs(return_value) > threshold`
   (this stack-then-filter shape is the post-fix anomaly logic; see
   KNOWN_ISSUES for the prior bug).

**Hardcoded:** none beyond what's in `PreprocessingConfig`.

**Known issues:** `flag_anomalies` corruption fixed in this session (see
KNOWN_ISSUES H-1).

**Tests:** `tests/test_preprocessing.py` — 6 tests covering coverage,
filtering, log returns, anomaly flag.

---

## `src/analysis.py`

**Entry point:** `run_analysis(config) -> None` (`analysis.py:127`).

**Reads:** `data/processed/log_returns.parquet`, `coverage_report.csv`.

**Writes:**
- `data/results/summary_stats.parquet` — per-ticker descriptive stats.
- `data/results/pearson_corr.parquet` — N×N correlation matrix.
- `data/results/distance_matrix.parquet` — N×N correlation distance.
- `data/results/top_bottom_pairs.csv` — top/bottom 10 pairs by correlation.
- `data/results/pipeline_metadata.json` — config snapshot, ticker/day counts,
  market summary stats.

**Method:**
- Descriptive stats: `mean`, `std`, `min`, `max`, `skewness`, `kurtosis`,
  annualised metrics (multiply by `ann_factor=252`).
- Correlation: `returns.corr(method='pearson', min_periods=200)`.
- Distance: `sqrt(2 * (1 - rho))`. Standard econophysics distance.
- Top/bottom pairs: upper triangle of corr, sorted, attach sector labels.
- Market summary: mean / median / std / min / max / count of upper-triangle
  correlations.

**Hardcoded:** ann factor and min-periods come from `AnalysisConfig`; the
`compute_distance_matrix` formula is hardcoded.

**Tests:** `tests/test_analysis.py` — 14 tests.

---

## `src/clustering.py`

**Entry point:** `run_clustering(config) -> None` (`clustering.py:154`).

**Reads:** `data/results/distance_matrix.parquet`, `pearson_corr.parquet`.

**Writes:**
- `linkage_matrix.npy`, `linkage_labels.json` — scipy linkage Z (n-1 × 4).
- `dendrogram_order.json` — leaf order for reordered heatmap.
- `cluster_assignments.csv` — `ticker, cluster_id, sector`.
- `mst_edges.csv`, `mst_node_metrics.csv`.

**Method:**
- Linkage: `scipy.cluster.hierarchy.linkage(condensed, method='single')`.
  NaN distances filled with `2.0` (max possible). Symmetry forced.
- Cluster assignments: `fcluster(Z, t=1.0, criterion='distance')`.
- MST: NetworkX `minimum_spanning_tree(G, algorithm='kruskal')`.
- Metrics: degree, betweenness centrality.

**Hardcoded:** linkage method `single`, `distance_threshold=1.0` for
fcluster. Listed in FUTURE_WORK F-6 (LOW: hoist threshold to YAML).

**Tests:** `tests/test_clustering.py` — 21 tests.

---

## `src/rolling_correlation.py`

**Entry point:** `run_rolling_analysis(config) -> None` (`rolling_correlation.py:326`).

**Reads:** `data/processed/log_returns.parquet`.

**Writes (orphan — see FUTURE_WORK F-1):**
- `rolling_market_stats_w60.parquet`, `_w120.parquet`, `_w252.parquet`.
- `rolling_sector_stats.parquet`.

These outputs are precomputed but the dashboard recomputes rolling stats
on-the-fly through `_compute_market_stats` / `_compute_sector` (cached).

**Library functions used at render time (these *are* live):**
- `compute_rolling_market_stats` (`rolling_correlation.py:35`).
- `compute_rolling_pair_correlation` (`rolling_correlation.py:134`).
- `compute_rolling_sector_stats` (`rolling_correlation.py:202`).
- `compute_window_correlation` (`rolling_correlation.py:298`).

`DEFAULT_EVENTS` (`rolling_correlation.py:28`) — three notable BIST events
(WHO pandemic, Russia-Ukraine, Turkey earthquakes) used as overlay markers.

**Hardcoded:** `min_periods` defaults to `max(30, window * min_periods_ratio)`.

**Tests:** `tests/test_rolling_correlation.py` — 25 tests.

---

## `src/pair_dislocation.py`

**Entry point:** `run_pair_dislocation(config) -> None` (`pair_dislocation.py:299`).

**Reads:** `data/processed/adj_close.parquet`, `data/results/pearson_corr.parquet`.

**Writes:**
- `dislocation_candidates.csv`, `dislocation_candidates.parquet`.

**Method:**
1. Filter pairs with `corr ≥ min_correlation`.
2. For each, compute log-price spread: `log(Pb) - β log(Pa) - α` with OLS hedge ratio.
3. Compute mean-reversion half-life via AR(1): `-ln(2)/ln(1+φ)` if `φ<0`, else `inf`.
4. Drop pairs outside `[min_half_life, max_half_life]`.
5. Compute rolling Z-score (window from `DislocationConfig.zscore_window`).
6. Detect entry/exit signals via the state machine (`detect_signals`).
7. Composite rank score = weighted combo of correlation, 1/half_life,
   spread_std, |current_zscore|, n_signals (weights 0.30/0.25/0.20/0.15/0.10).
8. Top-N candidates returned.

**Hardcoded:** rank-score weights (`pair_dislocation.py:285-291`) — listed in
FUTURE_WORK as worth hoisting if you want to tune the screen.

**Tests:** `tests/test_pair_dislocation.py` — 21 tests.

---

## `src/rmt_denoising.py`

**Entry point:** `run_rmt_denoising(config) -> None` (`rmt_denoising.py:134`).

**Reads:** `data/results/pearson_corr.parquet`, `data/processed/log_returns.parquet`.

**Writes:**
- `eigenvalue_spectrum.csv` — eigenvalue, is_signal, mp_upper, mp_lower, explained_variance_pct.
- `denoised_corr.parquet`.
- `denoised_mst_edges.csv`, `denoised_mst_node_metrics.csv` (the latter is
  orphan — see FUTURE_WORK F-1).

**Method:**
1. Eigendecompose the correlation matrix (`np.linalg.eigh`).
2. Marchenko–Pastur upper bound: `λ_max = (1 + 1/q + 2√(1/q))` with `q = T/N`.
3. Eigenvalues `> λ_max` are signal; below are noise.
4. Replace noise eigenvalues with their average (`method='constant'`),
   reconstruct `Q Λ_denoised Q^T`, force diagonal=1, clip to `[-1,1]`.
5. Build distance matrix and MST from the denoised correlation.

**Hardcoded:** `method='constant'` (alternative is `'zero'`); noise
eigenvalues clipped to non-negative; NaN correlations filled with 0 before
decomposition (`rmt_denoising.py:79`).

**Tests:** none. See FUTURE_WORK F-4.

---

## `src/partial_correlation.py`

**Entry point:** `run_partial_correlation(config) -> None` (`partial_correlation.py:126`).

**Reads:** `data/processed/log_returns.parquet`.

**Writes:**
- `partial_corr.parquet` — derived partial correlation matrix.
- `precision_matrix.parquet` — sparse precision (saved this session — was
  previously discarded).
- `partial_corr_edges.csv` — edges above `threshold=0.01`.
- `glasso_metadata.json` — alpha used, sparsity, n_edges.

**Method:**
- Drop NaN rows (full-row dropna). Warn if clean rows < tickers.
- Fit `GraphicalLassoCV(max_iter=200, cv=5)` (cross-validated alpha).
- Convert precision to partial correlation: `pcorr_ij = -Θ_ij / √(Θ_ii Θ_jj)`.

**Hardcoded:** `cv=5`, `max_iter=200`, `threshold=0.01` for edge extraction.
Listed in FUTURE_WORK F-2.

**Tests:** none.

---

## `src/wavelet_analysis.py`

**Entry point:** `run_wavelet_analysis(config) -> None` (`wavelet_analysis.py:120`).

**Reads:** `data/processed/log_returns.parquet`.

**Writes (per scale, scales 1–7):**
- `wavelet_corr_scale{ℓ}.parquet`.
- `wavelet_mst_edges_scale{ℓ}.csv`.
- `wavelet_mst_metrics_scale{ℓ}.csv` — orphan, see FUTURE_WORK F-1.
- `wavelet_metadata.json` — wavelet family, n_scales, scale labels.

**Method:**
1. Per ticker, `pywt.wavedec(series, 'db4', level=ℓ)` then reconstruct only
   the detail-band at level ℓ via `pywt.waverec` with all other coefficient
   arrays zeroed.
2. Compute correlation, distance, MST at each scale using the same code
   paths as the main correlation analysis (`compute_correlation_matrix`
   + `build_mst`).

**Scale → trading-day labels** (`wavelet_analysis.py:31`):
1=2-4d, 2=4-8d, 3=8-16d, 4=16-32d, 5=32-64d, 6=64-128d, 7=128-256d.

**Hardcoded:** wavelet `db4`, `max_level=min(pywt.dwt_max_level(T,'db4'),7)`,
NaN replaced with 0 before DWT (`wavelet_analysis.py:83`). Listed in
FUTURE_WORK F-2.

**Tests:** none.

---

## `src/transfer_entropy.py`

**Entry point:** `run_transfer_entropy(config) -> None` (`transfer_entropy.py:262`).

**Reads:** `data/processed/log_returns.parquet`, `config.universe`.

**Writes:**
- `transfer_entropy_matrix.parquet` — N×N asymmetric `te[i,j] = TE(i→j)`.
- `net_transfer_entropy_matrix.parquet` — `net[i,j] = te[i,j] - te[j,i]`.
- `te_network_edges.csv` — pruned edge list.
- `te_node_roles.csv` — per-ticker `te_out, te_in, net_te_flow, role`
  (`source` if net>0 else `sink`).

**Method:**
- Discretize series into `n_bins=3` equal-frequency bins.
- For each pair `(i,j)`: compute `TE(i→j)` via the four-term entropy
  decomposition.
- Significance: shuffle source `significance_shuffles=100` times, p-value =
  fraction of nulls ≥ observed TE. Insignificant entries are zeroed.
- The shuffle RNG is seeded via `config.transfer_entropy.seed` (default 42).

**Hardcoded:** discretization `n_bins=3`, `significance_level=0.05`. Now
plumbed through `TransferEntropyConfig` in `config/settings.yaml`.

**Complexity:** `O(N² · S · T)` where `N` is tickers, `S` shuffles, `T` days.
Currently the slowest stage at ~5–10 minutes for the BIST universe.

**Tests:** none. See FUTURE_WORK F-4.

---

## `src/snn_signals.py`

**Entry point:** `run_snn_signals(config, retrain=False, snn_cfg=None) -> dict` (`snn_signals.py:891`).

**Reads:**
- `data/processed/adj_close.parquet`
- `data/processed/log_returns.parquet`
- `data/results/dislocation_candidates.csv` (top-20 pairs)

**Writes:**
- `data/results/snn_metrics.json` — per-pair + aggregate metrics, full
  `SNNConfig` dump, `sample_pair`, `n_inputs`, `n_pairs`.
- `data/results/snn_pair_list.csv` — `ticker_a, ticker_b, pair_id` for the
  20 trained pairs.
- `data/results/snn_training_history.csv` — `epoch, train_loss, val_loss,
  val_acc, val_macro_f1, pair` (one row per epoch; ~11 rows with early
  stopping at patience=5).
- `data/results/snn_signals/{pair_id}.parquet` (×20) — daily per-pair
  signal: `date, zscore, prob_hold, prob_buy, prob_sell, signal,
  classical_signal`.
- `data/results/snn_model_weights/universal.pt` — trained PyTorch
  state-dict (single universal model; the per-pair `.pt` files from an
  earlier code path are deliberately not persisted).
- `data/results/snn_spike_raster_sample.parquet` — output-neuron spike
  raster for one sample window of the sample pair (`BRYAT_BRSAN`).
- `data/results/snn_membrane_sample.parquet` — output-layer membrane
  V(t) trace for the same sample window.

**Method:**
1. **Feature construction** (`build_input_features`, 11 raw features per
   pair per day): rolling-60 spread Z-score, its first difference, lags at
   5 and 20 days, rolling correlation of returns, log returns of both
   tickers, 20-day rolling spread vol, cross-sectional market dispersion,
   market breadth, and inverse half-life clipped to `[0, 0.3]`.
2. **Spike encoding** (`encode_features_to_spikes`, 11 → 45 channels):
   delta-modulation (Σ-Δ-style) on `zscore` and `dzscore` (2 channels each);
   Gaussian population coding on 8 slow features (5 fields each);
   saturating-ramp single channel on `inv_half_life`. Append 20-dim
   one-hot pair embedding → 65 total input channels.
3. **Labels** (`generate_mean_reversion_labels`, magnitude-aware K-day
   forward oracle): looking K=20 days forward, if `|Z_t| ≥ entry_z=1.2`
   and `Z` reverts by at least `min_reversion=0.8` Z-units → BUY (for
   `Z<0`) or SELL (for `Z>0`); else HOLD. Inference is strictly causal;
   only the supervised target uses the forward look.
4. **Architecture** (`build_lif_classifier`): `Linear(65 → 96)` →
   `snn.RLeaky(96, all_to_all=True)` (recurrent LIF, β=0.92, V_th=0.5,
   surrogate_grad=fast_sigmoid(slope=25)) → `Linear(96 → 3)` →
   `snn.Leaky(reset_mechanism="none")` (membrane-potential readout). Each
   sample unrolled across `window_size × n_timesteps = 5 × 20 = 100` SNN
   ticks.
5. **Training** (`train_snn`): Adam(lr=3e-3, wd=1e-4), focal loss
   (γ=2.0) with `sqrt(inv_freq)` class weights, 25 max epochs with
   early-stop patience 5. Single universal model trained on the pooled
   train splits of all 20 pairs (val/test stay time-ordered per pair).
6. **Per-pair inference + backtest** (`_evaluate_pair`): predict BUY /
   SELL / HOLD on the held-out test split; paper-trade against the
   classical `|Z|>2` rule (`_classical_signals_per_day`); report macro-F1
   per pair plus annualised Sharpe with 20-day non-overlapping holds.

**Hardcoded — `SNNConfig` dataclass (`snn_signals.py:80`):**
n_hidden=96, beta=0.92, v_threshold=0.5, n_timesteps=20, window_size=5,
use_universal_model=True, use_recurrent_hidden=True, readout="membrane",
input_scaling=2.0, class_weight_mode="sqrt_inv_freq", learning_rate=3e-3,
weight_decay=1e-4, n_epochs=25, batch_size=128, early_stop_patience=5,
seed=42, use_focal_loss=True, focal_gamma=2.0, delta_threshold=0.25,
n_population_fields=5, label_horizon=20, label_entry_z=1.2,
label_min_reversion=0.8, label_exit_z=0.5, train_ratio=0.7,
top_n_pairs=20, rolling_window=60, retrain=False.

None of these are exposed through `config/settings.yaml`; future YAML hoist
listed in FUTURE_WORK F-2.

**Dependencies:** `torch`, `snntorch` — both lazily imported via
`_require_torch()`. Not installed by default. Install with
`uv sync --extra snn`. Without them, the module imports clean but
`run_snn_signals` raises `ImportError`; `run_pipeline.py` wraps the call
in `try/except ImportError` so the rest of the pipeline still completes.

**Complexity:** training is `O(epochs × n_samples × window_size ×
n_timesteps × n_hidden²)` due to BPTT through the recurrent layer.
On a laptop CPU: ~12 min full retrain; ~30 s re-inference using the
cached `universal.pt`.

**Known caveats** (see also `docs/SNN_Report.md` §§ 11.3, 15):
- Δ-Sharpe of −1.11 on average — the SNN underperforms the simple
  `|Z|>2` rule on 15 of 20 pairs. Reported as a documented
  exploratory negative result with positive classification (macro-F1
  0.668 above majority-class baseline 0.27).
- Output layer uses continuous membrane readout, not pure spike-count
  readout — necessary for training stability but means the network is
  a hybrid spike-rate model, not strictly event-driven.
- Sharpe annualisation uses overlapping 20-day holds and is internally
  fair vs. the classical baseline but inflated in absolute terms.
- No transaction costs in the backtest.

**Tests:** `tests/test_snn_signals.py` — 12 tests (8 torch-free always
run; 4 torch-dependent skip when `torch` is not installed).

---

## Function reuse map

The most reused functions (handy when adding a new method):

| Function | File:line | Used by |
|---|---|---|
| `compute_correlation_matrix` | `analysis.py:46` | analysis, wavelet |
| `compute_distance_matrix` | `analysis.py:70` | analysis, rmt, wavelet |
| `build_mst` | `clustering.py:92` | clustering, rmt, wavelet |
| `mst_to_edge_df` | `clustering.py:138` | clustering, rmt, wavelet |
| `compute_mst_metrics` | `clustering.py:120` | clustering, rmt, wavelet |
| `compute_log_returns` | `preprocessing.py:92` | preprocessing only |
| `compute_rolling_pair_correlation` | `rolling_correlation.py:134` | dashboard, pair_analysis |
| `compute_window_correlation` | `rolling_correlation.py:298` | dashboard PIT snapshot |
| `compute_spread / compute_zscore / compute_half_life / detect_signals` | `pair_dislocation.py:48-205` | pair_dislocation pipeline + pair_analysis page |
