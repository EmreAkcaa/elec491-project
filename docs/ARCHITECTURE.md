# StoNeCoAl — Architecture

## Layered view

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Configuration                                                               │
│   config/settings.yaml ─┬─► src/config.py:load_config ─► PipelineConfig     │
│   config/universes/*.csv┘                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Pipeline (orchestrated by run_pipeline.py)                                  │
│   src/data_acquisition.py ─► src/data_validation.py ─► src/preprocessing.py │
│         │                                                       │           │
│         └─► data/raw/*.parquet                                  ▼           │
│                                                       data/processed/*      │
│                                                                 │           │
│   src/analysis.py ◄──────────────────────────────────────────── ┘           │
│         │                                                                   │
│         ├─► src/clustering.py     ─► MST + dendrogram + clusters            │
│         ├─► src/rolling_correlation.py ─► windowed market/sector stats      │
│         ├─► src/pair_dislocation.py    ─► dislocation_candidates            │
│         ├─► src/rmt_denoising.py       ─┐                                   │
│         ├─► src/partial_correlation.py ─┤  EEE methods (informal grouping)  │
│         ├─► src/wavelet_analysis.py    ─┤                                   │
│         ├─► src/transfer_entropy.py    ─┘                                   │
│         └─► src/snn_signals.py    ─► spike-coded LIF classifier (optional   │
│                                      `[snn]` extra: torch + snntorch)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Artifacts (the API between pipeline and app)                                │
│   data/processed/  adj_close, log_returns, raw_close, anomalies, coverage   │
│   data/results/    pearson_corr, distance_matrix, mst_*, dendrogram_*,      │
│                    cluster_assignments, rolling_*, dislocation_*,           │
│                    eigenvalue_spectrum, denoised_*, partial_corr,           │
│                    precision_matrix, wavelet_*, transfer_entropy_*,         │
│                    snn_metrics.json, snn_pair_list.csv, snn_training_*,     │
│                    snn_signals/*.parquet, snn_model_weights/universal.pt,   │
│                    snn_spike_raster_sample, snn_membrane_sample,            │
│                    summary_stats, top_bottom_pairs, *_metadata.json         │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Application (Streamlit, no compute except light rolling/MST layout)         │
│   app/utils.py            cached loaders, theming, render_chart, warnings   │
│   app/dashboard.py        Market Overview (6 tabs) + nav to Pair Analysis   │
│   app/pair_analysis.py    Pair Analysis page (5 tabs)                       │
│   app/eee_analysis.py     EEE Analysis sub-tab (5 sub-tabs: RMT/Glasso/     │
│                           Wavelet/TE/Neuromorphic Signals)                  │
│   app/chart_themes.py     palette, sidebar theme switcher                   │
│   app/chart_export.py     plotly→PNG hook used by render_chart              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 12-stage pipeline flow

`run_pipeline.py` (~50 lines) calls these in order:

| # | Stage | Module | Reads | Writes |
|---|---|---|---|---|
| 1 | acquisition | `src/data_acquisition.py:run_acquisition` | universe csv | `data/raw/prices_raw.parquet`, `xu100.parquet`, `fetch_metadata.json` |
| 2 | validation | `src/data_validation.py:validate_sample` | raw prices | `data/processed/validation_report.csv` |
| 3 | preprocessing | `src/preprocessing.py:run_preprocessing` | raw prices | `adj_close`, `raw_close`, `log_returns`, `coverage_report`, `anomalies` |
| 4 | analysis | `src/analysis.py:run_analysis` | log returns | `summary_stats`, `pearson_corr`, `distance_matrix`, `top_bottom_pairs`, `pipeline_metadata.json` |
| 5 | clustering | `src/clustering.py:run_clustering` | dist + corr | `linkage_matrix.npy`, `linkage_labels.json`, `dendrogram_order.json`, `cluster_assignments`, `mst_edges`, `mst_node_metrics` |
| 6 | rolling | `src/rolling_correlation.py:run_rolling_analysis` | log returns | `rolling_market_stats_w{60,120,252}`, `rolling_sector_stats` |
| 7 | pair dislocation | `src/pair_dislocation.py:run_pair_dislocation` | adj close + corr | `dislocation_candidates.{csv,parquet}` |
| 8 | RMT | `src/rmt_denoising.py:run_rmt_denoising` | corr | `eigenvalue_spectrum`, `denoised_corr`, `denoised_mst_*` |
| 9 | Glasso | `src/partial_correlation.py:run_partial_correlation` | log returns | `partial_corr`, `precision_matrix`, `partial_corr_edges`, `glasso_metadata.json` |
| 10 | wavelet | `src/wavelet_analysis.py:run_wavelet_analysis` | log returns | `wavelet_corr_scale{1..7}`, `wavelet_mst_edges_*`, `wavelet_mst_metrics_*`, `wavelet_metadata.json` |
| 11 | transfer entropy | `src/transfer_entropy.py:run_transfer_entropy` | log returns | `transfer_entropy_matrix`, `net_transfer_entropy_matrix`, `te_network_edges`, `te_node_roles` |
| 12 | SNN (neuromorphic) | `src/snn_signals.py:run_snn_signals` | log returns + `adj_close` + `dislocation_candidates` | `snn_metrics.json`, `snn_pair_list.csv`, `snn_training_history.csv`, `snn_signals/{pair}.parquet`, `snn_model_weights/universal.pt`, `snn_spike_raster_sample`, `snn_membrane_sample` |

Stages 8–12 are the "EEE Analysis" group (informal label, surfaced in dashboard
sub-tabs). Stage 12 (SNN) is wrapped in `try/except ImportError` inside
`run_pipeline.py` — the rest of the pipeline completes if the optional
`[snn]` extra (`torch + snntorch`) is not installed.

Phase D parameterised every stage's output path by `config.market_id`, so
artifacts now live under `data/<market>/{raw,processed,results}/` rather than
the legacy `data/{raw,processed,results}/`. BIST, S&P-500, and EEG universes
coexist on disk via different `market_id` values.

## Module dependency graph (intra-`src/`)

```
config.py ◄─── (every module imports PipelineConfig and PROJECT_ROOT)

analysis.py ─────► clustering.py
                          ▲
                          │
                  rmt_denoising.py
                          │
                  wavelet_analysis.py  (also imports analysis.compute_correlation_matrix)
                  
preprocessing.py is a leaf (only depends on config)
data_acquisition.py is a leaf
data_validation.py is a leaf
rolling_correlation.py is a leaf
pair_dislocation.py is a leaf
partial_correlation.py is a leaf (uses sklearn directly)
transfer_entropy.py is a leaf
snn_signals.py imports from pair_dislocation (compute_spread / compute_zscore /
                compute_half_life / detect_signals); torch + snntorch are
                lazily imported via `_require_torch()`
```

`clustering.py:build_mst` and `mst_to_edge_df` and `compute_mst_metrics` are
the most reused functions in the pipeline (called by RMT and wavelet stages).

## Pipeline-time vs render-time

| Computation | Where |
|---|---|
| Correlation matrix (full universe, full date range) | Pipeline → `pearson_corr.parquet` |
| Correlation matrix (sub-window selected by user) | App → `dashboard.py:_compute_corr` (cached) |
| MST edges (full date range) | Pipeline → `mst_edges.csv` |
| MST node layout (kamada-kawai) | App → `dashboard.py:_mst_layout` (cached) |
| Rolling market stats (configured windows: w∈{60,120,252}, step=5, pearson) | Pipeline → `rolling_market_stats_w*.parquet`; app reads via precompute-first dispatch when params match the grid |
| Rolling market stats (off-grid window/step/method) | App → `dashboard.py:_compute_market_stats` (cached fallback) |
| Pair correlation/spread/half-life | Pipeline screens top-N → `dislocation_candidates`. App computes per-pair on demand. |
| TE shuffle null distribution | Pipeline (seeded). App reads precomputed matrix. |
| Wavelet decomposition | Pipeline (one pass per scale). App reads `wavelet_corr_scale{1..7}.parquet`. |
| SNN training (universal model) | Pipeline (one pooled training run, ~12 min CPU). App reads `snn_metrics.json` and the per-pair signal parquets. Re-inference reuses cached `universal.pt` (~30 s). |
| Cross-market comparison table | Script (`scripts/sp500_vs_bist.py`) → `data/comparison_bist_vs_sp500.csv`. App's Cross-Market page reads the CSV plus each universe's eigenvalue spectrum + MST + crisis-window stats directly. |

## Glossary

- **BIST-100** — Borsa Istanbul 100, the headline equity index in Turkey.
- **XU100** — yfinance ticker for the BIST-100 index.
- **Log return** — `ln(P_t / P_{t-1})`. Time-additive; symmetric around zero.
- **MST** — Minimum Spanning Tree. Subgraph that connects all N nodes with N-1
  edges of minimum total weight; here weights are correlation-distances.
- **Correlation distance** — `d_ij = sqrt(2(1 - ρ_ij))`. Maps `ρ ∈ [-1, 1]` to
  `d ∈ [0, 2]`. Standard in econophysics.
- **Hierarchical clustering / linkage** — repeated nearest-neighbour merges on
  the distance matrix (single-linkage by default).
- **Dendrogram leaf order** — the optimal left-to-right ordering of tickers
  produced by `scipy.cluster.hierarchy.leaves_list`. Used to reorder the
  correlation heatmap so blocks are visible.
- **RMT denoising** — Random Matrix Theory. Eigenvalues of an empirical
  correlation matrix that fall inside the Marchenko–Pastur band are treated
  as noise; eigenvalues above the upper bound are signal.
- **Marchenko–Pastur distribution** — limiting eigenvalue distribution of
  `(1/T) X X'` where `X` is a `T × N` matrix of i.i.d. standard-normal entries.
  Bounds: `λ± = σ² (1 + 1/q ± 2 √(1/q))` with `q = T/N`.
- **Glasso (Graphical LASSO)** — L1-penalised maximum-likelihood estimator
  of a sparse precision (inverse covariance) matrix.
- **Precision matrix** — `Θ = Σ⁻¹`. Off-diagonal zeros encode conditional
  independence under Gaussianity.
- **Partial correlation** — derived from precision: `pcorr_ij = -Θ_ij / sqrt(Θ_ii Θ_jj)`.
- **Wavelet decomposition** — DWT splits a series into approximation and
  detail bands. We use Daubechies-4 (`db4`); detail at level `ℓ` corresponds
  roughly to `2^ℓ`–`2^(ℓ+1)` day cycles.
- **Transfer entropy** — Shannon-information measure of directed dependence:
  `TE(X→Y) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})`. Non-negative.
  Significance via shuffle-source null distribution.
- **OU process / half-life** — under an AR(1) assumption `Δs_t = φ s_{t-1} + ε`,
  the mean-reversion half-life is `-ln(2) / ln(1+φ)` if `φ < 0`.
- **Z-score (rolling)** — standardised spread `(s_t - μ_w) / σ_w` over a
  trailing window `w`. Used to detect dislocation entries/exits.
- **Cross-sectional dispersion** — `std(returns across stocks at time t)`.
  Higher = more idiosyncratic moves; lower = market moving in sync.
- **LIF (Leaky-Integrate-and-Fire) neuron** — discrete-time recurrence
  `V[k+1] = β·V[k] + W·x[k] − S[k]·V_th`, with binary spike `S[k+1] = 1` if
  `V[k+1] ≥ V_th` else 0. The algorithmic substrate of neuromorphic chips
  (Intel Loihi, IBM TrueNorth, SpiNNaker).
- **Surrogate gradient** — smooth differentiable approximation (here
  `fast_sigmoid(slope=25)`) substituted for the non-differentiable Heaviside
  spike during backprop-through-time training of SNNs. Neftci, Mostafa &
  Zenke 2019.
- **Delta-modulation encoder** — two binary channels per scalar: `up` fires
  on `Δx ≥ +θ`, `down` fires on `Δx ≤ −θ`. Σ-Δ-style asynchronous
  analog-to-digital conversion applied to a 1-D signal; matches DVS event
  cameras.
- **Recurrent LIF (`snn.RLeaky`)** — LIF hidden layer augmented with a
  learned within-layer recurrence matrix `V_rec`; adds short-term
  associative memory across SNN ticks.
- **Membrane-potential readout** — output layer with `reset_mechanism="none"`;
  reads the final analog membrane voltage rather than spike counts.
  Standard SNN-training practice (Eshraghian et al. 2023) — gives a smooth
  loss landscape that the surrogate gradient can actually descend.
- **Focal loss** — `L = −α_y · (1 − p_y)^γ · log p_y` (Lin et al. 2017).
  Down-weights confidently-correct examples so the gradient focuses on the
  hard, near-decision-boundary minority class.

For the math behind RMT / Glasso / Wavelet / TE see
[`EEE_METHODS.md`](EEE_METHODS.md). For the SNN method see
[`SNN_Report.md`](SNN_Report.md).
