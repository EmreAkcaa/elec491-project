# StoNeCoAl — Technical Report

**Project:** StoNeCoAl: Stockmarket Network and Correlation Analysis
**Course:** Koç University ELEC 491, Spring 2026
**Team:** Arda Rutkay Var (75628), Emre Akça (80567)
**Supervisor:** Emir Salih Mağden
**TA:** Dilem Eşlik

---

## Abstract

StoNeCoAl is a reproducible 12-stage data pipeline and interactive Streamlit
dashboard that characterises the dependence structure of the Borsa İstanbul
BIST-100 equity universe through a battery of estimators drawn from electrical
and electronics engineering: random matrix theory (RMT) for signal/noise
separation in the empirical correlation matrix; the graphical LASSO for sparse
inverse-covariance estimation as a Gaussian conditional-independence graph;
discrete wavelet decomposition for multi-resolution analysis of co-movement;
transfer entropy for directed information flow; and reservoir computing (Echo
State Network) and surrogate-gradient spiking neural networks for predictive
evaluation. The analyses are reported alongside Pearson correlation and the
classical Mantegna (1999) minimum-spanning-tree (MST) baseline.

Five years and three months of daily price data (2020-01-01 → 2026-03-01,
1,543 trading days) are ingested for 73 BIST constituents that pass a 90 %
coverage threshold. Headline empirical findings include (i) an effective
informational dimensionality of 6.5 modes out of 73 nominal dimensions;
(ii) a covariance-induced reduction in joint differential entropy of 21.5 nats
(31 bits) relative to an uncorrelated null; (iii) MST hubs that map to the
three largest Turkish industrial groups (Koç, Şişecam, Sabancı); (iv) a
monotonic decay of MST edge overlap from 60 % at daily scale to 22 % at annual
scale, demonstrating that BIST co-movement structure is not scale-invariant;
(v) a 0.318 → 0.622 jump in mean pairwise correlation around the February 2023
earthquakes; and (vi) a documented negative result for daily-frequency neural
return-prediction consistent with the efficient-market hypothesis. Every
quantitative claim in this report is supported by an artifact under
`data/results/`.

---

## 1. System overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ External data sources                                                        │
│   yfinance HTTP API ──► daily Adj-Close + Close (chunked 25-ticker batches)  │
│   İş Yatırım (HGDG)  ──► cross-validation sample (10 tickers, seed=42)       │
└────────────────────┬─────────────────────────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Pipeline (run_pipeline.py — 12 stages, deterministic given fixed raw data)   │
│  1. data_acquisition       → data/raw/prices_raw.parquet, xu100.parquet      │
│  2. data_validation        → validation_report.csv                           │
│  3. preprocessing          → adj_close, log_returns, coverage, anomalies     │
│  4. analysis               → pearson_corr, distance_matrix, summary_stats    │
│  5. clustering             → linkage_matrix, mst_edges, cluster_assignments  │
│  6. rolling_correlation    → rolling_market_stats_w{60,120,252}              │
│  7. pair_dislocation       → dislocation_candidates                          │
│  8. rmt_denoising          → eigenvalue_spectrum, denoised_corr, MST         │
│  9. partial_correlation    → partial_corr, precision_matrix, edges (Glasso)  │
│ 10. wavelet_analysis       → wavelet_corr_scale1..7, MST per scale           │
│ 11. transfer_entropy       → TE matrix, net flow, edges, node roles          │
│ 12. reservoir_computing    → rc_dispersion_predictions, rc_metrics           │
│                                                                              │
│ Extra (scripts/extra_analysis.py — derived metrics for this report)          │
│   • mutual_information_matrix.parquet (3-bin equal-frequency)                │
│   • mi_pearson_comparison.csv (empirical MI vs Gaussian lower bound)         │
│   • wavelet_entropy.csv (per-ticker H_w from corr-weighted scale proxy)      │
│   • crisis_window_stats.csv (±60-day correlation around named events)        │
│   • methods_comparison.csv (sector purity + Jaccard vs raw MST)              │
│   • it_summary.json (consolidated information-theoretic scalars)             │
└────────────────────┬─────────────────────────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Dashboard (Streamlit; app/dashboard.py, app/pair_analysis.py, eee_analysis)  │
│   • Market Overview (6 tabs: Data, Correlation, Clustering & Network,        │
│     Rolling, Pairs, EEE Analysis — 5 sub-tabs incl. Forecasting)             │
│   • Pair Analysis (5 sub-tabs: Overview, Correlation, Risk, Spread, Network) │
│   • Theme-aware chart export, cached loaders for every artifact              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Reproducibility.** Given a fixed `data/raw/prices_raw.parquet` and the seeded
TE configuration (`config/settings.yaml: transfer_entropy.seed = 42`), the
pipeline produces byte-identical outputs across runs. All 87 unit tests pass
under `uv run python -m pytest -q`.

**Universe.** 102 tickers in `config/universes/bist100.csv` (the BIST-100
membership exhibits cardinality 102 in vendor sources at our snapshot date).
Coverage filter at 90 % retains **73 tickers**. The 29 dropped tickers include
9 Energy, 6 Technology, 2 Pharmaceuticals, 2 Mining, 2 Food, 2 Building
Materials, 1 Transportation, 1 Sports, 1 Retail, 1 REIT, 1 Industrials, 1
Conglomerate — predominantly post-2022 IPOs lacking five years of history,
plus two delisted names (KOZAA, KOZAL). **The analysis universe therefore
under-represents Energy (15 → 6) and Technology (9 → 3) relative to the
nominal index.**

---

## 2. Information-theoretic characterisation of BIST market dependence

This section consolidates every estimator in the project under a single
information-theoretic frame. Each subsection states (a) the measured property
of the joint return distribution, (b) the estimator implementation, (c) the
empirical result with concrete numbers, and (d) caveats.

### 2.1 Information capacity and effective dimensionality (RMT)

**Measured property:** decomposition of the joint differential entropy into
orthogonal modes; classification of modes as signal vs. finite-sample noise.

**Estimator:** Eigendecomposition of the 73×73 Pearson correlation matrix by
`numpy.linalg.eigh` (`src/rmt_denoising.py`). Eigenvalues are compared against
the analytic Marchenko–Pastur upper bound

`λ₊ = σ² (1 + 1/q + 2·√(1/q))`

with `q = T/N = 1543/73 = 21.137` and `σ² = 1`, giving **λ₊ = 1.482**.

**Results** (`data/results/eigenvalue_spectrum.csv`):

| Quantity | Value |
|---|---|
| N (tickers) | 73 |
| T (days) | 1,543 |
| MP upper bound λ₊ | 1.482 |
| Eigenvalues above λ₊ (signal) | 4 |
| {λ₁, λ₂, λ₃, λ₄} | {27.86, 3.04, 1.81, 1.52} |
| Signal-variance share Σ λ_signal / N | **46.9 %** |
| Top-eigenvalue share λ₁ / N (market mode) | **38.2 %** |
| Participation ratio `(Σλ)² / Σλ²` = effective dimensionality | **6.52** |
| log det Σ = Σ log λ | −43.06 |
| Differential-entropy reduction `ΔH = −½ log det Σ` | **21.53 nats ≈ 31.06 bits** |

**Interpretation.** Out of 73 nominal degrees of freedom, the BIST log-return
panel carries informational content in only ~6.5 effective modes; the
remaining ~66 directions are statistically indistinguishable from
finite-sample noise under the MP null. The leading mode alone captures 38 %
of total variance — the *market mode*, the direction along which "everything
moves together." The 21.5-nat (31-bit) reduction in joint differential
entropy versus an uncorrelated null is a single scalar that quantifies the
total informational integration of the index.

**Caveat.** Differential entropy under multivariate Gaussianity is an
approximation: BIST returns are heavy-tailed (excess kurtosis well above 0),
so the true entropy reduction is plausibly larger in absolute magnitude. The
21.5-nat figure is a lower bound on the true Gaussian-equivalent integration.

### 2.2 Pairwise nonlinear dependence (Mutual Information)

**Measured property:** symmetric, model-free pairwise dependence including
nonlinear coupling.

**Estimator:** `compute_mutual_information_matrix` in
`scripts/extra_analysis.py`. Each ticker is discretised by equal-frequency
binning into n_bins = 3 (matching the discretisation used by the existing TE
module `src/transfer_entropy.py`). Pairwise MI is computed by the plug-in
estimator

`I(X;Y) = H(X) + H(Y) − H(X,Y)`

over the joint 3×3 histogram, yielding a 73×73 symmetric matrix
(`data/results/extra/mutual_information_matrix.parquet`).

**Results.** From `data/results/extra/it_summary.json`:

| Quantity | Empirical (3-bin) | Gaussian lower bound `−½ log(1−ρ²)` |
|---|---|---|
| Mean pairwise MI (nats) | **0.0503** | **0.0787** |
| Median pairwise MI (nats) | 0.0465 | 0.0705 |
| Max pairwise MI (nats) | 0.341 (AKBNK–YKBNK) | 0.671 (AKBNK–YKBNK) |
| Sum across 2,628 pairs (nats) | 132.2 | 206.8 |
| Number of pairs with `MI_emp / MI_gauss > 1.5` | **14** | — |

**Important interpretive note on the empirical-vs-Gaussian comparison.**
The empirical 3-bin estimator has a hard ceiling at `log 3 ≈ 1.10` nats per
pair (the maximum joint entropy of a 3×3 distribution). For strongly
linearly-correlated pairs the Gaussian formula `−½ log(1−ρ²)` is not
bounded and grows fast as `ρ → 1`. This is why the mean empirical MI is
*below* the mean Gaussian MI: the 3-bin discretiser saturates on strongly
linear pairs (AKBNK–YKBNK: emp 0.341 vs gauss 0.671 nats). The
`MI_emp / MI_gauss > 1.5` filter is therefore meaningful only for *weakly*
linearly-correlated pairs where the Gaussian formula is small enough that
non-Gaussian structure in the joint distribution can push the empirical
estimator above it. That is exactly the regime our filter targets.

**Top 5 pairs by ratio MI_emp / MI_gauss:**

| Pair | ρ | MI_emp (nats) | MI_gauss (nats) | ratio |
|---|---|---|---|---|
| HEKTS – ZOREN | +0.313 | 0.0865 | 0.0517 | **1.67** |
| HEKTS – TKFEN | +0.247 | 0.0621 | 0.0314 | **1.98** |
| HEKTS – BUCIM | +0.293 | 0.0750 | 0.0448 | **1.67** |
| HEKTS – PETKM | +0.298 | 0.0718 | 0.0465 | **1.54** |
| HEKTS – TKNSA | +0.250 | 0.0574 | 0.0323 | **1.78** |

**Interpretation.** A small population of 14 pairs (out of 2,628) exhibits
mutual information that exceeds the Gaussian lower bound by more than 50 %,
indicating dependence structure that Pearson correlation under-counts.

**Caveat — critical.** All five top-ratio pairs involve HEKTS (Hektaş, an
agricultural chemicals issuer). HEKTS has a residual unhandled corporate
action on 2024-09-09 (price 11.76 → 4.13; see §6.1). This produces a
discrete one-day discontinuity that the 3-bin MI estimator captures as
"extra structure" beyond the linear Pearson estimator. **The HEKTS-driven
excess MI is therefore plausibly an artifact of the data-quality issue
rather than genuine market nonlinearity.** After excluding HEKTS, the 9
remaining `MI_emp / MI_gauss > 1.5` pairs are smaller-magnitude excesses
that may reflect true heavy-tail or threshold-style coupling. We report both
in the artifact `mi_pearson_comparison.csv` and recommend the HEKTS data
correction as immediate future work.

### 2.3 Conditional independence (Graphical LASSO as Markov-blanket graph)

**Measured property:** the sparsity pattern of the inverse covariance matrix
`Θ = Σ⁻¹`. Under joint Gaussianity, `Θ_ij = 0` ⇔ `I(X_i ; X_j | rest) = 0`,
so the sparse precision matrix encodes a *conditional-independence graph*.

**Estimator:** `sklearn.GraphicalLassoCV(cv=5, max_iter=200)` in
`src/partial_correlation.py`. Partial correlation `ρ_ij|rest = −Θ_ij /
√(Θ_ii · Θ_jj)`. Edge retention threshold `|ρ_ij|rest| > 0.01`.

**Results** (`data/results/glasso_metadata.json`,
`data/results/partial_corr_edges.csv`):

| Quantity | Value |
|---|---|
| CV-selected α | 2.34 × 10⁻⁴ (near-zero L1 penalty) |
| Edges retained | 736 of 2,628 unique pairs (**28 %**) |
| Top partial correlations | BRYAT–BRSAN (0.50), VAKBN–HALKB (0.41), PGSUS–THYAO (0.32), EREGL–KRDMD (0.32), AKBNK–GARAN (0.26) |

**Interpretation.** The CV-selected α is small, so the resulting graph is
better described as a *regularised inverse covariance* than as a strictly
sparse conditional-independence structure. The retained 28 % of pair edges
identify direct intra-sector linkages — bank-bank (VAKBN–HALKB, AKBNK–GARAN),
steel-steel (EREGL–KRDMD), airlines-airlines (PGSUS–THYAO) — that survive
conditioning on the rest of the universe and are therefore not mediated by
common market or sector exposure. Most of the remaining 72 % of pairwise
correlations are statistically explained away by the market mode and
sector-level drivers.

### 2.4 Directed information flow (Transfer Entropy)

**Measured property:** asymmetric reduction in uncertainty about `Y_t` provided
by `X_{t-lag}` beyond `Y_{t-lag}` — Schreiber (2000) directed information
transfer.

**Estimator:** `src/transfer_entropy.py`. Equal-frequency 3-bin
discretisation; plug-in entropy estimator for the four-term Shannon
decomposition

`TE(X→Y) = H(Y_t, Y_lag) − H(Y_t, Y_lag, X_lag) − H(Y_lag) + H(Y_lag, X_lag)`

Significance assessed by 100 source-shuffle permutations (RNG seeded at 42
via `config.transfer_entropy.seed`). Edges with p ≥ 0.05 zeroed.

**Results** (`data/results/te_node_roles.csv`,
`data/results/te_network_edges.csv`):

| Quantity | Value |
|---|---|
| Significant directed pair edges | 609 of 5,256 ordered pairs (**11.6 %**) |
| Per-edge TE for top-100 retained edges | 0.012 – 0.014 nats |
| Per-ticker net flow `(TE_out − TE_in)` range | −0.19 to +0.13 nats |

**Top-5 net outflow tickers (sources):**

| Ticker | Sector | Net TE (nats) |
|---|---|---|
| THYAO | Airlines | +0.129 |
| BERA | Conglomerates | +0.117 |
| DOAS | Automotive | +0.089 |
| TUPRS | Energy | +0.079 |
| SISE | Industrials | +0.074 |

**Interpretation.** Detected directed coupling is weak in absolute terms
(per-edge effect sizes 0.01–0.02 nats, two orders of magnitude smaller than
the largest pairwise correlations expressed as Gaussian MI). The
top-outflow tickers cluster around airlines (THYAO), conglomerates with
energy-sector exposure (BERA), automotive (DOAS), and refining (TUPRS) —
plausibly reflecting Turkey-specific channels (currency, fuel-pass-through,
tourism) but with effect sizes too small to support strong economic claims.

**Caveat — documented in `docs/KNOWN_ISSUES.md` (M-1).** The source-shuffle
null breaks both cross-dependence and source autocorrelation; significance
is therefore overly liberal. The 609 retained edges include a non-trivial
fraction of edges whose "significance" reflects autocorrelation in `X`
rather than true `X → Y` information flow. Replacing the permutation null
with IAAFT (iterative amplitude-adjusted Fourier transform) surrogates
that preserve `X`'s spectral content (Kantz & Schreiber 2004) is left as
future work and is expected to materially reduce the number of significant
edges.

### 2.5 Multi-scale information distribution (Wavelet decomposition)

**Measured property:** scale-dependent variance and co-movement structure
of log returns.

**Estimator:** `src/wavelet_analysis.py`. Discrete wavelet transform using
Daubechies-4 (`db4`); for each ticker, only the detail coefficients at level
`ℓ ∈ {1, …, 7}` are reconstructed (via `pywt.waverec` with all other
coefficient arrays zeroed). A correlation matrix and MST are computed
per scale.

**Per-scale band labels:**

| Scale | Band | Cycle |
|---|---|---|
| 1 | 2–4 day | Sub-weekly noise / microstructure |
| 2 | 4–8 day | Weekly |
| 3 | 8–16 day | Bi-weekly to monthly |
| 4 | 16–32 day | Monthly |
| 5 | 32–64 day | Quarterly |
| 6 | 64–128 day | Semi-annual |
| 7 | 128–256 day | Annual |

**Inter-scale MST overlap.** Two metrics are reported:
(a) `common / |raw|` — what fraction of the raw MST's 72 edges is preserved
in the scale-decomposed MST (each tree has 72 edges by construction); and
(b) the Jaccard index `|A ∩ B| / |A ∪ B|`. The first is the more intuitive
"reproduction rate"; the second is the symmetric set-similarity coefficient.

| Comparison | Common edges (of 72) | (a) common / \|raw\| | (b) Jaccard |
|---|---|---|---|
| Raw MST vs scale-1 (2–4 d)         | 43 | **60 %** | 0.426 |
| Raw MST vs scale-2 (4–8 d)         | 46 | 64 %     | 0.469 |
| Raw MST vs scale-3 (8–16 d)        | 32 | 44 %     | 0.286 |
| Raw MST vs scale-5 (32–64 d)       | 23 | 32 %     | 0.190 |
| Raw MST vs scale-7 (128–256 d)     | 16 | **22 %** | 0.125 |
| Scale-1 vs scale-7 (daily vs annual) | 13 | **18 %** | (computed) |

The headline progression `60 % → 22 % → 18 %` uses metric (a) throughout
(consistent with the wavelet community's reporting convention); the
cross-method table in §5 uses metric (b) Jaccard (consistent with
network-science convention for set similarity). Both metrics tell the
same story: BIST's correlation network is monotonically less and less
similar to the unconditional MST as the analysis timescale grows.

**Per-ticker wavelet entropy** (`data/results/extra/wavelet_entropy.csv`).
Computed from a per-scale proxy weight `w_s(ticker) = mean |corr_s(ticker, ·)|`
normalised across scales, then `H_w = −Σ_s p_s log p_s`. Range (normalised
to `[0,1]` by `log(7)`): **min = 0.953, mean = 0.986, max = 0.996**. Top-5
most spectrally diverse tickers: INDES, SKBNK, SISE, EKGYO, ISCTR. Top-5
most spectrally concentrated: CCOLA, AGESA, DOAS, PAPIL, BRYAT.

**Interpretation.** MST edge overlap between the unconditional and
scale-decomposed networks decreases monotonically from 60 % at the daily
scale to 22 % at the annual scale; the two extreme scales share only 18 %
of edges. **This quantifies information non-redundancy across timescales:**
the network structure at one scale is largely distinct from the network
structure at another scale, validating the multi-resolution framework.
Short-scale structure (60 % overlap with unconditional) is dominated by
common market shocks; long-scale structure (22 % overlap) reflects
sectoral and macroeconomic linkages that average out at higher frequencies.

**Caveat.** Per-ticker wavelet entropy `H_w` values are clustered tightly
near the maximum entropy of `log 7 ≈ 1.95` nats — nearly all tickers show
roughly uniform variance across the seven scales by the proxy used.
This is partly because the per-scale weight was derived from a correlation
proxy (mean absolute row correlation of `wavelet_corr_scale{s}`) rather
than from per-ticker variance of the detail-band time series themselves
(which are not currently persisted by the pipeline). A direct
variance-of-detail-band calculation would require modifying
`src/wavelet_analysis.py` to also save the reconstructed detail series, and
is recommended future work. The per-ticker ranking should therefore be
treated as indicative rather than definitive.

### 2.6 Predictive evaluation as information-bottleneck (ESN + SNN)

**Measured property:** how much of the predictive information about
20-trading-day-ahead spread Z-score or next-day market dispersion is
recoverable from a high-dimensional feature vector versus from a single
scalar (the current Z-score).

**Estimators:**
- **ESN (`src/reservoir_computing.py`):** 300-neuron leaky reservoir,
  spectral radius 0.9, leak rate 0.3, ridge readout (`α = 10`). Target:
  next-day cross-sectional dispersion. 17 features (cross-sectional stats,
  vol windows, PCA factors). 5-fold walk-forward CV.
- **SNN (`src/snn_signals.py` on branch `arda/eee-analysis`):** 2-layer
  recurrent leaky integrate-and-fire (RLeaky 96 hidden, β = 0.92, V_th = 0.5),
  surrogate-gradient training (`fast_sigmoid`, slope 25), focal loss with
  `sqrt(inv_freq)` class weights, 25 max epochs with early-stop patience 5.
  Target: 3-class HOLD/BUY/SELL labels from a 20-day forward mean-reversion
  oracle. 11 raw features → 45 spike-encoded channels (delta-modulation +
  population encoding) + 20 pair one-hot. Trained as a single universal
  model across 20 pairs.

**Results.**

| Model | Headline metric | Baseline | Reading |
|---|---|---|---|
| ESN (dispersion) | aggregate R² = **0.063** | persistence R² = −0.91; mean R² = −0.07 | Near-zero out-of-sample skill; per-fold R² degrades 0.286 → 0.033 → −0.006 → −0.014 → −0.155 (overfitting) |
| ESN | Direction-of-change accuracy | 0.452 | Indistinguishable from 0.5 |
| ESN | Top-importance features | All four are volatility statistics | Model rediscovers volatility clustering (Engle 1982); no new predictive content beyond past vol |
| SNN (pair signals) | mean macro-F1 | 0.668 | Above majority-class baseline 0.263; above random 0.33 |
| SNN | mean Δ-Sharpe vs classical \|Z\|>2 | **−1.108** | SNN underperforms a one-line heuristic |
| SNN | pairs where SNN beats classical on Sharpe | **5 of 20** | Loses on 75 % of pairs |

**Interpretation under the information-bottleneck framework
(Tishby, Pereira, Bialek 2000).** A high-capacity dynamical-system model
(ESN: 300-dim reservoir, 17 input features) and a recurrent spike-coded
classifier (SNN: 96-dim hidden, 65 input channels) both fail to extract
predictive content beyond what a one-feature scalar threshold rule
captures. We read this as: *the mutual information between past returns
and 20-day-ahead mean-reversion targets at daily frequency is concentrated
in a near-one-bit channel — the current spread Z-score itself — and is
not increased by high-dimensional feature augmentation or nonlinear
function approximation.* This convergence of two architecturally different
neural failures into the same negative result is consistent with weak-form
efficient-market predictions at this horizon. We report this jointly as a
documented exploratory negative finding, not as a method failure.

**Caveats.**
- The ESN's PCA features are fit on the full return history before
  walk-forward CV begins (`KNOWN_ISSUES.md` M-2); reported R² is therefore
  an upper bound on strictly causal performance.
- The SNN's forecast oracle uses a 20-day forward look during label
  generation. Inference is causal; labels are not.
- The SNN branch (`arda/eee-analysis`, commit `f0b78ca`) is currently
  unmerged. See §7.

---

## 3. Network structure: hubs, MST topology, clustering

### 3.1 Raw MST hubs

`data/results/mst_node_metrics.csv`, sorted by betweenness centrality:

| Rank | Ticker | Sector | Degree | Betweenness | Real-world role |
|---|---|---|---|---|---|
| 1 | KCHOL | Conglomerates | 7 | **0.723** | Koç Holding — owns YKBNK, FROTO, TOFAŞ, TUPRS, ARÇELİK |
| 2 | SISE | Industrials | 8 | 0.587 | Şişecam — glass/soda-ash/chemicals industrial holding |
| 3 | SAHOL | Conglomerates | 6 | 0.516 | Sabancı Holding — owns AKBNK, ENRJSA, TEKNOSA |
| 4 | ARCLK | Consumer Durables | 2 | 0.222 | Arçelik (Koç white-goods subsidiary) |
| 5 | VESBE | Consumer Durables | 5 | 0.209 | Vestel Beyaz Eşya |
| 6 | PETKM | Chemicals | 4 | 0.182 | Petkim (SOCAR-owned; supplies TUPRS) |
| 7 | KRDMD | Steel | 3 | 0.180 | Kardemir |
| 8 | AKBNK | Banking | 3 | 0.180 | Akbank (Sabancı's bank) |
| 9 | AGHOL | Conglomerates | 3 | 0.158 | Anadolu Group |
| 10 | CIMSA | Building Materials | 3 | 0.133 | Çimsa |

**Interpretation.** The MST extracted purely from price-return correlation
identifies three of the largest Turkish industrial groups — Koç Holding,
Şişecam, Sabancı Holding — as the network's highest-betweenness nodes; their
subsidiaries (Arçelik for Koç, Akbank for Sabancı) appear as secondary hubs.
The statistical network topology recovers the actual ownership and supply-
chain structure of the Turkish economy. **This is a substantive validation
of the MST methodology on BIST data, and a finding specific to a market
where conglomerate holdings — not standalone banks — are the central
intermediaries.**

### 3.2 Hierarchical clustering — single-linkage chaining

`data/results/cluster_assignments.csv`:

| Cluster size distribution | Count |
|---|---|
| 1 mega-cluster (cluster_id = 5) | 44 of 73 tickers (60 %) |
| 2-member clusters | 2 |
| Singletons | 25 |
| Total cluster ids | 28 |

The mega-cluster mixes every sector: 7 banks, 4 conglomerates, both airlines,
6 energy, 3 durables, both telecoms, defense, 3 steel, 4 retail, food,
beverages, chemicals, 2 industrials, REIT, 2 automotive, 3 building materials.

**Interpretation and treatment.** This is the textbook *chaining* artifact of
single-linkage clustering on a dense correlation graph: at any distance
threshold low enough to retain meaningful structure, the strongest pairwise
links transitively connect almost every node through intermediaries. The
distance threshold `t = 1.0` (out of a maximum 2.0 from `d = √(2(1−ρ))`)
admits ~60 % of the universe into a single connected component.

**Reporting choice.** We surface the *dendrogram* (which preserves the
hierarchical structure regardless of cut threshold) and the *MST hub
analysis* (which is robust to chaining) rather than the cluster-id table in
the dashboard and report. The codebase supports drop-in replacement of
single-linkage with `average` or `ward` linkage and an explicit cluster-count
target (`fcluster(Z, t=K, criterion='maxclust')`); this is a recommended
future-work patch in `src/clustering.py`.

### 3.3 RMT-denoised MST — a methodological cautionary tale

`data/results/extra/methods_comparison.csv` (computed by
`scripts/extra_analysis.py`):

| MST variant | n edges | Sector purity (intra-sector edge fraction) | Jaccard vs raw MST |
|---|---|---|---|
| **Raw MST (Pearson distance)** | 72 | **0.403** | 1.000 |
| RMT-denoised MST (constant-replacement) | 72 | **0.153** | **0.161** |
| Wavelet scale-1 MST | 72 | 0.361 | 0.426 |
| Wavelet scale-2 MST | 72 | 0.361 | 0.469 |
| Wavelet scale-3 MST | 72 | 0.333 | 0.286 |
| Wavelet scale-4 MST | 72 | 0.278 | 0.274 |
| Wavelet scale-5 MST | 72 | 0.236 | 0.190 |
| Wavelet scale-6 MST | 72 | 0.319 | 0.161 |
| Wavelet scale-7 MST | 72 | 0.222 | 0.125 |
| Glasso partial-correlation edges | 736 | 0.125 | 0.096 |
| Transfer-entropy directed edges | 609 | 0.062 | 0.030 |

**Critical finding on RMT denoising.** The denoised MST has the *worst* sector
purity (15 %) of any MST-style network in the project — substantially below
the raw MST's 40 % — and shares only 16 % of edges with the raw MST. Inspection
of the denoised network reveals SISE as a degree-22 super-hub (vs. degree 8 in
the raw MST); the maximum off-diagonal correlation in the denoised matrix is
attenuated from 0.86 (AKBNK–YKBNK in the raw) to 0.57.

This is a known artifact of the `method='constant'` strategy in
`src/rmt_denoising.py`: replacing each noise eigenvalue with the *mean of
all noise eigenvalues* creates near-ties among many noise-pair distances,
which the MST algorithm resolves by routing through a single super-node. The
codebase also supports `method='zero'` as an alternative — noise eigenvalues
are set to zero rather than averaged, which truncates the matrix to the
4-dim signal subspace.

**We ran the head-to-head comparison** (`scripts/rmt_method_comparison.py`,
output `data/results/extra/rmt_method_comparison.csv`):

| Method | Max \|off-diag corr\| | MST max degree | Top-3 hubs | Sector purity | Jaccard vs raw |
|---|---|---|---|---|---|
| Raw | 0.86 | 8 | KCHOL, SISE, SAHOL | **0.403** | 1.000 |
| RMT `constant` | 0.57 (attenuated) | **22** (super-hub) | SISE, PETKM, KCHOL | 0.153 | 0.161 |
| RMT `zero` | **0.9999** (inflated) | 4 | ALARK, TKFEN, VESTL | 0.222 | 0.125 |

Cross-method Jaccard `(constant, zero) = 0.108` — the two denoised MSTs
are almost completely different from each other and from the raw.

**Neither replacement strategy dominates the raw MST.** `constant`
redistributes noise variance uniformly and produces a super-hub topology;
`zero` truncates to the 4-mode signal subspace, which makes many pairs
near-collinear in that subspace (max off-diagonal correlation 0.9999) and
moves the MST hubs from the real top-3 conglomerates (KCHOL, SISE, SAHOL)
to a different set of mid-cap industrials (ALARK, TKFEN, VESTL). With
`q = T/N = 21.1` the project's signal-to-noise ratio is high enough that
the noise band carries genuinely useful structural information that both
replacements destroy. **RMT denoising should therefore be presented as
a methodological alternative whose downstream topology differs from the
raw MST in two distinct ways, not as a "clean version" of the raw MST.**
A more principled denoising — e.g., the eigenvalue-shrinkage estimator of
Ledoit & Wolf (2004) or rotational-invariant estimators (Bun, Bouchaud,
Potters 2017) — is left as future work.

We retain both denoised MSTs in the dashboard alongside the raw MST as
side-by-side comparison views; the raw MST remains the canonical network
for sector-validation and hub-identification claims.

---

### 3.4 Cross-market comparison: BIST-73 vs S&P-485

Computed by `scripts/sp500_vs_bist.py` after running the same 12-stage pipeline
against `config/settings_sp500.yaml` (universe: 500 S&P 500 constituents, 15
dropped by the 90 % coverage filter as post-2022 IPOs/spin-offs, 485 surviving;
3 dual-class share duplicates GOOG/FOX/NWS pre-removed from the universe CSV
per `KNOWN_ISSUES.md` G-2).

| Metric | BIST-73 | S&P-485 |
|---|---|---|
| **Structural — RMT** | | |
| Effective dimensionality `D_eff = (Σλ)²/Σλ²` | **6.30** | **6.56** |
| Number of signal eigenvalues (λ > MP+) | 4 | 17 |
| Top eigenvalue share (market mode) | **38.9 %** | **38.1 %** |
| Signal-mode variance share | 47.6 % | 61.1 % |
| Differential-entropy reduction ΔH (Gaussian) | 22.0 nats | 307.3 nats |
| **Pairwise correlation** | | |
| Mean pairwise correlation | 0.374 | 0.365 |
| Median pairwise correlation | 0.368 | 0.364 |
| Std of pairwise correlations | 0.083 | 0.132 |
| Max pairwise correlation | 0.859 (banks-banks) | 0.927 (apartment REITs) |
| **Mutual information (Gaussian lower bound)** | | |
| Mean pairwise MI (nats) | 0.082 | 0.086 |
| Sum across all pairs (nats) | 214.7 | 10,086 |
| **Network — MST top-3 hubs (sector)** | KCHOL Conglomerates (0.72); SISE Industrials (0.59); SAHOL Conglomerates (0.52) | PRU Financials/insurance (0.65); AMP Financials/asset mgmt (0.63); PH Industrials/diversified (0.53) |
| MST sector purity | 0.40 | **0.80** |
| **Conditional independence — Glasso** | | |
| Number of edges retained | 744 | 845 |
| Glasso sector purity | 0.13 | **0.51** |
| **Directed coupling — Transfer Entropy** | | |
| Significant directed edges | 617 | 20,729 |
| TE sector purity | 0.06 | 0.11 |
| **Crisis-window mean pairwise correlation** | | |
| COVID-19 (during, 2020-03-11 + 60 d) | 0.547 | 0.648 |
| Russia-Ukraine (during, 2022-02-24 + 60 d) | 0.490 | 0.315 |
| Türkiye earthquakes (during, 2023-02-06 + 60 d) | **0.622** | 0.347 |
| Türkiye earthquakes (before, 2023-02-06 − 60 d) | 0.318 | 0.443 |

**Five publishable cross-market findings:**

1. **Effective informational dimensionality is a structural invariant ≈ 6–7
   modes in both markets.** Despite S&P-485 having 6.6× more tickers than
   BIST-73, the participation ratio `D_eff = (Σλ)² / Σλ²` is essentially
   identical (6.30 vs 6.56). The "informational rank" of an equity universe
   does not scale linearly with N — a real cross-market constant.

2. **Top eigenvalue share is also invariant at ≈ 38 % in both markets.**
   The market-mode share of variance is 38.9 % on BIST and 38.1 % on S&P-485.
   Both markets concentrate roughly the same fraction of their total
   informational content in a single "everything-moves-together" direction.

3. **The hub composition differs structurally.** BIST is hubbed on
   family-conglomerate holdings (Koç Holding, Şişecam, Sabancı Holding) —
   characteristic of an emerging market organised around family-controlled
   conglomerates. S&P is hubbed on insurance + asset-management names (PRU
   Prudential, AMP Ameriprise) plus the diversified industrial Parker-Hannifin
   (PH) — companies whose cash flows reflect economy-wide interest-rate
   and credit-spread regimes that propagate via correlation. **Not banks** as
   one might expect from US headline-equity narratives — the actual MST hubs
   are conglomerate-like financial intermediaries with cross-sector exposure.

4. **MST sector purity is 2× higher on S&P (0.80) than on BIST (0.40).**
   Same applies to Graphical LASSO partial-correlation edges (0.51 vs 0.13).
   S&P-485's GICS sector taxonomy clusters much more cleanly in correlation
   space than the Borsa Istanbul sector taxonomy clusters BIST-73 — a real
   developed-vs-emerging-market structural difference. (TE sector purity is
   low for both: 0.06 / 0.11 — TE captures directed dependencies that
   explicitly cross sectoral boundaries, so low purity is the correct
   expectation.)

5. **The 2023 February earthquakes are a BIST-specific signal.** Mean
   pairwise correlation jumped 0.318 → 0.622 (nearly doubled) on BIST in the
   60-day window after the magnitude-7.8 Kahramanmaraş quake. S&P showed no
   contemporaneous effect (0.443 → 0.347). The earthquake is a
   Türkiye-domestic shock that propagates through Turkish equity correlation
   structure without crossing to US markets — a clean demonstration that the
   StoNeCoAl toolkit, applied to two markets in parallel, can isolate
   country-specific systemic events from global shocks.

The full comparison table including before/during/after windows for COVID-19,
the Russia-Ukraine invasion, and the Türkiye earthquakes is persisted as
`data/comparison_bist_vs_sp500.csv`.

---

## 4. Crisis-window correlation analysis

`data/results/extra/crisis_window_stats.csv` — mean pairwise correlation
in ±60-day windows around three named events, computed from
`rolling_market_stats_w60.parquet`:

| Event | Date | Before (avg corr) | During (avg corr) | After (avg corr) |
|---|---|---|---|---|
| COVID-19 WHO pandemic declaration | 2020-03-11 | — (no pre-data) | **0.547** | 0.389 |
| Russia–Ukraine invasion | 2022-02-24 | 0.437 | **0.490** | 0.322 |
| Türkiye earthquakes (Kahramanmaraş) | 2023-02-06 | 0.318 | **0.622** | 0.536 |

(Baseline: the unconditional mean pairwise correlation across the full
1,543-day sample is **0.365**, from `pipeline_metadata.json`.)

**Interpretation.** All three events exhibit measurable correlation
tightening *during* the event window relative to the project's unconditional
baseline:
- **COVID-19**: the rolling-60 series starts on 2020-03-26 (first complete
  60-day window), so the "before" segment is empty. The "during" mean of
  0.547 is ~0.18 above the unconditional baseline and reverts within the
  60-day post-window.
- **Russia–Ukraine invasion**: a modest +0.05 elevation during the event,
  with a subsequent decline below baseline in the post window — consistent
  with a transient sentiment shock followed by sector divergence as the
  Turkish economy adjusted to the geopolitical regime.
- **Türkiye earthquakes (Kahramanmaraş, 7.8 + 7.5 M)**: the most striking
  signal in the dataset — average pairwise correlation roughly **doubled**
  from 0.318 (pre-event) to 0.622 (during), with **0.536 (sustained
  elevation 60-120 days after).** This is consistent with a market-wide
  re-pricing event affecting industrial, banking, and insurance exposures
  simultaneously.

**This is a substantive empirical finding specific to BIST: the February
2023 earthquakes produced the single largest correlation-tightening signal
observable in the project's analysis window.** It is the strongest
"information concentration during crisis" event in the dataset and offers
a quantitative complement to the established stylised fact (Forbes &
Rigobon 2002) on contagion during stress.

**Caveats.** The "before" baseline for the earthquake event is 0.318 —
*below* the unconditional sample mean of 0.365 — reflecting a relatively
quiet early-2023 period. The 0.62 "during" figure is therefore not directly
comparable to the COVID-19 0.547 because the relevant baseline differs.
Reporting all three events together with explicit before/during/after
windows is the honest disclosure.

---

## 5. Methods comparison — the comparative-laboratory result

`data/results/extra/methods_comparison.csv` (also presented in §3.3):

> Computed in `scripts/extra_analysis.py:methods_comparison`. Sector purity
> = fraction of edges connecting same-sector tickers, using the sector map
> from `cluster_assignments.csv`. Jaccard vs raw MST = `|A ∩ B| / |A ∪ B|`
> on edge sets.

**Headline observations:**
1. **Raw MST has the highest sector purity (40 %)** — the simplest method
   produces the most sectorally-coherent network on BIST.
2. **Wavelet scale-1 (60 % edge overlap, 36 % purity) is closest to the
   raw MST** — daily-scale correlation captures roughly the same network
   structure as the unconditional correlation, consistent with daily
   trading dominating the unconditional signal.
3. **MST sector purity degrades monotonically with scale** (40 → 36 → 33 →
   28 → 24 → 22 % across wavelet scales 1–7) — at longer scales the MST
   structure increasingly reflects macroeconomic exposure axes that cross
   conventional BIST sector boundaries.
4. **Denoised MST has the worst sector purity (15 %)** — see §3.3 for the
   methodological explanation.
5. **Glasso (12.5 % purity, 736 edges) and TE (6 % purity, 609 edges)**
   produce denser, lower-purity edge sets — they identify direct and
   directed dependencies that explicitly are *not* the same thing as
   "tickers in the same sector," so low purity is appropriate and not a
   defect.

This table converts an otherwise dispersed set of methods into a single
quantitative comparison and is the centrepiece "results" deliverable of
the project.

---

## 6. Limitations and honest disclosures

### 6.1 Residual data-quality artifacts in the anomalies output

`data/processed/anomalies.csv` contains four rows after the
`KNOWN_ISSUES.md` G-1 (resolved in Phase G, 2026-05-17) documents that the
original `data/bist/processed/anomalies.csv` contained 4 unhandled corporate
actions where yfinance `Adj Close` failed to back-adjust the corresponding
splits / bonus issues:

| Date | Ticker | Log-return | Pre price | Post price | Cause (audited) |
|---|---|---|---|---|---|
| 2024-08-01 | CCOLA | −2.38 | 828.44 | 76.65 | 10.81× bonus issue |
| 2024-09-09 | HEKTS | −1.05 | 11.76 | 4.13 | 2.84× bonus issue |
| 2022-09-01 | AYGAZ | +0.55 | 21.18 | 36.54 | 1.72× bonus issue (opposite sign) |
| 2021-04-30 | HEKTS | +0.37 | 1.62 | 2.33 | 1.45× bonus issue (opposite sign) |

A mathematically a ±50 %+ single-day "market move" of this magnitude is not
plausible. These artifacts contaminated (a) the log-return series for those
tickers on the affected days; (b) every pairwise correlation involving the
affected tickers (CCOLA's mean \|correlation\| with the rest of the universe
was **0.124 with the bug**, jumping to **0.361 after the mask** — a 3×
contamination-induced distortion that hid CCOLA's true relationship to the
Turkish consumer/conglomerate cluster).

**Fix shipped in Phase G.** A new `manual_anomaly_nulls` field on
`PreprocessingConfig` accepts `[ticker, "YYYY-MM-DD"]` overrides; the
`run_preprocessing` step sets the corresponding cells to NaN before
anomaly flagging. `config/settings.yaml` records the 4 corrections with
audit comments. After the fix: `anomalies.csv` is header-only (0 flagged
rows); CCOLA's true correlation structure is restored; MST hubs (KCHOL,
SISE, SAHOL) are unchanged. Downstream NaN handling verified across TE
(delta < 0.001 nats per masked cell), Glasso (3 rows dropped out of 1543),
and wavelets (fillna-zero at single-cell granularity is cosmetic).

### 6.2 Transfer-entropy permutation null is too liberal

Documented in `KNOWN_ISSUES.md` M-1. The 609 retained "significant" edges
include a non-trivial fraction whose significance reflects source
autocorrelation rather than information transfer. Replacing the permutation
null with an IAAFT-surrogate null (Schreiber & Schmitz 1996) preserving the
spectral content of the source is the recommended fix.

### 6.3 ESN PCA features leak history

Documented in `KNOWN_ISSUES.md` M-2. `PCA.fit_transform(returns_clean)` is
called once on the full panel before walk-forward CV begins. The reported
aggregate R² of 0.063 is therefore an upper bound on strictly-causal
performance. Per-fold R² already degrades to −0.155 by the final fold,
indicating that even the upward-biased number is not material; the
underlying conclusion (no short-horizon predictability) is robust.

### 6.4 Hardcoded parameters not exposed in YAML

Documented in `KNOWN_ISSUES.md` L-1 and `FUTURE_WORK.md` F-2. Several
non-trivial parameters (RMT noise-replacement method, Glasso α-grid,
wavelet family, SNN config, ESN config) remain as code defaults rather
than YAML entries. These should be hoisted to `config/settings.yaml`
before any production use.

### 6.5 Survivorship and IPO-recency bias in the universe

29 of 102 nominal universe tickers are dropped by the 90 % coverage filter,
and the drop is sectorally biased (9 Energy, 6 Technology). The analysis
universe over-represents established banking/conglomerate/industrial names
and under-represents recent IPOs. This is acknowledged in the proposal as
an accepted trade-off for the engineering scope, but the reporting in
the dashboard and report uses **"73 BIST-100 constituents with ≥90 %
coverage 2020-01-01 → 2026-03-01"** rather than "BIST-100" to avoid
overstating universe coverage.

### 6.6 Gaussianity assumptions

Several derived quantities use a Gaussian assumption on returns:
- Differential entropy reduction in §2.1 (lower bound under heavy tails)
- Gaussian MI lower bound in §2.2
- Conditional-independence interpretation of Glasso edges in §2.3

BIST returns exhibit excess kurtosis well above the Gaussian baseline of 0;
the assumption is a working approximation, not a verified property. The
empirical MI estimator in §2.2 does not rely on Gaussianity, which is why
the comparison `MI_emp / MI_gauss > 1.5` is informative as a heuristic
for non-Gaussian coupling (modulo the HEKTS data-quality caveat).

### 6.7 Pair-dislocation half-lives are long

Half-lives for the top dislocation candidates are 100–250 days
(`data/results/dislocation_candidates.csv`), too long for actual
mean-reversion trading at daily frequency. The pair-dislocation outputs
are presented as research indicators only, as specified in the project
proposal.

### 6.8 Unmerged SNN branch

The SNN module (commit `f0b78ca` on branch `arda/eee-analysis`) is
currently unmerged. Its accompanying `docs/SNN_Report.md` describes a
dashboard integration that does not exist in that commit. Recommendation:
either keep the branch unmerged and exclude SNN from the report
deliverable, or merge with the report-only fixes described in our
internal review (rewrite the Executive Summary to lead with the negative
Δ-Sharpe of −1.12, delete the fictional integration claims, delete the
20 orphan per-pair `.pt` files).

---

## 7. Reproducibility and engineering quality

- **Tests:** 87 passing under `uv run python -m pytest -q`. Test coverage:
  `src/analysis.py` (14), `src/clustering.py` (21), `src/preprocessing.py`
  (6), `src/rolling_correlation.py` (25), `src/pair_dislocation.py` (21).
  EEE-method modules (RMT, Glasso, wavelet, TE, RC) are not yet covered
  by unit tests — recommended future work (`FUTURE_WORK.md` F-4).
- **Determinism:** seeded RNGs (TE seed = 42; ESN seed = 42; SNN seed =
  42 on the unmerged branch); deterministic pipeline given fixed raw data.
- **Configuration:** single `config/settings.yaml` plus version-controlled
  `config/universes/bist100.csv`.
- **Caching:** Streamlit `@st.cache_data` on every loader in `app/utils.py`
  (37 loaders); heavy in-app computations (correlation, MST layout,
  rolling statistics, sector breakdown) cached via the same mechanism.
- **Lines of code:** ~7,742 across `src/` (12 modules), `app/` (6 files),
  `tests/` (5 files).

---

## 8. References (for inclusion in the final-report bibliography)

Carried over from the proposal (Mantegna 1999; Onnela et al. 2003; Bonanno
et al. 2003; Marti et al. 2020; Mantegna & Stanley 2000; Laloux et al.
1999; Tumminello et al. 2007), augmented by the information-theoretic and
EEE references invoked by the methods used in this report:

- Marčenko, V. A. & Pastur, L. A. (1967). *Distribution of eigenvalues for
  some sets of random matrices.* Math USSR-Sb, 1(4):457.
- Friedman, J., Hastie, T., Tibshirani, R. (2008). *Sparse inverse
  covariance estimation with the graphical lasso.* Biostatistics 9(3):432.
- Schreiber, T. (2000). *Measuring information transfer.* Phys. Rev. Lett.
  85, 461.
- Marschinski, R. & Kantz, H. (2002). *Analysing the information flow
  between financial time series.* Eur. Phys. J. B 30, 275.
- Schreiber, T. & Schmitz, A. (1996). *Improved surrogate data for
  nonlinearity tests.* Phys. Rev. Lett. 77, 635 (IAAFT).
- Daubechies, I. (1992). *Ten lectures on wavelets.* SIAM.
- Percival, D. B. & Walden, A. T. (2000). *Wavelet methods for time series
  analysis.* Cambridge University Press.
- Gençay, R., Whitcher, B., Selçuk, F. (2001). *Differentiating
  intraday seasonalities through wavelet multi-scaling.* Physica A.
- Rosso, O. A., Blanco, S., Yordanova, J., Kolev, V., Figliola, A.,
  Schürmann, M., Başar, E. (2001). *Wavelet entropy: a new tool for
  analysis of short-duration brain electrical signals.* J. Neurosci.
  Methods 105, 65–75.
- Jaeger, H. (2001). *The echo state approach to analysing and training
  recurrent neural networks.* GMD Report 148.
- Maass, W. (1997). *Networks of spiking neurons: the third generation
  of neural network models.* Neural Networks 10(9): 1659–1671.
- Neftci, E. O., Mostafa, H., Zenke, F. (2019). *Surrogate gradient
  learning in spiking neural networks.* IEEE Signal Processing Magazine
  36(6): 51–63.
- Tishby, N., Pereira, F. C., Bialek, W. (2000). *The information
  bottleneck method.* arXiv:physics/0004057.
- Engle, R. F. (1982). *Autoregressive conditional heteroscedasticity
  with estimates of the variance of United Kingdom inflation.*
  Econometrica 50(4): 987–1007.
- Forbes, K. J. & Rigobon, R. (2002). *No contagion, only interdependence:
  measuring stock market comovements.* Journal of Finance 57(5):
  2223–2261.
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*
  (2nd ed.). Wiley.

---

## Appendix A — Extra-analysis script and generated artifacts

`scripts/extra_analysis.py` reads existing pipeline outputs and writes:

| Output | Contents |
|---|---|
| `data/results/extra/it_summary.json` | Consolidated information-theoretic scalars (effective dimensionality, ΔH, MI summaries, nonlinear-coupling pair list) |
| `data/results/extra/mutual_information_matrix.parquet` | 73×73 empirical MI matrix (3-bin equal-frequency) |
| `data/results/extra/mi_pearson_comparison.csv` | Pair-level table: empirical MI, Gaussian MI lower bound, ratio, excess |
| `data/results/extra/wavelet_entropy.csv` | Per-ticker wavelet entropy (proxy from corr-weighted scale shares) |
| `data/results/extra/crisis_window_stats.csv` | ±60-day correlation stats around the three named events |
| `data/results/extra/methods_comparison.csv` | Cross-method MST-style comparison: edge count, sector purity, Jaccard vs raw MST |

All outputs are deterministic given the existing artifacts. Re-run with
`uv run python scripts/extra_analysis.py`.

---

*End of Technical Report.*
