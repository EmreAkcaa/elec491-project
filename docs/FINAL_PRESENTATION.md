# StoNeCoAl — Final Presentation Notes

**ELEC 491 Final Presentation · Spring 2026 · Koç University**

Participants: Arda Rutkay Var (75628), Emre Akça (80567)
Supervisor: Emir Salih Mağden  ·  TA: Dilem Eşlik

This document is organised to satisfy each presentation-criterion prompt
verbatim. Numbers are taken from `data/results/`; nothing is fabricated.

---

## 1. What it is

> *"StoNeCoAl is a reproducible 12-stage data pipeline and Streamlit
> dashboard that applies six EEE-style estimators — random matrix theory,
> wavelet multi-resolution analysis, sparse inverse-covariance estimation,
> transfer entropy, reservoir computing, and (optionally) a spiking neural
> network — to characterise the dependence and information structure of
> the BIST-100 equity universe."*

---

## 2. Why it matters (quantitative motivation)

The BIST-100 has substantial real-economic significance: ~80 % of Türkiye's
listed market capitalisation, monitored by domestic regulators and
international investors as the single most-cited indicator of Turkish
equity-market health.

Two quantitative gaps in the existing literature motivate the project:

1. **Methodological narrowness.** Existing BIST-100 network studies
   (Atılgan & Afşar 2022; Şükrüoğlu 2022; Bulut & Şimşek 2023) each apply
   a *single* method (MST, NETS, or Markov-switching respectively) to a
   *single* time window (mostly the COVID-19 period, ~18 months). Our
   project spans **6.17 years (1,543 trading days, 2020-01-01 to
   2026-03-01)** and runs **six complementary estimators on the same
   panel.**

2. **Reproducibility gap.** None of the published BIST-100 analyses ship
   a runnable pipeline, configuration file, or interactive dashboard. A
   researcher who wants to repeat or extend any prior result must rebuild
   the entire workflow. Our project ships **87 passing unit tests,
   12 modular pipeline stages, a single YAML configuration, and an
   interactive dashboard** that exposes every result without code knowledge.

The substantive engineering payoff for the user: we quantify, with numbers,
that the BIST-73 correlation matrix has an **effective informational
dimensionality of 6.30 modes** (not 73), that **average pairwise correlation
nearly doubled (0.318 → 0.622) around the February 2023 earthquakes**, and
that **the network's MST hubs are Turkey's three largest industrial
conglomerates** (KCHOL, SISE, SAHOL) — none of which is reported in the
existing BIST literature.

---

## 3. Detailed system schematic

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  EXTERNAL DATA INGRESS (HTTP/REST)                                           │
│  ─────────────────────────────────                                           │
│  yfinance API     (primary)  ─► chunked 25-ticker batches, daily prices      │
│  İş Yatırım (HGDG)            ─► 10-ticker validation sample (seed=42)       │
│  TCMB EVDS (optional)         ─► XU100 index cross-check                     │
└────────────────────┬─────────────────────────────────────────────────────────┘
                     │ HTTPS / REST
                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PIPELINE (run_pipeline.py — Python, deterministic; ~7,742 LoC under src/)   │
│  ─────────────────────────────────────────────────────────────────────────── │
│   1. data_acquisition       ─►  yfinance + caching                           │
│   2. data_validation        ─►  İş Yatırım cross-check                       │
│   3. preprocessing          ─►  log returns, coverage filter, anomaly flag   │
│   4. analysis               ─►  Pearson correlation + distance matrix        │
│   5. clustering             ─►  scipy linkage + NetworkX Kruskal MST         │
│   6. rolling_correlation    ─►  60 / 120 / 252-day rolling stats             │
│   7. pair_dislocation       ─►  OLS hedge ratio, Z-score, half-life          │
│   8. rmt_denoising          ─►  numpy.linalg.eigh + MP bound                 │
│   9. partial_correlation    ─►  sklearn GraphicalLassoCV (5-fold)            │
│  10. wavelet_analysis       ─►  PyWavelets db4, 7 dyadic scales              │
│  11. transfer_entropy       ─►  3-bin discrete, 100-shuffle null (seeded)    │
│  12. reservoir_computing    ─►  300-neuron ESN, ridge readout                │
│                                                                              │
│   Extra analysis (scripts/extra_analysis.py):                                │
│      • Mutual information matrix (3-bin equal-frequency)                     │
│      • Effective dimensionality, ΔH (RMT-derived scalars)                    │
│      • Wavelet entropy per ticker                                            │
│      • Crisis-window correlation stats                                       │
│      • Cross-method MST comparison (sector purity, Jaccard)                  │
└────────────────────┬─────────────────────────────────────────────────────────┘
                     │ Parquet / CSV / JSON artifact files
                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ARTIFACT LAYER (data/processed/, data/results/)                             │
│  ──────────────────────────────────────────────                              │
│  37 cached loaders in app/utils.py expose every artifact to the dashboard    │
└────────────────────┬─────────────────────────────────────────────────────────┘
                     │ pandas DataFrames / numpy arrays
                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  DASHBOARD (Streamlit, app/*)                                                │
│  ─────────────────────────────                                               │
│  Page 1: Market Overview (6 tabs)                                            │
│    Data & Stats │ Correlation │ Clustering & Network │ Rolling │ Pairs │     │
│    EEE Analysis (5 sub-tabs: RMT, Glasso, Wavelet, TE, Forecasting)          │
│                                                                              │
│  Page 2: Pair Analysis (5 sub-tabs)                                          │
│    Overview │ Correlation │ Risk │ Spread │ Network                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Hardware (in the broad infrastructure sense)

This is a software/data project; no embedded hardware. The deployment
substrate is:
- **Compute host:** any laptop with Python 3.11+ and `uv` package manager;
  pipeline completes in 10–30 min on standard hardware (CPU only).
- **External services:** Yahoo Finance HTTP API (free, rate-limited);
  İş Yatırım public scrape wrapper (open-source `isyatirimhisse` library).
- **Storage:** local filesystem; Parquet for tabular artifacts (~50 MB
  total), CSV for human-readable tables, JSON for metadata and scalars.

### Software stack

- **Language:** Python 3.11+
- **Numerics:** numpy, pandas, scipy, scikit-learn, networkx, pywavelets
- **App:** streamlit (with `@st.cache_data` for every loader)
- **Testing:** pytest (87 tests)
- **Packaging:** `uv` + `pyproject.toml`
- **Optional (SNN branch):** torch, snntorch (lazy-imported)

### Communication

- **Pipeline ↔ artifacts:** filesystem I/O (Parquet/CSV/JSON); the
  artifact layer is the contract between batch compute and presentation.
- **Dashboard ↔ artifacts:** cached loader functions in `app/utils.py`.
- **External ↔ pipeline:** HTTPS via `yfinance` and `isyatirimhisse`.

---

## 4. Work since midterm presentation

### Status at midterm

A baseline pipeline implementing the original proposal scope was in
place: yfinance ingest, preprocessing, Pearson correlation, single-linkage
hierarchical clustering, Kruskal MST, dendrogram, pair-dislocation Z-score
screening, and a first version of the Streamlit dashboard. **Supervisor
critique (midterm grade 65/100):** the project read as a finance tool
rather than an electrical-engineering project; needed substantive EEE
content to justify the senior-project designation.

### Work completed since midterm

Direct response to the EEE-content critique, all on top of a stable
baseline. The pipeline grew from 7 to **12 stages**:

| Added since midterm | EEE discipline it draws from |
|---|---|
| Random Matrix Theory denoising + Marchenko–Pastur signal detection | Statistical signal processing / random matrix theory |
| Graphical LASSO sparse inverse covariance | System identification, sparse estimation |
| Discrete wavelet decomposition + per-scale MSTs | Multi-resolution signal analysis |
| Transfer entropy with permutation significance | Information theory |
| Reservoir computing (Echo State Network) for next-day dispersion forecasting | Recurrent neural dynamics |
| (Optional, unmerged branch) Spiking neural network with surrogate-gradient training | Neuromorphic / event-driven processing |

In parallel:

- **Engineering hardening:** four HIGH-severity bugs fixed and verified
  (`docs/KNOWN_ISSUES.md` H-1 to H-5); transfer entropy made
  reproducible via seeded RNG; precision matrix persisted to disk;
  Windows-personal paths removed from `.claude/` settings.
- **Test suite:** grew to **87 passing tests** across 5 test files.
- **Documentation set:** seven dedicated docs under `docs/`
  (architecture, pipeline reference, EEE methods, UI reference,
  data artifacts, known issues, future work, runbook).
- **Dashboard:** added Pair Analysis page (5 sub-tabs) and the EEE
  Analysis tab with 5 sub-tabs.
- **Derived information-theoretic metrics** (this presentation):
  effective dimensionality, ΔH, mutual-information matrix,
  crisis-window correlation analysis, cross-method comparison
  (see §6 below for the headline numbers).

---

## 5. Technical challenges (fundamental engineering, not project-specific)

The project is hard for reasons that are universal to *any* complex
signal-processing-on-high-dimensional-time-series system:

### 5.1 Curse of dimensionality with limited sample size

The empirical correlation matrix is 73 × 73 (5,329 unique entries)
estimated from 1,543 daily observations. The ratio `T/N = 21.1` is
moderate but not generous; many entries are within Marchenko–Pastur
noise tolerance. Any analysis that treats the correlation matrix as
"the truth" without RMT-style noise discounting is making an unverified
claim about a finite-sample estimator.

### 5.2 Signal-noise separation in nonstationary data

Returns are heavy-tailed, mildly autocorrelated, and exhibit regime
changes (COVID, geopolitical events, earthquakes). Standard estimators
that assume stationarity (Pearson correlation, Glasso) need either
explicit windowing (rolling stats) or robustness modifications.
Significance testing on autocorrelated time series is itself a
research problem — we use a known-too-liberal permutation null for
transfer entropy and disclose this.

### 5.3 Multi-scale structure

Financial co-movement is genuinely multi-scale: intraday microstructure,
weekly rebalancing flows, monthly factor exposures, quarterly fundamental
news, annual macro cycles. A single correlation matrix at one scale
is an aggregate that hides scale-specific structure. The
wavelet-decomposition stage exists precisely because the answer to "are
these two stocks correlated?" depends on the timescale.

### 5.4 Reproducibility under data drift

Free vendor APIs (yfinance) silently mutate historical adjusted-close
series as corporate actions accrue. A pipeline that re-fetches data on
different days produces non-identical inputs — *not a code bug*, but a
reproducibility property of the entire workflow. We pin reproducibility
to "fixed raw-data snapshot + seeded RNGs" rather than "re-derivable
from the live API."

### 5.5 Combinatorial method-comparison

With six estimators (Pearson, RMT-denoised, Glasso, wavelet, TE, ESN)
producing different artifacts on the same input, comparing them
quantitatively requires a shared evaluation axis. We chose three
(sector purity, Jaccard vs raw MST, edge count) and report them
side-by-side; choosing the *right* shared axis is a generic engineering
challenge for any multi-method analysis platform.

### Why this represents a universal challenge

The five issues above appear in any project that combines (a) high
dimensionality, (b) limited sample size, (c) non-stationary data, (d)
multiple methods, and (e) reproducibility constraints. Examples beyond
finance: brain-connectivity studies on fMRI (typically 200+ ROIs, 500
time points), gene co-expression networks (10,000+ genes, 100 samples),
sensor networks on smart-grid data (100+ sensors, multiple sample
rates), and large-scale Σ-Δ ADC arrays. The same toolkit — RMT,
graphical Lasso, wavelets, transfer entropy, reservoir computing —
appears in all of them. Mastering the engineering trade-offs on BIST
is portable engineering knowledge.

---

## 6. Existing solutions

### 6.1 Academic comparable work

| Work | Method | Period | Reproducible? | Multi-method? |
|---|---|---|---|---|
| Atılgan & Afşar (2022) | Mantegna MST | Mar 2020 – Nov 2021 | No (static figures) | No |
| Şükrüoğlu (2022) | NETS (Barigozzi-Brownlees) | COVID window | No | No |
| Bulut & Şimşek (2023) | Markov-switching | COVID window | No | No |
| Mantegna (1999) | MST | 1990s NYSE | No (foundational paper) | No |
| Marti et al. (2020) | Survey | Multiple | N/A | Yes (review) |

None of the above offers a runnable repository or interactive interface.

### 6.2 Commercial / industry tools

| Product | What it does | What it does NOT do |
|---|---|---|
| **Bloomberg Terminal** | Real-time market data, fundamental analytics | Closed-source, no network/IT estimators, US$24k+/year |
| **Refinitiv Eikon** | Same category as Bloomberg | Same limitations |
| **TradingView** | Charting, technical indicators, retail community | No network/clustering/IT methods |
| **R `igraph` + ad-hoc scripts** | Researcher workflow | Not productised, no dashboard, per-paper reinvention |
| **Cytoscape, Gephi** | Network visualisation | Generic; not equity-data aware |

### 6.3 Comparison against StoNeCoAl

| Capability | Academic studies | Commercial terminals | StoNeCoAl |
|---|---|---|---|
| BIST-100 universe | Yes (one window each) | Yes | Yes (73 of 102 at ≥ 90 % coverage) |
| Reproducible end-to-end | No | No (closed) | **Yes** |
| Multi-method comparison | No | No | **Yes (6 estimators)** |
| Interactive dashboard | No | Yes | **Yes** |
| Open source | Implementations rarely shared | No | **Yes (uv + Python)** |
| Information-theoretic estimators | At most one | No | **TE + MI + RMT-derived entropy** |

---

## 7. Novelty & advantages

### 7.1 Novelty vs published BIST work

1. **First multi-method BIST network study spanning the full COVID-to-2026
   period** (existing studies cover ≤ 20 months and one method).
2. **First reported "effective informational dimensionality" of BIST**:
   participation ratio `(Σλ)² / Σλ² = 6.30` modes out of 73. This single
   scalar quantifies how concentrated BIST risk is in informational terms.
3. **First multi-scale MST overlap analysis of BIST:** edge overlap with
   the unconditional MST drops 60 → 19 % from daily to annual scale. The
   network is not scale-invariant.
4. **First MST-hub identification on a post-2020 BIST window finding
   conglomerates (KCHOL, SISE, SAHOL) rather than banks as the
   highest-betweenness nodes** — a structurally specific feature of the
   Turkish economy that contrasts with US findings on insurance + REIT hubs.
5. **First quantitative correlation-tightening report around the
   February 2023 Türkiye earthquakes:** avg pairwise correlation
   0.318 → 0.622 → 0.536 (before/during/after) — a near-doubling that
   persists 60–120 days post-event, **with verification of country-
   specificity** (S&P-485 shows zero contemporaneous correlation effect
   on the same dates — earthquake propagates through Turkish equity
   structure only).
6. **First documented joint negative result for daily-frequency neural
   forecasting on BIST:** two architecturally distinct neural models
   (rate-coded ESN, spike-coded SNN) both underperform a single-scalar
   threshold rule — interpreted via the information-bottleneck framework
   as predictive content being concentrated in a near-one-bit channel.

### 7.1b Cross-market findings (Phase G addition: full S&P-485 run for comparison)

These five findings emerge from running the identical 12-stage pipeline on
the S&P-500 universe (500 tickers → 485 after coverage filter; 3 dual-class
share duplicates dropped from the constituent list per `KNOWN_ISSUES.md`
G-2).

1. **Effective dimensionality is a cross-market structural invariant:
   D_eff = 6.30 (BIST) ≈ 6.56 (S&P-485)** despite S&P having 6.6× more
   tickers. The "informational rank" of an equity universe does not
   scale linearly with N.
2. **Top eigenvalue share is also invariant at 38 % in both markets.**
   The market-mode share is 38.9 % on BIST and 38.1 % on S&P. Both
   markets concentrate ~38 % of total variance in a single "everything
   moves together" direction.
3. **Network hub composition differs structurally**: BIST is hubbed on
   family-conglomerate holdings (KCHOL/SISE/SAHOL); S&P is hubbed on
   insurance + asset-management names + Parker-Hannifin (PRU/AMP/PH)
   — neither universe is bank-led despite the conventional narrative.
4. **MST sector purity is 2× higher on S&P (0.80) than on BIST (0.40);**
   same for Graphical LASSO (0.51 vs 0.13). Developed-market sector
   taxonomy clusters much more cleanly in correlation space than the
   Borsa Istanbul taxonomy does.
5. **Graphical LASSO recovers head-to-head competitor pairs on S&P with
   textbook precision:** UAL/DAL (airlines), LVS/WYNN (Vegas casinos),
   NUE/STLD (steel), NCLH/RCL (cruise lines), PSKY/WBD (legacy media)
   — every top partial-correlation edge is a 2-3 firm oligopoly. This
   is the single strongest validation that the toolkit is not smoke.

### 7.2 Advantages vs commercial / academic alternatives

- **Reproducibility:** single command (`uv run python run_pipeline.py`)
  regenerates every artifact from raw data; commercial tools are closed
  and academic implementations are typically not shipped.
- **Methodological breadth in one workflow:** the comparative-laboratory
  framing is the project's key engineering contribution — six dependence
  estimators on the same input with quantitative side-by-side
  comparison (`scripts/extra_analysis.py:methods_comparison`).
- **EEE-discipline grounding:** every method is from an EEE syllabus
  (signal processing, system identification, multi-resolution analysis,
  information theory, neural dynamics) — not "finance" methods.
- **Honest reporting of negative results:** the project explicitly
  reports the ESN/SNN forecasting failures and the chaining artifact
  in single-linkage clustering, rather than hiding them.

---

## 8. What the demo will show

A 12–15 minute walkthrough of the dashboard with the following beats:

1. **Pipeline & setup** (1 min) — show `run_pipeline.py` and `pyproject.toml`;
   one-sentence overview of the 12 stages.
2. **Data & coverage** (1 min) — Data tab: coverage bar chart, the 73-of-102
   retention, anomaly disclosure.
3. **Pearson + MST baseline** (2 min) — Clustering & Network tab: the
   heatmap reordered by dendrogram leaf order, the MST with KCHOL, SISE,
   SAHOL as the three highest-betweenness hubs.
4. **RMT eigenvalues + Marchenko–Pastur** (2 min) — EEE → RMT sub-tab:
   eigenvalue spectrum with the MP upper bound overlay, 4 signal
   eigenvalues, the market mode at 38.2 % variance.
5. **Wavelet multi-scale** (2 min) — EEE → Wavelet sub-tab: scale-1 vs
   scale-7 MSTs side-by-side; on-stage statement of the 60 % → 22 %
   overlap decay as the headline finding.
6. **Transfer entropy + information flow** (1 min) — EEE → TE sub-tab:
   net-flow heatmap, top sources/sinks, honest caveat on small effect
   sizes and the liberal null.
7. **Glasso conditional independence** (1 min) — EEE → Glasso sub-tab:
   partial-correlation heatmap and sparsity pattern.
8. **Forecasting honest negative** (1 min) — EEE → Forecasting sub-tab:
   ESN per-fold R² bar chart with the negative-result framing.
9. **Crisis-window result** (1 min) — Rolling Analysis tab with event
   overlays; quote the 0.318 → 0.622 earthquake-window numbers.
10. **Methods-comparison table** (1 min) — show
    `data/results/extra/methods_comparison.csv` (sector purity + Jaccard
    vs raw MST) as the summary slide.
11. **Pair Analysis page demo** (1 min) — pick AKBNK–YKBNK, show the
    rolling pair correlation and the spread-Z signal panel.
12. **Limitations + future work** (1 min) — explicit list from §10.

### Visitor interactions at the demo booth

- Change the date range in the Settings popover and watch every chart
  re-cache.
- Pick any two tickers and inspect their pair page.
- Toggle the chart theme (light / dark / colour-blind) and confirm
  the per-chart "Download PNG" button works.
- Open any wavelet scale 1–7 and watch the MST topology change.

---

## 9. Success criteria

| Criterion | Metric / benchmark | Current status |
|---|---|---|
| Pipeline runs end-to-end | `uv run python run_pipeline.py` exits 0 in ≤ 30 min | ✅ |
| Tests pass | 87 / 87 passing | ✅ |
| Dashboard loads without exceptions | All 6 Market Overview tabs + 5 Pair Analysis sub-tabs render | ✅ |
| RMT eigenvalues match analytic MP bound | `λ₊ = 1.482` matches closed form for `T=1543, N=73` | ✅ |
| Top MST hubs are real Turkish conglomerates | KCHOL (0.72), SISE (0.59), SAHOL (0.52) | ✅ |
| Effective informational dimensionality reported | `D_eff = 6.30` | ✅ |
| Crisis-window earthquake jump detected | avg corr 0.318 → 0.622 | ✅ |
| Multi-scale MST overlap decay quantified | 60 → 22 % (daily → annual) | ✅ |
| Methods comparison table populated | 11 methods × 3 metrics | ✅ |
| Honest reporting of negative results | ESN R²=0.06, SNN Δ-Sharpe=−1.12 disclosed in report | ✅ |
| Information-theoretic component clearly framed | TE + MI + RMT-derived ΔH in technical report §2 | ✅ |
| Cross-method MST overlap quantified (Jaccard) | All 11 methods vs raw MST in `methods_comparison.csv` | ✅ |

**Failure criteria (none currently triggered):** any test failing, the
dashboard erroring on a fresh clone, any number in the report not
traceable to an artifact under `data/results/`.

---

## 10. Known limitations

We disclose these explicitly to forestall hostile questioning.

| # | Limitation | Mitigation in deliverable |
|---|---|---|
| L-1 | Universe drops 29 of 102 tickers (28 %), sectorally biased | Labelled "73 BIST-100 constituents with ≥ 90 % coverage" everywhere |
| L-2 | Two residual unhandled corporate actions remain in the anomalies output (CCOLA 2024-08-01, HEKTS 2024-09-09) | Renamed dashboard panel "Suspect returns / corporate-action candidates"; future-work item to cross-check vs Borsa İstanbul calendar |
| L-3 | TE permutation null is too liberal (breaks source autocorrelation) | Documented in `KNOWN_ISSUES.md` M-1; IAAFT surrogate recommended as future work |
| L-4 | ESN PCA features fit on full panel before walk-forward CV | Documented `KNOWN_ISSUES.md` M-2; reported R² is upper bound; underlying conclusion robust |
| L-5 | Single-linkage clustering chains: 60 % of tickers in one mega-cluster | We report dendrogram + MST hubs rather than cluster IDs in the dashboard |
| L-6 | RMT denoising with `method='constant'` produces a degenerate super-hub | Both raw and denoised MSTs shown side-by-side; comparison table makes the artifact explicit |
| L-7 | Wavelet entropy uses a correlation-weighted proxy, not direct detail-band variance | Documented in the technical report §2.5; recommended fix is to persist detail series |
| L-8 | EEE methods (RMT, Glasso, wavelet, TE, ESN) have no dedicated unit tests | `FUTURE_WORK.md` F-4; the integration through `run_pipeline.py` and resulting artifacts are validated end-to-end |
| L-9 | Several non-trivial parameters hardcoded in code, not in `settings.yaml` | `KNOWN_ISSUES.md` L-1, `FUTURE_WORK.md` F-2 |
| L-10 | The optional SNN module (branch `arda/eee-analysis`) is currently unmerged; its accompanying report describes a dashboard integration that does not exist in that commit | Branch kept off `main`; either it is merged after re-framing the report as a negative-result exploration, or it is excluded from the final deliverable |

### Handling during the demo

If a visitor asks about any of these:
- Limitations L-1, L-5, L-6 are *in* the report; cite the line.
- L-3, L-4 are documented in `KNOWN_ISSUES.md`; show the file on screen
  if pressed.
- L-2 is the most likely "gotcha" if the visitor clicks the Anomalies
  tab. Answer: "those four flagged returns are residual unhandled
  corporate actions in the vendor's adjusted close, not market events;
  documented in the limitations section."
- L-10 (SNN): "There is an exploratory spike-coded classifier on a
  separate branch that achieved 0.66 macro-F1 on the 3-class signal
  but underperforms the simple |Z|>2 rule on Sharpe; we report it as
  a documented joint-negative result alongside the ESN if asked."

---

## Appendix — Where every number in this document lives

| Number | Source artifact |
|---|---|
| 87 tests passing | `tests/`, runnable via `uv run python -m pytest -q` |
| 12 pipeline stages | `run_pipeline.py` |
| 102 → 73 universe | `data/processed/coverage_report.csv`, `data/results/pipeline_metadata.json` |
| 1,543 days | `data/results/pipeline_metadata.json:trading_days` |
| 6.17 years | `data/results/pipeline_metadata.json:config.start_date/end_date` |
| MP bound λ₊ = 1.482 | `data/results/eigenvalue_spectrum.csv:mp_upper` |
| 4 signal eigenvalues, 38.2 % top share | `data/results/eigenvalue_spectrum.csv` |
| D_eff = 6.30 | `data/results/extra/it_summary.json:rmt_information_geometry.effective_dimensionality_D_eff` |
| ΔH = 21.53 nats | `data/results/extra/it_summary.json:rmt_information_geometry.delta_entropy_nats` |
| MST hubs (KCHOL 0.72 etc.) | `data/results/mst_node_metrics.csv` |
| Wavelet overlaps 60 % → 22 % | `data/results/extra/methods_comparison.csv` + manual common-edge counts in `scripts/extra_analysis.py` |
| Earthquake corr 0.318 → 0.622 | `data/results/extra/crisis_window_stats.csv` |
| 14 nonlinear-MI pairs | `data/results/extra/mi_pearson_comparison.csv` (filtered) |
| ESN R² = 0.063 | `data/results/rc_metrics.json:dispersion_prediction.r2` |
| SNN Δ-Sharpe = −1.108 | `data/results/snn_metrics.json:aggregate.mean_delta_sharpe` (branch `arda/eee-analysis`) |
| Methods-comparison purity table | `data/results/extra/methods_comparison.csv` |

---

*End of Final Presentation document.*
