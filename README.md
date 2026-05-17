# StoNeCoAl

**Stock Network Correlation Analysis** — a 12-stage pipeline and Streamlit
dashboard for analysing the correlation network of the BIST-100 (Borsa
Istanbul 100) equity universe.

The pipeline downloads daily prices via yfinance, validates a sample against
isyatirimhisse, computes log returns, builds correlation/distance matrices,
clusters tickers and constructs minimum spanning trees, then layers on
RMT-denoised correlation, Graphical-LASSO partial correlation, wavelet
multi-scale correlation, transfer-entropy directed flow, and a spiking
neural network classifier on the top dislocation pairs.

The dashboard renders 40+ charts across 2 pages: **Market Overview** and
**Pair Analysis**, with an **EEE Analysis** sub-tab that surfaces the five
advanced methods.

## Quick start

```bash
uv sync                                  # install core deps (Python 3.11+)
uv sync --extra snn                      # optional: add torch + snntorch for the SNN stage
uv run python run_pipeline.py            # ~10–30 min end-to-end (+~12 min if [snn] installed)
uv run streamlit run app/dashboard.py    # http://localhost:8501
uv run python -m pytest -q               # 96 passed, 3 skipped (99 total; 3 skips need torch)
```

## Pipeline at a glance

```
yfinance ─► raw prices ─► coverage filter ─► log returns ─► validation
                                                  │
              ┌───────────────────────────────────┼───────────────────────────────────┐
              ▼                                   ▼                                   ▼
       descriptive stats                  Pearson correlation                 anomaly detection
       summary_stats.parquet              pearson_corr.parquet                anomalies.csv
                                                  │
                ┌─────────────────┬───────────────┼────────────────┬─────────────────┐
                ▼                 ▼               ▼                ▼                 ▼
          hierarchical      MST + node       distance         rolling stats     pair dislocation
          clustering        metrics          matrix           (60/120/252)      (Z-score, half-life)
          dendrogram        mst_edges                         rolling_*.parquet dislocation_candidates
                                                                                       │
                ┌────────────────┬────────────────┬─────────────────┬─────────────────┘
                ▼                ▼                ▼                 ▼                  ▼
          RMT denoising   Glasso partial    wavelet (db4)      transfer entropy   SNN [optional, [snn] extra]
          MP bounds       correlation       7 scales           shuffle null       LIF + surrogate-gradient
          denoised_corr   precision_matrix  wavelet_*          te_*               snn_metrics, snn_signals/*
```

## Project tree

```
stonecal/
├── src/                      # Pipeline (12 modules; orchestrator: run_pipeline.py)
│   ├── config.py             # YAML loader, dataclasses, PROJECT_ROOT
│   ├── data_acquisition.py   # yfinance fetch
│   ├── data_validation.py    # cross-check vs isyatirimhisse
│   ├── preprocessing.py      # coverage filter, log returns, anomalies
│   ├── analysis.py           # descriptive stats, Pearson corr, distance matrix
│   ├── clustering.py         # hierarchical clustering + MST
│   ├── rolling_correlation.py# windowed market/sector/pair stats
│   ├── pair_dislocation.py   # spread / Z-score / half-life screening
│   ├── rmt_denoising.py      # Marchenko–Pastur bounds + reconstruction
│   ├── partial_correlation.py# GraphicalLassoCV
│   ├── wavelet_analysis.py   # PyWavelets db4, scales 1–7
│   ├── transfer_entropy.py   # binned discrete TE with shuffle null
│   └── snn_signals.py        # spike-coded LIF classifier (optional [snn] extra)
├── app/
│   ├── dashboard.py          # Market Overview entry (~1189 lines, 6 tabs)
│   ├── pair_analysis.py      # Pair Analysis page (5 tabs)
│   ├── eee_analysis.py       # EEE Analysis tab (5 sub-tabs incl. Neuromorphic Signals)
│   ├── utils.py              # 40 cached loaders + section helpers
│   ├── chart_themes.py       # palette, sidebar theme switcher
│   └── chart_export.py       # PNG export hook
├── tests/                    # 6 files, 99 tests (96 pass + 3 torch-skip)
├── config/
│   ├── settings.yaml         # main config (incomplete by design)
│   └── universes/bist100.csv # 102-ticker universe
├── data/
│   ├── raw/                  # yfinance dump
│   ├── processed/            # post-filter, log returns
│   └── results/              # everything the dashboard reads (incl. snn_*)
├── docs/                     # this directory's index, see below
├── CLAUDE.md                 # auto-loaded by Claude Code sessions
└── pyproject.toml            # uv / hatchling
```

## Documentation

| Doc | Read it for |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Conventions and pitfalls (auto-loaded by Claude). |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System shape, glossary. |
| [`docs/PIPELINE_REFERENCE.md`](docs/PIPELINE_REFERENCE.md) | Per-module behaviour and outputs. |
| [`docs/EEE_METHODS.md`](docs/EEE_METHODS.md) | Math behind the advanced methods. |
| [`docs/SNN_Report.md`](docs/SNN_Report.md) | Long-form Spiking Neural Network writeup. |
| [`docs/UI_REFERENCE.md`](docs/UI_REFERENCE.md) | Page × tab × chart inventory. |
| [`docs/DATA_ARTIFACTS.md`](docs/DATA_ARTIFACTS.md) | Schemas of every artifact. |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Install / run / deploy / troubleshoot. |
| [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) | Open MED/LOW bugs. |
| [`docs/FUTURE_WORK.md`](docs/FUTURE_WORK.md) | Roadmap. |

## License

See [`LICENSE`](LICENSE).
