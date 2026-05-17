---
title: StoNeCoAl
emoji: 📊
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.41.1
app_file: app/dashboard.py
pinned: false
license: mit
short_description: BIST-100 / S&P-500 / EEG correlation-network dashboard
---

# StoNeCoAl

**Stock Network Correlation Analysis** — a 12-stage pipeline and Streamlit
dashboard for analysing the correlation network of the BIST-100 (Borsa
Istanbul 100) equity universe, the S&P-500, and PhysioNet EEG.

> The YAML block above is read by [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-config-reference)
> when this repo is pushed as a Spaces remote. GitHub renders the body below as
> normal. See `docs/HUGGINGFACE_DEPLOY.md` for the full deploy runbook.

The pipeline downloads daily prices via yfinance, validates a sample against
isyatirimhisse, computes log returns, builds correlation/distance matrices,
clusters tickers and constructs minimum spanning trees, then layers on
RMT-denoised correlation, Graphical-LASSO partial correlation, wavelet
multi-scale correlation, transfer-entropy directed flow, and a spiking
neural network classifier on the top dislocation pairs.

The dashboard renders 40+ charts across 3 pages: **Market Overview**,
**Pair Analysis**, and **Cross-Market Comparison** (BIST vs S&P-500), with
an **EEE Analysis** sub-tab that surfaces the five advanced methods.

## Quick start

```bash
# One-time per machine: install Git LFS so the EEG bulk parquets pull on clone.
brew install git-lfs && git lfs install                              # macOS; use your distro's equivalent on Linux/Win

uv sync                                                              # install (Python 3.11+)
uv sync --extra snn                                                  # optional: pull torch + snntorch to enable the SNN stage
uv sync --extra eeg                                                  # optional: pull MNE to re-run the EEG pipeline locally
uv run python run_pipeline.py                                        # BIST (default; ~10-30 min, +~12 min if [snn] installed)
uv run python run_pipeline.py --config config/settings_sp500.yaml    # S&P-500 (~95 min including parallel TE on 12 cores)
uv run python run_pipeline_eeg.py                                    # EEG (optional rerun; ~15 min + ~5 min MNE download for cold cache)
uv run streamlit run app/dashboard.py                                # http://localhost:8501; sidebar selector flips between BIST / S&P / EEG
DASHBOARD_UNIVERSE=sp500 uv run streamlit run app/dashboard.py       # alternative: boot directly into the S&P universe
uv run python scripts/sp500_vs_bist.py                               # cross-market table (after both financial pipelines run)
uv run python -m pytest -q                                           # 120 tests (BIST + SNN + capability gates)
```

### Multi-universe data layout (Phase D)

The pipeline reads `market.market_id` from the active YAML and writes per-market
artifacts under `data/<market_id>/{raw,processed,results}/`. The default BIST
artifacts live in `data/bist/`. Running with `--config config/settings_sp500.yaml`
populates `data/sp500/`. The dashboard reads from `data/$DASHBOARD_UNIVERSE/`
(default `bist`) — set the env var to switch universes.

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
├── src/                      # Pipeline (11 modules; orchestrator: run_pipeline.py)
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
│   ├── dashboard.py          # Market Overview entry (~1189 lines, 6 sub-tabs)
│   ├── pair_analysis.py      # Pair Analysis page (5 sub-tabs)
│   ├── eee_analysis.py       # EEE Analysis tab (5 sub-tabs incl. Neuromorphic Signals)
│   ├── cross_market.py       # Cross-Market Comparison page (BIST vs S&P-500)
│   ├── universe_registry.py  # Registry of dashboard-switchable universes
│   ├── utils.py              # universe-aware cached loaders + section helpers
│   ├── chart_themes.py       # palette, sidebar theme switcher
│   └── chart_export.py       # PNG export hook
├── tests/                    # 100 tests (BIST + SNN combined)
├── config/
│   ├── settings.yaml             # BIST config (default)
│   ├── settings_sp500.yaml       # S&P-500 config
│   ├── settings_eeg.yaml         # EEG scaffold (Phase F)
│   ├── universes/bist100.csv     # 102-ticker BIST universe
│   ├── universes/sp500_full.csv  # 500-ticker S&P universe
│   └── universes/eeg_motor_left_right.csv
├── data/
│   ├── bist/{raw,processed,results}/    # BIST artifacts
│   ├── sp500/{raw,processed,results}/   # S&P artifacts
│   └── comparison_bist_vs_sp500.csv     # cross-market headline table
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
