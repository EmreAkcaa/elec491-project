# RUNBOOK

Operational guide for installing, running, and deploying StoNeCoAl.

## Prerequisites

- **Python 3.11+** (`pyproject.toml:requires-python = ">=3.11"`)
- **uv** for dependency management ([install instructions](https://docs.astral.sh/uv/))
- ~2 GB free disk for `data/` artifacts
- Internet access for the data-acquisition stage (yfinance + isyatirimhisse)

## Install

```bash
git clone <repo>
cd Repo_2          # or whatever you've named the directory
uv sync            # creates .venv and installs all deps from uv.lock
```

The lock file pins exact versions — `uv sync` is deterministic. If you need
the dev extras (only `pytest` currently):

```bash
uv sync --extra dev
```

### Optional: SNN extra (torch + snntorch)

The SNN module (`src/snn_signals.py`, stage 12 of the pipeline, surfaced
under the dashboard's **EEE Analysis → Neuromorphic Signals** sub-tab)
needs PyTorch and snntorch. They are not installed by default because
they pull ~700 MB of binaries that most contributors don't need; the
pre-computed SNN artifacts under `data/results/snn_*` let the dashboard
render the sub-tab without torch installed.

To enable training / re-inference locally:

```bash
uv sync --extra snn
```

Then `run_pipeline.py` will execute stage 12; without the extra, the
stage logs a warning and the rest of the pipeline still completes
(`try/except ImportError` in `run_pipeline.py`).

## Run the full pipeline

### BIST 100 (default)

```bash
uv run python run_pipeline.py
```

Writes to `data/bist/{raw,processed,results}/`.

### S&P 500

```bash
uv run python run_pipeline.py --config config/settings_sp500.yaml
```

Writes to `data/sp500/{raw,processed,results}/`. ~95 min wall time including
parallel TE on 12 cores (485 surviving tickers post-coverage filter).

### EEG (scaffold)

```bash
uv sync --extra eeg
uv run python run_pipeline_eeg.py
```

Writes to `data/eeg_motor_left_right/{processed,results}/`. Not yet wired
into the dashboard's universe selector (the SNN and Pair Analysis sub-tabs
assume stock semantics).

### Cross-market refresh

After running both BIST and S&P pipelines:

```bash
uv run python scripts/sp500_vs_bist.py
```

Regenerates `data/comparison_bist_vs_sp500.csv` — the Cross-Market dashboard
page's headline-numbers data source.

### Timings

Stages execute in the order documented in
[`ARCHITECTURE.md`](ARCHITECTURE.md). BIST total wall time ~10–30 min depending
on network and CPU (~12 additional minutes for SNN training if the `[snn]`
extra is installed). The slowest stages are **transfer entropy**
(`O(N² · S · T)` ≈ 10k pair-shuffles, typically 5–10 min on BIST, ~80 min on
S&P) and **SNN training** (~12 min CPU; subsequent re-inference is ~30 s when
the cached `universal.pt` exists).

Logging goes to stdout at INFO; redirect to a file if you want a record:

```bash
uv run python run_pipeline.py 2>&1 | tee run.log
```

## Run individual stages

Each `src/<module>.py` has a `if __name__ == "__main__"` block that loads
the config and runs that stage's `run_*` entry point. Use this when you've
modified one module and don't want to redo the whole chain:

```bash
uv run python -m src.preprocessing            # re-run preprocessing
uv run python -m src.partial_correlation      # re-run Glasso only
uv run python -m src.transfer_entropy         # slow — 5–10 min
uv run python -m src.snn_signals              # requires `[snn]` extra; ~12 min
```

Stages have data dependencies; check `data/processed/` and `data/results/`
exist for the inputs the stage needs (see PIPELINE_REFERENCE.md per-stage
"Reads").

## Run the tests

```bash
uv run python -m pytest -q
```

Expect **96 passed, 3 skipped** in ~10 seconds when the `[snn]` extra is
not installed (the 3 skips are the torch-dependent SNN tests). With
`uv sync --extra snn` all 99 tests run. There are no slow / integration
markers.

## Launch the dashboard

```bash
uv run streamlit run app/dashboard.py
```

Opens at `http://localhost:8501`. The dashboard reads exclusively from
`data/<universe>/processed/` and `data/<universe>/results/` — it doesn't
touch the network.

### Switching universes

The sidebar has a **Dataset** dropdown when more than one universe has a
populated `pipeline_metadata.json` on disk. BIST and S&P caches coexist —
switching back to the previous universe is a warm cache hit, not a re-read.

To pick the boot-time default without using the sidebar (e.g. for sharing
a deep-linked URL), set the env var:

```bash
DASHBOARD_UNIVERSE=sp500 uv run streamlit run app/dashboard.py    # boots into S&P
DASHBOARD_UNIVERSE=bist  uv run streamlit run app/dashboard.py    # boots into BIST (default)
```

The browser tab title is set once from this env var; the in-page header
chip updates immediately when the user changes the sidebar selector.

### Cross-Market page

Top-level nav has three pages: **Market Overview**, **Pair Analysis**,
**Cross-Market**. The Cross-Market page reads from both universes
simultaneously (not via the sidebar selector) and shows the headline
side-by-side comparison: D_eff invariance (~6.5 across both), MST sector-purity
divergence (BIST 0.40 vs S&P 0.80), eigenvalue spectra, crisis windows,
top hubs, and top dislocation candidate per universe.

To kill an orphan Streamlit process on Windows:

```bash
taskkill //F //IM streamlit.exe
```

On macOS/Linux:

```bash
pkill -f streamlit
```

## Streamlit Cloud deploy

`data/raw/`, `data/processed/`, `data/results/` are intentionally *not*
git-ignored at the project level (`.gitignore` lines are commented out) so
that Streamlit Cloud has the artifacts it needs without running the
pipeline. Cloud build:

1. Push to GitHub.
2. Create a Streamlit Cloud app pointing at `app/dashboard.py`.
3. Python version: 3.11. Runtime deps come from `pyproject.toml`.
4. No secrets needed (everything is local-file reads).

If you need to refresh data, run the pipeline locally and commit the new
parquet/csv files. Don't try to run yfinance from Streamlit Cloud — outbound
calls to Yahoo Finance are unreliable from Streamlit's egress IPs.

## Common errors

### `ModuleNotFoundError: No module named 'numpy'` in pytest

Run via `uv run python -m pytest`, not `pytest` directly. The latter may
pick up a system Python that doesn't have the deps.

### `ImportError: PyWavelets (pywt) is required`

`pip install PyWavelets` (already in `pyproject.toml` deps; should be
installed by `uv sync`).

### `RuntimeError: No data downloaded from any chunk`

yfinance is rate-limiting or down. Wait a few minutes and retry; if persistent,
check status at downdetector.

### `Universe has 102 tickers (expected 100)`

This is a warning, not an error. Don't "fix" it.

### Streamlit ScriptRunner: `cached function returned a mutable object`

Look for places where a loader returns a DataFrame that gets mutated
downstream. Always `.copy()` before in-place modification of cached returns.

## Recipes

### Add a new ticker

1. Append a row to `config/universes/bist100.csv` with the four required
   columns (`ticker, company_name, sector, provider_symbol`).
2. Re-run the pipeline.
3. The new ticker will be subject to the 90% coverage filter — if it doesn't
   have enough history it'll be dropped (see `coverage_report.csv`).

### Extend the date range

1. Edit `config/settings.yaml`:
   ```yaml
   data:
     start_date: "2018-01-01"   # earlier
     end_date: "2026-06-01"     # later
   ```
2. Delete `data/raw/prices_raw.parquet` (yfinance has no merge logic — the
   acquisition stage replaces the file).
3. Re-run the pipeline.

### Refresh data without changing range

```bash
rm data/raw/prices_raw.parquet data/raw/xu100.parquet data/raw/fetch_metadata.json
uv run python run_pipeline.py
```

### Speed up iteration

If you're iterating on EEE-method code, you don't need to re-run acquisition,
preprocessing, or analysis. Run only what you need:

```bash
uv run python -m src.rmt_denoising          # ~5s
uv run python -m src.partial_correlation    # ~30s
uv run python -m src.wavelet_analysis       # ~30s
uv run python -m src.transfer_entropy       # ~5–10 min
uv run python -m src.snn_signals            # ~12 min (requires `[snn]` extra)
```

### Wipe everything and start fresh

```bash
rm -rf data/bist/{raw,processed,results} data/sp500/{raw,processed,results} .venv
uv sync --all-extras                                                              # core + snn + eeg + dev
uv run python run_pipeline.py                                                     # BIST
uv run python run_pipeline.py --config config/settings_sp500.yaml                 # S&P-500 (~95 min)
uv run python scripts/sp500_vs_bist.py                                            # refresh cross-market table
```

## Configuration reference

`config/settings.yaml` — every field consumed by `src/config.py`:

| Section | Field | Default | Used by |
|---|---|---|---|
| `market` | `market_id, universe_file, index_ticker, provider_suffix, currency` | BIST | data_acquisition |
| `data` | `start_date, end_date, download_interval, store_raw_close` | 2020-01-01–2026-03-01, 1d | data_acquisition |
| `preprocessing` | `min_coverage_pct, anomaly_return_threshold, forward_fill` | 0.90, 0.30, false | preprocessing |
| `analysis` | `correlation_method, annualization_factor, corr_min_periods` | pearson, 252, 200 | analysis |
| `validation` | `enabled, sample_size` | true, 10 | data_validation |
| `rolling` | `windows, step, method, min_periods_ratio` | [60,120,252], 5, pearson, 0.6 | rolling_correlation |
| `dislocation` | `zscore_window, entry_zscore, exit_zscore, min_half_life, max_half_life, top_n_candidates, lookback_window, min_correlation` | 60, 2.0, 0.5, 5, 252, 20, 252, 0.5 | pair_dislocation |
| `transfer_entropy` | `lag, n_bins, significance_shuffles, significance_level, seed` | 1, 3, 100, 0.05, 42 | transfer_entropy |

What's **not** in YAML and lives as Python defaults (FUTURE_WORK F-2):

- RMT method (`'constant'`)
- Glasso `cv`, `max_iter`, edge `threshold`
- Wavelet family (`'db4'`), max level cap (7)
- Clustering `linkage_method='single'`, `distance_threshold=1.0`
- SNN: every field of `SNNConfig` (`snn_signals.py:80`) — architecture
  (`n_hidden`, `beta`, `v_threshold`, `n_timesteps`, `window_size`,
  `readout`, etc.), training (`learning_rate`, `weight_decay`,
  `n_epochs`, `early_stop_patience`, `seed`), encoders
  (`delta_threshold`, `n_population_fields`), and label-oracle
  (`label_horizon`, `label_entry_z`, `label_min_reversion`, `train_ratio`,
  `top_n_pairs`)
