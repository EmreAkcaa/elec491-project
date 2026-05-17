# CLAUDE.md — StoNeCoAl

This file is auto-loaded by every Claude Code session. Read it first.

## What this project is

**StoNeCoAl** = *Stock Network Correlation Analysis*. A senior project that
analyses correlation structure across the BIST-100 (Borsa Istanbul) stock
universe via a 12-stage Python pipeline plus a 3-page Streamlit dashboard.

Pipeline outputs (`data/results/`) are precomputed; the dashboard reads them.
Owners: emre and a teammate (taking over from a previous senior project).

## Run cheatsheet

```bash
uv sync                                  # install deps (torch + snntorch not included)
uv sync --extra snn                      # add the optional SNN extra
uv run python run_pipeline.py            # run the whole pipeline (~10–30 min; +~12 min if [snn])
uv run streamlit run app/dashboard.py    # launch the dashboard
uv run python -m pytest -q               # 96 passed, 3 skipped (99 total)
```

## Where to look first

- `docs/ARCHITECTURE.md` — system shape, module dep graph, glossary.
- `docs/PIPELINE_REFERENCE.md` — per-module reference for `src/*.py`.
- `docs/EEE_METHODS.md` — math behind RMT / Glasso / Wavelet / TE / SNN.
- `docs/SNN_Report.md` — long-form writeup of the Spiking Neural Network module.
- `docs/UI_REFERENCE.md` — page × tab × chart inventory.
- `docs/DATA_ARTIFACTS.md` — every file under `data/`, its schema, producer, consumer.
- `docs/RUNBOOK.md` — operations (install, run, deploy, common errors).
- `docs/KNOWN_ISSUES.md` — open MED/LOW bugs (HIGH severity items closed).
- `docs/FUTURE_WORK.md` — orphan files to wire up, params to hoist, perf wins.

## Conventions

- **`run_pipeline.py` is the orchestrator.** Each stage reads from
  `data/processed/` or `data/results/` and writes new artifacts to the same
  directories. One producer per artifact. Don't write to artifacts from the
  app layer.
- **No look-ahead.** Pipeline features are cross-sectional or rolling-up-to-`t`.
  Supervised targets are `.shift(-1)` or use an explicit forward-look oracle
  (only the SNN's mean-reversion label generator does this, with K=20 days).
  If you add a feature, verify the time alignment.
- **`config/settings.yaml` is incomplete by design.** Many parameters live in
  module-level defaults or `dataclass` defaults, not the YAML. See the per-module
  hardcoded-params lists in `docs/PIPELINE_REFERENCE.md`. When adding a new param,
  hoist to YAML if it's user-facing; otherwise leave it as a default.
- **Tests live in `tests/`** (6 files, 99 tests; 96 pass + 3 skip without torch).
  Stages without tests are listed in `docs/FUTURE_WORK.md` (F-4).
- **Chart rendering** goes through `app/utils.py:render_chart` for consistent
  theming and PNG export. Don't call `st.plotly_chart` directly.
- **Caching**: every loader in `app/utils.py` is `@st.cache_data`. Heavy
  computations in `app/dashboard.py` (correlation, MST layout, rolling stats)
  are cached too — keep the `_returns_json` ser/de pattern when adding more.

## Pitfalls (read before changing pipeline code)

- **"EEE" is an informal grouping label, not an acronym.** It gathers the more
  advanced methods (RMT denoising, Glasso, Wavelet, Transfer Entropy, Spiking
  Neural Network) into a single dashboard tab and a label inside
  `run_pipeline.py`. Don't try to expand it.
- **`anomalies.csv` was buggy** (`flag_anomalies` emitted ~112k rows of NaN).
  Fixed in `src/preprocessing.py` — file is now ~5–50 rows. If you see it
  blow up again, the bug is the masked-stack pattern.
- **All but one pipeline output is wired into the dashboard** as of the
  SNN-replaces-RC merge. The only remaining orphan is
  `data/results/distance_matrix.parquet` — see `docs/DATA_ARTIFACTS.md`
  "Orphan summary" and `docs/FUTURE_WORK.md` F-1.
- **Hardcoded params in EEE methods**: wavelet `db4`, RMT `method="constant"`,
  partial correlation `threshold=0.01`, every field of `SNNConfig`
  (`snn_signals.py:80`). Not yet driven by YAML. See FUTURE_WORK F-2.
- **SNN is optional.** `src/snn_signals.py` lazy-imports `torch` and `snntorch`
  via `_require_torch()`. `run_pipeline.py` wraps `run_snn_signals(config)` in
  `try/except ImportError`, so the pipeline still completes without the
  `[snn]` extra installed — but the SNN stage is skipped and stale
  pre-committed artifacts remain in `data/<market>/results/snn_*`. Don't remove the
  try/except.
- **Universe has 102 tickers, not 100.** `src/config.py:_load_universe` warns
  but doesn't fail. Don't "correct" this.
- **Universe drops below 100 after the 90% coverage filter** (typically ~73
  survive). Pipelines downstream of `src/preprocessing.py` only see the
  surviving tickers.
- **Transfer entropy is now seeded** (`config/settings.yaml:transfer_entropy.seed`).
  The shuffle null distribution uses `np.random.default_rng(seed)`. Don't
  switch back to global `np.random.permutation`.
- **SNN underperforms the simple `|Z|>2` rule** (mean Δ-Sharpe ≈ −1.11; wins
  on 5 of 20 pairs). This is reported honestly in `docs/SNN_Report.md` §1 and
  in the dashboard's Neuromorphic Signals sub-tab caption. Don't reframe it
  as a win.

## What's been confirmed about the codebase

- Total lines: ~9,288 across `src/` (12 modules incl. `snn_signals.py`),
  `app/` (6 files), `tests/` (6 files, 99 tests). The 1,101-line SNN module
  contributes the bulk of the recent growth.
- All non-torch tests pass under `uv run python -m pytest -q` (96 passed,
  3 skipped). With `uv sync --extra snn`, all 99 tests run.
- Pipeline is reproducible end-to-end given the same
  `data/raw/prices_raw.parquet` and the seeded TE / SNN configs.
