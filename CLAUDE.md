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
uv sync                                  # install deps
uv run python run_pipeline.py            # run the whole pipeline (~10–30 min)
uv run streamlit run app/dashboard.py    # launch the dashboard
uv run python -m pytest -q               # 87 tests
```

## Where to look first

- `docs/ARCHITECTURE.md` — system shape, module dep graph, glossary.
- `docs/PIPELINE_REFERENCE.md` — per-module reference for `src/*.py`.
- `docs/EEE_METHODS.md` — math behind RMT / Glasso / Wavelet / TE / RC.
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
- **No look-ahead.** RC features are cross-sectional or rolling-up-to-`t`.
  Targets are `.shift(-1)`, never the same-day signal. If you add a feature,
  verify the time alignment.
- **`config/settings.yaml` is incomplete by design.** Many parameters live in
  module-level defaults or `dataclass` defaults, not the YAML. See the per-module
  hardcoded-params lists in `docs/PIPELINE_REFERENCE.md`. When adding a new param,
  hoist to YAML if it's user-facing; otherwise leave it as a default.
- **Tests live in `tests/`** (87 passing). Stages without tests are listed in
  `docs/FUTURE_WORK.md` (F-4).
- **Chart rendering** goes through `app/utils.py:render_chart` for consistent
  theming and PNG export. Don't call `st.plotly_chart` directly.
- **Caching**: every loader in `app/utils.py` is `@st.cache_data`. Heavy
  computations in `app/dashboard.py` (correlation, MST layout, rolling stats)
  are cached too — keep the `_returns_json` ser/de pattern when adding more.

## Pitfalls (read before changing pipeline code)

- **"EEE" is an informal grouping label, not an acronym.** It gathers the more
  advanced methods (RMT denoising, Glasso, Wavelet, Transfer Entropy) into a
  single dashboard tab and a label inside `run_pipeline.py`. Don't try to
  expand it.
- **`anomalies.csv` was buggy** (`flag_anomalies` emitted ~112k rows of NaN).
  Fixed in `src/preprocessing.py` — file is now ~5–50 rows. If you see it
  blow up again, the bug is the masked-stack pattern.
- **18 pipeline outputs are orphans** — written by `src/` but never read by
  `app/`. See `docs/DATA_ARTIFACTS.md` and `docs/FUTURE_WORK.md` (F-1) for the
  list and proposed wiring.
- **Hardcoded params in EEE/RC**: `ESNConfig` (`src/reservoir_computing.py:48`)
  is constructed with all-defaults inside `run_reservoir_computing` — not yet
  driven by YAML. Same for wavelet `db4`, RMT `method="constant"`, partial
  correlation `threshold=0.01`. See FUTURE_WORK F-2.
- **Universe has 102 tickers, not 100.** `src/config.py:_load_universe` warns
  but doesn't fail. Don't "correct" this.
- **Universe drops below 100 after the 90% coverage filter** (typically ~73
  survive). Pipelines downstream of `src/preprocessing.py` only see the
  surviving tickers.
- **Transfer entropy is now seeded** (`config/settings.yaml:transfer_entropy.seed`).
  The shuffle null distribution uses `np.random.default_rng(seed)`. Don't
  switch back to global `np.random.permutation`.

## What's been confirmed about the codebase

- Total lines: ~7,742 across `src/` (12 modules), `app/` (6 files), `tests/` (5 files, 87 tests).
- All tests pass under `uv run python -m pytest -q`.
- Pipeline is reproducible end-to-end given the same `data/raw/prices_raw.parquet`
  and the seeded TE config.
