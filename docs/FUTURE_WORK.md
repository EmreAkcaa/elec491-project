# FUTURE_WORK

Roadmap of deferred work, organised by theme. Items in
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) at MED severity also appear here as
F-6 with concrete plans.

---

## F-1 — Wire the last remaining orphan output

Commit `8379448` ("New methods wired to dashboard") wired 20 of the 21
originally-inventoried orphan outputs into the dashboard (see
[`DATA_ARTIFACTS.md`](DATA_ARTIFACTS.md) "Orphan summary"; see
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) L-3 for the closed precompute/recompute
split). One file is still unwired:

| Orphan file | Suggested action |
|---|---|
| `data/results/distance_matrix.parquet` | Pick one of: **(a)** delete the producer save at `src/analysis.py:154` — the matrix is just `sqrt(2*(1 - corr))` derived from `pearson_corr.parquet`, which the dashboard already loads. **(b)** wire it as a "Raw distance" heatmap sibling of the new "Denoised Correlation Matrix" heatmap in EEE → RMT, using the existing `_plot_matrix_heatmap` helper (`app/eee_analysis.py:_plot_matrix_heatmap`) with `zmin=0, zmax=2, diverging=False`. **(a)** is cleaner; **(b)** is more pedagogically honest about what the denoiser is changing. |

For the historical wiring proposals (table archived in the commit diff),
see `git show 8379448 -- app/eee_analysis.py app/utils.py app/dashboard.py`.

---

## F-2 — Hoist hardcoded EEE params to `settings.yaml`

Add the following dataclasses to `src/config.py` (mirror the existing
`DislocationConfig` / `TransferEntropyConfig` pattern), wire them into
`PipelineConfig` and `load_config`, and replace the hardcoded calls.

```yaml
# config/settings.yaml additions
clustering:
  linkage_method: "single"
  distance_threshold: 1.0

rmt:
  method: "constant"   # or "zero"

partial_correlation:
  cv_folds: 5
  max_iter: 200
  edge_threshold: 0.01

wavelet:
  family: "db4"
  max_levels: 7
```

Each `run_*` function then reads from `config.<name>` instead of constructing
defaults locally.

---

## F-3 — Harden transfer entropy null distribution

For transfer entropy (KNOWN_ISSUES M-1): replace the shuffle null
with IAAFT (iterative amplitude-adjusted Fourier transform) surrogates,
preserving source autocorrelation. Use `pyunicorn` or implement directly
via `np.fft`.

---

## F-4 — Test gaps

Priority order (high-value first):

1. **`src/transfer_entropy.py`** — heaviest stage, no tests. At minimum
   test: `transfer_entropy(x, y, lag=1, n_bins=3)` on a known case where
   `y_t = x_{t-1}` should give large positive TE; symmetric independent
   inputs should give ~0.
2. **`src/rmt_denoising.py`** — test that `marchenko_pastur_bounds(T, N)`
   returns analytic values; test that `denoise_correlation` preserves
   diagonal=1 and clips correctly.
3. **`src/partial_correlation.py`** — test that the `pcorr_ij = -Θ_ij /
   √(Θ_ii Θ_jj)` derivation matches sklearn's output on a small fixture.
4. **`src/wavelet_analysis.py`** — test that `wavelet_decompose` returns
   one DataFrame per requested scale and that summing reconstructed
   details across all scales recovers (within tolerance) the original
   signal minus the approximation.
5. **`src/data_validation.py`** — mock `isyatirimhisse.fetch_stock_data`
   to test the alignment + status-classification logic without network.
6. **`src/data_acquisition.py`** — mock `yf.download`; test the chunking,
   ticker rename, and failures-list logic.
7. **`src/config.py`** — test missing/invalid YAML, duplicate-ticker
   universe, missing required columns.

UI smoke tests via `streamlit testing.AppTest` for: dashboard loads, all 6
tabs render, pair_analysis loads, EEE Analysis loads.

---

## F-5 — Performance

| Issue | Recommendation |
|---|---|
| Sector map rebuilt on every chart render | Cache `dict(zip(universe.ticker, universe.sector))` once in `utils.py` |
| MST kamada-kawai layout recomputed per session | Already cached via `_mst_layout`, but re-keys on the JSON payload — switch to a content-hash key for stability across reruns |
| `compute_rolling_pair_correlation` (Spearman/Kendall path) iterates per window | Vectorise via rank transform once, then reuse `rolling.corr` |
| TE `O(N² · S · T)` is the slowest stage at ~5–10 min | Two options: (a) parallelise pair loop with `joblib` / `multiprocessing` (low risk, ~4× speedup on 4 cores); (b) precompute discretized series once instead of inside the inner shuffle (~2× speedup) |
| Wavelet decomposition reruns full DWT per scale per ticker | Cache once per ticker and slice — saves ~7× on the inner loop |

---

## F-6 — MED-severity items from KNOWN_ISSUES

These are concrete enough to schedule:

- **M-1** TE temporal-surrogate null — covered above in F-3.
- **M-3** `store_raw_close` half-honoured — small refactor, drop the flag
  or skip the download.
- **M-4** `_fetch_isyatirim_data` broad except — narrow to known exception
  classes; add a debug-level traceback.
- **M-5** Streamlit session-state race — convert deferred flag to button
  callback that calls `st.rerun()` after `st.session_state['nav_page'] =
  'Pair Analysis'`.
- **M-6** `distance_threshold` to YAML — covered above in F-2 under
  `clustering:`.

---

## F-7 — Stretch ideas (not requested by user)

- **Dynamic universe** — refresh the BIST-100 membership list periodically
  rather than freezing at one CSV.
- **Multi-market support** — `MarketConfig` is already general; adding S&P
  500 or Nikkei would be a CSV swap plus index ticker change.
- **Live mode** — add a "today" sidebar button that re-runs the acquisition
  + minimum compute path and writes a delta. Not great for Streamlit Cloud
  due to yfinance reliability; better as a CLI cron job.
- **Sector-map UI editor** — let the user remap tickers to sectors
  interactively and re-render the affected charts.
- **Comparison mode** — let the user select two date ranges and overlay
  rolling stats / MSTs side-by-side.
