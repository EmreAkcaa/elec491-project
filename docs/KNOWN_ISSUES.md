# KNOWN_ISSUES

Bugs and rough edges, ordered by severity. Items closed in the most recent
session are at the top with status `FIXED`.

---

## HIGH severity (closed this session)

### H-1 — `anomalies.csv` corruption — FIXED

| | |
|---|---|
| File:line | `src/preprocessing.py:107-120` |
| Symptom | `data/processed/anomalies.csv` contained **112,640 rows**, the vast majority with empty `return_value`. |
| Cause | `returns.where(mask).stack().dropna()` retained NaN rows under pandas 2.x's transitional `.stack()` semantics. |
| Fix | Stack first, then filter on `abs(return_value) > threshold`. Output is now ~5 rows. |
| Verification | `wc -l data/processed/anomalies.csv` → 5 (header + 4 valid). All 87 tests still pass. |

### H-2 — Transfer entropy unseeded RNG — FIXED

| | |
|---|---|
| File:line | `src/transfer_entropy.py:161` (was `np.random.permutation(x)`) |
| Symptom | TE matrices (`transfer_entropy_matrix.parquet`, `net_transfer_entropy_matrix.parquet`, `te_network_edges.csv`, `te_node_roles.csv`) differed across runs even with identical inputs. |
| Cause | Global `np.random.permutation` uses the legacy global state, never seeded. |
| Fix | Added `TransferEntropyConfig(seed=42, ...)` to `src/config.py` and `transfer_entropy:` block in `config/settings.yaml`; `compute_transfer_entropy_matrix` now creates `rng = np.random.default_rng(seed)` and uses `rng.permutation`. `run_transfer_entropy` reads from `config.transfer_entropy`. |
| Verification | Two runs with `seed=42` → byte-identical TE matrix. Two runs with different seeds → differ. |

### H-3 — Dead Glasso precision matrix — FIXED

| | |
|---|---|
| File:line | `src/partial_correlation.py:135` |
| Symptom | `fit_graphical_lasso` returned a `precision` DataFrame, but `run_partial_correlation` discarded it without saving. |
| Cause | Oversight — only `partial_corr` was persisted. |
| Fix | Added `precision.to_parquet(DATA_RESULTS / "precision_matrix.parquet")` after the partial-correlation save, plus `load_precision_matrix()` loader in `app/utils.py`. |
| Verification | `data/results/precision_matrix.parquet` exists after pipeline run. UI consumer is FUTURE_WORK F-1. |

### H-4 — Windows/personal paths leaked into `.claude/` — FIXED

| | |
|---|---|
| Files | `.claude/settings.json`, `.claude/settings.local.json` |
| Symptom | `.claude/settings.json` had `"additionalDirectories": ["C:\\\\Users\\\\Rutkay\\\\.claude"]` and a Windows-only Bash `ls` permission. `.claude/settings.local.json` was tracked and full of `C:\Users\Rutkay\Desktop\ELEC491\Repo_2\...` paths. |
| Fix | Removed the Windows entries from `settings.json` (kept the portable allowlist). `git rm --cached settings.local.json`; added `.claude/settings.local.json` to `.gitignore`. Renamed `.idea/Repo_2.iml` → `.idea/stonecal.iml` and updated `.idea/modules.xml`. |
| Verification | `git ls-files .claude` → only `settings.json`. New clones won't pick up Windows-specific permission lists. |

### H-5 — `fetch_all_tickers` misleading docstring — FIXED

| | |
|---|---|
| File:line | `src/data_acquisition.py:18-23` |
| Symptom | Annotated `-> pd.DataFrame`, docstring said "Returns a wide DataFrame…", but the function actually returns `tuple[pd.DataFrame, list[str]]` (the failures list). The single caller (`run_acquisition`, line 162) already unpacked correctly, so this was a lying-docs bug, not a runtime bug. |
| Fix | Updated annotation and docstring to `tuple[pd.DataFrame, list[str]]` with proper `Returns` section. |

---

## MED severity (open)

### M-1 — TE shuffle null is too easy

| | |
|---|---|
| File:line | `src/transfer_entropy.py:158-164` |
| Issue | The shuffle null permutes the entire source series, breaking both cross-dependence and source autocorrelation. A correct null tests `X ⊥ Y | Y_lag` while preserving `X`'s temporal structure (e.g. IAAFT or block bootstrap). |
| Impact | Significance test is too liberal — many "significant" edges may be artefacts of source autocorrelation. |
| Recommendation | Replace with a temporal surrogate (FUTURE_WORK F-3 details). |

### M-2 — RC features may leak future information

| | |
|---|---|
| File:line | `src/reservoir_computing.py:268-273` (`build_market_features`) |
| Issue | `pca.fit_transform(returns_clean)` fits PCA loadings using the entire history once, before walk-forward CV starts. The `pca_{i+1}` columns therefore contain in-sample-derived features at every test fold. The `dispersion.shift(-1)` target is correctly aligned, so the look-ahead is on inputs, not labels — but it's still a rigorous mistake. |
| Impact | Reported R² may overstate true out-of-sample performance. |
| Recommendation | Move PCA fit inside `walk_forward_validation` per fold (FUTURE_WORK F-3). |

### M-3 — `store_raw_close` flag is half-honoured

| | |
|---|---|
| File:line | `src/data_acquisition.py:53-78`, `src/preprocessing.py:21-27, 142-148` |
| Issue | The config flag exists but acquisition always pulls both `Adj Close` and `Close` from yfinance; preprocessing then conditionally saves `raw_close.parquet`. The flag really only controls the save step. |
| Impact | Wasted bandwidth and disk if user expects the flag to skip the download. |
| Recommendation | Either skip the `Close` download when flag is false, or drop the flag and always save. |

### M-4 — `_fetch_isyatirim_data` swallows all exceptions

| | |
|---|---|
| File:line | `src/data_validation.py:39-63` |
| Issue | `except Exception as e` covers network errors, parse errors, and bugs alike; user only sees a warning. |
| Recommendation | Narrow to `(requests.RequestException, KeyError, ValueError)`; log full traceback at DEBUG level. |

### M-5 — Streamlit session-state race

| | |
|---|---|
| File:line | `app/dashboard.py:127-136` |
| Issue | `st.session_state.pop("_goto_pair_analysis", False)` followed by setting `nav_page` works most of the time, but if a tab body sets the flag *after* this dispatch line has run in the same script execution, the jump is delayed by one rerun. |
| Recommendation | Move the dispatch logic into a callback fired by the jump button, or use `st.rerun()` immediately after setting the flag. |

### M-6 — `distance_threshold=1.0` not in YAML

| | |
|---|---|
| File:line | `src/clustering.py:84, 186` |
| Issue | Default for `fcluster` cut. Users tuning the analysis can't change cluster count without code edits. |
| Recommendation | Add `clustering.distance_threshold` (or `n_clusters`) to YAML; mirror as `ClusteringConfig` (FUTURE_WORK F-2). |

---

## LOW severity (open)

### L-1 — Hardcoded EEE / RC parameters not in YAML

See FUTURE_WORK F-2 for the full hoist list.

- RMT: `method='constant'` (`rmt_denoising.py:144`)
- Glasso: `cv=5, max_iter=200, edge_threshold=0.01` (`partial_correlation.py:60-62, 90`)
- Wavelet: `wavelet='db4', max_level cap = 7` (`wavelet_analysis.py:71, 131`)
- Reservoir: every field of `ESNConfig` (`reservoir_computing.py:48-62`)

### L-2 — `compute_transfer_entropy_matrix` has both significance arms even when `significance_shuffles=0`

| File:line | `src/transfer_entropy.py:158` |
| Note | Behaviour is fine (`if significance_shuffles > 0:` guards the inner loop). Just verbose; could short-circuit higher up. |

### L-3 — Dashboard rolling-stats precompute/recompute split — RESOLVED in commit `8379448`

| | |
|---|---|
| File:line | `app/dashboard.py:797-993` (precompute-first dispatch); loaders at `app/utils.py:593-610` |
| Resolution | Commit `8379448` implements a precompute-first path. The dashboard now reads `rolling_market_stats_w{60,120,252}.parquet` when `window∈{60,120,252} ∧ step=5 ∧ method="pearson" ∧ not expanding`, and `rolling_sector_stats.parquet` when `window=252 ∧ step=5 ∧ method="pearson"`. Off-grid parameters fall back to the on-the-fly `_compute_market_stats` / `_compute_sector` caches. A caption indicates which path ran. |

### L-4 — RC uses legacy `np.random.RandomState`

| File:line | `src/reservoir_computing.py:95` |
| Note | Other modules now use `np.random.default_rng`. Switching keeps the same reproducibility properties and removes the legacy API. Low priority. |

### L-5 — Test gaps for 7 of 12 src modules

See FUTURE_WORK F-4. Testing exists for: analysis, clustering, pair_dislocation, preprocessing, rolling_correlation. Missing: data_acquisition, data_validation, config, rmt_denoising, partial_correlation, wavelet_analysis, transfer_entropy, reservoir_computing.
