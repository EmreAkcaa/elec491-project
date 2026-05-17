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

### L-1 — Hardcoded EEE parameters not in YAML

See FUTURE_WORK F-2 for the full hoist list.

- RMT: `method='constant'` (`rmt_denoising.py:144`)
- Glasso: `cv=5, max_iter=200, edge_threshold=0.01` (`partial_correlation.py:60-62, 90`)
- Wavelet: `wavelet='db4', max_level cap = 7` (`wavelet_analysis.py:71, 131`)

### L-2 — `compute_transfer_entropy_matrix` has both significance arms even when `significance_shuffles=0`

| File:line | `src/transfer_entropy.py:158` |
| Note | Behaviour is fine (`if significance_shuffles > 0:` guards the inner loop). Just verbose; could short-circuit higher up. |

### L-3 — Dashboard rolling-stats precompute/recompute split — RESOLVED in commit `8379448`

| | |
|---|---|
| File:line | `app/dashboard.py:797-993` (precompute-first dispatch); loaders at `app/utils.py:593-610` |
| Resolution | Commit `8379448` implements a precompute-first path. The dashboard now reads `rolling_market_stats_w{60,120,252}.parquet` when `window∈{60,120,252} ∧ step=5 ∧ method="pearson" ∧ not expanding`, and `rolling_sector_stats.parquet` when `window=252 ∧ step=5 ∧ method="pearson"`. Off-grid parameters fall back to the on-the-fly `_compute_market_stats` / `_compute_sector` caches. A caption indicates which path ran. |

### L-5 — Test gaps for 6 of 11 src modules

See FUTURE_WORK F-4. Testing exists for: analysis, clustering, pair_dislocation, preprocessing, rolling_correlation. Missing: data_acquisition, data_validation, config, rmt_denoising, partial_correlation, wavelet_analysis, transfer_entropy.

---

## RESOLVED (Phase G, 2026-05-17)

### G-1 — BIST anomalies output was 4 unhandled corporate actions, not market events — RESOLVED

| | |
|---|---|
| File:line | `data/bist/processed/anomalies.csv` (output); `src/preprocessing.py:135-160` (fix); `config/settings.yaml:18-26` (override list) |
| Symptom | The 4 entries flagged in `anomalies.csv` (CCOLA 2024-08-01 log-return −2.38, HEKTS 2024-09-09 −1.05, AYGAZ 2022-09-01 +0.55, HEKTS 2021-04-30 +0.37) were not market anomalies but unhandled corporate-action artifacts. yfinance `Adj Close` failed to back-adjust four BIST corporate actions: CCOLA's 10.81× bonus issue (828.44 → 76.65), HEKTS's 2.84× and 1.45× bonus issues, AYGAZ's 1.72× bonus issue. Inspected pre/post adjusted-close prices to confirm. The contaminated cells inflated \|log-return\| values 0.4–2.4, dragging affected tickers' mean correlation down (CCOLA's mean \|corr\| was 0.124 with the bug, 0.361 after the mask — a 3× hidden distortion). |
| Fix | Added `manual_anomaly_nulls: list = field(default_factory=list)` to `PreprocessingConfig`. `run_preprocessing` iterates the list and sets `log_returns.loc[ts, ticker] = NaN` for each `[ticker, "YYYY-MM-DD"]` entry, **after** `compute_log_returns` and **before** `flag_anomalies`. Default empty list preserves backward compatibility for `settings_sp500.yaml` and `settings_eeg.yaml`. BIST `settings.yaml` declares the 4 corrections inline with audit comments. |
| Verification | Post-rerun: `data/bist/processed/anomalies.csv` is header-only (0 flagged rows); the 4 (ticker, date) cells are NaN; CCOLA mean \|corr\| = 0.361 (was 0.124); MST hubs (KCHOL, SISE, SAHOL) unchanged; downstream NaN handling validated (TE delta per masked cell ≈ +0.001 nats; Glasso loses 3 of 1543 rows = 0.2 %). |

### G-2 — S&P-500 universe had 3 dual-class share duplicates — RESOLVED

| | |
|---|---|
| File:line | `config/universes/sp500_full.csv` |
| Symptom | The Wikipedia S&P 500 constituent list contains the company entries for Alphabet, Fox Corp, and News Corp **twice** — once for Class A and once for Class B/C shares. Including both in the universe caused mechanical 0.99+ pairwise correlations (GOOGL-GOOG 0.996, FOXA-FOX 0.988, NWSA-NWS 0.974) and produced a nonsense pair-dislocation #1 candidate of GOOGL-GOOG (share-class arbitrage with current Z-score −2.26 and 26-day half-life — literally tradable but meaningless). |
| Fix | Dropped GOOG, FOX, NWS from the universe CSV (kept GOOGL, FOXA, NWSA — the voting/more-liquid classes by convention). BRK-B and BF-B are NOT duplicates (their A-class counterparts are not in S&P 500) and were kept. Universe: 503 → 500. |
| Verification | Post-rerun: top-10 correlations contain no 0.99+ artifacts (max now UDR-EQR 0.927, apartment REITs); pair-dislocation #1 is CMS-AEP (electric utilities, 35-day half-life, actually tradeable); RMT structure unchanged (D_eff = 6.56, top eig share 38.1 %); MST hubs unchanged (PRU/AMP/PH). |
