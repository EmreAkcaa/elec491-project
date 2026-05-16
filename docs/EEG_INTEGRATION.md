# EEG Integration (Phase F)

**Status:** SCAFFOLD. The pipeline code, configuration, universe CSV, and a
runnable orchestrator are all in place. End-to-end verification against an
actual PhysioNet download has not been performed in the implementation
session — the 5 sanity checks below MUST pass before claiming Phase F is
validated.

## What this phase adds

The StoNeCoAl toolkit is largely domain-agnostic: Pearson correlation, MST,
RMT denoising, Graphical LASSO partial correlation, wavelet multi-resolution
analysis, transfer entropy, and mutual information all work on any
`(time × node)` panel. Phase F demonstrates this by applying the same six
estimators to brain electrical activity from the **PhysioNet EEG Motor
Movement/Imagery Database**
(<https://www.physionet.org/content/eegmmidb/1.0.0/>): 109 subjects, 14
runs each, 64 channels, 160 Hz, EDF+ format, Open Data Commons Attribution
(ODC-By) license.

The pitch becomes:

> *We applied the same dependence-estimation toolkit to two financial
> universes (BIST-73, S&P-100) and one biomedical signal source (64-channel
> motor-imagery EEG from PhysioNet). The estimators recovered structurally
> meaningful results in all three: conglomerate hubs in Türkiye, mega-cap
> hubs in the US, and posterior alpha-band hubs in the brain. The toolkit
> is domain-agnostic and the methods are all drawn from the EEE syllabus
> (signal processing, system identification, information theory, multi-
> resolution analysis).*

## Files added

| File | Role |
|---|---|
| `pyproject.toml` (`[eeg]` extra) | `mne >= 1.6`. Install with `uv sync --extra eeg` (~50 MB). |
| `src/config.py` (`EEGConfig` dataclass + `data.source` field) | Knobs for task type, subject IDs, runs per condition, sampling rate, bandpass/notch frequencies, CAR reference toggle, cache raw. |
| `src/eeg_acquisition.py` | MNE wrapper: download via `mne.datasets.eegbci.load_data`, concatenate runs per subject, bandpass + notch + common-average reference, write `(time × channel)` parquet that downstream stages consume. |
| `config/universes/eeg_motor_left_right.csv` | 64-channel 10-10 EEG montage with anatomical-region labels (Frontal, Central, Temporal, Parietal, Parieto-occipital, Occipital, Frontopolar, Centroparietal). The `sector` column repurposes the equity-universe schema for anatomical region. |
| `config/settings_eeg.yaml` | Per-task EEG settings; default task is `left_right` with 10 curated subjects. |
| `run_pipeline_eeg.py` | EEG orchestrator: runs `run_eeg_acquisition` then the same downstream stages used by `run_pipeline.py`. **Skips** `pair_dislocation` and `snn_signals` (financial-only). |
| `docs/EEG_INTEGRATION.md` | (this file) |

## How to run

```bash
# 1. One-time: install MNE-Python (~50 MB)
uv sync --extra eeg

# 2. Run the EEG pipeline. First invocation downloads ~3.4 GB from PhysioNet
#    (cached under ~/mne_data/MNE-eegbci-data/). Subsequent runs reuse the cache.
uv run python run_pipeline_eeg.py                          # default task: left_right
uv run python run_pipeline_eeg.py --task feet_fists        # another task
uv run python run_pipeline_eeg.py --task baseline          # eyes-open/eyes-closed

# 3. Dashboard: switch universes via env var
DASHBOARD_UNIVERSE=eeg_motor_left_right uv run streamlit run app/dashboard.py
```

Per-task artifacts land under `data/eeg_motor_<task>/{raw,processed,results}/`
thanks to the Phase D path parameterisation.

## Architecture

### Why the file name is "log_returns.parquet"

The downstream stages (`analysis`, `clustering`, `rmt_denoising`,
`partial_correlation`, `wavelet_analysis`, `transfer_entropy`) all read
`processed/log_returns.parquet`. Rather than refactor every stage to accept
a different filename, `run_eeg_acquisition` writes its `(time × channel)`
bandpassed-voltage panel to that file. The downstream code is agnostic to
whether the values are returns or volts — it just computes correlations,
MSTs, etc. Documenting this aliasing once is cleaner than changing the
contract.

### Why these tasks, these subjects

PhysioNet's EEG-Motor-Imagery has 109 subjects × 14 runs each. We use 10
curated subjects × 3 runs per task (≈287k samples × 64 channels per pooled
universe), giving `T/N ≈ 4480` — well above the `T/N > 100` threshold for
stable RMT. The three task universes (`left_right`, `feet_fists`, `baseline`)
give three independent runs of the entire pipeline so the team can compare
network structure across cognitive states.

### Preprocessing chain (minimum viable, defensible)

- **Bandpass 1–50 Hz** (`mne.io.Raw.filter`) — removes DC drift and
  aliasing above Nyquist/2.
- **Notch at 50 Hz** (Türkiye / EU grid; switch to 60 Hz in US) — removes
  line noise. Wrong notch frequency leaves a power-line peak that
  artificially inflates correlation at 50/60 Hz harmonics.
- **Common-average reference** (`raw.set_eeg_reference("average")`) —
  standard in EEG functional-connectivity work; reduces shared environmental
  noise.
- **No ICA, no artifact rejection, no epoching beyond the per-recording
  boundary.** Deliberately minimal: ICA needs interactive cleanup and would
  be ~half a week of work. Simple amplitude thresholding can be added later
  if sanity-check #2 (motor desynchronisation) is noisy.

### Modules that don't apply

`src/pair_dislocation.py` and `src/snn_signals.py` are skipped in
`run_pipeline_eeg.py` because they assume financial pair-spread semantics
(OLS hedge ratio, Z-score, mean-reversion half-life, classical |Z|>2
rule). The EEG pipeline still produces the seven correlation/network/
information-theoretic artifacts that the dashboard surfaces.

## Sanity-check protocol (run after the first end-to-end pipeline)

Phase F is **not** complete until **at least 4 of these 5** checks pass on
the resulting `data/eeg_motor_<task>/results/` artifacts. If <3 pass, roll
back to "future work" rather than ship.

| # | Check | What to verify | Reference |
|---|---|---|---|
| 1 | Inter-hemispheric homologous coherence | Baseline eyes-closed task: pairs `(FP1, FP2)`, `(C3, C4)`, `(P3, P4)`, `(O1, O2)` cluster as low-distance in the raw MST | Thatcher 1986 |
| 2 | Motor-imagery contralateral desynchronisation | `left_right` task: rolling correlation `corr(C3, C4)` *drops* during motor-imagery blocks vs rest. T-test p<0.05 | Pfurtscheller & Lopes da Silva 1999 |
| 3 | RMT signal mode | `λ₁` exceeds MP upper bound; ratio `λ₁/λ₂ > 5` (broadband co-activation captured by leading mode) | Plerou et al. 2002 (cross-domain) |
| 4 | Resting-state MST hubs | Baseline task: posterior channels (P3, P4, Pz, O1, O2) account for ≥ 2 of top-5 hubs by degree centrality | Stam 2014 (network neuroscience) |
| 5 | TE directionality | Motor-imagery task: `TE(FP/F → C) > TE(C → FP/F)` — planning regions lead motor cortex | Bressler & Seth 2011 |

The team should write up the results of these 5 checks in a `docs/EEG_VERIFICATION.md` after the first pipeline run.

## Known landmines

| Landmine | Mitigation |
|---|---|
| 50 Hz vs 60 Hz notch | Config knob `eeg.notch_hz`; default 50 (Türkiye). Set to 60 if running in US. |
| PhysioNet ODC-By attribution | Add `Schalk et al. (2004), "BCI2000: a general-purpose brain-computer interface (BCI) system", IEEE Trans. Biomed. Eng.` to `README.md` + dashboard footer + final report bibliography. |
| Inter-subject variability is large | Curate 5–10 clean subjects; the default list `[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]` is arbitrary — visually inspect each subject's recordings before declaring sanity checks done. |
| 3.4 GB raw data download | The download is one-shot via MNE's cache (`~/mne_data/`). `data/eeg_motor_<task>/raw/` only stores per-subject concatenated `.fif` files (~50 MB each); add `data/eeg_motor_*/raw/` to `.gitignore` if you don't want them committed. |
| MNE-Python learning curve | The 5-line load pattern (`mne.datasets.eegbci.load_data`) is the official tutorial path; <https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html> has worked examples. |
| EDF+ malformed headers (rare) | `load_raw_for_subject` wraps the call in try/except via the per-subject loop in `run_eeg_acquisition`; bad subjects are logged and skipped, the pipeline continues. |
| Modules expect `(time × node)` panel; EEG produces `(sample × channel)` | Same shape, different units. `analysis.compute_correlation_matrix` and friends don't care. |
| `processed/log_returns.parquet` semantically misleading for EEG | Documented above. Don't rename; the pipeline contract is fixed. |

## Day-by-day plan (~7 days actual work)

| Day | Task | Output |
|---|---|---|
| 1 | Run `run_pipeline_eeg.py --task left_right`; debug MNE channel-name standardisation | `data/eeg_motor_left_right/processed/log_returns.parquet` |
| 2 | Run sanity check #1 (baseline) + #2 (motor desync) | Pass/fail per check |
| 3 | Run pipeline for `--task feet_fists` and `--task baseline` | All 3 task universes populated |
| 4 | Run sanity checks #3, #4, #5 | Pass/fail per check |
| 5 | Write `docs/EEG_VERIFICATION.md` with sanity-check results | Documented |
| 6 | Build EEG-specific dashboard page (if Phase E's universe selector is already in) | `app/eeg_analysis.py` |
| 7 | Polish, add EEG section to `docs/TECHNICAL_REPORT.md` and final-presentation slide | Reports updated |

## Phase F completion criteria

- [ ] `uv sync --extra eeg` succeeds
- [ ] `uv run python run_pipeline_eeg.py` completes end-to-end without errors
- [ ] All seven downstream artifacts exist under `data/eeg_motor_<task>/results/` for at least one task
- [ ] At least 4 of 5 sanity checks pass; results documented in `docs/EEG_VERIFICATION.md`
- [ ] PhysioNet attribution added to `README.md` and the final report
- [ ] `DASHBOARD_UNIVERSE=eeg_motor_left_right uv run streamlit run app/dashboard.py` loads cleanly and renders the four EEE sub-tabs that don't depend on financial semantics (RMT, Glasso, Wavelet, TE). The SNN and Pair Analysis sub-tabs will gracefully say "no data" for EEG.

If <3 sanity checks pass: roll back EEG, present as "future work" in the
final presentation slide, use the freed days to polish BIST + S&P comparison.
