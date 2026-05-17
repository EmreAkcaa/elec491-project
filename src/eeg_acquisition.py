"""EEG acquisition + preprocessing for PhysioNet EEG-Motor-Imagery (Phase F).

Wraps `mne.datasets.eegbci.load_data` so the StoNeCoAl pipeline can consume
brain electrical activity the same way it consumes financial returns. The
downstream stages (analysis, clustering, RMT, Glasso, wavelet, transfer entropy,
mutual information) are domain-agnostic and run unchanged on the resulting
(time × channel) DataFrame, which we write to
`data/<eeg_market_id>/processed/log_returns.parquet` so existing loaders find
it. (Yes, the file name is a misnomer for EEG — keeping it preserves the
pipeline contract; the panel itself is bandpassed voltage, not log returns.)

Status: SCAFFOLD — code has not been end-to-end verified against an actual
PhysioNet download in this session. See docs/EEG_INTEGRATION.md for the
sanity-check protocol (5 neuroscience checks) that the team must run to
declare Phase F validated.

Run:
    uv sync --extra eeg
    uv run python run_pipeline_eeg.py --task left_right

Outputs (per task):
    data/eeg_motor_<task>/raw/<subject>_<run>_raw.fif       # MNE-native cache
    data/eeg_motor_<task>/processed/log_returns.parquet    # (time × 64ch) bandpassed
    data/eeg_motor_<task>/processed/coverage_report.csv    # trivial (all 64ch always)
    data/eeg_motor_<task>/processed/adj_close.parquet      # = log_returns (alias)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


# Task-type → list of PhysioNet runs (1-indexed)
# Source: https://www.physionet.org/content/eegmmidb/1.0.0/
TASK_RUNS = {
    "left_right": [4, 8, 12],          # left vs right fist (motor imagery)
    "feet_fists": [6, 10, 14],         # feet vs both fists (motor imagery)
    "baseline":   [1, 2],              # eyes-open + eyes-closed
}


def _require_mne():
    """Lazy import of MNE-Python; raise informative error if missing."""
    try:
        import mne  # noqa: F401
        from mne.datasets import eegbci  # noqa: F401
        return mne, eegbci
    except ImportError as e:
        raise ImportError(
            "EEG support requires MNE-Python. Install with: uv sync --extra eeg"
        ) from e


def download_subject(subject_id: int, runs: list[int]) -> list[Path]:
    """Download one subject's runs via MNE; returns local EDF+ paths.

    Uses MNE's built-in mirror of PhysioNet so we don't have to manage URLs.
    Cached under `~/mne_data/MNE-eegbci-data/` by default.
    """
    mne, eegbci = _require_mne()
    # MNE >= 1.10 renamed `subject` (singular) -> `subjects` (list).
    # update_path=True suppresses an interactive input() prompt that breaks
    # in non-TTY runs (nohup / CI).
    paths = eegbci.load_data(subjects=[subject_id], runs=runs, update_path=True)
    return [Path(p) for p in paths]


def load_raw_for_subject(subject_id: int, runs: list[int]):
    """Load + concatenate the requested runs for one subject; returns an mne.Raw."""
    mne, eegbci = _require_mne()
    paths = download_subject(subject_id, runs)
    raws = [mne.io.read_raw_edf(p, preload=True, verbose="ERROR") for p in paths]
    # PhysioNet EDF+ labels channels as e.g. 'Fc5.', 'C3..' (with trailing dots).
    # Strip them so they match our canonical universe CSV.
    for r in raws:
        eegbci.standardize(r)
    raw_concat = mne.concatenate_raws(raws)
    return raw_concat


def preprocess(raw, config: PipelineConfig):
    """Apply bandpass + notch + CAR reference (returns the same Raw, in place)."""
    cfg = config.eeg
    raw = raw.copy()
    raw.filter(
        l_freq=cfg.bandpass_low_hz,
        h_freq=cfg.bandpass_high_hz,
        picks="eeg",
        verbose="ERROR",
    )
    if cfg.notch_hz and cfg.notch_hz > 0:
        raw.notch_filter(freqs=cfg.notch_hz, picks="eeg", verbose="ERROR")
    if cfg.car_reference:
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return raw


def raw_to_dataframe(raw, universe_channels: list[str]) -> pd.DataFrame:
    """Convert a preprocessed mne.Raw to (time × channel) DataFrame.

    Only the channels present in `universe_channels` are retained; channels
    in the recording but not in the universe are dropped, and vice versa
    (missing channels are filled with NaN — propagates to downstream coverage).
    """
    # MNE returns shape (n_channels, n_samples). Transpose to (n_samples, n_channels).
    available = raw.ch_names
    keep = [c for c in universe_channels if c in available]
    missing = [c for c in universe_channels if c not in available]
    if missing:
        logger.warning(
            "Universe channels missing from recording: %s (filled with NaN)",
            missing,
        )

    data = raw.get_data(picks=keep).T  # (n_samples, len(keep))
    times = raw.times
    df = pd.DataFrame(data, index=pd.Index(times, name="time_s"), columns=keep)
    # Add missing channels as NaN so the DataFrame matches the universe shape
    for c in missing:
        df[c] = np.nan
    return df[universe_channels]  # reorder to canonical universe order


def run_eeg_acquisition(config: PipelineConfig) -> None:
    """Full EEG acquisition + preprocessing pipeline step.

    1. For each curated subject, load + concatenate the requested task runs
    2. Preprocess (bandpass + notch + CAR)
    3. Convert to (time × channel) DataFrame
    4. Concatenate across subjects (rows stacked, subject-id label dropped)
    5. Write to data/<market_dir>/processed/log_returns.parquet (and as
       adj_close.parquet too so loaders that look there still work)

    Writes per-subject `_raw.fif` to `data/<market_dir>/raw/` for inspection.
    """
    logger.info("=== EEG Acquisition (PhysioNet EEG-Motor-Imagery) ===")
    config.data_raw.mkdir(parents=True, exist_ok=True)
    config.data_processed.mkdir(parents=True, exist_ok=True)

    cfg = config.eeg
    runs = cfg.runs_per_condition or TASK_RUNS.get(cfg.task_type, TASK_RUNS["left_right"])
    universe_channels = config.tickers

    subject_dfs = []
    for sid in cfg.subject_ids:
        try:
            logger.info("Loading subject %03d, runs %s …", sid, runs)
            raw = load_raw_for_subject(sid, runs)
            raw = preprocess(raw, config)
            if cfg.cache_raw:
                cache_path = config.data_raw / f"S{sid:03d}_concatenated_raw.fif"
                raw.save(cache_path, overwrite=True, verbose="ERROR")
            df = raw_to_dataframe(raw, universe_channels)
            df["_subject"] = sid  # tracking column; dropped at concat
            subject_dfs.append(df)
        except Exception as exc:
            logger.warning("Skipping subject %03d due to load error: %s", sid, exc)

    if not subject_dfs:
        raise RuntimeError("No subjects loaded — verify PhysioNet download succeeded.")

    pooled = pd.concat(subject_dfs, axis=0, ignore_index=False)
    pooled = pooled.drop(columns=["_subject"])  # keep raw values; subject id stays in cache
    logger.info(
        "Pooled EEG panel: %d samples × %d channels (across %d subjects)",
        pooled.shape[0], pooled.shape[1], len(subject_dfs),
    )

    # Write to the file names the downstream pipeline expects.
    # Both "log_returns" and "adj_close" are aliased to the same bandpassed-
    # voltage panel; the downstream pipeline doesn't care which is called what.
    out_lr = config.data_processed / "log_returns.parquet"
    out_ac = config.data_processed / "adj_close.parquet"
    pooled.to_parquet(out_lr)
    pooled.to_parquet(out_ac)
    logger.info("Saved EEG panel to %s and %s", out_lr, out_ac)

    # Trivial coverage report (every channel always present; helps the dashboard)
    coverage = pd.DataFrame({
        "ticker": pooled.columns,
        "total_days": len(pooled),
        "available_days": pooled.notna().sum().values,
        "coverage_pct": (pooled.notna().sum() / len(pooled)).round(4).values,
    })
    coverage.to_csv(config.data_processed / "coverage_report.csv", index=False)
    logger.info("Saved coverage report (EEG: trivially 1.0 for all present channels).")
