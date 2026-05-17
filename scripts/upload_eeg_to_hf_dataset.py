"""One-time uploader: push EEG processed parquets to a companion HF Dataset.

Why
===
HF Spaces caps per-repo storage at 1 GB total (unmovable, even on paid plans).
Our 2 EEG processed parquets are 308 MB each (616 MB combined) which blows
past that limit when combined with the rest of the dashboard.

The canonical workaround documented by HF is "put large bulk data in a
companion Dataset repo (50 GB per-file limit) and have the Space fetch it
on startup". This script does the push side; the fetch side is wired up in
``app/dashboard.py:_materialise_eeg_data_if_needed()``.

Run once
========
After creating the dataset repo manually at
https://huggingface.co/new-dataset (Owner: you, Name: stonecoal-eeg,
License: MIT, Public), run::

    uv run python scripts/upload_eeg_to_hf_dataset.py

The script reads HF_DEPLOY_TOKEN from your environment (or fall back to your
huggingface-cli login), reads the EEG processed parquets from local disk,
and pushes them to ``huggingface.co/datasets/<your-owner>/stonecoal-eeg``.

Re-run any time the EEG pipeline produces new outputs and you want the
deployed dashboard to reflect them.

Configuration
=============
- ``EEG_DATASET_REPO`` env var (default: ``FlyingSubmarine33/stonecoal-eeg``)
- ``HF_DEPLOY_TOKEN`` env var with **Write** scope on that dataset repo
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EEG_PROCESSED_DIR = PROJECT_ROOT / "data" / "eeg_motor_left_right" / "processed"

DEFAULT_DATASET_REPO = "FlyingSubmarine33/stonecoal-eeg"
DATASET_REPO = os.environ.get("EEG_DATASET_REPO", DEFAULT_DATASET_REPO)


def main() -> int:
    # Defer the import so the script's --help / docstring works even when
    # huggingface_hub isn't installed yet.
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("huggingface_hub not installed. Install with: uv pip install huggingface_hub")
        return 1

    token = os.environ.get("HF_DEPLOY_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        log.warning(
            "No HF_DEPLOY_TOKEN or HF_TOKEN in environment. Falling back to "
            "huggingface-cli login credentials at ~/.cache/huggingface/token. "
            "If that's also empty the upload will fail with 401."
        )

    if not EEG_PROCESSED_DIR.exists():
        log.error(
            "%s does not exist. Run the EEG pipeline first: "
            "uv sync --extra eeg && uv run python run_pipeline_eeg.py",
            EEG_PROCESSED_DIR,
        )
        return 1

    files_to_upload = list(EEG_PROCESSED_DIR.glob("*.parquet")) + list(
        EEG_PROCESSED_DIR.glob("*.csv")
    )
    if not files_to_upload:
        log.error("No parquet/csv files found under %s", EEG_PROCESSED_DIR)
        return 1

    total_mb = sum(f.stat().st_size for f in files_to_upload) / 1024 / 1024
    log.info("Uploading to dataset repo: %s", DATASET_REPO)
    log.info("Source dir: %s", EEG_PROCESSED_DIR)
    log.info("Files (%d, %.1f MB total):", len(files_to_upload), total_mb)
    for f in files_to_upload:
        log.info("  %s  (%.1f MB)", f.name, f.stat().st_size / 1024 / 1024)

    api = HfApi(token=token)

    # Create the dataset repo if it doesn't exist yet (no-op if it does).
    try:
        api.create_repo(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        log.info("Dataset repo confirmed (created if missing): %s", DATASET_REPO)
    except Exception as exc:
        log.error("Could not create/verify dataset repo: %s", exc)
        log.error("Manually create it at https://huggingface.co/new-dataset (Owner: <you>, Name: stonecoal-eeg, Public)")
        return 1

    log.info("Pushing files to %s ...", DATASET_REPO)
    commit_info = api.upload_folder(
        folder_path=str(EEG_PROCESSED_DIR),
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="EEG bulk parquets — uploaded via scripts/upload_eeg_to_hf_dataset.py",
        allow_patterns=["*.parquet", "*.csv"],
    )
    log.info("Upload complete.")
    log.info("Commit: %s", commit_info.commit_url)
    log.info("Dataset URL: https://huggingface.co/datasets/%s", DATASET_REPO)
    log.info("")
    log.info("Next step: the deployed dashboard's preload will fetch these files")
    log.info("on first launch via huggingface_hub.snapshot_download(...).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
