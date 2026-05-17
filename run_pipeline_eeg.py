"""EEG pipeline entry point (Phase F).

Loads + preprocesses PhysioNet EEG-Motor-Imagery data, then runs the same
domain-agnostic downstream stages used by the financial pipeline (analysis,
clustering, RMT, Glasso, wavelet, transfer entropy). The SNN, pair-dislocation,
and validation stages are intentionally skipped for EEG because they assume
financial semantics (pair spreads, sectors, İş Yatırım cross-check).

Usage:
    uv sync --extra eeg                                              # one-time, ~50 MB
    uv run python run_pipeline_eeg.py                                # default: left_right task
    uv run python run_pipeline_eeg.py --config config/settings_eeg.yaml
    uv run python run_pipeline_eeg.py --task feet_fists              # other task universes

Status: SCAFFOLD. The code path is in place but has not been end-to-end
verified against an actual PhysioNet download in this session. See
docs/EEG_INTEGRATION.md for the 5-step sanity check the team must run.
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(argv=None):
    from src.config import load_config
    from src.eeg_acquisition import run_eeg_acquisition
    from src.analysis import run_analysis
    from src.clustering import run_clustering
    from src.rolling_correlation import run_rolling_analysis
    from src.rmt_denoising import run_rmt_denoising
    from src.partial_correlation import run_partial_correlation
    from src.wavelet_analysis import run_wavelet_analysis
    from src.transfer_entropy import run_transfer_entropy

    parser = argparse.ArgumentParser(description="StoNeCoAl EEG pipeline.")
    parser.add_argument("--config", default="config/settings_eeg.yaml",
                        help="EEG settings YAML.")
    parser.add_argument("--task", default=None,
                        help="Override eeg.task_type (left_right | feet_fists | baseline).")
    args = parser.parse_args(argv)

    logger.info("========== StoNeCoAl EEG Pipeline ==========")

    config = load_config(args.config)
    if args.task:
        config.eeg.task_type = args.task
    logger.info(
        "Config: %s (market_dir=%s, task=%s), %d channels, results -> %s",
        config.market.market_id, config.market_dir, config.eeg.task_type,
        len(config.tickers), config.data_results,
    )

    # ── EEG-specific acquisition + preprocessing (~5 min after download) ──
    run_eeg_acquisition(config)

    # ── Domain-agnostic analysis stages (reused unchanged from financial pipeline) ──
    run_analysis(config)
    run_clustering(config)
    run_rolling_analysis(config)
    run_rmt_denoising(config)
    run_partial_correlation(config)
    run_wavelet_analysis(config)
    run_transfer_entropy(config)

    # NOTE: pair_dislocation, snn_signals are NOT run for EEG —
    # they assume financial pair-spread semantics that don't apply to brain channels.

    logger.info("========== EEG Pipeline Complete ==========")
    logger.info("Dashboard: DASHBOARD_UNIVERSE=%s uv run streamlit run app/dashboard.py",
                config.market_dir)


if __name__ == "__main__":
    main()
