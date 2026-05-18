"""CLI entry point: fetch -> validate -> preprocess -> analyze.

Usage:
    uv run python run_pipeline.py                                    # BIST (default)
    uv run python run_pipeline.py --config config/settings_sp500.yaml  # S&P 500
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None):
    from src.config import load_config
    from src.data_acquisition import run_acquisition
    from src.data_validation import validate_sample
    from src.preprocessing import run_preprocessing
    from src.analysis import run_analysis
    from src.clustering import run_clustering
    from src.rolling_correlation import run_rolling_analysis
    from src.pair_dislocation import run_pair_dislocation
    from src.pit_snapshots import run_pit_snapshots
    from src.rmt_denoising import run_rmt_denoising
    from src.partial_correlation import run_partial_correlation
    from src.wavelet_analysis import run_wavelet_analysis
    from src.transfer_entropy import run_transfer_entropy
    from src.info_theory import run_info_theory

    parser = argparse.ArgumentParser(description="StoNeCoAl pipeline.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a settings YAML (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--skip-pit",
        action="store_true",
        help=(
            "Skip the PIT snapshot grid stage (Phase 3 slim). "
            "Useful for dev iterations where you don't need to refresh "
            "Time Machine's instant-scrubbing cache."
        ),
    )
    args = parser.parse_args(argv)

    logger.info("========== StoNeCoAl Pipeline ==========")

    config = load_config(args.config)
    logger.info(
        "Config loaded: %s (market_dir=%s), %d tickers, results -> %s",
        config.market.market_id,
        config.market_dir,
        len(config.tickers),
        config.data_results,
    )

    run_acquisition(config)
    validate_sample(config)
    run_preprocessing(config)
    run_analysis(config)
    run_clustering(config)
    run_rolling_analysis(config)
    run_pair_dislocation(config)

    # --- Phase 3 (slim) — PIT snapshot grid for Time Machine ---
    # Gated to bist + sp500 only (other markets skip silently).
    # Skip via --skip-pit for dev iterations.
    if not args.skip_pit:
        run_pit_snapshots(config)
    else:
        logger.info("PIT snapshots skipped via --skip-pit flag")

    # --- EEE Analysis Methods ---
    run_rmt_denoising(config)
    run_partial_correlation(config)
    run_wavelet_analysis(config)
    run_transfer_entropy(config)
    run_info_theory(config)

    # --- Spiking Neural Network (pair-signal classifier) ---
    # Optional: requires torch + snntorch (install with `uv sync --extra snn`).
    # The pipeline completes regardless of whether torch is installed.
    try:
        from src.snn_signals import run_snn_signals
        run_snn_signals(config)
    except ImportError as exc:
        logger.warning(
            "SNN step skipped: %s. Install with `uv sync --extra snn` to enable.",
            exc,
        )

    logger.info("========== Pipeline Complete ==========")


if __name__ == "__main__":
    main()
