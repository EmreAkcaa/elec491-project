"""CLI entry point: fetch -> validate -> preprocess -> analyze."""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    from src.config import load_config
    from src.data_acquisition import run_acquisition
    from src.data_validation import validate_sample
    from src.preprocessing import run_preprocessing
    from src.analysis import run_analysis
    from src.clustering import run_clustering
    from src.rolling_correlation import run_rolling_analysis
    from src.pair_dislocation import run_pair_dislocation
    from src.rmt_denoising import run_rmt_denoising
    from src.partial_correlation import run_partial_correlation
    from src.wavelet_analysis import run_wavelet_analysis
    from src.transfer_entropy import run_transfer_entropy
    from src.reservoir_computing import run_reservoir_computing

    logger.info("========== StoNeCoAl Pipeline ==========")

    config = load_config()
    logger.info("Config loaded: %s, %d tickers", config.market.market_id, len(config.tickers))

    run_acquisition(config)
    validate_sample(config)
    run_preprocessing(config)
    run_analysis(config)
    run_clustering(config)
    run_rolling_analysis(config)
    run_pair_dislocation(config)

    # --- EEE Analysis Methods ---
    run_rmt_denoising(config)
    run_partial_correlation(config)
    run_wavelet_analysis(config)
    run_transfer_entropy(config)

    # --- Reservoir Computing (prediction layer) ---
    run_reservoir_computing(config)

    logger.info("========== Pipeline Complete ==========")


if __name__ == "__main__":
    main()
