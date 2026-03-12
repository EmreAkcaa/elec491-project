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

    logger.info("========== StoNeCoAl Pipeline ==========")

    config = load_config()
    logger.info("Config loaded: %s, %d tickers", config.market.market_id, len(config.tickers))

    run_acquisition(config)
    validate_sample(config)
    run_preprocessing(config)
    run_analysis(config)
    run_clustering(config)
    run_rolling_analysis(config)

    logger.info("========== Pipeline Complete ==========")


if __name__ == "__main__":
    main()
