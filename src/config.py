"""Configuration loader for the StoNeCoAl pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class MarketConfig:
    market_id: str
    universe_file: str
    index_ticker: str
    provider_suffix: str
    currency: str


@dataclass
class DataConfig:
    start_date: str
    end_date: str
    download_interval: str
    store_raw_close: bool


@dataclass
class PreprocessingConfig:
    min_coverage_pct: float
    anomaly_return_threshold: float
    forward_fill: bool


@dataclass
class AnalysisConfig:
    correlation_method: str
    annualization_factor: int
    corr_min_periods: int


@dataclass
class ValidationConfig:
    enabled: bool
    sample_size: int


@dataclass
class RollingConfig:
    windows: list = field(default_factory=lambda: [60, 120, 252])
    step: int = 5
    method: str = "pearson"
    min_periods_ratio: float = 0.6


@dataclass
class DislocationConfig:
    zscore_window: int = 60
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    min_half_life: int = 5
    max_half_life: int = 252
    top_n_candidates: int = 20
    lookback_window: int = 252
    min_correlation: float = 0.5


@dataclass
class PipelineConfig:
    market: MarketConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    analysis: AnalysisConfig
    validation: ValidationConfig
    universe: pd.DataFrame = field(repr=False)
    universe_metadata: dict = field(default_factory=dict)
    rolling: RollingConfig = field(default_factory=RollingConfig)
    dislocation: DislocationConfig = field(default_factory=DislocationConfig)

    @property
    def tickers(self) -> list[str]:
        return self.universe["ticker"].tolist()

    @property
    def provider_symbols(self) -> list[str]:
        return self.universe["provider_symbol"].tolist()

    @property
    def index_provider_symbol(self) -> str:
        return self.market.index_ticker + self.market.provider_suffix


def _load_universe(universe_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load and validate the universe CSV."""
    df = pd.read_csv(universe_path)

    required_cols = {"ticker", "company_name", "sector", "provider_symbol"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Universe CSV missing columns: {missing}")

    # Validate uniqueness of instrument_id (ticker)
    dupes = df[df["ticker"].duplicated(keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"Duplicate instrument_id (ticker) values: {dupes['ticker'].tolist()}"
        )

    count = len(df)
    metadata = {
        "source": str(universe_path),
        "freeze_date": pd.Timestamp.now().isoformat(),
        "count": count,
    }

    if count != 100:
        logger.warning(
            "Universe has %d tickers (expected 100). "
            "This is logged in metadata but not a hard error.",
            count,
        )

    logger.info("Loaded universe: %d tickers from %s", count, universe_path.name)
    return df, metadata


def load_config(
    settings_path: str | Path | None = None,
) -> PipelineConfig:
    """Load pipeline configuration from YAML and universe CSV."""
    if settings_path is None:
        settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    else:
        settings_path = Path(settings_path)

    with open(settings_path) as f:
        raw = yaml.safe_load(f)

    universe_path = PROJECT_ROOT / raw["market"]["universe_file"]
    universe_df, universe_meta = _load_universe(universe_path)

    rolling_raw = raw.get("rolling", {})
    dislocation_raw = raw.get("dislocation", {})

    config = PipelineConfig(
        market=MarketConfig(**raw["market"]),
        data=DataConfig(**raw["data"]),
        preprocessing=PreprocessingConfig(**raw["preprocessing"]),
        analysis=AnalysisConfig(**raw["analysis"]),
        validation=ValidationConfig(**raw["validation"]),
        universe=universe_df,
        universe_metadata=universe_meta,
        rolling=RollingConfig(**rolling_raw),
        dislocation=DislocationConfig(**dislocation_raw),
    )

    return config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    print(f"Market: {cfg.market.market_id}")
    print(f"Tickers: {len(cfg.tickers)}")
    print(f"Date range: {cfg.data.start_date} → {cfg.data.end_date}")
    print(f"Coverage threshold: {cfg.preprocessing.min_coverage_pct}")
