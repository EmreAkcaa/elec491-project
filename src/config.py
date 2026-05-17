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
    source: str = "yfinance"  # yfinance | physionet — drives src/eeg_acquisition vs src/data_acquisition


@dataclass
class PreprocessingConfig:
    min_coverage_pct: float
    anomaly_return_threshold: float
    forward_fill: bool
    manual_anomaly_nulls: list = field(default_factory=list)
    # ^ List of [ticker, "YYYY-MM-DD"] entries. The (ticker, date) cell in
    # `log_returns` is set to NaN before anomaly detection runs. Use to mask
    # unhandled corporate actions that yfinance Adj-Close failed to back-
    # adjust (e.g. CCOLA 2024-08-01 10.81x bonus issue). Old YAMLs without
    # this key continue to load — default factory yields [].


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
class TransferEntropyConfig:
    lag: int = 1
    n_bins: int = 3
    significance_shuffles: int = 100
    significance_level: float = 0.05
    seed: int = 42


@dataclass
class EEGConfig:
    """PhysioNet EEG-Motor-Imagery-specific knobs (Phase F)."""
    task_type: str = "left_right"       # left_right | feet_fists | baseline
    subject_ids: list = field(default_factory=lambda: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    runs_per_condition: list = field(default_factory=lambda: [4, 8, 12])  # left_right runs
    sampling_rate_hz: int = 160
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 50.0
    notch_hz: float = 50.0              # 50 = Türkiye / EU grid; 60 = US
    car_reference: bool = True
    cache_raw: bool = True


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
    transfer_entropy: TransferEntropyConfig = field(default_factory=TransferEntropyConfig)
    eeg: EEGConfig = field(default_factory=EEGConfig)

    @property
    def tickers(self) -> list[str]:
        return self.universe["ticker"].tolist()

    @property
    def provider_symbols(self) -> list[str]:
        return self.universe["provider_symbol"].tolist()

    @property
    def index_provider_symbol(self) -> str:
        return self.market.index_ticker + self.market.provider_suffix

    # ---- Per-market data path properties (Phase D) ----
    # Each market gets its own data/<market_dir>/{raw,processed,results} tree
    # so multiple universes (BIST, S&P 500, EEG) can coexist on disk.
    @property
    def market_dir(self) -> str:
        """Lower-cased market_id used as the filesystem namespace."""
        return self.market.market_id.lower()

    @property
    def data_raw(self) -> Path:
        return PROJECT_ROOT / "data" / self.market_dir / "raw"

    @property
    def data_processed(self) -> Path:
        return PROJECT_ROOT / "data" / self.market_dir / "processed"

    @property
    def data_results(self) -> Path:
        return PROJECT_ROOT / "data" / self.market_dir / "results"


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
    transfer_entropy_raw = raw.get("transfer_entropy", {})
    eeg_raw = raw.get("eeg", {})

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
        transfer_entropy=TransferEntropyConfig(**transfer_entropy_raw),
        eeg=EEGConfig(**eeg_raw),
    )

    return config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    print(f"Market: {cfg.market.market_id}")
    print(f"Tickers: {len(cfg.tickers)}")
    print(f"Date range: {cfg.data.start_date} → {cfg.data.end_date}")
    print(f"Coverage threshold: {cfg.preprocessing.min_coverage_pct}")
