"""Generate BIST_USD and BIST_GOLD universe variants from base BIST data.

Phase 4 of the mutable-candy rescue. The script:

1. Loads BIST adjusted-close prices and log-returns.
2. Fetches USDTRY and gold base assets (cached after first run).
3. Re-expresses each ticker's price series in USD (divide by USDTRY) and
   in gold-TRY (divide by gold_usd * USDTRY). Recomputes log-returns
   directly from the re-expressed prices so apply_numeraire and the
   chain-rule match exactly.
4. Writes the resulting processed/ tree for the two variant universes.
5. Runs analysis → clustering → rolling → dislocation → RMT → Glasso →
   wavelet → transfer-entropy → info-theory on each variant.

Usage:
    uv run python scripts/run_basis_variants.py             # both variants
    uv run python scripts/run_basis_variants.py --only usd  # USD only
    uv run python scripts/run_basis_variants.py --skip-pipeline  # just write processed/

Run-time on a recent laptop: ~5-10 min per variant (dominated by
transfer-entropy parallelism). The base currency transform itself is < 1 s.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.basis_transform import to_log_returns
from src.config import PROJECT_ROOT, load_config
from src.numeraire_acquisition import (
    fetch_all_base_assets,
    gold_in_try,
    load_base_asset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


VARIANT_CONFIGS = {
    "usd": "config/settings_bist_usd.yaml",
    "gold": "config/settings_bist_gold.yaml",
}


def _ensure_base_assets(start: str, end: str) -> None:
    """Fetch USDTRY + gold_usd if their cached parquets are missing."""
    try:
        load_base_asset("usd_try")
        load_base_asset("gold_usd")
    except FileNotFoundError:
        logger.info("Base assets missing — fetching from yfinance.")
        fetch_all_base_assets(start, end)


def _build_re_expressed_panel(
    adj_close: pd.DataFrame,
    base_series: pd.Series,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide every ticker's price by the aligned base series, then
    recompute log-returns. Returns (new_adj_close, new_log_returns)."""
    common = adj_close.index.intersection(base_series.index)
    if common.empty:
        raise RuntimeError(
            f"No overlapping dates between BIST prices and {label} base series."
        )
    base_aligned = base_series.loc[common].astype(float).replace(0, np.nan).dropna()
    panel_aligned = adj_close.loc[base_aligned.index]
    re_expressed = panel_aligned.div(base_aligned, axis=0)
    log_returns = to_log_returns(re_expressed).dropna(how="all")
    logger.info(
        "%s: kept %d / %d rows; %d → %d valid tickers",
        label, len(re_expressed), len(adj_close),
        adj_close.shape[1], re_expressed.shape[1],
    )
    return re_expressed, log_returns


def _materialise_variant(
    variant_name: str,
    new_adj_close: pd.DataFrame,
    new_log_returns: pd.DataFrame,
    source_dir: Path,
    target_dir: Path,
) -> None:
    """Write the variant's processed/ tree. Re-uses BIST's coverage report
    and validation report (the universe membership is the same)."""
    proc = target_dir / "processed"
    raw = target_dir / "raw"
    proc.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    new_adj_close.to_parquet(proc / "adj_close.parquet")
    new_log_returns.to_parquet(proc / "log_returns.parquet")
    new_adj_close.to_parquet(proc / "raw_close.parquet")  # raw == adj for the variant

    for fname in ("coverage_report.csv", "anomalies.csv", "validation_report.csv"):
        src = source_dir / "processed" / fname
        if src.exists():
            shutil.copy2(src, proc / fname)

    # Anomaly mask file may not exist when there are no anomalies; that's fine.
    logger.info("%s: processed/ written under %s", variant_name, proc)


def _run_pipeline_stages(config) -> None:
    """Run analysis → ... → info_theory on a variant config. Skips
    acquisition + validation + preprocessing because we wrote the
    processed panel ourselves."""
    from src.analysis import run_analysis
    from src.clustering import run_clustering
    from src.rolling_correlation import run_rolling_analysis
    from src.pair_dislocation import run_pair_dislocation
    from src.rmt_denoising import run_rmt_denoising
    from src.partial_correlation import run_partial_correlation
    from src.wavelet_analysis import run_wavelet_analysis
    from src.transfer_entropy import run_transfer_entropy
    from src.info_theory import run_info_theory

    run_analysis(config)
    run_clustering(config)
    run_rolling_analysis(config)
    run_pair_dislocation(config)
    run_rmt_denoising(config)
    run_partial_correlation(config)
    run_wavelet_analysis(config)
    run_transfer_entropy(config)
    run_info_theory(config)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="BIST base currency variant runner.")
    parser.add_argument(
        "--only",
        choices=["usd", "gold"],
        default=None,
        help="Run only one variant (default: both).",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Just write the processed/ tree; don't run analysis…info_theory.",
    )
    args = parser.parse_args(argv)

    bist_cfg = load_config("config/settings.yaml")
    bist_dir = PROJECT_ROOT / "data" / "bist"
    adj_close = pd.read_parquet(bist_dir / "processed" / "adj_close.parquet")
    logger.info(
        "Loaded BIST adj_close: %d days × %d tickers (%s → %s)",
        len(adj_close), adj_close.shape[1],
        adj_close.index.min().date(), adj_close.index.max().date(),
    )

    _ensure_base_assets(start="2019-12-01", end=bist_cfg.data.end_date)
    usd_try = load_base_asset("usd_try")
    gold_try = gold_in_try(start="2019-12-01", end=bist_cfg.data.end_date)

    variants = ["usd", "gold"] if args.only is None else [args.only]

    for variant in variants:
        cfg_path = VARIANT_CONFIGS[variant]
        variant_cfg = load_config(cfg_path)
        target_dir = PROJECT_ROOT / "data" / variant_cfg.market_dir
        base = usd_try if variant == "usd" else gold_try
        label = f"BIST in {'USD' if variant == 'usd' else 'gold (XAU/TRY)'}"
        new_adj, new_log = _build_re_expressed_panel(adj_close, base, label)
        _materialise_variant(
            variant_name=variant,
            new_adj_close=new_adj,
            new_log_returns=new_log,
            source_dir=bist_dir,
            target_dir=target_dir,
        )
        if args.skip_pipeline:
            continue
        logger.info("=== Running pipeline stages for %s ===", variant_cfg.market.market_id)
        _run_pipeline_stages(variant_cfg)
        logger.info("=== %s complete ===", variant_cfg.market.market_id)


if __name__ == "__main__":
    main()
