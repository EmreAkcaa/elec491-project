"""Numéraire transform: re-express a log-return panel in a new base asset.

Phase 4 of the mutable-candy rescue. ``apply_numeraire`` subtracts the
log-return of the chosen base asset from every ticker's log-return — the
exact equivalent of dividing all prices by the base asset before taking
log-returns. Used to ask the empirical question "how much of BIST co-
movement is FX vs equity?" by comparing TRY, USD and gold variants of
the same panel.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def to_log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Log return = log(P_t / P_{t-1}). First row is NaN by convention."""
    return np.log(prices.astype(float)).diff()


def apply_numeraire(
    log_returns: pd.DataFrame,
    base_log_returns: pd.Series,
) -> pd.DataFrame:
    """Re-express a log-return panel in a different numéraire.

    Mathematically:

        new_returns[t, ticker] = log_returns[t, ticker] − base_log_returns[t]

    which is identically log( price[t, ticker] / base[t] ) - log( price[t-1, ticker] / base[t-1] ).

    Aligns on the intersection of the panel's index and the base series'
    index, drops rows where the base is NaN, and returns a DataFrame of
    the same columns as ``log_returns``.
    """
    if not isinstance(log_returns, pd.DataFrame):
        raise TypeError(
            f"log_returns must be a DataFrame, got {type(log_returns).__name__}"
        )
    if not isinstance(base_log_returns, pd.Series):
        raise TypeError(
            f"base_log_returns must be a Series, got "
            f"{type(base_log_returns).__name__}"
        )

    common = log_returns.index.intersection(base_log_returns.index)
    if common.empty:
        raise ValueError(
            "log_returns and base_log_returns have no overlapping dates"
        )

    base_aligned = base_log_returns.loc[common].dropna()
    panel_aligned = log_returns.loc[base_aligned.index]
    # Subtract element-wise broadcast across columns.
    out = panel_aligned.sub(base_aligned, axis=0)

    dropped = len(log_returns) - len(out)
    if dropped:
        logger.info(
            "apply_numeraire: kept %d / %d rows after aligning to base index "
            "(dropped %d non-overlapping or base-NaN rows)",
            len(out), len(log_returns), dropped,
        )
    return out


def numeraire_descriptor(label: str, base_name: str) -> str:
    """Stable human-readable tag for the resulting basis (e.g. for filenames
    or chart captions): \"bist_in_usd\", \"sp500_in_gold\"."""
    return f"{label}_in_{base_name}"
