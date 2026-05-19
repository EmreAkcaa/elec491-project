"""Pair Analysis — two-leg deep-dive on a ticker pair.

PHASE 2 page wrapper. Hidden from the nav when the active universe
has `has_pair_trading=False` (e.g., EEG). Loads coverage + adj_close +
log_returns then delegates to `pair_analysis.render(...)`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import load_adj_close, load_coverage, load_log_returns
from pair_analysis import render

adj_close = load_adj_close()
full_returns = load_log_returns()
coverage_df = load_coverage()
min_date = adj_close.index.min().date()
max_date = adj_close.index.max().date()

render(adj_close, full_returns, coverage_df, min_date, max_date)
