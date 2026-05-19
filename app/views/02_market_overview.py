"""Market Overview — full-period static analysis (5 sub-tabs).

PHASE 2 page wrapper. Loads the universe-keyed log returns + adj close
panels and delegates to `market_overview.render()` (extracted from
dashboard.py in Stage 1 of this PR). The page itself owns the date-range
picker in its header strip — see `market_overview.render()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import current_universe, load_adj_close, load_log_returns
from universe_registry import get_universe
from market_overview import render

adj_close = load_adj_close()
full_returns = load_log_returns()
min_date = adj_close.index.min().date()
max_date = adj_close.index.max().date()
active_universe = get_universe(current_universe())

render(
    full_returns=full_returns,
    adj_close=adj_close,
    min_date=min_date,
    max_date=max_date,
    active_universe=active_universe,
)
