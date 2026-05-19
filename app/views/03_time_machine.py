"""Time Machine — date-driven correlation network analysis.

PHASE 2 page wrapper. Phase 3 (slim) precomputed PIT snapshots make
this near-instant on BIST(TRY) + S&P at window=252. Other universes
fall back to live compute transparently — see time_machine.render().
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import load_adj_close, load_log_returns
from time_machine import render

adj_close = load_adj_close()
full_returns = load_log_returns()
min_date = adj_close.index.min().date()
max_date = adj_close.index.max().date()

render(adj_close, full_returns, min_date, max_date)
