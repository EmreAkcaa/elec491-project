"""Cross-Market — BIST 100 vs S&P 500 side-by-side comparison.

PHASE 2 page wrapper. Delegates to `cross_market.render()` which is
universe-independent (it reads BOTH BIST and S&P data directly via
the `_load_*(universe)` underscore-prefixed loaders in app/utils.py).
This page is hidden from the nav when the active universe has
`eligible_for_cross_market=False` (gating handled in main.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cross_market import render

render()
