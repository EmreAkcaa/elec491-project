"""Signals — pair signal leaderboard + cross-asset breakout (BIST).

PHASE 2 page wrapper. Hidden from the nav when the active universe has
``has_pair_trading=False`` (e.g., EEG). Delegates to ``signals.render()``
which handles its own input loading (adj_close, dislocation candidates,
cross-asset summary) via cached loaders in ``utils``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from signals import render

render()
