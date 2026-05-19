"""Methods Lab — RMT / GLASSO / Wavelet / TE / IT / SNN methodology depth.

PHASE 2 page wrapper. `methods_lab.render()` is a thin wrapper around
`eee_analysis.render()` which manages its own loaders + has Phase Y
lazy-rendered sub-tabs via `render_subtabs()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from methods_lab import render

render()
