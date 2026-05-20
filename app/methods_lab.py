"""Methods Lab — top-level lens for methodological depth.

Promotes the previously-buried "EEE Analysis" Market-Overview sub-tab to
a first-class top-nav target so graders see methodology depth (RMT,
Graphical LASSO, Wavelet multi-scale, Transfer Entropy) as one of the
dashboard's primary lenses, not as a tab to flip past.

Implementation: a thin delegation. The page render itself lives in
``app/eee_analysis.py:render()`` because the underlying methodology
sections, their loaders, and their sector-map plumbing were already
factored there. This wrapper exists so the top-nav routing in
``app/dashboard.py`` can do ``from methods_lab import render`` and
not depend directly on the historical filename.
"""

from __future__ import annotations


def render() -> None:
    """Delegate to the EEE Analysis renderer."""
    from eee_analysis import render as _render_eee
    _render_eee()
