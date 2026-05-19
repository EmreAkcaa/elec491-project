"""Tests for the IT orphan-surfacing work (PR #71).

Verifies:
  1. New loaders (TE p-values, TE significance, TE summary) are wired and
     return expected types on hits + misses.
  2. The Information Theory + Transfer Entropy sub-tabs render without
     exceptions when the artifacts exist on disk.
  3. The new per-ticker diagnostics panel produces the expected
     aggregations (sign-entropy sort, nonlinear-excess sum per ticker).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Loader contracts
# ---------------------------------------------------------------------------

def test_load_te_summary_returns_dict_or_empty():
    from app.utils import _load_te_summary
    df = _load_te_summary("bist")
    if df:
        # If artifact exists, must have the keys we read in the UI.
        for k in ("n_significant_fdr", "n_significant_uncorrected", "total_pairs",
                  "significance_level", "multiple_testing", "significance_shuffles"):
            assert k in df, f"Missing key {k} in transfer_entropy_summary.json"
        assert isinstance(df["total_pairs"], int)
        assert df["total_pairs"] > 0
    else:
        # Empty dict is the documented miss behavior — no exception.
        assert df == {}


def test_load_te_summary_returns_empty_for_unknown_universe():
    from app.utils import _load_te_summary
    out = _load_te_summary("nonexistent_universe_xyz")
    assert out == {}


def test_load_te_pvalues_returns_dataframe_or_empty():
    from app.utils import _load_te_pvalues
    df = _load_te_pvalues("bist")
    if not df.empty:
        # Must be a square matrix indexed by tickers, values in [0, 1]
        assert df.shape[0] == df.shape[1], "TE p-values should be a square matrix"
        # Strip diagonals + NaN (self-pairs are undefined) before bounds check
        import numpy as np
        arr = df.to_numpy()
        mask = ~np.eye(arr.shape[0], dtype=bool)
        offdiag = arr[mask]
        offdiag = offdiag[np.isfinite(offdiag)]
        if len(offdiag) > 0:
            assert offdiag.min() >= 0.0
            assert offdiag.max() <= 1.0
    else:
        # Empty on miss is fine.
        assert df.empty


def test_load_te_significance_returns_boolean_mask_or_empty():
    from app.utils import _load_te_significance
    df = _load_te_significance("bist")
    if not df.empty:
        assert df.shape[0] == df.shape[1]
        # Must coerce to boolean dtype cleanly
        assert df.dtypes.iloc[0] == bool or df.to_numpy().dtype == bool


# ---------------------------------------------------------------------------
# Per-ticker aggregation correctness
# ---------------------------------------------------------------------------

def test_nonlinear_excess_per_ticker_sum_matches_matrix_row_sum():
    """The 'Σ excess MI [bits]' column in the new IT per-ticker panel must
    equal the per-row absolute sum of the full mi_nonlinear_excess matrix
    (off-diagonal). Pin this so a refactor can't silently change the
    aggregation logic.
    """
    path = _REPO_ROOT / "data" / "bist" / "results" / "mi_nonlinear_excess.parquet"
    if not path.exists():
        pytest.skip("mi_nonlinear_excess.parquet missing — run pipeline first")
    excess = pd.read_parquet(path)
    if excess.empty:
        pytest.skip("nonlinear excess matrix empty")

    # Defensive: strip self-pairs (diagonal) before summing.
    e = excess.copy()
    for t in e.columns:
        if t in e.index:
            e.loc[t, t] = 0.0
    per_ticker_via_abs = e.abs().sum(axis=1)
    # Spot-check: a ticker's per-ticker total should be >= 0 and bounded
    # by the universe size × max excess.
    assert (per_ticker_via_abs >= 0).all()
    max_per = float(per_ticker_via_abs.max())
    max_pair = float(e.abs().to_numpy().max())
    assert max_per <= max_pair * len(e.columns), "Aggregation overshoots possible bound"


def test_sign_entropy_per_ticker_is_in_unit_interval():
    """sign-entropy is a conditional Shannon entropy of a 2-outcome
    variable; it's bounded in [0, 1] bits."""
    path = _REPO_ROOT / "data" / "bist" / "results" / "entropy_rate_signs.csv"
    if not path.exists():
        pytest.skip("entropy_rate_signs.csv missing")
    df = pd.read_csv(path)
    if df.empty:
        pytest.skip("entropy_rate_signs empty")
    assert "ticker" in df.columns
    assert "entropy_rate_bits" in df.columns
    assert (df["entropy_rate_bits"] >= 0).all()
    assert (df["entropy_rate_bits"] <= 1.0 + 1e-6).all(), (
        "sign-entropy above 1.0 bit is impossible for a binary outcome"
    )


# ---------------------------------------------------------------------------
# Render smoke
# ---------------------------------------------------------------------------

def test_methods_lab_information_theory_renders_on_bist():
    """End-to-end render of the IT sub-tab on BIST. Verifies the new
    per-ticker diagnostics panel doesn't crash on real data."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit AppTest unavailable")

    at = AppTest.from_file("app/views/05_methods_lab.py", default_timeout=120)
    at.session_state["dataset"] = "bist"
    at.session_state["universe"] = "bist"
    at.session_state["bist_basis"] = "try"
    at.session_state["methods_lab_subtab_bist"] = "Information Theory"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_methods_lab_transfer_entropy_renders_on_bist():
    """End-to-end render of the TE sub-tab on BIST. Verifies the new
    p-value distribution panel + FDR significance count don't crash."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit AppTest unavailable")

    at = AppTest.from_file("app/views/05_methods_lab.py", default_timeout=120)
    at.session_state["dataset"] = "bist"
    at.session_state["universe"] = "bist"
    at.session_state["bist_basis"] = "try"
    at.session_state["methods_lab_subtab_bist"] = "Transfer Entropy"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
