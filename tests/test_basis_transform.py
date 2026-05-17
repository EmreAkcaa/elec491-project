"""Tests for the numéraire transform (Phase 4 mutable-candy).

Pins the canonical invariants of ``apply_numeraire``:

- Zero base = identity (the new returns equal the input returns).
- Constant base (positive constant for prices → log-return zero) is the
  same as zero base.
- Subtracting a constant *log-return* shifts every entry by that
  constant.
- Index alignment: only rows where both panel and base are defined
  survive; mismatched indices raise.
- The composite chain price→log_returns→apply_numeraire is the same as
  divide-then-log_returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.basis_transform import apply_numeraire, numeraire_descriptor, to_log_returns


@pytest.fixture
def panel():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0, 0.01, (10, 3)),
        index=idx,
        columns=list("ABC"),
    )


class TestApplyNumeraire:
    def test_zero_base_is_identity(self, panel):
        base = pd.Series(0.0, index=panel.index, name="zero")
        out = apply_numeraire(panel, base)
        pd.testing.assert_frame_equal(out, panel)

    def test_constant_base_shifts_mean(self, panel):
        base = pd.Series(0.001, index=panel.index, name="const")
        out = apply_numeraire(panel, base)
        diff = (panel - out).mean().mean()
        assert diff == pytest.approx(0.001, rel=1e-9, abs=1e-12)

    def test_alignment_drops_non_overlapping(self, panel):
        # Base shifted far enough that no row overlaps.
        far_idx = pd.date_range("2099-01-01", periods=10, freq="B")
        base = pd.Series(0.0, index=far_idx, name="shifted")
        with pytest.raises(ValueError, match="no overlapping"):
            apply_numeraire(panel, base)

    def test_alignment_keeps_overlap(self, panel):
        # 5 of the panel's 10 rows overlap with the base index.
        overlap_idx = panel.index[5:].append(
            pd.date_range("2099-01-01", periods=20, freq="B")
        )
        base = pd.Series(0.001, index=overlap_idx, name="partial")
        out = apply_numeraire(panel, base)
        assert len(out) == 5
        assert all(idx in panel.index for idx in out.index)

    def test_drops_base_nan_rows(self, panel):
        base = pd.Series(0.001, index=panel.index, name="nans")
        base.iloc[3] = np.nan
        out = apply_numeraire(panel, base)
        assert len(out) == len(panel) - 1
        assert panel.index[3] not in out.index

    def test_subtract_then_recompose(self, panel):
        """Apply numéraire and add base back → recover original panel."""
        base = pd.Series(
            np.linspace(0.001, 0.002, len(panel)),
            index=panel.index, name="lin",
        )
        re_expressed = apply_numeraire(panel, base)
        recovered = re_expressed.add(base.loc[re_expressed.index], axis=0)
        pd.testing.assert_frame_equal(recovered, panel.loc[re_expressed.index])

    def test_price_chain_consistency(self):
        """log(P/B).diff() should equal log(P).diff() − log(B).diff()."""
        idx = pd.date_range("2020-01-01", periods=20, freq="B")
        rng = np.random.default_rng(1)
        prices = pd.DataFrame(
            100.0 * np.exp(rng.normal(0, 0.01, (20, 2)).cumsum(axis=0)),
            index=idx, columns=list("XY"),
        )
        base_prices = pd.Series(
            10.0 * np.exp(rng.normal(0, 0.01, 20).cumsum()),
            index=idx, name="BASE",
        )

        # Direct chain: divide prices, then log-return.
        ratio_prices = prices.div(base_prices, axis=0)
        ratio_returns = to_log_returns(ratio_prices).dropna()

        # Composed chain: log-return both, then subtract.
        panel_rets = to_log_returns(prices)
        base_rets = to_log_returns(base_prices)
        composed = apply_numeraire(panel_rets, base_rets).dropna()

        # Drop NaN rows in either, match index, compare numerically.
        common = ratio_returns.index.intersection(composed.index)
        pd.testing.assert_frame_equal(
            ratio_returns.loc[common],
            composed.loc[common],
            check_exact=False, atol=1e-12,
        )

    def test_wrong_input_type_raises(self):
        with pytest.raises(TypeError, match="must be a DataFrame"):
            apply_numeraire(pd.Series([1.0]), pd.Series([1.0]))
        with pytest.raises(TypeError, match="must be a Series"):
            apply_numeraire(pd.DataFrame({"A": [1.0]}), pd.DataFrame({"B": [1.0]}))


class TestDescriptor:
    def test_descriptor_format(self):
        assert numeraire_descriptor("bist", "usd_try") == "bist_in_usd_try"
        assert numeraire_descriptor("sp500", "gold_usd") == "sp500_in_gold_usd"
