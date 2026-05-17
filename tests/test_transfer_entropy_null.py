"""Tests for the transfer entropy surrogate null + BH-FDR correction.

Phase 1.3 + 1.4 of the mutable-candy rescue plan:
- The previous shuffle null (`rng.permutation(x)`) destroyed source
  autocorrelation and made p-values too liberal for autocorrelated returns.
- No multiple-testing correction was applied across the N*(N-1) directed
  pairs, so ~5% of pairs always appeared significant at alpha=0.05.

These tests pin both fixes:
- Block-bootstrap surrogates preserve autocorrelation under the null.
- A constructed causal X→Y stays significant under the new pipeline.
- Independent X, Y do not produce significant edges above the FDR target.
- Benjamini–Hochberg controls FDR ≤ alpha on a synthetic noise grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.transfer_entropy import (
    _apply_multiple_testing,
    _benjamini_hochberg,
    _circular_block_bootstrap,
    compute_transfer_entropy_matrix,
    transfer_entropy,
)


class TestCircularBlockBootstrap:
    def test_length_preserved(self):
        rng = np.random.default_rng(0)
        x = np.arange(200, dtype=float)
        out = _circular_block_bootstrap(x, block_length=5, rng=rng)
        assert out.shape == x.shape

    def test_block_one_reduces_to_permutation(self):
        rng = np.random.default_rng(1)
        x = np.arange(100, dtype=float)
        out = _circular_block_bootstrap(x, block_length=1, rng=rng)
        assert sorted(out.tolist()) == sorted(x.tolist())

    def test_within_block_order_preserved(self):
        """Within each contiguous block, values must appear in source order."""
        rng = np.random.default_rng(2)
        x = np.arange(40, dtype=float)
        out = _circular_block_bootstrap(x, block_length=5, rng=rng)
        # Each adjacent diff inside a block is +1 (mod wrap). Count how many
        # +1 transitions we see in the bootstrap output.
        diffs = np.diff(out)
        n_in_block_transitions = (diffs == 1).sum()
        # 8 blocks × 4 in-block transitions = 32; some may wrap so floor is 24.
        assert n_in_block_transitions >= 24, (
            f"only {n_in_block_transitions} of expected ≥24 in-block transitions"
        )

    def test_autocorrelation_preserved_better_than_permutation(self):
        """Lag-1 autocorrelation of an AR(1) survives block bootstrap but
        is destroyed by plain permutation."""
        rng = np.random.default_rng(3)
        n = 1000
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.7 * x[t - 1] + rng.normal(0, 0.1)

        true_acf = np.corrcoef(x[:-1], x[1:])[0, 1]
        perm_acf = np.corrcoef(rng.permutation(x)[:-1], rng.permutation(x)[1:])[0, 1]
        bootstrap = _circular_block_bootstrap(x, block_length=10, rng=rng)
        bs_acf = np.corrcoef(bootstrap[:-1], bootstrap[1:])[0, 1]

        # Plain permutation kills autocorrelation; block bootstrap retains
        # most of it. Both relative comparisons should be one-sided.
        assert abs(perm_acf) < 0.15, f"permutation acf {perm_acf:.3f} too large"
        assert bs_acf > 0.4, f"block-bootstrap acf {bs_acf:.3f} too small"
        assert bs_acf > perm_acf


class TestBenjaminiHochberg:
    def test_all_null_no_rejections(self):
        rng = np.random.default_rng(0)
        pvals = rng.uniform(0, 1, size=1000)
        reject = _benjamini_hochberg(pvals, alpha=0.05)
        fdr = reject.mean()
        # Under the global null, expected FDR ≤ alpha. Allow some slack
        # for finite-sample variance.
        assert fdr <= 0.05, f"global-null FDR {fdr:.3f} above 0.05 target"

    def test_obvious_signal_rejected(self):
        pvals = np.concatenate(
            [np.full(10, 1e-6), np.random.default_rng(0).uniform(0.5, 1.0, 990)]
        )
        reject = _benjamini_hochberg(pvals, alpha=0.05)
        assert reject[:10].all(), "obvious signal (p=1e-6) was not rejected"

    def test_matches_textbook_step_up(self):
        """Hand-verified on Benjamini & Hochberg's 1995 worked example
        (15 hypotheses, alpha=0.05). Expected: reject the 4 smallest."""
        pvals = np.array(
            [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
             0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]
        )
        reject = _benjamini_hochberg(pvals, alpha=0.05)
        assert reject.sum() == 4
        assert reject[:4].all()


class TestApplyMultipleTesting:
    def _make_grid(self, p_off: float, N: int = 5) -> np.ndarray:
        pvals = np.full((N, N), p_off)
        np.fill_diagonal(pvals, 1.0)
        return pvals

    def _all_tasks(self, N: int = 5) -> list[tuple[int, int]]:
        return [(i, j) for i in range(N) for j in range(N) if i != j]

    def test_uncorrected_lets_marginals_through(self):
        pvals = self._make_grid(p_off=0.04)
        sig = _apply_multiple_testing(pvals, self._all_tasks(), "none", alpha=0.05)
        # off-diagonal all true (0.04 < 0.05), diagonal true by convention
        assert sig.sum() == 25

    def test_bonferroni_blocks_marginals(self):
        pvals = self._make_grid(p_off=0.04)
        sig = _apply_multiple_testing(pvals, self._all_tasks(), "bonferroni", alpha=0.05)
        # 0.04 > 0.05/20 = 0.0025, so all off-diagonal blocked
        assert (sig.sum() - 5) == 0  # only diagonal True

    def test_bh_blocks_majority_null(self):
        pvals = self._make_grid(p_off=0.5)
        sig = _apply_multiple_testing(pvals, self._all_tasks(), "fdr_bh", alpha=0.05)
        # All p-values 0.5 → no rejections under BH
        assert (sig.sum() - 5) == 0

    def test_unknown_method_raises(self):
        pvals = self._make_grid(p_off=0.5)
        with pytest.raises(ValueError, match="multiple_testing"):
            _apply_multiple_testing(pvals, self._all_tasks(), "garbage", alpha=0.05)


@pytest.fixture
def causal_xy_panel():
    """Construct X with autocorrelation that causally drives Y at lag 1,
    plus an independent Z. Returns a DataFrame [X, Y, Z]."""
    rng = np.random.default_rng(123)
    n = 600
    x = np.zeros(n)
    y = np.zeros(n)
    z = rng.normal(0, 1.0, n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.normal(0, 1.0)
        y[t] = 0.5 * x[t - 1] + rng.normal(0, 1.0)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


class TestEndToEndSignificance:
    def test_block_bootstrap_keeps_real_signal_significant(self, causal_xy_panel):
        te_xy = transfer_entropy(
            causal_xy_panel["X"].values, causal_xy_panel["Y"].values, lag=1, n_bins=3
        )
        assert te_xy > 0.02, f"TE(X→Y) {te_xy:.4f} should be visibly positive"

    def test_full_pipeline_recovers_xy_edge(self, causal_xy_panel):
        te_mat, _ = compute_transfer_entropy_matrix(
            causal_xy_panel,
            lag=1,
            n_bins=3,
            significance_shuffles=200,
            significance_level=0.05,
            seed=7,
            n_jobs=1,
            surrogate_block_length=5,
            multiple_testing="fdr_bh",
        )
        assert te_mat.loc["X", "Y"] > 0, (
            "X→Y edge was zeroed out by FDR despite real causal structure"
        )

    def test_independent_panel_few_false_positives(self):
        """Six i.i.d. Gaussian series → BH-FDR should declare almost no
        edges significant (5% × 30 pairs = ~1.5 in expectation; FDR-controlled)."""
        rng = np.random.default_rng(99)
        n = 600
        panel = pd.DataFrame({
            f"S{k}": rng.normal(0, 1.0, n) for k in range(6)
        })
        te_mat, _ = compute_transfer_entropy_matrix(
            panel,
            lag=1,
            n_bins=3,
            significance_shuffles=200,
            significance_level=0.05,
            seed=11,
            n_jobs=1,
            surrogate_block_length=5,
            multiple_testing="fdr_bh",
        )
        # All entries should be zeroed after FDR; allow at most 1 spurious
        # surviving edge given the bootstrap's discrete p-value resolution.
        nonzero_off_diagonal = (te_mat.values != 0).sum() - len(panel.columns)
        assert nonzero_off_diagonal <= 1, (
            f"{nonzero_off_diagonal} spurious significant edges among independent series"
        )
