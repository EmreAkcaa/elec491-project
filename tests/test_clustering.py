"""Tests for clustering and network module using synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    build_mst,
    compute_linkage,
    compute_mst_metrics,
    get_cluster_assignments,
    get_leaf_order,
    mst_to_edge_df,
)


@pytest.fixture
def synthetic_returns():
    """Create synthetic log returns for 5 tickers with known structure."""
    rng = np.random.default_rng(123)
    n_days = 500
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    # A,B highly correlated; C,D moderately; E independent
    base = rng.normal(0, 0.02, n_days)
    returns = pd.DataFrame(
        {
            "A": base + rng.normal(0, 0.005, n_days),
            "B": base + rng.normal(0, 0.005, n_days),
            "C": 0.5 * base + rng.normal(0, 0.015, n_days),
            "D": 0.5 * base + rng.normal(0, 0.015, n_days),
            "E": rng.normal(0, 0.02, n_days),
        },
        index=dates,
    )
    return returns


@pytest.fixture
def corr_matrix(synthetic_returns):
    return synthetic_returns.corr()


@pytest.fixture
def dist_matrix(corr_matrix):
    return np.sqrt(2 * (1 - corr_matrix))


class TestLinkage:
    def test_shape(self, dist_matrix):
        Z, labels = compute_linkage(dist_matrix)
        n = len(labels)
        assert Z.shape == (n - 1, 4)

    def test_labels_match(self, dist_matrix):
        Z, labels = compute_linkage(dist_matrix)
        assert labels == dist_matrix.columns.tolist()

    def test_linkage_methods(self, dist_matrix):
        """All supported methods should produce valid linkage."""
        for method in ["single", "complete", "average"]:
            Z, labels = compute_linkage(dist_matrix, method=method)
            assert Z.shape[0] == len(labels) - 1


class TestLeafOrder:
    def test_all_tickers_present(self, dist_matrix):
        Z, labels = compute_linkage(dist_matrix)
        order = get_leaf_order(Z, labels)
        assert set(order) == set(labels)
        assert len(order) == len(labels)

    def test_no_duplicates(self, dist_matrix):
        Z, labels = compute_linkage(dist_matrix)
        order = get_leaf_order(Z, labels)
        assert len(order) == len(set(order))

    def test_correlated_tickers_adjacent(self, dist_matrix):
        """A and B (highly correlated) should be close in leaf order."""
        Z, labels = compute_linkage(dist_matrix, method="single")
        order = get_leaf_order(Z, labels)
        idx_a = order.index("A")
        idx_b = order.index("B")
        assert abs(idx_a - idx_b) <= 2


class TestClusterAssignments:
    def test_all_tickers_assigned(self, dist_matrix):
        Z, labels = compute_linkage(dist_matrix)
        clusters = get_cluster_assignments(Z, labels, n_clusters=3)
        assert len(clusters) == len(labels)
        assert set(clusters["ticker"]) == set(labels)

    def test_n_clusters(self, dist_matrix):
        Z, labels = compute_linkage(dist_matrix)
        clusters = get_cluster_assignments(Z, labels, n_clusters=3)
        assert clusters["cluster_id"].nunique() <= 3

    def test_ab_same_cluster(self, dist_matrix):
        """A and B should end up in the same cluster."""
        Z, labels = compute_linkage(dist_matrix)
        clusters = get_cluster_assignments(Z, labels, n_clusters=3)
        cluster_a = clusters.loc[clusters["ticker"] == "A", "cluster_id"].iloc[0]
        cluster_b = clusters.loc[clusters["ticker"] == "B", "cluster_id"].iloc[0]
        assert cluster_a == cluster_b

    def test_distance_threshold(self, dist_matrix):
        Z, labels = compute_linkage(dist_matrix)
        clusters = get_cluster_assignments(Z, labels, distance_threshold=0.5)
        assert clusters["cluster_id"].nunique() >= 1


class TestMST:
    def test_node_count(self, dist_matrix):
        mst = build_mst(dist_matrix)
        assert mst.number_of_nodes() == len(dist_matrix)

    def test_edge_count(self, dist_matrix):
        """MST should have exactly N-1 edges."""
        mst = build_mst(dist_matrix)
        assert mst.number_of_edges() == len(dist_matrix) - 1

    def test_connected(self, dist_matrix):
        mst = build_mst(dist_matrix)
        assert nx.is_connected(mst)

    def test_weights_positive(self, dist_matrix):
        mst = build_mst(dist_matrix)
        for _, _, data in mst.edges(data=True):
            assert data["weight"] >= 0

    def test_ab_edge_exists(self, dist_matrix):
        """A-B have smallest distance, so should be in MST."""
        mst = build_mst(dist_matrix)
        assert mst.has_edge("A", "B")


class TestMSTMetrics:
    def test_columns(self, dist_matrix):
        mst = build_mst(dist_matrix)
        metrics = compute_mst_metrics(mst)
        assert "ticker" in metrics.columns
        assert "degree" in metrics.columns
        assert "betweenness_centrality" in metrics.columns

    def test_all_nodes(self, dist_matrix):
        mst = build_mst(dist_matrix)
        metrics = compute_mst_metrics(mst)
        assert len(metrics) == mst.number_of_nodes()

    def test_degree_sum(self, dist_matrix):
        """Sum of degrees should equal 2 * number of edges."""
        mst = build_mst(dist_matrix)
        metrics = compute_mst_metrics(mst)
        assert metrics["degree"].sum() == 2 * mst.number_of_edges()

    def test_betweenness_range(self, dist_matrix):
        mst = build_mst(dist_matrix)
        metrics = compute_mst_metrics(mst)
        assert (metrics["betweenness_centrality"] >= 0).all()
        assert (metrics["betweenness_centrality"] <= 1).all()


class TestMSTEdgeDf:
    def test_edge_count(self, dist_matrix, corr_matrix):
        mst = build_mst(dist_matrix)
        edges = mst_to_edge_df(mst, corr_matrix)
        assert len(edges) == mst.number_of_edges()

    def test_columns(self, dist_matrix, corr_matrix):
        mst = build_mst(dist_matrix)
        edges = mst_to_edge_df(mst, corr_matrix)
        assert set(edges.columns) == {"source", "target", "distance", "correlation"}


# Need to import networkx for the connectivity test
import networkx as nx


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIST_RESULTS = PROJECT_ROOT / "data" / "bist" / "results"
SP500_RESULTS = PROJECT_ROOT / "data" / "sp500" / "results"
BIST_UNIVERSE = PROJECT_ROOT / "config" / "universes" / "bist100.csv"
SP500_UNIVERSE = PROJECT_ROOT / "config" / "universes" / "sp500_full.csv"


def _load_artifact_clusters(results_dir: Path, universe_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_dir / "cluster_assignments.csv")
    if "sector" not in df.columns or df["sector"].isna().any():
        uni = pd.read_csv(universe_path)
        df = df.merge(uni[["ticker", "sector"]], on="ticker", how="left", suffixes=("_old", ""))
    return df.dropna(subset=["sector", "cluster_id"])


@pytest.mark.skipif(
    not (BIST_RESULTS / "cluster_assignments.csv").exists(),
    reason="BIST pipeline artifacts not present",
)
class TestBistClusteringPostFix:
    """Regression tests pinning the Phase 1.1 clustering fix.

    Before: single-linkage + distance_threshold=1.0 → 45 of 73 tickers in one
    cluster across 17 sectors; ARI vs sector ≈ 0.034.
    After: ward + n_clusters=20 → max cluster ≈ 12; banks co-cluster;
    ARI ≈ 0.27, NMI ≈ 0.72.
    """

    def test_max_cluster_under_15(self):
        df = _load_artifact_clusters(BIST_RESULTS, BIST_UNIVERSE)
        max_size = df.groupby("cluster_id").size().max()
        assert max_size <= 15, f"max cluster size {max_size} > 15 (chaining returned)"

    def test_banks_mostly_co_cluster(self):
        df = _load_artifact_clusters(BIST_RESULTS, BIST_UNIVERSE)
        banks = ["AKBNK", "GARAN", "YKBNK", "VAKBN", "HALKB", "SKBNK", "ISCTR"]
        bank_rows = df[df["ticker"].isin(banks)]
        cluster_counts = bank_rows["cluster_id"].value_counts()
        # at least 6 of the 7 BIST major banks should share a single cluster
        assert cluster_counts.iloc[0] >= 6, (
            f"only {cluster_counts.iloc[0]} of {len(bank_rows)} banks "
            f"share the largest bank-cluster (distribution: {cluster_counts.to_dict()})"
        )

    def test_ari_recovers_sector_structure(self):
        from sklearn.metrics import adjusted_rand_score

        df = _load_artifact_clusters(BIST_RESULTS, BIST_UNIVERSE)
        ari = adjusted_rand_score(df["sector"], df["cluster_id"])
        assert ari >= 0.20, (
            f"BIST sector-recovery ARI {ari:.3f} below 0.20 floor "
            "(pre-fix baseline was ~0.034)"
        )


@pytest.mark.skipif(
    not (SP500_RESULTS / "cluster_assignments.csv").exists(),
    reason="S&P pipeline artifacts not present",
)
class TestSP500ClusteringPostFix:
    """Mirror of TestBistClusteringPostFix for the S&P 500 universe."""

    def test_max_cluster_under_50(self):
        df = _load_artifact_clusters(SP500_RESULTS, SP500_UNIVERSE)
        max_size = df.groupby("cluster_id").size().max()
        assert max_size <= 50, (
            f"S&P max cluster size {max_size} > 50 "
            "(pre-fix baseline was 446 of 485 in one mega-cluster)"
        )

    def test_ari_recovers_sector_structure(self):
        from sklearn.metrics import adjusted_rand_score

        df = _load_artifact_clusters(SP500_RESULTS, SP500_UNIVERSE)
        ari = adjusted_rand_score(df["sector"], df["cluster_id"])
        assert ari >= 0.25, (
            f"S&P sector-recovery ARI {ari:.3f} below 0.25 floor "
            "(pre-fix baseline was ~0.007)"
        )


class TestClusteringConfigPlumbing:
    """The Phase 1.1 fix added ClusteringConfig. Confirm run_clustering honours it."""

    def test_config_defaults_to_ward_maxclust(self):
        from src.config import ClusteringConfig

        cfg = ClusteringConfig()
        assert cfg.linkage_method == "ward"
        assert cfg.n_clusters == 20
        assert cfg.criterion == "maxclust"

    def test_load_config_picks_up_clustering_block(self):
        from src.config import load_config

        cfg = load_config(PROJECT_ROOT / "config" / "settings.yaml")
        assert cfg.clustering.linkage_method == "ward"
        assert cfg.clustering.n_clusters == 20
