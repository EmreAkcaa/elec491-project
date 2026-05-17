"""Clustering and network: hierarchical clustering, dendrogram, MST."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
import networkx as nx

from src.config import PipelineConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)



def compute_linkage(
    dist: pd.DataFrame, method: str = "single"
) -> tuple[np.ndarray, list[str]]:
    """Compute hierarchical clustering linkage from a distance matrix.

    Parameters
    ----------
    dist : pd.DataFrame
        Square symmetric distance matrix (tickers as index/columns).
    method : str
        Linkage method: 'single', 'complete', 'average', 'ward'.
        Note: 'ward' requires Euclidean distances (which ours are).

    Returns
    -------
    Z : np.ndarray
        Scipy linkage matrix (n-1 x 4).
    labels : list[str]
        Ticker labels in the order they appear in the distance matrix.
    """
    labels = dist.columns.tolist()

    # Handle NaN: fill with max distance (2.0) for missing correlations
    dist_clean = dist.fillna(2.0).values
    # Ensure perfect symmetry
    dist_clean = (dist_clean + dist_clean.T) / 2
    np.fill_diagonal(dist_clean, 0.0)

    # Convert to condensed form for scipy
    condensed = squareform(dist_clean, checks=False)

    Z = linkage(condensed, method=method)
    logger.info(
        "Computed %s linkage: %d tickers, %d merges",
        method, len(labels), len(Z),
    )
    return Z, labels


def get_leaf_order(Z: np.ndarray, labels: list[str]) -> list[str]:
    """Get the optimal leaf ordering from the linkage for heatmap reordering."""
    order_idx = leaves_list(Z)
    return [labels[i] for i in order_idx]


def get_cluster_assignments(
    Z: np.ndarray,
    labels: list[str],
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Cut the dendrogram to assign clusters.

    Provide either n_clusters or distance_threshold, not both.
    If neither is given, defaults to distance_threshold=1.0 (mid-range).
    """
    if n_clusters is not None:
        cluster_ids = fcluster(Z, t=n_clusters, criterion="maxclust")
    elif distance_threshold is not None:
        cluster_ids = fcluster(Z, t=distance_threshold, criterion="distance")
    else:
        cluster_ids = fcluster(Z, t=1.0, criterion="distance")

    df = pd.DataFrame({"ticker": labels, "cluster_id": cluster_ids})
    n = df["cluster_id"].nunique()
    logger.info("Assigned %d clusters to %d tickers", n, len(labels))
    return df


def build_mst(dist: pd.DataFrame) -> nx.Graph:
    """Build a minimum spanning tree from the distance matrix.

    Uses NetworkX's built-in MST (Kruskal's algorithm).
    """
    labels = dist.columns.tolist()

    # Build complete weighted graph from distance matrix
    G = nx.Graph()
    G.add_nodes_from(labels)

    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            d = dist.iloc[i, j]
            if np.isfinite(d):
                G.add_edge(labels[i], labels[j], weight=d)

    mst = nx.minimum_spanning_tree(G, algorithm="kruskal")
    logger.info(
        "MST: %d nodes, %d edges, total weight=%.3f",
        mst.number_of_nodes(),
        mst.number_of_edges(),
        sum(d["weight"] for _, _, d in mst.edges(data=True)),
    )
    return mst


def compute_mst_metrics(mst: nx.Graph) -> pd.DataFrame:
    """Compute node-level metrics from the MST.

    Metrics: degree, betweenness centrality.
    """
    degree = dict(mst.degree())
    betweenness = nx.betweenness_centrality(mst)

    df = pd.DataFrame(
        {
            "ticker": list(degree.keys()),
            "degree": list(degree.values()),
            "betweenness_centrality": [betweenness[n] for n in degree],
        }
    )
    return df.sort_values("degree", ascending=False).reset_index(drop=True)


def mst_to_edge_df(mst: nx.Graph, corr: pd.DataFrame) -> pd.DataFrame:
    """Convert MST edges to a DataFrame with distance and correlation."""
    rows = []
    for u, v, data in mst.edges(data=True):
        rho = corr.loc[u, v] if u in corr.index and v in corr.columns else np.nan
        rows.append(
            {
                "source": u,
                "target": v,
                "distance": data["weight"],
                "correlation": rho,
            }
        )
    return pd.DataFrame(rows)


def run_clustering(config: PipelineConfig) -> None:
    """Full clustering and network pipeline step."""
    logger.info("=== Clustering & Network ===")
    config.data_results.mkdir(parents=True, exist_ok=True)

    # Load precomputed artifacts
    dist = pd.read_parquet(config.data_results / "distance_matrix.parquet")
    corr = pd.read_parquet(config.data_results / "pearson_corr.parquet")

    # Drop tickers with all-NaN distances (if any)
    valid_mask = dist.notna().any(axis=1) & dist.notna().any(axis=0)
    dist = dist.loc[valid_mask, valid_mask]
    corr = corr.loc[valid_mask, valid_mask]
    logger.info("Using %d tickers for clustering", len(dist))

    # --- Hierarchical clustering ---
    Z, labels = compute_linkage(dist, method="single")

    # Save linkage matrix
    np.save(config.data_results / "linkage_matrix.npy", Z)
    # Save labels order
    with open(config.data_results / "linkage_labels.json", "w") as f:
        json.dump(labels, f)
    logger.info("Saved linkage matrix and labels")

    # Leaf order for heatmap reordering
    leaf_order = get_leaf_order(Z, labels)
    with open(config.data_results / "dendrogram_order.json", "w") as f:
        json.dump(leaf_order, f)
    logger.info("Saved dendrogram leaf order")

    # Cluster assignments (default threshold)
    clusters = get_cluster_assignments(Z, labels, distance_threshold=1.0)
    # Add sector info from universe
    sector_map = dict(zip(config.universe["ticker"], config.universe["sector"]))
    clusters["sector"] = clusters["ticker"].map(sector_map)
    clusters.to_csv(config.data_results / "cluster_assignments.csv", index=False)
    logger.info(
        "Saved cluster assignments: %d clusters",
        clusters["cluster_id"].nunique(),
    )

    # --- Minimum spanning tree ---
    mst = build_mst(dist)

    # MST edges
    edges = mst_to_edge_df(mst, corr)
    edges.to_csv(config.data_results / "mst_edges.csv", index=False)
    logger.info("Saved MST edges")

    # MST node metrics
    metrics = compute_mst_metrics(mst)
    metrics["sector"] = metrics["ticker"].map(sector_map)
    metrics.to_csv(config.data_results / "mst_node_metrics.csv", index=False)
    logger.info("Saved MST node metrics")

    logger.info("Clustering & network complete")
