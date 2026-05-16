"""Head-to-head: RMT denoising method='constant' vs method='zero'.

Both produce a denoised correlation, a denoised MST, and a set of node metrics.
We compare them on:
    - eigenvalue preservation (signal modes should be identical)
    - max off-diagonal correlation (super-hub indicator)
    - MST top-3 hubs by betweenness
    - MST sector purity
    - Jaccard overlap with raw MST
    - Jaccard overlap with each other
    - Trace preservation
"""

from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rmt_denoising import denoise_correlation, marchenko_pastur_bounds
from src.clustering import build_mst, compute_mst_metrics
from src.analysis import compute_distance_matrix

RESULTS = ROOT / "data" / "results"
EXTRA = RESULTS / "extra"
EXTRA.mkdir(exist_ok=True, parents=True)


def edges_of(g: nx.Graph) -> set[tuple[str, str]]:
    return {tuple(sorted((str(u), str(v)))) for u, v in g.edges()}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def main():
    corr = pd.read_parquet(RESULTS / "pearson_corr.parquet")
    returns = pd.read_parquet(ROOT / "data" / "processed" / "log_returns.parquet")
    clusters = pd.read_csv(RESULTS / "cluster_assignments.csv")
    sector = dict(zip(clusters["ticker"].astype(str), clusters["sector"].astype(str)))
    T, N = len(returns), len(corr.columns)

    mp_lo, mp_hi = marchenko_pastur_bounds(T, N)
    print(f"T={T}  N={N}  MP bounds [{mp_lo:.4f}, {mp_hi:.4f}]")

    # Raw MST for comparison
    raw_dist = compute_distance_matrix(corr)
    raw_mst = build_mst(raw_dist)
    raw_metrics = compute_mst_metrics(raw_mst).set_index("ticker")
    raw_edges = edges_of(raw_mst)
    max_raw = float(np.abs(np.triu(corr.values, k=1)).max())
    raw_purity = sum(
        sector.get(u) == sector.get(v) for u, v in raw_edges
    ) / max(1, len(raw_edges))

    rows = []
    edge_sets = {"raw": raw_edges}
    for method in ("constant", "zero"):
        den_corr, spectrum = denoise_correlation(corr, T=T, method=method)
        den_dist = compute_distance_matrix(den_corr)
        den_mst = build_mst(den_dist)
        den_metrics = compute_mst_metrics(den_mst).set_index("ticker")
        e = edges_of(den_mst)
        edge_sets[method] = e

        # Per-method stats
        max_offdiag = float(np.abs(np.triu(den_corr.values, k=1)).max())
        trace = float(np.trace(den_corr.values))
        purity = sum(sector.get(u) == sector.get(v) for u, v in e) / max(1, len(e))

        top_hubs = (
            den_metrics["betweenness_centrality"]
            .sort_values(ascending=False)
            .head(3)
        )
        max_degree = int(den_metrics["degree"].max())
        max_degree_ticker = den_metrics["degree"].idxmax()
        max_betw = float(den_metrics["betweenness_centrality"].max())

        rows.append({
            "method": method,
            "n_signal_eig": int(spectrum["is_signal"].sum()),
            "max_off_diag_corr": max_offdiag,
            "trace": trace,
            "mst_max_degree": max_degree,
            "mst_max_degree_ticker": max_degree_ticker,
            "mst_max_betweenness": max_betw,
            "mst_top_hub_1": top_hubs.index[0] if len(top_hubs) > 0 else None,
            "mst_top_hub_2": top_hubs.index[1] if len(top_hubs) > 1 else None,
            "mst_top_hub_3": top_hubs.index[2] if len(top_hubs) > 2 else None,
            "sector_purity": purity,
            "jaccard_vs_raw": jaccard(e, raw_edges),
        })

    out = pd.DataFrame(rows)
    # Add raw row for reference
    raw_row = {
        "method": "raw",
        "n_signal_eig": None,
        "max_off_diag_corr": max_raw,
        "trace": float(np.trace(corr.values)),
        "mst_max_degree": int(raw_metrics["degree"].max()),
        "mst_max_degree_ticker": raw_metrics["degree"].idxmax(),
        "mst_max_betweenness": float(raw_metrics["betweenness_centrality"].max()),
        "mst_top_hub_1": raw_metrics["betweenness_centrality"].sort_values(ascending=False).index[0],
        "mst_top_hub_2": raw_metrics["betweenness_centrality"].sort_values(ascending=False).index[1],
        "mst_top_hub_3": raw_metrics["betweenness_centrality"].sort_values(ascending=False).index[2],
        "sector_purity": raw_purity,
        "jaccard_vs_raw": 1.0,
    }
    out = pd.concat([pd.DataFrame([raw_row]), out], ignore_index=True)
    print("\n=== RAW vs CONSTANT vs ZERO ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    j_cz = jaccard(edge_sets["constant"], edge_sets["zero"])
    print(f"\nJaccard(constant MST, zero MST) = {j_cz:.4f}")

    out_path = EXTRA / "rmt_method_comparison.csv"
    out.to_csv(out_path, index=False)
    with open(EXTRA / "rmt_method_comparison.json", "w") as f:
        json.dump({
            "table": out.to_dict(orient="records"),
            "jaccard_constant_vs_zero": j_cz,
            "n_tickers": N, "T_days": T,
            "mp_upper": mp_hi, "mp_lower": mp_lo,
        }, f, indent=2, default=str)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
