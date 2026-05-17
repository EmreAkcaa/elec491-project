"""sp500_vs_bist.py — cross-market comparison.

Loads pipeline artifacts from data/bist/ and data/sp500/ (run the BIST and S&P
pipelines first) and produces a single comparison table for the technical
report and final-presentation slide.

Run:
    # 1. Make sure both pipelines have run end-to-end:
    uv run python run_pipeline.py                                    # BIST
    uv run python run_pipeline.py --config config/settings_sp500.yaml # S&P (top-100)

    # 2. Compute comparison:
    uv run python scripts/sp500_vs_bist.py
    # -> writes data/comparison_bist_vs_sp500.csv and prints the table
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _has_universe(market_dir: str) -> bool:
    return (ROOT / "data" / market_dir / "results" / "pearson_corr.parquet").exists()


def _load(market_dir: str) -> dict:
    """Load the headline artifacts for one universe."""
    rdir = ROOT / "data" / market_dir / "results"
    pdir = ROOT / "data" / market_dir / "processed"
    out: dict = {"market_dir": market_dir}

    if (rdir / "pipeline_metadata.json").exists():
        with open(rdir / "pipeline_metadata.json") as f:
            meta = json.load(f)
        out["meta"] = meta

    if (rdir / "pearson_corr.parquet").exists():
        out["pearson"] = pd.read_parquet(rdir / "pearson_corr.parquet")

    if (rdir / "eigenvalue_spectrum.csv").exists():
        out["eigenvalues"] = pd.read_csv(rdir / "eigenvalue_spectrum.csv")

    if (rdir / "mst_node_metrics.csv").exists():
        out["mst_metrics"] = pd.read_csv(rdir / "mst_node_metrics.csv")

    if (rdir / "mst_edges.csv").exists():
        out["mst_edges"] = pd.read_csv(rdir / "mst_edges.csv")

    if (rdir / "cluster_assignments.csv").exists():
        out["clusters"] = pd.read_csv(rdir / "cluster_assignments.csv")

    if (rdir / "partial_corr_edges.csv").exists():
        out["partial_edges"] = pd.read_csv(rdir / "partial_corr_edges.csv")

    if (rdir / "te_network_edges.csv").exists():
        out["te_edges"] = pd.read_csv(rdir / "te_network_edges.csv")

    if (pdir / "log_returns.parquet").exists():
        out["log_returns"] = pd.read_parquet(pdir / "log_returns.parquet")

    return out


def _effective_dimensionality(eig_df: pd.DataFrame) -> dict:
    lam = eig_df["eigenvalue"].to_numpy()
    sum_l = float(lam.sum())
    sum_l2 = float((lam ** 2).sum())
    d_eff = sum_l ** 2 / sum_l2
    n_signal = int(eig_df["is_signal"].sum())
    signal_share = float(lam[eig_df["is_signal"].to_numpy(dtype=bool)].sum() / sum_l)
    top_share = float(lam.max() / sum_l)
    log_det = float(np.log(np.clip(lam, 1e-12, None)).sum())
    return {
        "N": int(len(lam)),
        "D_eff": d_eff,
        "n_signal_eigenvalues": n_signal,
        "signal_variance_share": signal_share,
        "top_eigenvalue_share": top_share,
        "delta_entropy_nats": -0.5 * log_det,
    }


def _correlation_summary(corr: pd.DataFrame) -> dict:
    arr = corr.to_numpy()
    iu = np.triu_indices_from(arr, k=1)
    vals = arr[iu]
    vals = vals[np.isfinite(vals)]
    return {
        "n_pairs": int(vals.size),
        "mean_corr": float(vals.mean()),
        "median_corr": float(np.median(vals)),
        "std_corr": float(vals.std()),
        "max_abs_corr": float(np.abs(vals).max()),
    }


def _gaussian_mi_summary(corr: pd.DataFrame) -> dict:
    arr = corr.to_numpy()
    iu = np.triu_indices_from(arr, k=1)
    rho = arr[iu]
    rho = rho[np.isfinite(rho)]
    mi = -0.5 * np.log(np.clip(1 - rho ** 2, 1e-12, None))
    return {
        "mean_gaussian_mi_nats": float(mi.mean()),
        "max_gaussian_mi_nats": float(mi.max()),
        "sum_gaussian_mi_nats": float(mi.sum()),
    }


def _top_mst_hubs(metrics: pd.DataFrame, k: int = 5) -> list[dict]:
    top = (
        metrics.sort_values("betweenness_centrality", ascending=False).head(k)
    )
    return [
        {
            "ticker": str(r["ticker"]),
            "betweenness": float(r["betweenness_centrality"]),
            "degree": int(r["degree"]),
            "sector": str(r.get("sector", "")),
        }
        for _, r in top.iterrows()
    ]


def _sector_purity(edges: pd.DataFrame, sector_map: dict) -> float:
    if edges.empty:
        return float("nan")
    matched = counted = 0
    for _, r in edges.iterrows():
        sa, sb = sector_map.get(str(r["source"])), sector_map.get(str(r["target"]))
        if sa and sb:
            counted += 1
            if sa == sb:
                matched += 1
    return matched / counted if counted else float("nan")


def _crisis_window(market_dir: str, event_date: str, window_days: int = 60) -> dict:
    """Mean pairwise correlation in ±window-day buckets around an event."""
    p = ROOT / "data" / market_dir / "results" / "rolling_market_stats_w60.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    ev = pd.Timestamp(event_date)
    bef = df.loc[ev - pd.Timedelta(days=window_days):ev]
    dur = df.loc[ev:ev + pd.Timedelta(days=window_days)]
    aft = df.loc[ev + pd.Timedelta(days=window_days):ev + pd.Timedelta(days=2 * window_days)]
    return {
        f"{event_date}_before": float(bef["avg_corr"].mean()) if not bef.empty else float("nan"),
        f"{event_date}_during": float(dur["avg_corr"].mean()) if not dur.empty else float("nan"),
        f"{event_date}_after":  float(aft["avg_corr"].mean()) if not aft.empty else float("nan"),
    }


def main():
    have_bist = _has_universe("bist")
    have_sp500 = _has_universe("sp500")

    if not have_bist or not have_sp500:
        print("Missing required artifacts:")
        print(f"  data/bist/results/  : {'OK' if have_bist else 'MISSING'}")
        print(f"  data/sp500/results/ : {'OK' if have_sp500 else 'MISSING'}")
        print()
        print("Run both pipelines first:")
        print("  uv run python run_pipeline.py                                    # BIST")
        print("  uv run python run_pipeline.py --config config/settings_sp500.yaml # S&P")
        return 1

    print("Loading both universes…")
    B = _load("bist")
    S = _load("sp500")

    rows = []
    for label, data in (("BIST", B), ("S&P-500", S)):
        row = {"universe": label}
        if "eigenvalues" in data:
            row.update(_effective_dimensionality(data["eigenvalues"]))
        if "pearson" in data:
            row.update(_correlation_summary(data["pearson"]))
            row.update(_gaussian_mi_summary(data["pearson"]))
        if "mst_metrics" in data and "mst_edges" in data:
            row["top_mst_hubs"] = "; ".join(
                f"{h['ticker']} ({h['betweenness']:.2f})"
                for h in _top_mst_hubs(data["mst_metrics"], k=3)
            )
            sector_map = (
                dict(zip(data["clusters"]["ticker"].astype(str),
                         data["clusters"]["sector"].astype(str)))
                if "clusters" in data else {}
            )
            row["mst_sector_purity"] = _sector_purity(data["mst_edges"], sector_map)
            if "partial_edges" in data:
                row["glasso_n_edges"] = int(len(data["partial_edges"]))
                row["glasso_sector_purity"] = _sector_purity(data["partial_edges"], sector_map)
            if "te_edges" in data:
                row["te_n_edges"] = int(len(data["te_edges"]))
                row["te_sector_purity"] = _sector_purity(data["te_edges"], sector_map)
        # Crisis windows on both universes (COVID + Russia-Ukraine are shared events;
        # earthquake is Türkiye-only so BIST gets it but S&P likely won't show much).
        for ev_date in ("2020-03-11", "2022-02-24", "2023-02-06"):
            row.update(_crisis_window(data["market_dir"], ev_date))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("universe").T
    print("\n=== Cross-Market Comparison: BIST vs S&P-500 ===")
    print(df.to_string())

    out_path = ROOT / "data" / "comparison_bist_vs_sp500.csv"
    df.to_csv(out_path)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
