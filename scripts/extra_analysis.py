"""extra_analysis.py — derived metrics for the final report.

Consumes existing pipeline artifacts under data/results/ and data/processed/.
Produces a single JSON of scalar metrics plus a few derived parquet/csv files.

Run:
    uv run python scripts/extra_analysis.py

Outputs:
    data/results/extra/it_summary.json
    data/results/extra/mutual_information_matrix.parquet
    data/results/extra/mi_pearson_comparison.csv
    data/results/extra/wavelet_entropy.csv
    data/results/extra/crisis_window_stats.csv
    data/results/extra/methods_comparison.csv

All computations are deterministic given the existing artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Post Phase D: pipeline writes per-market under data/<market>/...
# Default to BIST; override with EXTRA_UNIVERSE env var (e.g. "sp500" later).
import os as _os
_UNIV = _os.environ.get("EXTRA_UNIVERSE", "bist")
DATA_RESULTS = ROOT / "data" / _UNIV / "results"
DATA_PROCESSED = ROOT / "data" / _UNIV / "processed"
EXTRA_DIR = DATA_RESULTS / "extra"
EXTRA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_returns() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / "log_returns.parquet")


def load_pearson() -> pd.DataFrame:
    return pd.read_parquet(DATA_RESULTS / "pearson_corr.parquet")


def load_eigenvalues() -> pd.DataFrame:
    return pd.read_csv(DATA_RESULTS / "eigenvalue_spectrum.csv")


def load_universe() -> pd.DataFrame:
    return pd.read_csv(ROOT / "config" / "universes" / "bist100.csv")


def load_clusters() -> pd.DataFrame:
    return pd.read_csv(DATA_RESULTS / "cluster_assignments.csv")


def load_mst_edges(path: Path) -> set[tuple[str, str]]:
    df = pd.read_csv(path)
    # store edges as sorted tuples so undirected comparison works
    return {tuple(sorted((str(s), str(t)))) for s, t in zip(df["source"], df["target"])}


def load_rolling_market(window: int) -> pd.DataFrame:
    df = pd.read_parquet(DATA_RESULTS / f"rolling_market_stats_w{window}.parquet")
    df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# 1. RMT-derived information-theoretic scalars
# ---------------------------------------------------------------------------


def rmt_scalars(eig_df: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Effective informational dimensionality + Gaussian differential-entropy reduction.

    Effective dimensionality (participation ratio):
        D_eff = (sum lambda_i)^2 / sum (lambda_i^2)

    Differential entropy under N(0, Σ) with unit-variance marginals:
        H = (N/2) log(2 pi e) + (1/2) log det Σ
        log det Σ = sum log lambda_i
        ΔH (uncorrelated vs correlated) = -(1/2) sum log lambda_i  [in nats]

    Returns scalars in nats and bits.
    """
    lam = eig_df["eigenvalue"].to_numpy()
    N = len(lam)
    T = len(returns)
    sum_l = lam.sum()
    sum_l2 = (lam ** 2).sum()
    d_eff = float(sum_l ** 2 / sum_l2)
    # Clamp eigenvalues to a tiny positive floor to avoid log(0).
    safe = np.clip(lam, 1e-12, None)
    log_det = float(np.log(safe).sum())
    dH_nats = -0.5 * log_det
    dH_bits = dH_nats / math.log(2)

    # Per-mode information share (signal eigenvalues only).
    signal_mask = eig_df["is_signal"].to_numpy(dtype=bool)
    signal_eig = lam[signal_mask]
    signal_share = float(signal_eig.sum() / N)

    return {
        "N_tickers": int(N),
        "T_days": int(T),
        "trace_sigma": float(sum_l),
        "frobenius_sq_sigma": float(sum_l2),
        "effective_dimensionality_D_eff": d_eff,
        "n_signal_eigenvalues": int(signal_mask.sum()),
        "signal_variance_share": signal_share,
        "top_eigenvalue": float(lam.max()),
        "top_eigenvalue_share": float(lam.max() / N),
        "log_det_sigma": log_det,
        "delta_entropy_nats": dH_nats,
        "delta_entropy_bits": dH_bits,
    }


# ---------------------------------------------------------------------------
# 2. Mutual information via 3-bin equal-frequency binning (matches TE)
# ---------------------------------------------------------------------------


def equal_freq_bin(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-frequency binning. Returns int labels in [0, n_bins)."""
    finite = np.isfinite(x)
    if finite.sum() < n_bins:
        return np.full(len(x), -1, dtype=np.int64)
    # Use quantile edges, drop ties carefully.
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(x[finite], qs)
    edges = np.unique(edges)  # collapse tied quantile edges
    labels = np.digitize(x, edges)
    labels[~finite] = -1
    return labels.astype(np.int64)


def entropy_plugin(counts: np.ndarray) -> float:
    """Shannon entropy in nats from non-negative count array."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def mutual_information_matrix(returns: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    """Empirical MI matrix using same binning convention as transfer_entropy.py."""
    cols = list(returns.columns)
    N = len(cols)
    discretized: dict[str, np.ndarray] = {}
    marginal_entropy: dict[str, float] = {}
    for c in cols:
        labels = equal_freq_bin(returns[c].to_numpy(), n_bins)
        discretized[c] = labels
        valid = labels[labels >= 0]
        if valid.size == 0:
            marginal_entropy[c] = 0.0
            continue
        counts = np.bincount(valid, minlength=n_bins)
        marginal_entropy[c] = entropy_plugin(counts)

    mi = np.zeros((N, N), dtype=np.float64)
    for i, ci in enumerate(cols):
        mi[i, i] = marginal_entropy[ci]
        xi = discretized[ci]
        for j in range(i + 1, N):
            cj = cols[j]
            xj = discretized[cj]
            mask = (xi >= 0) & (xj >= 0)
            if mask.sum() < 30:
                value = 0.0
            else:
                pair_codes = xi[mask] * n_bins + xj[mask]
                counts = np.bincount(pair_codes, minlength=n_bins * n_bins)
                H_joint = entropy_plugin(counts)
                value = marginal_entropy[ci] + marginal_entropy[cj] - H_joint
                value = max(value, 0.0)
            mi[i, j] = value
            mi[j, i] = value
    return pd.DataFrame(mi, index=cols, columns=cols)


def compare_mi_vs_pearson(mi: pd.DataFrame, pearson: pd.DataFrame) -> pd.DataFrame:
    """Pair-level table: empirical MI vs Gaussian MI lower bound."""
    cols = mi.columns
    rows = []
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            rho = float(pearson.iloc[i, j])
            if not np.isfinite(rho):
                continue
            mi_gauss = -0.5 * math.log(max(1.0 - rho * rho, 1e-12))
            mi_emp = float(mi.iloc[i, j])
            rows.append(
                {
                    "ticker_a": a,
                    "ticker_b": b,
                    "pearson_rho": rho,
                    "mi_empirical_nats": mi_emp,
                    "mi_gaussian_nats": mi_gauss,
                    "ratio_emp_over_gauss": mi_emp / mi_gauss if mi_gauss > 1e-9 else float("nan"),
                    "excess_mi_nats": mi_emp - mi_gauss,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Wavelet entropy per ticker
# ---------------------------------------------------------------------------


def per_ticker_wavelet_variance() -> pd.DataFrame:
    """Approximate per-ticker variance contribution at each scale.

    The wavelet pipeline persists *correlation* matrices per scale, not the
    detail series themselves. We approximate per-ticker scale variance by the
    diagonal of (Cov_scale = Corr_scale * sqrt(var_scale_i * var_scale_j)),
    but since we only have correlations (diagonal = 1 by construction), we use
    the *average off-diagonal correlation magnitude* per ticker as a proxy
    weight for the scale's contribution. Less ideal than direct variance but
    is the cleanest derivation from the persisted artifacts. Documented as a
    proxy in the report.
    """
    scales = sorted(int(p.stem.replace("wavelet_corr_scale", ""))
                    for p in DATA_RESULTS.glob("wavelet_corr_scale*.parquet"))
    if not scales:
        return pd.DataFrame()
    weight_per_scale: dict[int, pd.Series] = {}
    for s in scales:
        corr = pd.read_parquet(DATA_RESULTS / f"wavelet_corr_scale{s}.parquet")
        np.fill_diagonal(corr.values, np.nan)
        weight_per_scale[s] = corr.abs().mean(axis=1)
    df = pd.DataFrame(weight_per_scale)
    df = df.div(df.sum(axis=1), axis=0)  # normalize across scales per ticker
    # Per-ticker wavelet entropy (Rosso et al. 2001 form, in nats)
    eps = 1e-12
    H = -(df * np.log(df.clip(lower=eps))).sum(axis=1)
    out = df.copy()
    out.columns = [f"share_scale{s}" for s in scales]
    out["wavelet_entropy_nats"] = H
    out["wavelet_entropy_normalized"] = H / math.log(len(scales))
    out.index.name = "ticker"
    return out


# ---------------------------------------------------------------------------
# 4. Crisis window stats (±60 trading days around named events)
# ---------------------------------------------------------------------------


CRISIS_EVENTS = [
    {"name": "COVID-19 WHO pandemic declaration", "date": "2020-03-11"},
    {"name": "Russia–Ukraine invasion",            "date": "2022-02-24"},
    {"name": "Turkey earthquakes",                  "date": "2023-02-06"},
]


def crisis_window_stats(window_rolling: int = 60, window_days: int = 60) -> pd.DataFrame:
    """For each event, compute mean pairwise corr in ±window-day buckets.

    Uses the w=60 rolling-correlation series (data starts 2020-03-26).
    'before' window for the COVID event is therefore truncated and reported as such.
    """
    rolling = load_rolling_market(window_rolling)
    rows = []
    for ev in CRISIS_EVENTS:
        ev_date = pd.Timestamp(ev["date"])
        bef = rolling.loc[ev_date - pd.Timedelta(days=window_days):ev_date]
        dur = rolling.loc[ev_date:ev_date + pd.Timedelta(days=window_days)]
        aft = rolling.loc[
            ev_date + pd.Timedelta(days=window_days):ev_date + pd.Timedelta(days=2 * window_days)
        ]
        for label, segment in (("before", bef), ("during", dur), ("after", aft)):
            if segment.empty or "avg_corr" not in segment.columns:
                avg = float("nan")
                med = float("nan")
            else:
                avg = float(segment["avg_corr"].mean())
                med = float(segment["median_corr"].mean())
            rows.append(
                {
                    "event": ev["name"],
                    "event_date": ev["date"],
                    "phase": label,
                    "n_obs": int(len(segment)),
                    "avg_corr": avg,
                    "median_corr": med,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Methods-comparison table (cross-method MST overlap + sector purity)
# ---------------------------------------------------------------------------


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def mst_method_edges() -> dict[str, set]:
    out: dict[str, set] = {}
    out["raw"] = load_mst_edges(DATA_RESULTS / "mst_edges.csv")
    out["denoised"] = load_mst_edges(DATA_RESULTS / "denoised_mst_edges.csv")
    for s in range(1, 8):
        p = DATA_RESULTS / f"wavelet_mst_edges_scale{s}.csv"
        if p.exists():
            out[f"wavelet_s{s}"] = load_mst_edges(p)
    # Glasso/TE produce arbitrary-size edge lists, not strict MSTs;
    # we use them as "method-level edge sets" for comparative purity.
    return out


def method_edge_set_from_csv(path: Path, src: str = "source", tgt: str = "target") -> set:
    df = pd.read_csv(path)
    return {tuple(sorted((str(a), str(b)))) for a, b in zip(df[src], df[tgt])}


def sector_purity(edges: set, sector_map: dict[str, str]) -> float:
    if not edges:
        return float("nan")
    matched = 0
    counted = 0
    for a, b in edges:
        sa, sb = sector_map.get(a), sector_map.get(b)
        if sa and sb:
            counted += 1
            if sa == sb:
                matched += 1
    return matched / counted if counted else float("nan")


def gini(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    if x.sum() <= 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


def methods_comparison() -> pd.DataFrame:
    cluster_df = load_clusters()
    sector_map = dict(zip(cluster_df["ticker"].astype(str), cluster_df["sector"].astype(str)))
    method_edges = mst_method_edges()
    # Add Glasso and TE edge lists as additional method "networks"
    method_edges["glasso"] = method_edge_set_from_csv(
        DATA_RESULTS / "partial_corr_edges.csv"
    )
    method_edges["transfer_entropy"] = method_edge_set_from_csv(
        DATA_RESULTS / "te_network_edges.csv"
    )

    raw = method_edges["raw"]
    rows = []
    # Edge weight load: where possible, attach mean edge weight (corr / partial / te)
    for name, edges in method_edges.items():
        rows.append(
            {
                "method": name,
                "n_edges": len(edges),
                "sector_purity": sector_purity(edges, sector_map),
                "jaccard_vs_raw_mst": jaccard(edges, raw),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Gaussian-MI summary from existing Pearson matrix (no new code)
# ---------------------------------------------------------------------------


def gaussian_mi_summary(pearson: pd.DataFrame) -> dict:
    cols = list(pearson.columns)
    N = len(cols)
    vals = []
    pair_ab = []
    for i in range(N):
        for j in range(i + 1, N):
            rho = float(pearson.iloc[i, j])
            if not np.isfinite(rho):
                continue
            mi_g = -0.5 * math.log(max(1.0 - rho * rho, 1e-12))
            vals.append(mi_g)
            pair_ab.append((cols[i], cols[j], rho, mi_g))
    arr = np.asarray(vals)
    top5 = sorted(pair_ab, key=lambda r: -r[3])[:5]
    return {
        "mean_gaussian_mi_nats": float(arr.mean()),
        "median_gaussian_mi_nats": float(np.median(arr)),
        "max_gaussian_mi_nats": float(arr.max()),
        "min_gaussian_mi_nats": float(arr.min()),
        "sum_gaussian_mi_nats": float(arr.sum()),
        "n_pairs": int(arr.size),
        "top_5_pairs_gaussian_mi": [
            {"a": a, "b": b, "rho": r, "mi_nats": m} for a, b, r, m in top5
        ],
    }


def empirical_mi_summary(mi: pd.DataFrame) -> dict:
    cols = list(mi.columns)
    vals = []
    pair_ab = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = float(mi.iloc[i, j])
            vals.append(v)
            pair_ab.append((cols[i], cols[j], v))
    arr = np.asarray(vals)
    top5 = sorted(pair_ab, key=lambda r: -r[2])[:5]
    return {
        "mean_empirical_mi_nats": float(arr.mean()),
        "median_empirical_mi_nats": float(np.median(arr)),
        "max_empirical_mi_nats": float(arr.max()),
        "min_empirical_mi_nats": float(arr.min()),
        "sum_empirical_mi_nats": float(arr.sum()),
        "n_pairs": int(arr.size),
        "top_5_pairs_empirical_mi": [
            {"a": a, "b": b, "mi_nats": m} for a, b, m in top5
        ],
    }


def excess_pairs(mi_vs_pearson: pd.DataFrame, ratio_threshold: float = 1.5) -> pd.DataFrame:
    sub = mi_vs_pearson.copy()
    sub = sub[sub["mi_gaussian_nats"] > 0.02]  # ignore numerically tiny denominators
    sub = sub[sub["ratio_emp_over_gauss"] > ratio_threshold]
    return sub.sort_values("excess_mi_nats", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading artifacts…")
    returns = load_returns()
    pearson = load_pearson()
    eig_df = load_eigenvalues()
    print(f"  returns: {returns.shape}  pearson: {pearson.shape}  eigenvalues: {len(eig_df)}")

    print("\n[1] RMT-derived informational scalars")
    rmt = rmt_scalars(eig_df, returns)
    for k, v in rmt.items():
        if isinstance(v, float):
            print(f"   {k:40s} = {v:.4f}")
        else:
            print(f"   {k:40s} = {v}")

    print("\n[2] Mutual information matrix (3-bin equal-frequency)…")
    mi = mutual_information_matrix(returns, n_bins=3)
    mi.to_parquet(EXTRA_DIR / "mutual_information_matrix.parquet")
    print(f"   saved {EXTRA_DIR / 'mutual_information_matrix.parquet'}")

    print("\n[3] Empirical MI vs Gaussian MI comparison…")
    mi_vs = compare_mi_vs_pearson(mi, pearson)
    mi_vs.to_csv(EXTRA_DIR / "mi_pearson_comparison.csv", index=False)
    print(f"   saved {EXTRA_DIR / 'mi_pearson_comparison.csv'} ({len(mi_vs)} pairs)")

    gauss_sum = gaussian_mi_summary(pearson)
    emp_sum = empirical_mi_summary(mi)
    excess = excess_pairs(mi_vs, ratio_threshold=1.5)
    print(f"   pairs with empirical MI > 1.5 × Gaussian MI: {len(excess)}")
    if len(excess) > 0:
        print("   top 5 nonlinear-excess pairs:")
        for _, r in excess.head(5).iterrows():
            print(
                f"     {r.ticker_a:6s} {r.ticker_b:6s}  ρ={r.pearson_rho:+.3f}  "
                f"MI_emp={r.mi_empirical_nats:.4f}  MI_gauss={r.mi_gaussian_nats:.4f}  "
                f"ratio={r.ratio_emp_over_gauss:.2f}"
            )

    print("\n[4] Wavelet entropy per ticker…")
    we = per_ticker_wavelet_variance()
    if we.empty:
        print("   no wavelet artifacts found; skipping")
    else:
        we.to_csv(EXTRA_DIR / "wavelet_entropy.csv")
        norm = we["wavelet_entropy_normalized"]
        print(
            f"   per-ticker H_w (normalized to [0,1] by log(n_scales)): "
            f"min={norm.min():.3f}  mean={norm.mean():.3f}  max={norm.max():.3f}"
        )
        print(f"   top-5 most spectrally diverse tickers: "
              f"{list(norm.sort_values(ascending=False).head().index)}")
        print(f"   top-5 most spectrally concentrated tickers: "
              f"{list(norm.sort_values().head().index)}")

    print("\n[5] Crisis-window correlation stats…")
    cw = crisis_window_stats(window_rolling=60, window_days=60)
    cw.to_csv(EXTRA_DIR / "crisis_window_stats.csv", index=False)
    for _, r in cw.iterrows():
        print(
            f"   {r['event_date']}  {r['phase']:6s}  n={r['n_obs']:3d}  "
            f"avg_corr={r['avg_corr']:.4f}"
        )

    print("\n[6] Methods comparison (sector purity + Jaccard vs raw MST)…")
    mc = methods_comparison()
    mc.to_csv(EXTRA_DIR / "methods_comparison.csv", index=False)
    print(mc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n[7] Saving consolidated IT summary…")
    summary = {
        "rmt_information_geometry": rmt,
        "gaussian_mi_summary": gauss_sum,
        "empirical_mi_summary": emp_sum,
        "nonlinear_coupling": {
            "ratio_threshold": 1.5,
            "n_pairs_above_threshold": int(len(excess)),
            "top_pairs": excess.head(10).to_dict(orient="records"),
        },
    }
    with open(EXTRA_DIR / "it_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"   saved {EXTRA_DIR / 'it_summary.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
