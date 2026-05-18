"""Precomputed MST graph layouts for fast dashboard rendering.

PHASE Y (Y2) — see ``/Users/emre/.claude/plans/...spicy-finch.md``.

NetworkX's spring_layout / kamada_kawai_layout are O(N²) or O(N³) per
call. On S&P 485 nodes that's ~1-2 s per layout; multiplied by the
~12 MSTs the dashboard renders (main + denoised + 7 wavelets + TE +
cross-market ×2) = 12-24 s of layout compute per page load.

This stage runs the layouts ONCE at pipeline time and serialises the
node positions to JSON. The dashboard's renderers read the JSON and
skip the layout call entirely → 50 ms disk read instead of 1-2 s
NetworkX compute, for each of the ~12 MSTs.

Output layout::

    data/<universe>/results/layouts/
        main_mst.json
        denoised_mst.json
        wavelet_mst_scale1.json
        ...
        wavelet_mst_scale7.json
        te_network.json

JSON schema (per file)::

    {
        "positions": {"AKBNK": [0.12, -0.45], "GARAN": [0.34, 0.21], ...},
        "n_nodes": 73,
        "n_edges": 72,
        "algorithm": "kamada_kawai" | "spring",
        "seed": 42,
        "source_file": "mst_edges.csv"
    }

Gating: only the 4 finance universes (bist, bist_usd, bist_gold, sp500)
generate layouts. EEG is on live compute (small electrode counts make
it fast enough; deferred to a follow-up if EEG demo paths get heavy).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


# Only finance universes get precomputed layouts.
_LAYOUT_MARKETS: set[str] = {"bist", "bist_usd", "bist_gold", "sp500"}

# Match dashboard.py:_mst_layout heuristic so live + precomputed give
# identical positions (when both seed=42).
_SPRING_THRESHOLD = 200
_SPRING_ITERATIONS = 80
_SPRING_SEED = 42

# Set of MST sources to precompute, mapping `source_name` → CSV filename
# (relative to `data/<universe>/results/`). Files that don't exist for a
# given universe (e.g., wavelets if Wavelet stage didn't run) are skipped
# silently.
_SOURCES: tuple[tuple[str, str], ...] = (
    ("main_mst", "mst_edges.csv"),
    ("denoised_mst", "denoised_mst_edges.csv"),
    ("wavelet_mst_scale1", "wavelet_mst_edges_scale1.csv"),
    ("wavelet_mst_scale2", "wavelet_mst_edges_scale2.csv"),
    ("wavelet_mst_scale3", "wavelet_mst_edges_scale3.csv"),
    ("wavelet_mst_scale4", "wavelet_mst_edges_scale4.csv"),
    ("wavelet_mst_scale5", "wavelet_mst_edges_scale5.csv"),
    ("wavelet_mst_scale6", "wavelet_mst_edges_scale6.csv"),
    ("wavelet_mst_scale7", "wavelet_mst_edges_scale7.csv"),
    ("te_network", "te_network_edges.csv"),
)


def _compute_layout_for_edges(
    edges_df: pd.DataFrame,
    *,
    weight_col: str = "distance",
) -> tuple[dict[str, tuple[float, float]], str, int]:
    """Build a graph from edges and compute layout positions.

    Returns (positions_dict, algorithm_name, n_nodes). Falls back to
    "distance" → "weight" col detection so this works for both MST and
    TE network CSVs (which use different column names).
    """
    if not HAS_NETWORKX or edges_df.empty:
        return {}, "none", 0

    # Auto-detect weight column — MST CSVs use "distance"; TE CSVs use
    # "te_value" or "weight" or "value".
    candidates = [weight_col, "weight", "te_value", "value", "edge_weight"]
    actual_weight_col = next(
        (c for c in candidates if c in edges_df.columns), None,
    )

    G = nx.Graph()
    for _, r in edges_df.iterrows():
        src = str(r["source"])
        tgt = str(r["target"])
        w = float(r[actual_weight_col]) if actual_weight_col else 1.0
        G.add_edge(src, tgt, weight=w)

    if G.number_of_nodes() == 0:
        return {}, "none", 0

    # Match the live-compute heuristic in app/dashboard.py:_mst_layout so
    # precomputed and live paths give identical positions.
    if G.number_of_nodes() > _SPRING_THRESHOLD:
        pos_raw = nx.spring_layout(
            G, weight="weight", iterations=_SPRING_ITERATIONS, seed=_SPRING_SEED,
        )
        algo = "spring"
    else:
        pos_raw = nx.kamada_kawai_layout(G, weight="weight")
        algo = "kamada_kawai"

    # Coerce numpy arrays/types → plain Python floats for clean JSON.
    pos = {str(node): [float(x), float(y)] for node, (x, y) in pos_raw.items()}
    return pos, algo, G.number_of_nodes()


def _process_one_source(
    *,
    universe_results: Path,
    source_name: str,
    source_filename: str,
    out_dir: Path,
) -> Optional[dict]:
    """Compute + write one layout file. Returns a small summary dict or
    None if the source CSV is missing (universe doesn't have this artifact)."""
    src_path = universe_results / source_filename
    if not src_path.exists():
        return None

    try:
        edges_df = pd.read_csv(src_path)
    except Exception as exc:  # noqa: BLE001 — log + skip on per-file failure
        logger.warning("Could not read %s: %s", src_path, exc)
        return None

    if edges_df.empty or "source" not in edges_df.columns or "target" not in edges_df.columns:
        logger.warning(
            "%s missing required columns (source, target); skipping.", src_path,
        )
        return None

    t0 = time.perf_counter()
    positions, algorithm, n_nodes = _compute_layout_for_edges(edges_df)
    if not positions:
        return None

    payload = {
        "positions": positions,
        "n_nodes": n_nodes,
        "n_edges": len(edges_df),
        "algorithm": algorithm,
        "seed": _SPRING_SEED,
        "source_file": source_filename,
    }
    out_path = out_dir / f"{source_name}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f)
    elapsed = time.perf_counter() - t0
    return {
        "source_name": source_name,
        "n_nodes": n_nodes,
        "algorithm": algorithm,
        "elapsed_s": elapsed,
        "out_path": out_path,
    }


def run_mst_layouts(config: PipelineConfig) -> None:
    """Pipeline stage: precompute graph layouts for the dashboard's MSTs.

    Gate: only runs for `config.market.market_id in _LAYOUT_MARKETS`.
    """
    market_id = config.market.market_id.lower()
    if market_id not in _LAYOUT_MARKETS:
        logger.info(
            "MST layouts SKIPPED for market_id=%r (precompute set: %s)",
            market_id, sorted(_LAYOUT_MARKETS),
        )
        return

    if not HAS_NETWORKX:
        logger.warning(
            "NetworkX not available — MST layout precompute skipped.",
        )
        return

    logger.info("=== MST Layouts — %s ===", market_id)

    out_dir = config.data_results / "layouts"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[dict] = []
    skipped: list[str] = []
    t0 = time.perf_counter()

    for source_name, source_filename in _SOURCES:
        summary = _process_one_source(
            universe_results=config.data_results,
            source_name=source_name,
            source_filename=source_filename,
            out_dir=out_dir,
        )
        if summary is None:
            skipped.append(source_name)
            continue
        written.append(summary)
        logger.info(
            "  %s: %d nodes via %s in %.1f ms → %s",
            source_name, summary["n_nodes"], summary["algorithm"],
            summary["elapsed_s"] * 1000, summary["out_path"].name,
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "MST layouts complete for %s: %d written, %d skipped, %.1fs total.",
        market_id, len(written), len(skipped), elapsed,
    )
    if skipped:
        logger.info(
            "  skipped (no source CSV present): %s", ", ".join(skipped),
        )
