"""StoNeCoAl — multi-universe correlation-network dashboard (BIST 100, S&P 500, EEG).

PHASE 2 entry point. Replaces the single-script `dashboard.py` with a
multi-page architecture using Streamlit's native `st.navigation`.

Layout:
  app/main.py            ← THIS FILE. Shared infrastructure + page router.
  app/views/01_*.py      ← thin per-page wrappers that delegate to existing render() functions
  app/views/02_*.py      ...
  app/views/05_*.py      ...
  app/cross_market.py    ← per-page modules (unchanged)
  app/market_overview.py ← extracted from dashboard.py in Stage 1 of this PR
  app/time_machine.py
  app/pair_analysis.py
  app/methods_lab.py
  app/eee_analysis.py

What lives here:
  1. EEG bulk-data materialisation (HF Spaces lazy-download).
  2. Universe registry init (`available_universes`, boot defaults).
  3. `st.set_page_config` + `inject_custom_css` (Streamlit's "only one set_page_config
     per app" rule means this MUST live in main.py — pages MUST NOT call it).
  4. Phase Y Y3 two-tier pre-warm hook.
  5. Sidebar render (dataset selector + BIST numéraire sub-switcher).
  6. Capability-gated page list build + page-disappearance pending stash
     (the Phase S #1 round-trip semantic carried forward to multi-page).
  7. `st.navigation(pages, position="sidebar").run()` — Streamlit's native
     nav widget. Page list appears in the sidebar below the dataset controls.

Native sidebar navigation chosen over a custom top-nav because:
  - Streamlit 1.41.1 only supports `position={"sidebar","hidden"}`; `"top"` arrived
    in 1.46+. A custom top-nav would require `st.switch_page()` calls which still
    cost a full Python rerun per click — defeating the perf benefit.
  - Native sidebar nav handles URL routing, browser back/forward, and deep links
    for free (e.g., `?page=time_machine`).
  - User explicit choice 2026-05-19: stable + predictable + idiomatic Streamlit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

_APP_DIR      = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ──────────────────────────────────────────────────────────────────────────────
# EEG bulk-data materialisation
# ──────────────────────────────────────────────────────────────────────────────
# HF Spaces caps per-repo storage at 1 GB. Our 2 EEG processed parquets are
# 308 MB each — too big to ship in the Space repo. The canonical HF workaround
# is to put bulk data in a companion Dataset repo (50 GB per file) and have
# the Space download it on first launch.
#
# Local dev: parquets already on disk → no-op early-return.
# HF Spaces:  files absent → snapshot_download from EEG_DATASET_REPO once;
#             cached under ~/.cache/huggingface on subsequent reruns.
# Fallback:   if the download fails, EEG silently drops from the sidebar
#             selector (available_universes() detects the absence and filters).
def _materialise_eeg_data_if_needed() -> None:
    if os.environ.get("STONECOAL_SKIP_EEG_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return

    eeg_dir = _PROJECT_ROOT / "data" / "eeg_motor_left_right" / "processed"
    sentinel = eeg_dir / "log_returns.parquet"
    if sentinel.exists() and sentinel.stat().st_size > 1_000_000:
        return  # local dev, or already-cached HF Spaces rebuild

    repo_id = os.environ.get("EEG_DATASET_REPO", "FlyingSubmarine33/stonecoal-eeg")
    print(f"[EEG] Bulk parquets not on disk; fetching from dataset repo {repo_id} …")
    try:
        from huggingface_hub import snapshot_download
        eeg_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(eeg_dir),
            allow_patterns=["*.parquet", "*.csv"],
        )
        print(f"[EEG] Materialised bulk data from {repo_id} into {eeg_dir}")
    except Exception as exc:  # noqa: BLE001 — best-effort; failure is non-fatal
        print(f"[EEG] Could not fetch from {repo_id}: {exc}")
        print(f"[EEG] Dashboard will run with BIST + S&P only. To enable EEG, "
              f"upload the parquets with: uv run python scripts/upload_eeg_to_hf_dataset.py")


_materialise_eeg_data_if_needed()


from utils import (  # noqa: E402
    current_universe, inject_custom_css,
)
from universe_registry import available_universes, get_universe  # noqa: E402


def _cap(u, attr, default):
    """Defensive capability lookup. Falls back to `default` rather than
    crashing on AttributeError (Streamlit Cloud sometimes ships a stale
    Universe class without a newer Phase I field)."""
    return getattr(u, attr, default)


# ══════════════════════════════════════════════════════════════════════════════
# Page config & global styling
# ══════════════════════════════════════════════════════════════════════════════

# Universe initialisation MUST run before st.set_page_config so the browser
# tab title reflects the active universe on first paint.
_AVAIL_UNIVERSES = available_universes()
_AVAIL_KEYS      = [u.key for u in _AVAIL_UNIVERSES] or ["bist"]
_LEGACY_ENV      = os.environ.get("DASHBOARD_UNIVERSE", "bist")

# Map a legacy single-key env var to the (dataset, bist_basis) pair.
_LEGACY_TO_PAIR = {
    "bist":      ("bist", "try"),
    "bist_usd":  ("bist", "usd"),
    "bist_gold": ("bist", "gold"),
}
if _LEGACY_ENV in _LEGACY_TO_PAIR:
    _BOOT_DATASET, _BOOT_BASIS = _LEGACY_TO_PAIR[_LEGACY_ENV]
elif _LEGACY_ENV in _AVAIL_KEYS:
    _BOOT_DATASET, _BOOT_BASIS = _LEGACY_ENV, "try"
else:
    _BOOT_DATASET = _AVAIL_KEYS[0] if not _AVAIL_KEYS[0].startswith("bist") else "bist"
    _BOOT_BASIS = "try"

# Defensive session_state init.
try:
    if "dataset" not in st.session_state:
        st.session_state["dataset"] = _BOOT_DATASET
    if "bist_basis" not in st.session_state:
        st.session_state["bist_basis"] = _BOOT_BASIS
    _disk_datasets = set()
    for _k in _AVAIL_KEYS:
        _disk_datasets.add("bist" if _k.startswith("bist") else _k)
    if st.session_state["dataset"] not in _disk_datasets:
        st.session_state["dataset"] = _BOOT_DATASET
    _active_universe_key = current_universe()
    if _active_universe_key not in _AVAIL_KEYS:
        _active_universe_key = _AVAIL_KEYS[0]
except Exception:  # noqa: BLE001 — SessionInfo not yet initialised
    _active_universe_key = (
        _LEGACY_ENV if _LEGACY_ENV in _AVAIL_KEYS else _AVAIL_KEYS[0]
    )

_active_universe = get_universe(_active_universe_key)

st.set_page_config(
    page_title=f"StoNeCoAl — {_cap(_active_universe, 'short_label', 'BIST 100')}",
    page_icon="<svg xmlns='http://www.w3.org/2000/svg'/>",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE Y (Y3) — Two-tier background pre-warm
# ══════════════════════════════════════════════════════════════════════════════
# Tier 1: log_returns + metadata for ALL universes (cheap, ~50 KB each).
# Tier 2: heavy artifacts (batch_corr, MSTs, clusters, eigenvalues, summary)
#         for the ACTIVE universe only — avoids 5× memory on cold container.
# Guarded by a session_state flag so we only fire ONCE per session.
if "_prewarm_dispatched" not in st.session_state:
    st.session_state["_prewarm_dispatched"] = True
    try:
        import concurrent.futures as _cf
        from utils import (  # noqa: E402
            _load_log_returns, _load_metadata,
            _load_batch_corr, _load_mst_edges, _load_mst_metrics,
            _load_cluster_assignments, _load_dendrogram_order,
            _load_eigenvalue_spectrum, _load_summary_stats,
        )
        _prewarm_keys = [u.key for u in _AVAIL_UNIVERSES]
        _prewarm_executor = _cf.ThreadPoolExecutor(
            max_workers=min(8, len(_prewarm_keys) * 2 + 6),
        )
        for _k in _prewarm_keys:
            _prewarm_executor.submit(_load_log_returns, _k)
            _prewarm_executor.submit(_load_metadata, _k)
        try:
            _active_key = current_universe()
            if _active_key in _prewarm_keys:
                for _deep_loader in (
                    _load_batch_corr, _load_mst_edges, _load_mst_metrics,
                    _load_cluster_assignments, _load_dendrogram_order,
                    _load_eigenvalue_spectrum, _load_summary_stats,
                ):
                    _prewarm_executor.submit(_deep_loader, _active_key)
        except Exception:  # noqa: BLE001 — deep warm is best-effort
            pass
    except Exception as _prewarm_exc:  # noqa: BLE001 — never crash boot
        print(f"[prewarm] background warm-up failed: {_prewarm_exc!r}")


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar: 3-dataset selector + BIST numéraire sub-switcher
# ══════════════════════════════════════════════════════════════════════════════
# Sidebar runs on every page navigation (main.py re-executes per Streamlit's
# multi-page model). The sidebar widgets persist their values in session_state
# (`dataset`, `bist_basis`) which `current_universe()` resolves to the active
# universe key — keys that all `@st.cache_data` loaders use for cache identity.

_DATASET_LABELS = {
    "bist":                "BIST 100 — Türkiye",
    "sp500":               "S&P 500 — United States",
    "eeg_motor_left_right": "EEG Motor Imagery — PhysioNet",
}
_BASIS_LABELS = {"try": "TRY", "usd": "USD", "gold": "Gold"}

# Derive which TOP-LEVEL datasets are present on disk.
_dataset_options: list[str] = []
if any(k.startswith("bist") for k in _AVAIL_KEYS):
    _dataset_options.append("bist")
if "sp500" in _AVAIL_KEYS:
    _dataset_options.append("sp500")
if "eeg_motor_left_right" in _AVAIL_KEYS:
    _dataset_options.append("eeg_motor_left_right")
if not _dataset_options:
    _dataset_options = ["bist"]

with st.sidebar:
    # StoNeCoAl branding strip — replaces the top-nav header from the
    # single-script dashboard. Sits above the dataset selector so it's
    # always visible regardless of which page is active.
    st.markdown(
        f"<div style='display:flex; flex-direction:column; gap:2px; "
        f"padding:0; margin-bottom:0.5rem;'>"
        f"<span style='font-size:1.15rem; font-weight:800; letter-spacing:-0.02em; "
        f"color:#2B2D42;'>StoNeCoAl</span>"
        f"<span style='font-size:0.68rem; color:#8D99AE; letter-spacing:0.06em;'>"
        f"{_cap(_active_universe, 'short_label', 'BIST 100').upper()} NETWORK ANALYSIS"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    if len(_dataset_options) > 1:
        st.markdown("**Dataset**")
        st.selectbox(
            "Dataset",
            _dataset_options,
            format_func=lambda k: _DATASET_LABELS.get(k, k),
            key="dataset",
            label_visibility="collapsed",
        )
    elif _dataset_options:
        _only = _dataset_options[0]
        st.markdown(f"**Dataset:** {_DATASET_LABELS.get(_only, _only)}")

    # BIST sub-switcher.
    if st.session_state.get("dataset") == "bist":
        _basis_options = ["try"]
        if "bist_usd" in _AVAIL_KEYS:
            _basis_options.append("usd")
        if "bist_gold" in _AVAIL_KEYS:
            _basis_options.append("gold")
        if len(_basis_options) > 1:
            st.markdown("**Base currency**")
            st.segmented_control(
                "Base currency",
                _basis_options,
                format_func=lambda b: _BASIS_LABELS.get(b, b),
                key="bist_basis",
                label_visibility="collapsed",
            )

    # Re-read the active universe AFTER both sidebar controls.
    # PORT arda/ui-cleanup item 5: sidebar dataset description caption
    # removed for cleaner left-rail. `Universe.description` stays in
    # universe_registry.py for any other consumer that wants it.
    _active_universe = get_universe(current_universe())
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# Page list + capability gating + page-disappearance pending stash
# ══════════════════════════════════════════════════════════════════════════════
# Streamlit native sidebar nav handles the page-switching widget below this
# section. We just need to:
#   1. Decide which pages to show (capability-gated per active universe).
#   2. When the user flips dataset/basis and a previously-active page
#      disappears (e.g., Cross-Market unavailable on bist_usd), stash their
#      pick so a round-trip back restores it.
#
# This mirrors the Phase S #1 top-nav clamp/pending mechanism but adapted
# for multi-page semantics where Streamlit's URL state, not our session_state,
# is the source of truth for the active page. We use `st.switch_page()` to
# programmatically restore stashed pages on round-trip.

_overview_label = (
    "Market Overview"
    if _cap(_active_universe, 'domain', 'finance') == "finance"
    else "Network Overview"
)

# All possible pages, keyed by canonical title (independent of EEG re-label
# above so the stash logic is consistent across datasets).
_PAGE_PATHS: dict[str, str] = {
    "Cross-Market":    "views/01_cross_market.py",
    "Market Overview": "views/02_market_overview.py",
    "Time Machine":    "views/03_time_machine.py",
    "Signals":         "views/06_signals.py",
    "Pair Analysis":   "views/04_pair_analysis.py",
    "Methods Lab":     "views/05_methods_lab.py",
}

# Build the visible-pages list for the active universe.
# Signals sits between Time Machine and Pair Analysis — "see the network →
# see time evolution → see what's signaling now → deep-dive one pair →
# methods" reads as a natural progression.
visible_titles: list[str] = []
if _cap(_active_universe, 'eligible_for_cross_market', True):
    visible_titles.append("Cross-Market")
visible_titles.append("Market Overview")
visible_titles.append("Time Machine")
if _cap(_active_universe, 'has_pair_trading', True):
    visible_titles.append("Signals")
    visible_titles.append("Pair Analysis")
visible_titles.append("Methods Lab")


# ── Page-disappearance pending stash (Phase S #1 semantic, multi-page port) ─
# Per-dataset key so each dataset remembers what page the user was on.
# When the user is on Pair Analysis (BIST) and flips to EEG (no pair trading),
# Streamlit auto-redirects to the default page; we stash "Pair Analysis" so
# flipping BIST → EEG → BIST restores them to Pair Analysis.
_dataset_key = st.session_state.get("dataset", _BOOT_DATASET)
_last_page_key = f"last_page_{_dataset_key}"
_pending_key = f"{_last_page_key}__pending"
_remembered = st.session_state.get(_last_page_key)

# If we have a pending stash AND it's visible again AND we're currently on
# the default page (first visible), restore the stash.
_restore_target: str | None = None
if _pending_key in st.session_state:
    _pending_value = st.session_state[_pending_key]
    if _pending_value in visible_titles and (
        _remembered is None or _remembered == visible_titles[0]
    ):
        _restore_target = _pending_value
        st.session_state.pop(_pending_key, None)
    elif _remembered and _remembered != visible_titles[0] and _remembered in visible_titles:
        # User navigated away from default; drop stale stash.
        st.session_state.pop(_pending_key, None)


# ══════════════════════════════════════════════════════════════════════════════
# Build st.Page objects + run navigation
# ══════════════════════════════════════════════════════════════════════════════
# Build StreamlitPage objects in the order they should appear in the sidebar.
# `default=True` on the first visible page marks it as the landing page when
# the user hits a URL with no `?page=...` query param.

_pages: list = []
for i, title in enumerate(visible_titles):
    page_label = _overview_label if title == "Market Overview" else title
    _pages.append(
        st.Page(
            _PAGE_PATHS[title],
            title=page_label,
            default=(i == 0),
        )
    )

# `st.navigation` with `position="sidebar"` (default) renders the page list
# in the sidebar below our custom widgets above. The returned StreamlitPage
# is what we call `.run()` on to dispatch.
_pg = st.navigation(_pages, position="sidebar")

# Track the active page in session_state for the next render's pending logic.
# Map the displayed label back to the canonical title so the stash works
# across EEG's "Network Overview" re-label.
_active_title_displayed = _pg.title
_canonical_to_displayed = {t: (_overview_label if t == "Market Overview" else t)
                           for t in visible_titles}
_displayed_to_canonical = {v: k for k, v in _canonical_to_displayed.items()}
_active_title_canonical = _displayed_to_canonical.get(_active_title_displayed, _active_title_displayed)

# If we have a restore target AND we're not already there, programmatically
# navigate. Streamlit's `switch_page` raises a special exception that halts
# the current render and triggers a rerun to the new page.
if _restore_target is not None and _restore_target != _active_title_canonical:
    _restore_path = _PAGE_PATHS[_restore_target]
    st.switch_page(_restore_path)

# If the previously-remembered page disappeared on this dataset (capability
# change), stash it for round-trip restore.
if _remembered and _remembered not in visible_titles and _remembered != visible_titles[0]:
    if _pending_key not in st.session_state:
        st.session_state[_pending_key] = _remembered

# Update tracker for next render.
st.session_state[_last_page_key] = _active_title_canonical

_pg.run()
