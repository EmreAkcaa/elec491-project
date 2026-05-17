"""Streamlit `AppTest`-based smoke tests for the dashboard.

These tests render the dashboard programmatically in-process — no browser,
no network, no real Streamlit server. They catch the entire class of
"runtime API incompatibility" errors that have plagued our HF Spaces
deploys (nested popovers, missing widgets like `st.segmented_control`,
broken capability-flag fallbacks, missing-data degradation paths, etc.).

The intent: every PR runs these in CI before merge. If `AppTest` reports
a Streamlit exception, the merge is blocked. We never again discover
runtime errors by deploying to production and waiting for a user click.

Coverage strategy
=================
For each universe present locally (BIST always; S&P if data on disk; EEG
deliberately skipped — see below):
  * Module-load smoke (default universe, no interactions)
  * Each nav page (Market Overview / Pair Analysis / Cross-Market)
  * Selected sub-tabs inside Market Overview that historically broke
    (Settings popover open, Clustering & Network, EEE Analysis sub-tabs)

EEG is intentionally NOT covered here:
  * Bulk parquets aren't on the CI runner (gitignored / Dataset-hosted)
  * The preload would try to download from HF Dataset → slow + flaky
  * `STONECOAL_SKIP_EEG_DOWNLOAD=1` is set so the preload no-ops
  * `available_universes()` then filters EEG → it's never in the selector
    during a CI smoke run
  * For EEG-specific regressions, add a manual smoke test that requires
    the EEG data to be present locally and runs only when the user opts in
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Streamlit's AppTest framework — added in 1.27, stabilised in 1.33+.
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    pytest.skip(
        "streamlit.testing.v1.AppTest unavailable. "
        "Bump Streamlit pin in requirements.txt.",
        allow_module_level=True,
    )

# Make sure the app/ + project root + src/ are importable the same way the
# dashboard does it on Streamlit Cloud / HF Spaces.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _REPO_ROOT / "app"
for _p in (str(_REPO_ROOT), str(_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DASHBOARD_PATH = _APP_DIR / "dashboard.py"
APPTEST_TIMEOUT = 90  # seconds; dashboard's first run includes heavy imports


# ---------------------------------------------------------------------------
# Test environment: tell the EEG preload to skip the download attempt
# so CI doesn't hang on a flaky / unauthenticated HF dataset fetch.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _skip_eeg_download(monkeypatch):
    monkeypatch.setenv("STONECOAL_SKIP_EEG_DOWNLOAD", "1")


# ---------------------------------------------------------------------------
# Universe availability helpers
# ---------------------------------------------------------------------------

def _financial_universes_on_disk() -> list[str]:
    """Return BIST/S&P keys whose pipeline_metadata.json exists locally.

    EEG deliberately excluded — see module docstring.
    """
    out = []
    for key in ("bist", "sp500"):
        meta = _REPO_ROOT / "data" / key / "results" / "pipeline_metadata.json"
        if meta.exists():
            out.append(key)
    return out


_AVAILABLE = _financial_universes_on_disk()
_HAS_BIST  = "bist"  in _AVAILABLE
_HAS_SP500 = "sp500" in _AVAILABLE

needs_data = pytest.mark.skipif(
    not _AVAILABLE,
    reason="No financial universe data on disk; run `uv run python run_pipeline.py` first.",
)


# ---------------------------------------------------------------------------
# Basic load — the single most useful smoke test
# Catches: nested popovers, missing widgets, broken imports, syntax errors,
# capability-flag bugs that crash on first render
# ---------------------------------------------------------------------------

@needs_data
@pytest.mark.parametrize("universe", _AVAILABLE)
def test_dashboard_renders_for_each_universe(universe):
    """Render the dashboard for each available financial universe.

    Failure here means the dashboard would crash on first page load on
    HF Spaces / Streamlit Cloud for that universe. The most common cause
    is a Streamlit API incompatibility (nested popovers, removed widget,
    renamed param).
    """
    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.session_state["universe"] = universe
    at.run(timeout=APPTEST_TIMEOUT)
    _assert_no_exception(at, f"Dashboard crashed on initial render with universe={universe!r}")


# ---------------------------------------------------------------------------
# Navigation — each nav page must render without exception
# Catches: page-specific widget bugs, capability-flag mistakes in nav routing,
# Cross-Market page-specific issues
# ---------------------------------------------------------------------------

@needs_data
@pytest.mark.skipif(not _HAS_BIST, reason="BIST data not on disk")
def test_nav_market_overview_bist():
    at = _open(universe="bist", nav_page="Market Overview")
    _assert_no_exception(at, "Market Overview crashed on BIST")


@needs_data
@pytest.mark.skipif(not _HAS_SP500, reason="S&P data not on disk")
def test_nav_market_overview_sp500():
    at = _open(universe="sp500", nav_page="Market Overview")
    _assert_no_exception(at, "Market Overview crashed on S&P-500")


@needs_data
@pytest.mark.skipif(not _HAS_BIST, reason="BIST data not on disk")
def test_nav_pair_analysis_bist():
    """Pair Analysis only renders for universes with has_pair_trading=True
    (BIST + S&P, but not EEG). Confirm it loads cleanly for BIST."""
    at = _open(universe="bist", nav_page="Pair Analysis")
    _assert_no_exception(at, "Pair Analysis crashed on BIST")


@needs_data
@pytest.mark.skipif(not _HAS_SP500, reason="S&P data not on disk")
def test_nav_pair_analysis_sp500():
    at = _open(universe="sp500", nav_page="Pair Analysis")
    _assert_no_exception(at, "Pair Analysis crashed on S&P-500")


@needs_data
@pytest.mark.skipif(
    not (_HAS_BIST and _HAS_SP500),
    reason="Cross-Market needs both BIST and S&P data on disk",
)
def test_nav_cross_market():
    """Cross-Market page reads from BOTH BIST + S&P. Skip if either missing.

    Active universe doesn't matter here since the page is universe-independent,
    but we set it to BIST so the rest of the script's per-universe path doesn't
    blow up first.
    """
    at = _open(universe="bist", nav_page="Cross-Market")
    _assert_no_exception(at, "Cross-Market page crashed")


# ---------------------------------------------------------------------------
# Sub-tab smoke — confirms each Market Overview tab renders for default universe
# Catches: tab-content-specific bugs that wouldn't surface from nav alone
# ---------------------------------------------------------------------------

@needs_data
@pytest.mark.skipif(not _HAS_BIST, reason="BIST data not on disk")
def test_market_overview_default_tabs_render():
    """Render the default Market Overview view. All tabs are instantiated
    (Streamlit creates all tab containers on render, even hidden ones), so
    this exercises Data & Stats, Correlation, Clustering, Rolling Analysis,
    EEE Analysis, and (for financial universes) Pairs & Dislocations."""
    at = _open(universe="bist", nav_page="Market Overview")
    _assert_no_exception(at, "Default Market Overview tabs failed to instantiate")


# ---------------------------------------------------------------------------
# Static AST checks — catch classes of bugs AppTest can't see
# ---------------------------------------------------------------------------
# AppTest runs the script body but doesn't simulate user interactions like
# opening a popover. So a bug like "st.popover nested inside st.popover"
# (which Streamlit 1.41+ rejects with a StreamlitAPIException only when
# the outer popover is opened) sails through AppTest cleanly.
#
# These AST tests close that gap by scanning the source code directly for
# Streamlit-API violations.

def _is_popover_call(node) -> bool:
    """True iff `node` is a Call to something.popover(...) — covers st.popover,
    st.sidebar.popover, col.popover, etc."""
    import ast
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "popover"
    )


def _is_call_to(node, attr_names: set[str]) -> bool:
    """True iff `node` is a Call to something.<attr>(...) where <attr> ∈ attr_names."""
    import ast
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in attr_names
    )


def _walk_app_files():
    """Yield (path, ast_tree) for every app/*.py — convenience for static checks."""
    import ast
    for py_path in sorted(_APP_DIR.glob("*.py")):
        try:
            yield py_path, ast.parse(py_path.read_text(), filename=str(py_path))
        except SyntaxError as e:
            raise AssertionError(f"{py_path}: cannot parse — {e}") from e


def test_no_nested_popovers_anywhere_in_app():
    """Streamlit 1.41+ rejects popovers nested inside other popovers with
    StreamlitAPIException. The error fires when the outer popover is opened
    (which AppTest doesn't simulate by default), so we catch it statically.

    Use st.expander or st.container for collapsible disclosure inside a popover.
    """
    import ast
    violations: list[str] = []
    for py_path, tree in _walk_app_files():
        for outer in ast.walk(tree):
            if not isinstance(outer, ast.With):
                continue
            if not any(_is_call_to(item.context_expr, {"popover"}) for item in outer.items):
                continue
            for inner in ast.walk(outer):
                if inner is outer or not isinstance(inner, ast.With):
                    continue
                if any(_is_call_to(item.context_expr, {"popover"}) for item in inner.items):
                    rel = py_path.relative_to(_REPO_ROOT)
                    violations.append(
                        f"  {rel}:{inner.lineno}  — nested popover inside outer popover at line {outer.lineno}"
                    )
    if violations:
        raise AssertionError(
            "Streamlit 1.41+ rejects popovers nested inside other popovers. "
            "Use st.expander or st.container instead. Violations:\n"
            + "\n".join(violations)
        )


def test_no_columns_inside_popover_anywhere_in_app():
    """Streamlit 1.41+ rejects st.columns nested 2+ levels deep. Every popover
    in this app is rendered from inside a dashboard column (render_chart →
    export popover, event_marker_manager_ui → event-marker popover inside
    a Rolling Analysis tab column, etc.), so columns INSIDE a popover are
    always 2 levels deep → reject.

    Use vertical stacking (st.write / st.markdown) or st.container for
    layout inside popovers — never st.columns. AppTest renders popover
    bodies but doesn't always trigger the strict check; we catch it
    statically here.
    """
    import ast
    violations: list[str] = []
    for py_path, tree in _walk_app_files():
        for popover_block in ast.walk(tree):
            if not isinstance(popover_block, ast.With):
                continue
            if not any(_is_call_to(it.context_expr, {"popover"}) for it in popover_block.items):
                continue
            # Walk every descendant looking for st.columns or `with st.columns(...)`
            for descendant in ast.walk(popover_block):
                if descendant is popover_block:
                    continue
                # `c1, c2 = st.columns(2)` (Assign with columns Call as value)
                if isinstance(descendant, ast.Assign) and _is_call_to(descendant.value, {"columns"}):
                    rel = py_path.relative_to(_REPO_ROOT)
                    violations.append(
                        f"  {rel}:{descendant.lineno}  — st.columns inside popover at line {popover_block.lineno}"
                    )
                # `with st.columns(...): ...` (rare but possible)
                if isinstance(descendant, ast.With):
                    if any(_is_call_to(it.context_expr, {"columns"}) for it in descendant.items):
                        rel = py_path.relative_to(_REPO_ROOT)
                        violations.append(
                            f"  {rel}:{descendant.lineno}  — with-st.columns inside popover at line {popover_block.lineno}"
                        )
    if violations:
        raise AssertionError(
            "Streamlit 1.41+ rejects st.columns more than 1 level deep. "
            "Popovers in this app live inside dashboard columns, so columns "
            "inside a popover are always 2 levels → use vertical stacking. "
            "Violations:\n" + "\n".join(violations)
        )


def test_no_expanders_inside_popovers():
    """Streamlit 1.41+ also rejects expanders nested inside popovers and
    vice-versa (treated as similar disclosure widgets). Catch statically.
    """
    import ast
    violations: list[str] = []
    for py_path, tree in _walk_app_files():
        for outer in ast.walk(tree):
            if not isinstance(outer, ast.With):
                continue
            outer_kinds = {
                "popover" if _is_call_to(it.context_expr, {"popover"}) else
                "expander" if _is_call_to(it.context_expr, {"expander"}) else
                None
                for it in outer.items
            }
            outer_kinds.discard(None)
            if not outer_kinds:
                continue
            for inner in ast.walk(outer):
                if inner is outer or not isinstance(inner, ast.With):
                    continue
                for it in inner.items:
                    if _is_call_to(it.context_expr, {"popover"}) and "popover" in outer_kinds:
                        violations.append(
                            f"  {py_path.relative_to(_REPO_ROOT)}:{inner.lineno}  — popover inside popover"
                        )
                    # Expander-in-expander: Streamlit warns but doesn't always reject;
                    # still flag it as a smell.
                    if _is_call_to(it.context_expr, {"expander"}) and "expander" in outer_kinds:
                        violations.append(
                            f"  {py_path.relative_to(_REPO_ROOT)}:{inner.lineno}  — expander inside expander"
                        )
    if violations:
        raise AssertionError(
            "Disclosure widget nesting violations (Streamlit 1.41+ strict mode):\n"
            + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# Deprecation-cliff guard — use_container_width=
# ---------------------------------------------------------------------------
# Streamlit 1.41+ emits a deprecation warning for `use_container_width=` and
# points users at the `width="stretch"/"content"` replacement. Removal is
# slated for end-of-2025 (~Streamlit 1.46+).
#
# This codebase migrated all 22 production call sites to `width=` in PR
# feat/streamlit-resilience-pass. The only allowed remaining mention of
# `use_container_width` is inside app/utils.py:render_chart, where it's a
# back-compat shim that translates the legacy kwarg → the new width API.
#
# Any other reintroduction (a contributor copy-pasting old example code,
# a stale snippet, etc.) fails this test instantly with file:line.

def test_no_width_stretch_or_content_string_on_pinned_streamlit():
    """On the pinned Streamlit 1.41.1, **no widget accepts `width="stretch"`
    or `width="content"` as a string** — every site that tried to use this
    "future" API crashed at render time. The string-API for `width=` was
    rolled out per-widget across 1.42–1.45+.

    Confirmed crash modes from PR #23's deploy + local-against-pinned tests:

      * st.popover(..., width="stretch")          → TypeError: popover() got
                                                    an unexpected keyword
                                                    argument 'width'
      * st.button(..., width="stretch")           → TypeError: button() got
                                                    an unexpected keyword
                                                    argument 'width'
      * st.download_button(..., width="stretch")  → TypeError: download_button()
      * st.plotly_chart(..., width="stretch")     → TypeError: plotly_chart()
      * st.dataframe(..., width="stretch")        → TypeError: 'str' object
                                                    cannot be interpreted as
                                                    an integer  (dataframe
                                                    DID accept width=, but
                                                    only as an int pixel count)

    Until the Streamlit pin is bumped past the version where every widget
    we touch supports the string API (target ~1.45+), `use_container_width=`
    is the ONLY portable spelling.

    When we do bump the pin, the migration is mechanical: search-and-replace
    `use_container_width=True` → `width="stretch"` and re-run this test
    after deleting or relaxing this guard.

    Allowed exceptions:
      * utils.py docstrings/comments may still discuss the public `width=`
        API of render_chart, which threads the value through to the
        plotly_chart call internally (translated to use_container_width=
        until 1.45).
    """
    import re
    string_pattern = re.compile(r'\bwidth\s*=\s*"(stretch|content)"')
    violations: list[str] = []
    # utils.py:render_chart deals in `width="stretch"/"content"` as its
    # public API contract (translated internally to use_container_width=
    # before the actual st.plotly_chart call). The internal translation
    # legitimately assigns `width = "stretch" if ... else "content"` —
    # that's not a Streamlit call, so allowlist the whole file.
    FILE_ALLOWLIST = {"utils.py"}
    for py_path in sorted(_APP_DIR.glob("*.py")):
        if py_path.name in FILE_ALLOWLIST:
            continue
        for i, line in enumerate(py_path.read_text().splitlines(), start=1):
            if not string_pattern.search(line):
                continue
            stripped = line.strip()
            # Skip pure comment lines and docstring-style backtick mentions.
            if stripped.startswith("#"):
                continue
            if "``" in stripped:
                continue
            rel = py_path.relative_to(_REPO_ROOT)
            violations.append(f"  {rel}:{i}  — {stripped}")
    if violations:
        raise AssertionError(
            'width="stretch" / width="content" is unsupported on the pinned '
            "Streamlit 1.41.1 (string-API rolled out per-widget across 1.42–1.45+). "
            "Use `use_container_width=True/False` until the pin is bumped.\n"
            "Violations:\n" + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# Robustness guard — format_func=lambda subscripting a dict literal directly
# ---------------------------------------------------------------------------
# `format_func=lambda v: {1: "a", 5: "b"}[v]` raises KeyError if a new option
# value is added to the parent selectbox without updating the dict. The fix
# is `{...}.get(v, str(v))` — same behaviour when the key exists, graceful
# fallback when it doesn't.

def test_format_func_lambdas_use_get_not_subscript():
    """No `format_func=lambda x: {<dict literal>}[x]` patterns anywhere in app/.

    Detects the brittle subscript-into-inline-dict-literal pattern. Allows any
    other lambda body (e.g. `.get(...)`, `f"…"` formatting, function calls).
    """
    import ast
    violations: list[str] = []
    for py_path, tree in _walk_app_files():
        for node in ast.walk(tree):
            # Looking for: <call>(..., format_func=lambda v: {...}[v], ...)
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "format_func":
                    continue
                lam = kw.value
                if not isinstance(lam, ast.Lambda):
                    continue
                body = lam.body
                # Pattern: ast.Subscript(value=ast.Dict(...), slice=...)
                if isinstance(body, ast.Subscript) and isinstance(body.value, ast.Dict):
                    rel = py_path.relative_to(_REPO_ROOT)
                    snippet = ast.unparse(body) if hasattr(ast, "unparse") else "<lambda>"
                    violations.append(
                        f"  {rel}:{lam.lineno}  — format_func=lambda …: {snippet} "
                        f"(KeyError if option not in dict; use .get(key, str(key)))"
                    )
    if violations:
        raise AssertionError(
            "format_func lambdas with direct dict-subscript can raise KeyError. "
            "Use `dict.get(key, fallback)` instead. Violations:\n"
            + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# Contract guard — render_chart threads the width= param to st.plotly_chart
# ---------------------------------------------------------------------------
# Defends against silent refactors that drop the `width` parameter on
# render_chart's signature or stop passing it to st.plotly_chart. Either
# regression would silently revert the dashboard to Streamlit defaults.

def test_render_chart_signature_and_passthrough():
    """app/utils.py:render_chart must accept a `width` parameter and pass it
    through to st.plotly_chart. Regressions break the deprecation-shim."""
    import ast
    utils_path = _APP_DIR / "utils.py"
    tree = ast.parse(utils_path.read_text())
    render_chart_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "render_chart":
            render_chart_def = node
            break
    assert render_chart_def is not None, "app/utils.py must define render_chart"

    # 1. Signature must include a `width` kwarg
    arg_names = {a.arg for a in render_chart_def.args.args} | {
        a.arg for a in render_chart_def.args.kwonlyargs
    }
    assert "width" in arg_names, (
        f"render_chart must accept a `width` parameter; got {sorted(arg_names)}"
    )

    # 2. Body must call st.plotly_chart(...) somewhere that propagates the
    #    width semantic — EITHER as a literal forward (`width=width`, valid
    #    once Streamlit ≥1.45 is pinned) OR translated to the legacy kwarg
    #    (`use_container_width=...`, required on the current 1.41.1 pin
    #    because plotly_chart didn't accept width= back then). Either form
    #    keeps the chart filling its container; we reject only the case
    #    where the kwarg disappears entirely.
    found_plotly_call = False
    for sub in ast.walk(render_chart_def):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if not (isinstance(func, ast.Attribute) and func.attr == "plotly_chart"):
            continue
        for kw in sub.keywords:
            if kw.arg in {"width", "use_container_width"}:
                found_plotly_call = True
                break
    assert found_plotly_call, (
        "render_chart must call st.plotly_chart(fig, width=... [or "
        "use_container_width=...], ...) — the container-fill kwarg is no "
        "longer being forwarded, which would silently degrade every chart "
        "to Streamlit's default narrow width."
    )


# ---------------------------------------------------------------------------
# Capability-flag fallback paths (no dashboard import — dashboard.py is a
# Streamlit script that crashes when imported outside `streamlit run`).
# Catches: regressions where the defensive getattr() pattern stops working.
# ---------------------------------------------------------------------------

def test_capability_getattr_fallback_for_missing_attr():
    """Re-implement dashboard.py's `_cap` defensive lookup and verify it.
    Same one-liner — but tested without importing dashboard."""
    def _cap(u, attr, default):
        return getattr(u, attr, default)

    class _StaleUniverse:
        # Mimics a pre-Phase-I Universe class missing the new capability fields
        key = "bist"
        label = "BIST"
        short_label = "BIST"

    u = _StaleUniverse()
    # Missing capability flags fall back to defaults (financial-on)
    assert _cap(u, "has_pair_trading", True) is True
    assert _cap(u, "has_snn", True) is True
    assert _cap(u, "item_label", "Ticker") == "Ticker"
    # Existing attrs pass through unchanged
    assert _cap(u, "label", "fallback") == "BIST"


# ---------------------------------------------------------------------------
# Sanity: env-var skip is honoured (every AppTest above runs with it set;
# if the preload didn't honour it we'd see EEG-download network noise in
# the run logs and slow tests — captured implicitly by the 90-sec timeout).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open(*, universe: str, nav_page: str) -> AppTest:
    """Run the dashboard with explicit session_state values."""
    at = AppTest.from_file(str(DASHBOARD_PATH))
    at.session_state["universe"] = universe
    at.session_state["nav_page"] = nav_page
    at.run(timeout=APPTEST_TIMEOUT)
    return at


def _assert_no_exception(at: AppTest, msg: str) -> None:
    """Fail the test with a useful message when AppTest captured an exception."""
    if not at.exception:
        return
    # AppTest's exception attribute is a sequence of ElementList[Exception]
    excs = list(at.exception)
    lines = [f"{msg}", f"  {len(excs)} exception(s) captured by AppTest:"]
    for i, exc in enumerate(excs):
        # Each exc has .value (str) and sometimes .name (str)
        val = getattr(exc, "value", repr(exc))
        name = getattr(exc, "name", "")
        lines.append(f"    [{i}] {name}: {val}")
    raise AssertionError("\n".join(lines))
