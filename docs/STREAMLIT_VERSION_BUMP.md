# Streamlit version bump procedure

We discovered the hard way that pinning Streamlit loosely (`>=`) causes
deploys to break when Streamlit ships a new strict-mode check or removes
a deprecated widget. This doc captures the *one* procedure to follow when
bumping the Streamlit version.

## The rule

Streamlit is pinned **exactly** in three places, all of which must match:

| Where | What |
|---|---|
| `requirements.txt` | `streamlit==1.41.1` |
| `README.md` YAML | `sdk_version: 1.41.1` |
| `pyproject.toml` (if present) | `"streamlit==1.41.1"` |

A mismatch means local dev, Streamlit Cloud, and HF Spaces install
different versions and we end up debugging API-incompatibility bugs in
production.

## When to bump

- A new Streamlit feature you want to use (e.g. `st.dataframe` got
  column-config-v2 in 1.45).
- A security advisory you care about.
- **Never** "just to be on the latest" — every bump is a risk.

## How to bump (no shortcuts)

### 1. Read the changelog from current → target

https://github.com/streamlit/streamlit/releases — read every release
between your current pin and your target. Note any:
- **Breaking changes** (removed widgets, renamed params)
- **New strict-mode checks** (1.41 added the popover-nesting rejection)
- **Deprecations** that fire warnings on widgets you use

### 2. Bump the three pins in one PR

Single commit, three files. Do not split.

```bash
# In your branch:
sed -i '' 's/streamlit==1\.41\.1/streamlit==1.45.0/' requirements.txt
sed -i '' 's/sdk_version: 1\.41\.1/sdk_version: 1.45.0/' README.md
# pyproject.toml if present
```

### 3. Run the smoke suite locally

```bash
uv sync --all-extras
uv run python -m pytest tests/test_dashboard_smoke.py -v
```

This is the test suite specifically designed to catch Streamlit-API
incompatibilities. It includes:
- **AppTest-based** end-to-end renders of every universe + every nav page
- **Static AST checks** for things AppTest can't catch (e.g. nested popovers)

If the suite fails, the changelog told you which API broke. Fix it in
the same PR. If the suite passes locally, it'll pass in CI.

### 4. Manual click-through

The AppTest suite covers the most common breakages but not 100%. Spin
up the dashboard locally:

```bash
uv run streamlit run app/dashboard.py
```

…and click every tab, every sub-tab, every popover. Anything that
crashes in the browser should get a new test added to
`tests/test_dashboard_smoke.py` so the next bump catches it
automatically.

### 5. Open the PR + let CI gate it

The `CI smoke tests` workflow (`.github/workflows/ci.yml`) runs on every
PR. If the smoke suite fails on CI, do NOT merge. Fix the regression in
the same PR.

After merge, the `Deploy to Hugging Face Spaces` workflow re-runs the
same smoke suite as a gate before pushing to HF. Belt-and-suspenders:
even if branch protection isn't enabled, broken code can't deploy.

## Bug surface to add to the smoke suite

When you find a new Streamlit-API breakage in production, **add a test
for it** to `tests/test_dashboard_smoke.py`. The suite is documented in
its own module docstring; pattern is:

- **API existence bugs** (widget removed) → AppTest catches automatically
  via "exception captured during render"
- **Strict-mode violations** (nested popovers, etc.) → write a static
  AST check like `test_no_nested_popovers_anywhere_in_app`. AppTest
  doesn't see these because it doesn't simulate user interactions.
- **Stateful UI bugs** (e.g. session_state-key collisions) → add an
  AppTest scenario that sets the relevant session_state and reruns.

The suite ratchets up with every bug we find. After 5-10 production bugs
caught + tested, the surface area of "stuff that breaks on deploy" should
shrink to near-zero.

---

## Widget-API stability matrix

The dashboard's runtime dependencies on Streamlit APIs, with the minimum
version that supports each, and any known deprecation. Consult this table
**before** bumping `streamlit==X.Y.Z` — if the bump crosses any "removed
in" line below, the migration in the same PR is mandatory.

### APIs we use (sorted by introduction version)

| API | Min version | Status @ 1.41.1 | Notes |
|---|---|---|---|
| `st.selectbox`, `st.dataframe`, `st.button`, `st.checkbox`, `st.slider`, `st.radio`, `st.text_input`, `st.markdown`, `st.write`, `st.metric`, `st.columns`, `st.tabs`, `st.expander`, `st.form`, `st.form_submit_button`, `st.date_input`, `st.color_picker`, `st.select_slider` | ≤ 1.16 | ✅ Stable | Core widgets — safe across the foreseeable future |
| `@st.cache_data`, `@st.cache_resource` | 1.18 | ✅ Stable | Replaces legacy `@st.cache` |
| `st.status` | 1.22 | ✅ Stable | Spinner-like context manager |
| `st.popover` | 1.32 | ⚠️ **Strict** | 1.41+ rejects: nested in popover, columns inside (when popover is itself in a column) |
| `st.column_config.NumberColumn` (+ `format=`) | 1.36 | ✅ Stable | Format-string syntax stable through 1.41.1 |
| `st.container(border=True)` | 1.40 | ✅ Stable | `border=` kwarg specifically |
| **`st.segmented_control`** | **1.41** | ✅ Stable | **At pinned version — downgrade past 1.41 breaks the dashboard** |

### Deprecated APIs we still touch

| API | Deprecated | Removed | Our exposure | Migration |
|---|---|---|---|---|
| `use_container_width=True/False` on `st.dataframe`, `st.popover`, `st.button`, `st.download_button`, `st.plotly_chart` | 1.41 (warning) | end-2025 (~1.46+) | Migrated to `width="stretch"/"content"` in PR `feat/streamlit-resilience-pass`. Only remaining mention: back-compat shim in `app/utils.py:render_chart`. Test: `test_no_use_container_width_outside_shim`. | `True` → `"stretch"`; `False` → `"content"` |

### Strict-mode rules enforced from Streamlit 1.41+

These rules fire `StreamlitAPIException` at the first violating render. AppTest
doesn't always trigger them (popover/expander bodies aren't auto-opened), so
each is also guarded by a static AST check in `tests/test_dashboard_smoke.py`.

| Rule | AST check | Hits seen in production |
|---|---|---|
| `st.popover` cannot contain another `st.popover` | `test_no_nested_popovers_anywhere_in_app` | PR #19 (Data Freshness inside Settings) |
| `st.columns` cannot be nested 2+ levels deep, **including transitively through `st.popover`** | `test_no_columns_inside_popover_anywhere_in_app` | PR #21 (chart_export, event markers) |
| `st.expander` cannot contain another `st.expander`; expander-in-popover smells | `test_no_expanders_inside_popovers` | None yet |
| `st.form` cannot contain another `st.form` | (no AST check yet; no current violations) | None yet |
| `st.tabs` cannot be nested directly | (no AST check yet; no current violations) | None yet |

### Robustness guards (not Streamlit-specific, but caught by the same suite)

| Pattern | AST check | Notes |
|---|---|---|
| `format_func=lambda x: {a:b}[x]` (KeyError if option not in dict) | `test_format_func_lambdas_use_get_not_subscript` | Use `dict.get(x, str(x))` |
| `render_chart` signature must accept `width` and forward to `st.plotly_chart` | `test_render_chart_signature_and_passthrough` | Defends the deprecation-shim contract |
| Universe `getattr` defensive fallback | `test_capability_getattr_fallback_for_missing_attr` | Defends against stale-module-class cache |

### Bump-safety procedure (use this every Streamlit upgrade)

1. Read changelogs from current → target version (look for "removed", "breaking", "now rejects").
2. Update the three pins (requirements.txt, README.md, pyproject.toml if present).
3. `uv sync && uv run python -m pytest -v` — the 15 smoke tests + 3 AST checks must stay green.
4. If new strict-mode rules landed → add an AST check to the matrix above.
5. If a widget you depended on was removed → migrate, and add a static check for the new API.
6. Push to a PR. CI gate (`.github/workflows/ci.yml`) re-runs the suite.
7. Merge. Deploy gate re-runs the suite. HF Spaces rebuilds.

Step 4 is the critical one — every production bug we hit becomes a permanent
member of the matrix above. The matrix grows; the production bug rate shrinks.
