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
