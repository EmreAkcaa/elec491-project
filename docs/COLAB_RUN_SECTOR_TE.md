# Colab runbook: K=10,000 sector transfer-entropy on BIST

**Audience**: a fresh Claude (or human) session running on Google Colab. You
have:
- A Google Colab tab open (Pro recommended for CPU headroom, but free will
  finish).
- A GitHub personal-access token with `repo` scope for
  `EmreAkcaa/elec491-project`. Treat it like a password.
- No prior context about the StoNeCoAl project.

**Mission**: run sector-aggregated transfer entropy with 10,000 surrogate
shuffles on the BIST returns panel, write the result back to the repo, push
the commit. ~40 minutes of CPU. One file changes
(`data/bist/results/te_sector_matrix.parquet`). One follow-up doc edit.

Nothing else in this conversation matters. This file is self-contained.

---

## Why we're doing this

The local pipeline already produces `te_sector_matrix.parquet` at K=1000
surrogate shuffles. On 156 directed sector pairs, the Benjamini–Hochberg
FDR cutoff for the strongest edge is `α / m = 0.05 / 156 ≈ 3.2 × 10⁻⁴`. At
K=1000, the minimum achievable p-value is `1 / (K + 1) ≈ 1.0 × 10⁻³` — an
order of magnitude **above** the FDR cutoff, so 0 edges survive correction
by mathematical resolution alone, not by absence of signal. 26 of 156
edges pass uncorrected.

At K=10,000 the minimum p-value drops to `~1.0 × 10⁻⁴`, comfortably below
the BH cutoff. Expected outcome: **5–15 sector edges clear FDR**, including
the strongest `Conglomerates → Insurance / Consumer Durables / Retail /
Steel` edges. That turns the current "uncorrected pattern" finding into a
clean "FDR-survivor directed sector flow" — the cleanest thesis-defensible
IT-native result on this dataset.

---

## Step 0 — Make the token available to the Colab notebook

In the left-hand panel of Colab, click the 🔑 **Secrets** icon. Add a secret
named `GITHUB_TOKEN` with your PAT as the value. Enable
"Notebook access" for it.

The cells below read it via `from google.colab import userdata` →
`userdata.get("GITHUB_TOKEN")`. Never hardcode the token in a notebook cell.

---

## Step 1 — Clone the repo

Paste this into the first Colab cell and run:

```python
import os
from google.colab import userdata

TOKEN = userdata.get("GITHUB_TOKEN")
if not TOKEN:
    raise SystemExit("GITHUB_TOKEN not set in Colab secrets — see Step 0.")

REPO = "EmreAkcaa/elec491-project"
WORKDIR = "/content/elec491-project"

if not os.path.exists(WORKDIR):
    !git clone https://{TOKEN}@github.com/{REPO}.git {WORKDIR}
os.chdir(WORKDIR)
!git pull --rebase
print("Repo ready at", os.getcwd())
```

Verify it printed the repo path and listed `data/`, `src/`, `app/`,
`docs/` etc.

---

## Step 2 — Install dependencies

```python
!pip install -q numpy pandas pyarrow joblib scipy
```

We only need the libraries `compute_sector_te_matrix` and its helpers
touch. `pyarrow` is for parquet IO.

---

## Step 3 — Sanity-check the inputs

We need two files. They are checked into the repo, so the clone in Step 1
already pulled them.

```python
import pandas as pd

returns = pd.read_parquet("data/bist/processed/log_returns.parquet")
clusters = pd.read_csv("data/bist/results/cluster_assignments.csv")

print(f"log_returns: {returns.shape[0]} days × {returns.shape[1]} tickers")
print(f"date range: {returns.index.min()} → {returns.index.max()}")
print(f"cluster_assignments: {len(clusters)} rows, sectors = {clusters['sector'].nunique()}")

sector_map = dict(zip(clusters["ticker"], clusters["sector"]))
print("First 5 sector entries:", list(sector_map.items())[:5])
```

Expected: ~1543 days × 73 tickers, 22 unique sectors. If the numbers are
materially different, something has changed in the upstream pipeline and
this runbook may not apply cleanly — stop and let a human review before
running.

---

## Step 4 — The compute

The function `compute_sector_te_matrix` lives in `src/transfer_entropy.py`.
It aggregates tickers into equal-weight sector portfolios (sectors with
≥3 tickers; BIST gives 13), then runs pairwise transfer entropy with
surrogate-null testing + BH-FDR.

```python
import sys
sys.path.insert(0, "/content/elec491-project")

from src.transfer_entropy import compute_sector_te_matrix
import time

t0 = time.perf_counter()
te_sector_df = compute_sector_te_matrix(
    returns=returns,
    sector_map=sector_map,
    min_tickers_per_sector=3,
    n_shuffles=10_000,        # ← the only change vs the local pipeline
    lag=1,
    n_bins=3,
    block_length=5,
    multiple_testing="fdr_bh",
    significance_level=0.05,
    seed=42,
)
elapsed = time.perf_counter() - t0
print(f"Wall time: {elapsed/60:.1f} min, output shape: {te_sector_df.shape}")
```

**Expected wall time**:
- Colab Pro (8 vCPU): **30–45 minutes**.
- Colab free (2 vCPU): **2–4 hours**. Don't close the tab.

The function uses `joblib(n_jobs=-1)` internally, so all available CPUs
get used.

**While it runs**, the Colab cell will show no progress bar by default. To
see something live, drop `joblib` to `n_jobs=1` and `verbose=10` — but that's
~10× slower. The default parallel run produces no console output until
done. That's fine; check the tab is still active every 10–15 min.

---

## Step 5 — Inspect the result

```python
n_fdr = int(te_sector_df["significant_fdr"].sum())
n_unc = int(te_sector_df["significant_uncorrected"].sum())
print(f"FDR survivors:        {n_fdr} / {len(te_sector_df)}")
print(f"Uncorrected (α=0.05): {n_unc} / {len(te_sector_df)}")

print("\nTop 15 FDR-surviving edges (or top 15 by TE if none FDR-survive):")
if n_fdr > 0:
    top = te_sector_df[te_sector_df["significant_fdr"]].sort_values("te", ascending=False).head(15)
else:
    top = te_sector_df.sort_values("te", ascending=False).head(15)
print(top[["source", "target", "te", "p_value", "significant_fdr"]].to_string(index=False))
```

What you should see:
- `n_fdr` likely in `[5, 15]`.
- Top edges should include some of: `Conglomerates → Insurance`,
  `Conglomerates → Consumer Durables`, `Insurance → Consumer Durables`,
  `Conglomerates → Retail / Steel`, `Defense → Technology`, `Energy →
  Insurance`.
- Schema columns must match: `[source, target, te, p_value,
  significant_fdr, significant_uncorrected, n_tickers_source,
  n_tickers_target]`.

If `n_fdr == 0` even at K=10,000, the math case for proceeding has not
been met — stop, report back, don't commit.

---

## Step 6 — Write the parquet back

```python
out_path = "data/bist/results/te_sector_matrix.parquet"
te_sector_df.to_parquet(out_path, index=False)
print(f"Wrote {out_path}")
```

Verify by reading it back:

```python
check = pd.read_parquet(out_path)
print(check.shape, list(check.columns))
```

---

## Step 7 — Update the docs

`docs/INFORMATION_THEORY.md` currently has §11 ("Sector-aggregated TE")
with the K=1000 numbers (0 FDR survivors, 26 uncorrected). Update the
table with your fresh K=10,000 numbers.

The simplest way from inside Colab is to read + edit + write:

```python
import re
doc_path = "docs/INFORMATION_THEORY.md"
text = open(doc_path).read()

# Replace the headline finding line. Match the K=1000 phrasing.
new_summary = (
    f"**Current BIST findings** (K=10,000, {n_fdr} FDR survivors, "
    f"{n_unc} of 156 uncorrected significant). Top edges by TE:"
)
text = re.sub(
    r"\*\*Current BIST findings\*\* \(K=1000.*?Top edges by TE:",
    new_summary,
    text,
    flags=re.DOTALL,
)

# Also update the bracketed sentence further down that mentions K=10,000
# would clear FDR — replace it with the past-tense version.
text = text.replace(
    "K=10,000 (one-time Colab job, ~40 min) would clear FDR on the strongest edges.",
    f"K=10,000 ran on Colab via `docs/COLAB_RUN_SECTOR_TE.md` ({n_fdr} FDR survivors on the strongest sector edges).",
)

with open(doc_path, "w") as f:
    f.write(text)
print("Updated", doc_path)
```

If the regex doesn't match cleanly (the doc may have been edited in the
meantime), open the file in Colab's editor and manually replace the K=1000
table with a K=10,000 one — the structure is the same, just different
numbers.

---

## Step 8 — Commit and push

```python
import subprocess

# Configure git identity (one-time per Colab session).
!git config user.email "colab-bot@stonecoal.local"
!git config user.name "stonecoal-colab"

# Show what changed.
!git status --short
!git diff --stat

# Stage + commit.
!git add data/bist/results/te_sector_matrix.parquet docs/INFORMATION_THEORY.md
commit_msg = f"sector TE at K=10000: {n_fdr} FDR-significant directed sector flows"
!git commit -m "{commit_msg}"

# Push.
!git push origin main
```

If the push is rejected because `main` is protected, push to a branch
and open a PR instead:

```python
branch = "data/sector-te-k10000"
!git checkout -b {branch}
!git push -u origin {branch}
print(f"Open a PR: https://github.com/EmreAkcaa/elec491-project/compare/{branch}?expand=1")
```

---

## Troubleshooting

**`git push` says `403` or `Authentication failed`**: the token doesn't
have `repo` scope, OR the token-in-URL didn't get carried through. Re-run
Step 1 with a fresh token; confirm the URL printed by `git remote -v`
includes the token.

**Colab session timed out mid-compute**: re-run from Step 1. The compute
is fully deterministic with `seed=42`, so a fresh run produces an
identical parquet. Colab Pro has ~24h continuous-session budget; the
~40-min job fits comfortably.

**`compute_sector_te_matrix` raises ImportError**: you likely skipped
the `sys.path.insert` line in Step 4. Add it; re-run the cell.

**Result schema doesn't match `[source, target, te, p_value,
significant_fdr, significant_uncorrected, n_tickers_source,
n_tickers_target]`**: the upstream pipeline has changed since this
runbook was written. Pause and ping the human — do NOT commit a
schema-broken parquet.

**`n_fdr == 0` even at K=10,000**: the headline case for running this on
Colab was that K=10k should clear FDR. If it doesn't, something
non-trivial has changed upstream (different sector mapping, different
returns universe, etc.). Don't commit; let the human investigate.

**Out of memory** (highly unlikely at 13×13 sector grid): bump Colab to
Pro+ or run with `joblib(n_jobs=4)` to reduce parallel memory pressure.

---

## What you're NOT doing

- Not running the full TE pipeline at K=10k on the 5256-pair ticker
  grid. That's a separate decision and was explicitly skipped (see plan
  file). This runbook is sector-only.
- Not touching `src/`, `app/`, or any test files. Pipeline code is
  unchanged; we're just rerunning the existing function with more
  shuffles.
- Not running other IT stages (MI, regime KL, etc.). Only sector TE.
- Not deleting or replacing any other parquet.

The blast radius is: one parquet update + one docs edit + one commit.
That's it.
