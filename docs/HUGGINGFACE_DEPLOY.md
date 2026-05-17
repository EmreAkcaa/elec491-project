# Hugging Face Spaces deploy runbook

This is the secondary deployment target for the StoNeCoAl dashboard (the
primary one is Streamlit Cloud at https://share.streamlit.io/).

**Why HF Spaces:** the free CPU-basic tier ships with **16 GB RAM** (vs Streamlit
Cloud's effective ~500-700 MB after Python + libs). EEG processed parquets
(593,280 × 64 ≈ 300 MB raw, multiplied by JSON serialisation in the dashboard's
cache layer) push Streamlit Cloud past its OOM limit; HF Spaces handles them
comfortably even without the `_downsample_if_oversize` shim.

**Native Git LFS support** is the other big win — HF Spaces' build env has
`git-lfs` preinstalled, so LFS pull happens automatically on every push. No
`packages.txt` workarounds, no LFS-pointer-detection fallbacks needed.

---

## Architecture: Space + companion Dataset

HF Spaces has a **hard 1 GB per-repo storage limit** that can't be raised
even on paid plans. Our 2 EEG processed parquets are 308 MB each (616 MB
combined) which blows past that on its own.

The canonical HF workaround — and what this project does — is to store the
bulk data in a separate **Dataset repo** (which allows 50 GB per file) and
have the Space download it on first launch.

```
   GitHub repo (this one) ─────► HF Space (UI + code, < 100 MB)
                                       │
                                       │ huggingface_hub.snapshot_download() at startup
                                       ▼
                              HF Dataset (EEG bulk parquets, ~620 MB)
```

The Space's `app/dashboard.py:_materialise_eeg_data_if_needed()` runs at
module top, checks if the EEG parquets are on disk, and downloads them
from the Dataset repo if missing. After the first download HF caches them
in `~/.cache/huggingface/` for subsequent reruns.

Local dev is unaffected — the parquets already exist on disk so the
preload is a fast no-op stat check.

## One-time setup

### 1. Create a Hugging Face account

- https://huggingface.co/join (use any GitHub/Google login)

### 2. Create an access token (write scope)

- https://huggingface.co/settings/tokens
- "New token" → **Type: Write** → name it e.g. `elec491-deploy`
- **Copy the token NOW** — you only see it once. Looks like `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx`
- Save it somewhere safe (password manager)

### 3. Create the Space

- https://huggingface.co/new-space
- **Owner:** your HF username (or an org if you have one)
- **Space name:** `stonecoal` (or whatever — must be lowercase, no spaces)
- **License:** MIT (or whatever your repo uses)
- **SDK:** select **Streamlit**
- **Space hardware:** select **CPU basic (16GB RAM, free)** — the default
- **Visibility:** **Public** (required for the free tier; private spaces cost $)
- Click **Create Space**

You now have a Git repo at `https://huggingface.co/spaces/<owner>/<spacename>`.
It's seeded with a hello-world `app.py` you'll overwrite by pushing this repo.

### 4. Create the companion EEG Dataset

- https://huggingface.co/new-dataset
- **Owner:** same HF username as the Space
- **Dataset name:** `stonecoal-eeg` (or whatever — but it must match what's
  in `app/dashboard.py:_materialise_eeg_data_if_needed()` or be overridden
  via the `EEG_DATASET_REPO` env var)
- **License:** MIT
- **Visibility:** **Public** (the Space downloads via the anonymous read
  endpoint; no token needed at runtime if public)
- Click **Create dataset**

The Dataset starts empty. You'll fill it in step 5.

### 5. Upload EEG bulk parquets to the Dataset

Run once from the repo root:

```bash
# Uses HF_DEPLOY_TOKEN env var (or huggingface-cli login fallback) to push
# the local EEG parquets to the Dataset repo. Takes ~2-3 min for the 616 MB.
HF_DEPLOY_TOKEN=hf_xxx \
EEG_DATASET_REPO=FlyingSubmarine33/stonecoal-eeg \
uv run python scripts/upload_eeg_to_hf_dataset.py
```

(Re-run any time the EEG pipeline regenerates fresh parquets that you want
the deployed dashboard to pick up.)

Verify in the browser: https://huggingface.co/datasets/FlyingSubmarine33/stonecoal-eeg
should list `adj_close.parquet`, `log_returns.parquet`, `coverage_report.csv`.

---

## Deploy (automatic, via GitHub Actions — recommended)

This repo ships a GitHub Actions workflow at
`.github/workflows/deploy-hf-spaces.yml` that mirrors `main` → the HF Space
on every push. Set it up once and never re-type the push commands.

### One-time GitHub setup

1. **GitHub repo → Settings → Environments → New environment**
   - Name: `top` (this is what the workflow references)
   - Add secret: **`HF_DEPLOY_TOKEN`** = the `hf_xxx…` Write-scope token from
     step 2 of "One-time setup" above.

2. **GitHub repo → Settings → Secrets and variables → Actions → Variables tab**
   - Add variable: **`HF_SPACE_REPO`** = `<your-hf-username>/<space-name>`
     e.g. `EmreAkcaa/stonecoal`

That's it. Both are configured per-repo; nothing is hard-coded.

### How it runs

- **Auto-deploy** on every push to `main` (after PR merge).
- **Manual trigger** from the GitHub Actions UI (Actions → "Deploy to Hugging
  Face Spaces" → Run workflow). Useful for redeploys without a new commit.
- **Concurrency**: one deploy in flight at a time. Pushes that land while a
  deploy is running queue up.

### What the workflow does

1. Checks out the repo **with LFS objects materialised** (so the EEG parquets
   are real files, not pointers).
2. Re-runs `git lfs install && git lfs pull` for safety.
3. Adds the HF Space as a `git remote` with the token embedded in the URL.
4. `git push hf HEAD:main --force` — pushes the GitHub `main` tree byte-for-
   byte onto the HF Space's `main` branch. The first push is `--force`
   because the HF Space repo is initialised with an unrelated "hello world"
   commit; subsequent pushes are normal fast-forwards (force is harmless).

The HF Space then auto-rebuilds (its own build, separate from GitHub Actions).

### Watch a deploy

After merging a PR, go to the GitHub repo's **Actions** tab → "Deploy to
Hugging Face Spaces" → click the latest run.

The job's **Summary** step prints two links:
- `https://huggingface.co/spaces/<repo>` — the live dashboard
- `https://huggingface.co/spaces/<repo>/logs` — HF's own build log

If a step fails, the **Failure hint** step prints a 4-row troubleshooting
checklist.

---

## Deploy (manual fallback — if Actions are unavailable)

From the repo root (any branch with the HF YAML frontmatter in README.md):

```bash
# 1. Add HF as a remote (one-time; safe to re-run)
git remote add hf https://huggingface.co/spaces/<owner>/<spacename>
# (replace <owner> and <spacename> with your values)

# 2. Push. You'll be asked for credentials:
#    username: <your-hf-username>
#    password: <the hf_xxx token from step 2 above — NOT your account password>
git push hf main

# Or push a specific branch as 'main' on HF:
git push hf deploy/huggingface-spaces:main
```

The first push uploads everything including LFS objects (~600 MB EEG parquets +
~80 MB other artifacts). Expect 2-5 minutes upload time + 2-3 minutes build time
on the HF side.

Subsequent pushes are deltas — fast.

### Avoid re-typing the token every push (optional but recommended)

```bash
# Cache HF credentials in macOS Keychain / Linux libsecret for 1 year
git config --global credential.helper 'osxkeychain'  # macOS
git config --global credential.helper 'cache --timeout=31536000'  # Linux fallback
```

Or pre-encode the token into the remote URL (less secure — only on personal
machines you trust):

```bash
git remote set-url hf https://<owner>:<hf_xxx_token>@huggingface.co/spaces/<owner>/<spacename>
```

---

## Watch the build

- Go to `https://huggingface.co/spaces/<owner>/<spacename>` in a browser.
- Top-right shows the build status: **Building → Running**.
- Click **Logs** (top-right) to watch live build output:
  - `git lfs pull` — should materialise all LFS objects automatically
  - `pip install -r requirements.txt` — installs runtime deps
  - `streamlit run app/dashboard.py` — kicks off the Streamlit server
- First build: ~3-5 min.
- Subsequent builds: ~30-60 s (cached pip wheels).

---

## Verify the deploy

Once status shows **Running**, click the app frame:

- Sidebar shows **Dataset** dropdown with all 3 universes (BIST 100, S&P 500, EEG MI)
- Click EEG MI: dashboard switches; **no OOM crash** (the whole point of moving
  off Streamlit Cloud)
- Click each main tab + EEE sub-tabs: charts render
- Cross-Market page: BIST vs S&P side-by-side (EEG correctly excluded)

If anything crashes, **Logs** tab shows the actual error.

---

## Update the deploy

**If you set up the GitHub Actions workflow** (recommended path above): just
push to GitHub `main` — Actions takes care of mirroring to HF Spaces. Watch
the Actions tab for the deploy run, then HF's own build logs for the
container start.

**If you're using the manual fallback**:

```bash
git push hf main
```

HF auto-detects the new commit and rebuilds. Watch **Logs** for status.

---

## Cost

**Free** for public Spaces on CPU basic hardware (16 GB RAM, 2 vCPU, no GPU).
Hard limits: 50 GB storage, unlimited LFS bandwidth for public spaces.

If you ever want to upgrade for faster rendering or GPU access:

| Tier | Hardware | Cost |
|---|---|---|
| **CPU basic** (current) | 16 GB RAM, 2 vCPU | Free |
| **CPU upgrade** | 32 GB RAM, 8 vCPU | ~$0.03/hr |
| **T4 small (GPU)** | 16 GB RAM, 16 GB GPU | ~$0.40/hr |
| **A10G small** | 24 GB RAM, 24 GB GPU | ~$1.05/hr |

You manage hardware tier from the Space settings page.

---

## Decommissioning Streamlit Cloud (optional)

Once HF Spaces is verified working, you can:

- **Keep both deploys** — HF Spaces as primary, Streamlit Cloud as fallback /
  legacy. Both auto-rebuild on push to GitHub `main`. Useful for portfolio
  variety.
- **Delete the Streamlit Cloud app** — Manage app → Delete. Frees the
  `share.streamlit.io` slot.

This runbook assumes you keep both unless you say otherwise.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Actions workflow fails at "Verify HF_SPACE_REPO is configured" | Repo variable not set | Settings → Secrets and variables → Actions → **Variables** tab → add `HF_SPACE_REPO` = `<user>/<space>` |
| Actions workflow fails at "Push to Hugging Face Space" with auth error | Token missing or wrong scope | Settings → Environments → `top` → confirm `HF_DEPLOY_TOKEN` exists and has **Write** scope (regenerate at https://huggingface.co/settings/tokens) |
| Actions workflow fails at LFS pull | LFS bandwidth quota exhausted (rare on public spaces) | Wait an hour and re-run; or check HF status |
| `Permission denied` on `git push hf` (manual path) | Wrong token, or token not write-scoped | Regenerate token with **Write** type at https://huggingface.co/settings/tokens |
| Build hangs at "Cloning LFS objects..." | Repo's LFS bandwidth exhausted (rare on public spaces) | Wait it out, or check HF status page |
| App starts but EEG tab is missing from the sidebar | `available_universes()` filtered EEG out → either LFS didn't pull or parquets are stubs | Click **Logs**; if you see "EEG bulk parquets are LFS pointer stubs", file a GitHub issue. Workaround: `git lfs pull` locally and re-push. |
| `streamlit: command not found` in build logs | `requirements.txt` malformed | Verify the file parses with `pip install -r requirements.txt --dry-run` locally |
| Generic "App failed to start" with no useful log | `app_file:` in README YAML points at a non-existent path | Verify `app_file: app/dashboard.py` matches the actual file location |

---

## Architecture notes

HF Spaces builds run in an ephemeral Docker container. They:

1. Clone your repo (with `git-lfs` preinstalled — LFS pull happens here).
2. Read `README.md` YAML frontmatter to determine SDK + entrypoint.
3. `pip install -r requirements.txt`.
4. Run `streamlit run <app_file>` with a wrapped server.

Persistent state between sessions lives **only** in your repo. There's no
writeable filesystem outside `/tmp` (which is per-request, not shared). Our
dashboard already reads from baked-in parquets so this is fine.

Session state (`st.session_state`) survives within a single browser session,
gets wiped on rebuild — same as Streamlit Cloud.
