# Information Theory in StoNeCoAl

This document is the single source of truth for what each information-theory
(IT) quantity in the StoNeCoAl pipeline measures, how it's computed, what it
says about Borsa Istanbul as of the latest run, and where the honest limits
of the method lie.

## Why information theory, on top of correlation

Pearson correlation answers one specific question: *how much do two series
move together **linearly**?* That's a useful starting point but it discards
two things we care about:

1. **Non-linear coupling.** Two series can be deterministically related and
   yet have correlation 0 (e.g., `Y = X²` on symmetric `X`). Mutual
   information catches that.
2. **Direction of influence.** Correlation is symmetric — it cannot tell
   you which series drives which. Transfer entropy is the natural
   extension that can.

Information theory's basic move is to measure everything in terms of
**uncertainty reduction** rather than linear co-movement. The unit is the
**bit** (base-2 log) or **nat** (base-e log). 1 bit = the uncertainty of
a fair coin flip. 0 bits = certainty.

## The 7 measures we compute

All seven run as Stage 13 of the pipeline. Outputs land under
`data/<universe>/results/`. The Streamlit dashboard surfaces them in
**Methods Lab → Information Theory** and **Methods Lab → Transfer Entropy**.

### 1. Pairwise mutual information `I(X; Y)`

The amount of overlap between `X` and `Y`, measured in bits:

```
I(X; Y) = H(X) + H(Y) − H(X, Y)
```

Where `H(·)` is the Shannon entropy of a discretised series. We discretise
returns into 4 equal-frequency bins for the IT pipeline (3 for transfer
entropy, see §6). Properties:

- `I ≥ 0` always.
- `I = 0` iff `X ⊥ Y`.
- `I = log₂(k)` for `k` bins iff `X` determines `Y` perfectly.
- **Symmetric**: `I(X; Y) = I(Y; X)`.
- No linearity assumption.

**Artifact**: `mi_matrix.parquet` — symmetric N×N table indexed by ticker,
diagonal = marginal entropy `H(X_i)`, off-diagonal = pairwise MI in bits.

**Estimator bias**: plug-in MI is biased upward by `~(k − 1)² / (2T)` bits
per pair for `k` bins and `T` samples. On BIST (4 bins × 1543 days)
the per-pair bias is `~9 / 3086 ≈ 0.003 bits`, small relative to the
typical MI magnitude.

### 2. Gaussian baseline MI `I_gauss(X; Y; ρ)`

The MI that the same pair would have if `X` and `Y` were *jointly Gaussian*
with Pearson correlation `ρ`. Closed form:

```
I_gauss(ρ) = −½ log(1 − ρ²)  [in nats]
           = −½ log₂(1 − ρ²)  [in bits]
```

**Artifact**: `mi_gaussian_matrix.parquet` — same shape as `mi_matrix`.

The point of computing this is to compare it against the empirical MI.

### 3. Non-linear excess `MI − MI_gauss`

```
nonlinear_excess(X, Y) = I_empirical(X; Y) − I_gauss(ρ(X, Y))
```

Positive excess means there's pairwise dependence the linear model can't
explain. If `Y` really were a linear+Gaussian function of `X`, the
excess would be ~0 (modulo estimator bias).

**Artifacts**:
- `mi_nonlinear_excess.parquet` — full N×N excess matrix.
- `mi_nonlinear_excess_top.csv` — top-14 pairs ranked by excess.

**Current BIST finding (2026-02 run)**: top non-linear pair is
**BRSAN / HEKTS at 0.044 bits**. To calibrate: 1 bit is the maximum
possible coupling under the 4-bin discretisation. 0.044 bits is real
but small — most BIST pair dependence is approximately linear. The
top-10 list skews to small-caps in steel and building materials, which
is where non-linear shocks (illiquidity, sector-specific events) are
expected.

### 4. Effective dimensionality `D_eff`

The participation ratio of the eigenvalues of the correlation matrix:

```
D_eff = (Σ λ_i)² / Σ λ_i²
```

Properties:
- All correlations identical → 1 dominant eigenvalue → `D_eff ≈ 1`.
- All assets independent → equal eigenvalues → `D_eff = N`.
- Empirical financial systems sit between, much closer to 1 than to N.

**Artifact**: scalar `d_eff` in `it_summary.json`.

**Current BIST finding**: `D_eff = 6.3` out of 73 tickers. The 73 BIST
stocks collapse to about 6 effectively-independent dimensions of
variation — a few macro factors (TRY moves, banking sentiment,
commodity cycles, energy chain, conglomerate themes) drive most of the
variance. Matches what's expected of a small-cap-heavy emerging market.

### 5. Joint structure `ΔH = −½ log det Σ`

The non-trivial part of the joint Gaussian entropy of the N-asset system
(the trivial parts are constants in N and in the marginal variances).
Negative `ΔH` ≡ correlation matrix has eigenvalues bunched near zero ≡
the joint distribution is concentrated on a low-dimensional manifold.

```
ΔH = −½ log det Σ   [nats; same sign convention used in app/utils]
```

For numerical stability on near-singular Σ (e.g., post-CAR EEG), we
apply Ledoit–Wolf shrinkage + a small ridge.

**Artifact**: scalar `log_det_term` in `it_summary.json`.

**Current BIST finding**: `ΔH = 21.97 nats`. Big positive value confirms
strong cross-asset structure (high redundancy across the 73 tickers).

### 6. Transfer entropy `TE(X → Y)`

The directional analogue of mutual information: how much does knowing
`X`'s past help predict `Y`'s future, *beyond* what `Y`'s own past
already tells you?

```
TE(X → Y) = H(Y_t | Y_lag) − H(Y_t | Y_lag, X_lag)
          = H(Y_t, Y_lag) − H(Y_t, Y_lag, X_lag) − H(Y_lag) + H(Y_lag, X_lag)
```

Properties:
- `TE ≥ 0` always.
- `TE(X → Y) = 0` iff `X`'s past adds no predictive information.
- **Asymmetric**: `TE(X → Y)` and `TE(Y → X)` can differ.
- Higher-order: depends on the joint distribution of three variables,
  so estimator variance is much worse than for MI.

We compute TE at lag 1 with 3 bins (equal-frequency). Lag and bin count
are knobs in `config/settings.yaml:transfer_entropy`.

**Significance testing**: each TE value is compared against a null
distribution built from `K` surrogate series. The surrogate is a
**circular block bootstrap** of `X` with block length 5 (≈ one trading
week), which preserves `X`'s autocorrelation under H₀ (a naive i.i.d.
permutation destroys autocorrelation and inflates significance — see
`docs/KNOWN_ISSUES.md` M-1, fixed 2026-05-17).

**Multiple-testing correction**: across the `N(N-1)` directed pairs,
we apply Benjamini–Hochberg FDR at α = 0.05. Alternative options
`bonferroni` and `none` are config-selectable.

**Artifacts**:
- `transfer_entropy_raw.parquet` — full pairwise TE values pre-FDR.
- `transfer_entropy_pvalues.parquet` — per-pair p-values from the surrogate test.
- `transfer_entropy_significance.parquet` — boolean mask post-FDR.
- `transfer_entropy_matrix.parquet` — FDR-thresholded matrix.
- `net_transfer_entropy_matrix.parquet` — `net[i, j] = TE(i→j) − TE(j→i)`.
- `te_network_edges.csv` — directed edge list ranked by `|net_te|`.
- `te_node_roles.csv` — per-ticker `te_out`, `te_in`, `net_te_flow`, role ∈ {source, sink}.
- `transfer_entropy_summary.json` — counts and method metadata.

**Resolution-vs-significance trade-off**: with K surrogate shuffles
the minimum achievable p-value is `1 / (K + 1)`. To pass FDR on N(N−1)
simultaneous tests, the smallest BH cutoff is `α / (N(N−1))`. On BIST:
73 tickers → 5,256 directed pairs → smallest BH cutoff ≈ 9.5e-6.

| K (shuffles) | min p-value | BH cutoff achievable? |
|---|---|---|
| 100 | 0.0099 | no (cutoff 1e-5 unreachable) |
| 1,000 | 0.001 | borderline for top edges |
| 10,000 | 1e-4 | yes for top ~5–20 |
| 100,000 | 1e-5 | yes for top ~50+ |

Production K is **1,000** as of 2026-05-19. **On the full 5,256-pair
grid, K=1000 still produces 0 significant edges after BH-FDR** —
670 pass uncorrected α=0.05 but the multiple-testing correction
demands p < 9.5e-6 for the very top edge, and K=1000 floors at
1e-3. This is a fundamental scale issue: surviving FDR on this many
simultaneous tests would require ~K=100,000+ surrogate shuffles
(~30 hours of CPU at current parallelism).

**What does work** — restricting the hypothesis set BEFORE testing.
Gate-1 validation on 10 high-correlation pairs at K=1000 surfaced
3 directional flows surviving FDR on the small batch:
**TUPRS → AYGAZ (p=0.001, refiner drives gas distributor),
KCHOL → AKBNK (p=0.005, holding drives bank subsidiary),
BRSAN → BRYAT (p=0.007, steel maker drives tire maker)** —
all sectorally meaningful. The FDR cutoff for 20 tests
(20-pair × 2 directions) is 0.05/20 = 0.0025, comfortably above
the K=1000 resolution floor.

The takeaway: **for honest TE on this universe, use either a
pre-selected hypothesis set (top-N most-correlated pairs) at K=1000,
or accept the multiple-testing cost and bump K to ~100,000 for the
full grid**. The dashboard surfaces both: the network plot ranks
ALL pairs by raw TE magnitude (no FDR claim made), and the
p-value distribution panel shows the resolution-limited null
graphically.

### 7. Sign-entropy rate `H(sign_t | sign_{t-1})`

A per-ticker check on the weak form of the efficient market hypothesis:
does today's return *direction* tell you anything about tomorrow's?

```
H(sign_t | sign_{t-1}) = H(sign_t, sign_{t-1}) − H(sign_{t-1})
```

The unconditional `H(sign)` is at most 1 bit (a fair coin flip between
up and down). If `H(sign_t | sign_{t-1}) ≈ 1` bit, today's sign tells
you essentially nothing about tomorrow's — the series passes a weak
EMH check.

**Artifact**: `entropy_rate_signs.csv` — one row per ticker with
`entropy_rate_bits`. Aggregate mean lives in `it_summary.json:mean_sign_entropy_rate_bits`.

**Current BIST finding**: mean = **0.9964 bits/day**. The 73-ticker
panel collectively gives away only 0.0036 bits of directional
information per day, which is the textbook weak-EMH-passes result.
The most "directional" stock is **SKBNK** at 0.9875 bits — about a
1.25% deviation from a fair coin flip, far too small to trade after
costs.

### Regime KL divergence (special-cased)

A system-level measure of how much the cross-asset *structure* shifts
between a calm period and a crisis period. We model each window's returns
as a multivariate Gaussian with covariance `Σ_calm` and `Σ_crisis`, then
compute the closed-form KL between them:

```
D_KL(N(0, Σ_calm) ‖ N(0, Σ_crisis)) =
  ½ [ tr(Σ_crisis⁻¹ Σ_calm) − N + log(det Σ_crisis / det Σ_calm) ]
```

Both covariances get Ledoit–Wolf shrinkage to tame singularity in high
dimensions.

**Artifact**: `regime_kl.json` — list of `{label, date, calm_window,
crisis_window, kl_nats, n_tickers}` per crisis event.

**Hardcoded events**: COVID-19 (2020-03-11), Russia-Ukraine
(2022-02-24), Türkiye earthquakes (2023-02-06). 180-day calm window
before, 60-day crisis window starting on the event date.

**Current BIST findings (nats)**:
- Ukraine invasion: **KL = 287 nats** (large structural shift)
- Türkiye earthquakes: **KL = 531 nats** (~2× the Ukraine shift)

Both are very large. The earthquake event reconfigured the BIST
covariance structure roughly twice as drastically as the Ukraine
invasion did.

### 8. Permutation entropy `PE(X)` — added PR #73

A complementary, **binning-free** complexity measure. Take each window of
4 consecutive returns; rank the 4 values to get a permutation (one of 24
possible orderings). Build the histogram over all observed permutations
across the series, take Shannon entropy, normalise by log₂(24).

```
PE_norm(X) = H(ordinal-pattern distribution) / log₂(D!)
```

PE_norm ∈ [0, 1]. 1.0 = patterns uniform (maximally unpredictable).
0 = single pattern dominates (deterministic).

**Why this is complementary to sign-entropy**: sign-entropy coarse-grains
returns to 2 states and looks at lag-1 transitions only. PE catches
**4-bar ordinal patterns** that the sign-only view discards. A series
like `+, ++, ++, +, +, +, ++, …` has high sign-entropy (mostly positive,
lag-1 sign is uninformative) but the ordinal structure can still be
detected by PE if the magnitude trajectory has a repeated shape.

**Artifact**: `data/<universe>/results/permutation_entropy.csv` —
columns `[ticker, permutation_entropy_norm, n_observations]`. Aggregate
mean is in `it_summary.json:mean_permutation_entropy_norm`.

**Current BIST findings**:
- Mean = **0.9978** — BIST is close to ordinally random in aggregate.
- Range = 0.993 (ZOREN — most patterned) → 0.999 (BRSAN — most random).
- Distinct from sign-entropy: only **5 of the top-15 most-predictable tickers overlap** between the two measures. Sign-entropy and PE catch genuinely different structural patterns:
  - Both lists (clearest non-random): **AGHOL, AKFGY, ASELS, CCOLA, VESBE**
  - PE only (multi-bar patterns missed by sign-only view): **AKSA, ASUZU, BRISA, BRYAT, PAPIL, PGSUS, TOASO, TUPRS, YKBNK, ZOREN**

The dashboard renders these side-by-side in the Information Theory
sub-tab; tickers in 2+ of the three top-15 lists get a `★` annotation.

### 9. Predictability stylised facts beyond sign-entropy — added PR #74

Sign-entropy at lag-1 with 2-state coarse-graining is the weakest possible
predictability measure: it ignores return magnitude and only looks back
one day. Three classic financial diagnostics fill the gap:

**Volatility clustering** — autocorrelation of `|returns|`:
```
ACF(|r|, lag=k) = corr(|r_t|, |r_{t-k}|)
```
Bread-and-butter finance: tomorrow's |return| is strongly predictable
from today's, even when tomorrow's *direction* is not. We report at
lags 1, 5, 22 to capture the decay rate.

**Hurst exponent (rescaled-range R/S)**:
```
H = slope of log(R/S) vs log(n)  on log-spaced window sizes
```
Single number per ticker:
- H ≈ 0.5 → random walk
- H > 0.55 → persistent (long-range positive memory, trending)
- H < 0.45 → anti-persistent (mean-reverting)

**Raw return autocorrelation** `ACF(r, lag=k)`: direct test of return
predictability at lags 1, 5, 22. Sign-entropy only captures the SIGN
part of this; magnitude information shows up in non-zero ACF(r) even
when the direction looks coin-flip.

**Artifact**: `data/<universe>/results/predictability_diagnostics.csv`
with columns `[ticker, sign_entropy_bits, acf_returns_lag1,
acf_abs_returns_lag1, acf_abs_returns_lag5, acf_abs_returns_lag22,
hurst_exponent]`. Universe-aggregate fractions in `it_summary.json`:
`frac_tickers_with_volatility_clustering`,
`frac_tickers_persistent_hurst`.

**Current BIST findings (2026-02 run, 73 tickers)**:

| Measure | Value | What it says |
|---|---|---|
| Mean `ACF(|r|, lag=1)` | **0.21** | Strong volatility clustering across the panel |
| Mean `ACF(|r|, lag=5)` | 0.12 | Decays with lag, still positive at one week |
| Mean `ACF(|r|, lag=22)` | 0.06 | Still positive at one month |
| Tickers with `ACF(|r|, lag=1) > 0.20` | **37 / 73 (51%)** | More than half show strong volatility clustering |
| Mean Hurst | **0.60** | Persistent regime across the universe |
| Tickers with Hurst > 0.55 | **64 / 73 (88%)** | Long-range memory is the norm, not the exception |
| Tickers with Hurst in [0.45, 0.55] | 9 / 73 | True random-walk behaviour is rare |
| Tickers with Hurst < 0.45 (mean-reverting) | 0 / 73 | No single ticker mean-reverts at the single-asset level |
| Mean `ACF(r, lag=1)` | 0.045 | Modest but real |
| Tickers with `|ACF(r, lag=1)| > 0.050` (Bartlett 95%) | **32 / 73 (44%)** | Direct lag-1 return autocorrelation is significant on ~half the universe |

**The honest reframe**: the "BIST passes weak EMH" reading from
sign-entropy ≈ 1.0 is incomplete. **Volatility is forecastable, the
long-range regime is persistent (not random walk), and the lag-1 return
autocorrelation is non-trivial on nearly half the universe.** The
direction of any individual day's return is hard to predict — but
"hard to predict directionally" is a much weaker claim than "the
return process is unpredictable." All three diagnostics are surfaced
side-by-side with sign-entropy in the dashboard so the qualification
is visible.

**Most-misleading ticker examples** (sign-entropy ≈ 1, ACF(|r|) > 0.30):
TKNSA, SKBNK, ISGYO, BRSAN, PAPIL, SASA, BTCIM, ASUZU.

## Where each measure surfaces in the dashboard

| Surface | Measures shown |
|---|---|
| **Methods Lab → Information Theory** → KPI strip | D_eff, ΔH, mean sign-entropy rate, ticker count |
| **Methods Lab → Information Theory** → MI panel | MI matrix heatmap (off-diagonal), MI vs Gaussian scatter, top non-linear-excess table |
| **Methods Lab → Information Theory** → rolling panel | D_eff(t), ΔH(t) on dual axes with crisis markers |
| **Methods Lab → Information Theory** → regime panel | KL divergence per documented crisis |
| **Methods Lab → Information Theory** → per-ticker (PR #71) | Most-predictable tickers by sign-entropy + most-non-linearly-coupled tickers by Σ excess |
| **Methods Lab → Transfer Entropy** → KPI strip | sources / sinks / magnitude-ranked edges / FDR-significant count |
| **Methods Lab → Transfer Entropy** → network plot | Top-N directed edges with arrows; node colour = role, size = degree |
| **Methods Lab → Transfer Entropy** → net flow heatmap | `net[i, j] = TE(i→j) − TE(j→i)` |
| **Methods Lab → Transfer Entropy** → p-value panel (PR #71) | Histogram of TE p-values + significant-edges table |
| **Cross-Market** → KPI strip | D_eff comparison BIST vs S&P |

## Honest limitations

- **Estimator bias scales with sample size**. All plug-in estimators
  overestimate H, MI, and TE by an amount proportional to the number
  of histogram cells divided by sample size. We use small bin counts
  (4 for IT, 3 for TE) to keep this under control on 1,543 trading
  days. Going to higher bin counts is tempting but bad — bias blows up.

- **Discretisation matters**. Equal-frequency binning is robust to fat
  tails but throws away magnitude information. KDE estimators would be
  more powerful but much slower and harder to validate.

- **TE assumes lag is known**. We compute at lag 1 only. Slow lead-lag
  relationships (weekly, monthly) are invisible. A lag-sweep extension
  would be a natural future direction.

- **TE doesn't separate direct from indirect influence**. If X drives
  Y and Z drives Y independently, TE(Z → Y) and TE(X → Y) are both
  positive; TE doesn't tell you whether X and Z are correlated. The
  fix is *conditional* transfer entropy `TE(X → Y | Z)`, which we
  don't compute (estimator variance is severe at our sample size).

- **Regime KL uses Gaussian assumption**. The Σ-only KL ignores fat
  tails. Crisis-period returns are heavy-tailed by definition, so the
  Gaussian KL probably underestimates the true distributional shift.
  Computing KL between empirical distributions would require either
  density estimation or non-parametric tests; not implemented.

## Lag-sweep transfer entropy (PR #73)

The production TE pipeline tests directional flow at lag=1 only. Our
pair-trading half-lives are 30–150 days and the crisis windows last 60+
days, so a daily lag is the *fastest* timescale we could plausibly
measure. We extend with a **lag-sweep** at {1, 5, 22} days (daily /
weekly / monthly) on a hand-picked top-correlation hypothesis set
(10 pairs).

**Why a small hypothesis set**: the BH-FDR cutoff on N=5256 directed
pairs is ≈ 9.5e-6, requiring K=100,000+ surrogate shuffles to clear.
On a 20-test batch (10 pairs × 2 directions), the cutoff is α/20 = 0.0025,
comfortably above the K=1000 minimum p-value of 0.001. So pre-selected
sets surface signal that the full-grid test multiple-testing-suppresses.

**Methodology**:
- Same `_te_one_pair` machinery used by the production stage (circular
  block bootstrap, BH-FDR per-lag).
- FDR is applied **independently per lag** — a finding at lag=5 doesn't
  compete with a finding at lag=1.

**Current BIST findings on the top-10 correlation hypothesis set**:

| Lag (days) | FDR survivors | Uncorrected sig at α=0.05 |
|---|---|---|
| 1 | **2** | 5 |
| 5 (weekly) | 0 | 0 |
| 22 (monthly) | 0 | 3 |

**The 2 lag-1 FDR survivors**:
- **KCHOL → AKBNK** (holding → bank subsidiary), p=0.004
- **BRYAT → BRSAN** (tire maker → steel maker), p=0.003

**Headline interpretation**: BIST directional information flow is
**concentrated at the daily scale**. Weekly flow is undetectable
at this sample size and pre-selected set; monthly flow shows 3 marginal
candidates uncorrected but FDR kills them.

**Important alignment note**: the earlier (pre-fix) Gate-1 result
reported TUPRS → AYGAZ as a third FDR survivor at p=0.001. That was a
**date-alignment artifact** — the lag-sweep helper was using
dropna-per-series + tail-alignment, which mispaired observations on
dates with different NaN patterns. After joint-dropna fix the
TUPRS→AYGAZ point estimate drops from 0.041 to 0.008 (uncorrected
p=0.017, no longer surviving FDR). Lesson: validate alignment on
real data with NaN holes before reporting findings.

## Rolling transfer entropy (PR #73)

The full-sample TE answer is "this directional flow exists across
2020–2026." A more honest claim requires **time-localisation**: does the
flow persist throughout the period, or does it concentrate around
specific regimes?

**Methodology**:
- 252-day sliding window, 21-day stride → ~64 windows on BIST.
- K=500 surrogates per window (lower than full-sample K=1000 — at
  smaller N the surrogate distribution converges faster).
- **Hypothesis discipline**: only run on the pairs that already
  surfaced via the full-sample lag-1 test (G1 survivors). Running on
  the full grid is wasteful and exacerbates the multiple-testing problem.

**Current BIST findings** (across 62 windows):

| Pair (direction) | Mean TE | Max TE | # sig (p<0.05) |
|---|---|---|---|
| BRYAT → BRSAN | 0.036 | 0.068 | **14/62** |
| TUPRS → AYGAZ | 0.026 | 0.051 | 8/62 |
| BRSAN → BRYAT (reverse) | 0.029 | 0.057 | 5/62 |
| KCHOL → AKBNK | 0.025 | 0.044 | 2/62 |
| TUPRS ← AYGAZ (reverse) | 0.027 | 0.058 | 6/62 |
| KCHOL ← AKBNK (reverse) | 0.024 | 0.048 | 1/62 |

**BRYAT → BRSAN is the most persistent** directional flow — significant
in 14 of 62 windows (~23% of time). KCHOL → AKBNK is the cleanest
full-sample finding but the time-localisation is weak (only 2/62
windows individually significant), suggesting the directional flow is
spread thinly across the period rather than concentrated.

The chart in the dashboard plots TE(t) for each (pair, direction) with
crisis-date vertical lines (Ukraine 2022-02, Türkiye earthquake
2023-02) for visual comparison with the rolling D_eff panel.

## Bootstrap confidence intervals (PR #73)

Point estimates of MI and TE come without error bars in the
production pipeline. Adding **circular-block-bootstrap CIs**
(block length = 5, K=500 iterations, joint resampling that preserves
the pair structure) gives the obvious thesis-defense answer to "is
this estimate robust or sampling noise?"

**Joint vs source-only bootstrap**: a TE bootstrap that shuffles only
the source series gives the surrogate-null distribution (which the
existing pipeline already uses for p-values). A **joint** bootstrap
that resamples (source, target) with the SAME block index preserves
the directional information and gives a true CI on the point estimate.
PR #73's `bootstrap_te` uses the joint approach.

**Current BIST findings**:

*MI nonlinear excess CIs (top-14 pairs)*:
- **9 of 14** pairs have CIs that exclude zero → robust nonlinear coupling
- 5 of 14 have CIs straddling zero → consistent with sampling noise
- Top pair **BRSAN/HEKTS**: excess = 0.044 bits, CI = [0.019, 0.077] — robust.

*TE CIs (3 pairs from G1 + uncorrected-interesting)*:
- KCHOL → AKBNK: TE = 0.0092, CI = [0.0042, 0.0166] — robust
- BRYAT → BRSAN: TE = 0.0091, CI = [0.0045, 0.0160] — robust
- TUPRS → AYGAZ: TE = 0.0079, CI = [0.0034, 0.0144] — robust (even though uncorrected-only at full-grid significance)

The dashboard renders both CI tables in the IT and TE sub-tabs
respectively, flagging "CI excludes 0?" as a defensibility marker.

## What we don't compute (and could)

- **Conditional MI `I(X; Y | Z)`** — would isolate direct from indirect
  pairwise coupling. We have partial correlation (GLASSO stage); the
  IT analogue is the next step.
- **Conditional transfer entropy `TE(X → Y | Z)`** — isolates direct
  causal-looking flow.
- **Multi-lag TE / lag sweep** — slow lead-lag relationships.
- **Rolling TE** — time-varying directional flow (analogous to the
  rolling D_eff we already plot). Currently snapshot only.
- **Granger causality** — the standard linear baseline TE generalises.
  Useful as a sanity check; not in the pipeline.

## References

- Shannon, C.E. (1948). "A Mathematical Theory of Communication."
  *Bell System Technical Journal*.
- Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*,
  2nd ed., Wiley.
- Schreiber, T. (2000). "Measuring Information Transfer."
  *Physical Review Letters* 85, 461.
- Bossomaier, T., Barnett, L., Harré, M., & Lizier, J.T. (2016).
  *An Introduction to Transfer Entropy*. Springer.
- Politis, D.N. & Romano, J.P. (1992). "A Circular Block-Resampling
  Procedure for Stationary Data." In *Exploring the Limits of Bootstrap*.
- Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery
  Rate: a Practical and Powerful Approach to Multiple Testing."
  *Journal of the Royal Statistical Society B*, 57(1).
- Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance
  Matrix." *Journal of Portfolio Management* 30(4).

## Maintenance / production knobs

All IT knobs live in `config/settings.yaml`:

```yaml
transfer_entropy:
  lag: 1                          # TE lookback step (days)
  n_bins: 3                       # discretisation cells per series
  significance_shuffles: 1000     # surrogate test count; see resolution table above
  significance_level: 0.05        # α for both uncorrected and corrected tests
  surrogate_block_length: 5       # circular bootstrap block (≈ 1 trading week)
  multiple_testing: fdr_bh        # fdr_bh | bonferroni | none
  seed: 42                        # reproducibility
```

Re-run only Stage 13 with:

```bash
uv run python -c "
from src.config import load_config
from src.info_theory import run_info_theory
from src.transfer_entropy import run_transfer_entropy
cfg = load_config()
run_info_theory(cfg)
run_transfer_entropy(cfg)
"
```

On BIST at K=1000 the TE stage takes ~20 min on a modern laptop
(joblib parallel across cores). On S&P (485 tickers) the same stage
takes 5-15× longer; bump K only when necessary.
