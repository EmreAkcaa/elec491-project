# StoNeCoAl Final Presentation — Slide Deck Outline

**Format:** 13 slides, ~15 minutes total (~1 min per slide + 1 min for Q&A
warm-up). Times are upper bounds; the demo (slide 12) is the longest stop.

**Audience:** A different professor than the supervisor (per Phase 0
clarification), so we lead finance-first and use the IT layer + system
schematic as the substantive answer to "where's the EEE?" rather than as
a philosophical thesis. No cocky claims; honest negatives stay honest.

**Source numbers:** Every figure in the speaker notes is keyed to a live
artifact under `data/<market>/results/` or to `docs/EEG_VERIFICATION.md`.
Regenerate anything by running `uv run python run_pipeline.py` (or
`scripts/run_basis_variants.py`) and re-reading the JSON / parquet.

---

## 1. Title + headline sentence (45 s)

**Title:** *StoNeCoAl — Stock Network Correlation Analysis*
*A reproducible multi-domain signal-processing pipeline applied to
financial and biomedical sensor data.*

**Bottom of slide:** Authors · ELEC 491 · Koç University · 2026

**Speaker note:** "We applied the same 12-stage signal-processing chain
to three sensor streams: BIST 100, S&P 500, and 64-channel motor-imagery
EEG from PhysioNet. The pipeline produces sector-coherent networks on
both markets and anatomically plausible structure on EEG."

---

## 2. Midterm response — what changed since 65/100 (60 s)

**Three bullets, large font:**

1. **Supervisor:** *"not an EEE project, no hardware."* → Our response:
   the sensor + DSP chain *is* the EEE engineering. New system schematic
   (slide 3) makes that explicit.
2. **TA:** *"add an information theory perspective."* → Built a full IT
   layer (MI matrix, D_eff, ΔH, KL divergence). Slide 8.
3. **Three real bugs were silently corrupting the proposal's primary
   validation criterion.** All fixed. Slide 11.

**Speaker note:** "We're not here to defend the midterm score; we're
here to show the response. Concrete and measurable."

---

## 3. System schematic — the chain (90 s)

**Figure:** `docs/figures/system_schematic.svg` full-page.

**Speaker note (pointing at boxes top-to-bottom):**
> "Sensors → acquisition → statistical signal processing → information
> theory → network extraction → optional neuromorphic inference →
> visualisation. Every stage is real EEE: 160 Hz ADC, bandpass + notch
> + common-average reference DSP, random matrix theory, Shannon
> entropy, transfer entropy, spike-coded inference. No hardware
> fakery — the EEG cap is a literal biomedical sensor (PhysioNet
> BCI2000); the exchange tick aggregates are sensor-derived
> economic data."

---

## 4. Methods at a glance — what runs in each stage (60 s)

**Two-column slide.**

| Stage | Method | Purpose |
|---|---|---|
| Acquisition | yfinance / İş Yatırım / MNE | Pull raw price + EEG panels |
| Preprocessing | Coverage filter, anomaly mask, bandpass/notch/CAR | Clean signal |
| Statistical SP | Pearson, RMT, Glasso, wavelet | Linear dependence structure |
| **Information Theory** | **MI, D_eff, ΔH, TE (FDR-corrected), regime KL** | **Non-linear + multivariate summary** |
| Networks | Mantegna distance, MST, Ward clustering | Recovered topology |
| Inference | LIF spiking NN with surrogate gradient | Optional — neuromorphic substrate |

**Speaker note:** "Everything in this table is what an EEE master's
qual would expect. The bolded row is the layer we added in response to
the TA's note."

---

## 5. BIST result — sector recovery (75 s)

**Figure:** MST coloured by sector + the ARI / NMI KPI strip from the
hero panel on Market Overview.

**Numbers:**
- 73 surviving tickers after 90% coverage filter
- 20 Ward clusters
- **ARI = 0.271, NMI = 0.719** vs official BIST sectors
- 6 of 7 major banks (AKBNK, GARAN, YKBNK, VAKBN, HALKB, ISCTR) share
  cluster 1; SKBNK splits off (Şekerbank — much smaller free float)

**Speaker note:** "The proposal's primary validation criterion was
'the MST recovers known sectors.' Before our fix, that check failed —
single-linkage chaining put 45 of 73 tickers in one cluster across 17
sectors. After switching to Ward, the proposal-promised banking
sanity check passes for 6 of 7 banks, conglomerates separate cleanly,
energy / steel / beverages each form their own clusters."

---

## 6. Cross-market BIST vs S&P (90 s)

**Two MSTs side-by-side, both sector-coloured.** Plus a 2-column KPI
strip:

| Metric | BIST | S&P 500 |
|---|---:|---:|
| Surviving tickers | 73 | 485 |
| ARI vs sector | **0.27** | **0.31** |
| NMI vs sector | 0.72 | 0.60 |
| D_eff | 6.30 | 6.56 |
| Avg pairwise ρ | 0.374 | 0.220 |
| Top eigenvalue share | 38.9% | ~18% |
| Türkiye-quake spike (mean abs corr) | **0.44 → 0.66 → 0.58** | 0.44 → 0.35 → 0.31 (flat) |

**Speaker note:** "Same pipeline, two markets. Important contrast: the
2023 Türkiye earthquake is *isolated to BIST* — mean absolute
correlation jumps from 0.44 to 0.66 during the event window. The S&P
shows no response. This is the validation that our toolkit catches
market-specific stress events, not generic noise."

---

## 7. Numéraire experiment — BIST in TRY / USD / Gold (90 s)

**Figure:** the 3-bar eigenvalue-spectrum overlay + KPI strip from the
new Numéraire Sensitivity panel on the Cross-Market page.

**Numbers:**

| Numéraire | D_eff | Avg ρ | Top eig share |
|---|---:|---:|---:|
| TRY | 6.30 | 0.374 | **38.87%** |
| USD | 4.77 | 0.436 | **45.11%** |
| Gold | 3.71 | 0.502 | **51.47%** |

**Speaker note:** "Naïve hypothesis: stripping the TRY leg should
remove a common factor and reduce the top eigenvalue's share. We re-
expressed BIST returns in USD and in gold and ran the full pipeline on
the variants. The hypothesis is *refuted*: top eigenvalue share goes
*up*, not down. Interpretation: TRY volatility is a *dispersion
source* for BIST equities — exporters benefit from TRY weakness while
importers suffer, so removing the currency leg amplifies the residual
global-equity-risk common factor. **This is the kind of empirical
finance question the pipeline lets us actually test.**"

---

## 8. Information Theory layer (75 s)

**Figure:** 4-panel composite from the new IT sub-tab —
KPI strip (D_eff / ΔH / sign-entropy) · MI vs Pearson scatter ·
rolling D_eff(t) · regime KL table.

**Numbers (BIST):**
- D_eff = 6.30 (matches the figure in `docs/TECHNICAL_REPORT.md` §2.1)
- ΔH = 21.97 nats
- Mean sign-entropy rate = **0.996 bits/day** ≈ 1 bit = weak-form-EMH
  fingerprint: tomorrow's direction is independent of today's
- 14 non-linear-excess pairs identified (BRSAN-HEKTS leads at 0.044
  bits above Gaussian baseline)
- Regime KL: Ukraine invasion = 287 nats; Türkiye earthquake = 531
  nats (≈ 1.85× more disruptive)

**Speaker note:** "The TA asked for an information-theory perspective.
This is the substantive answer: every quantity here is in nats or bits
and has a real interpretation. Mutual information catches non-linear
coupling that Pearson misses on 14 BIST pairs. The sign-entropy rate
of ≈ 1 bit/day across all tickers is the canonical weak-form-EMH
fingerprint — knowing today's direction tells you nothing about
tomorrow's."

---

## 9. EEG methodology portability (60 s)

**Figure:** Anatomical MST for EEG + the sanity-check report table from
`docs/EEG_VERIFICATION.md`.

**Numbers:** 3 PASS · 1 PARTIAL · 0 FAIL · 1 SKIPPED on the 5 sanity
checks from `docs/EEG_INTEGRATION.md`:

| # | Check | Status |
|---|---|---|
| 1 | Inter-hemispheric homologous coherence | PASS (Fp1~Fp2 r=0.86, O1~O2 r=0.85) |
| 2 | Motor-imagery desynchronisation | SKIPPED (3.4 GB raw cache deferred) |
| 3 | RMT signal mode | PARTIAL (λ₁=27 vs MP=1; ratio 2.93 vs target 5) |
| 4 | Posterior MST hubs | PASS (Pz, Oz in top-5) |
| 5 | TE(F→C) > TE(C→F) | PASS |

**Speaker note:** "Same 12-stage pipeline, no modifications, applied to
PhysioNet 64-channel EEG. Homologous hemispheric pairs co-cluster,
midline hubs dominate the MST, transfer-entropy directionality matches
the canonical frontal-to-motor information flow. We do *not* claim a
working motor-imagery classifier — that's out of scope. We claim the
toolkit is sensor-agnostic and produces neuroscience-plausible
structure on brain data."

---

## 10. SNN — honest negative result (60 s)

**Figure:** The SNN sub-tab's `st.warning` block screenshot + per-pair
Δ-Sharpe bar chart.

**Numbers:**
- Macro-F1 = 0.660 BIST / 0.625 S&P (random baseline 0.33) — *the
  classification problem is learnable*
- Mean Δ-Sharpe = **−0.27 BIST / −0.84 S&P** — *the trading edge is not*
- Wins 10/20 pairs on BIST (3 with Δ-Sharpe > +1.0:
  EKGYO_HALKB +1.16, SISE_KRDMD +1.12, KCHOL_AKBNK +1.07);
  7/20 on S&P

**Speaker note:** "We trained a spiking neural network on the
dislocation features. The classifier learns the problem (F1 well
above random), but the trading edge doesn't survive. We report this
as an honest negative result — consistent with weak-form efficiency
at daily frequency. We retain the SNN for methodological breadth, not
as an alpha signal."

---

## 11. Engineering rigour — what we fixed (60 s)

**Four bullets:**

1. **Clustering pathology** — switched single-linkage chaining to
   Ward+20. BIST max cluster 45→12; S&P max cluster 446→36; ARI 8×
   higher (BIST) and 44× higher (S&P).
2. **TE multiple-testing** — added BH-FDR correction across the
   N*(N-1) directed pairs. Pre-fix declared 647 BIST edges "significant"
   at p<0.05; post-fix shows 0 survive FDR at the 100-shuffle resolution
   (a methodological finding we report honestly rather than spin).
3. **TE shuffle null** — replaced `np.random.permutation` (destroyed
   source autocorrelation) with a circular-block-bootstrap surrogate
   (block length 5).
4. **SNN doc numbers** — corrected stale Δ-Sharpe and per-pair-win
   counts everywhere they appeared. (`−1.11 / 5-of-20` was an older
   pair selection; current is `−0.27 / 10-of-20` BIST and `−0.84 /
   7-of-20` S&P, per live `snn_metrics.json`.)

**Plus reproducibility numbers:**
- 196 tests, 100% passing
- Single-command runs: `uv run python run_pipeline.py [--config ...]`
- 5 universes registered (BIST, S&P, BIST/USD, BIST/Gold, EEG)
- All commits land on a feature branch; never push to main

---

## 12. Live demo (90 s — longest stop) {#demo}

**Demo script — rehearse exactly these clicks, exactly these
sentences:**

| Step | Action | Sentence |
|---|---|---|
| 1 | Open dashboard, land on Cross-Market | "Here's the central finance question — how does BIST co-movement compare to a developed market." |
| 2 | Point to the sector-purity bar contrast / Türkiye-quake row | "BIST is conglomerate-led and stress-reactive; S&P is sector-coherent and stable." |
| 3 | Scroll to the Numéraire sub-tab | "When we re-express BIST in USD or gold, the top eigenvalue share *rises* — TRY noise was hiding the global-equity common factor." |
| 4 | Switch universe to EEG via sidebar | "Same pipeline on a 64-channel motor-imagery EEG. Anatomical clusters emerge with no code change." |
| 5 | Click EEE Analysis → Information Theory sub-tab | "And here's the IT layer: pairwise MI in bits, effective dimensionality, ΔH, regime KL." |

**Backup:** Pre-recorded demo video on a USB stick. If anything crashes
on the room's machine, switch to video and keep narrating.

---

## 13. Limitations, honest negatives, future work (60 s)

**Three columns:**

| Limitations | Honest negatives | Future work |
|---|---|---|
| Survivorship bias in both market universes | SNN underperforms `|Z|>2` on average | OOS pair-trading backtest with transaction costs |
| 100 surrogate shuffles is a coarse FDR resolution | Wavelet not informative for daily financial data | Motor-imagery decoding on the EEG cap |
| Synthetic Σ samples for Ledoit-Wolf shrinkage in regime KL | Pearson and MI mostly agree (Spearman corr 0.66) — most coupling is linear | Bootstrap MST stability |
| No physical neuromorphic deployment | Numéraire shifts common-mode the wrong way | Real-time / streaming dashboard |

**Speaker note:** "Everything that didn't work is documented — in the
report, in `docs/KNOWN_ISSUES.md`, and in the dashboard captions.
What you see is what reproduces."

---

## Q&A backup slides (only if asked)

- A1: Why Ward + 20 clusters? (Hyperparameter choice via the
  proposal's "banks cluster together" sanity check.)
- A2: Why BH-FDR instead of Bonferroni? (BH controls FDR ≤ α and is
  less conservative; we still report 0 edges survive, so honesty
  isn't at risk.)
- A3: What's the relationship between the IT layer and the rest of
  the pipeline? (IT summarises the joint distribution that the
  network methods extract — they're complementary, not competing.)
- A4: Why XAU/TRY for the gold numéraire (not XAU/USD)? (We compare
  BIST-Turkish-equity returns in three local-currency-equivalent
  numéraires. XAU/USD would compare to US returns, which is what
  S&P already does.)
- A5: What runs on Loihi 2 in practice? (Nothing yet — that's the
  out-of-scope target deployment substrate. The architecture is
  already designed for it: spike encoders, LIF neurons, recurrent
  topology — only the readout layer would need a rate-to-spike
  conversion before deployment.)

---

## Production checklist

- [ ] Build the schematic SVG (`uv run python scripts/draw_system_schematic.py`)
- [ ] Pull the live dashboard screenshots for slides 5, 7, 8, 9, 10
- [ ] Boot the dashboard on the room's machine ≥ 24 h before final
- [ ] Record a 90-second demo video as backup
- [ ] Time the full deck end-to-end at least 5 times
- [ ] Print 5 handout copies of the abstract + 4-row results table
