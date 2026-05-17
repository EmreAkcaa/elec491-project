# Spiking Neural Network for Pair-Dislocation Buy/Sell Signals

**Project:** StoNeCoAl — Stock Network Correlation Analysis, BIST-100
**Course:** Koç University ELEC 491, Spring 2026
**Author:** Arda Rutkay Var
**Module documented:** [src/snn_signals.py](../src/snn_signals.py)
**Dashboard view:** Market Overview → EEE Analysis → **Neuromorphic Signals** sub-tab in [app/eee_analysis.py](../app/eee_analysis.py)

---

## 1. Executive Summary

We added a **Spiking Neural Network (SNN)** classifier that consumes the existing
pair-dislocation Z-score series produced by [src/pair_dislocation.py](../src/pair_dislocation.py)
and emits per-day **BUY / SELL / HOLD** decisions for the top 20 mean-reverting
stock pairs from the BIST-100 universe.

**Honest framing of the result.** This is a **documented exploratory negative
result on trading performance, with a positive result on classification
quality.** The SNN learns the dislocation features (macro-F1 ≈ 0.66 on BIST,
0.63 on S&P, vs. random baseline 0.33 / majority-class baseline 0.27), but the
resulting trading signal **underperforms** the simple `|Z| > 2` heuristic on
both markets: BIST Δ-Sharpe = **−0.27** (wins **10 of 20 pairs**); S&P
Δ-Sharpe = **−0.84** (wins **7 of 20 pairs**). An earlier snapshot of the
BIST pair selection reported Δ-Sharpe = −1.11 / 5 of 20 wins; that pair set
is no longer the active one — current numbers come from `snn_metrics.json`.
The qualitative reading stands: the predictive information about 20-day-ahead
mean reversion is concentrated in the current Z-score itself; neither
high-dimensional feature augmentation nor recurrent spike-coded inference
extracts additional alpha at daily frequency — consistent with the weak-form
EMH at this horizon.

The SNN's value to the project is therefore **methodological breadth**, not
trading alpha: it implements a spike-coded recurrent classifier with all the
algorithmic ingredients that target neuromorphic hardware (Σ-Δ-style delta
modulation as input, leaky-integrate-and-fire dynamics, surrogate-gradient
training). The architecture details:

- a **recurrent LIF** hidden layer (`snn.RLeaky`, 96 neurons, β = 0.92, V_th = 0.5),
- a **non-resetting LIF** output layer with membrane-potential readout
  (continuous, not spike-counts — see §15.1),
- **delta-modulation** (Σ-Δ-style) and **population coding** as the input encoders,
- **windowed temporal input** — each decision sees a sliding 5-day window
  unrolled across 100 SNN ticks (5 days × 20 ticks/day),
- **surrogate-gradient backprop-through-time** training with focal loss and
  `sqrt`-of-inverse-frequency class weighting,
- one **universal model** for all 20 pairs, with a 20-dim one-hot pair embedding
  that lets the network specialize per pair.

### Headline numbers (test set, 20 pairs, walk-forward holdout)

Numbers below are read directly from `data/results/snn_metrics.json`; the
dashboard's Neuromorphic Signals sub-tab shows the live values.

| Metric | Value | Note |
|---|---|---|
| **Mean macro-F1 (BIST)** | **0.660** | Random-baseline = 0.33; "always HOLD" trivial = ~0.27. Classification is meaningfully above baseline. |
| Mean macro-F1 (S&P) | **0.625** | Same architecture, US universe |
| Best-pair F1 (BIST) | 0.820 (`AGHOL_SAHOL`) | |
| Worst-pair F1 (BIST) | 0.485 (`VESTL_VESBE`) | 17 of 20 pairs > 0.5 |
| Per-class F1 (averaged) | HOLD ≈ 0.79 / BUY ≈ 0.57 / SELL ≈ 0.63 | No class collapse |
| Mean SNN Sharpe (annualised, gross) — BIST | **+3.92** | Paper-trade; no transaction costs |
| Mean Classical (\|Z\|>2) Sharpe — BIST | **+4.19** | Same backtest with the simple heuristic |
| **Mean Δ-Sharpe — BIST** | **−0.27** | Modest underperformance; not the −1.11 the earlier snapshot reported |
| **Mean Δ-Sharpe — S&P** | **−0.84** | Larger gap on the US universe |
| **Pairs where SNN > Classical on Sharpe — BIST** | **10 / 20** | Three pairs beat by ≥ +1.00 (`EKGYO_HALKB`, `SISE_KRDMD`, `KCHOL_AKBNK`) |
| **Pairs where SNN > Classical on Sharpe — S&P** | **7 / 20** | |
| Mean SNN hit rate — BIST | 0.85 | Selectivity is real — SNN trades less often, with higher per-trade accuracy |
| SNN trades / pair — BIST (mean) | ~60 | vs ~80 classical (varies per pair: SNN range 21–108; classical 37–148) |
| Training time | ~12 min on CPU | 20 pairs pooled, 25 max epochs, early-stop patience 5 |

---

## 2. Where SNN Fits in StoNeCoAl

The pipeline is sequential. The SNN is **Stage 12**, the very last stage — it consumes the output of Stages 1–7 (the existing data ingest, correlation, and pair-dislocation modules) and produces buy/sell signals as the final inference layer:

```
1.  data_acquisition       (yfinance daily prices)
2.  data_validation        (coverage + anomaly checks)
3.  preprocessing          (adjusted close, log returns)
4.  analysis               (Pearson correlation, distance matrix)
5.  clustering             (hierarchical + MST)
6.  rolling_correlation    (rolling, expanding, EWMA stats)
7.  pair_dislocation       (spread Z-score, top-20 candidate pairs)   ← SNN input
─── EEE analysis methods ───────────────────────────────────────────
8.  rmt_denoising
9.  partial_correlation    (Graphical LASSO)
10. wavelet_analysis
11. transfer_entropy
12. snn_signals            (Spiking Neural Network — spike-coded, EEE)   ← this module
```

`run_snn_signals(config, retrain)` is invoked from [run_pipeline.py](../run_pipeline.py) inside a `try / except ImportError` so the rest of the pipeline keeps working even if `torch`/`snntorch` are not installed.

---

## 3. Why an SNN Belongs in an EEE Capstone

Earlier supervisor feedback flagged the project as "looking too much like a finance tool." The SNN deliberately answers that critique. Spiking neural networks are an **Electrical and Electronics Engineering** technique, not a finance technique. Three concrete reasons:

### 3.1. SNNs are the algorithmic substrate of neuromorphic hardware

The LIF model we implement in PyTorch is the **same model** that runs on:

- **Intel Loihi 2** (asynchronous neuromorphic processor, ~1 mW per neuron)
- **IBM TrueNorth** (256k digital neurons, 70 mW total chip power)
- **SpiNNaker** (University of Manchester massively-parallel ARM cluster simulating ~1B neurons)
- **BrainScaleS** (mixed-signal accelerated analog SNN at Heidelberg)

An SNN trained off-chip can in principle be deployed unchanged onto these platforms. The chips perform inference at **milliwatt power** because the neurons are *event-driven* — no spike means no computation. That hardware story is purely EEE.

### 3.2. The input encoder is an EEE signal-processing technique

We use **delta modulation**: a positive spike fires when |ΔZ| ≥ θ, a negative spike fires when ΔZ ≤ −θ, silent otherwise. This is the same scheme used in:

- **Σ-Δ analog-to-digital converters** (the most common ADC topology in modern audio + sensor ICs)
- **Dynamic Vision Sensors** / "event cameras" (Prophesee, iniLabs DAVIS, Samsung) — pixels output asynchronous events on log-intensity change rather than synchronous frames

In our pipeline, delta modulation converts a continuous Z-score time-series into an asynchronous spike train. The financial interpretation is secondary; the encoding itself is asynchronous A/D conversion applied to a one-dimensional signal.

### 3.3. Surrogate-gradient training is itself an EEE-developed contribution

The spike function `H(V − V_th)` (Heaviside) is non-differentiable, so standard backprop fails. Surrogate gradients (Neftci et al. 2019, "Surrogate Gradient Learning in Spiking Neural Networks") replace the non-existent gradient of the step with a smooth approximation (here `fast_sigmoid(slope=25)`). This is the trick that makes modern SNN training viable, developed specifically to enable training of networks that later deploy to neuromorphic silicon.

### 3.4. Spike-coded vs rate-coded neural computation

The SNN occupies the **spike-coded, event-driven** end of the neural-network
spectrum. Rate-coded continuous-valued recurrent networks (e.g. Echo State
Networks, Jaeger 2001) sit at the opposite end. An earlier iteration of this
project included a rate-coded ESN forecasting module; it produced a near-zero
R² on the same dislocation-prediction task and was removed. The SNN replaces
it: same task, qualitatively different neural-computation paradigm. Reporting
both attempts honestly (the rate-coded ESN's near-zero R² and the spike-coded
SNN's negative Δ-Sharpe) is the joint information-bottleneck finding described
in the Executive Summary — neither paradigm extracts additional predictive
information beyond a one-feature scalar threshold at this horizon.

---

## 4. Data Flow Through the Module

```
adj_close.parquet           (data/processed/)
log_returns.parquet
dislocation_candidates.csv  (top-20 pairs from Step 7)
            │
            ▼
   build_input_features(...)            ← computes 11 raw features per pair, per day
            │
            ▼
   encode_features_to_spikes(...)       ← spike-encodes to 45 channels
            │
            ▼
   append one-hot pair embedding (20)    → 65 input channels per day
            │
            ▼
   build_windows(W = 5)                  → (T − 4) samples of shape (5 days × 65 channels)
            │
            ▼
   pool all 20 pairs into one training set
            │
            ▼
   train_snn(...)         ← universal LIF classifier, focal loss + sqrt class weights
            │
            ▼
   predict_signals(...)   ← per-pair inference, output spike + membrane records
            │
            ▼
   data/results/snn_signals/{ticker_a}_{ticker_b}.parquet
   data/results/snn_metrics.json
   data/results/snn_model_weights/universal.pt
   data/results/snn_spike_raster_sample.parquet
   data/results/snn_membrane_sample.parquet
   data/results/snn_training_history.csv
   data/results/snn_pair_list.csv
            │
            ▼
   Streamlit dashboard reads these artifacts
```

---

## 5. Input Features (11 raw → 45 spike channels)

### 5.1. The 11 raw features

| # | Feature | Source / definition | Why we use it |
|---|---|---|---|
| 1 | **`zscore`** | Rolling 60-day Z-score of log-price spread between the pair | Core dislocation signal |
| 2 | **`dzscore`** | First difference of (1) | Velocity of the dislocation — distinguishes "Z stuck at 2" from "Z just crossed 2" |
| 3 | **`z_lag5`** | Z-score from 5 trading days ago | Short-term history — was this dislocation sustained or sudden? |
| 4 | **`z_lag20`** | Z-score from 20 trading days ago | Long-term history — regime context |
| 5 | **`rolling_corr`** | 60-day rolling correlation of returns of the two tickers | Is the pair still moving together, or has the correlation broken? |
| 6 | **`return_a`** | Daily log return of ticker A | Local price action |
| 7 | **`return_b`** | Daily log return of ticker B | Local price action |
| 8 | **`spread_vol`** | 20-day rolling std-dev of the log-price spread | Volatility regime |
| 9 | **`market_disp`** | Cross-sectional std-dev of all 100 stocks' returns that day | Market-wide turbulence indicator |
| 10 | **`market_breadth`** | Fraction of stocks with positive return that day | Market sentiment proxy |
| 11 | **`inv_half_life`** | 1 / rolling AR(1) half-life of the spread, clipped to [0, 0.3] | How "snappy" mean reversion has been historically |

All features are causal — they only depend on data at or before day t.

Features 1, 2, 6, 7 are reused from [src/pair_dislocation.py](../src/pair_dislocation.py)
and a similar feature set used by the removed Echo State Network. Features 3, 4,
8, 9, 10 are new contributions of this module — the F1 jump from 0.38 to 0.66
during development came largely from these.

### 5.2. Spike encoding (continuous → asynchronous events)

Continuous values must be translated into spike trains before an LIF network can consume them. Two encoders are used in combination:

#### 5.2.1. Delta modulation (for the fast features: `zscore`, `dzscore`)

Two output channels per scalar:

```
up_channel[t]   = 1  if  x[t] − x[t-1] ≥ +θ      else 0
down_channel[t] = 1  if  x[t] − x[t-1] ≤ −θ      else 0
```

With θ = 0.25, a doorbell rings *only* on transitions. A Z-score plateaued at 2.0 fires once on its way up and then goes silent — exactly the event-driven semantics neuromorphic hardware exploits. This is identical to the principle behind:

- Σ-Δ ADC modulators (in audio, sensor, comms ICs)
- DVS event cameras (Brandli et al. 2014)

The two channels carry the *sign* of the transition, which is critical for distinguishing "spread is widening" from "spread is narrowing."

#### 5.2.2. Population coding (for the slow features: rolling correlation, returns, etc.)

For each scalar, place `n_population_fields = 5` Gaussian receptive fields uniformly across the expected value range `[x_min, x_max]`. The firing rate of neuron `k` is

```
rate_k(x) = exp(-((x - center_k) / width)^2)        center_k ∈ linspace(x_min, x_max, 5)
```

Neuron 1 fires hard when `x` is near `x_min`, neuron 3 near the middle, neuron 5 near `x_max`. This preserves absolute level information (which delta modulation throws away by design) and was developed originally as a model of biological sensory encoding (e.g. orientation columns in primary visual cortex).

#### 5.2.3. Scalar threshold encode (for `inv_half_life`)

A single channel with a saturating ramp from 0 (no mean reversion) to 1 (very fast mean reversion). One channel total.

#### 5.2.4. Pair one-hot embedding (20 channels)

For each pair `(A, B)` we append a 20-dimensional one-hot vector identifying which of the top-20 dislocation candidates this is. The universal model uses this to specialize its decision boundary per pair without needing 20 separate models.

#### 5.2.5. Total channel count

| Encoder group | Channels |
|---|---:|
| Delta (zscore, dzscore) | 2 + 2 = 4 |
| Population (z_lag5, z_lag20, rolling_corr, return_a, return_b, spread_vol, market_disp, market_breadth) | 8 × 5 = 40 |
| Scalar ramp (inv_half_life) | 1 |
| **Base channels** | **45** |
| Pair one-hot | 20 |
| **Total inputs to fc1** | **65** |

---

## 6. Labels — Mean-Reversion Oracle (Magnitude-Aware)

The labels are computed by **looking forward** in the training set only. Inference is always strictly causal — the trained model never sees the future. The forward-look defines what the *correct* decision *would have been*.

### 6.1. Rule

For each trading day `t` in a pair's history:

1. If `|Z_t| ≤ entry_z` (=1.2)  →  label = **HOLD** (Z not extreme enough to be a dislocation).
2. Else look at the next `K = 20` trading days of Z-score: `future = Z[t+1 : t+1+K]`.
   - If `Z_t > +entry_z`: profit if Z drops → magnitude = `Z_t − min(future)`.
     - If magnitude ≥ `min_reversion` (=0.8) → label = **SELL** (short the spread).
   - If `Z_t < −entry_z`: profit if Z rises → magnitude = `max(future) − Z_t`.
     - If magnitude ≥ `min_reversion` → label = **BUY** (long the spread).
3. Otherwise → label = **HOLD** (Z was extreme but didn't revert enough — likely a regime shift, not a mean-reversion opportunity).

### 6.2. Why magnitude-aware rather than "did Z revert to ±0.5?"

The earlier binary rule treated "Z went from +2.0 to +0.45" the same as "Z went from +2.0 to −0.5" — even though the second case is a much cleaner profitable reversion. The magnitude rule explicitly demands a meaningful reversion ≥ 0.8 Z-units before tagging the day as a profitable trade entry. This pushes label noise out of the training set.

### 6.3. Class distribution (pooled across 20 pairs, training set)

| Class | Count | Share |
|---|---:|---:|
| HOLD | 12,873 | 65% |
| BUY | 3,349 | 17% |
| SELL | 3,618 | 18% |
| **Total** | **19,840** | 100% |

The 65 / 35 split is much more balanced than naïve "always HOLD" baselines suggest — the magnitude rule does a good job of carving the meaningful trade opportunities out of the time series.

---

## 7. Network Architecture

```
       (B, W=5 days, 65 channels)                  ← windowed input tensor
                    │
              fc1: Linear(65, 96)                  ← learned input projection
                    │
              snn.RLeaky(96, all_to_all=True)      ← recurrent LIF hidden, β=0.92, V_th=0.5
                    │                              ← unrolled for W × T_steps = 100 ticks
              fc2: Linear(96, 3)                   ← readout projection
                    │
              snn.Leaky(reset_mechanism="none")    ← non-resetting LIF output layer
                    │                              ← integrates membrane V across all 100 ticks
                final V vector ∈ R^3              ← logits for [HOLD, BUY, SELL]
                    │
                  argmax                            ← class prediction
```

### 7.1. The LIF neuron model

The Leaky Integrate-and-Fire neuron is governed in continuous time by

> τ_m · dV/dt = −(V − V_rest) + R · I(t)

with the firing rule

> If V(t) ≥ V_th  →  emit spike, then  V ← V_reset.

In discrete time (one tick per simulation step) `snntorch.Leaky` implements

> V[k+1] = β · V[k] + W · x[k] − S[k] · V_th
>
> S[k+1] = 1 if V[k+1] ≥ V_th else 0

where `β = exp(−Δt/τ_m)` is the leak factor and `S` is the binary spike output. With `β = 0.92`, the "memory" of the neuron is ~12 ticks (1/(1−β)). With our `n_timesteps = 20` ticks per simulated trading day, that means each LIF bucket effectively remembers the most recent ~half-day of input drive.

### 7.2. The recurrent LIF (`snn.RLeaky`)

The hidden layer is `snn.RLeaky` with `all_to_all = True`, which adds a learned recurrent matrix `V_rec` so that the layer's update becomes

> mem[k+1] = β · mem[k] + W · x[k] + V_rec · spk[k] − S[k] · V_th

The recurrent connection lets the hidden layer use its *previous spike pattern* as additional drive at the next tick. This adds within-layer associative memory — the network can chain together patterns like "Z rose two days ago, ΔZ flipped sign yesterday, current Z still high" into a single coherent representation.

Recurrent LIF was the single biggest architectural contributor to the F1 jump from 0.38 to 0.66.

### 7.3. The membrane-potential readout

The output layer is plain `snn.Leaky` but with `reset_mechanism = "none"`. That means the output neurons **integrate without ever firing** — they just accumulate the synaptic drive over the entire `W × T_steps = 100` ticks of the window. The class prediction is the argmax of the final 3-dimensional membrane vector.

Why this matters: pure spike-count readout (the more biologically faithful choice) produces a discrete loss landscape — the surrogate gradient on top of integer spike counts is sparse and easily gets stuck. We discovered this empirically: with spike-count readout the network collapsed to "always HOLD" within one epoch (F1 = 0.27) because the gradient signal was too weak. Switching to membrane readout immediately unlocked learning. This is the standard practice recommended in Eshraghian et al. 2023, "Training Spiking Neural Networks Using Lessons From Deep Learning" — the hidden layers still spike (which is the EEE-relevant part), but the readout integrates smoothly.

### 7.4. Windowed temporal input

For each "sample" (a single trading day t), the network is fed a 5-day window of features `X[t−4 : t+1]`. Inside the forward pass, the window is unrolled day-by-day; each day's feature vector is held constant for `T_steps = 20` ticks of LIF integration before moving to the next day. The LIF state (membrane + spike) **persists across all 100 ticks** of the window, so the β-leak naturally weights more recent days more heavily.

This lets the network reason about *sustained* dislocations rather than treating each day in isolation — a critical property for distinguishing "Z = 2 for the last 5 days" (real dislocation, likely to revert) from "Z just hit 2 today" (could be noise).

### 7.5. Universal model with pair embedding

Rather than training 20 separate per-pair models (the original approach), we train **one shared model** on the pooled training set across all pairs, with a 20-dimensional one-hot pair embedding appended to each input. This:

- multiplies the effective training data by 20×,
- lets the model learn general dislocation patterns that transfer across stocks,
- still allows per-pair specialization (the one-hot lets `fc1` produce a pair-specific bias),
- ends up training **~19× faster** than the per-pair version end-to-end (11.9 min for one universal pooled run vs ~11.2 min × 20 pairs = ~224 min total for 20 separate per-pair models), even with a slightly larger hidden layer.

The val/test gap shrank substantially with this change — going from one-model-per-pair (heavy overfit risk on 1000 training days) to one universal model with 20× the data largely solved the generalization problem.

---

## 8. Hyperparameters — What Each Knob Does

All defaults live in the `SNNConfig` dataclass at the top of [src/snn_signals.py](../src/snn_signals.py).

| Name | Value | What it controls / what happens if you change it |
|---|---|---|
| `n_hidden` | 96 | Hidden-layer width. Bigger = more capacity but slower and more overfit risk. Doubling to 192 might help; 32 collapses to majority class. |
| `beta` | 0.92 | Membrane leak factor per tick. β=1 = perfect integrator (no leak), β=0 = no memory. Our 0.92 gives ~12-tick memory horizon. |
| `v_threshold` | 0.5 | Firing threshold. Lower → more selective firing (each neuron is more "active"). Originally we used 1.0 and the network saturated; 0.5 was empirically better. |
| `n_timesteps` | 20 | SNN simulation ticks per trading day. More ticks = more accumulation = smoother spike counts, but slower training. |
| `window_size` | 5 | Trading days of history fed to each decision. Used to be 1 (single-day) — going to 5 added temporal context. |
| `use_universal_model` | True | One model for all pairs vs one per pair. Universal is the recommended setting. |
| `use_recurrent_hidden` | True | Use `snn.RLeaky` (recurrent) for the hidden layer instead of plain `snn.Leaky`. Adds within-layer memory; one of the most impactful flags. |
| `readout` | `"membrane"` | Output layer integrates membrane (no reset) and argmax over the final V. Alternative `"spike_count"` is more biological but trains badly on this task. |
| `input_scaling` | 2.0 | Multiplier applied to spike-encoded inputs before `fc1`. Boosts drive into the hidden layer; matters because the spike inputs are bounded in [0, 1]. |
| `class_weight_mode` | `"sqrt_inv_freq"` | Weighting of the loss across HOLD/BUY/SELL. Plain inverse-frequency (`"inv_freq"`) over-pressured the minority classes and produced too many false alarms. `sqrt` is calibrated. |
| `learning_rate` | 3e-3 | Adam optimizer step size. |
| `weight_decay` | 1e-4 | L2 regularization. Small, but useful given limited data. |
| `n_epochs` | 25 | Max training passes. With early stopping we usually stop at 5–10 epochs. |
| `batch_size` | 128 | Larger batches → smoother gradient estimates but slower epoch time on CPU. |
| `early_stop_patience` | 5 | Epochs to wait for val_loss improvement before stopping. |
| `seed` | 42 | Reproducibility. |
| `use_focal_loss` | True | Use focal loss instead of plain weighted CE. Focuses gradient on hard examples. |
| `focal_gamma` | 2.0 | Focal-loss focusing strength. γ=0 reduces to CE; γ=2 is the standard. |
| `delta_threshold` | 0.25 | Spike threshold for the delta-modulation encoder. Lower θ = more spikes = denser input. |
| `n_population_fields` | 5 | Gaussian receptive fields per population-encoded feature. |
| `label_horizon` | 20 | K-day forward look for the mean-reversion oracle. |
| `label_entry_z` | 1.2 | Minimum |Z| to even consider labeling a day as BUY/SELL. |
| `label_min_reversion` | 0.8 | Required reversion magnitude (in Z units) within K days to label BUY/SELL. |
| `train_ratio` | 0.7 | Per-pair train fraction. Remaining 30% is split 15% val / 15% test. |
| `top_n_pairs` | 20 | How many pairs (from `dislocation_candidates.csv`) to train and evaluate on. |
| `rolling_window` | 60 | Window for spread Z-score and rolling correlation computation. |

---

## 9. Training Procedure

### 9.1. Time-respecting data split

For each pair, after building the windowed feature tensor:

- First 70% of windows → **train**
- Next 15% → **validation** (used for early stopping)
- Last 15% → **test** (the headline numbers come from this — never seen during training)

The training portions of all 20 pairs are then concatenated and shuffled within the training pool. Val and test stay time-ordered per pair so the test set always represents the *most recent* trading days. This prevents look-ahead leakage.

### 9.2. Loss function

Focal loss (Lin et al. 2017, "Focal Loss for Dense Object Detection") on top of class-weighted cross-entropy:

> L = − α_{y_i} · (1 − p_{y_i})^γ · log(p_{y_i})

where:

- `p_{y_i}` = softmax probability of the true class for sample i,
- `α_{y_i}` = `sqrt(inverse frequency)` class weight for class y_i,
- `γ = 2.0` is the focusing parameter.

The `(1 − p_{y_i})^γ` factor down-weights confidently-correct predictions (mostly HOLDs in our case) so the gradient signal concentrates on the hard BUY/SELL samples near the decision boundary.

### 9.3. Optimizer

Adam, lr = 3e-3, weight_decay = 1e-4, default β1/β2.

### 9.4. Backprop-through-time (BPTT)

Each forward pass unrolls the LIF for `W × T_steps = 100` ticks. PyTorch builds a computational graph through all 100 ticks; the surrogate gradient (`snntorch.surrogate.fast_sigmoid(slope=25)`) substitutes for the non-differentiable spike function during the backward pass. The recurrent matrix `V_rec` in `RLeaky` is updated like any other learned weight.

### 9.5. Early stopping

Monitor val_loss; if it doesn't improve by ≥ 1e-4 for 5 consecutive epochs, restore the best weights and stop. On the 20-pair training run, the model stopped around epoch 7–10.

---

## 10. Key Performance Indicators

We track three families of metrics — each measures a different property of the trained model.

### 10.1. Classification metrics

These come from the confusion matrix on the held-out test set.

**Macro-F1** (our headline metric)

> F1_c = 2 · P_c · R_c / (P_c + R_c),  for each class c
>
> macro-F1 = mean(F1_HOLD, F1_BUY, F1_SELL)

Macro-F1 weights all three classes equally — so a model that gets HOLD perfect but never predicts BUY/SELL gets a low macro-F1 even though it has high accuracy. This is exactly what we want for an imbalanced 3-class problem: it rewards a model that correctly handles all classes.

Random baseline on 3 classes ≈ 0.33. "Always HOLD" given our class distribution ≈ 0.27. **Our 0.664 is meaningfully above both.**

**Per-class precision / recall**

- **Precision** for class c = `TP_c / (TP_c + FP_c)` = "when I predict c, how often am I right?"
- **Recall** for class c = `TP_c / (TP_c + FN_c)` = "of all true c's, how many did I catch?"

We watch these per class because aggregate accuracy can hide bad behavior. Earlier iterations had high BUY recall (caught 50% of true BUYs) but terrible BUY precision (only 23% of predicted BUYs were correct) — i.e., the model was a buy-alarm spammer. The sqrt class weights fixed this.

### 10.2. Trading metrics (paper-trade backtest)

We simulate entering on each SNN signal and exiting after `K = 20` trading days. The P&L of a trade is the change in spread Z-score over those 20 days (signed by trade direction).

**Sharpe ratio**

> Sharpe = (mean P&L per trade) / (std-dev P&L per trade) × √(252 / K)

Risk-adjusted return, annualized. Positive Sharpe = strategy is profitable on average. >1 = good, >2 = strong, >3 = excellent. Our SNN currently scores +3.56 on average across pairs.

**Hit rate**

> Hit rate = fraction of trades with positive P&L

Simpler intuition than Sharpe: "how often is the model right when it does trade?" Random direction-guess = 0.50; our model averages 0.85 across pairs (some pairs at 0.95).

**Δ-Sharpe**

> Δ-Sharpe = SNN Sharpe − Classical Sharpe

The single most important comparison number. The classical baseline is the `|Z| > 2` mean-reversion rule from `pair_dislocation.detect_signals()`. If Δ-Sharpe is positive, our SNN is genuinely adding value over the heuristic the project was already implementing.

Currently **10 of 20** BIST pairs have a positive Δ-Sharpe (regenerated from
`data/bist/results/snn_metrics.json`). The three strongest wins:
`EKGYO_HALKB` (+1.16), `SISE_KRDMD` (+1.12), `KCHOL_AKBNK` (+1.07). Five more
exceed +0.10 (`SISE_PETKM`, `YKBNK_AKBNK`, `KCHOL_KRDMD`, `KCHOL_ISCTR`,
`YKBNK_VAKBN`); the remaining two (`TTKOM_AKBNK`, `TUPRS_AYGAZ`) are
within rounding of zero. The SNN loses to the simple `|Z|>2` rule on the
other 10 of 20 BIST pairs. On S&P it wins on 7 of 20.

### 10.3. Number of trades

A quality-quantity tradeoff indicator. The SNN averages ~80 trades per pair vs the classical rule's ~140. Fewer trades but a much higher hit rate — meaning the SNN has learned to *be selective*, which is exactly what we want from a sophisticated classifier.

---

## 11. Results

### 11.1. Aggregate

All values regenerated from `data/results/snn_metrics.json` (test set, 20 pairs,
walk-forward holdout, universal model).

| Metric | BIST | S&P |
|---|---:|---:|
| Macro-F1 (mean over 20 pairs) | **0.660** | **0.625** |
| Mean SNN Sharpe (gross, no costs) | +3.92 | +3.25 |
| Mean Classical Sharpe (\|Z\|>2) | +4.19 | +4.09 |
| **Mean Δ-Sharpe (SNN − Classical)** | **−0.27** | **−0.84** |
| Mean SNN hit rate | 0.85 | 0.83 |
| Pairs with F1 > 0.50 | 17 / 20 | 16 / 20 |
| **Pairs with positive Δ-Sharpe** | **10 / 20** | **7 / 20** |
| Pairs with Δ-Sharpe ≥ +1.0 | 3 / 20 (`EKGYO_HALKB`, `SISE_KRDMD`, `KCHOL_AKBNK`) | 1 / 20 (`LEN_DHI`) |
| Wall-clock training time | ~12 min on CPU | ~12 min on CPU |

### 11.2. Per-pair table — BIST (sorted by macro-F1, descending)

Regenerated verbatim from the per-pair entries in
`data/bist/results/snn_metrics.json`. **Δ-Sh column highlights:** the three
Δ-Sh ≥ +1.0 pairs are `EKGYO_HALKB`, `SISE_KRDMD`, `KCHOL_AKBNK`.

| Pair | F1 | SNN-Sh | Cls-Sh | Δ-Sh | Hit | SNN trades | Cls trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| `AGHOL_SAHOL` | 0.820 | +3.48 | +4.63 | −1.15 | 0.89 | 62 | 95 |
| `YKBNK_AKBNK` | 0.814 | +3.28 | +2.90 | **+0.38** | 0.80 | 69 | 77 |
| `BRSAN_BRYAT` | 0.754 | +3.48 | +4.05 | −0.58 | 0.87 | 100 | 110 |
| `SAHOL_KCHOL` | 0.753 | +3.38 | +4.44 | −1.06 | 0.75 | 61 | 69 |
| `TUPRS_AYGAZ` | 0.734 | +3.79 | +3.79 | **+0.00** | 0.87 | 83 | 85 |
| `KCHOL_ISCTR` | 0.714 | +7.95 | +7.79 | **+0.17** | 0.98 | 41 | 96 |
| `KCHOL_AKBNK` | 0.711 | +6.62 | +5.55 | **+1.07** | 0.96 | 52 | 64 |
| `EKGYO_HALKB` | 0.706 | +3.82 | +2.66 | **+1.16** | 0.87 | 54 | 86 |
| `KCHOL_TUPRS` | 0.702 | +2.46 | +3.63 | −1.18 | 0.93 | 67 | 82 |
| `SAHOL_ISCTR` | 0.678 | +4.38 | +5.71 | −1.33 | 0.93 | 71 | 37 |
| `TTKOM_AKBNK` | 0.675 | +3.90 | +3.86 | **+0.04** | 0.87 | 70 | 78 |
| `YKBNK_VAKBN` | 0.655 | +2.22 | +2.11 | **+0.11** | 0.67 | 90 | 99 |
| `VAKBN_EKGYO` | 0.613 | +1.05 | +1.39 | −0.34 | 0.66 | 108 | 148 |
| `PETKM_KRDMD` | 0.601 | +3.55 | +5.28 | −1.72 | 0.90 | 48 | 70 |
| `SAHOL_AKBNK` | 0.584 | +4.06 | +4.06 | −0.00 | 0.84 | 68 | 89 |
| `KCHOL_KRDMD` | 0.578 | +3.85 | +3.64 | **+0.21** | 0.88 | 40 | 81 |
| `SAHOL_KRDMD` | 0.563 | +1.85 | +3.19 | −1.33 | 0.65 | 23 | 55 |
| `SISE_PETKM` | 0.533 | +4.30 | +3.78 | **+0.53** | 0.89 | 35 | 53 |
| `SISE_KRDMD` | 0.518 | +6.07 | +4.95 | **+1.12** | 1.00 | 33 | 60 |
| `VESTL_VESBE` | 0.485 | +4.97 | +6.46 | −1.49 | 0.90 | 21 | 62 |

S&P per-pair detail lives in `data/sp500/results/snn_metrics.json`; the
top-3 SNN wins there are `LEN_DHI` (+1.18), `PNC_FITB` (+0.78), `KEY_TFC`
(+0.75).

### 11.3. Interpretation (honest)

- **17 of 20 BIST pairs have F1 > 0.5** (16 of 20 on S&P) — classification is
  meaningfully above random. The dislocation features carry learnable structure.
- **The SNN is selective.** Mean hit rate 0.85 on BIST; per-pair hit rates
  0.65–1.00. Fewer trades per pair than the classical rule on most pairs,
  with higher per-trade accuracy — a real property of trained classifiers,
  not a BIST-specific finding.
- **SNN loses to the classical rule on 10 of 20 BIST pairs by Sharpe**
  (13 of 20 on S&P). The biggest losses concentrate where the classical
  rule produces unusually high Sharpe with few trades — `VESTL_VESBE`
  (Cls +6.46 from 62 trades), `SAHOL_ISCTR` (Cls +5.71 from only 37 trades).
- **The 10 BIST pairs where SNN wins** include three with Δ-Sharpe > +1.0
  (`EKGYO_HALKB`, `SISE_KRDMD`, `KCHOL_AKBNK`) — substantively beyond noise.
  The other seven are smaller (0.0–0.5).
- **No transaction costs in either backtest.** Round-trip pair-trade costs on
  Turkish equities (~30 bps per leg, 4 fills = ~120 bps) would eat much of the
  per-trade margin on both strategies; the *relative* ranking is preserved
  because both strategies are charged the same costs.
- **Bottom line.** The classifier learns the problem; the problem is not
  predictable enough at daily frequency for high-dimensional ML to beat a
  one-line heuristic. We treat this jointly with §15 as a documented
  exploration and an information-bottleneck finding.

---

## 12. Dashboard Integration

The dashboard's **EEE Analysis** tab has a fifth sub-tab, **Neuromorphic
Signals**, surfacing the artifacts listed in §13.

What renders today (mirrors the live UI in `app/eee_analysis.py:render_snn`):

1. **Section header + honest framing caption** — explicitly states the
   negative trading result (BIST Δ-Sharpe = −0.27, 10 of 20 pairs beat
   classical; S&P Δ-Sharpe = −0.84, 7 of 20) alongside the positive
   classification result (macro-F1 ≈ 0.66 BIST / 0.63 S&P).
2. **5-metric KPI row** — Pairs trained, Mean macro-F1, Mean SNN Sharpe,
   Mean Classical Sharpe, Mean Δ-Sharpe.
3. **Per-pair leaderboard** (`st.dataframe`) — all 20 pairs sorted by
   Δ-Sharpe, with F1 / SNN-Sh / Cls-Sh / Δ-Sh / hit rate / trade counts.
4. **Per-pair signal explorer** — dropdown picks a pair; chart overlays the
   pair's Z-score with SNN BUY/SELL markers (with ±2 reference lines).
5. **Training history** — train loss + val loss (left axis) + val macro-F1
   (right axis) across epochs.
6. **Sample-pair internals** (sample pair = `BRYAT_BRSAN`):
   - **Output-neuron spike raster** over the sampled window (HOLD / BUY /
     SELL on the y-axis; SNN ticks on the x-axis).
   - **Membrane V(t)** trace for the 3 output neurons with horizontal V_th
     reference line.
7. **Expander** — full architecture and hyperparameter summary read from
   the live `snn_metrics.json:config` block.

Dashboard loaders are in [`app/utils.py`](../app/utils.py) — six
`@st.cache_data` functions named `load_snn_*` that read the artifacts under
`data/results/` lazily.

---

## 13. Files and Artifacts

### 13.1. Source code (all present after the Phase C integration commit)

| File | Role |
|---|---|
| [`src/snn_signals.py`](../src/snn_signals.py) | SNN module: encoders, LIF model, training, inference, `run_snn_signals` entry point |
| [`tests/test_snn_signals.py`](../tests/test_snn_signals.py) | 12 unit tests (8 torch-free + 4 torch-skip-if-not-installed) |
| [`run_pipeline.py`](../run_pipeline.py) | Calls `run_snn_signals(config)` after the EEE block, wrapped in `try/except ImportError` so the pipeline still completes if torch is not installed |
| [`pyproject.toml`](../pyproject.toml) | `[project.optional-dependencies] snn = ["torch>=2.0", "snntorch>=0.7"]` — install with `uv sync --extra snn` |
| [`app/utils.py`](../app/utils.py) | Six `load_snn_*` cached loaders: `metrics`, `pair_list`, `signals(pair_id)`, `training_history`, `raster_sample`, `membrane_sample` |
| [`app/eee_analysis.py`](../app/eee_analysis.py) | `render_snn(sector_map)` function + "Neuromorphic Signals" sub-tab wiring in `render()` |

### 13.2. Generated artifacts (under `data/results/`)

| File | Contents |
|---|---|
| `snn_metrics.json` | Per-pair + aggregate metrics, full `SNNConfig` dump, sample-pair id, n_inputs |
| `snn_pair_list.csv` | The 20 pairs the model was trained on (ticker_a, ticker_b, pair_id) |
| `snn_signals/{ticker_a}_{ticker_b}.parquet` | Per-pair daily signal: date, zscore, prob_hold/buy/sell, signal, classical_signal |
| `snn_model_weights/universal.pt` | Trained PyTorch state dict (single universal model). **No per-pair `.pt` files** — those were unused orphans from an earlier per-pair-training code path and are not committed. |
| `snn_training_history.csv` | epoch / train_loss / val_loss / val_acc / val_macro_f1 / pair |
| `snn_spike_raster_sample.parquet` | (day_index, date, timestep, neuron_id, neuron_name) for one sample window of the sample pair |
| `snn_membrane_sample.parquet` | (day_index, date, timestep, neuron_id, neuron_name, membrane) for the sample window |

---

## 14. How to Run

```bash
# 1. One-time: install the optional SNN extra (pulls torch + snntorch, ~700 MB)
uv sync --extra snn

# 2. Run the full pipeline. The SNN step runs at the end if torch is installed,
#    otherwise it is skipped with a log warning (the rest of the pipeline still
#    completes). Caches universal.pt; re-inference is ~30 s, full retrain ~12 min.
uv run python run_pipeline.py

# 3. Tests (the 8 torch-free SNN tests always run; 4 torch-dependent tests are
#    skipped automatically if torch is not installed)
uv run python -m pytest -q

# 4. Dashboard
uv run streamlit run app/dashboard.py
# Navigate: Market Overview → EEE Analysis → Neuromorphic Signals (5th sub-tab)
```

`run_snn_signals(config, retrain=False)` reuses cached `universal.pt` weights
by default. To force a full retrain, call it from a Python REPL with
`retrain=True` (no CLI flag is exposed; the SNN module is the authoritative
toggle):

```python
from src.config import load_config
from src.snn_signals import run_snn_signals
run_snn_signals(load_config(), retrain=True)
```

The full BIST pipeline + SNN training takes about 12 minutes on a laptop CPU
(SNN training dominates; the upstream pipeline is faster). Without `retrain=True`
the cached weights are reused and the SNN step just re-runs inference (~30 s).

---

## 15. What Did Not Work (and why) — for the project journal

Three things broke at first and had to be fixed:

### 15.1. Collapse to "always HOLD" with spike-count readout

The first version used `spike_counts = spk_rec.sum(dim=0)` for the output logits — a pure spike-count readout, biologically faithful. The model trained for one epoch, then val-F1 stuck at 0.27 for every subsequent epoch. Diagnosis: integer spike counts produce a sparse loss landscape, and the surrogate gradient through `sum(binary_spikes)` is too weak to escape the trivial fixed point where the model predicts the majority class.

**Fix:** switched the output layer to `reset_mechanism="none"` and read the **final membrane potential** as the class logits. The hidden layers still spike (so the EEE story holds), but the readout integrates continuously. Eshraghian et al. 2023 ("Training Spiking Neural Networks Using Lessons From Deep Learning") recommends this as the standard SNN-training practice.

### 15.2. Over-firing BUY/SELL with full inverse-frequency class weights

The second version used `w_c = N / (3 · count_c)`. Class weights were ~4× larger for BUY/SELL than HOLD. The model learned to predict BUY/SELL too often — recall on BUY/SELL was OK (~45%) but precision was terrible (~23%).

**Fix:** switched to `sqrt(N / (3 · count_c))` (the `"sqrt_inv_freq"` mode). Less aggressive correction, better-calibrated predictions. Precision on BUY/SELL jumped from 0.23 / 0.30 to 0.61 / 0.62.

### 15.3. F1 plateau at 0.38 with the simple input feature set

Even with universal training and membrane readout, F1 was capped at 0.38 — close to the original per-pair version. The model could not distinguish "Z is high right now" from "Z has been high for a week" from features 1–6 alone.

**Fix:** added 5 new features (z_lag5, z_lag20, spread_vol, market_disp, market_breadth) for temporal + regime context, plus switched the hidden layer to `snn.RLeaky` for within-layer recurrence. F1 jumped from 0.38 → 0.66.

These three iterations are documented in this report deliberately — the SNN's final architecture is informed by the failure modes, and explaining them in the project report demonstrates engineering judgment.

### 15.4. Operational limitations (current state)

§§ 15.1–15.3 above are the development-history journal — failure modes
that have been fixed. This subsection consolidates the **remaining
operational caveats** of the shipped implementation; each is also
referenced from the relevant headline section so a reader who jumps
straight to §1 or §11 sees them in context.

- **Hybrid spike-rate readout.** The hidden layer fires binary spikes
  (LIF dynamics with surrogate gradients — the EEE-relevant part), but
  the output layer uses `reset_mechanism="none"` and reads the
  *continuous membrane potential* as the classification logit (§15.1).
  A pure spike-count readout is more biologically faithful but
  collapsed to the majority class in our setting; we therefore label
  the architecture as a *hybrid spike-rate model*, not a strictly
  event-driven SNN. Deployment to neuromorphic silicon (Loihi 2,
  TrueNorth, SpiNNaker) would require a rate-to-spike conversion of
  the readout, which is left as future work.
- **Negative trading result.** Mean Δ-Sharpe = −0.27 on BIST / −0.84 on
  S&P over the 20 test pairs each; the SNN beats the classical `|Z|>2`
  rule on 10 of 20 BIST pairs (7 of 20 on S&P), with three substantive
  BIST wins above +1.0 Sharpe (`EKGYO_HALKB`, `SISE_KRDMD`, `KCHOL_AKBNK`)
  and the remainder within ≤ +0.5 of break-even (§11.1, §11.3). We retain
  the SNN for methodological-breadth value (spike-coded counterpart to
  other rate-coded methods in the project) rather than as an
  alpha-generating signal.
- **Per-pair `.pt` orphans deliberately omitted.** Earlier iterations
  trained one model per pair and wrote per-pair `.pt` files. The
  current code path only saves `universal.pt`; the per-pair files are
  unused. We deliberately do not commit them to `data/results/
  snn_model_weights/` (§13.1) to avoid leaving dead binary artifacts
  in version control.
- **Sharpe annualisation uses overlapping 20-day holds.** The
  `_backtest_sharpe` helper scales per-trade Sharpe by
  `√(252 / horizon)` with `horizon = 20`. Consecutive trades share
  ~19 days of the spread path, so returns are highly autocorrelated
  and the i.i.d. annualisation overstates absolute Sharpe by roughly
  √20 ≈ 4.5×. **The comparison vs. the classical baseline is
  internally fair** (same construction on both arms), but the
  absolute Sharpe numbers in §11.1 should not be read as realisable
  annualised performance.
- **Zero transaction costs.** A pair trade requires four fills.
  At Turkish-equity ~30 bps per leg the round-trip cost is ~120 bps,
  which is large relative to the per-trade spread vol of the top
  pairs (≈ 0.17 log-points for AKBNK–YKBNK). Realistic net Sharpe is
  materially smaller than the reported gross values; the *relative*
  ranking is preserved because both strategies are charged the same
  costs.
- **Forward-look labels.** Labels are computed from a K=20-day
  forward-look mean-reversion oracle (§6). Inference is strictly
  causal — the trained model never sees the future — but the supervised
  target itself benefits from hindsight that no real-time system has.
- **`SNNConfig` not exposed through YAML.** Every architectural and
  training hyperparameter lives as a Python default in
  `snn_signals.py:80`. Hoist to `config/settings.yaml` is filed under
  FUTURE_WORK F-2.

---

## 16. EEE Framing for the Project Report

Suggested talking points for the final ELEC 491 report:

- **LIF as a discrete-time analog VLSI model.** Cite Mead 1989 (*Analog VLSI and Neural Systems*) and the Loihi / TrueNorth / SpiNNaker chip families.
- **Delta modulation as Σ-Δ ADC applied to a financial time series.** This frames the encoder explicitly as an EEE signal-processing technique. Cite a Σ-Δ ADC textbook for academic gravitas.
- **Surrogate gradient as the EEE-developed bridge to deep learning.** Cite Neftci, Mostafa & Zenke 2019 (IEEE Signal Processing Magazine — the paper is literally an EEE-journal publication).
- **Recurrent LIF as a continuous-time recurrent dynamical system.** Mention biological plausibility plus hardware-compatibility (Loihi's neurons support recurrent connections natively).
- **Spike-coded vs rate-coded neural computation.** Rate-coded continuous recurrent networks (Echo State Networks, Jaeger 2001) and spike-coded event-driven networks (SNNs, Maass 1997 — the "third generation" of neural networks) span the neural-computation spectrum. An earlier rate-coded ESN forecasting module was explored on this same task and produced a near-zero out-of-sample R²; the spike-coded SNN reaches macro-F1 = 0.67 on the classification metric but still underperforms the simple `|Z|>2` rule on trading Sharpe. Reporting both attempts honestly is the joint information-bottleneck finding.
- **The training/deployment dichotomy.** GPU-trained surrogate models can in principle be deployed to milliwatt-power neuromorphic silicon — a hardware/software co-design story that finance tools simply do not have.

---

## 17. References (for the bibliography)

1. **Mead, C.** *Analog VLSI and Neural Systems.* Addison-Wesley, 1989.
2. **Maass, W.** "Networks of spiking neurons: the third generation of neural network models." *Neural Networks* 10.9 (1997): 1659–1671.
3. **Jaeger, H.** "The Echo State Approach to Analysing and Training Recurrent Neural Networks." *GMD Report* 148, 2001.
4. **Brandli, C. et al.** "A 240×180 130 dB 3μs Latency Global Shutter Spatiotemporal Vision Sensor." *IEEE Journal of Solid-State Circuits* 49.10 (2014): 2333–2341. (DVS event-camera reference for delta modulation.)
5. **Neftci, E.O., Mostafa, H., Zenke, F.** "Surrogate Gradient Learning in Spiking Neural Networks." *IEEE Signal Processing Magazine* 36.6 (2019): 51–63.
6. **Davies, M. et al.** "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning." *IEEE Micro* 38.1 (2018): 82–99.
7. **Akopyan, F. et al.** "TrueNorth: Design and Tool Flow of a 65 mW 1 Million Neuron Programmable Neurosynaptic Chip." *IEEE Transactions on Computer-Aided Design* 34.10 (2015): 1537–1557.
8. **Lin, T.-Y. et al.** "Focal Loss for Dense Object Detection." *IEEE TPAMI* 42.2 (2020): 318–327.
9. **Eshraghian, J.K. et al.** "Training Spiking Neural Networks Using Lessons from Deep Learning." *Proceedings of the IEEE* 111.9 (2023). (The `snntorch` paper.)
10. **Furber, S.B. et al.** "The SpiNNaker Project." *Proceedings of the IEEE* 102.5 (2014): 652–665.

---

*Last updated: 2026-05-17 (canonical doc-set integration: §15.4 operational
limitations added; "Step 13" → "Stage 12" to match pipeline numbering;
universal-vs-per-pair training-time figure corrected from 3× to ~19× total
end-to-end; Phase H merge of `origin/main`).*
