"""Spiking Neural Network buy/sell signals for dislocation pairs.

A compact Leaky-Integrate-and-Fire (LIF) classifier trained with surrogate-gradient
backprop-through-time to predict mean-reversion BUY / SELL / HOLD decisions on
the spread Z-score of historically correlated stock pairs.

Why SNNs for stock signals (EEE framing)
────────────────────────────────────────
- LIF neurons are the algorithmic substrate of neuromorphic hardware
  (Intel Loihi, IBM TrueNorth, SpiNNaker) — milliwatt-power event-driven inference.
- Delta-modulation input encoding mirrors Σ-Δ ADCs and DVS event cameras:
  a spike fires only when |Δsignal| crosses a threshold.  No event, no compute.
- Surrogate-gradient training (Neftci et al. 2019) supplies the differentiable
  approximation that lets us train these otherwise non-differentiable networks
  end-to-end with PyTorch autograd.
- Sits on the spike-coded, event-driven end of the neural-computation spectrum
  (vs. rate-coded continuous recurrence, e.g. Echo State Networks — explored
  in an earlier iteration of this project and removed because the dispersion-
  forecasting R^2 was near zero on the same task).

This module is intentionally light on financial machinery: the inputs come from
the existing pair-dislocation analysis (Z-score, ΔZ, rolling correlation,
log returns, half-life proxy) and the labels come from a mean-reversion oracle
defined over future Z-score trajectory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PipelineConfig, PROJECT_ROOT
from src.pair_dislocation import (
    compute_spread,
    compute_zscore,
    compute_half_life,
    detect_signals,
)

logger = logging.getLogger(__name__)

# Path constants are derived per-call from the active `config` (config.data_*)
# so that BIST, S&P, EEG universes can coexist on disk without collisions.

# Class encoding: order matters for label arrays and confusion matrix axes
HOLD, BUY, SELL = 0, 1, 2
CLASS_NAMES = ["HOLD", "BUY", "SELL"]


# ── Lazy torch / snntorch import ────────────────────────────────────────


def _require_torch():
    """Import torch + snntorch on demand; raise informative ImportError if missing."""
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        import snntorch as snn  # noqa: F401
        import snntorch.functional as SF  # noqa: F401
        import snntorch.surrogate as surrogate  # noqa: F401
        return torch, nn, snn, SF, surrogate
    except ImportError as e:
        raise ImportError(
            "SNN signals require torch + snntorch. "
            'Install with `uv sync --extra snn` (or `pip install ".[snn]"`).'
        ) from e


# ── Configuration ───────────────────────────────────────────────────────


@dataclass
class SNNConfig:
    """Spiking Neural Network hyperparameters."""

    # Architecture
    n_hidden: int = 96
    beta: float = 0.92               # LIF leak rate
    v_threshold: float = 0.5          # lower → more selective firing, less saturation
    n_timesteps: int = 20            # SNN ticks per trading day (per day inside the window)
    window_size: int = 5             # # consecutive trading days fed per decision
    use_universal_model: bool = True # one model across all pairs (with pair-id one-hot)
    use_recurrent_hidden: bool = True # snn.RLeaky in hidden layer (within-layer recurrence)
    readout: str = "membrane"        # "membrane" (use final V) or "spike_count"
    input_scaling: float = 2.0        # multiplier applied to spike-encoded inputs before fc1
    class_weight_mode: str = "sqrt_inv_freq"  # "inv_freq" | "sqrt_inv_freq" | "none"

    # Training
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    n_epochs: int = 25
    batch_size: int = 128
    early_stop_patience: int = 5
    seed: int = 42
    use_focal_loss: bool = True
    focal_gamma: float = 2.0

    # Encoders
    delta_threshold: float = 0.25
    n_population_fields: int = 5

    # Labels (magnitude-aware mean-reversion oracle)
    label_horizon: int = 20            # K trading days forward look
    label_entry_z: float = 1.2         # lowered → more BUY/SELL examples
    label_min_reversion: float = 0.8   # required |Z| reversion magnitude toward 0
    label_exit_z: float = 0.5          # kept for backward compatibility

    # Scope
    train_ratio: float = 0.7           # final train/test split
    top_n_pairs: int = 20              # pairs trained from dislocation_candidates.csv
    rolling_window: int = 60           # window for spread Z-score and rolling corr
    retrain: bool = False

    # Sample artifact pair (used for spike-raster + V(t) plots in dashboard).
    # Empty → first pair in candidates is used.
    sample_pair: tuple[str, str] = field(default_factory=tuple)


# ── Encoders (pure numpy — torch-free) ─────────────────────────────────


def delta_encode(x: np.ndarray, theta: float) -> np.ndarray:
    """Delta-modulation encoder: 2 spike channels per scalar input.

    Channel 0 fires (=1) when x_t - x_{t-1} >= +theta, else 0.
    Channel 1 fires when x_t - x_{t-1} <= -theta, else 0.

    Mirrors Σ-Δ ADC / DVS-camera asynchronous A/D conversion.

    Parameters
    ----------
    x : ndarray of shape (T,)
    theta : float, positive crossing threshold

    Returns
    -------
    spikes : ndarray of shape (T, 2)  — first row is all zeros (no Δ at t=0)
    """
    x = np.asarray(x, dtype=np.float32)
    dx = np.diff(x, prepend=x[:1])
    up = (dx >= theta).astype(np.float32)
    down = (dx <= -theta).astype(np.float32)
    return np.stack([up, down], axis=-1)


def population_encode(
    x: np.ndarray,
    n_fields: int,
    x_min: float,
    x_max: float,
    width: Optional[float] = None,
) -> np.ndarray:
    """Population encoder: Gaussian-receptive-field firing probability per neuron.

    Each of `n_fields` neurons has a Gaussian tuning curve centered uniformly
    across [x_min, x_max].  Output is a deterministic firing rate in [0, 1]
    that the SNN reads as a sub-threshold drive (no Bernoulli sampling — we
    inject the rate directly as a continuous spike-amplitude input, which the
    LIF integrates over `n_timesteps`).

    Returns
    -------
    rates : ndarray of shape (T, n_fields), values in [0, 1]
    """
    x = np.asarray(x, dtype=np.float32)
    centers = np.linspace(x_min, x_max, n_fields, dtype=np.float32)
    if width is None:
        width = float((x_max - x_min) / max(1, n_fields - 1))
    if width <= 0:
        width = 1.0
    # Gaussian receptive field: exp(-((x - c)/width)^2)
    diff = x[:, None] - centers[None, :]
    rates = np.exp(-((diff / width) ** 2))
    return rates.astype(np.float32)


def _scalar_threshold_encode(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Single-channel saturating ramp: 0 below `low`, 1 above `high`, linear between.

    Used for the half-life proxy (1/HL) which is always non-negative.
    """
    x = np.asarray(x, dtype=np.float32)
    if high <= low:
        return (x >= high).astype(np.float32)[:, None]
    rate = np.clip((x - low) / (high - low), 0.0, 1.0)
    return rate[:, None].astype(np.float32)


def build_input_features(
    adj_close: pd.DataFrame,
    log_returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int = 60,
) -> pd.DataFrame:
    """Compute the continuous feature matrix for one pair, ready for spike encoding.

    Columns (11 raw features):
        zscore         — rolling Z-score of log-price spread
        dzscore        — first difference of zscore
        z_lag5         — Z-score 5 trading days ago     (autoregressive history)
        z_lag20        — Z-score 20 trading days ago    (autoregressive history)
        rolling_corr   — rolling correlation of log returns
        return_a       — log return ticker A
        return_b       — log return ticker B
        spread_vol     — 20-day rolling std of spread   (volatility regime)
        market_disp    — cross-sectional std of all returns that day  (regime)
        market_breadth — fraction of stocks with positive return      (sentiment)
        inv_half_life  — 1 / rolling half-life (clipped to [0, 0.3])

    All NaN rows are dropped.  Index = trading date.
    """
    spread, _, _ = compute_spread(adj_close, ticker_a, ticker_b)
    zscore = compute_zscore(spread, window=window)
    dzscore = zscore.diff()
    z_lag5 = zscore.shift(5)
    z_lag20 = zscore.shift(20)

    ra = log_returns[ticker_a]
    rb = log_returns[ticker_b]
    rolling_corr = ra.rolling(window, min_periods=max(10, window // 2)).corr(rb)
    spread_vol = spread.rolling(20, min_periods=10).std()

    # Market-wide cross-sectional regime features (pandas skips NaN tickers)
    market_disp = log_returns.std(axis=1)
    market_breadth = (log_returns > 0).sum(axis=1) / log_returns.notna().sum(axis=1)

    # Rolling half-life: estimated on a trailing window so it is causal
    hl_series = pd.Series(index=spread.index, dtype=float)
    hl_window = max(120, window * 2)
    for i in range(hl_window, len(spread)):
        hl_series.iloc[i] = compute_half_life(spread.iloc[i - hl_window : i])
    inv_hl = (1.0 / hl_series).clip(lower=0.0, upper=0.3)

    feats = pd.DataFrame({
        "zscore": zscore,
        "dzscore": dzscore,
        "z_lag5": z_lag5,
        "z_lag20": z_lag20,
        "rolling_corr": rolling_corr,
        "return_a": ra,
        "return_b": rb,
        "spread_vol": spread_vol,
        "market_disp": market_disp,
        "market_breadth": market_breadth,
        "inv_half_life": inv_hl,
    })
    return feats.dropna()


def encode_features_to_spikes(features: pd.DataFrame, cfg: SNNConfig) -> np.ndarray:
    """Convert continuous features → fixed-rate spike-amplitude tensor.

    Output shape: (T_days, n_channels=45)  — replicated across n_timesteps
    in the model forward pass.

    Channels:
        zscore         → delta             (2)
        dzscore        → delta             (2)
        z_lag5         → population        (5)
        z_lag20        → population        (5)
        rolling_corr   → population        (5)
        return_a       → population        (5)
        return_b       → population        (5)
        spread_vol     → population        (5)
        market_disp    → population        (5)
        market_breadth → population        (5)
        inv_half_life  → saturating ramp   (1)
    """
    n_pop = cfg.n_population_fields
    parts = [
        delta_encode(features["zscore"].values, cfg.delta_threshold),
        delta_encode(features["dzscore"].values, cfg.delta_threshold),
        population_encode(features["z_lag5"].values, n_pop, x_min=-3.0, x_max=3.0),
        population_encode(features["z_lag20"].values, n_pop, x_min=-3.0, x_max=3.0),
        population_encode(features["rolling_corr"].values, n_pop, x_min=-1.0, x_max=1.0),
        population_encode(features["return_a"].values, n_pop, x_min=-0.05, x_max=0.05),
        population_encode(features["return_b"].values, n_pop, x_min=-0.05, x_max=0.05),
        population_encode(features["spread_vol"].values, n_pop, x_min=0.0, x_max=0.1),
        population_encode(features["market_disp"].values, n_pop, x_min=0.0, x_max=0.05),
        population_encode(features["market_breadth"].values, n_pop, x_min=0.0, x_max=1.0),
        _scalar_threshold_encode(features["inv_half_life"].values, low=0.0, high=0.2),
    ]
    return np.concatenate(parts, axis=-1).astype(np.float32)


# ── Mean-reversion oracle labels ────────────────────────────────────────


def generate_mean_reversion_labels(zscore: pd.Series, cfg: SNNConfig) -> pd.Series:
    """Label each day as HOLD / BUY / SELL by magnitude of forward Z-reversion.

    Uses future data — VALID ONLY AT TRAINING TIME.  Inference is strictly causal.

    Rule (magnitude-aware, K-day forward look):
        Z_t > +entry_z and Z drops by at least `min_reversion` within K days →
            SELL  (short the spread, profit from Z falling)
        Z_t < -entry_z and Z rises by at least `min_reversion` within K days →
            BUY   (long the spread, profit from Z rising)
        otherwise → HOLD

    The magnitude check beats the old "reverted to ±exit_z?" binary test because
    it preserves information about how much profit was available — and it
    excludes "barely-touched-the-threshold" examples that produced label noise.
    """
    z = zscore.dropna().values.astype(np.float64)
    n = len(z)
    K = int(cfg.label_horizon)
    labels = np.zeros(n, dtype=np.int64)  # HOLD = 0

    for t in range(n - 1):
        z_t = z[t]
        if abs(z_t) <= cfg.label_entry_z:
            continue
        future = z[t + 1 : min(t + 1 + K, n)]
        if future.size == 0:
            continue
        if z_t > 0:
            # Want Z to drop. Best-case profit = z_t - min(future).
            profit = float(z_t - future.min())
            if profit >= cfg.label_min_reversion:
                labels[t] = SELL
        else:
            # Want Z to rise.  Best-case profit = max(future) - z_t.
            profit = float(future.max() - z_t)
            if profit >= cfg.label_min_reversion:
                labels[t] = BUY

    return pd.Series(labels, index=zscore.dropna().index, name="label")


def build_windows(X: np.ndarray, W: int) -> np.ndarray:
    """Build sliding windows of shape (T - W + 1, W, n_features) from (T, n_features).

    Each window[i] = X[i : i + W], aligned so that window[i] ends at row i + W - 1.
    """
    T = len(X)
    if T < W:
        return np.zeros((0, W, X.shape[1]), dtype=X.dtype)
    return np.stack([X[t - W + 1 : t + 1] for t in range(W - 1, T)], axis=0)


# ── Model ───────────────────────────────────────────────────────────────


def build_lif_classifier(n_inputs: int, cfg: SNNConfig):
    """Construct a 2-layer LIF classifier with windowed temporal input.

    Forward pass accepts (batch, window, n_inputs) and unrolls the LIF buckets
    for `window * n_timesteps` ticks total, with state persisting across days
    inside the window.  This lets β-decay carry sustained-extreme information
    forward across multiple trading days — the temporal context the per-day
    version lacked.

    Returns the model (subclass of nn.Module) — lazy-imports torch.
    """
    torch, nn, snn, _, surrogate = _require_torch()
    spike_grad = surrogate.fast_sigmoid(slope=25)

    class LIFClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(n_inputs, cfg.n_hidden)
            self.use_recurrent = cfg.use_recurrent_hidden
            if self.use_recurrent:
                self.lif1 = snn.RLeaky(
                    beta=cfg.beta, threshold=cfg.v_threshold,
                    spike_grad=spike_grad,
                    linear_features=cfg.n_hidden,
                    all_to_all=True,
                    init_hidden=False,
                )
            else:
                self.lif1 = snn.Leaky(
                    beta=cfg.beta, threshold=cfg.v_threshold,
                    spike_grad=spike_grad, init_hidden=False,
                )
            self.fc2 = nn.Linear(cfg.n_hidden, 3)
            self.lif2 = snn.Leaky(
                beta=cfg.beta, threshold=cfg.v_threshold,
                spike_grad=spike_grad, init_hidden=False,
                # Output layer integrates without reset → smooth readout signal.
                reset_mechanism="none",
            )
            self.n_timesteps = cfg.n_timesteps
            self.window_size = cfg.window_size
            self.input_scaling = cfg.input_scaling
            self.readout = cfg.readout

        def forward(self, x):
            """x: (B, window, n_inputs).

            Returns
            -------
            spk_rec : (window * n_timesteps, B, 3) output spikes
            mem_rec : (window * n_timesteps, B, 3) output membrane potentials
            """
            if x.dim() == 2:
                x = x.unsqueeze(1)
            B, W, _ = x.shape
            if self.use_recurrent:
                spk1, mem1 = self.lif1.init_rleaky()
            else:
                mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            spk_rec, mem_rec = [], []
            for d in range(W):
                x_d = x[:, d, :] * self.input_scaling   # (B, n_inputs)
                for _ in range(self.n_timesteps):
                    cur1 = self.fc1(x_d)
                    if self.use_recurrent:
                        spk1, mem1 = self.lif1(cur1, spk1, mem1)
                    else:
                        spk1, mem1 = self.lif1(cur1, mem1)
                    cur2 = self.fc2(spk1)
                    spk2, mem2 = self.lif2(cur2, mem2)
                    spk_rec.append(spk2)
                    mem_rec.append(mem2)
            return torch.stack(spk_rec), torch.stack(mem_rec)

        def readout_logits(self, spk_rec, mem_rec):
            """Combine the recorded outputs into class logits.

            "spike_count" → sum of binary spikes (discrete, noisy gradient)
            "membrane"    → final membrane potential (smooth, dense gradient)
            """
            if self.readout == "membrane":
                # Final-tick membrane (output layer has reset_mechanism="none")
                return mem_rec[-1]
            return spk_rec.sum(dim=0)

    return LIFClassifier()


def focal_loss_on_spike_counts(spike_counts, targets, alpha_weights, gamma: float = 2.0):
    """Focal loss applied to output spike-count logits.

    L = -alpha_t * (1 - p_t)^gamma * log(p_t),   p_t = softmax(spike_counts)[true_class]

    Class weights `alpha_weights` handle imbalance; the (1-p_t)^gamma factor
    down-weights easy examples (correct, high-confidence predictions) so the
    gradient focuses on the hard BUY/SELL examples we actually care about.
    """
    torch, *_ = _require_torch()
    import torch.nn.functional as F
    log_probs = F.log_softmax(spike_counts, dim=1)
    probs = log_probs.exp()
    target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal = (1.0 - target_probs).clamp(min=1e-8) ** gamma
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    if alpha_weights is not None:
        nll = nll * alpha_weights[targets]
    return (focal * nll).mean()


# ── Training / inference ────────────────────────────────────────────────


def _class_weights(labels: np.ndarray, mode: str = "sqrt_inv_freq"):
    """Class weights from label distribution.

    "inv_freq"       — full inverse-frequency (aggressive — over-pushes minorities)
    "sqrt_inv_freq"  — sqrt of inv-freq (calibrated; recommended default)
    "none"           — uniform weights
    """
    torch, *_ = _require_torch()
    counts = np.array([(labels == c).sum() for c in range(3)], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = counts.sum() / (3.0 * counts)
    if mode == "none":
        weights = np.ones_like(inv_freq)
    elif mode == "inv_freq":
        weights = inv_freq
    else:
        weights = np.sqrt(inv_freq)
    return torch.tensor(weights, dtype=torch.float32)


def train_snn(X_train, y_train, X_val, y_val, cfg: SNNConfig, model=None):
    """Train an SNN classifier on windowed numpy arrays.

    Parameters
    ----------
    X_train, X_val : ndarray (N, window, n_inputs)
    y_train, y_val : ndarray (N,) ∈ {0, 1, 2}
    model : pre-built nn.Module or None (built from X_train shape if None)

    Returns
    -------
    model : trained nn.Module
    history : list of {epoch, train_loss, val_loss, val_acc, val_macro_f1}
    """
    torch, nn, *_ = _require_torch()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if model is None:
        n_inputs = X_train.shape[-1]
        model = build_lif_classifier(n_inputs, cfg)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    class_w = _class_weights(y_train, mode=cfg.class_weight_mode)
    ce_fn = nn.CrossEntropyLoss(weight=class_w)

    def compute_loss(spike_counts, targets):
        if cfg.use_focal_loss:
            return focal_loss_on_spike_counts(
                spike_counts, targets, class_w, gamma=cfg.focal_gamma,
            )
        return ce_fn(spike_counts, targets)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_v = torch.tensor(X_val, dtype=torch.float32) if len(X_val) else None
    y_v = torch.tensor(y_val, dtype=torch.long) if len(y_val) else None

    n = len(X_tr)
    history = []
    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(cfg.n_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            optimiser.zero_grad()
            spk_rec, mem_rec = model(xb)
            logits = model.readout_logits(spk_rec, mem_rec)
            loss = compute_loss(logits, yb)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(1, n_batches)

        val_loss = float("nan")
        val_acc = float("nan")
        val_f1 = float("nan")
        if X_v is not None and len(X_v) > 0:
            model.eval()
            with torch.no_grad():
                # Chunked forward pass to bound memory
                preds_list = []
                losses = []
                for start in range(0, len(X_v), cfg.batch_size):
                    xv, yv = X_v[start : start + cfg.batch_size], y_v[start : start + cfg.batch_size]
                    spk_rec, mem_rec = model(xv)
                    logits = model.readout_logits(spk_rec, mem_rec)
                    losses.append(float(compute_loss(logits, yv).item()) * len(xv))
                    preds_list.append(logits.argmax(dim=1).cpu().numpy())
                val_loss = sum(losses) / len(X_v)
                pred = np.concatenate(preds_list)
                y_v_np = y_v.cpu().numpy()
                val_acc = float((pred == y_v_np).mean())
                val_f1 = float(_macro_f1(_confusion_matrix(y_v_np, pred))["macro_f1"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        })
        logger.info(
            "      epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_F1=%.3f",
            epoch, cfg.n_epochs, train_loss, val_loss, val_f1,
        )

        if not np.isnan(val_loss) and val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience and X_v is not None and len(X_v) > 0:
                logger.info("    Early stop at epoch %d (best val_loss=%.4f)", epoch, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_signals(model, X: np.ndarray, cfg: SNNConfig, batch_size: int = 256) -> dict:
    """Run the SNN forward (windowed input) and return spike counts, classes, probs, raster."""
    torch, *_ = _require_torch()
    model.eval()
    spk_chunks, mem_chunks, prob_chunks, pred_chunks = [], [], [], []
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        for start in range(0, len(X_t), batch_size):
            xb = X_t[start : start + batch_size]
            spk_rec, mem_rec = model(xb)
            logits = model.readout_logits(spk_rec, mem_rec)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            spk_chunks.append(spk_rec.cpu().numpy())
            mem_chunks.append(mem_rec.cpu().numpy())
            prob_chunks.append(probs.cpu().numpy())
            pred_chunks.append(pred.cpu().numpy())
    return {
        "probs": np.concatenate(prob_chunks),
        "pred": np.concatenate(pred_chunks),
        "spk_rec": np.concatenate(spk_chunks, axis=1) if spk_chunks else np.zeros((0, 0, 3)),
        "mem_rec": np.concatenate(mem_chunks, axis=1) if mem_chunks else np.zeros((0, 0, 3)),
    }


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _macro_f1(cm: np.ndarray) -> dict:
    f1s, precisions, recalls = [], [], []
    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    return {
        "macro_f1": float(np.mean(f1s)),
        "per_class_f1": [float(x) for x in f1s],
        "per_class_precision": [float(x) for x in precisions],
        "per_class_recall": [float(x) for x in recalls],
    }


def _backtest_sharpe(
    signals: np.ndarray,
    spread: np.ndarray,
    horizon: int,
) -> dict:
    """Simple spread paper-trade: enter on BUY/SELL, exit after `horizon` days.

    Returns Sharpe (annualised assuming 252 trading days) and hit rate.
    """
    rets = []
    for t in range(len(signals) - horizon):
        sig = signals[t]
        if sig == HOLD:
            continue
        # BUY = long spread (expect spread up) → P&L = spread[t+h] - spread[t]
        # SELL = short spread (expect spread down) → P&L = spread[t] - spread[t+h]
        delta = spread[t + horizon] - spread[t]
        pnl = delta if sig == BUY else -delta
        rets.append(pnl)
    if not rets:
        return {"sharpe": 0.0, "hit_rate": 0.0, "n_trades": 0, "avg_pnl": 0.0}
    arr = np.asarray(rets, dtype=np.float64)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sharpe = (mu / sd * np.sqrt(252 / horizon)) if sd > 0 else 0.0
    return {
        "sharpe": float(sharpe),
        "hit_rate": float((arr > 0).mean()),
        "n_trades": int(len(arr)),
        "avg_pnl": mu,
    }


# ── Pipeline entry point ────────────────────────────────────────────────


def _classical_signals_per_day(zscore: pd.Series) -> pd.Series:
    """Convert detect_signals() state-machine output into per-day class labels.

    Once entered, position is held until exit signal — every held day inherits
    the entry direction.  HOLD when flat.  Used as baseline for comparison.
    """
    events = detect_signals(zscore, entry_threshold=2.0, exit_threshold=0.5)
    z_clean = zscore.dropna()
    out = pd.Series(HOLD, index=z_clean.index, dtype=np.int64)
    if events.empty:
        return out
    state = HOLD
    event_iter = iter(events.itertuples(index=False))
    next_event = next(event_iter, None)
    for date in z_clean.index:
        while next_event is not None and pd.Timestamp(next_event.date) == date:
            sig = next_event.signal
            if sig == "long_entry":
                state = BUY
            elif sig == "short_entry":
                state = SELL
            elif sig in ("long_exit", "short_exit"):
                state = HOLD
            next_event = next(event_iter, None)
        out.loc[date] = state
    return out


def _build_pair_dataset(
    pair_id: str,
    ticker_a: str,
    ticker_b: str,
    pair_idx: int,
    n_pairs_total: int,
    adj_close: pd.DataFrame,
    log_returns: pd.DataFrame,
    cfg: SNNConfig,
) -> Optional[dict]:
    """Build the windowed feature tensor + labels for one pair.

    Returns a dict with:
        pair_id, ticker_a, ticker_b,
        X_win        : (N, W, n_base + n_pairs) — windowed input with pair one-hot
        y            : (N,) labels
        classical    : (N,) per-day classical baseline classes
        zscore       : (N,) Z-score values aligned with windowed output
        dates        : (N,) trading dates aligned with windowed output
    """
    feats = build_input_features(
        adj_close, log_returns, ticker_a, ticker_b, window=cfg.rolling_window
    )
    if len(feats) < max(200, cfg.window_size + 50):
        logger.warning("  %s: insufficient feature rows (%d) — skipping", pair_id, len(feats))
        return None

    X_per_day = encode_features_to_spikes(feats, cfg)              # (T, n_base)
    # Append one-hot pair embedding so the universal model can specialize per pair.
    onehot = np.zeros(n_pairs_total, dtype=np.float32)
    onehot[pair_idx] = 1.0
    pair_block = np.tile(onehot[None, :], (len(X_per_day), 1))      # (T, n_pairs)
    X_with_id = np.concatenate([X_per_day, pair_block], axis=1)     # (T, n_base + n_pairs)

    W = cfg.window_size
    X_win = build_windows(X_with_id, W)                              # (T - W + 1, W, ...)

    zscore_series = feats["zscore"]
    labels_full = generate_mean_reversion_labels(zscore_series, cfg).values
    classical_full = _classical_signals_per_day(zscore_series).values
    dates_full = feats.index

    # The window ending at row t covers days [t-W+1 .. t] → predictions are for row t
    y = labels_full[W - 1 :]
    classical = classical_full[W - 1 :]
    zscore_vals = zscore_series.values[W - 1 :]
    dates = dates_full[W - 1 :]

    return {
        "pair_id": pair_id,
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "X_win": X_win.astype(np.float32),
        "y": y.astype(np.int64),
        "classical": classical.astype(np.int64),
        "zscore": zscore_vals.astype(np.float32),
        "dates": dates,
    }


def _evaluate_pair(
    pair_data: dict,
    model,
    cfg: SNNConfig,
    test_start_idx: int,
) -> dict:
    """Run inference for one pair on its full history; compute test-set metrics.

    Test set = rows [test_start_idx : end].  Sharpe/F1 are computed on the test
    portion only (no train leakage); the full-history predictions are still
    written to disk for visualization.
    """
    out = predict_signals(model, pair_data["X_win"], cfg)
    pred = out["pred"]
    probs = out["probs"]

    y = pair_data["y"]
    classical = pair_data["classical"]
    zscore = pair_data["zscore"]

    test_pred = pred[test_start_idx:]
    test_y = y[test_start_idx:]
    test_classical = classical[test_start_idx:]
    test_zscore = zscore[test_start_idx:]

    cm = _confusion_matrix(test_y, test_pred)
    f1d = _macro_f1(cm)
    snn_bt = _backtest_sharpe(test_pred, test_zscore, horizon=cfg.label_horizon)
    cls_bt = _backtest_sharpe(test_classical, test_zscore, horizon=cfg.label_horizon)

    return {
        "out": out,
        "pred": pred,
        "probs": probs,
        "metrics": {
            "macro_f1": f1d["macro_f1"],
            "per_class_f1": f1d["per_class_f1"],
            "per_class_precision": f1d["per_class_precision"],
            "per_class_recall": f1d["per_class_recall"],
            "confusion_matrix": cm.tolist(),
            "snn_sharpe": snn_bt["sharpe"],
            "snn_hit_rate": snn_bt["hit_rate"],
            "snn_n_trades": snn_bt["n_trades"],
            "classical_sharpe": cls_bt["sharpe"],
            "classical_hit_rate": cls_bt["hit_rate"],
            "classical_n_trades": cls_bt["n_trades"],
            "delta_sharpe": snn_bt["sharpe"] - cls_bt["sharpe"],
            "n_test": int(len(test_y)),
        },
    }


def _write_raster_and_membrane(
    pair_data: dict,
    out: dict,
    cfg: SNNConfig,
    data_results: Path,
):
    """Write spike raster + membrane V(t) artifacts for the sample pair.

    Picks a single high-|Z| window (one sample point covering W trading days).
    Spike timeline is W * T_steps ticks; we reshape into (day-in-window, timestep).
    """
    zscore = pair_data["zscore"]
    if len(zscore) == 0:
        return
    centre = int(np.argmax(np.abs(zscore)))
    centre = max(0, min(len(zscore) - 1, centre))

    spk = out["spk_rec"]  # (W*T_steps, N, 3)
    mem = out["mem_rec"]
    if spk.shape[1] == 0:
        return

    W = cfg.window_size
    T_steps = cfg.n_timesteps
    spk_sample = spk[:, centre, :].reshape(W, T_steps, 3)   # (W days, T_steps, 3)
    mem_sample = mem[:, centre, :].reshape(W, T_steps, 3)

    sample_dates = pair_data["dates"][max(0, centre - W + 1) : centre + 1]
    if len(sample_dates) < W:
        # Pad with the centre date if window extends before history start
        sample_dates = list(sample_dates) + [sample_dates[-1]] * (W - len(sample_dates))

    raster_rows = []
    for d in range(W):
        for t in range(T_steps):
            for n in range(3):
                if float(spk_sample[d, t, n]) > 0:
                    raster_rows.append({
                        "day_index": d,
                        "date": sample_dates[d],
                        "timestep": t,
                        "neuron_id": n,
                        "neuron_name": CLASS_NAMES[n],
                    })
    pd.DataFrame(raster_rows).to_parquet(
        data_results / "snn_spike_raster_sample.parquet", index=False
    )

    mem_rows = []
    for d in range(W):
        for t in range(T_steps):
            for n in range(3):
                mem_rows.append({
                    "day_index": d,
                    "date": sample_dates[d],
                    "timestep": t,
                    "neuron_id": n,
                    "neuron_name": CLASS_NAMES[n],
                    "membrane": float(mem_sample[d, t, n]),
                })
    pd.DataFrame(mem_rows).to_parquet(
        data_results / "snn_membrane_sample.parquet", index=False
    )


def run_snn_signals(
    config: PipelineConfig,
    retrain: bool = False,
    snn_cfg: Optional[SNNConfig] = None,
) -> dict:
    """Pipeline Step 13: train one universal SNN classifier across all top-N pairs.

    Phase 1: build windowed feature tensors per pair (with one-hot pair embedding).
    Phase 2: pool train portions across all pairs and train ONE universal model
             with focal loss + windowed temporal context.
    Phase 3: run inference per pair using the universal model, write artifacts.

    `snn_cfg` overrides the default SNNConfig (useful for smoke tests).  If
    `retrain=False` and a cached `snn_universal_weights.pt` exists, training is
    skipped and only inference is re-run.
    """
    logger.info("── Spiking Neural Network Signals (universal model) ──")
    torch, *_ = _require_torch()

    cfg = snn_cfg if snn_cfg is not None else SNNConfig()
    cfg.retrain = retrain

    # Per-market data paths derived from config
    data_processed = config.data_processed
    data_results = config.data_results
    snn_signals_dir = data_results / "snn_signals"
    snn_weights_dir = data_results / "snn_model_weights"

    adj_path = data_processed / "adj_close.parquet"
    ret_path = data_processed / "log_returns.parquet"
    disloc_path = data_results / "dislocation_candidates.csv"
    for p in (adj_path, ret_path, disloc_path):
        if not p.exists():
            logger.warning("Missing required artifact %s — skipping SNN step.", p)
            return {}

    adj_close = pd.read_parquet(adj_path)
    log_returns = pd.read_parquet(ret_path)
    candidates = pd.read_csv(disloc_path).head(cfg.top_n_pairs)
    if candidates.empty:
        logger.warning("No dislocation candidates — skipping SNN step.")
        return {}

    snn_signals_dir.mkdir(parents=True, exist_ok=True)
    snn_weights_dir.mkdir(parents=True, exist_ok=True)

    n_pairs_total = len(candidates)
    sample_pair_id = f"{candidates.iloc[0]['ticker_a']}_{candidates.iloc[0]['ticker_b']}"

    # ── Phase 1: per-pair dataset construction ────────────────────────
    logger.info("Building per-pair windowed datasets (W=%d, T_steps=%d)…",
                cfg.window_size, cfg.n_timesteps)
    pair_datasets: list[dict] = []
    pair_list = []
    for idx, row in candidates.reset_index(drop=True).iterrows():
        ta, tb = row["ticker_a"], row["ticker_b"]
        pair_id = f"{ta}_{tb}"
        pair_list.append({"ticker_a": ta, "ticker_b": tb, "pair_id": pair_id})
        try:
            d = _build_pair_dataset(
                pair_id, ta, tb, idx, n_pairs_total,
                adj_close, log_returns, cfg,
            )
        except Exception as e:
            logger.exception("  %s: dataset build failed (%s)", pair_id, e)
            continue
        if d is None:
            continue
        pair_datasets.append(d)
        n_buy = int((d["y"] == BUY).sum())
        n_sell = int((d["y"] == SELL).sum())
        n_hold = int((d["y"] == HOLD).sum())
        logger.info(
            "  %s: %d windows  (HOLD=%d  BUY=%d  SELL=%d)",
            pair_id, len(d["X_win"]), n_hold, n_buy, n_sell,
        )

    if not pair_datasets:
        logger.warning("No pair datasets built — skipping training.")
        return {}

    n_inputs = pair_datasets[0]["X_win"].shape[-1]
    logger.info(
        "Pooled training: %d pairs, %d input channels (%d base + %d pair-id), W=%d",
        len(pair_datasets), n_inputs, n_inputs - n_pairs_total, n_pairs_total, cfg.window_size,
    )

    # ── Phase 2: pool, time-split per pair, train universal model ─────
    universal_weights_path = snn_weights_dir / "universal.pt"
    history: list[dict] = []

    if universal_weights_path.exists() and not cfg.retrain:
        model = build_lif_classifier(n_inputs, cfg)
        model.load_state_dict(torch.load(universal_weights_path, weights_only=True))
        logger.info("Loaded cached universal weights from %s", universal_weights_path.name)
    else:
        X_tr_parts, y_tr_parts, X_va_parts, y_va_parts = [], [], [], []
        test_starts = {}  # pair_id → start index of the held-out test set inside that pair
        for d in pair_datasets:
            n = len(d["X_win"])
            train_end = int(n * cfg.train_ratio)
            val_end = int(n * (cfg.train_ratio + 0.15))
            X_tr_parts.append(d["X_win"][:train_end])
            y_tr_parts.append(d["y"][:train_end])
            X_va_parts.append(d["X_win"][train_end:val_end])
            y_va_parts.append(d["y"][train_end:val_end])
            test_starts[d["pair_id"]] = val_end

        X_tr = np.concatenate(X_tr_parts, axis=0)
        y_tr = np.concatenate(y_tr_parts, axis=0)
        X_va = np.concatenate(X_va_parts, axis=0)
        y_va = np.concatenate(y_va_parts, axis=0)

        # Shuffle train to mix pairs (val/test stay time-ordered)
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(X_tr))
        X_tr = X_tr[perm]
        y_tr = y_tr[perm]

        logger.info(
            "Universal train: %d windows  val: %d windows  (pooled across %d pairs)",
            len(X_tr), len(X_va), len(pair_datasets),
        )
        cls_counts = np.bincount(y_tr, minlength=3)
        logger.info(
            "  train class counts: HOLD=%d  BUY=%d  SELL=%d",
            cls_counts[HOLD], cls_counts[BUY], cls_counts[SELL],
        )

        model, history = train_snn(X_tr, y_tr, X_va, y_va, cfg)
        torch.save(model.state_dict(), universal_weights_path)
        for row in history:
            row["pair"] = "_universal_"

    # Need test_starts for both the cached and freshly-trained paths.
    if universal_weights_path.exists() and not cfg.retrain:
        test_starts = {}
        for d in pair_datasets:
            n = len(d["X_win"])
            test_starts[d["pair_id"]] = int(n * (cfg.train_ratio + 0.15))

    # ── Phase 3: per-pair inference + metrics + artifacts ─────────────
    logger.info("Running per-pair inference + writing artifacts…")
    per_pair_metrics: dict[str, dict] = {}
    macro_f1_vals = []
    snn_sharpes, cls_sharpes = [], []

    for d in pair_datasets:
        pair_id = d["pair_id"]
        ev = _evaluate_pair(d, model, cfg, test_start_idx=test_starts[pair_id])
        m = ev["metrics"]
        m["pair_id"] = pair_id
        m["ticker_a"] = d["ticker_a"]
        m["ticker_b"] = d["ticker_b"]
        per_pair_metrics[pair_id] = m
        macro_f1_vals.append(m["macro_f1"])
        snn_sharpes.append(m["snn_sharpe"])
        cls_sharpes.append(m["classical_sharpe"])

        # Per-pair signal artifact (full history of predictions)
        pred = ev["pred"]
        probs = ev["probs"]
        sig_df = pd.DataFrame({
            "date": d["dates"],
            "zscore": d["zscore"],
            "prob_hold": probs[:, HOLD],
            "prob_buy": probs[:, BUY],
            "prob_sell": probs[:, SELL],
            "signal": [CLASS_NAMES[i] for i in pred],
            "classical_signal": [CLASS_NAMES[i] for i in d["classical"]],
        })
        sig_df.to_parquet(snn_signals_dir / f"{pair_id}.parquet", index=False)

        # Spike raster + membrane V(t) for the sample pair only
        if pair_id == sample_pair_id:
            _write_raster_and_membrane(d, ev["out"], cfg, data_results)
            per_pair_metrics[pair_id]["sample_pair"] = pair_id

        logger.info(
            "  %s done — F1=%.3f  SNN-Sh=%.2f  Cls-Sh=%.2f  ΔSh=%+.2f  trades=%d (cls %d)",
            pair_id, m["macro_f1"], m["snn_sharpe"], m["classical_sharpe"],
            m["delta_sharpe"], m["snn_n_trades"], m["classical_n_trades"],
        )

    mean_f1 = float(np.mean(macro_f1_vals)) if macro_f1_vals else 0.0
    mean_snn_sh = float(np.mean(snn_sharpes)) if snn_sharpes else 0.0
    mean_cls_sh = float(np.mean(cls_sharpes)) if cls_sharpes else 0.0
    logger.info(
        "Aggregate (test-set, %d pairs): mean macro-F1=%.3f  SNN-Sh=%+.2f  Cls-Sh=%+.2f  ΔSh=%+.2f",
        len(macro_f1_vals), mean_f1, mean_snn_sh, mean_cls_sh, mean_snn_sh - mean_cls_sh,
    )

    summary = {
        "n_pairs": len(per_pair_metrics),
        "per_pair": per_pair_metrics,
        "config": asdict(cfg),
        "sample_pair": sample_pair_id,
        "model_type": "universal",
        "n_inputs": int(n_inputs),
        "aggregate": {
            "mean_macro_f1": mean_f1,
            "mean_snn_sharpe": mean_snn_sh,
            "mean_classical_sharpe": mean_cls_sh,
            "mean_delta_sharpe": mean_snn_sh - mean_cls_sh,
        },
    }
    with open(data_results / "snn_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    pd.DataFrame(pair_list).to_csv(data_results / "snn_pair_list.csv", index=False)
    if history:
        pd.DataFrame(history).to_csv(data_results / "snn_training_history.csv", index=False)

    logger.info(
        "SNN signals saved to data/results/snn_signals/ (%d pairs).",
        len(per_pair_metrics),
    )
    return summary
