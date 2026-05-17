"""Tests for the SNN signals module — encoder + label generator + (optional) forward pass.

Tests in this file are split into two tiers:
1. Torch-free tests for the pure-numpy encoders and label generator (always run).
2. Torch-dependent tests for the LIF classifier forward pass (skipped if torch
   is not installed — see the `snn` optional extra in pyproject.toml).
"""

import importlib

import numpy as np
import pandas as pd
import pytest

from src.snn_signals import (
    HOLD, BUY, SELL,
    SNNConfig,
    delta_encode,
    population_encode,
    encode_features_to_spikes,
    generate_mean_reversion_labels,
)


# ── Torch-free tests ────────────────────────────────────────────────────


def test_delta_encode_basic():
    x = np.array([0.0, 0.5, 0.6, 0.0, -0.5, -0.6], dtype=np.float32)
    spikes = delta_encode(x, theta=0.4)
    assert spikes.shape == (6, 2)
    # First step: no Δ (prepend = self) → no spike
    assert spikes[0].tolist() == [0.0, 0.0]
    # Step 1: jump 0 → 0.5 (Δ=0.5 ≥ 0.4) → up channel fires
    assert spikes[1].tolist() == [1.0, 0.0]
    # Step 2: 0.5 → 0.6 (Δ=0.1, below threshold) → no spike
    assert spikes[2].tolist() == [0.0, 0.0]
    # Step 3: 0.6 → 0.0 (Δ=-0.6 ≤ -0.4) → down channel fires
    assert spikes[3].tolist() == [0.0, 1.0]


def test_delta_encode_is_sparse_for_flat_signal():
    """Flat signal should produce essentially no spikes — sanity check on event-driven semantics."""
    x = np.ones(200, dtype=np.float32)
    spikes = delta_encode(x, theta=0.1)
    assert spikes.sum() == 0.0


def test_population_encode_shape_and_range():
    x = np.linspace(-1, 1, 50)
    rates = population_encode(x, n_fields=5, x_min=-1.0, x_max=1.0)
    assert rates.shape == (50, 5)
    # All firing rates must be in [0, 1]
    assert rates.min() >= 0.0
    assert rates.max() <= 1.0
    # The neuron whose centre is closest to x[0]=-1 should fire strongest at t=0
    assert int(np.argmax(rates[0])) == 0
    # And the neuron whose centre is closest to x[-1]=1 should fire strongest at the end
    assert int(np.argmax(rates[-1])) == 4


def test_encode_features_to_spikes_channel_count():
    """encode_features_to_spikes concatenates 11 feature encoders → 45 channels."""
    n = 50
    feats = pd.DataFrame({
        "zscore": np.linspace(-3, 3, n),
        "dzscore": np.zeros(n),
        "z_lag5": np.zeros(n),
        "z_lag20": np.zeros(n),
        "rolling_corr": np.linspace(-1, 1, n),
        "return_a": np.zeros(n),
        "return_b": np.zeros(n),
        "spread_vol": np.linspace(0, 0.05, n),
        "market_disp": np.linspace(0, 0.03, n),
        "market_breadth": np.linspace(0, 1, n),
        "inv_half_life": np.linspace(0, 0.2, n),
    })
    cfg = SNNConfig()
    spikes = encode_features_to_spikes(feats, cfg)
    # 2 + 2 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 5 + 1 = 45
    assert spikes.shape == (n, 45)
    assert spikes.dtype == np.float32


def test_label_generator_buy_on_negative_z_reverting():
    """Z drops to -2 then reverts: that day should be labeled BUY."""
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    z = pd.Series(
        # day 5: Z=-2.0; days 6-10: reverts toward 0
        [0.0] * 5 + [-2.0] + [-1.5, -1.0, -0.5, 0.0, 0.0] + [0.0] * 19,
        index=idx,
    )
    cfg = SNNConfig(label_horizon=10, label_entry_z=1.5, label_exit_z=0.5)
    labels = generate_mean_reversion_labels(z, cfg)
    assert labels.iloc[5] == BUY
    # Day 0 with Z=0 → HOLD
    assert labels.iloc[0] == HOLD


def test_label_generator_sell_on_positive_z_reverting():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    z = pd.Series(
        [0.0] * 5 + [2.5] + [2.0, 1.0, 0.4, 0.0, 0.0] + [0.0] * 19,
        index=idx,
    )
    cfg = SNNConfig(label_horizon=10, label_entry_z=1.5, label_exit_z=0.5)
    labels = generate_mean_reversion_labels(z, cfg)
    assert labels.iloc[5] == SELL


def test_label_generator_hold_when_no_reversion():
    """Z stays at extreme without reverting → HOLD (no spurious BUY/SELL)."""
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    z = pd.Series([-2.5] * 30, index=idx)
    cfg = SNNConfig(label_horizon=10, label_entry_z=1.5, label_exit_z=0.5)
    labels = generate_mean_reversion_labels(z, cfg)
    # No reversion within K days → every day should be HOLD
    assert (labels == HOLD).all()


# ── Torch-dependent tests ───────────────────────────────────────────────


HAS_TORCH = importlib.util.find_spec("torch") is not None and \
    importlib.util.find_spec("snntorch") is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch + snntorch not installed")
def test_lif_classifier_forward_pass_shape():
    """Windowed forward: (B, W, n_inputs) → spk/mem (W*T_steps, B, 3) — binary spikes."""
    from src.snn_signals import build_lif_classifier
    import torch

    cfg = SNNConfig(n_timesteps=10, window_size=3, n_hidden=8)
    model = build_lif_classifier(n_inputs=20, cfg=cfg)
    x = torch.randn(4, 3, 20)  # batch=4, window=3, 20 input channels
    spk, mem = model(x)
    assert spk.shape == (30, 4, 3)  # W * T_steps = 30
    assert mem.shape == (30, 4, 3)
    spk_unique = set(spk.unique().tolist())
    assert spk_unique.issubset({0.0, 1.0})


@pytest.mark.skipif(not HAS_TORCH, reason="torch + snntorch not installed")
def test_lif_classifier_accepts_legacy_2d_input():
    """Pre-windowing call shape (B, n_inputs) should still work (gets unsqueezed to W=1)."""
    from src.snn_signals import build_lif_classifier
    import torch
    cfg = SNNConfig(n_timesteps=10, window_size=1, n_hidden=8)
    model = build_lif_classifier(n_inputs=20, cfg=cfg)
    x = torch.randn(2, 20)
    spk, mem = model(x)
    assert spk.shape == (10, 2, 3)


def test_build_windows_basic():
    from src.snn_signals import build_windows
    X = np.arange(15).reshape(5, 3).astype(np.float32)  # T=5, n_features=3
    W = 3
    wins = build_windows(X, W)
    assert wins.shape == (3, 3, 3)  # T - W + 1 = 3 windows
    # First window covers rows 0..2
    assert (wins[0] == X[0:3]).all()
    # Last window covers rows 2..4
    assert (wins[2] == X[2:5]).all()


def test_focal_loss_zero_when_perfect():
    """When the predicted spike counts massively favor the correct class, focal loss → 0."""
    if not HAS_TORCH:
        pytest.skip("torch required")
    from src.snn_signals import focal_loss_on_spike_counts
    import torch
    # 3 samples, each with strong preference for the correct class
    spike_counts = torch.tensor([
        [100.0, 0.0, 0.0],   # class 0
        [0.0, 100.0, 0.0],   # class 1
        [0.0, 0.0, 100.0],   # class 2
    ])
    targets = torch.tensor([0, 1, 2])
    weights = torch.tensor([1.0, 1.0, 1.0])
    loss = focal_loss_on_spike_counts(spike_counts, targets, weights, gamma=2.0)
    assert float(loss.item()) < 1e-5


def test_magnitude_label_requires_min_reversion():
    """If Z dips just past entry_z but bounces less than min_reversion, label stays HOLD."""
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    # Z goes to -1.5 then only reverts to -1.3 → reversion = 0.2 < min_reversion=0.8
    z = pd.Series([0.0] * 5 + [-1.5] + [-1.4, -1.35, -1.3, -1.3] + [-1.3] * 20, index=idx)
    cfg = SNNConfig(label_horizon=10, label_entry_z=1.2, label_min_reversion=0.8)
    labels = generate_mean_reversion_labels(z, cfg)
    # Day 5 should stay HOLD because the reversion magnitude (0.2) is below threshold
    assert labels.iloc[5] == HOLD
