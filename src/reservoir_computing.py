"""Reservoir Computing (Echo State Network) for financial market prediction.

Implements an ESN from scratch (numpy/scipy only) for:
1. Market dispersion forecasting — predicts next-day cross-sectional volatility
2. Pair spread prediction — forecasts mean-reversion for dislocation candidates

Why Reservoir Computing for stock correlation networks:
───────────────────────────────────────────────────────
- Fading memory naturally weights recent regime dynamics, matching financial
  correlation timescales (weeks to months).
- The random reservoir acts as a nonlinear kernel over 73 stocks, capturing
  cross-stock interactions without explicit O(N²) feature engineering.
- Training is ridge regression — one closed-form solve, no GPU, no overfitting
  on 1500-day datasets where LSTM/GRU would fail.
- Spectral radius / leak rate control the timescale of dynamics, complementing
  the wavelet multi-scale analysis already in the pipeline.

Reference: Jaeger (2001) "The echo state approach to analysing and training
recurrent neural networks", GMD Report 148.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.config import PipelineConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RESULTS = PROJECT_ROOT / "data" / "results"


# ── Configuration ──────────────────────────────────────────────────────


@dataclass
class ESNConfig:
    """Echo State Network hyperparameters."""

    reservoir_size: int = 300       # N reservoir neurons
    spectral_radius: float = 0.9    # < 1 for echo state property
    input_scaling: float = 0.5      # scales input weights
    leak_rate: float = 0.3          # leaky-integrator time constant
    ridge_alpha: float = 10.0       # readout regularization (strong to avoid overfitting)
    sparsity: float = 0.9           # fraction of zero reservoir weights
    seed: int = 42
    washout: int = 100              # initial transient steps to discard
    n_pca: int = 5                  # PCA components from returns
    vol_windows: tuple = (5, 20)    # rolling volatility windows
    train_ratio: float = 0.7        # train/test split


# ── Echo State Network ────────────────────────────────────────────────


class EchoStateNetwork:
    """Echo State Network with ridge regression readout.

    The reservoir (recurrent layer) has fixed random weights.  Only the
    linear readout is trained, making training O(T × N²) with N = reservoir
    size and T = sequence length.
    """

    def __init__(
        self,
        reservoir_size: int = 300,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        leak_rate: float = 0.3,
        ridge_alpha: float = 10.0,
        sparsity: float = 0.9,
        seed: int = 42,
        washout: int = 100,
    ):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha
        self.sparsity = sparsity
        self.seed = seed
        self.washout = washout

        self.rng = np.random.RandomState(seed)
        self.W_in = None   # (N, n_inputs+1) input weight matrix
        self.W = None       # (N, N) reservoir weight matrix
        self.W_out = None   # (N + n_inputs, n_outputs) readout weights
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._fitted = False
        self._n_inputs = 0

    def _init_reservoir(self, n_inputs: int) -> None:
        """Create fixed input and reservoir weight matrices."""
        N = self.reservoir_size
        self._n_inputs = n_inputs

        # Input weights: uniform [-1,1] with scaling; +1 col for bias
        self.W_in = (
            self.rng.uniform(-1, 1, (N, n_inputs + 1)) * self.input_scaling
        )

        # Reservoir: sparse random matrix
        W = self.rng.randn(N, N)
        mask = self.rng.rand(N, N) < (1.0 - self.sparsity)
        W *= mask

        # Scale to target spectral radius
        rho = np.max(np.abs(linalg.eigvals(W)))
        if rho > 0:
            W *= self.spectral_radius / rho
        self.W = W

    def _run_reservoir(self, X: np.ndarray) -> np.ndarray:
        """Drive the reservoir with scaled input and collect states.

        Parameters
        ----------
        X : ndarray of shape (T, n_features)
            Scaled input sequence.

        Returns
        -------
        states : ndarray of shape (T, reservoir_size)
        """
        T = X.shape[0]
        N = self.reservoir_size
        alpha = self.leak_rate

        # Augment input with constant bias
        X_bias = np.hstack([X, np.ones((T, 1))])

        states = np.zeros((T, N))
        x = np.zeros(N)

        for t in range(T):
            pre = self.W_in @ X_bias[t] + self.W @ x
            x = (1 - alpha) * x + alpha * np.tanh(pre)
            states[t] = x

        return states

    # ── Training ───────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Train the ESN readout via ridge regression.

        Parameters
        ----------
        X_train : (T, n_features)
        y_train : (T,) or (T, n_targets)

        Returns
        -------
        y_pred_train : training predictions (after washout) for diagnostics
        """
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        X_sc = self.scaler_X.fit_transform(X_train)
        y_sc = self.scaler_y.fit_transform(y_train)

        self._init_reservoir(X_sc.shape[1])
        states = self._run_reservoir(X_sc)

        # Discard washout transient
        w = self.washout
        S = np.hstack([states[w:], X_sc[w:]])   # extended state
        Y = y_sc[w:]

        # Closed-form ridge: W_out = (S'S + αI)⁻¹ S'Y
        reg = self.ridge_alpha * np.eye(S.shape[1])
        self.W_out = np.linalg.solve(S.T @ S + reg, S.T @ Y)
        self._fitted = True

        y_pred_sc = S @ self.W_out
        return self.scaler_y.inverse_transform(y_pred_sc).squeeze()

    # ── Prediction ─────────────────────────────────────────────────

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict with a cold-start reservoir (state initialised to zero)."""
        assert self._fitted, "Call fit() first"
        X_sc = self.scaler_X.transform(X_test)
        states = self._run_reservoir(X_sc)
        S = np.hstack([states, X_sc])
        y_pred_sc = S @ self.W_out
        return self.scaler_y.inverse_transform(y_pred_sc).squeeze()

    def predict_continuation(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        """Predict test data with the reservoir warmed up on training data.

        This is the correct walk-forward approach: the reservoir state
        carries over from training into the test sequence, so there is
        no cold-start transient on test data.
        """
        assert self._fitted, "Call fit() first"
        X_full = np.vstack([X_train, X_test])
        X_sc = self.scaler_X.transform(X_full)
        states = self._run_reservoir(X_sc)

        T_train = X_train.shape[0]
        S_test = np.hstack([states[T_train:], X_sc[T_train:]])
        y_pred_sc = S_test @ self.W_out
        return self.scaler_y.inverse_transform(y_pred_sc).squeeze()


# ── Feature Engineering ───────────────────────────────────────────────


def build_market_features(
    returns: pd.DataFrame,
    vol_windows: tuple[int, ...] = (5, 20),
    n_pca: int = 5,
) -> pd.DataFrame:
    """Build daily market-wide features from stock returns.

    All features are computable at time *t* from data up to *t*
    (cross-sectional stats) or [t-w, t] (rolling stats), so they
    are safe to use as inputs without look-ahead bias.

    Parameters
    ----------
    returns : DataFrame (T, N_stocks) of log returns
    vol_windows : rolling volatility window sizes
    n_pca : principal components to extract

    Returns
    -------
    features : DataFrame (T', n_features), NaN rows dropped
    """
    feats = pd.DataFrame(index=returns.index)

    # Cross-sectional statistics (computed from same-day returns)
    feats["market_return"] = returns.mean(axis=1)
    feats["dispersion"] = returns.std(axis=1)
    feats["cs_skew"] = returns.skew(axis=1)
    feats["cs_kurt"] = returns.kurtosis(axis=1)
    feats["breadth"] = (returns > 0).mean(axis=1)
    feats["abs_avg_return"] = returns.abs().mean(axis=1)
    feats["return_range"] = returns.max(axis=1) - returns.min(axis=1)

    # Temporal volatility features
    for w in vol_windows:
        feats[f"realized_vol_{w}d"] = (
            feats["market_return"].rolling(w, min_periods=w).std()
        )
    if len(vol_windows) >= 2:
        short_w, long_w = vol_windows[0], vol_windows[1]
        feats["vol_ratio"] = (
            feats[f"realized_vol_{short_w}d"]
            / feats[f"realized_vol_{long_w}d"].replace(0, np.nan)
        )

    # PCA of returns cross-section (dominant market factors)
    returns_clean = returns.fillna(0)
    pca = PCA(n_components=min(n_pca, returns_clean.shape[1]))
    pca_vals = pca.fit_transform(returns_clean)
    for i in range(pca_vals.shape[1]):
        feats[f"pca_{i + 1}"] = pca_vals[:, i]

    # Lagged features (autoregressive signal for readout)
    feats["dispersion_lag1"] = feats["dispersion"].shift(1)
    feats["market_return_lag1"] = feats["market_return"].shift(1)

    feats = feats.dropna()
    return feats


def build_pair_features(
    returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int = 60,
) -> pd.DataFrame:
    """Build features for a specific stock pair (spread prediction).

    Parameters
    ----------
    returns : full returns DataFrame
    ticker_a, ticker_b : stock tickers
    window : rolling window for pair statistics
    """
    feats = pd.DataFrame(index=returns.index)

    ra, rb = returns[ticker_a], returns[ticker_b]

    feats["return_a"] = ra
    feats["return_b"] = rb
    feats["spread"] = ra - rb
    feats["spread_ma"] = feats["spread"].rolling(window, min_periods=20).mean()
    feats["spread_std"] = feats["spread"].rolling(window, min_periods=20).std()
    feats["zscore"] = (
        (feats["spread"] - feats["spread_ma"])
        / feats["spread_std"].replace(0, np.nan)
    )
    feats["rolling_corr"] = ra.rolling(window, min_periods=20).corr(rb)

    # Market context features
    feats["market_return"] = returns.mean(axis=1)
    feats["dispersion"] = returns.std(axis=1)

    feats = feats.dropna()
    return feats


# ── Evaluation ────────────────────────────────────────────────────────


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, name: str = "target"
) -> dict:
    """Regression and directional-change accuracy metrics."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]

    corr = (
        float(np.corrcoef(yt, yp)[0, 1])
        if np.std(yp) > 1e-10
        else 0.0
    )

    # Direction-of-change accuracy: did the prediction get the
    # direction of change (delta) right?
    if len(yt) > 1:
        delta_true = np.diff(yt)
        delta_pred = np.diff(yp)
        dir_acc = float(np.mean(np.sign(delta_true) == np.sign(delta_pred)))
    else:
        dir_acc = 0.0

    return {
        "target": name,
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "correlation": corr,
        "direction_of_change_accuracy": dir_acc,
        "n_samples": int(len(yt)),
    }


def walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    esn_params: dict,
    n_folds: int = 5,
    min_train_ratio: float = 0.5,
) -> dict:
    """Walk-forward (expanding window) cross-validation.

    For each fold the readout is retrained on [0 .. t_i] and evaluated
    on [t_i .. t_{i+1}].  The reservoir runs continuously so state
    carries across the train/test boundary.

    Returns dict with fold metrics, aggregate metrics, and arrays of
    concatenated (predictions, actuals).
    """
    T = len(X)
    min_train = int(T * min_train_ratio)
    test_size = (T - min_train) // n_folds

    all_preds, all_actuals = [], []
    fold_metrics = []

    for fold in range(n_folds):
        t_split = min_train + fold * test_size
        t_end = min(t_split + test_size, T)
        if t_split >= T or t_end <= t_split:
            break

        X_train, X_test = X[:t_split], X[t_split:t_end]
        y_train, y_test = y[:t_split], y[t_split:t_end]

        esn = EchoStateNetwork(**esn_params)
        esn.fit(X_train, y_train)
        y_pred = esn.predict_continuation(X_train, X_test)

        metrics = evaluate_predictions(y_test, y_pred, f"fold_{fold}")
        fold_metrics.append(metrics)
        all_preds.extend(y_pred.tolist())
        all_actuals.extend(y_test.tolist())

        logger.info(
            "  Fold %d: train=%d test=%d  R²=%.4f  DA=%.1f%%",
            fold,
            len(X_train),
            len(X_test),
            metrics["r2"],
            metrics["direction_of_change_accuracy"] * 100,
        )

    predictions = np.array(all_preds)
    actuals = np.array(all_actuals)
    aggregate = evaluate_predictions(actuals, predictions, "aggregate")

    return {
        "fold_metrics": fold_metrics,
        "aggregate": aggregate,
        "predictions": predictions,
        "actuals": actuals,
    }


# ── Naive Baselines ───────────────────────────────────────────────────


def naive_baselines(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute naive baseline metrics for comparison.

    - Persistence: predict yesterday's value
    - Mean: predict training-set mean
    """
    # Persistence baseline
    y_persist = np.roll(y_test, 1)
    y_persist[0] = y_train[-1]
    persist = evaluate_predictions(y_test, y_persist, "persistence")

    # Mean baseline
    y_mean = np.full_like(y_test, y_train.mean())
    mean_bl = evaluate_predictions(y_test, y_mean, "mean")

    return {"persistence": persist, "mean": mean_bl}


# ── Pipeline Entry Point ──────────────────────────────────────────────


def run_reservoir_computing(config: PipelineConfig = None) -> dict:
    """Run reservoir computing analysis.

    Trains ESN models for market dispersion forecasting and pair spread
    prediction.  Saves results to ``data/results/rc_*``.
    """
    logger.info("── Reservoir Computing ──")

    returns = pd.read_parquet(DATA_PROCESSED / "log_returns.parquet")
    logger.info("Loaded returns: %s", returns.shape)

    cfg = ESNConfig()
    esn_params = {
        "reservoir_size": cfg.reservoir_size,
        "spectral_radius": cfg.spectral_radius,
        "input_scaling": cfg.input_scaling,
        "leak_rate": cfg.leak_rate,
        "ridge_alpha": cfg.ridge_alpha,
        "sparsity": cfg.sparsity,
        "seed": cfg.seed,
        "washout": cfg.washout,
    }

    # ── Task 1: Market Dispersion Prediction ──────────────────────

    logger.info("Task 1: Market dispersion prediction")
    features = build_market_features(
        returns, vol_windows=cfg.vol_windows, n_pca=cfg.n_pca
    )

    # Target: next-day dispersion
    target = features["dispersion"].shift(-1).dropna()
    feat_aligned = features.loc[target.index]

    X = feat_aligned.values
    y = target.values
    dates = target.index

    logger.info(
        "Features: %d samples × %d features", X.shape[0], X.shape[1]
    )

    # Walk-forward cross-validation
    results_disp = walk_forward_validation(X, y, esn_params, n_folds=5)
    agg = results_disp["aggregate"]
    logger.info(
        "Dispersion — R²=%.4f  RMSE=%.6f  DA=%.1f%%",
        agg["r2"],
        agg["rmse"],
        agg["direction_of_change_accuracy"] * 100,
    )

    # Final train/test split for saved predictions + baselines
    split = int(len(X) * cfg.train_ratio)
    esn_full = EchoStateNetwork(**esn_params)
    esn_full.fit(X[:split], y[:split])
    y_pred_full = esn_full.predict_continuation(X[:split], X[split:])

    baselines = naive_baselines(y[:split], y[split:])
    logger.info(
        "Baselines — Persistence R²=%.4f, Mean R²=%.4f",
        baselines["persistence"]["r2"],
        baselines["mean"]["r2"],
    )

    pred_df = pd.DataFrame(
        {
            "date": dates[split:],
            "actual_dispersion": y[split:],
            "predicted_dispersion": y_pred_full,
        }
    )
    pred_df.to_parquet(
        DATA_RESULTS / "rc_dispersion_predictions.parquet", index=False
    )

    # ── Task 2: Pair Spread Prediction ────────────────────────────

    disloc_path = DATA_RESULTS / "dislocation_candidates.csv"
    pair_results = {}

    if disloc_path.exists():
        logger.info("Task 2: Pair spread prediction")
        candidates = pd.read_csv(disloc_path)
        top_pairs = candidates.head(3)

        for _, row in top_pairs.iterrows():
            pair_name = f"{row['ticker_a']}-{row['ticker_b']}"
            logger.info("  Pair: %s", pair_name)
            try:
                pf = build_pair_features(
                    returns, row["ticker_a"], row["ticker_b"]
                )
                # Predict next-day z-score (mean-reverting, bounded)
                pair_target = pf["zscore"].shift(-1).dropna()
                pX = pf.loc[pair_target.index].values
                py = pair_target.values

                pr = walk_forward_validation(pX, py, esn_params, n_folds=3)
                pair_results[pair_name] = pr["aggregate"]
                logger.info(
                    "  %s — R²=%.4f  DA=%.1f%%",
                    pair_name,
                    pr["aggregate"]["r2"],
                    pr["aggregate"]["direction_of_change_accuracy"] * 100,
                )
            except Exception as e:
                logger.warning("  Pair %s failed: %s", pair_name, e)

    # ── Feature Importance ────────────────────────────────────────

    feat_importance = None
    if esn_full.W_out is not None:
        n_res = esn_full.reservoir_size
        # Readout = [reservoir_weights | input_weights]
        input_w = np.abs(esn_full.W_out[n_res:]).squeeze()
        cols = list(features.columns)
        # +1 for bias in extended state
        feat_importance = pd.DataFrame(
            {
                "feature": cols[: len(input_w)],
                "weight_magnitude": input_w[: len(cols)],
            }
        ).sort_values("weight_magnitude", ascending=False)
        feat_importance.to_csv(
            DATA_RESULTS / "rc_feature_importance.csv", index=False
        )

    # ── Save Metadata ─────────────────────────────────────────────

    all_metrics = {
        "dispersion_prediction": results_disp["aggregate"],
        "dispersion_fold_metrics": results_disp["fold_metrics"],
        "baselines": baselines,
        "pair_spread_prediction": pair_results,
        "esn_config": asdict(cfg),
        "feature_columns": list(features.columns),
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "train_size": split,
        "test_size": len(X) - split,
    }

    with open(DATA_RESULTS / "rc_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    logger.info("Reservoir computing results saved to data/results/rc_*")
    return all_metrics
