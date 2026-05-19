"""Information-theoretic summaries of multivariate return panels.

Stage 13 of the StoNeCoAl pipeline. Adds a substantive answer to the TA's
"information theory perspective" suggestion by summarising the joint
distribution of a returns panel in four complementary measures:

- pairwise mutual information (plug-in estimator + closed-form Gaussian baseline)
- effective dimensionality D_eff (participation ratio of the eigenvalue spectrum)
- joint differential entropy ΔH = −½ log det Σ (joint Gaussian structure)
- regime KL divergence between Gaussian covariances (calm vs. crisis)
- rolling D_eff(t) + ΔH(t) for the dashboard's IT-over-time panel
- sign-sequence entropy rate (a one-bit-per-day weak-form-EMH fingerprint)

All entropy primitives are reused from ``src.transfer_entropy`` to avoid
duplicating the discretisation and entropy code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.transfer_entropy import (
    _discretize,
    _joint_entropy,
    _shannon_entropy,
)

logger = logging.getLogger(__name__)

LOG2_E = float(np.log2(np.e))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def pairwise_mi_value(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 4,
    *,
    units: str = "bits",
) -> float:
    """Plug-in mutual information estimator I(X;Y) for two 1D series.

    Equal-frequency binning into `n_bins` followed by the standard
    I(X;Y) = H(X) + H(Y) − H(X,Y) decomposition. Entropy primitives in
    `transfer_entropy.py` return nats; we convert to bits by default
    (set ``units='nats'`` to keep the native unit).
    """
    if units not in ("bits", "nats"):
        raise ValueError(f"units must be 'bits' or 'nats', got {units!r}")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return 0.0
    x_b = _discretize(x[mask], n_bins=n_bins)
    y_b = _discretize(y[mask], n_bins=n_bins)
    h_x = _shannon_entropy(x_b)
    h_y = _shannon_entropy(y_b)
    h_xy = _joint_entropy(x_b, y_b)
    mi_nats = max(0.0, h_x + h_y - h_xy)
    return mi_nats * LOG2_E if units == "bits" else mi_nats


def pairwise_mi_matrix(
    returns: pd.DataFrame,
    n_bins: int = 4,
    *,
    units: str = "bits",
) -> pd.DataFrame:
    """Pairwise MI matrix for an N-ticker returns panel.

    Symmetric; diagonal = H(X_i) so the matrix is comparable cell-to-cell.
    """
    cols = returns.columns.tolist()
    n = len(cols)
    out = np.zeros((n, n))
    raw = [returns.iloc[:, k].values for k in range(n)]
    for i in range(n):
        out[i, i] = _self_entropy(raw[i], n_bins=n_bins, units=units)
        for j in range(i + 1, n):
            mi = pairwise_mi_value(raw[i], raw[j], n_bins=n_bins, units=units)
            out[i, j] = mi
            out[j, i] = mi
    return pd.DataFrame(out, index=cols, columns=cols)


def _self_entropy(x: np.ndarray, n_bins: int, units: str) -> float:
    mask = np.isfinite(x)
    if mask.sum() < 30:
        return 0.0
    x_b = _discretize(x[mask], n_bins=n_bins)
    h = _shannon_entropy(x_b)
    return h * LOG2_E if units == "bits" else h


def gaussian_mi_from_corr(rho: float, *, units: str = "bits") -> float:
    """Closed-form mutual information between two Gaussian variables with
    Pearson correlation `rho`.

    I_gauss = −½ log(1 − ρ²). Reported in bits by default.
    """
    rho = max(min(float(rho), 0.9999), -0.9999)
    mi_nats = -0.5 * np.log(1.0 - rho * rho)
    mi_nats = max(0.0, float(mi_nats))
    return mi_nats * LOG2_E if units == "bits" else mi_nats


def gaussian_mi_matrix(corr: pd.DataFrame, *, units: str = "bits") -> pd.DataFrame:
    """Per-pair Gaussian MI baseline I_gauss = −½ log(1 − r²)."""
    vals = corr.values
    out = -0.5 * np.log(np.clip(1.0 - vals * vals, 1e-12, None))
    np.fill_diagonal(out, 0.0)
    if units == "bits":
        out = out * LOG2_E
    return pd.DataFrame(out, index=corr.index, columns=corr.columns)


def nonlinear_excess(
    mi: pd.DataFrame,
    mi_gauss: pd.DataFrame,
) -> pd.DataFrame:
    """Per-pair MI minus the Gaussian-MI proxy.

    Positive excess flags pairs whose joint distribution carries more
    information than a Gaussian with the same Pearson correlation —
    i.e., non-linear coupling Pearson misses.
    """
    common_idx = mi.index.intersection(mi_gauss.index)
    common_col = mi.columns.intersection(mi_gauss.columns)
    a = mi.loc[common_idx, common_col]
    b = mi_gauss.loc[common_idx, common_col]
    return a.subtract(b)


def top_nonlinear_pairs(
    excess: pd.DataFrame,
    top_k: int = 14,
) -> pd.DataFrame:
    """Return the `top_k` (ticker_a, ticker_b, excess) rows from the upper
    triangle of the excess matrix, sorted by excess descending."""
    rows = []
    cols = excess.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append((cols[i], cols[j], float(excess.iloc[i, j])))
    df = pd.DataFrame(rows, columns=["ticker_a", "ticker_b", "nonlinear_excess"])
    return df.sort_values("nonlinear_excess", ascending=False).head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Spectrum-derived measures
# ---------------------------------------------------------------------------

def d_eff(eigenvalues: np.ndarray) -> float:
    """Effective dimensionality of a covariance / correlation spectrum.

    Participation ratio: ``(Σλ)² / Σλ²``. Equals N for a perfectly isotropic
    (diagonal) spectrum; collapses toward 1 as a single eigenvalue dominates.
    Concentration of the spectrum is what RMT denoising tries to push away
    from; D_eff turns that into a single intuitive number.
    """
    eig = np.asarray(eigenvalues, dtype=float)
    eig = eig[eig > 0]
    if eig.size == 0:
        return 0.0
    return float((eig.sum() ** 2) / (eig ** 2).sum())


def joint_diff_entropy(cov: np.ndarray) -> float:
    """Differential entropy of a zero-mean multivariate Gaussian with
    covariance ``cov``, in nats. Equals ½ log((2πe)^k det Σ).

    For correlation matrices, the (2πe)^k constant is non-informative, so
    the dashboard usually plots ΔH = −½ log det Σ alone. We return the
    full Gaussian entropy here so the function is mathematically faithful;
    callers wanting the −½ log det version can use ``log_det_term`` below.
    """
    k = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("nan")
    return float(0.5 * (k * np.log(2 * np.pi * np.e) + logdet))


def log_det_term(cov: np.ndarray, *, ridge: float = 1e-6) -> float:
    """``−½ log det Σ`` — the joint-structure piece reported in the
    dashboard. Negative when the system is highly correlated (the
    covariance matrix's determinant is small).

    When ``Σ`` is exactly singular (det ≤ 0 numerically), we add a tiny
    ridge ``ridge * I`` and recompute. This handles the
    common-average-referenced EEG case where one channel is a linear
    combination of the others, which exactly zeros one eigenvalue.
    """
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        k = cov.shape[0]
        sign_r, logdet_r = np.linalg.slogdet(cov + ridge * np.eye(k))
        if sign_r <= 0:
            return float("nan")
        return float(-0.5 * logdet_r)
    return float(-0.5 * logdet)


def kl_gaussian_covariances(cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    """KL divergence between two zero-mean Gaussians ``N(0, Σ_a) ‖ N(0, Σ_b)``.

    Returns the closed-form
        ½ [ tr(Σ_b^{-1} Σ_a) − k + log(det Σ_b / det Σ_a) ]
    in nats. Strictly non-negative; zero iff ``cov_a == cov_b``.
    """
    cov_a = np.asarray(cov_a, dtype=float)
    cov_b = np.asarray(cov_b, dtype=float)
    if cov_a.shape != cov_b.shape:
        raise ValueError(
            f"covariance shape mismatch: {cov_a.shape} vs {cov_b.shape}"
        )
    k = cov_a.shape[0]
    try:
        b_inv = np.linalg.inv(cov_b)
    except np.linalg.LinAlgError:
        # Regularise mildly when Σ_b is singular (rare on real returns).
        b_inv = np.linalg.pinv(cov_b + 1e-8 * np.eye(k))
    sign_a, logdet_a = np.linalg.slogdet(cov_a)
    sign_b, logdet_b = np.linalg.slogdet(cov_b)
    if sign_a <= 0 or sign_b <= 0:
        return float("nan")
    trace_term = float(np.trace(b_inv @ cov_a))
    kl = 0.5 * (trace_term - k + (logdet_b - logdet_a))
    # Numerical noise can drag the result a hair below zero; clip.
    return float(max(0.0, kl))


# ---------------------------------------------------------------------------
# Rolling + sign-sequence statistics
# ---------------------------------------------------------------------------

def rolling_d_eff_dh(
    log_returns: pd.DataFrame,
    *,
    window: int = 60,
    step: int = 5,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Sliding-window D_eff(t) and −½ log det Σ(t).

    Each row is the right edge of a `window`-long window; correlation is
    Pearson on the surviving (non-NaN) sub-matrix. Empty windows return
    NaN for both stats.
    """
    if min_periods is None:
        min_periods = window // 2
    dates = log_returns.index
    n = len(log_returns)
    rows = []
    for end in range(window, n + 1, step):
        sub = log_returns.iloc[end - window: end]
        if sub.notna().sum().min() < min_periods:
            rows.append({
                "date": dates[end - 1],
                "d_eff": float("nan"),
                "log_det_term": float("nan"),
            })
            continue
        corr = sub.corr(min_periods=min_periods).fillna(0.0)
        eigs = np.linalg.eigvalsh(corr.values)
        rows.append({
            "date": dates[end - 1],
            "d_eff": d_eff(eigs),
            "log_det_term": log_det_term(corr.values),
        })
    return pd.DataFrame(rows).set_index("date")


def sign_entropy_rate(series: np.ndarray) -> float:
    """Conditional entropy H(sign_t | sign_{t-1}) in bits.

    ≈ 1 bit per day is the canonical weak-form-EMH fingerprint for daily
    log returns: tomorrow's sign is independent of today's. Returns NaN
    on series shorter than 30 observations.
    """
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 30:
        return float("nan")
    signs = np.sign(s).astype(int)
    signs[signs == 0] = 1  # treat zero-return days as positive (rare)
    past = signs[:-1]
    cur = signs[1:]
    # H(cur | past) = H(cur, past) − H(past). Convert nats → bits.
    h_joint = _joint_entropy(past, cur)
    h_past = _shannon_entropy(past)
    return float(max(0.0, (h_joint - h_past)) * LOG2_E)


# ---------------------------------------------------------------------------
# Permutation entropy (G2 — PR #73)
# ---------------------------------------------------------------------------

def permutation_entropy(
    series: np.ndarray,
    embedding_dim: int = 4,
    delay: int = 1,
    *,
    normalize: bool = True,
) -> float:
    """Permutation entropy of a 1-D series.

    For each window of `embedding_dim` consecutive observations (with `delay`
    skips), rank-order the values to get a permutation. Build the histogram
    over all `embedding_dim!` permutations and return the Shannon entropy
    of that histogram in bits.

    Properties:
      * Range: [0, log2(embedding_dim!)] (e.g. [0, log2(24)] for D=4).
      * Normalised version divides by log2(D!) so the result is in [0, 1].
      * Robust to monotone transformations of the series — only the
        ordering matters, not the magnitudes (no binning bias).
      * Established complexity measure for time-series (Bandt & Pompe 2002).

    Returns NaN when the series is too short (< D+1 effective samples).
    Ties broken by NumPy's stable rank (argsort of argsort).
    """
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    if embedding_dim < 2 or delay < 1:
        raise ValueError("embedding_dim must be ≥ 2 and delay must be ≥ 1")
    eff_len = (embedding_dim - 1) * delay + 1
    if n < eff_len + 1:
        return float("nan")

    # Build (N - eff_len + 1) windows, each of size D.
    n_windows = n - eff_len + 1
    # Vectorised indexing: window i = s[i : i + eff_len : delay]
    idx = np.arange(0, eff_len, delay)[None, :] + np.arange(n_windows)[:, None]
    windows = s[idx]  # shape (n_windows, embedding_dim)

    # Encode each window's permutation as a single integer in a positional
    # base-D numeral system. argsort gives the ranks; we hash them to an
    # int by Σ rank_i * D^i. This is faster than tuple-key counting on
    # large windows and avoids hash collisions.
    ranks = np.argsort(np.argsort(windows, axis=1), axis=1)
    powers = embedding_dim ** np.arange(embedding_dim)
    codes = (ranks * powers).sum(axis=1)

    _, counts = np.unique(codes, return_counts=True)
    probs = counts / counts.sum()
    h_bits = float(-(probs * np.log2(probs)).sum())

    if normalize:
        # log2(D!) is the maximum possible entropy on D! patterns.
        from math import factorial
        max_h = float(np.log2(factorial(embedding_dim)))
        return h_bits / max_h if max_h > 0 else 0.0
    return h_bits


def permutation_entropy_per_ticker(
    returns: pd.DataFrame,
    embedding_dim: int = 4,
    delay: int = 1,
) -> pd.DataFrame:
    """Per-ticker permutation entropy on the log-returns panel.

    Returns a DataFrame with columns ``[ticker, permutation_entropy_norm,
    n_observations]``. Tickers with insufficient observations get NaN PE.
    """
    rows: list[dict] = []
    for ticker in returns.columns:
        series = returns[ticker].to_numpy(dtype=float)
        finite = np.isfinite(series)
        n_obs = int(finite.sum())
        pe = permutation_entropy(
            series[finite], embedding_dim=embedding_dim, delay=delay, normalize=True,
        )
        rows.append({
            "ticker": ticker,
            "permutation_entropy_norm": pe,
            "n_observations": n_obs,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Beyond sign-entropy: predictability stylised facts (PR #74)
# ---------------------------------------------------------------------------
# Sign-entropy at lag-1 with 2-state coarse-graining is the weakest possible
# predictability measure. It ignores return magnitude entirely and only looks
# back one day. Three classic financial stylised facts the dashboard was
# missing:
#
#   1. VOLATILITY CLUSTERING — autocorrelation of |returns| persists even
#      when raw returns are nearly uncorrelated. This is the bread-and-butter
#      "GARCH" effect; absent from sign-entropy by construction.
#
#   2. LONG-RANGE MEMORY (Hurst exponent) — single number per series
#      classifying it as persistent (H > 0.55, trending), random-walk
#      (0.45 ≤ H ≤ 0.55), or mean-reverting (H < 0.45).
#
#   3. RAW RETURN AUTOCORRELATION at multiple lags — the direct test of
#      whether yesterday's RETURN MAGNITUDE+SIGN predicts today's. Sign-
#      entropy only captures the sign part of this.

def hurst_rs(
    series: np.ndarray,
    *,
    min_n: int = 20,
    max_n: int = 200,
    n_points: int = 20,
) -> float:
    """Hurst exponent via simple rescaled-range (R/S) analysis.

    For each window size ``n`` in a log-spaced grid [min_n, max_n], split
    the series into non-overlapping chunks of size ``n``, compute the
    rescaled range R/S in each, average across chunks. Fit a power law:
    log(R/S) = H · log(n) + c. Return the slope H.

    Interpretation:
      * H ≈ 0.5: random walk (no long-range memory)
      * H > 0.5: persistent (positive long-range autocorrelation, trending)
      * H < 0.5: anti-persistent (mean-reverting)

    Returns NaN on series too short to fit (< max_n samples) or when the
    R/S fit doesn't have enough points.

    Note: simplified R/S is biased upward for short series. For the BIST
    panel (~1543 days) the bias is small but reported numbers should be
    treated as ordinal — H_A < H_B is more reliable than H_A < 0.5.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < max_n:
        return float("nan")

    ns = np.unique(np.logspace(np.log10(min_n), np.log10(max_n), n_points).astype(int))
    rs_vals: list[tuple[int, float]] = []
    for nn in ns:
        chunks = n // nn
        if chunks < 1:
            continue
        rs_subs: list[float] = []
        for c in range(chunks):
            chunk = x[c * nn:(c + 1) * nn]
            mean = chunk.mean()
            cumdev = (chunk - mean).cumsum()
            r = cumdev.max() - cumdev.min()
            s = chunk.std()
            if s > 0:
                rs_subs.append(r / s)
        if rs_subs:
            rs_vals.append((nn, float(np.mean(rs_subs))))

    if len(rs_vals) < 5:
        return float("nan")
    ns_arr = np.array([v[0] for v in rs_vals], dtype=float)
    rs_arr = np.array([v[1] for v in rs_vals], dtype=float)
    coeffs = np.polyfit(np.log(ns_arr), np.log(rs_arr), 1)
    return float(coeffs[0])


def _autocorr(series: np.ndarray, lag: int) -> float:
    """Standard lag-k Pearson autocorrelation. NaN on series too short
    or with zero variance."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= lag + 1:
        return float("nan")
    x0 = x[:-lag]
    x1 = x[lag:]
    if x0.std() == 0 or x1.std() == 0:
        return float("nan")
    return float(np.corrcoef(x0, x1)[0, 1])


def predictability_diagnostics_per_ticker(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Per-ticker predictability stylised-facts beyond sign-entropy.

    Columns:
      * ``ticker``
      * ``sign_entropy_bits`` — copied here for one-stop comparison
      * ``acf_returns_lag1`` — direct lag-1 return autocorr (small or 0
        under weak EMH; sign-entropy only captures the sign part)
      * ``acf_abs_returns_lag1`` — VOLATILITY CLUSTERING at lag 1
      * ``acf_abs_returns_lag5`` — same at one week
      * ``acf_abs_returns_lag22`` — same at one month (decay rate)
      * ``hurst_exponent`` — long-range memory classification

    For BIST: the typical sign-entropy ≈ 1.0 result conceals that |r| is
    strongly autocorrelated and Hurst > 0.5 for nearly every ticker.
    These three diagnostics surface that.
    """
    rows: list[dict] = []
    for ticker in returns.columns:
        s = returns[ticker].to_numpy(dtype=float)
        rows.append({
            "ticker": ticker,
            "sign_entropy_bits": sign_entropy_rate(s),
            "acf_returns_lag1": _autocorr(s, 1),
            "acf_abs_returns_lag1": _autocorr(np.abs(s), 1),
            "acf_abs_returns_lag5": _autocorr(np.abs(s), 5),
            "acf_abs_returns_lag22": _autocorr(np.abs(s), 22),
            "hurst_exponent": hurst_rs(s),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (G4 — PR #73)
# ---------------------------------------------------------------------------

def bootstrap_mi_excess(
    series_x: np.ndarray,
    series_y: np.ndarray,
    *,
    n_iter: int = 500,
    block_length: int = 5,
    n_bins: int = 4,
    units: str = "bits",
    seed: int = 42,
) -> dict:
    """Circular-block-bootstrap 95% CI for empirical-minus-Gaussian MI.

    Returns ``{point, ci_low, ci_high, n_iter, includes_zero}``. Both X
    and Y are bootstrapped jointly (same block index across them) so the
    pair structure is preserved within blocks.

    Reuses :func:`src.transfer_entropy._circular_block_bootstrap` to stay
    consistent with the surrogate-null methodology used by transfer entropy.
    """
    from src.transfer_entropy import _circular_block_bootstrap

    x = np.asarray(series_x, dtype=float)
    y = np.asarray(series_y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < block_length * 4:
        return {
            "point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "n_iter": 0, "includes_zero": True,
        }

    # Point estimate from the full series.
    rho = float(np.corrcoef(x, y)[0, 1])
    mi_emp_point = pairwise_mi_value(x, y, n_bins=n_bins, units=units)
    mi_gauss_point = gaussian_mi_from_corr(rho, units=units)
    excess_point = mi_emp_point - mi_gauss_point

    # Joint bootstrap: pick block indices once, slice both x and y.
    rng = np.random.default_rng(seed)
    n = x.size
    n_blocks = int(np.ceil(n / block_length))
    samples = np.empty(n_iter)
    for it in range(n_iter):
        starts = rng.integers(0, n, size=n_blocks)
        # Build the per-iter index list once, then slice both x and y.
        block_idx = (starts[:, None] + np.arange(block_length)[None, :]) % n
        idx = block_idx.ravel()[:n]
        x_bs = x[idx]
        y_bs = y[idx]
        try:
            rho_bs = float(np.corrcoef(x_bs, y_bs)[0, 1])
            mi_e = pairwise_mi_value(x_bs, y_bs, n_bins=n_bins, units=units)
            mi_g = gaussian_mi_from_corr(rho_bs, units=units)
            samples[it] = mi_e - mi_g
        except Exception:
            samples[it] = np.nan

    samples = samples[np.isfinite(samples)]
    ci_low = float(np.percentile(samples, 2.5)) if samples.size else float("nan")
    ci_high = float(np.percentile(samples, 97.5)) if samples.size else float("nan")
    return {
        "point": float(excess_point),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_iter": int(samples.size),
        "includes_zero": bool(ci_low <= 0.0 <= ci_high),
    }


# ---------------------------------------------------------------------------
# Top-level pipeline stage
# ---------------------------------------------------------------------------

def _resolve_crisis_dates(
    returns_index: pd.DatetimeIndex,
    crisis_dates: Iterable[Mapping[str, str]],
    calm_window_days: int = 180,
    crisis_window_days: int = 60,
    min_calm_days: int = 30,
) -> list[dict]:
    """For each crisis date, return the calm/crisis index slices that
    bracket it within `returns_index`.

    Skips events whose crisis_date is outside the panel entirely. When
    the configured calm window stretches before panel_start, the calm
    slice is truncated to start at panel_start; if the truncated calm
    slice still has at least `min_calm_days` of room, it is kept.
    """
    out: list[dict] = []
    if returns_index.empty:
        return out
    panel_start = returns_index[0]
    panel_end = returns_index[-1]
    for entry in crisis_dates:
        label = entry["label"]
        date = pd.Timestamp(entry["date"])
        if date <= panel_start or date >= panel_end:
            continue
        ideal_calm_start = date - pd.Timedelta(days=calm_window_days + crisis_window_days)
        calm_end = date - pd.Timedelta(days=crisis_window_days)
        calm_start = max(ideal_calm_start, panel_start)
        if (calm_end - calm_start).days < min_calm_days:
            continue
        crisis_start = date
        crisis_end = min(panel_end, date + pd.Timedelta(days=crisis_window_days))
        out.append({
            "label": label,
            "date": date.isoformat(),
            "calm_start": calm_start.isoformat(),
            "calm_end": calm_end.isoformat(),
            "crisis_start": crisis_start.isoformat(),
            "crisis_end": crisis_end.isoformat(),
        })
    return out


def _ledoit_wolf_shrink(cov: np.ndarray) -> np.ndarray:
    """Ledoit–Wolf shrinkage toward the scaled identity.

    Returns ``(1 − α) Σ + α (tr Σ / k) I`` where α is chosen by the
    sklearn Ledoit-Wolf estimator. Crucial for high-dim regimes where
    sample size T is comparable to or smaller than the dimensionality
    k — the raw sample covariance is then ill-conditioned and its KL
    explodes by orders of magnitude.
    """
    from sklearn.covariance import LedoitWolf

    k = cov.shape[0]
    if k <= 1:
        return cov
    trace_mean = float(np.trace(cov)) / k
    # We need a sample to drive LedoitWolf; sample from N(0, cov) seeded
    # so the shrinkage coefficient is deterministic per cov.
    rng = np.random.default_rng(0)
    # 5× dimensionality of synthetic samples → stable shrinkage estimate
    n = max(5 * k, 200)
    try:
        x = rng.multivariate_normal(np.zeros(k), cov, size=n)
        lw = LedoitWolf().fit(x)
        alpha = float(lw.shrinkage_)
    except (np.linalg.LinAlgError, ValueError):
        alpha = 0.3  # sensible fallback for severely singular cov
    return (1 - alpha) * cov + alpha * trace_mean * np.eye(k)


def _compute_regime_kl(
    log_returns: pd.DataFrame,
    crisis_specs: list[dict],
) -> list[dict]:
    out: list[dict] = []
    for spec in crisis_specs:
        calm = log_returns.loc[spec["calm_start"]:spec["calm_end"]].dropna(axis=1, how="all")
        crisis = log_returns.loc[spec["crisis_start"]:spec["crisis_end"]].dropna(axis=1, how="all")
        common = calm.columns.intersection(crisis.columns)
        if len(common) < 2 or len(calm) < 30 or len(crisis) < 10:
            out.append({**spec, "kl": float("nan"), "n_tickers": len(common)})
            continue
        cov_calm = calm[common].cov(min_periods=10).fillna(0.0).values
        cov_crisis = crisis[common].cov(min_periods=5).fillna(0.0).values
        # Ledoit-Wolf shrinkage tames the high-dim singularity that
        # otherwise lets the inverse-trace term in the closed-form KL
        # blow up to O(10^5) when T_crisis ≪ N_tickers.
        cov_calm = _ledoit_wolf_shrink(cov_calm)
        cov_crisis = _ledoit_wolf_shrink(cov_crisis)
        kl = kl_gaussian_covariances(cov_calm, cov_crisis)
        out.append({**spec, "kl": kl, "n_tickers": len(common)})
    return out


DEFAULT_CRISIS_DATES = (
    {"label": "COVID-19 crash", "date": "2020-03-11"},
    {"label": "Ukraine invasion", "date": "2022-02-24"},
    {"label": "Türkiye earthquake", "date": "2023-02-06"},
)


def run_info_theory(config: PipelineConfig) -> None:
    """Compute and persist the Information-Theory layer for one market."""
    logger.info("=== Information Theory ===")
    config.data_results.mkdir(parents=True, exist_ok=True)

    returns = pd.read_parquet(config.data_processed / "log_returns.parquet")
    if returns.empty:
        logger.warning("Info-theory: empty returns panel; skipping")
        return

    # Pearson correlation (precomputed by run_analysis); fall back to recomputing
    # if missing so this stage can be re-run independently.
    pearson_path = config.data_results / "pearson_corr.parquet"
    if pearson_path.exists():
        corr = pd.read_parquet(pearson_path)
        common = returns.columns.intersection(corr.columns)
        returns = returns[common]
        corr = corr.loc[common, common]
    else:
        corr = returns.corr(min_periods=config.analysis.corr_min_periods)
        corr = corr.fillna(0.0)

    n_bins = config.transfer_entropy.n_bins + 1  # 4 bins by default (TE uses 3)

    # 1. Pairwise MI matrix (plug-in) + Gaussian baseline + nonlinear excess
    mi = pairwise_mi_matrix(returns, n_bins=n_bins, units="bits")
    mi_g = gaussian_mi_matrix(corr, units="bits")
    excess = nonlinear_excess(mi, mi_g)
    top_excess = top_nonlinear_pairs(excess, top_k=14)

    mi.to_parquet(config.data_results / "mi_matrix.parquet")
    mi_g.to_parquet(config.data_results / "mi_gaussian_matrix.parquet")
    excess.to_parquet(config.data_results / "mi_nonlinear_excess.parquet")
    top_excess.to_csv(
        config.data_results / "mi_nonlinear_excess_top.csv", index=False
    )

    # 2. Eigen-spectrum measures (lifted from pearson_corr)
    eigs = np.linalg.eigvalsh(corr.values)
    d_eff_val = d_eff(eigs)
    dh_val = log_det_term(corr.values)
    joint_h_nats = joint_diff_entropy(corr.values)

    # 3. Rolling D_eff and ΔH(t)
    window = 60
    step = 5
    rolling = rolling_d_eff_dh(returns, window=window, step=step)
    rolling.to_parquet(config.data_results / "rolling_info_theory.parquet")

    # 4. Crisis-window KL divergences. Only meaningful for finance universes
    # with a DatetimeIndex (EEG/sample-indexed panels skip this entirely).
    if isinstance(returns.index, pd.DatetimeIndex):
        crisis_specs = _resolve_crisis_dates(returns.index, DEFAULT_CRISIS_DATES)
        crisis_kls = _compute_regime_kl(returns, crisis_specs)
    else:
        crisis_kls = []
    with open(config.data_results / "regime_kl.json", "w") as f:
        json.dump(crisis_kls, f, indent=2)

    # 5. Mean entropy rate of return signs (weak-form-EMH fingerprint)
    sign_h = pd.Series(
        {ticker: sign_entropy_rate(returns[ticker].values) for ticker in returns.columns}
    )
    sign_h_mean = float(sign_h.dropna().mean()) if sign_h.notna().any() else float("nan")
    sign_h.to_csv(
        config.data_results / "entropy_rate_signs.csv",
        header=["entropy_rate_bits"],
        index_label="ticker",
    )

    # 6. Permutation entropy per ticker (PR #73 — G2)
    # Complementary to sign-entropy: captures 4-bar ordinal patterns that
    # sign-entropy (2-state, lag-1 only) throws away.
    pe_df = permutation_entropy_per_ticker(returns, embedding_dim=4, delay=1)
    pe_df.to_csv(
        config.data_results / "permutation_entropy.csv",
        index=False,
    )
    pe_mean = float(pe_df["permutation_entropy_norm"].dropna().mean()) if pe_df["permutation_entropy_norm"].notna().any() else float("nan")

    # 7. Predictability diagnostics beyond sign-entropy (PR #74)
    # Sign-entropy at lag-1 with 2 states is misleading: it can read ≈1.0
    # while |returns| autocorrelation says ≈0.3 (volatility clustering)
    # and Hurst says ≈0.6 (persistent trending). These three measures
    # together give an honest picture of what's predictable about each
    # ticker's return process.
    predict_df = predictability_diagnostics_per_ticker(returns)
    predict_df.to_csv(
        config.data_results / "predictability_diagnostics.csv",
        index=False,
    )
    vol_cluster_pct = float(
        (predict_df["acf_abs_returns_lag1"].dropna() > 0.20).mean()
    ) if predict_df["acf_abs_returns_lag1"].notna().any() else float("nan")
    hurst_persistent_pct = float(
        (predict_df["hurst_exponent"].dropna() > 0.55).mean()
    ) if predict_df["hurst_exponent"].notna().any() else float("nan")

    summary = {
        "n_tickers": int(len(returns.columns)),
        "n_observations": int(len(returns)),
        "d_eff": d_eff_val,
        "log_det_term": dh_val,
        "joint_gaussian_entropy_nats": joint_h_nats,
        "mean_sign_entropy_rate_bits": sign_h_mean,
        "mean_permutation_entropy_norm": pe_mean,
        "permutation_entropy_embedding_dim": 4,
        "permutation_entropy_delay": 1,
        "frac_tickers_with_volatility_clustering": vol_cluster_pct,
        "frac_tickers_persistent_hurst": hurst_persistent_pct,
        "mi_bits_units": True,
        "n_bins": int(n_bins),
        "rolling_window": int(window),
        "rolling_step": int(step),
        "top_nonlinear_pairs": top_excess.to_dict(orient="records"),
        "regime_kl_nats": crisis_kls,
    }
    with open(config.data_results / "it_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Information theory: D_eff=%.2f, ΔH=%.3f nats, mean sign-entropy=%.3f bits, "
        "%d crisis KLs, top nonlinear excess=%.4f bits",
        d_eff_val,
        dh_val,
        sign_h_mean,
        sum(1 for r in crisis_kls if np.isfinite(r.get("kl", float("nan")))),
        float(top_excess["nonlinear_excess"].iloc[0]) if not top_excess.empty else float("nan"),
    )
