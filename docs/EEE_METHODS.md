# EEE_METHODS — RMT, Glasso, Wavelet, Transfer Entropy

> **"EEE" is an informal grouping label.** It bundles the more advanced
> methods (RMT denoising, Graphical LASSO, Wavelet decomposition, Transfer
> Entropy) into a single dashboard sub-tab and a labelled block in
> `run_pipeline.py`. It is **not** an acronym. Don't try to decode it.

---

## 1. RMT denoising — `src/rmt_denoising.py`

### Theory

Empirical correlation matrices computed from finite samples are noisy. Random
Matrix Theory tells us that if `X` is a `T × N` matrix of i.i.d. standard
normals, the eigenvalues of `(1/T) X^T X` distribute according to the
**Marchenko–Pastur** law as `T, N → ∞` with `q = T/N` fixed. The MP
distribution has compact support `[λ-, λ+]` where:

```
λ± = σ² ( 1 + 1/q ± 2 √(1/q) )
```

For a normalised correlation matrix, `σ² = 1`. Eigenvalues falling **inside**
`[λ-, λ+]` are statistically indistinguishable from noise — they reflect the
finite-sample variance of an underlying uncorrelated structure. Eigenvalues
**above** `λ+` carry signal: market-wide, sector, or pair-specific
co-movement.

### Implementation

`denoise_correlation` (`rmt_denoising.py:47`):
1. Symmetrise and clean the empirical correlation; fill diagonal with 1.
2. Eigendecompose with `np.linalg.eigh`.
3. Compute `λ±` from `T` (returns rows) and `N` (tickers).
4. Mark eigenvalues `> λ+` as signal.
5. Replace each noise eigenvalue with the **mean of all noise eigenvalues**
   (`method='constant'`). Trace is preserved.
6. Reconstruct `Q Λ_denoised Q^T`, force diagonal=1, clip to `[-1, 1]`.

### Hardcoded params

- `method='constant'` — alternative `'zero'` exists but is not selected from YAML.
- NaN correlations filled with 0 before decomposition. This biases
  ill-defined entries toward zero, which is benign for downstream MST but
  worth knowing.

### Caveats

- MP assumes i.i.d. Gaussian residuals. Returns are heavy-tailed; the MP
  bound is therefore approximate.
- The denoised matrix is **not guaranteed positive semi-definite** after
  diagonal renormalisation and clipping; numerical instabilities can leak
  in. Treat it as a regularised estimator, not a probabilistic model.
- Effective sample size `T` after the coverage filter ≠ raw days; we use
  `len(returns)` directly.

### What to verify or improve

- Compare `method='constant'` vs `'zero'` empirically (FUTURE_WORK F-2).
- Bootstrap the MP bound: instead of using analytic `λ+`, simulate the null
  distribution of the largest noise eigenvalue under the empirical marginal
  distributions.

### References

- Marčenko, V. A. & Pastur, L. A. (1967). *Distribution of eigenvalues for
  some sets of random matrices.* Math USSR-Sb, 1(4):457.
- Laloux, L. et al. (1999). *Noise dressing of financial correlation
  matrices.* Phys. Rev. Lett. 83, 1467.
- Plerou, V. et al. (2002). *Random matrix approach to cross correlations
  in financial data.* Phys. Rev. E 65, 066126.

---

## 2. Graphical LASSO — `src/partial_correlation.py`

### Theory

Under joint Gaussianity of returns, the **precision matrix** `Θ = Σ⁻¹` has
the property that `Θ_ij = 0` ⇔ `i ⫫ j | rest`. So a sparse precision
matrix expresses a sparse conditional-independence graph: edges are direct
dependencies that can't be explained by mediating variables.

The Graphical LASSO (Banerjee et al. 2008; Friedman et al. 2008) estimates
`Θ` by penalised maximum likelihood:

```
Θ̂ = argmax_Θ  log det Θ - tr(S Θ) - α ‖Θ‖₁,off
```

where `S` is the empirical covariance and `α > 0` controls sparsity. The L1
penalty drives off-diagonal entries to exactly zero.

Partial correlations follow:

```
ρ_ij·rest = -Θ_ij / √(Θ_ii Θ_jj)
```

### Implementation

`fit_graphical_lasso` (`partial_correlation.py:24`):
- Drop full-row NaN observations (full dropna).
- `GraphicalLassoCV(max_iter=200, cv=5)` selects α via 5-fold CV unless an
  explicit `alpha` is passed.
- The pipeline saves both the precision matrix
  (`precision_matrix.parquet`, added this session) and the partial
  correlation derived matrix.

### Hardcoded params

- `cv=5`, `max_iter=200`, edge-extraction `threshold=0.01`.

### Caveats

- Returns aren't Gaussian. The estimator still works as a regulariser but
  the conditional-independence interpretation weakens.
- `dropna()` (full row) discards observations conservatively; a
  pairwise-complete approach would retain more data but breaks the
  positive-definiteness assumption sklearn relies on.
- Sample size needed: `T ≫ N`. With ~73 surviving tickers and ~1500 days
  this is comfortable.

### What to verify or improve

- Compare CV-selected α against StARS or BIC selection.
- Investigate whether centering / standardising before fit changes results.

### References

- Friedman, J., Hastie, T., Tibshirani, R. (2008). *Sparse inverse
  covariance estimation with the graphical lasso.* Biostatistics 9(3):432.
- Banerjee, O. et al. (2008). *Model selection through sparse maximum
  likelihood estimation for multivariate Gaussian or binary data.* JMLR
  9:485.

---

## 3. Wavelet multi-scale correlation — `src/wavelet_analysis.py`

### Theory

The Discrete Wavelet Transform (DWT) decomposes a signal `x_t` into
approximation and detail coefficients across dyadic frequency bands. The
detail coefficients at level `ℓ` capture variation in the band `2^ℓ` to
`2^(ℓ+1)` time units. For daily returns this maps roughly to:

| Level | Band | Cycle interpretation |
|---|---|---|
| 1 | 2–4 days | Short-term noise / microstructure |
| 2 | 4–8 days | Weekly cycles |
| 3 | 8–16 days | Bi-weekly to monthly |
| 4 | 16–32 days | Monthly |
| 5 | 32–64 days | Quarterly |
| 6 | 64–128 days | Semi-annual |
| 7 | 128–256 days | Annual |

By computing correlation matrices on the detail coefficients, we see how
co-movement structure varies with timescale.

### Implementation

`wavelet_decompose` (`wavelet_analysis.py:42`):
- For each level `ℓ`, run `pywt.wavedec(series, 'db4', level=ℓ)`.
- Reconstruct **only** the detail at level `ℓ` by zeroing all other coefficient
  arrays and calling `pywt.waverec`. Truncate to original length.
- Stack reconstructed series into a `(T × N)` DataFrame per scale.

Correlation + MST per scale uses the same code as the main analysis (see
`compute_wavelet_scale_mst`).

### Hardcoded params

- `wavelet='db4'` (Daubechies-4).
- `max_level = min(pywt.dwt_max_level(T, 'db4'), 7)`.
- NaN replaced with 0 before DWT.

### Caveats

- DWT requires equal-length, NaN-free signals; the zero-fill is benign for
  short windows of NaN but biases scale-1 detail near long gaps.
- Edge effects: `db4` has support length 8, so the first/last ~8 coefficients
  per scale are influenced by boundary conditions.
- `db4` is a default choice — alternatives (`sym8`, `haar`) would yield
  different scale resolutions / oscillation characteristics. Not currently
  surfaced in YAML.

### What to verify or improve

- Try `sym8` for better symmetry around peaks.
- Use Maximal Overlap DWT (MODWT) instead of regular DWT — it's
  shift-invariant and avoids the dyadic-length restriction. PyWavelets
  supports `swt`.

### References

- Daubechies, I. (1992). *Ten lectures on wavelets.* SIAM.
- Percival, D. B. & Walden, A. T. (2000). *Wavelet methods for time series
  analysis.* Cambridge.
- Gençay, R. et al. (2001). *Asymmetry of information flow between volatilities
  across time scales.* Quantitative Finance.

---

## 4. Transfer entropy — `src/transfer_entropy.py`

### Theory

For two stationary time series `X` and `Y`, the **transfer entropy** from
`X` to `Y` at lag `k` is:

```
TE(X→Y) = H(Y_t | Y_{t-k}) - H(Y_t | Y_{t-k}, X_{t-k})
```

It quantifies the reduction in uncertainty about `Y_t` provided by `X_{t-k}`
beyond what `Y_{t-k}` already supplies. Non-negative; zero iff `X` and `Y`
are conditionally independent given `Y`'s own past. Asymmetric:
`TE(X→Y) ≠ TE(Y→X)` in general.

We compute it as a **discrete** (binned) estimator:

```
TE = H(Y_t, Y_lag) - H(Y_t, Y_lag, X_lag) - H(Y_lag) + H(Y_lag, X_lag)
```

with `H(·)` Shannon entropy estimated by plug-in from joint histograms.

### Implementation

- Per series: equal-frequency binning into `n_bins=3` (terciles).
- Per pair: build joint distributions via `np.unique` on stacked columns,
  estimate plug-in entropies, combine into TE.
- Significance: shuffle `X` source `significance_shuffles=100` times,
  compute null distribution of TE under H₀ of no information transfer.
  P-value = fraction of nulls ≥ observed.
- Insignificant entries (p ≥ 0.05) zeroed.
- Net TE matrix: `net[i,j] = TE(i→j) - TE(j→i)`. Positive = `i` leads `j`.

The shuffle RNG is seeded via `config.transfer_entropy.seed` (default 42)
since this session — the previous code used global `np.random.permutation`
which made results irreproducible.

### Hardcoded params

- Discretisation strategy: equal-frequency, fixed at 3 bins.
- Significance test: shuffle source only (not target) — this gives a
  surrogate that preserves the marginal distribution of `X` but breaks any
  cross-dependence.

### Caveats

- **Discrete TE has bias proportional to `1/N`** where `N` is the number of
  observations per bin. With `T ≈ 1500` and 3 bins, joint distributions
  have ~167 observations per `(Y_t, Y_lag, X_lag)` triple — adequate but
  not generous.
- **The shuffle null is too easy**. It tests `X ⊥ Y | Y_lag`, but a
  permutation of `X` also breaks autocorrelation in `X`. A better null is
  a **temporal surrogate** that preserves `X`'s autocorrelation (e.g.
  IAAFT). Filed as MED-severity in KNOWN_ISSUES.
- TE is sensitive to lag choice. We use `lag=1`. Multi-lag analysis would
  catch slower cross-correlations.

### What to verify or improve

- Replace permutation null with IAAFT or block bootstrap.
- Sweep `lag ∈ {1, 5, 22}` to expose multi-scale information flow.
- Compare with kernel-based estimators or KSG estimator for robustness.

### References

- Schreiber, T. (2000). *Measuring information transfer.* Phys. Rev. Lett.
  85, 461.
- Marschinski, R. & Kantz, H. (2002). *Analysing the information flow
  between financial time series.* Eur. Phys. J. B 30, 275.
- Bossomaier, T. et al. (2016). *An introduction to transfer entropy.*
  Springer.

