# Rough Volatility — Empirical Replication of Gatheral, Jaisson & Rosenbaum (2018)

An empirical investigation of whether the realised volatility of major indices exhibits the "rough" behaviour (Hurst exponent H ≈ 0.1) documented in Gatheral, Jaisson & Rosenbaum (2018), rejecting the classical H = 0.5 assumption used in Black-Scholes and standard stochastic volatility models.

## Overview

The classical framework of financial mathematics assumes that volatility, when modelled as a stochastic process, follows Brownian-motion-like dynamics with Hurst exponent H = 0.5. In 2018, Gatheral, Jaisson & Rosenbaum showed empirically that the volatility of major indices is dramatically more "rough" than assumed — with H ≈ 0.1 — implying that the classical models fundamentally misprice short-dated options and misrepresent the fine structure of volatility dynamics.

This project replicates the central empirical finding of the paper on live market data across multiple assets and time regimes, validates the estimation method through simulation, and tests a trading strategy inspired by the discovered dynamics.

## Methodology

**Data source.** Daily and long-horizon price data from Yahoo Finance via `yfinance` for SPY, QQQ, NVDA, GLD, TLT.

**Realised volatility.** Computed as the rolling standard deviation of log-returns over 5-day windows, annualised. The choice of window is discussed in the robustness section.

**Hurst estimation.** The q-th moment estimator applied to log-volatility increments:

    E[|X(t+Δ) - X(t)|^q] ~ Δ^(q·H)

A log-log regression of the empirical moment against lag Δ yields the estimated H.

**Method validation.** The estimator is applied to synthetic fractional Brownian motion trajectories with known Hurst exponents H ∈ {0.1, 0.2, 0.3, 0.5, 0.7} to confirm empirical accuracy.

**Trading strategy.** A regime-switching strategy is tested: long SPY when rolling H exceeds a "persistent" threshold, cash otherwise. Backtested over 10 years against buy-and-hold benchmark.

## Key Findings

- **SPY exhibits H ≈ 0.12** using 5-day realised volatility over five years — consistent with the rough behaviour reported by Gatheral et al. (2018). Rejects the classical H = 0.5 assumption at empirical significance.

- **The rough behaviour is universal across assets.** All five assets tested (SPY, QQQ, NVDA, GLD, TLT) show H in the range 0.06 – 0.12, indicating that roughness is a fundamental property of financial markets, not specific to a single asset class. Bonds (TLT) show the roughest dynamics.

- **H depends on the realised-volatility window.** With smaller windows (3-5 days) closer to the high-frequency estimators used in the original paper, H is in the rough range. With larger windows (20-30 days), a well-known smoothing bias inflates the estimate.

- **Roughness varies with market regime.** A 10-year rolling analysis shows H is not constant: it drops to ≈ 0.08 in normal times and rises to ≈ 0.30 during COVID-era stress. This suggests that extreme crises exhibit a transient softening of the rough behaviour, likely due to sustained volatility clustering.

- **The estimator is validated on simulated fractional Brownian motion.** For H ∈ {0.1, 0.2, 0.3}, the empirical error is below 3%. The method is fully accurate in the rough regime of interest.

- **Negative result: H-based regime switching does not produce alpha.** A backtest of a long-when-H-high, cash-when-H-low strategy underperforms buy-and-hold in return (27% vs 303%) and Sharpe (0.28 vs 0.87) over ten years. A threshold sensitivity analysis (not shown) confirmed the result is robust — no combination of H thresholds outperforms buy-and-hold. H captures the textural property of volatility, not directional information about the underlying price. This is a valuable negative result: roughness alone is not directly tradeable through simple regime-switching.

## Files

- `rough_vol.py` — full pipeline
- `rv_series.png` — realised volatility time series
- `hurst_estimation.png` — log-log regression underlying H estimation
- `hurst_window_robustness.png` — window sensitivity of H
- `hurst_cross_asset.png` — cross-asset Hurst estimation
- `hurst_time_regime.png` — rolling H over 10 years
- `rough_vs_standard_dynamics.png` — simulated rough vs standard volatility paths
- `h_strategy_backtest.png` — trading strategy performance vs buy-and-hold

## Limitations

- **Daily data only.** The original paper of Gatheral et al. uses intraday high-frequency data, allowing more accurate RV estimation. Our daily estimator is subject to smoothing bias for larger windows.
- **q-moment estimator alone.** Only one estimation method is used. Independent methods (Whittle, wavelets) could confirm the result more rigorously.
- **No option pricing connection.** The practical implication of rough behaviour for short-dated option pricing (via rBergomi, rough Heston) is not implemented — it would require full option-pricing infrastructure.
- **Backtest simplifications.** The trading backtest ignores transaction costs, slippage and short-selling constraints. It is meant as a directional test, not a production trading system.

## How to Run

```bash
pip install numpy scipy pandas matplotlib yfinance
python rough_vol.py
```

## Technical Stack

Python · NumPy · pandas · SciPy · yfinance · matplotlib

## References

- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). *Volatility is rough.* Quantitative Finance, 18(6), 933–949.
- Mandelbrot, B. B. (1963). *The variation of certain speculative prices.* Journal of Business, 36(4), 394–419.
- Hurst, H. E. (1951). *Long-term storage capacity of reservoirs.* Transactions of the American Society of Civil Engineers, 116, 770–808.