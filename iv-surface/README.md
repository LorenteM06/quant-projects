# SPY Volatility Surface Analysis with SVI Parametrization

Quantitative analysis of the SPY implied volatility surface using the Stochastic Volatility Inspired (SVI) parametrization introduced by Gatheral (2004). The project constructs, calibrates and diagnoses the surface, then applies it to detect mispriced options in live market data, with cross-ticker comparison against QQQ.

## Overview

For six selected expiries spanning from 1 week to 1 year, implied volatilities are extracted from market prices via Black-Scholes inversion, then fitted with the SVI parametric form. The calibrated surface enables four concrete outputs:

1. Term structure analysis of skew (ρ) and ATM volatility
2. Rigorous butterfly-arbitrage diagnostics per maturity (Gatheral 2014)
3. Automated detection of options trading far from the model, filtered by liquidity criteria
4. Cross-ticker comparison against QQQ to contextualise structural differences

## Methodology

**Data source.** Live option chain from Yahoo Finance via `yfinance`. Six maturities selected to approximate 1w, 1m, 2m, 3m, 6m and 1y.

**Implied volatility extraction.** Black-Scholes is inverted numerically using Brent's method.

**SVI fitting.** For each expiry, total variance w = σ²·T is fitted against log-moneyness k = ln(K/S):

    w(k) = a + b · (ρ(k - m) + √((k - m)² + σ²))

Parameters estimated by non-linear least squares with bounds.

**Model diagnostics.** RMSE(IV) and butterfly-arbitrage check via Gatheral (2014) equation (2.2).

**Mispricing detection.** Live quotes compared against SVI-implied volatilities. Signals filtered by volume (> 10) and bid-ask spread (< 5% of price).

**Cross-ticker comparison.** Full pipeline applied to QQQ; ρ and ATM volatility compared across maturities.

## Key Findings

- **Persistent negative skew across all maturities** — OTM puts trade at higher implied volatilities than OTM calls.
- **Skew flattens with maturity** — a well-documented stylised fact.
- **Butterfly arbitrage detected at short maturities** — standard SVI produces g(k) < 0 for expiries under one month, consistent with Gatheral & Jacquier (2014).
- **RMSE(IV) ranges from 0.7% to 1.5%** — comparable to production-grade calibration.
- **Cross-ticker asymmetry** — QQQ exhibits higher ATM volatility (sector concentration in tech) while SPY shows more pronounced skew (systemic hedging demand from institutional flows).
- **Actionable mispricing signals detected** — after liquidity filtering, small number of options trade materially outside the SVI curve, exported to `signals.csv`.

## Limitations

- **Snapshot analysis.** The pipeline captures the surface at a single point in time. Detected mispricings are not backtested — persistence and mean-reversion of signals require multi-day sampling.
- **Standard SVI, not SSVI.** The classical parametrization violates butterfly arbitrage at short maturities. Production settings would use Surface SVI (SSVI) or Gatheral-Jacquier arbitrage-free formulations.
- **Yahoo Finance data quality.** Bid-ask spreads and volumes are proxies for true liquidity; some quotes may be stale, particularly for far-OTM strikes.
- **Constant risk-free rate.** A single r = 5% is used across all maturities. A proper implementation would use the treasury curve.
- **No dividend adjustment.** SPY dividends are ignored; more accurate pricing would incorporate the projected dividend yield.

## Files

- `iv_surface.py` — full pipeline
- `iv_surface_3d.png` — 3D implied volatility surface
- `iv_heatmap.png` — 2D heatmap
- `term_structure.png` — SVI ρ and ATM volatility vs maturity
- `mispricings_map.png` — detected dislocations in (strike, time) space
- `spy_vs_qqq.png` — cross-ticker comparison
- `signals.csv` — quality-filtered mispricing signals

## How to Run

```bash
pip install numpy scipy matplotlib yfinance pandas
python iv_surface.py
```

## Technical Stack

Python · NumPy · SciPy (optimize, stats) · pandas · yfinance · matplotlib

## References

- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility parameterization.* Global Derivatives.
- Gatheral, J., & Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.* Quantitative Finance, 14(1), 59-71.