# SPY Volatility Surface Analysis with SVI Parametrization

Full quantitative analysis of the SPY implied volatility surface using the Stochastic Volatility Inspired (SVI) parametrization introduced by Gatheral (2004).

## Overview

This project constructs and analyses the implied volatility surface of SPY (S&P 500 ETF) using live option market data. For each of six selected expiries spanning from 1 week to 1 year, implied volatilities are extracted from market prices via Black-Scholes inversion, then fitted with the SVI parametric form. The resulting surface enables interpolation of arbitrary strike/maturity combinations and reveals structural properties of the market's volatility expectations.

## Methodology

**Data source.** Live option chain from Yahoo Finance via `yfinance`. Six maturities are selected to approximate 1w, 1m, 2m, 3m, 6m and 1y.

**Implied volatility extraction.** For each call option, Black-Scholes is inverted numerically using Brent's method to find the volatility that reproduces the observed market price.

**SVI fitting.** For each expiry, total variance w = σ²·T is fitted against log-moneyness k = ln(K/S) using the SVI parametric form:

    w(k) = a + b · (ρ(k - m) + √((k - m)² + σ²))

Five parameters (a, b, ρ, m, σ) are estimated by non-linear least squares minimisation with bounds.

**Surface construction.** Individual SVI fits are combined into a continuous surface evaluated on a grid of log-moneyness and time to expiry.

## Key Findings

- **Persistent negative skew across maturities.** The ρ parameter is systematically negative, confirming the well-known equity index volatility skew: out-of-the-money puts trade at higher implied volatilities than out-of-the-money calls.

- **Term structure of skew flattens with maturity.** Short-dated smiles (1-2 weeks) exhibit pronounced curvature that gradually flattens for longer expiries — a stylised fact widely documented in the volatility literature.

- **ATM volatility term structure is non-monotonic.** The at-the-money implied volatility rises through medium maturities but decreases at long horizons, consistent with either upcoming event risk or reduced liquidity in far-dated options.

## Files

- `iv_surface.py` — full pipeline (data download, IV extraction, SVI fitting, visualisation)
- `smiles_all.png` — volatility smiles for all expiries with SVI fits overlaid
- `iv_surface_3d.png` — full 3D implied volatility surface
- `term_structure.png` — SVI ρ and ATM volatility as functions of maturity

## How to Run

```bash
pip install numpy scipy matplotlib yfinance
python iv_surface.py
```

## Technical Stack

Python · NumPy · SciPy (optimize, stats) · yfinance · matplotlib

## References

- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives.* Presentation at Global Derivatives.
- Gatheral, J., & Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.* Quantitative Finance, 14(1), 59-71.