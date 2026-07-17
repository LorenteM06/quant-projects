# Optimal Execution: Almgren-Chriss Model with Cross-Asset Analysis

A full implementation and quantitative analysis of the Almgren-Chriss (2000) optimal execution framework, extended with cross-asset comparison, historical backtesting, and intraday sensitivity analysis on live market data.

## Overview

The Almgren-Chriss model formalises the fundamental trade-off in institutional trade execution: selling faster increases market impact costs, while selling slower increases exposure to price volatility. The model derives an analytical closed-form solution — a family of hyperbolic-cosine trajectories parametrised by risk aversion λ.

This project moves beyond textbook implementation to answer concrete questions relevant to practitioners:

1. When does the optimal strategy meaningfully differ from naive TWAP or VWAP?
2. How does the optimal execution differ across assets of different liquidity?
3. Does the model's theoretical advantage hold up when backtested on real historical prices?
4. How does the optimal cost vary with time of day?

## Methodology

**Model.** Price dynamics with permanent impact γ·v(t) and temporary impact η·v(t). Objective: minimise E[C] + λ·Var[C], where C is the total execution cost. Solved analytically:

    x(t) = X₀ · sinh(κ(T-t)) / sinh(κT),   κ = √(λσ²/η)

Implemented in a numerically stable form to handle large κ·T without overflow.

**Parameter estimation.** For each asset, σ is estimated from historical returns; η is estimated from the average bid-ask spread and daily volume (η ≈ spread / (2·ADV)); γ ≈ η/10 as a standard approximation.

**Cross-asset study.** Four assets spanning the liquidity spectrum: SPY, QQQ, NVDA, RIVN.

**Backtest.** 100 random 5-hour windows sampled from 30 days of SPY 15-min bars. Costs of TWAP and AC simulated against actual price paths.

**Intraday analysis.** 60 days of 15-min SPY data grouped by time of day. Parameters re-estimated per intraday bin (η varies significantly with volume).

## Key Findings

- **The value of Almgren-Chriss is risk control, not expected cost reduction.** In the SPY backtest, AC produced 6× lower cost standard deviation than TWAP ($231k vs $1.43M) despite similar expected costs. Institutional users care about predictability, not just average cost.

- **For liquid assets, TWAP is nearly optimal.** SPY, QQQ, and NVDA require λ ~ 10⁻¹² for AC to meaningfully deviate from TWAP — well below any realistic institutional aversion. The model adds most value on illiquid names.

- **Critical urgency differs by orders of magnitude across assets.** RIVN's critical λ (1.3×10⁻⁹) is ~200× that of SPY (9.2×10⁻¹²). Small-caps require far more aggressive scheduling than TWAP even at moderate risk aversion.

- **Time of day matters enormously.** Executing 100k SPY shares at the closing hour (15:30) costs 229% less than executing at midday (12:30), consistent with well-known intraday liquidity patterns. The closing auction concentrates volume and depresses temporary impact.

- **Institutional benchmarks (TWAP, VWAP) are structurally similar for liquid ETFs.** VWAP and TWAP produce nearly identical distributions because SPY's intraday volume pattern is relatively flat outside open/close.

## Practical Recommendations

The analysis yields concrete execution rules for a first-year quant trader:

| Situation | Recommendation |
|---|---|
| Large SPY/QQQ/NVDA order | Use TWAP or wait for closing auction |
| Small-cap execution (>1% ADV) | Use Almgren-Chriss with λ ≥ 10⁻⁹ |
| Risk-averse client (predictability matters) | Always Almgren-Chriss, never TWAP |
| No intraday timing constraint | Execute in the last 30 minutes of trading |
| Avoid at all costs | Executing large orders at midday (12:00–13:00) |

## Files

- `execution.py` — full pipeline (model, benchmarks, backtest, intraday)
- `lambda_sensitivity.png` — optimal trajectories across risk aversion levels
- `efficient_frontier.png` — cost/variance trade-off curve
- `vwap_comparison.png` — TWAP vs VWAP vs Almgren-Chriss
- `cross_asset.png` — trajectories and urgency across four assets
- `backtest_distribution.png` — TWAP vs AC cost distributions over 100 historical windows
- `intraday_analysis.png` — execution cost and liquidity profile by time of day

## Limitations

- **Linear impact assumption.** Almgren-Chriss assumes both temporary and permanent impact scale linearly with trading velocity. In practice, impact is empirically closer to a square-root function of order size (Torre 1997, Grinold-Kahn 2000). Production implementations typically use square-root impact models.
- **No interior optimal horizon.** For the liquid assets tested, the objective E[C] + λ·Var[C] is monotonically decreasing in T at all realistic λ. An interior optimum would emerge only with orders >10% of ADV, higher-volatility regimes, or enforced end-of-day constraints.
- **Constant parameters.** σ and η are treated as constant within each execution window. In reality both are intraday-varying — a limitation partially addressed by the intraday analysis in Section 8.
- **No adverse selection.** The model assumes a passive schedule that ignores order book dynamics and information leakage. Extensions such as Obizhaeva-Wang (2013) address this.

## How to Run

```bash
pip install numpy scipy pandas matplotlib yfinance
python execution.py
```

## Technical Stack

Python · NumPy · pandas · SciPy · yfinance · matplotlib

## References

- Almgren, R., & Chriss, N. (2000). *Optimal execution of portfolio transactions.* Journal of Risk, 3(2), 5–39.
- Obizhaeva, A. A., & Wang, J. (2013). *Optimal trading strategy and supply/demand dynamics.* Journal of Financial Markets, 16(1), 1–32.