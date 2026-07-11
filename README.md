# Black-Scholes Options Pricer

An interactive web app for pricing European options using the Black-Scholes model, built with Python and Streamlit.

## What it does

- Calculates **call and put prices** using the Black-Scholes analytical formula
- Computes all **5 Greeks**: Delta, Gamma, Theta, Vega, Rho
- Verifies **put-call parity** in real time
- Prices options using **Monte Carlo simulation** (up to 50,000 paths) and shows convergence to the analytical price
- Fetches **live market data** for any stock ticker and compares theoretical prices against real market prices
- Visualises Delta, Vega and Theta across different stock prices and expiries

## How to run

```bash
pip install streamlit numpy scipy matplotlib yfinance
streamlit run app.py
```

## Tech stack

Python · Streamlit · NumPy · SciPy · Matplotlib · yfinance

## Key findings

- Monte Carlo converges to the Black-Scholes price as simulations increase
- Real market prices deviate systematically from the model across strikes, consistent with the volatility smile — a known limitation of Black-Scholes

## Parameters

| Parameter | Description |
|---|---|
| S | Current stock price |
| K | Strike price |
| T | Time to expiry (years) |
| r | Risk-free interest rate |
| σ | Volatility (annualised) |