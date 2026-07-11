import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def delta_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

def gamma(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def theta_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365

def rho_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * T * np.exp(-r*T) * norm.cdf(d2) / 100

st.title("Black-Scholes Options Pricer")
st.markdown("Interactive options pricing model with Greeks, Monte Carlo simulation and real market data.")

st.sidebar.header("Parameters")
S = st.sidebar.slider("Stock Price (S)", 50, 500, 100)
K = st.sidebar.slider("Strike Price (K)", 50, 500, 100)
T = st.sidebar.slider("Time to Expiry (years)", 0.1, 3.0, 1.0)
r = st.sidebar.slider("Risk-free Rate", 0.0, 0.1, 0.05)
sigma = st.sidebar.slider("Volatility", 0.05, 0.8, 0.2)

call = bs_call(S, K, T, r, sigma)
put  = bs_put(S, K, T, r, sigma)

st.subheader("Option Prices")
col1, col2 = st.columns(2)
col1.metric("Call Price", f"${call:.4f}")
col2.metric("Put Price",  f"${put:.4f}")

st.subheader("Greeks")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Delta", f"{delta_call(S, K, T, r, sigma):.4f}")
col2.metric("Gamma", f"{gamma(S, K, T, r, sigma):.4f}")
col3.metric("Vega",  f"{vega(S, K, T, r, sigma):.4f}")
col4.metric("Theta", f"{theta_call(S, K, T, r, sigma):.4f}")
col5.metric("Rho",   f"{rho_call(S, K, T, r, sigma):.4f}")

st.subheader("Put-Call Parity Verification")
left  = call - put
right = S - K * np.exp(-r * T)
col1, col2 = st.columns(2)
col1.metric("C - P", f"${left:.4f}")
col2.metric("S - Ke^(-rT)", f"${right:.4f}")
if abs(left - right) < 0.01:
    st.success("✓ Put-call parity holds")
else:
    st.error("✗ Put-call parity violated")

S_range = np.linspace(50, 500, 200)

st.subheader("Delta vs Stock Price")
deltas = [delta_call(s, K, T, r, sigma) for s in S_range]
fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(S_range, deltas, color='steelblue', linewidth=2)
ax1.axvline(x=S, color='crimson', linestyle='--', alpha=0.7, label=f'S = {S}')
ax1.set_xlabel("Stock Price")
ax1.set_ylabel("Delta")
ax1.legend()
ax1.grid(alpha=0.3)
st.pyplot(fig1)

st.subheader("Vega vs Stock Price")
vegas = [vega(s, K, T, r, sigma) for s in S_range]
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(S_range, vegas, color='darkorange', linewidth=2)
ax2.axvline(x=S, color='crimson', linestyle='--', alpha=0.7, label=f'S = {S}')
ax2.set_xlabel("Stock Price")
ax2.set_ylabel("Vega")
ax2.legend()
ax2.grid(alpha=0.3)
st.pyplot(fig2)

st.subheader("Theta vs Time to Expiry")
T_range = np.linspace(0.01, 3.0, 200)
thetas = [theta_call(S, K, t, r, sigma) for t in T_range]
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.plot(T_range, thetas, color='crimson', linewidth=2)
ax3.axvline(x=T, color='steelblue', linestyle='--', alpha=0.7, label=f'T = {T}')
ax3.set_xlabel("Time to Expiry (years)")
ax3.set_ylabel("Theta ($/day)")
ax3.legend()
ax3.grid(alpha=0.3)
st.pyplot(fig3)

st.subheader("Monte Carlo vs Black-Scholes")
n_sims = st.slider("Number of simulations", 100, 50000, 10000, step=100)
Z  = np.random.standard_normal(n_sims)
ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
payoff   = np.maximum(ST - K, 0)
mc_price = np.exp(-r*T) * np.mean(payoff)
col1, col2 = st.columns(2)
col1.metric("Monte Carlo Price",   f"${mc_price:.4f}")
col2.metric("Black-Scholes Price", f"${call:.4f}")

sizes     = np.logspace(2, np.log10(n_sims), 30).astype(int)
mc_prices = []
for n in sizes:
    Z  = np.random.standard_normal(n)
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    mc_prices.append(np.exp(-r*T) * np.mean(np.maximum(ST - K, 0)))

fig4, ax4 = plt.subplots(figsize=(8, 4))
ax4.plot(sizes, mc_prices, color='steelblue', linewidth=2, label='Monte Carlo')
ax4.axhline(y=call, color='crimson', linestyle='--', linewidth=2, label=f'Black-Scholes: ${call:.4f}')
ax4.set_xlabel("Number of simulations")
ax4.set_ylabel("Call Price")
ax4.set_xscale('log')
ax4.legend()
ax4.grid(alpha=0.3)
st.pyplot(fig4)