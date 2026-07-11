"""
SPY Volatility Surface Analysis with SVI Parametrization
Full pipeline: IV extraction, SVI calibration, arbitrage checks, and mispricing detection.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# ══════════════════════════════════════════
# 1. CORE FUNCTIONS
# ══════════════════════════════════════════

def bs_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def implied_vol(market_price, S, K, T, r):
    """Extract implied volatility from a market price using Brent's method."""
    try:
        return brentq(lambda sigma: bs_call(S, K, T, r, sigma) - market_price, 0.001, 5.0)
    except:
        return None

def svi(k, a, b, rho, m, sigma):
    """SVI raw parametrization (Gatheral 2004). Returns total variance w."""
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def fit_svi(k_data, w_data):
    """Calibrate SVI parameters via non-linear least squares."""
    def error(params):
        a, b, rho, m, sigma = params
        return np.sum((svi(k_data, a, b, rho, m, sigma) - w_data)**2)
    result = minimize(
        error,
        x0=[0.01, 0.1, -0.5, 0.0, 0.1],
        bounds=[(-1, 1), (0.001, 5), (-0.999, 0.999), (-1, 1), (0.001, 5)],
        method='L-BFGS-B'
    )
    return result.x

def butterfly_check(a, b, rho, m, sigma_svi, k_range):
    """Check butterfly arbitrage via Gatheral (2014) g(k) function."""
    k = k_range
    w = svi(k, a, b, rho, m, sigma_svi)
    dw_dk = b * (rho + (k - m) / np.sqrt((k - m)**2 + sigma_svi**2))
    d2w_dk2 = b * sigma_svi**2 / ((k - m)**2 + sigma_svi**2)**1.5
    g = (1 - k * dw_dk / (2 * w))**2 - (dw_dk**2 / 4) * (1/w + 1/4) + d2w_dk2 / 2
    return g, np.all(g >= 0)

# ══════════════════════════════════════════
# 2. DATA DOWNLOAD
# ══════════════════════════════════════════

print("Downloading SPY option data...")
ticker = yf.Ticker("SPY")
S = ticker.history(period="1d")['Close'].iloc[-1]
r = 0.05
today = datetime.now()

print(f"SPY spot: ${S:.2f}")

# Select 6 maturities: ~1w, 1m, 2m, 3m, 6m, 1y
target_days = [7, 30, 60, 90, 180, 365]
selected_expiries = []
for target in target_days:
    best, best_diff = None, float('inf')
    for exp in ticker.options:
        days = (datetime.strptime(exp, "%Y-%m-%d") - today).days
        if days > 0 and abs(days - target) < best_diff:
            best_diff, best = abs(days - target), exp
    if best and best not in selected_expiries:
        selected_expiries.append(best)

print(f"Maturities selected: {selected_expiries}\n")

# ══════════════════════════════════════════
# 3. SVI CALIBRATION PER MATURITY
# ══════════════════════════════════════════

surface_data = []
for expiry in selected_expiries:
    T = (datetime.strptime(expiry, "%Y-%m-%d") - today).days / 365
    chain = ticker.option_chain(expiry).calls
    calls = chain[(chain['strike'] > S*0.85) & (chain['strike'] < S*1.15) & (chain['lastPrice'] > 0.5)].copy()
    calls['iv'] = calls.apply(lambda row: implied_vol(row['lastPrice'], S, row['strike'], T, r), axis=1)
    calls = calls.dropna(subset=['iv'])
    if len(calls) < 5:
        continue
    calls['log_moneyness'] = np.log(calls['strike'] / S)
    calls['total_variance'] = calls['iv']**2 * T
    params = fit_svi(calls['log_moneyness'].values, calls['total_variance'].values)
    surface_data.append({
        'expiry': expiry, 'T': T,
        'strikes': calls['strike'].values,
        'log_moneyness': calls['log_moneyness'].values,
        'iv_market': calls['iv'].values,
        'svi_params': params
    })
    print(f"  {expiry}: T={T:.3f}, n={len(calls)}, params={[f'{p:.3f}' for p in params]}")

# ══════════════════════════════════════════
# 4. MODEL DIAGNOSTICS (RMSE + BUTTERFLY)
# ══════════════════════════════════════════

print("\n" + "="*60)
print("MODEL DIAGNOSTICS")
print("="*60)
k_test = np.linspace(-0.2, 0.2, 200)
for data in surface_data:
    a, b, rho, m, sigma_svi = data['svi_params']
    w_model = svi(data['log_moneyness'], *data['svi_params'])
    w_market = data['iv_market']**2 * data['T']
    rmse_iv = np.sqrt(np.mean((np.sqrt(w_model / data['T']) - data['iv_market'])**2))
    g, arb_free = butterfly_check(*data['svi_params'], k_test)
    status = "arb-free" if arb_free else "ARBITRAGE"
    print(f"  T={data['T']:.3f}y: RMSE(IV)={rmse_iv:.4f} | {status} (min g={g.min():+.4f})")

# ══════════════════════════════════════════
# 5. TERM STRUCTURE ANALYSIS
# ══════════════════════════════════════════

Ts = [d['T'] for d in surface_data]
rhos = [d['svi_params'][2] for d in surface_data]
atm_ivs = [np.sqrt(svi(0, *d['svi_params']) / d['T']) for d in surface_data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(Ts, rhos, 'o-', color='crimson', linewidth=2, markersize=8)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel("Time to expiry (years)")
ax1.set_ylabel("SVI ρ (skew asymmetry)")
ax1.set_title("Term Structure of Skew")
ax1.grid(alpha=0.3)
ax2.plot(Ts, atm_ivs, 'o-', color='steelblue', linewidth=2, markersize=8)
ax2.set_xlabel("Time to expiry (years)")
ax2.set_ylabel("ATM implied volatility")
ax2.set_title("Term Structure of ATM Volatility")
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('term_structure.png', dpi=150)
plt.show()

# ══════════════════════════════════════════
# 6. VOLATILITY SURFACE (3D + HEATMAP)
# ══════════════════════════════════════════

from mpl_toolkits.mplot3d import Axes3D

k_grid = np.linspace(-0.15, 0.15, 60)
T_grid = np.linspace(min(Ts), max(Ts), 40)
K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
IV_mesh = np.zeros_like(K_mesh)
Ts_arr = np.array(Ts)
for i, T_val in enumerate(T_grid):
    idx = np.argmin(np.abs(Ts_arr - T_val))
    w = svi(k_grid, *surface_data[idx]['svi_params'])
    IV_mesh[i, :] = np.sqrt(w / T_val)

# 3D
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(K_mesh, T_mesh, IV_mesh, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_xlabel("Log-moneyness")
ax.set_ylabel("Time to expiry (years)")
ax.set_zlabel("Implied Volatility")
ax.set_title("SPY Implied Volatility Surface (SVI)")
fig.colorbar(surf, ax=ax, shrink=0.5)
plt.tight_layout()
plt.savefig('iv_surface_3d.png', dpi=150)
plt.show()

# Heatmap
fig, ax = plt.subplots(figsize=(11, 6))
im = ax.pcolormesh(K_mesh, T_mesh, IV_mesh, cmap='viridis', shading='auto')
ax.axvline(x=0, color='white', linestyle='--', alpha=0.6, linewidth=1)
ax.set_xlabel("Log-moneyness  ln(K/S)")
ax.set_ylabel("Time to expiry (years)")
ax.set_title("SPY Implied Volatility Surface — Heatmap")
fig.colorbar(im, ax=ax, label="Implied Volatility")
plt.tight_layout()
plt.savefig('iv_heatmap.png', dpi=150)
plt.show()

# ══════════════════════════════════════════
# 7. MISPRICING DETECTION (QUALITY-FILTERED)
# ══════════════════════════════════════════

print("\n" + "="*60)
print("MISPRICING DETECTION")
print("Filtered by volume > 10, spread < 5% of price")
print("="*60 + "\n")

filtered_signals = []
for data in surface_data:
    expiry, T, params = data['expiry'], data['T'], data['svi_params']
    chain = ticker.option_chain(expiry).calls.copy()
    chain['spread_pct'] = (chain['ask'] - chain['bid']) / chain['lastPrice']
    quality = chain[
        (chain['strike'] > S*0.85) & (chain['strike'] < S*1.15) &
        (chain['lastPrice'] > 0.5) & (chain['volume'] > 10) &
        (chain['spread_pct'] < 0.05) & (chain['bid'] > 0)
    ].copy()
    if len(quality) == 0:
        continue
    quality['iv'] = quality.apply(lambda row: implied_vol(row['lastPrice'], S, row['strike'], T, r), axis=1)
    quality = quality.dropna(subset=['iv'])
    if len(quality) == 0:
        continue
    quality['log_moneyness'] = np.log(quality['strike'] / S)
    quality['iv_svi'] = np.sqrt(svi(quality['log_moneyness'].values, *params) / T)
    quality['residual'] = quality['iv'] - quality['iv_svi']
    dislocations = quality[quality['residual'].abs() > 0.02]
    if len(dislocations) > 0:
        print(f"T={T:.3f}y ({expiry}):")
        for _, row in dislocations.iterrows():
            direction = "OVERPRICED" if row['residual'] > 0 else "UNDERPRICED"
            print(f"  Strike ${row['strike']:.2f}: IV={row['iv']:.4f}, SVI={row['iv_svi']:.4f}, "
                  f"diff={row['residual']:+.4f} | vol={int(row['volume'])} → {direction}")
            filtered_signals.append({
                'expiry': expiry, 'T': T, 'strike': row['strike'],
                'iv_market': row['iv'], 'iv_svi': row['iv_svi'],
                'residual': row['residual'], 'volume': int(row['volume']),
                'direction': direction
            })
        print()

print(f"Total quality signals: {len(filtered_signals)}")

# Export signals to CSV
import pandas as pd
if filtered_signals:
    pd.DataFrame(filtered_signals).to_csv('signals.csv', index=False)
    print("Signals saved to signals.csv")

# ══════════════════════════════════════════
# 8. MISPRICING VISUALIZATION
# ══════════════════════════════════════════

if filtered_signals:
    fig, ax = plt.subplots(figsize=(11, 6))
    for sig in filtered_signals:
        color = 'crimson' if sig['direction'] == 'OVERPRICED' else 'steelblue'
        ax.scatter(sig['strike'], sig['T'], s=abs(sig['residual'])*3000,
                   color=color, alpha=0.7, edgecolor='black')
    ax.axvline(x=S, color='gray', linestyle='--', alpha=0.5, label=f'Spot: ${S:.2f}')
    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to expiry (years)")
    ax.set_title("Detected Mispricings (size = deviation magnitude)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('mispricings_map.png', dpi=150)
    plt.show()

print("\nAll outputs saved.")


# ══════════════════════════════════════════
# 9. CROSS-TICKER COMPARISON (SPY vs QQQ)
# ══════════════════════════════════════════

print("\n" + "="*60)
print("CROSS-TICKER COMPARISON: SPY vs QQQ")
print("="*60 + "\n")

def analyze_ticker(symbol):
    tk = yf.Ticker(symbol)
    spot = tk.history(period="1d")['Close'].iloc[-1]
    results = []
    for target in [30, 90, 180]:
        best, best_diff = None, float('inf')
        for exp in tk.options:
            days = (datetime.strptime(exp, "%Y-%m-%d") - today).days
            if days > 0 and abs(days - target) < best_diff:
                best_diff, best = abs(days - target), exp
        if not best:
            continue
        T = (datetime.strptime(best, "%Y-%m-%d") - today).days / 365
        chain = tk.option_chain(best).calls
        calls = chain[(chain['strike'] > spot*0.85) & (chain['strike'] < spot*1.15) & (chain['lastPrice'] > 0.5)].copy()
        calls['iv'] = calls.apply(lambda row: implied_vol(row['lastPrice'], spot, row['strike'], T, r), axis=1)
        calls = calls.dropna(subset=['iv'])
        if len(calls) < 5:
            continue
        calls['log_moneyness'] = np.log(calls['strike'] / spot)
        calls['total_variance'] = calls['iv']**2 * T
        params = fit_svi(calls['log_moneyness'].values, calls['total_variance'].values)
        atm_iv = np.sqrt(svi(0, *params) / T)
        results.append({'T': T, 'rho': params[2], 'atm_iv': atm_iv})
    return spot, results

spy_spot, spy_res = analyze_ticker("SPY")
qqq_spot, qqq_res = analyze_ticker("QQQ")

print(f"{'Metric':<20} {'SPY':>12} {'QQQ':>12}")
print("-"*46)
print(f"{'Spot price':<20} ${spy_spot:>10.2f} ${qqq_spot:>10.2f}")
for i in range(min(len(spy_res), len(qqq_res))):
    T = spy_res[i]['T']
    print(f"\nMaturity T={T:.2f}y:")
    print(f"{'  ρ (skew)':<20} {spy_res[i]['rho']:>12.4f} {qqq_res[i]['rho']:>12.4f}")
    print(f"{'  ATM IV':<20} {spy_res[i]['atm_iv']:>12.4f} {qqq_res[i]['atm_iv']:>12.4f}")

# Visual comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot([r['T'] for r in spy_res], [r['rho'] for r in spy_res], 'o-', label='SPY', color='steelblue', linewidth=2)
ax1.plot([r['T'] for r in qqq_res], [r['rho'] for r in qqq_res], 'o-', label='QQQ', color='crimson', linewidth=2)
ax1.set_xlabel("Time to expiry (years)")
ax1.set_ylabel("SVI ρ")
ax1.set_title("Skew: SPY vs QQQ")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot([r['T'] for r in spy_res], [r['atm_iv'] for r in spy_res], 'o-', label='SPY', color='steelblue', linewidth=2)
ax2.plot([r['T'] for r in qqq_res], [r['atm_iv'] for r in qqq_res], 'o-', label='QQQ', color='crimson', linewidth=2)
ax2.set_xlabel("Time to expiry (years)")
ax2.set_ylabel("ATM implied volatility")
ax2.set_title("ATM Volatility: SPY vs QQQ")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('spy_vs_qqq.png', dpi=150)
plt.show()