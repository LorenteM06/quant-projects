import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# ══════════════════════════════════════════
# 1. FUNCIONES BASE
# ══════════════════════════════════════════

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def implied_vol(market_price, S, K, T, r):
    try:
        return brentq(lambda sigma: bs_call(S, K, T, r, sigma) - market_price, 0.001, 5.0)
    except:
        return None

def svi(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def fit_svi(k_data, w_data):
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

# ══════════════════════════════════════════
# 2. DESCARGAR DATOS DE SPY
# ══════════════════════════════════════════

print("Downloading SPY option data...")
ticker = yf.Ticker("SPY")
S = ticker.history(period="1d")['Close'].iloc[-1]
r = 0.05
today = datetime.now()

print(f"SPY spot price: ${S:.2f}")
print(f"Total expiries available: {len(ticker.options)}")

# Coger vencimientos espaciados: 1 semana, 1 mes, 2 meses, 3 meses, 6 meses, 1 año
target_days = [7, 30, 60, 90, 180, 365]
selected_expiries = []

for target in target_days:
    best_expiry = None
    best_diff = float('inf')
    for exp in ticker.options:
        exp_date = datetime.strptime(exp, "%Y-%m-%d")
        days = (exp_date - today).days
        if days > 0 and abs(days - target) < best_diff:
            best_diff = abs(days - target)
            best_expiry = exp
    if best_expiry and best_expiry not in selected_expiries:
        selected_expiries.append(best_expiry)

print(f"\nSelected expiries: {selected_expiries}")

# ══════════════════════════════════════════
# 3. CALCULAR IVs Y AJUSTAR SVI PARA CADA VENCIMIENTO
# ══════════════════════════════════════════

surface_data = []

for expiry in selected_expiries:
    exp_date = datetime.strptime(expiry, "%Y-%m-%d")
    T = (exp_date - today).days / 365
    
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
        'expiry': expiry,
        'T': T,
        'strikes': calls['strike'].values,
        'log_moneyness': calls['log_moneyness'].values,
        'iv_market': calls['iv'].values,
        'svi_params': params
    })
    
    print(f"  {expiry}: T={T:.3f}, {len(calls)} points, params={[f'{p:.3f}' for p in params]}")

# ══════════════════════════════════════════
# 4. VISUALIZACIÓN 2D — TODAS LAS SONRISAS
# ══════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(surface_data)))

for i, data in enumerate(surface_data):
    k_range = np.linspace(data['log_moneyness'].min(), data['log_moneyness'].max(), 100)
    w_svi = svi(k_range, *data['svi_params'])
    iv_svi = np.sqrt(w_svi / data['T'])
    
    ax.plot(data['log_moneyness'], data['iv_market'], 'o', color=colors[i], alpha=0.6)
    ax.plot(k_range, iv_svi, '-', color=colors[i], linewidth=2, label=f"T={data['T']:.2f}y")

ax.set_xlabel("Log-moneyness  ln(K/S)")
ax.set_ylabel("Implied Volatility")
ax.set_title("SPY Volatility Smiles Across Expiries (SVI Fit)")
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('smiles_all.png', dpi=150)
plt.show()

# ══════════════════════════════════════════
# 5. VISUALIZACIÓN 3D — SUPERFICIE COMPLETA
# ══════════════════════════════════════════

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

k_grid = np.linspace(-0.15, 0.15, 50)
T_grid = np.array([d['T'] for d in surface_data])
K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
IV_mesh = np.zeros_like(K_mesh)

for i, data in enumerate(surface_data):
    w = svi(k_grid, *data['svi_params'])
    IV_mesh[i, :] = np.sqrt(w / data['T'])

surf = ax.plot_surface(K_mesh, T_mesh, IV_mesh, cmap='viridis', alpha=0.9, edgecolor='none')
ax.set_xlabel("Log-moneyness")
ax.set_ylabel("Time to expiry (years)")
ax.set_zlabel("Implied Volatility")
ax.set_title("SPY Implied Volatility Surface (SVI)")
fig.colorbar(surf, ax=ax, shrink=0.5)
plt.tight_layout()
plt.savefig('iv_surface_3d.png', dpi=150)
plt.show()

print("\nSaved: smiles_all.png, iv_surface_3d.png")


# ══════════════════════════════════════════
# 6. ANÁLISIS: TERM STRUCTURE DEL SKEW
# ══════════════════════════════════════════

Ts = [d['T'] for d in surface_data]
rhos = [d['svi_params'][2] for d in surface_data]  # rho es el 3er parámetro
atm_ivs = [np.sqrt(svi(0, *d['svi_params']) / d['T']) for d in surface_data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel izquierdo: rho vs T
ax1.plot(Ts, rhos, 'o-', color='crimson', linewidth=2, markersize=8)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel("Time to expiry (years)")
ax1.set_ylabel("SVI rho (skew asymmetry)")
ax1.set_title("Term Structure of Skew")
ax1.grid(alpha=0.3)

# Panel derecho: ATM vol vs T
ax2.plot(Ts, atm_ivs, 'o-', color='steelblue', linewidth=2, markersize=8)
ax2.set_xlabel("Time to expiry (years)")
ax2.set_ylabel("At-the-money implied volatility")
ax2.set_title("Term Structure of ATM Volatility")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('term_structure.png', dpi=150)
plt.show()

print("\nTerm structure analysis:")
for i, data in enumerate(surface_data):
    print(f"  T={data['T']:.3f}y: rho={rhos[i]:.3f}, ATM IV={atm_ivs[i]:.3f}")