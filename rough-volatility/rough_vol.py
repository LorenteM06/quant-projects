"""
Rough Volatility — Empirical Replication of Gatheral, Jaisson & Rosenbaum (2018)
==================================================================================
Test whether realised volatility of major indices exhibits Hurst exponent
H ≈ 0.1 ("rough" behaviour), rejecting the classical H = 0.5 assumption of
Black-Scholes and standard stochastic volatility models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ══════════════════════════════════════════
# 1. DOWNLOAD SPY DATA
# ══════════════════════════════════════════

print("Downloading SPY daily data (5 years)...")
spy = yf.Ticker("SPY").history(period="5y")
print(f"Loaded {len(spy)} daily bars")
print(f"Date range: {spy.index[0].date()} to {spy.index[-1].date()}")

# Log-returns diarios
spy['log_return'] = np.log(spy['Close'] / spy['Close'].shift(1))
spy = spy.dropna()

print(f"\nDaily return stats:")
print(f"Mean: {spy['log_return'].mean()*252*100:.2f}% annualised")
print(f"Vol:  {spy['log_return'].std()*np.sqrt(252)*100:.2f}% annualised")


# ══════════════════════════════════════════
# 2. REALISED VOLATILITY SERIES
# ══════════════════════════════════════════

window = 5  # 5-day rolling window
spy['rv'] = spy['log_return'].rolling(window).std() * np.sqrt(252)
spy = spy.dropna()

plt.figure(figsize=(12, 5))
plt.plot(spy.index, spy['rv'], color='steelblue', linewidth=1)
plt.xlabel('Date')
plt.ylabel('Realised Volatility (annualised)')
plt.title(f'SPY {window}-day Realised Volatility')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('rv_series.png', dpi=150)
plt.show()

print(f"\nRealised vol series stats:")
print(f"Mean: {spy['rv'].mean()*100:.2f}%")
print(f"Min:  {spy['rv'].min()*100:.2f}%")
print(f"Max:  {spy['rv'].max()*100:.2f}%")


# ══════════════════════════════════════════
# 3. HURST EXPONENT ESTIMATION
# ══════════════════════════════════════════

# Trabajamos con log-volatilidad (más estable estadísticamente)
log_vol = np.log(spy['rv']).dropna()

# Método clásico: calcular momentos de incrementos a distintas escalas
def estimate_hurst(series, q=2, max_lag=100):
    """
    Estima H via el q-ésimo momento de incrementos.
    Para H estable: E[|X(t+delta) - X(t)|^q] ~ delta^(q*H)
    """
    lags = np.arange(1, max_lag)
    moments = []
    for lag in lags:
        increments = series.values[lag:] - series.values[:-lag]
        moment = np.mean(np.abs(increments)**q)
        moments.append(moment)
    
    log_lags = np.log(lags)
    log_moments = np.log(moments)
    slope, intercept = np.polyfit(log_lags, log_moments, 1)
    H = slope / q
    return H, log_lags, log_moments, slope

H, log_lags, log_moments, slope = estimate_hurst(log_vol, q=2, max_lag=100)

print(f"\nHurst exponent estimation:")
print(f"H = {H:.4f}")
print(f"Interpretation: {'Rough (Gatheral)' if H < 0.3 else 'Standard' if H > 0.4 else 'Intermediate'}")

# Plot de la regresión
plt.figure(figsize=(10, 5))
plt.plot(log_lags, log_moments, 'o', color='steelblue', label='Empirical moments')
plt.plot(log_lags, slope*log_lags + np.polyfit(log_lags, log_moments, 1)[1], '-', 
         color='crimson', linewidth=2, label=f'Fit: slope={slope:.3f}, H={H:.3f}')
plt.xlabel('log(lag)')
plt.ylabel('log(E[|increment|²])')
plt.title(f'Hurst estimation from log-volatility (SPY, H={H:.3f})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hurst_estimation.png', dpi=150)
plt.show()


# ══════════════════════════════════════════
# 4. ROBUSTNESS: WINDOW SENSITIVITY
# ══════════════════════════════════════════

print("\n" + "="*60)
print("ROBUSTNESS TEST: Hurst vs realised-vol window")
print("="*60)

spy_raw = yf.Ticker("SPY").history(period="5y")
spy_raw['log_return'] = np.log(spy_raw['Close'] / spy_raw['Close'].shift(1))
spy_raw = spy_raw.dropna()

windows = [3, 5, 10, 20, 30]
results_windows = []

for w in windows:
    rv = spy_raw['log_return'].rolling(w).std() * np.sqrt(252)
    log_rv = np.log(rv).dropna()
    H_w, _, _, _ = estimate_hurst(log_rv, q=2, max_lag=min(100, len(log_rv)//5))
    results_windows.append({'window': w, 'H': H_w})
    print(f"  Window {w} days: H = {H_w:.4f}")

df_w = pd.DataFrame(results_windows)

plt.figure(figsize=(9, 5))
plt.plot(df_w['window'], df_w['H'], 'o-', color='steelblue', linewidth=2, markersize=8)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Standard (H=0.5)')
plt.axhline(y=0.1, color='crimson', linestyle='--', alpha=0.5, label='Rough (H≈0.1)')
plt.xlabel('Realised-vol window (days)')
plt.ylabel('Estimated Hurst exponent')
plt.title('Robustness: H stability across RV windows (SPY)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hurst_window_robustness.png', dpi=150)
plt.show()

# ══════════════════════════════════════════
# 5. CROSS-ASSET HURST ESTIMATION
# ══════════════════════════════════════════

print("\n" + "="*60)
print("CROSS-ASSET HURST ESTIMATION (window=5)")
print("="*60)

tickers = ["SPY", "QQQ", "NVDA", "GLD", "TLT"]
window = 5
results_assets = []

for ticker in tickers:
    data = yf.Ticker(ticker).history(period="5y")
    if len(data) < 100:
        continue
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()
    rv = data['log_return'].rolling(window).std() * np.sqrt(252)
    log_rv = np.log(rv).dropna()
    H_t, _, _, _ = estimate_hurst(log_rv, q=2, max_lag=100)
    results_assets.append({'ticker': ticker, 'H': H_t, 'n_obs': len(log_rv)})
    print(f"  {ticker}: H = {H_t:.4f} (n={len(log_rv)})")

df_assets = pd.DataFrame(results_assets)

plt.figure(figsize=(9, 5))
plt.bar(df_assets['ticker'], df_assets['H'], color='steelblue', edgecolor='black')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Standard (H=0.5)')
plt.axhline(y=0.1, color='crimson', linestyle='--', alpha=0.5, label='Rough (H≈0.1)')
plt.xlabel('Asset')
plt.ylabel('Estimated Hurst exponent')
plt.title('Cross-asset Hurst exponent (RV window = 5 days)')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('hurst_cross_asset.png', dpi=150)
plt.show()




# ══════════════════════════════════════════
# 6. TIME-REGIME ANALYSIS
# ══════════════════════════════════════════

print("\n" + "="*60)
print("TIME-REGIME ANALYSIS: H over rolling windows")
print("="*60)

# H rolling: calculamos H para ventanas de 250 días (1 año) que se desplazan
spy_full = yf.Ticker("SPY").history(period="10y")
spy_full['log_return'] = np.log(spy_full['Close'] / spy_full['Close'].shift(1))
spy_full = spy_full.dropna()
spy_full['rv'] = spy_full['log_return'].rolling(5).std() * np.sqrt(252)
log_rv_full = np.log(spy_full['rv']).dropna()

window_size = 250  # 1 año
step = 50  # avanzar 50 días entre estimaciones
H_over_time = []
dates_H = []

for start in range(0, len(log_rv_full) - window_size, step):
    segment = log_rv_full.iloc[start:start + window_size]
    H_seg, _, _, _ = estimate_hurst(segment, q=2, max_lag=50)
    H_over_time.append(H_seg)
    dates_H.append(log_rv_full.index[start + window_size // 2])

plt.figure(figsize=(12, 5))
plt.plot(dates_H, H_over_time, 'o-', color='steelblue', linewidth=1.5, markersize=5)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Standard (H=0.5)')
plt.axhline(y=0.1, color='crimson', linestyle='--', alpha=0.5, label='Rough (H≈0.1)')
plt.axhline(y=np.mean(H_over_time), color='green', linestyle=':', alpha=0.7, 
            label=f'Mean H = {np.mean(H_over_time):.3f}')
plt.xlabel('Date (window midpoint)')
plt.ylabel('Estimated Hurst exponent')
plt.title('Rolling Hurst estimation (SPY, 1-year windows)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hurst_time_regime.png', dpi=150)
plt.show()

print(f"\nRolling H statistics (SPY 10 years, 1-year windows):")
print(f"Mean H: {np.mean(H_over_time):.4f}")
print(f"Std H:  {np.std(H_over_time):.4f}")
print(f"Min H:  {np.min(H_over_time):.4f}")
print(f"Max H:  {np.max(H_over_time):.4f}")





# ══════════════════════════════════════════
# 7. METHOD VALIDATION: SIMULATION
# ══════════════════════════════════════════

print("\n" + "="*60)
print("METHOD VALIDATION: simulate fBM with known H")
print("="*60)

def fbm_hosking(n, H, T=1.0):
    """
    Simulate fractional Brownian motion via Hosking's method.
    Simple but slow — adequate for validation purposes.
    """
    dt = T / n
    gamma = 0.5 * (np.arange(n+1)[np.newaxis, :]**(2*H) - 
                   2*np.arange(n)[:, np.newaxis]**(2*H) + 
                   np.abs(np.arange(-1, n)[np.newaxis, :] - 
                          np.arange(n)[:, np.newaxis])**(2*H) 
                   if False else 0)
    # Método alternativo más simple: Cholesky
    times = np.arange(1, n+1) * dt
    cov = 0.5 * (np.abs(times[:, None])**(2*H) + 
                 np.abs(times[None, :])**(2*H) - 
                 np.abs(times[:, None] - times[None, :])**(2*H))
    L = np.linalg.cholesky(cov + 1e-10*np.eye(n))
    Z = np.random.standard_normal(n)
    return L @ Z

# Test para distintos H conocidos
true_Hs = [0.1, 0.2, 0.3, 0.5, 0.7]
n_sim = 1000
np.random.seed(42)

print(f"\n{'True H':<10} {'Estimated H':<15} {'Error':<10}")
print("-" * 35)
for true_H in true_Hs:
    # Simular varias trayectorias y promediar el estimador
    estimated_Hs = []
    for _ in range(10):
        fbm = fbm_hosking(n_sim, true_H)
        H_est, _, _, _ = estimate_hurst(pd.Series(fbm), q=2, max_lag=100)
        estimated_Hs.append(H_est)
    mean_est = np.mean(estimated_Hs)
    error = mean_est - true_H
    print(f"{true_H:<10} {mean_est:<12.4f}  {error:+.4f}")



# ══════════════════════════════════════════
# 8. TRADING STRATEGY: H-BASED REGIME SWITCHING
# ══════════════════════════════════════════

print("\n" + "="*60)
print("TRADING STRATEGY: H-based regime switching")
print("="*60)

# Reutilizamos los datos de 10 años de SPY con H rolling
# Ya tenemos: dates_H, H_over_time

# Alinear H con precios
H_series = pd.Series(H_over_time, index=dates_H)
spy_prices_full = yf.Ticker("SPY").history(period="10y")['Close']

# Rellenar H a diaria (forward fill)
H_daily = H_series.reindex(spy_prices_full.index, method='ffill')

# Definir señales
H_high_threshold = 0.25
H_low_threshold = 0.15

# Estrategia: long SPY cuando H > 0.25, cash cuando H < 0.15, mantener posición previa en zona intermedia
signal = pd.Series(index=H_daily.index, dtype=float)
current_position = 0
for i, h in enumerate(H_daily):
    if pd.isna(h):
        signal.iloc[i] = current_position
        continue
    if h > H_high_threshold:
        current_position = 1
    elif h < H_low_threshold:
        current_position = 0
    signal.iloc[i] = current_position

# Retornos diarios SPY
returns_spy = spy_prices_full.pct_change().fillna(0)

# Retornos estrategia (con lag para evitar look-ahead)
strategy_returns = signal.shift(1) * returns_spy
strategy_returns = strategy_returns.fillna(0)

# Métricas
buy_hold_returns = returns_spy
buy_hold_cumret = (1 + buy_hold_returns).cumprod()
strategy_cumret = (1 + strategy_returns).cumprod()

def sharpe_ratio(returns, rf=0.0):
    return (returns.mean() - rf/252) / returns.std() * np.sqrt(252)

def max_drawdown(cumret):
    running_max = cumret.expanding().max()
    drawdown = (cumret - running_max) / running_max
    return drawdown.min()

print(f"\nStrategy: Long SPY when H > {H_high_threshold}, Cash when H < {H_low_threshold}")
print(f"\n{'Metric':<25} {'Buy & Hold':<15} {'Strategy':<15}")
print("-" * 55)
print(f"{'Total return':<25} {(buy_hold_cumret.iloc[-1]-1)*100:>10.2f}%   {(strategy_cumret.iloc[-1]-1)*100:>10.2f}%")
print(f"{'Annualized return':<25} {(buy_hold_returns.mean()*252)*100:>10.2f}%   {(strategy_returns.mean()*252)*100:>10.2f}%")
print(f"{'Annualized volatility':<25} {(buy_hold_returns.std()*np.sqrt(252))*100:>10.2f}%   {(strategy_returns.std()*np.sqrt(252))*100:>10.2f}%")
print(f"{'Sharpe ratio':<25} {sharpe_ratio(buy_hold_returns):>10.3f}    {sharpe_ratio(strategy_returns):>10.3f}")
print(f"{'Max drawdown':<25} {max_drawdown(buy_hold_cumret)*100:>10.2f}%   {max_drawdown(strategy_cumret)*100:>10.2f}%")
print(f"{'Time in market':<25} {100:>10.2f}%   {signal.mean()*100:>10.2f}%")

# Visualización
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(buy_hold_cumret.index, buy_hold_cumret, label='Buy & Hold', color='steelblue', linewidth=2)
ax1.plot(strategy_cumret.index, strategy_cumret, label='H-based strategy', color='crimson', linewidth=2)
ax1.set_ylabel('Cumulative return')
ax1.set_title('H-based regime switching vs Buy & Hold')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(H_daily.index, H_daily, color='gray', linewidth=1, alpha=0.7, label='Rolling H')
ax2.axhline(y=H_high_threshold, color='crimson', linestyle='--', alpha=0.7, label=f'Long threshold (H={H_high_threshold})')
ax2.axhline(y=H_low_threshold, color='steelblue', linestyle='--', alpha=0.7, label=f'Cash threshold (H={H_low_threshold})')
ax2.fill_between(signal.index, 0, signal, alpha=0.2, color='green', label='In market')
ax2.set_ylabel('H / signal')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('h_strategy_backtest.png', dpi=150)
plt.show()



