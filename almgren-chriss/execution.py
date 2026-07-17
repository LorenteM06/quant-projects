import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ══════════════════════════════════════════
# 1. PARÁMETROS INICIALES (escenario sintético)
# ══════════════════════════════════════════

X0 = 100_000       # Acciones a vender
T = 5              # Horizonte total (horas)
N = 50             # Número de intervalos
sigma = 0.5        # Volatilidad ($/√hora)
gamma = 2.5e-7     # Impacto permanente
eta = 2.5e-6       # Impacto temporal

print(f"Selling {X0} shares over {T} hours in {N} intervals")
print(f"Interval size: {T/N} hours = {T/N * 60:.1f} minutes each")

# ══════════════════════════════════════════
# 2. FUNCIONES BASE
# ══════════════════════════════════════════

def ac_holdings(kappa, X0, T, times):
    """Trayectoria Almgren-Chriss numéricamente estable."""
    tau = T - times
    if kappa * T > 500:
        result = np.zeros_like(times)
        result[times < T/len(times)] = X0
        return result
    return X0 * (np.exp(kappa * tau - kappa * T) - np.exp(-kappa * tau - kappa * T)) / \
                (1 - np.exp(-2 * kappa * T))

def strategy_stats(kappa, X0, T, sigma, eta, N):
    """Coste esperado y varianza de la estrategia."""
    times = np.linspace(0, T, N+1)
    dt = T / N
    x = ac_holdings(kappa, X0, T, times)
    trades = -np.diff(x)
    velocities = trades / dt
    expected_cost = np.sum(eta * velocities * trades)
    variance = sigma**2 * np.sum(x[1:]**2) * dt
    return expected_cost, variance

def estimate_params(ticker_symbol, period="6mo"):
    """Estima sigma, eta, gamma para un ticker."""
    tk = yf.Ticker(ticker_symbol)
    hist = tk.history(period=period)
    daily_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    sigma_daily = daily_returns.std()
    price = hist['Close'].iloc[-1]
    sigma_price = sigma_daily * price
    sigma_hourly = sigma_price / np.sqrt(6.5)
    avg_volume = hist['Volume'].mean()
    spread = 0.01 if avg_volume > 10e6 else 0.05
    eta = spread / (2 * avg_volume / 6.5)
    gamma = eta / 10
    return {
        'ticker': ticker_symbol,
        'price': price,
        'sigma_hourly': sigma_hourly,
        'avg_volume': avg_volume,
        'eta': eta,
        'gamma': gamma
    }

# ══════════════════════════════════════════
# 3. SENSITIVITY TO LAMBDA
# ══════════════════════════════════════════

times = np.linspace(0, T, N+1)

# TWAP como referencia
twap_holdings_ref = X0 - np.cumsum(np.full(N, X0 / N))

lambdas = [1e-8, 1e-6, 1e-4, 1e-2]
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(lambdas)))

plt.figure(figsize=(11, 6))
for lam, color in zip(lambdas, colors):
    k = np.sqrt(lam * sigma**2 / eta)
    holdings = ac_holdings(k, X0, T, times[1:])
    plt.plot(times[1:], holdings, 'o-', color=color, label=f'AC λ={lam:.0e}', linewidth=2)
plt.plot(times[1:], twap_holdings_ref, '--', color='gray', label='TWAP', alpha=0.7)
plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Time (hours)')
plt.ylabel('Shares remaining')
plt.title('Optimal execution: sensitivity to risk aversion (λ)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lambda_sensitivity.png', dpi=150)
plt.show()

# ══════════════════════════════════════════
# 4. EFFICIENT FRONTIER
# ══════════════════════════════════════════

lambdas_range = np.logspace(-10, -2, 100)
costs = []
variances = []

for lam in lambdas_range:
    k = np.sqrt(lam * sigma**2 / eta)
    c, v = strategy_stats(k, X0, T, sigma, eta, N)
    costs.append(c)
    variances.append(v)

costs = np.array(costs)
variances = np.array(variances)

plt.figure(figsize=(10, 6))
plt.plot(variances, costs, '-', color='seagreen', linewidth=2, label='Efficient frontier')

for lam in [1e-8, 1e-6, 1e-4]:
    k = np.sqrt(lam * sigma**2 / eta)
    c, v = strategy_stats(k, X0, T, sigma, eta, N)
    plt.plot(v, c, 'o', color='crimson', markersize=8)
    plt.annotate(f'λ={lam:.0e}', (v, c), textcoords="offset points", xytext=(10, 5), fontsize=9)

plt.xlabel('Variance of cost (risk)')
plt.ylabel('Expected cost')
plt.title('Efficient Frontier: cost vs risk trade-off')
plt.xscale('log')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('efficient_frontier.png', dpi=150)
plt.show()

# ══════════════════════════════════════════
# 5. INSTITUTIONAL BENCHMARKS (TWAP vs VWAP vs AC)
# ══════════════════════════════════════════

print("\n" + "="*60)
print("INSTITUTIONAL BENCHMARKS: TWAP vs VWAP vs Almgren-Chriss")
print("="*60)

# Patrón de volumen intradía histórico
spy_intra_vwap = yf.Ticker("SPY").history(period="30d", interval="15m").reset_index()
spy_intra_vwap['hour'] = spy_intra_vwap['Datetime'].dt.hour + spy_intra_vwap['Datetime'].dt.minute / 60
volume_pattern = spy_intra_vwap.groupby('hour')['Volume'].mean()
volume_weights = volume_pattern / volume_pattern.sum()

N_vwap = len(volume_weights)
vwap_trades = X0 * volume_weights.values
vwap_holdings = X0 - np.cumsum(vwap_trades)
times_vwap = np.linspace(0, T, N_vwap+1)

lam_vwap = 1e-6
spy_p = estimate_params("SPY")
k_vwap = np.sqrt(lam_vwap * spy_p['sigma_hourly']**2 / spy_p['eta'])
ac_holdings_ref = ac_holdings(k_vwap, X0, T, times_vwap[1:])
twap_holdings_ref2 = X0 - np.cumsum(np.full(N_vwap, X0/N_vwap))

plt.figure(figsize=(11, 6))
plt.plot(times_vwap[1:], vwap_holdings, 'o-', color='darkorange', label='VWAP', linewidth=2)
plt.plot(times_vwap[1:], twap_holdings_ref2, '--', color='gray', label='TWAP', linewidth=2)
plt.plot(times_vwap[1:], ac_holdings_ref, 'o-', color='seagreen', label=f'AC λ={lam_vwap:.0e}', linewidth=2)
plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Time (hours)')
plt.ylabel('Shares remaining')
plt.title('Institutional benchmarks: TWAP vs VWAP vs Almgren-Chriss')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('vwap_comparison.png', dpi=150)
plt.show()

def cost_of_schedule(trades, sigma, eta, T, N, X0):
    dt = T / N
    velocities = trades / dt
    x = X0 - np.cumsum(trades)
    cost = np.sum(eta * velocities * trades)
    var = sigma**2 * np.sum(x**2) * dt
    return cost, var

twap_c, twap_v = cost_of_schedule(np.full(N_vwap, X0/N_vwap), spy_p['sigma_hourly'], spy_p['eta'], T, N_vwap, X0)
vwap_c, vwap_v = cost_of_schedule(vwap_trades, spy_p['sigma_hourly'], spy_p['eta'], T, N_vwap, X0)
ac_trades = -np.diff(np.concatenate([[X0], ac_holdings_ref]))
ac_c, ac_v = cost_of_schedule(ac_trades, spy_p['sigma_hourly'], spy_p['eta'], T, N_vwap, X0)

print(f"\nExecution stats for {X0:,} SPY shares:")
print(f"{'Strategy':<10} {'E[Cost]':<15} {'Std':<15}")
print("-" * 40)
print(f"{'TWAP':<10} ${twap_c:<12.4f}  ${np.sqrt(twap_v):<12.2f}")
print(f"{'VWAP':<10} ${vwap_c:<12.4f}  ${np.sqrt(vwap_v):<12.2f}")
print(f"{'AC':<10} ${ac_c:<12.4f}  ${np.sqrt(ac_v):<12.2f}")

# ══════════════════════════════════════════
# 6. CROSS-ASSET TRADING URGENCY
# ══════════════════════════════════════════

print("\n" + "="*60)
print("CROSS-ASSET TRADING URGENCY")
print("="*60)

tickers = ["SPY", "QQQ", "NVDA", "RIVN"]
assets = [estimate_params(t) for t in tickers]

print(f"\n{'Ticker':<8} {'Price':<12} {'Vol/hr':<10} {'Volume(M)':<12} {'eta':<12}")
print("-" * 55)
for a in assets:
    print(f"{a['ticker']:<8} ${a['price']:<10.2f} ${a['sigma_hourly']:<8.4f} "
          f"{a['avg_volume']/1e6:<10.1f} {a['eta']:<10.2e}")

# Trayectorias óptimas (0.5% ADV, lambda fijo)
T_cross = 5
N_cross = 30
times_cross = np.linspace(0, T_cross, N_cross+1)
lam_cross = 1e-6

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for a in assets:
    X0_cross = int(0.005 * a['avg_volume'])
    k = np.sqrt(lam_cross * a['sigma_hourly']**2 / a['eta'])
    holdings = ac_holdings(k, X0_cross, T_cross, times_cross[1:])
    holdings_pct = holdings / X0_cross
    axes[0].plot(times_cross[1:], holdings_pct, 'o-', label=a['ticker'], linewidth=2)

axes[0].set_xlabel('Time (hours)')
axes[0].set_ylabel('Shares remaining (% of order)')
axes[0].set_title(f'Optimal trajectories (λ={lam_cross:.0e}, order = 0.5% ADV)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Half-completion time vs lambda (urgencia)
def half_completion_time(kappa, T):
    if kappa * T < 1e-6:
        return 0.5
    times_fine = np.linspace(0, T, 1000)
    holdings = ac_holdings(kappa, 1.0, T, times_fine)
    idx = np.argmin(np.abs(holdings - 0.5))
    return times_fine[idx] / T

lambdas_urg = np.logspace(-14, -4, 40)

for a in assets:
    hcs = []
    for lam in lambdas_urg:
        k = np.sqrt(lam * a['sigma_hourly']**2 / a['eta'])
        hcs.append(half_completion_time(k, T_cross))
    axes[1].plot(lambdas_urg, hcs, 'o-', label=a['ticker'], linewidth=2)

axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='TWAP (t½=0.5)')
axes[1].set_xlabel('Risk aversion (λ)')
axes[1].set_ylabel('Half-completion time (fraction of T)')
axes[1].set_title('Trading urgency: how fast position is halved')
axes[1].set_xscale('log')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cross_asset.png', dpi=150)
plt.show()

print(f"\nCritical urgency (order = 1% ADV, T={T_cross}h):")
print(f"{'Ticker':<8} {'λ critical':<15}")
print("-" * 30)
for a in assets:
    lam_crit_search = np.logspace(-14, -4, 200)
    for lam in lam_crit_search:
        k = np.sqrt(lam * a['sigma_hourly']**2 / a['eta'])
        hc = half_completion_time(k, T_cross)
        if hc < 0.35:
            print(f"{a['ticker']:<8} {lam:.1e}")
            break

# ══════════════════════════════════════════
# 7. HISTORICAL BACKTEST
# ══════════════════════════════════════════

print("\n" + "="*60)
print("HISTORICAL BACKTEST: SPY execution over 30 days")
print("="*60)

spy_intraday = yf.Ticker("SPY").history(period="30d", interval="15m")
print(f"Loaded {len(spy_intraday)} 15-min bars")

X0_bt = 500_000
T_bt = 5
N_bt = 20
lam_bt = 1e-9

k_bt = np.sqrt(lam_bt * spy_p['sigma_hourly']**2 / spy_p['eta'])
times_bt = np.linspace(0, T_bt, N_bt+1)
holdings_bt = ac_holdings(k_bt, X0_bt, T_bt, times_bt)
trades_bt = -np.diff(holdings_bt)
trades_twap = np.full(N_bt, X0_bt / N_bt)

n_simulations = 100
np.random.seed(42)

ac_costs = []
twap_costs = []

for _ in range(n_simulations):
    start_idx = np.random.randint(0, len(spy_intraday) - N_bt)
    window_prices = spy_intraday['Close'].iloc[start_idx:start_idx+N_bt].values
    if len(window_prices) < N_bt:
        continue
    ref_price = window_prices[0]
    ac_cost_sim = np.sum(trades_bt * (ref_price - window_prices))
    twap_cost_sim = np.sum(trades_twap * (ref_price - window_prices))
    ac_costs.append(ac_cost_sim)
    twap_costs.append(twap_cost_sim)

ac_costs = np.array(ac_costs)
twap_costs = np.array(twap_costs)

print(f"\nBacktest over {len(ac_costs)} random 5h windows:")
print(f"{'Strategy':<10} {'Mean cost':<15} {'Std':<15}")
print("-" * 40)
print(f"{'TWAP':<10} ${np.mean(twap_costs):<12.2f}  ${np.std(twap_costs):<12.2f}")
print(f"{'AC':<10} ${np.mean(ac_costs):<12.2f}  ${np.std(ac_costs):<12.2f}")

fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(twap_costs, bins=30, alpha=0.5, label=f'TWAP (σ=${np.std(twap_costs):.0f})', color='steelblue')
ax.hist(ac_costs, bins=30, alpha=0.5, label=f'AC (σ=${np.std(ac_costs):.0f})', color='seagreen')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Execution cost ($)')
ax.set_ylabel('Frequency')
ax.set_title(f'Distribution of execution costs ({len(ac_costs)} historical windows)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('backtest_distribution.png', dpi=150)
plt.show()

# ══════════════════════════════════════════
# 8. INTRADAY EXECUTION ANALYSIS
# ══════════════════════════════════════════

print("\n" + "="*60)
print("INTRADAY EXECUTION: cost vs time of day")
print("="*60)

spy_hd = yf.Ticker("SPY").history(period="60d", interval="15m").reset_index()
spy_hd['hour'] = spy_hd['Datetime'].dt.hour + spy_hd['Datetime'].dt.minute / 60
spy_hd['returns'] = np.log(spy_hd['Close'] / spy_hd['Close'].shift(1))

bins = np.arange(9.5, 16.5, 0.5)
spy_hd['bin'] = pd.cut(spy_hd['hour'], bins=bins, labels=bins[:-1])

intraday_stats = spy_hd.groupby('bin', observed=True).agg(
    volatility=('returns', 'std'),
    avg_volume=('Volume', 'mean'),
    avg_price=('Close', 'mean')
).reset_index()

intraday_stats['sigma_per_bin'] = intraday_stats['volatility'] * intraday_stats['avg_price']
intraday_stats['spread'] = 0.01
intraday_stats['eta_bin'] = intraday_stats['spread'] / (2 * intraday_stats['avg_volume'] * 4)

X0_intra = 100_000
T_intra = 0.5
N_intra = 6
lam_intra = 1e-8

costs_by_hour = []
for _, row in intraday_stats.iterrows():
    if pd.isna(row['sigma_per_bin']) or pd.isna(row['eta_bin']):
        continue
    k = np.sqrt(lam_intra * row['sigma_per_bin']**2 / row['eta_bin'])
    c, v = strategy_stats(k, X0_intra, T_intra, row['sigma_per_bin'], row['eta_bin'], N_intra)
    costs_by_hour.append({'hour': float(row['bin']), 'cost': c, 'variance': v})

costs_df = pd.DataFrame(costs_by_hour)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(costs_df['hour'], costs_df['cost'], width=0.4, color='steelblue', edgecolor='black')
ax1.set_xlabel('Time of day (hours, ET)')
ax1.set_ylabel('Expected execution cost ($)')
ax1.set_title(f'Cost of executing {X0_intra:,} SPY shares by time of day')
ax1.grid(alpha=0.3, axis='y')

ax2.plot(intraday_stats['bin'].astype(float), intraday_stats['avg_volume']/1e6, 'o-', color='seagreen', label='Volume', linewidth=2)
ax2.set_xlabel('Time of day')
ax2.set_ylabel('Avg volume per 15-min bin (M shares)', color='seagreen')
ax2.tick_params(axis='y', labelcolor='seagreen')
ax2b = ax2.twinx()
ax2b.plot(intraday_stats['bin'].astype(float), intraday_stats['sigma_per_bin'], 'o-', color='crimson', linewidth=2)
ax2b.set_ylabel('Volatility ($)', color='crimson')
ax2b.tick_params(axis='y', labelcolor='crimson')
ax2.set_title('Intraday liquidity and volatility profile')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('intraday_analysis.png', dpi=150)
plt.show()

print(f"\nBest time to execute: hour {costs_df.loc[costs_df['cost'].idxmin(), 'hour']}")
print(f"Worst time to execute: hour {costs_df.loc[costs_df['cost'].idxmax(), 'hour']}")
print(f"Cost difference: {(costs_df['cost'].max() / costs_df['cost'].min() - 1) * 100:.1f}%")