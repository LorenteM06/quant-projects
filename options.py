import numpy as np
from scipy.stats import norm

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

print(bs_call(S=100, K=100, T=1, r=0.05, sigma=0.2))
print(bs_put(S=100, K=100, T=1, r=0.05, sigma=0.2))


def delta_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

print(delta_call(S=100, K=100, T=1, r=0.05, sigma=0.2))

def gamma(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

print(gamma(S=100, K=100, T=1, r=0.05, sigma=0.2))

def vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

print(vega(S=100, K=100, T=1, r=0.05, sigma=0.2))

def theta_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365

print(theta_call(S=100, K=100, T=1, r=0.05, sigma=0.2))

def rho_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * T * np.exp(-r*T) * norm.cdf(d2) / 100

print(rho_call(S=100, K=100, T=1, r=0.05, sigma=0.2))

# Put-call parity verification
call = bs_call(S=100, K=100, T=1, r=0.05, sigma=0.2)
put = bs_put(S=100, K=100, T=1, r=0.05, sigma=0.2)

left  = call - put
right = 100 - 100 * np.exp(-0.05 * 1)

print(f"C - P = {left:.4f}")
print(f"S - Ke^(-rT) = {right:.4f}")

import matplotlib.pyplot as plt

# Rango de precios de la acción
S_range = np.linspace(50, 150, 200)

# Plot Delta vs precio de la acción
deltas = [delta_call(S, K=100, T=1, r=0.05, sigma=0.2) for S in S_range]

plt.figure(figsize=(8, 5))
plt.plot(S_range, deltas, color='steelblue', linewidth=2)
plt.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='S = K = 100')
plt.title('Delta vs Precio de la acción')
plt.xlabel('Precio de la acción (S)')
plt.ylabel('Delta')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('delta_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot Vega vs precio de la acción
vegas = [vega(S, K=100, T=1, r=0.05, sigma=0.2) for S in S_range]

plt.figure(figsize=(8, 5))
plt.plot(S_range, vegas, color='darkorange', linewidth=2)
plt.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='S = K = 100')
plt.title('Vega vs Precio de la acción')
plt.xlabel('Precio de la acción (S)')
plt.ylabel('Vega')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('vega_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot Theta vs tiempo hasta vencimiento
T_range = np.linspace(0.01, 2, 200)

thetas = [theta_call(S=100, K=100, T=t, r=0.05, sigma=0.2) for t in T_range]

plt.figure(figsize=(8, 5))
plt.plot(T_range, thetas, color='crimson', linewidth=2)
plt.title('Theta vs Tiempo hasta vencimiento')
plt.xlabel('Tiempo hasta vencimiento (años)')
plt.ylabel('Theta (€/día)')
plt.grid(alpha=0.3)
plt.savefig('theta_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot Gamma vs precio de la acción
gammas = [gamma(S, K=100, T=1, r=0.05, sigma=0.2) for S in S_range]

plt.figure(figsize=(8, 5))
plt.plot(S_range, gammas, color='seagreen', linewidth=2)
plt.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='S = K = 100')
plt.title('Gamma vs Precio de la acción')
plt.xlabel('Precio de la acción (S)')
plt.ylabel('Gamma')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('gamma_plot.png', dpi=150, bbox_inches='tight')
plt.show()

def mc_call(S, K, T, r, sigma, n=10000):
    Z = np.random.standard_normal(n)
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r*T) * np.mean(payoff)

print(f"Black-Scholes: {bs_call(100, 100, 1, 0.05, 0.2):.4f}")
print(f"Monte Carlo:   {mc_call(100, 100, 1, 0.05, 0.2):.4f}")

# Convergencia de Monte Carlo
simulaciones = [100, 500, 1000, 5000, 10000, 50000]
precios_mc = [mc_call(100, 100, 1, 0.05, 0.2, n) for n in simulaciones]
precio_bs = bs_call(100, 100, 1, 0.05, 0.2)

plt.figure(figsize=(8, 5))
plt.plot(simulaciones, precios_mc, 'o-', color='steelblue', linewidth=2, label='Monte Carlo')
plt.axhline(y=precio_bs, color='crimson', linestyle='--', linewidth=2, label=f'Black-Scholes: {precio_bs:.4f}')
plt.xscale('log')
plt.title('Convergencia Monte Carlo vs Black-Scholes')
plt.xlabel('Número de simulaciones')
plt.ylabel('Precio de la call')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('convergencia_mc.png', dpi=150, bbox_inches='tight')
plt.show()