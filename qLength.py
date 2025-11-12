import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

L = np.array([0.20, 0.24, 0.28, 0.32, 0.36, 0.40])
Q = np.array([346, 376, 440, 456, 516, 551])
Q_err = 20
L_err = 0.001

def linear(L, m, b):
    return m * L + b

params, cov = curve_fit(linear, L, Q)
m, b = params
perr = np.sqrt(np.diag(cov))

L_dense = np.linspace(0, max(L), 300)
Q_fit = linear(L_dense, m, b)

plt.figure(figsize=(8,6))
plt.rcParams.update({'font.size': 30})
plt.errorbar(L, Q, xerr=L_err, yerr=Q_err, fmt='o', color='red', markersize=1, label='Measured data')
plt.plot(L_dense, Q_fit, color='blue', linewidth=2.5, label=f'Fit: Q = {m:.1f}L + {b:.1f}\n(m ± {perr[0]:.1f}, b ± {perr[1]:.1f})')
plt.xlabel("Length [m]")
plt.ylabel("Q Factor")
plt.title("Q Factor vs. Length")
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

residuals = Q - linear(L, m, b)
plt.figure(figsize=(8,4))
plt.rcParams.update({'font.size': 30})
plt.errorbar(L, residuals, xerr=L_err, yerr=Q_err, fmt='o', color='black', markersize=1)
plt.axhline(0, color='blue', linestyle='--', linewidth=2.0)
plt.xlabel("Length [m]")
plt.ylabel("Residual (Data - Fit)")
plt.title("Residuals of Q vs. Length Fit")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"m (slope) = {m:.2f} ± {perr[0]:.2f}")
print(f"b (intercept) = {b:.2f} ± {perr[1]:.2f}")