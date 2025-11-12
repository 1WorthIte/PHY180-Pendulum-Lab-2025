import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

L = np.array([0.20, 0.24, 0.28, 0.32, 0.36, 0.40])
T = np.array([0.907, 0.94, 1.06, 1.10, 1.21, 1.39])
T_err = 0.03
L_err = 0.0005

def power_law(L, k, n):
    return k * L**n

params, cov = curve_fit(power_law, L, T, p0=(2.0, 0.5))
k, n = params
perr = np.sqrt(np.diag(cov))

L_dense = np.linspace(0, max(L), 200)
T_fit = power_law(L_dense, k, n)

plt.figure(figsize=(8,6))
plt.rcParams.update({'font.size': 30})
plt.errorbar(L, T, xerr=L_err, yerr=T_err, fmt='o', color='red', markersize=3, label='Measured data')
plt.plot(L_dense, T_fit, color='blue',
         label=f'Fit: T = {k:.3f} L^{n:.3f}\n(k ± {perr[0]:.3f}, n ± {perr[1]:.3f})')
plt.xlabel("Length [m]")
plt.ylabel("Period [s]")
plt.title("Period vs. Length")
plt.legend()
plt.xlim(0, None)
plt.ylim(0, None)
plt.show()

plt.figure(figsize=(8,6))
plt.rcParams.update({'font.size': 30})
plt.errorbar(np.log(L), np.log(T), xerr=L_err/L, yerr=T_err/T, fmt='o', color='red', markersize=1)
plt.plot(np.log(L_dense), np.log(T_fit), color='blue')
plt.xlabel("ln (L)")
plt.ylabel("ln (T)")
plt.title("Log–Log Plot: Period vs. Length")
plt.grid(True)
plt.xlim(-2, None)
plt.ylim(-0.5, None)
plt.show()

# Residuals plot
T_predicted = power_law(L, k, n)
residuals = T - T_predicted

plt.figure(figsize=(8,6))
plt.rcParams.update({'font.size': 30})
plt.errorbar(L, residuals, xerr=L_err, yerr=T_err, fmt='o', color='red', markersize=3)
plt.axhline(y=0, color='blue', linestyle='-', linewidth=2)
plt.xlabel("Length [m]")
plt.ylabel("Residuals [s]")
plt.title("Residuals: Period vs. Length")
plt.grid(True)
plt.show()

print(f"k = {k:.3f} ± {perr[0]:.3f}")
print(f"n = {n:.3f} ± {perr[1]:.3f}")