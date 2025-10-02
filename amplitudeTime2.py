import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Load data
data = np.loadtxt("pendulum_extended.txt")
t = data[:, 0]
angles = data[:, 1]

# Convert to radians relative to vertical
angles = (angles - 90) * np.pi / 180

# Find peaks
peaks, _ = find_peaks(angles)
peak_times = t[peaks]
peak_values = np.abs(angles[peaks])

# Exponential decay model
def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

# Fit
params, cov = curve_fit(exp_decay, peak_times, peak_values, p0=[peak_values[0], 1.0])
A_fit, tau = params
perr = np.sqrt(np.diag(cov))
A_err, tau_err = perr

# Calculate mean period & Q
periods = np.diff(peak_times)
T_mean = np.mean(periods)
Q = (np.pi * tau) / T_mean

# Error bars (constant angular resolution ~0.5° = 0.0087 rad)
yerr = 0.00872664626 * np.ones_like(peak_values)

print(f"Tau = {tau:.3f} ± {tau_err:.3f} s")
print(f"Mean Period = {T_mean:.3f} s")
print(f"Q Factor = {Q:.3f}")

# Residuals
residuals = peak_values - exp_decay(peak_times, A_fit, tau)

plt.figure(figsize=(12, 4))
plt.rcParams.update({'font.size': 25})
plt.errorbar(peak_times, residuals, yerr=yerr, fmt='o', color="black", markersize=4)
plt.axhline(0, linestyle='--', color="blue", linewidth=2.5)
plt.xlabel("Time [s]")
plt.ylabel("Residual (Data - Fit) [rad]")
plt.title("Residuals of Exponential Fit")
plt.grid(True)
plt.show()