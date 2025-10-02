import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

data = np.loadtxt("pendulum_extended.txt")
t = data[:, 0]
angles = data[:, 1]

angles = (angles - 90) * np.pi / 180

peaks, _ = find_peaks(angles)
peak_times = t[peaks]
peak_values = np.abs(angles[peaks])

def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

params, cov = curve_fit(exp_decay, peak_times, peak_values, p0=[peak_values[0], 1.0])
A_fit, tau = params

perr = np.sqrt(np.diag(cov))
A_err, tau_err = perr

periods = np.diff(peak_times)
T_mean = np.mean(periods)
Q = (np.pi * tau) / T_mean

yerr = 0.00872664626 * np.ones_like(peak_values)

print(f"Tau = {tau:.3f} ± {tau_err:.3f} s")
print(f"Mean Period = {T_mean:.3f} s")
print(f"Q Factor = {Q:.3f}")

plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.size': 30})
plt.errorbar(peak_times, peak_values, yerr=yerr, fmt='o', markersize = 3,
             label="Peak amplitudes (with error bars)", color="red", 
             ecolor="red", capsize=0, zorder=1)

plt.plot(peak_times, exp_decay(peak_times, A_fit, tau),
         label=f"Fit: τ={tau:.2f}±{tau_err:.2f}s, Q={Q:.2f}",
         color="blue", linewidth=2.5, zorder=2)

plt.xlabel("Time [s]")
plt.ylabel("Amplitude [rad]")
plt.title("Amplitude vs. Time")
plt.legend()
plt.grid(True)
plt.show()