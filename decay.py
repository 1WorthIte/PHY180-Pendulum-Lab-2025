import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

data = np.loadtxt("pendulum_extended.txt")
t = data[:, 0]
angles = data[:, 1]

for i in range(len(data)):
    angles[i] = (angles[i] - 90) * np.pi / 180

plt.figure(figsize=(12, 6))
plt.plot(t, angles, label="Pendulum Angle", color="red")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.title("Decaying Pendulum Oscillations")

peaks, _ = find_peaks(angles)
peak_times = t[peaks]
peak_values = np.abs(angles[peaks])

def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

params, _ = curve_fit(exp_decay, peak_times, peak_values, p0=[peak_values[0], 1.0])
A_fit, tau = params

plt.plot(t, exp_decay(t, A_fit, tau), label=f"Envelope: tau = {tau:.3f}s", color="blue", linestyle="--")
plt.legend()
plt.grid(True)
plt.show()
print(f"Estimated tau (time constant) = {tau:.3f} s")
print(peaks)