import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Quadratic model
def quadratic(t, C, B, A):
    return C*t**2 + B*t + A

# Data
x = np.array([-1.39626, -1.0472, -0.69813, -0.52360, -0.34907, -0.17453,
               0.17453, 0.34907, 0.52360, 0.69813, 1.0472, 1.39626])
y = np.array([1.4664, 1.3896, 1.3344, 1.3116, 1.2952, 1.284,
              1.2796, 1.2964, 1.312, 1.3332, 1.394, 1.4712])
xerr = 0.00873*2
yerr = 0.0096

# Fit
params, cov = curve_fit(quadratic, x, y, p0=(1e-4, 0, 1.3))
C, B, A = params
y_fit = quadratic(x, C, B, A)
residuals = y - y_fit

# Residual plot
plt.rcParams.update({'font.size': 25})
plt.errorbar(x, residuals, yerr=yerr, fmt='o', color="black")
plt.axhline(0, linestyle='--', color="blue")
plt.xlabel("Angle [rad]")
plt.ylabel("Residual (Data - Fit) [s]")
plt.title("Residuals of Period vs. Angle")
plt.grid(True)
plt.show()
