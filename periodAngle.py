import fit_black_box as bb

def linear(t, m, b):
    return m*t + b

def quadratic(t, C, B, A):
    return C*t**2 + B*t + A

def expon(t, a, b):
    return a*bb.np.exp(-t/b)

x = bb.np.array([-1.39626, -1.0472, -0.69813, -0.52360, -0.34907, -0.17453, 0.17453, 0.34907, 0.52360, 0.69813, 1.0472, 1.39626])
y = bb.np.array([1.4664, 1.3896, 1.3344, 1.3116, 1.2952, 1.284, 1.2796, 1.2964, 1.312, 1.3332, 1.394, 1.4712])
xerr = 0.0087
yerr = 0.03

bb.plot_fit(quadratic, x, y, xerr, yerr,
            init_guess=(1e-4, 0, 1.3), 
            y_min=1.25, y_max=1.5)