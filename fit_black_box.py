# -*- coding: utf-8 -*-
"""
This program will find the best fit of a given function to a given set
of data (including errorbars). It prints the results, with uncertainties.
Then it plots the graph and displays it to the screen, and also saves
a copy to a file in the local directory. Below the main graph is a 
residuals graph, the difference between the data and the best fit line.

There is also a function which will load data from a file. More convenient.
The first line of the file is ignored (assuming it's the name of the variables).
After that the data file needs to be formatted: 
number space number space number space number newline
Do NOT put commas in your data file!! You can use tabs instead of spaces.
The data file should be in the same directory as this python file.
The data should be in the order:
x_data y_data x_uncertainty y_uncertainty
"""


import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
from pylab import loadtxt

def load_data(filename):
    data=loadtxt(filename, usecols=(0,1,2,3), skiprows=1, unpack=True)
    return data


def plot_fit(my_func, xdata, ydata, xerror=None, yerror=None, init_guess=None, font_size=14,
             xlabel="Angle (rad)", ylabel="Period (s)", 
             title="Period vs. Angle",
             y_min=None, y_max=None):    
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['figure.figsize'] = 10, 7  # a bit shorter since we only need one plot
               
    popt, pcov = optimize.curve_fit(my_func, xdata, ydata, sigma=yerror, p0=init_guess, absolute_sigma=True)
    puncert = np.sqrt(np.diagonal(pcov))
    
    print("Best fit parameters, with uncertainties, but not rounded off properly:")
    for i in range(len(popt)):
        print(popt[i], "+/-", puncert[i])
    
    start = min(xdata)
    stop = max(xdata)
    xs = np.linspace(start, stop, 1000) 
    curve = my_func(xs, *popt) 
    
    # ---- single plot only ----
    fig, ax1 = plt.subplots()
    
    ax1.errorbar(xdata, ydata, yerr=yerror, xerr=xerror, fmt=".", label="measured values", color="red")
    ax1.plot(xs, curve, label="best fit", color="blue")
    
    ax1.legend(loc='upper right')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    
    # custom y-axis limits if provided
    if y_min is not None and y_max is not None:
        ax1.set_ylim(y_min, y_max)
    
    fig.tight_layout()
    plt.show()
    fig.savefig("graph.png")

    return None
