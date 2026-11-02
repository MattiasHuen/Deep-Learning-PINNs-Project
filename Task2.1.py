# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:23:57 2026

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 0.5, 10)
mu = 5
m = 1
k = 500

w_0 = np.sqrt(k / m)
sigma = mu / (2 * m)
omega = np.sqrt(w_0**2 - sigma**2)
phi = np.arctan(-sigma / omega)
A = 1 / (2 * np.cos(phi))

x_t = np.exp(-sigma * t) * 2 * A * np.cos(phi + omega * t)

p = np.polyfit(t, x_t, 9)
tfit = np.linspace(0, 0.5, 300)
xfit = np.polyval(p, tfit)

plt.scatter(t, x_t, label="Data points")
plt.plot(tfit, xfit, label="Fit",linestyle="--")
plt.xlabel("t [s]")
plt.ylabel("x(t) [m]")
plt.legend()
plt.show()