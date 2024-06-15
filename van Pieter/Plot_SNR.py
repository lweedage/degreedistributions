import matplotlib.pyplot as plt
import mpmath
import scipy.stats
import numpy as np
import networkx as nx
import math
import scipy.special
from scipy.integrate import quad
import seaborn
import matplotlib


matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['lines.markersize'] = 7
matplotlib.rcParams['figure.autolayout'] = True

colors = seaborn.color_palette('rocket')
colors.reverse()

def fsnr(x, k, phi, alpha, c):
    if x < c:
        return 2 * (phi * x**(-2/alpha))**k / (alpha *x*math.factorial(k-1)) * math.exp(-phi * x**(-2/alpha))
    if x == c:
        return 1 - mpmath.gammainc(k, lbs * math.pi)/math.factorial(k-1)
    else:
        return 0

alpha = 2
lbs = 10**(-2)
c = 10**(3.5)
phi = lbs * math.pi * c**(2/alpha)

max = 50
delta = 1000
x = np.arange(0, max, max/delta)
fig, ax = plt.subplots()

y = np.zeros((5, delta))

for k in [1, 2, 3, 4, 5]:
    y[k-1] = [fsnr(10**(X/10), k, phi, alpha, c) for X in x]
    plt.plot(x, y[k-1] , label = str('$j =$ ' + str(k)), color = colors[k-1])
plt.xlabel('SNR (in dB)')
plt.ylabel('$f_{SNR_j}$')
plt.legend()
# plt.ylim(0, 0.02)
plt.show()

