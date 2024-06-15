import networkx as nx
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib.patches
import math
from scipy.stats import chisquare
import numpy as np
import mpmath
import time
import pickle
import os
import scipy
import seaborn

matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['lines.markersize'] = 7
matplotlib.rcParams['figure.autolayout'] = True

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 10
colors = seaborn.color_palette('rocket')
colors.reverse()
markers = ['o', 's', 'p', 'd', '*']


def cdf_gamma(a, b, x):
    return float(mpmath.gammainc(a, 0, x * b, regularized = True))

def pdf_gamma(a, b, x):
    return b**a/gamma(a) * x**(a-1)*math.exp(-b*x)


def three_parameter_fit(k, mid_x, y, parameters):
    besta, bestb = 0, 0
    best_chi = math.inf
    for a in parameters:
        for b in [a/k]:
            realised = [cdf_gamma(a, b, x) for x in mid_x]
            y = y / sum(y) * sum(realised)

            # realised = np.divide(realised, sum(realised))

            chi, p_value = chisquare(y, realised)
            if chi < best_chi:
                besta = a
                bestb = b
                best_chi = chi
    print('chi =', best_chi)
    return besta, bestb

def poisson(n, k, labda):
    return (k * labda)**n/math.factorial(n) * math.exp(-k * labda)


points = 800000
resolution = 0.1

k_max = 50

nbins = 75

area = pickle.load(open(str('area_points_till_far=' + str(points) + 'resolution=' + str(resolution) + '.p'), 'rb'))

parameters = np.arange(1, 25, 0.5)

mink = [3.3, 6.8, 11, 14.5, 20.5]
maxk = [3.7, 7.5, 11.8, 15.5, 21.5]

minleqk = mink
maxleqk = maxk

mink = [3.45, 6.1, 8.55, 10.9, 15.5]
maxk = [3.65, 6.3, 8.75, 11.1, 15.7]

minleqk = [3.3, 6.9, 10.9, 15.0, 21]
maxleqk = [3.7, 7.4, 11.2, 15.5, 21.5]

delta = 100

prec = 0.05

lu = 0.1
labda = 1

ak = [3.53, 7.19, 11.06, 15.21, 21.17]
bk = [3.53, 7.19/2, 11.06/3, 15.21/4, 21.17/5]

# ------------------ NORMAL SIZE DISTRIBUTION ----------------------
# for k in [49]:
#     print(k)
#     area[k] = area[k] / (sum(area[k])/len(area[k]))
#     y, x = np.histogram(area[k], density=True, bins = 'auto')
#     # plt.hist(area[k], density = True, bins = nbins, alpha = 0.3, label = str('k =' +  str(k+1)), color = colors[k], cumulative=True)
#     mid_x = [(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)]
#     X = np.arange(0, 3, 3/delta)
#     ynew = y
#     y = np.cumsum(y)/sum(y)
#     besta, bestb = three_parameter_fit(1, mid_x, y, np.arange(mink[k], maxk[k], prec))
#
#     xnew = [x for i, x in enumerate(mid_x) if i % 2 != 0]
#     ynew = [y for i, y in enumerate(y) if i % 2 != 0]
#
#     while len(xnew) > 100:
#         xnew = [x for i, x in enumerate(xnew) if i % 2 != 0]
#         ynew = [y for i, y in enumerate(ynew) if i % 2 != 0]
#
#     plt.scatter(xnew, ynew, color = colors[k], marker = markers[k], label = str('$k = $ ' +  str(k+1)), facecolors='none')
#     plt.plot(X, [cdf_gamma(besta, bestb, x) for x in X], color = colors[k])
#     print('For k =', k + 1, 'the best parameters are a =', besta, 'b =', bestb)
#     # plt.plot(X, [pdf_poisson(1, x) for x in X], label = 'Poisson approximation')
# plt.legend()
# plt.xlim((-0.1, 3.1))
# plt.xlabel('Area size')
# plt.savefig('Ak.png')
# plt.show()

# ---------------- SUM OF SIZE DISTRIBUTIONS ------------------------------
print('Now we look at the cumulative area distribution')
area_cumulative = np.zeros(points)
blub = 0
for k in range(5):
    area_cumulative = area_cumulative + area[k]
    y, x = np.histogram(area_cumulative, density=True, bins = 'auto')
    # plt.hist(area_cumulative, density = True, bins = len(x)-1, alpha = 0.3, label = str('k =' +  str(k+1)), color = colors[k], cumulative=True)
    mid_x = [(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)]

    y = np.cumsum(y)/sum(y)

    X = np.arange(0, max(x), max(x)/delta)
    besta, bestb = ak[k], bk[k]


    # besta, bestb = three_parameter_fit(k+1, mid_x, y, np.arange(minleqk[k], maxleqk[k], prec))

    realised = [pdf_gamma(besta, bestb, x) for x in mid_x]

    print('For k <=', k + 1, 'the best parameters are a =', besta, 'b =', bestb)

    xnew = [x for i, x in enumerate(mid_x) if i % 2 != 0]
    ynew = [y for i, y in enumerate(y) if i % 2 != 0]

    while len(xnew) > 100:
        xnew = [x for i, x in enumerate(xnew) if i % 2 != 0]
        ynew = [y for i, y in enumerate(ynew) if i % 2 != 0]

    plt.scatter(xnew, ynew, color = colors[k], marker = markers[k], label = str('$k = $ ' +  str(k+1)), facecolors='none')
    plt.plot(X, [cdf_gamma(besta, bestb, x) for x in X], color = colors[k])
    plt.ylabel('P$(X_{\leq k} \leq x)$')
    if k == 49:
        pickle.dump(area_cumulative, open(str('area_cumulative_50.p'),'wb'), protocol=4)



    # degree_sim = np.zeros(15)
    # print(len(x), len(y))
    # print(sum(y)*(x[2]-x[1]))
    # deltax = (x[1] - x[0])
    # for n in range(15):
    #     for i in range(len(mid_x)):
    #         a = x[i]
    #         p = y[i]
    #         degree_sim[n] += (labda * a)**n/math.factorial(n) * math.exp(-labda * a) * p
    # print(degree_sim)
    #
    # plt.scatter(range(15), degree_sim/5, color = colors[blub], label = str('$k = $' +  str(k+1)), facecolors='none')
    # plt.plot(range(15), [poisson(n, k + 1, labda) for n in range(15)])
    blub += 1
plt.xlabel('Area size')
plt.legend()
plt.savefig('Aleqk.png')
plt.show()

# area_cumulative = pickle.load(open(str('area_cumulative_50.p'), 'rb'))
# y, x = np.histogram(area_cumulative, density=True, bins = 'auto')
# # plt.hist(area_cumulative, density = True, bins = len(x)-1, alpha = 0.3, label = str('k =' +  str(k+1)), color = colors[k], cumulative=True)
# mid_x = [(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)]
# X = np.arange(min(x), max(x), max(x)/delta)
#
# besta, bestb = three_parameter_fit(50, mid_x, y, np.arange(100, 1000, prec))
#
# print('For k <=', 50, 'the best parameters are a =', besta, 'b =', bestb)
# y = np.cumsum(y)/sum(y)
#
# xnew = [x for i, x in enumerate(mid_x) if i % 2 != 0]
# ynew = [y for i, y in enumerate(y) if i % 2 != 0]
#
# while len(xnew) > 100:
#     xnew = [x for i, x in enumerate(xnew) if i % 2 != 0]
#     ynew = [y for i, y in enumerate(ynew) if i % 2 != 0]
#
# plt.scatter(xnew, ynew, color = colors[0], marker = markers[0], label = str('$k = $ ' + str(50)), facecolors='none')
# plt.plot(X, [cdf_gamma(besta, bestb, x) for x in X], color = colors[0])
#
# plt.xlabel('Area size')
# plt.legend()
# plt.savefig('Aleqk.png')
# plt.show()