import math

import matplotlib.pylab as pylab
import numpy as np
from matplotlib import pyplot as plt

params = {'legend.fontsize': '18',
          'axes.labelsize': '20',
          'axes.titlesize': '20',
          'xtick.labelsize': '20',
          'ytick.labelsize': '20',
          'lines.markersize': 8,
          'figure.autolayout': True}
pylab.rcParams.update(params)

colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']


def distance(x, y, xbs, ybs):
    x = np.minimum((x - np.array(xbs)) % xDelta, (np.array(xbs) - x) % xDelta)
    y = np.minimum((y - np.array(ybs)) % yDelta, (np.array(ybs) - y) % yDelta)
    return np.sqrt(x ** 2 + y ** 2)


def analytic(d):
    labda = lu / lbs
    first_term = math.gamma(d + ak[k]) / (math.gamma(d + 1) * math.gamma(ak[k]))
    second_term = ak[k] ** (ak[k]) * (k * labda) ** d / (k * labda + ak[k]) ** (ak[k] + d)
    return first_term * second_term


xmin, xmax = 0, 100
ymin, ymax = 0, 100

xDelta, yDelta = xmax - xmin, ymax - ymin

ak = {1: 3.53, 2: 7.19, 3: 11.06, 4: 15.21, 5: 21.17}

lu = 0.1
lbs = 0.01
data = []
for _ in range(100):
    n_users = np.random.poisson(lu * (xDelta * yDelta))
    n_bs = np.random.poisson(lbs * (xDelta * yDelta))
    pos_u = np.random.uniform(xmin, xmax, (n_users, 2))
    pos_bs = np.random.uniform(xmin, xmax, (n_bs, 2))
    xbs = [x for x, y in pos_bs]
    ybs = [y for x, y in pos_bs]

    degrees = np.zeros(n_bs)
    k = 5

    for u in range(n_users):
        xu, yu = pos_u[u]
        distances = distance(xu, yu, xbs, ybs)
        sorted_bs = np.argsort(distances)
        for b in sorted_bs[:k]:
            degrees[b] += 1

    for blub in degrees:
        data.append(blub)

plt.hist(data, bins=25, alpha=0.7, color=colors[-1], label='Real', density=True)
plt.plot(range(int(max(data))), [analytic(d) for d in range(int(max(data)))], color=colors[2], label='Poisson-Gamma',
         linewidth=3)
plt.xlabel('Degree')
plt.legend(loc='upper right')
plt.xlim((0, 125))
plt.savefig('k=5.png', dpi=500)
plt.show()
