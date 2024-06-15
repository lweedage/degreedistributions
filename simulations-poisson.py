import math
import mpmath
import matplotlib.pylab as pylab
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import seaborn

matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['legend.fontsize'] = 18  # using a size in points
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['lines.markersize'] = 7
matplotlib.rcParams['figure.autolayout'] = True
# matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
markers = ['o', 's', 'p', 'd', '*']

colors = seaborn.color_palette('rocket')
colors.reverse()


def distance(x, y, xbs, ybs):
    x = np.minimum((x - np.array(xbs)) % xDelta, (np.array(xbs) - x) % xDelta)
    y = np.minimum((y - np.array(ybs)) % yDelta, (np.array(ybs) - y) % yDelta)
    return x**2 + y**2


def poisson_gamma(d, k):
    labda = lu / lbs
    first_term = mpmath.gamma(d + ak[k]) / (mpmath.gamma(d + 1) * mpmath.gamma(ak[k]))

    log_teller = ak[k] * math.log(ak[k]) + d * math.log(k * labda)
    log_noemer = (ak[k] + d) * math.log(k * labda + ak[k])
    log_second_term = log_teller - log_noemer
    second_term = math.exp(log_second_term)

    return first_term * second_term


def poisson_erlang(n, k):
    labda = lu / lbs
    first_term = mpmath.gamma(2 * k + n) / (math.factorial(n) * mpmath.gamma(2 * k))
    # second_term = (2 * lbs) ** (2 * k) * lu ** n / ((2 * lbs + lu) ** (2 * k + n))

    log_teller = 2 * k * math.log(2 * lbs) + n * math.log(lu)
    log_noemer = (2 * k + n) * math.log(2 * lbs + lu)
    second_term = math.exp(log_teller - log_noemer)
    return first_term * second_term


xmin, xmax = 0, 500
ymin, ymax = 0, 500

xDelta, yDelta = xmax - xmin, ymax - ymin

ak = {1: 3.53, 2: 7.19, 3: 11.06, 4: 15.21, 5: 21.17, 50: 50}

lu = 0.1
lbs = 0.01
Kaas = [1, 5, 50]
data = {i: [] for i in Kaas}
for _ in range(2):
    n_users = np.random.poisson(lu * (xDelta * yDelta))
    n_bs = np.random.poisson(lbs * (xDelta * yDelta))
    pos_u = np.random.uniform(xmin, xmax, (n_users, 2))
    pos_bs = np.random.uniform(xmin, xmax, (n_bs, 2))
    xbs = [x for x, y in pos_bs]
    ybs = [y for x, y in pos_bs]

    print(n_users)

    degrees = np.zeros(n_bs)

    for k in Kaas:
        print(f'k={k}')
        for u in range(n_users):
            xu, yu = pos_u[u]
            distances = distance(xu, yu, xbs, ybs)
            sorted_bs = np.argsort(distances)
            for b in sorted_bs[:k]:
                degrees[b] += 1

        for blub in degrees:
            data[k].append(blub)

index = 0
for k in Kaas:
    plt.hist(data[k], bins=25, alpha=0.5, color=colors[index], label=f'k={k}', density=True)
    plt.plot(range(int(max(data[k]))), [poisson_gamma(d, k) for d in range(int(max(data[k])))], color=colors[index],
             label='Poisson-Gamma',
             linewidth=3, linestyle=':')
    plt.plot(range(int(max(data[k]))), [poisson_erlang(d, k) for d in range(int(max(data[k])))], color=colors[index],
             label='Poisson-Erlang',
             linewidth=3)
    plt.xlabel('Degree')
    plt.ylabel('pdf')
    plt.legend(loc='upper right')
    # plt.xlim((0, 125))
    plt.savefig(f'k={k}.pdf')
    plt.show()
    index += 1
