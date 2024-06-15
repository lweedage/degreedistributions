import numpy as np
import matplotlib.pyplot as plt
import math

import scipy.stats
from scipy.special import gamma

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 10
markers = ['o', 's', 'p', 'd', '*']


def find_area_leq(i,n):
    area = 0
    for k in range(2* n - 1):
        area += s[(i - n + 2 + k) % pointsBS]
    return 0.5 * area

def distribution_A(x):
    return 4* x * labda**2 * math.exp(-labda * 2 * x) # I don't understand why this is 4...

def distribution_A_leq(x, k, lbs):
    return (2 * lbs)**(2*k)/gamma(2*k) * x**(2*k-1) * math.exp(-2*lbs *x)

def cdf_A(t):
    return (1 - math.exp(-2*labda * t) *(1 + 2*labda * t))  # if it is 4 in the distribution it also works for the cdf

length = 1e5
lbs = 0.1
lu = 0.5
labda = lu/lbs

pointsBS = int(lbs * length)
pointsU = int(lu * length)

length = int(length)

xbs = length * scipy.stats.uniform.rvs(0, 1, ((pointsBS, 1)))  # x coordinates of Poisson points
xu = length * scipy.stats.uniform.rvs(0, 1, ((pointsU, 1)))  # x coordinates of Poisson points
xbs = sorted([xbs[i,0] for i in range(len(xbs))])

s = np.zeros(pointsBS)

for i in range(pointsBS):
    s[i] = (xbs[i] - xbs[i-1]) % length

def find_area(i, n):
    return 0.5*(s[(i - n + 1) % pointsBS] + s[(i + n) % pointsBS])

A = np.zeros((pointsBS, 5))
Aleq = np.zeros((pointsBS, 5))

delta = 1000

X = np.arange(0, max(s), max(s)/delta)
X2 = np.arange(0, 2 * max(s), 2 * max(s)/delta)

for n in [1,2,3,4,5]:
    A[:, n-1] = [find_area(i, n) for i in range(pointsBS)]
    Aleq[:, n-1] = Aleq[:, n-2] + A[:, n-1]
    plt.hist(Aleq[:, n-1], alpha = 0.3, density = True, color= colors[n - 1], bins = 30)
    plt.plot(X2, [distribution_A_leq(x, n, lbs) for x in X2], label = str('k = ' + str(n)), color = colors[n-1])
plt.legend()
plt.show()


def find_distance(xu, xbs):
    x = np.minimum((xu-np.array(xbs))%length, (np.array(xbs)-xu)%length)
    return x

def find_bs(xu, xbs, max_connections):
    indices = find_distance(xu, xbs).argsort()
    return indices[:max_connections]

def find_distance_to_bs(xu, xbs, max_connections):
    indices = find_distance(xu, xbs)
    indices.sort()
    return indices[:max_connections]

def degree_distr(n, j, lbs, lu):
    return (2*lbs)**(2*j)/gamma(2*j) * (lu**n)/math.factorial(n) * gamma(2*j+n)/(lu + 2*lbs)**(2*j+n)

def poisson(n, k, lbs, lu):
    labda = lu/lbs
    return (k * labda)**n/math.factorial(n) * math.exp(-k * labda)

degree = np.zeros((pointsBS, 5))

fig, ax = plt.subplots()

for i in range(pointsU):
    bs = find_bs(xu[i], xbs[i-1], 5)
    for j in range(5):
        degree[bs[j], j:] += 1

X = np.arange(0, np.max(degree), 1)
x1 = np.arange(0, np.max(degree), 5)

for j in range(5):
    k = j
    plt.hist(degree[:, j], alpha = 0.3, density = True, label = str('k = ' + str(k + 1)), color= colors[j], bins = 25)
    plt.plot(X, [degree_distr(x, k + 1, lbs, lu) for x in X],  color= colors[j])
    plt.scatter(X, [poisson(x, k + 1, lbs, lu) for x in X], color = colors[j])
print(lbs, lu)
plt.xlim((0, np.max(degree)))
plt.legend()
plt.show()