import math

import matplotlib.pylab as pylab
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import seaborn
from math import sin, radians
import matplotlib.patches as mpatches

matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['legend.fontsize'] = 18  # using a size in points
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['lines.markersize'] = 7
matplotlib.rcParams['figure.autolayout'] = True
# matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
markers = ['o', 's', 'p', 'd', '*']

colors = seaborn.color_palette('hsv', n_colors=10)
colors.reverse()


def distlist(coord1, xs, ys):
    xdelta = np.max(xs)
    ydelta = np.max(ys)
    x = np.minimum((coord1[0] - xs) % xdelta, (xs - coord1[0]) % xdelta)
    y = np.minimum((coord1[1] - ys) % ydelta, (ys - coord1[1]) % ydelta)
    return np.sqrt(x ** 2 + y ** 2)


xmin, xmax = 0, 1000
ymin, ymax = 0, 1000

xDelta, yDelta = xmax - xmin, ymax - ymin

ak = {1: 3.53, 2: 7.19, 3: 11.06, 4: 15.21, 5: 21.17, 50: 50}

lu = 0.1
lbs = 0.01
Kaas = [1, 2, 3, 4, 5]
data = {i: [] for i in Kaas}
n_bs = np.random.poisson(lbs * (xDelta * yDelta))
pos_bs = np.random.uniform(xmin, xmax, (n_bs, 2))
xbs = [x for x, y in pos_bs]
ybs = [y for x, y in pos_bs]

BS = 1
delta = 50
x_mesh = np.arange(0, xmax, delta)
y_mesh = np.arange(0, ymax, delta)
xs, ys = np.meshgrid(x_mesh, y_mesh, sparse=False, indexing='xy')

rescale_x = np.max(xbs) / np.max(xs)
rescale_y = np.max(xbs) / np.max(ys)

area = {i: {j: 0 for j in range(n_bs)} for i in Kaas}

print(len(xs.flatten()))

for x, y in zip(xs.flatten(), ys.flatten()):
    dists = np.argsort(distlist((x, y), xbs, ybs))
    for k in Kaas:
        area[k][dists[k]] += 1

print(area)
