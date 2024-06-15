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

def create_hex_grid(dist):
    xcoords = np.arange(0, 100, dist * 1.5)
    ycoords = np.arange(0, 100, dist * sin(radians(60)) * 0.5)

    def create_grid(xcoords, ycoords, dist):
        points = []
        for count, y in enumerate(ycoords):
            if count % 2 == 0:  # even
                for x in xcoords:
                    points.append((x, y))
            else:
                for x in xcoords:
                    points.append((x + (dist * 0.75), y))
        return points

    grid_points = create_grid(xcoords, ycoords, dist)
    return grid_points
def distlist(coord1, xs, ys):
    xdelta = np.max(xs)
    ydelta = np.max(ys)
    x = np.minimum((coord1[0] - xs) % xdelta, (xs - coord1[0]) % xdelta)
    y = np.minimum((coord1[1] - ys) % ydelta, (ys - coord1[1]) % ydelta)
    return np.sqrt(x ** 2 + y ** 2)


Kaas = range(10)

dist = 1
number_of_bs = 100
np.random.seed(10)
x_bs, y_bs = np.random.uniform(0, 100, number_of_bs), np.random.uniform(0, 100,
                                                                          number_of_bs)

BS = 50


delta = 0.1
x_mesh = np.arange(x_bs[BS] - 25, x_bs[BS] + 25, delta)
y_mesh = np.arange(y_bs[BS] - 25, y_bs[BS] + 25, delta)

xs, ys = np.meshgrid(x_mesh, y_mesh, sparse=False, indexing='xy')

c = []

rescale_x = np.max(x_bs) / np.max(xs)
rescale_y = np.max(y_bs) / np.max(ys)

list_x, list_y, c = [], [], []

for x, y in zip(xs.flatten(), ys.flatten()):
    dists = np.argsort(distlist((x, y), x_bs, y_bs))
    found = False
    for k in Kaas:
        if dists[k] == BS:
            list_x.append(x)
            list_y.append(y)
            c.append(colors[k])

# for k in Kaas:
fig, ax = plt.subplots()

plt.scatter(list_x, list_y, c=c, alpha=0.75, s=0.5, zorder=1)
plt.scatter(x_bs, y_bs, color='black', alpha=0.5, s=20, zorder=2)

plt.scatter(x_bs[BS], y_bs[BS], color='red', s=20, zorder=3)
plt.xlim(x_bs[BS] - 25, x_bs[BS] + 25)
plt.ylim(y_bs[BS] - 25, y_bs[BS] + 25)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# plt.axis('equal')

patches = []
for k in Kaas:
    patches.append(mpatches.Patch(color=colors[k], alpha = 0.75, label=f'$k={k+1}$'))

ax.legend(handles = patches, loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('colors_PPP.pdf')
plt.show()
