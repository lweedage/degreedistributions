import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import networkx as nx
import math
import scipy.special
import failure_models as fm
import os
import pickle
import seaborn
import matplotlib
import plot_ecdf_lists

matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['lines.markersize'] = 7
matplotlib.rcParams['figure.autolayout'] = True

lijstk = [1, 2, 3, 4, 5]

delta = 25
xMin, xMax = 0, 1500
yMin, yMax = 0, 1500

xDelta = xMax - xMin
yDelta = yMax - yMin
total_bandwidth = 20 * 10 ** 6

alpha = 2
c = 10**(3.5)

labdaBS = 0.01
labdaU = 0.1

lbs = np.arange(1e-6, 1e-1, 1e-1 / delta)
pointsBS = labdaBS * (yDelta * xDelta)
pointsBS = int(pointsBS)
labdaBS = pointsBS / (yDelta * xDelta)

def initialise_graph(points):
    xx = xDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + xMin  # x coordinates of Poisson points
    yy = yDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + yMin  # y coordinates of Poisson points

    x = [xx[i][0] for i in range(points)]
    y = [yy[i][0] for i in range(points)]

    return x, y

def make_graph(xbs, ybs, xpop, ypop, pointsBS, pointsPop):
    G = nx.Graph()
    colorlist = list()
    nodesize = list()
    for node in range(pointsBS):
        G.add_node(node, pos=(xbs[node], ybs[node]))
        colorlist.append('b')
        nodesize.append(20)
    for node in range(pointsPop):
        G.add_node(node + pointsBS, pos=(xpop[node], ypop[node]))
        colorlist.append('g')
        nodesize.append(3)
    return G, colorlist, nodesize

def find_all(pointsPop, pointsBS, G, Gnew, labdaBS, xbs, ybs, xpop, ypop, total_bandwidth, k):
    channel = [0 for i in range(pointsPop)]
    logSNR = [0 for i in range(pointsPop)]
    W = [0 for i in range(pointsPop)]
    for edge in Gnew.edges():
        u, v = edge[0], edge[1] - pointsBS
        channel[v] += find_shannon_capacity(u, v, G, labdaBS, xbs, ybs, xpop, ypop)
        logSNR[v] += math.log2(1 + find_snr(u, v, xbs, ybs, xpop, ypop))
        W[v] += (total_bandwidth / (G.degree(u) * labdaBS))
    return channel, logSNR, W

def from_memory(filename):
    if os.path.exists(filename):
        file = pickle.load(open(filename, 'rb'))
        return file

def find_distance(x, y, xbs, ybs):
    x = np.minimum((x - np.array(xbs)) % xDelta, (np.array(xbs) - x) % xDelta)
    y = np.minimum((y - np.array(ybs)) % yDelta, (np.array(ybs) - y) % yDelta)
    return np.sqrt(x ** 2 + y ** 2)

def find_shannon_capacity(u, v, G, labdaBS, xbs, ybs, xpop, ypop):
    SNR = find_snr(u, v, xbs, ybs, xpop, ypop)
    return (total_bandwidth / (G.degree(u) * labdaBS)) * math.log2(1 + SNR)

def find_snr(u, v, xbs, ybs, xpop, ypop):
    dist = find_distance(xbs[u], ybs[u], xpop[v], ypop[v])
    if dist <= 1:
        return c
    else:
        return c * (dist) ** (-alpha)

def find_bs(xpop, ypop, xbs, ybs, max_connections):
    indices = find_distance(xpop, ypop, xbs, ybs).argsort()
    return indices[:max_connections]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
          '#17becf']
colors = seaborn.color_palette('rocket')
colors.reverse()

markers = ['o', 's', 'p', 'd', '*']
pointsPop = int(labdaU * (yDelta * xDelta))

xbs, ybs = initialise_graph(pointsBS)
xpop, ypop = initialise_graph(pointsPop)

graph, colorlist, nodesize = make_graph(xbs, ybs, xpop, ypop, pointsBS, pointsPop)
fig, ax = plt.subplots()
channels = list()

for k in lijstk:
    channel_sum = list()
    logSNR_sum = []
    W_sum = []


    if os.path.exists(
            str('Simulations/graph' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                k) + '.p')):
        print('The graph for k =', k, 'is already stored in memory')
        G = pickle.load(
            open(str('Simulations/graph' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                 'rb'))
        pointsPop = pickle.load(
            open(str(
                'Simulations/pointsPop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                 'rb'))
        pointsBS = pickle.load(
            open(str(
                'Simulations/pointsBS' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                 'rb'))
        xbs = pickle.load(
            open(str('Simulations/xbs' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                 'rb'))
        ybs = pickle.load(
            open(str('Simulations/ybs' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                 'rb'))
        xpop = pickle.load(
            open(str('Simulations/xpop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                 'rb'))
        ypop = pickle.load(
            open(str('Simulations/ypop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                 'rb'))
    else:
        xbs, ybs = initialise_graph(pointsBS)
        G, colorlist, nodesize = make_graph(xbs, ybs, xpop, ypop, pointsBS, pointsPop)

        for node in range(pointsPop):
            bss = find_bs(xpop[node], ypop[node], xbs, ybs, k)
            for bs in bss:
                G.add_edge(node + pointsBS, bs)

        pickle.dump(G, open(
            str('Simulations/graph' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
            'wb'), protocol=4)
        pickle.dump(pointsBS,
                    open(str('Simulations/pointsBS' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                        k) + '.p'), 'wb'), protocol=4)
        pickle.dump(pointsPop, open(
            str('Simulations/pointsPop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                k) + '.p'), 'wb'), protocol=4)
        pickle.dump(xbs, open(
            str('Simulations/xbs' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
            'wb'), protocol=4)
        pickle.dump(ybs, open(
            str('Simulations/ybs' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
            'wb'), protocol=4)
        pickle.dump(xpop, open(
            str('Simulations/xpop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
            'wb'), protocol=4)
        pickle.dump(ypop, open(
            str('Simulations/ypop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
            'wb'), protocol=4)

    channel, logSNR, W = find_all(pointsPop, pointsBS, G, G, labdaBS, xbs, ybs, xpop, ypop, total_bandwidth, k)
    y, x = np.histogram(np.array(channel), density=True, bins='auto')
    y = np.cumsum(y)
    mid_x = [(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)]

    # plt.hist(channel, color=colors[k - 1], label=str('k = ' + str(k)), bins = 100,  histtype=u'step', density=True, cumulative=True)
    # plt.plot(y, mid_x)
    print('cv = ', np.std(channel) / np.mean(channel))
    print(sum(channel)/len(channel))
    # ax11.plot(lbs, channel_sum, color=colors[k - 1])
    channels.append(channel)
plot_ecdf_lists.plot_ecdf_lists(channels, 'cdfcksum')
# plt.xlim((0, 0.5e10))
# plt.legend()
# plt.yscale('log')
# plt.xlabel('$C_{sum}^k$')
# plt.show()