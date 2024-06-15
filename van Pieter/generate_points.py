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
import csv

matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['lines.markersize'] = 7
matplotlib.rcParams['figure.autolayout'] = True

pi = math.pi

a = [3.5, 7.2, 11.1, 15.2, 21.2]


# --------- Functions --------------------
def initialise_graph(points):
    xx = xDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + xMin  # x coordinates of Poisson points
    yy = yDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + yMin  # y coordinates of Poisson points

    x = [xx[i][0] for i in range(points)]
    y = [yy[i][0] for i in range(points)]

    return x, y

def initialise_graph_triangular(radius):
    xbs, ybs = list(), list()
    dy = math.sqrt(3/4) * radius
    for i in range(0, int(xDelta/radius) + 1):
        for j in range(0, int(yDelta/dy) + 1):
            xbs.append(i*radius + 0.5*(j%2) * radius)
            ybs.append(j*dy)
    return xbs, ybs

def initialise_graph_real():
    First = True
    lon = []
    lat = []

    coords_lon = [6.833857108678182, 6.929472494113721]  # coordinates of larger Enschede
    coords_lat = [52.189007057488325, 52.250844234818864]
    xDelta = coords_lon[1] - coords_lon[0]
    yDelta = coords_lat[1] - coords_lat[0]

    print(xDelta, yDelta)

    with open('Data_Suzan/204.csv') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for row in csvReader:
            if not First:
                if row[0] == 'LTE':
                    long = float(row[6])
                    lati = float(row[7])
                    lon.append(long)
                    lat.append(lati)
            else:
                First = False
    pointsBS = len(lon)
    xbs, ybs = [], []

    for i in range(pointsBS):
        long = lon[i]
        lati = lat[i]
        if coords_lon[0] <= long <= coords_lon[1] and coords_lat[0] <= lati <= coords_lat[1]:
            xbs.append(long)
            ybs.append(lati)
    # xbs = np.array(xbs)/max(xbs) * 25
    xbs = np.interp(xbs, [min(xbs), max(xbs)], [0, 25])
    ybs = np.interp(ybs, [min(ybs), max(ybs)], [0, 25])
    return xbs, ybs

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


def draw_graph(G, colorlist, nodesize):
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, nx.get_node_attributes(G, 'pos'), nodelist=G.nodes(), node_size=nodesize,
                           node_color=colorlist, ax=ax)
    nx.draw_networkx_edges(G, nx.get_node_attributes(G, 'pos'), edge_color='gray', alpha=0.3)
    ax.set_xlim([xMin, xMax]), ax.set_ylim([yMin, yMax])
    plt.show()


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


# ----------------------- OTHER FUNCTIONS ---------------------------------------
def number_of_disconnected_users(G, pointsBS):
    discon_list = [1 for u, v in G.degree if u >= pointsBS and v == 0]
    return len(discon_list)


def find_bs(xpop, ypop, xbs, ybs, max_connections):
    indices = find_distance(xpop, ypop, xbs, ybs).argsort()
    return indices[:max_connections]


def average(x):
    if len(x) > 0:
        return sum(x) / len(x)
    else:
        return 0


def average_nonempty(x):
    X = [i for i in x if i != 0]
    if len(X) > 0:
        return sum(X) / len(X)
    else:
        return 0


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


# -------------------------- THE PROGRAM --------------------------------------

xMin, xMax = 0, 25  # in meters
yMin, yMax = 0, 25  # in meters

xDelta = xMax - xMin
yDelta = yMax - yMin

labdaBS = 0.1  # = how many BS per square meter
labdaU = 1  # = how many users per square meter

total_bandwidth = 20 * 10 ** 6

alpha = 2
c = 10 ** (3.5)

pointsBS = labdaBS * (yDelta * xDelta)
pointsPop = labdaU * (yDelta * xDelta)

pointsBS = int(pointsBS)
pointsPop = int(pointsPop)

labdaBS = pointsBS / (yDelta * xDelta)
labdaU = pointsPop / (yDelta * xDelta)

radius = 4

# xbs, ybs = initialise_graph(pointsBS)
# xbs, ybs = initialise_graph_triangular(radius)
xbs, ybs = initialise_graph_real()
xpop, ypop = initialise_graph(pointsPop)

delta = 25

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
          '#17becf']
colors = seaborn.color_palette('rocket')
colors.reverse()
markers = ['o', 's', 'p', 'd', '*']

pointsBS = len(xbs)
G, colorlist, nodesize = make_graph(xbs, ybs, xpop, ypop, pointsBS, pointsPop)

max_iterations = 100

for node in range(pointsPop):
    bss = find_bs(xpop[node], ypop[node], xbs, ybs, pointsBS)
    for bs in bss:
        G.add_edge(node + pointsBS, bs)

draw_graph(G, colorlist, nodesize)
# xbs.insert(0, 'xbs')
# ybs.insert(0, 'ybs')
# xpop.insert(0, 'xu')
# ypop.insert(0, 'yu')

print(pointsBS)
print(pointsPop)
np.savetxt('xbs.txt', xbs, fmt="%s")
np.savetxt('ybs.txt', ybs, fmt="%s")
np.savetxt('xu.txt', xpop, fmt="%s")
np.savetxt('yu.txt', ypop, fmt="%s")



