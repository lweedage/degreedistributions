import networkx as nx
import scipy.stats
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn
import matplotlib.patches as mpatches

pointsPop = 10000

xMin, xMax = 0, 1000     # in meters
yMin, yMax = 0, 1000     # in meters

xDelta = xMax - xMin
yDelta = yMax - yMin

radius = 20

def initialise_graph(points):
    xx = xDelta * scipy.stats.uniform.rvs(-1, 2, ((points, 1))) + xMin  # x coordinates of Poisson points
    yy = yDelta * scipy.stats.uniform.rvs(-1, 2, ((points, 1))) + yMin  # y coordinates of Poisson points

    x = [xx[i][0] for i in range(points)]
    y = [yy[i][0] for i in range(points)]

    return x, y


def initialise_graph_users(radius):
    x, y = list(), list()
    for i in range(int(xDelta/radius) + 1):
        for j in range(int(yDelta/radius) + 1):
            x.append(i * radius)
            y.append(j * radius % yDelta)
    return x, y

def initialise_graph_bs_hexagonal(radius):
    xbs, ybs = list(), list()
    dy = math.sqrt(3/4) * radius
    for i in range(-10, int(xDelta/radius) + 11):
        for j in range(-10, int(yDelta/dy) + 11):
            if i % 3 == j % 2:
                continue
            xbs.append(i*radius + 0.5*(j%2) * radius)
            ybs.append(j*dy)
    return xbs, ybs

def initialise_graph_bs_square(radius):
    xbs, ybs = list(), list()
    dy = math.sqrt(3/4) * radius
    for i in range(-10, int(xDelta/radius) + 11):
        for j in range(-10, int(yDelta/dy) + 11):
            xbs.append(i*radius)
            ybs.append(j*radius %yDelta)
    return xbs, ybs

def initialise_graph_bs_triangular(radius):
    xbs, ybs = list(), list()
    dy = math.sqrt(3/4) * radius
    for i in range(-10, int(xDelta/radius) + 11):
        for j in range(-10, int(yDelta/dy) + 11):
            xbs.append(i*radius + 0.5*(j%2) * radius)
            ybs.append(j*dy)
    return xbs, ybs

def make_graph(xbs, ybs, xpop, ypop):
    G = nx.Graph()
    colorlist = list()
    nodesize = list()
    nodeshape = list()
    for node in range(len(xbs)):
        G.add_node(node, pos=(xbs[node], ybs[node]))
        colorlist.append('#8B008B')
        nodesize.append(20)
    for node in range(len(xpop)):
        G.add_node(node + len(xbs), pos=(xpop[node], ypop[node]))
        colorlist.append('g')
        nodesize.append(0)
    return G, colorlist, nodesize

def draw_graph(G, ax, colorlist, nodesize):
    nx.draw_networkx_nodes(G, nx.get_node_attributes(G, 'pos'), nodelist = G.nodes(), node_size= nodesize, node_color=colorlist, ax=ax)
    nx.draw_networkx_edges(G, nx.get_node_attributes(G, 'pos'), edge_color='gray')
    ax.set_xlim([xMin, xMax]), ax.set_ylim([yMin, yMax])


def find_distance(x, y, xbs, ybs):
    x = np.minimum((x-np.array(xbs)), (np.array(xbs)-x))
    y = np.minimum((y-np.array(ybs)), (np.array(ybs)-y))
    return np.sqrt(x ** 2 + y ** 2)

def find_bs(xpop, ypop, xbs, ybs, max_connections):
    indices = find_distance(xpop, ypop, xbs, ybs).argsort()
    return indices[:max_connections]

np.random.seed(1) #seed 1 testbs 43
k = 10
number_of_iterations = 1
test_bs = 325
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 10
colors = seaborn.color_palette("hls", 10)
colors.reverse()

for iterations in range(number_of_iterations):
    xpop, ypop = initialise_graph_users(1)
    xbs, ybs = initialise_graph_bs_triangular(200)
    print(xbs)
    print(ybs)
    pointsBS = len(xbs)
    pointsPop = len(xpop)

    print(xbs[test_bs], ybs[test_bs])
    G, colorlist, nodesize = make_graph(xbs, ybs, xpop, ypop)
    for node in range(pointsPop):
        base_stations = find_bs(xpop[node], ypop[node], xbs, ybs, k)
        for i in range(k):
            if base_stations[i] == test_bs:
                colorlist[pointsBS + node] = colors[i]
                nodesize[pointsBS + node] = 1
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    draw_graph(G, ax, colorlist, nodesize)
    plt.plot(xbs[test_bs], ybs[test_bs], 'o', color = '#8B008B')


    patch1 = mpatches.Patch(color=colors[0], label='$k=1$')
    patch2 = mpatches.Patch(color=colors[1], label='$k=2$')
    patch3 = mpatches.Patch(color=colors[2], label='$k=3$')
    patch4 = mpatches.Patch(color=colors[3], label='$k=4$')
    patch5 = mpatches.Patch(color=colors[4], label='$k=5$')
    patch6 = mpatches.Patch(color=colors[5], label='$k=6$')
    patch7 = mpatches.Patch(color=colors[6], label='$k=7$')
    patch8 = mpatches.Patch(color=colors[7], label='$k=8$')
    patch9 = mpatches.Patch(color=colors[8], label='$k=9$')
    patch10 = mpatches.Patch(color=colors[9], label='$k=10$')

    plt.legend(handles=[patch1, patch2, patch3, patch4, patch5, patch6, patch7, patch8, patch9, patch10], bbox_to_anchor=(1.01, 1), loc='upper left')

    # plt.title(str("Users connect to " + str(k) + " base stations"))
    plt.show()


