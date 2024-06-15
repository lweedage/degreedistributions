import networkx as nx
import scipy.stats
import matplotlib.pyplot as plt
import math
import numpy as np

pointsPop = 10000

xMin, xMax = 0, 1000     # in meters
yMin, yMax = 0, 1000     # in meters

xDelta = xMax - xMin
yDelta = yMax - yMin

radius = 20

def initialise_graph(points):
    xx = xDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + xMin  # x coordinates of Poisson points
    yy = yDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + yMin  # y coordinates of Poisson points

    x = [xx[i][0] for i in range(points)]
    y = [yy[i][0] for i in range(points)]

    return x, y


def initialise_graph_users(radius):
    x, y = list(), list()
    for i in range(int(xDelta/radius + 1)):
        for j in range(int(yDelta/radius)+1):
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
    for i in range(int(xDelta/radius) + 1):
        for j in range(int(yDelta/dy) + 1):
            xbs.append(i*radius)
            ybs.append(j*radius %yDelta)
    return xbs, ybs

def initialise_graph_bs_triangular(radius):
    xbs, ybs = list(), list()
    dy = math.sqrt(3/4) * radius
    for i in range(-2, int(xDelta/radius) + 3):
        for j in range(-2, int(yDelta/dy) + 3):
            xbs.append(i*radius + 0.5*(j%2) * radius)
            ybs.append(j*dy)
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
        nodesize.append(0)
    return G, colorlist, nodesize

def draw_graph(G, ax, colorlist, nodesize):
    nx.draw_networkx_nodes(G, nx.get_node_attributes(G, 'pos'), nodelist = G.nodes(), node_size= nodesize, node_color=colorlist, ax=ax)
    nx.draw_networkx_edges(G, nx.get_node_attributes(G, 'pos'), edge_color='gray')
    ax.set_xlim([xMin, xMax]), ax.set_ylim([yMin, yMax])


def find_distance(x, y, xbs, ybs):
    x = np.minimum((x-np.array(xbs))%xDelta, (np.array(xbs)-x)%xDelta)
    y = np.minimum((y-np.array(ybs))%yDelta, (np.array(ybs)-y)%yDelta)
    return np.sqrt(x ** 2 + y ** 2)

def find_bs(xpop, ypop, xbs, ybs, max_connections):
    indices = find_distance(xpop, ypop, xbs, ybs).argsort()
    return indices[:max_connections]

k = 10
number_of_iterations = 1
test_bs = 28

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 10
for iterations in range(number_of_iterations):
    print(iterations)
    xpop, ypop = initialise_graph_users(10)
    # xbs, ybs = initialise_graph_bs_square(radius)
    xbs, ybs = initialise_graph_bs_triangular(200)
    pointsBS = len(xbs)
    pointsPop = len(xpop)
    G, colorlist, nodesize = make_graph(xbs, ybs, xpop, ypop, pointsBS, pointsPop)
    for node in range(pointsPop):
        base_stations = find_bs(xpop[node], ypop[node], xbs, ybs, k)
        for i in range(k):
            if base_stations[i] == test_bs:
                colorlist[pointsBS + node] = colors[i]
                nodesize[pointsBS + node] = 1
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    draw_graph(G, ax, colorlist, nodesize)
    # plt.title(str("Users connect to " + str(k) + " base stations"))
    plt.show()


