import networkx as nx
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.patches
import math
import numpy as np
import scipy
import mpmath
import time
import pickle
import os
import progressbar

def initialise_graph(points):
    xx = xDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + xMin  # x coordinates of Poisson points
    yy = yDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + yMin  # y coordinates of Poisson points

    x = [xx[i][0] for i in range(points)]
    y = [yy[i][0] for i in range(points)]

    return x, y

def make_graph(xbs, ybs, pointsBS):
    G = nx.Graph()
    colorlist = list()
    nodesize = list()
    for node in range(pointsBS):
        G.add_node(node, pos=(xbs[node], ybs[node]))
        colorlist.append('k')
        nodesize.append(50)
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

def find_distance_to_bs(x, y, xbs, ybs, max_connections):
    indices = find_distance(x, y, xbs, ybs)
    indices.sort()
    return indices[:max_connections]

def find_k_order_area_bs(x1, y1, x2, y2, min_progress, max_progress):
    mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
    closest_points = find_bs(mid_x, mid_y, xbs, ybs, max_k + 1)
    if x2 - x1 <= resolution and y2 - y1 <= resolution:
        for k in range(max_k):
            area[k-1, closest_points[k-1] + pointsBS * (iteration)] += (x2-x1) * (y2-y1)
        bar.update(max_progress)
        return
    nearest_distances = find_distance_to_bs(mid_x, mid_y, xbs, ybs, max_k + 1)
    all_in_headroom = True
    for k in range(max_k):
        head_room = min(abs(nearest_distances[k + 1] - nearest_distances[k]),
                        abs(nearest_distances[k] - nearest_distances[k - 1])) / 2
        if (x2-mid_x)**2 + (y2-mid_y)**2 >= head_room**2:
            all_in_headroom = False
    if all_in_headroom:
        for k in range(max_k):
            area[k-1, closest_points[k-1] + pointsBS * (iteration)] += (x2-x1) * (y2-y1)
    else:
        mid_progress = min_progress+(max_progress-min_progress)/2
        if x2 - x1 >= y2 - y1:
            find_k_order_area_bs(x1, y1, mid_x, y2, min_progress, mid_progress)
            find_k_order_area_bs(mid_x, y1, x2, y2, mid_progress, max_progress)
        else:
            find_k_order_area_bs(x1, y1, x2, mid_y, min_progress, mid_progress)
            find_k_order_area_bs(x1, mid_y, x2, y2, mid_progress, max_progress)
    bar.update(max_progress)

max_k = 5

number_of_iterations = 1000 #int(input('Number of iterations?'))
pointsBS = 100 #int(input('Number of points?'))
resolution = 0.1 #float(input('Resolution?'))

xMin, xMax = 0, math.sqrt(pointsBS)     # in meters
yMin, yMax = 0, math.sqrt(pointsBS)     # in meters

xDelta = xMax - xMin
yDelta = yMax - yMin


for blub in range(100000):
    bar = progressbar.ProgressBar(max_value=1.0)
    area = np.zeros((max_k, number_of_iterations * pointsBS))
    for iteration in range(number_of_iterations):
        xbs, ybs = initialise_graph(pointsBS)
        find_k_order_area_bs(xMin, yMin, xMax, yMax, 1.0 * iteration / number_of_iterations, 1.0 * (iteration+1) / number_of_iterations)

    pickle.dump(area, open(str('area_points_till_far=' + str(number_of_iterations * pointsBS) + 'resolution=' + str(resolution) + 'try' + str(blub) + '.p'),'wb'), protocol=4)
    print('Finished with iteration', blub)