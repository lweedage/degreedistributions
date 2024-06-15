import csv
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import geopy.distance
import pickle
import progressbar



def make_graph(xbs, ybs):
    G = nx.Graph()
    colorlist, nodesize = [], []
    for node in range(len(xbs)):
        G.add_node(node, pos=(xbs[node], ybs[node]))
        colorlist.append('#8B008B')
        nodesize.append(20)
    return colorlist, nodesize, G

def draw_graph(G, ax, colorlist, nodesize):
    nx.draw_networkx_nodes(G, nx.get_node_attributes(G, 'pos'), nodelist = G.nodes(), ax=ax, node_size=nodesize, node_color=colorlist, node_shape='+')
    nx.draw_networkx_edges(G, nx.get_node_attributes(G, 'pos'), edge_color='gray')

def find_distance(x, y, xbs, ybs):
    x = np.minimum((x-np.array(xbs))%xDelta, (np.array(xbs)-x)%xDelta)
    y = np.minimum((y-np.array(ybs))%yDelta, (np.array(ybs)-y)%yDelta)
    return np.sqrt(x ** 2 + y ** 2)

def find_bs(xpop, ypop, xbs, ybs, max_connections):
    indices = find_distance(xpop, ypop, xbs, ybs).argsort()
    return indices[:max_connections]

First = True

lon = []
lat = []

Enschede_city = True

if Enschede_city:
    coords_lon = [6.833857108678182, 6.929472494113721]  # coordinates of larger Enschede
    coords_lat = [52.189007057488325, 52.250844234818864]
xDelta = coords_lon[1] - coords_lon[0]
yDelta = coords_lat[1] - coords_lat[0]

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

colorlist, nodesize, G = make_graph(xbs, ybs)
fig, ax = plt.subplots()
draw_graph(G, ax, colorlist, nodesize)
epsilon = 0.001
plt.xlim(min(xbs) - epsilon, max(xbs) + epsilon)
plt.ylim(min(ybs) - epsilon, max(ybs) + epsilon)
plt.show()

pointsBS = len(ybs)

print('There are', pointsBS, 'base stations')
print('The area is', geopy.distance.distance((coords_lon[0], coords_lat[1]), (coords_lon[1], coords_lat[1])).km, 'by', geopy.distance.distance((coords_lon[0], coords_lat[0]), (coords_lon[0], coords_lat[1])).km, 'km')

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
number_of_iterations = int(input('Number of iterations?'))
resolution = float(input('Resolution?'))

area = np.zeros((max_k, number_of_iterations * pointsBS))

bar = progressbar.ProgressBar(max_value=1.0)
for iteration in range(number_of_iterations):
    find_k_order_area_bs(coords_lon[0], coords_lat[0], coords_lon[1], coords_lat[1], 1.0 * iteration / number_of_iterations, 1.0 * (iteration+1) / number_of_iterations)

pickle.dump(area, open(str('Enschede_area_points=' + str(number_of_iterations * pointsBS) + 'resolution=' + str(resolution) + '.p'),'wb'), protocol=4)