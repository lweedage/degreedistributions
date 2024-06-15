import networkx as nx
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.patches
import math
import numpy as np
from scipy.special import gamma, loggamma
import scipy
import mpmath
import time
import pickle
import os
import progressbar
import csv
from scipy.stats import chisquare
import seaborn
import geopy.distance

ak = [3.527, 7.187, 11.062, 15.212, 21.166]

def finda(k):
    if k <= 5:
        a = ak[k-1]
    else:
        a = 5.8294 * k - 5.7847
    return a


matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['lines.markersize'] = 10
matplotlib.rcParams['figure.autolayout'] = True

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 10
colors = seaborn.color_palette('rocket') * 10
colors.reverse()

def initialise_graph(points):
    xx = xDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + xMin  # x coordinates of Poisson points
    yy = yDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + yMin  # y coordinates of Poisson points

    x = [xx[i][0] for i in range(points)]
    y = [yy[i][0] for i in range(points)]

    return x, y

def initialise_graph_real(points):
    xmin, xmax = coords_lon[0], coords_lon[1]
    ymin, ymax = coords_lat[0], coords_lat[1]

    xdelta = xmax - xmin
    ydelta = ymax - ymin

    xx = xdelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + xmin  # x coordinates of Poisson points
    yy = ydelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + ymin  # y coordinates of Poisson points

    x = [xx[i][0] for i in range(points)]
    y = [yy[i][0] for i in range(points)]

    return x, y

def initialise_graph_bs_hexagonal(radius):
    xbs, ybs = list(), list()
    dy = math.sqrt(3/4) * radius
    for i in range(0, int(xDelta/radius)):
        for j in range(0, int(yDelta/dy)):
            if i % 3 == j % 2:
                continue
            xbs.append(i*radius + 0.5*(j%2) * radius)
            ybs.append(j*dy)
    return xbs, ybs

def make_graph(xbs, ybs, xpop, ypop):
    G = nx.Graph()
    colorlist = list()
    nodesize = list()
    for node in range(len(xbs)):
        G.add_node(node, pos=(xbs[node], ybs[node]))
        colorlist.append('k')
        nodesize.append(50)
    for node in range(len(xpop)):
        G.add_node(node + len(xbs), pos=(xpop[node], ypop[node]))
        colorlist.append('g')
        nodesize.append(1)
    return G, colorlist, nodesize

def draw_graph(G, ax, colorlist, nodesize):
    nx.draw_networkx_nodes(G, nx.get_node_attributes(G, 'pos'), nodelist = G.nodes(), node_size= nodesize, node_color=colorlist, ax=ax)
    nx.draw_networkx_edges(G, nx.get_node_attributes(G, 'pos'), edge_color='gray')
    ax.set_xlim([xMin, xMax]), ax.set_ylim([yMin, yMax])

def find_distance(x, y, xbs, ybs):
    x = np.minimum((x-np.array(xbs))%xDelta, (np.array(xbs)-x)%xDelta)
    y = np.minimum((y-np.array(ybs))%yDelta, (np.array(ybs)-y)%yDelta)
    return np.sqrt(x ** 2 + y ** 2)

def find_distance_shadowing(x, y, xbs, ybs, sigma):
    p = np.random.lognormal(1, sigma, len(xbs))
    x = np.minimum((x-np.array(xbs))%xDelta, (np.array(xbs)-x)%xDelta)
    y = np.minimum((y-np.array(ybs))%yDelta, (np.array(ybs)-y)%yDelta)
    return np.sqrt(x ** 2 + y ** 2) / p

def find_distance_shadowing_m(x, y, xbs, ybs, sigma):
    dlat = math.radians(np.minimum((x-np.array(xbs))%xDelta, (np.array(xbs)-x)%xDelta))
    dlon = math.radians(np.minimum((y-np.array(ybs))%yDelta, (np.array(ybs)-y)%yDelta))
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d

def find_bs(xpop, ypop, xbs, ybs, max_connections):
    indices = find_distance(xpop, ypop, xbs, ybs).argsort()
    return indices[:max_connections]

def find_bs_approx(xpop, ypop, xbs, ybs, max_connections, sigma):
    indices = find_distance_shadowing(xpop, ypop, xbs, ybs, sigma).argsort()
    return indices[:max_connections]

def pdf_poisson(k, labda, n):
    blub = n * math.log(k * labda) - loggamma(n+1)
    return math.exp(blub - k* labda)

def pdf_poisson_1d(k, lbs, lu, n):
    blub = loggamma(2*k+n) - loggamma(n+1) - loggamma(2*k)
    blub2 = 2*k*math.log(2*lbs) + n * math.log(lu) - (2*k+n)*math.log(2*lbs + lu)
    return math.exp(blub + blub2)

def degree_dist(k, labda, n):
    ak = finda(k)
    t = ak * math.log(ak/(k * labda + ak))
    if n > 1:
        blub1 = loggamma(n+ak) - (loggamma(n+1) + loggamma(ak))
    else:
        blub1 = math.log(gamma(n+ak)/(math.gamma(n+1)*gamma(ak)))
    return  math.exp(t + blub1) * (k*labda/(k * labda + ak))**n

def from_memory(filename):
    if os.path.exists(filename):
        file = pickle.load(open(filename, 'rb'))
        return file

def find_real_data_points():
    First = True

    lon = []
    lat = []
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
    return xbs, ybs

radius = 8.7

coords_lon = [5.932781187928603, 6.636705933262508]  # coordinates of Zwolle/Enschede
coords_lat = [52.030995432999276, 52.509922175518874]

coords_lon = [6.833857108678182, 6.929472494113721]  # coordinates of larger Enschede
coords_lat = [52.189007057488325, 52.250844234818864]


xMin, xMax = 0, 1500     # in meters
yMin, yMax = 0, 1500     # in meters

xDelta = xMax - xMin
yDelta = yMax - yMin

A = xDelta * yDelta

labdaU = 0.1
labdaBS = 0.01
pointsPop = int((xDelta * yDelta) * labdaU)
pointsBS = int((xDelta * yDelta) * labdaBS)

Hexagonal = False
Make_Graph = False
Real_Data = True

sigma = 0.1
k = 50
blub = 0
for iteration in range(1):
    # for k in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]:
    for k in [1, 5, 25, 50]:
        labdaU = 0.01
        # labdaBS = 0.010126666666666667 # hexagonal
        # labdaBS = 0.01 # poisson
        labdaBS = 0.0002662222222222222 # real
        pointsPop = int((xDelta * yDelta) * labdaU)
        pointsBS = int((xDelta * yDelta) * labdaBS)
        # print(pointsPop)
        # # print('k =', k)
        # if Make_Graph:
        #     if Hexagonal:
        #         filename = str('Simulations/graph_hexagonal_lu=' + str(labdaU) + 'radius=' + str(radius) + 'area=' + str(xDelta) + 'by' + str(yDelta) + 'k=' + str(k) + 'try' + str(iteration) +  '.p')
        #     elif Real_Data:
        #         filename = str('Simulations/graph_real_lu=' + str(labdaU) + 'BSs=' + str(pointsBS) + 'area=' + str(xDelta) + 'by' + str(yDelta) + 'k=' + str(k) + 'try' + str(iteration) + 'sigma' + str(sigma) + '.p')
        #     else:
        #         filename = str('Simulations/graph_poisson_lu=' + str(labdaU) + 'lbs=' + str(labdaBS) + 'area=' + str(xDelta) + 'by' + str(yDelta) + 'k=' + str(k) + 'try' + str(iteration) +  '.p')
        #         # filename = str('Simulations/graph_poisson_lu=' + str(labdaU) + 'lbs=' + str(labdaBS) + 'area=' + str(xDelta) + 'by' + str(yDelta) + 'k=' + str(k) + 'points=' + str(877500) +  '.p')
        #
        #     print('Making the graph...')
        #     if Hexagonal:
        #         xbs, ybs = initialise_graph_bs_hexagonal(radius)
        #         xpop, ypop = initialise_graph(int(pointsPop))
        #
        #     elif Real_Data:
        #         xbs, ybs = find_real_data_points()
        #         xpop, ypop = initialise_graph_real(int(pointsPop))
        #     else:
        #         xbs, ybs = initialise_graph(pointsBS)
        #         xpop, ypop = initialise_graph(pointsPop)
        #     print('There are', len(xbs), 'base stations')
        #     pointsBS = len(xbs)
        #     pointsPop = len(xpop)
        #     labdaBS = pointsBS/(xDelta * yDelta)
        #     labdaU = pointsPop/(xDelta * yDelta)
        #     print('labdabs = ', labdaBS)
        #     G, colorlist, nodesize = make_graph(xbs, ybs, xpop, ypop)
        #     print('Adding edges...')
        #     for node in range(pointsPop):
        #         if Real_Data:
        #             base_stations = find_bs_approx(xpop[node], ypop[node], xbs, ybs, k, sigma)
        #         else:
        #             base_stations = find_bs(xpop[node], ypop[node], xbs, ybs, k)
        #         for bs in base_stations:
        #             G.add_edge(bs, node + pointsBS)
        #     with open(filename, 'wb') as f:
        #         pickle.dump(G, f, protocol=4)
        #     print('Finding degrees...')
        #     degrees = [deg for node, deg in G.degree() if node < pointsBS]
        #     degrees = np.array(degrees)
        #
        #     if Hexagonal:
        #         file_name = str('degreelist_hexagonal_lu=' + str(labdaU) + 'lbs=' + str(labdaBS) + 'area=' + str(xDelta) + 'by' + str(yDelta) + 'k=' + str(k) + 'try=' + str(iteration) + '.p')
        #         area = xDelta * yDelta
        #     elif Real_Data:
        #         file_name = str('degreelist_real_lu=' + str(labdaU) + 'lbs=' + str(labdaBS) + 'area=' + str(xDelta) + 'by' + str(yDelta) + 'k=' + str(k) + 'try=' + str(iteration) +'sigma' + str(sigma) +  '.p')
        #         area = (coords_lon[1] - coords_lon[0]) * (coords_lat[1] - coords_lat[0])
        #     else:
        #         file_name = str('degreelist_poisson_lu=' + str(labdaU) + 'lbs=' + str(labdaBS) + 'area=' + str(xDelta) + 'by' + str(yDelta) + 'k=' + str(k) + 'try=' + str(iteration) + '.p')
        #         area = xDelta * yDelta
        #     with open(file_name, 'wb') as f:
        #         pickle.dump(degrees, f, protocol=4)
        #
        # else:
        #     if Hexagonal:
        #         file_name = str('degreelist_hexagonal_lu=' + str(labdaU) + 'lbs=' + str(labdaBS) + 'area=' + str(xDelta) + 'by' + str(yDelta) + 'k=' + str(k) + 'try=' + str(0) + '.p')
        #         area = xDelta * yDelta
        #     elif Real_Data:
        #         file_name = str('degreelist_real_lu=' + str(labdaU) + 'lbs=' + str(labdaBS) + 'area=' + str(xDelta) + 'by' + str(yDelta) + 'k=' + str(k) + 'try=' + str(0) + 'sigma' + str(sigma) + '.p')
        #         area = (coords_lon[1] - coords_lon[0]) * (coords_lat[1] - coords_lat[0])
        #     else:
        #         file_name = str('degreelist_poisson_lu=' + str(labdaU) + 'lbs=' + str(labdaBS) + 'area=' + str(xDelta) + 'by' + str(yDelta) + 'k=' + str(k) + 'try=' + str(0) + '.p')
        #         area = xDelta * yDelta
        #     degrees = from_memory(file_name)
        # exp_degree = sum(degrees)/len(degrees)
        # print('Expected degree = ', exp_degree)
        # pointsBS = len(degrees)
        # pointsPop = sum(degrees)/k
        # labdaBS = pointsBS/area
        # labdaU = pointsPop/area
        # print('There are', len(degrees), 'base stations')
        #
        # X = np.arange(min(degrees), max(degrees), 1)
        # y, x = np.histogram(np.array(degrees), density = True, bins= X)
        # mid_x = [(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)]
        # # plt.scatter(mid_x, y, color = colors[blub])
        # # fig, ax = plt.subplots()
        # if k <= 5:
        #     plt.hist(degrees, density = True, alpha = 0.3, bins = 25, color = colors[blub], label = str('$k =$ ' + str(k)))
        # else:
        #     plt.hist(degrees, density = True, alpha = 0.3, bins = 25, color = colors[blub], label = str('$k =$ ' + str(k)))
        # # if k <= 100:
        #     # plt.plot(X, [degree_dist(k, labdaU/labdaBS, x) for x in X], ':', color = colors[blub], label='Poisson-Gamma')
        # plt.plot(X, [pdf_poisson(k, labdaU/labdaBS, x) for x in X], '+', color = colors[blub], label = 'Poisson')
        # # plt.plot(X, [pdf_poisson_1d(k, labdaBS, labdaU, x) for x in X], color = colors[blub], label = 'Poisson-Erlang')
        #
        # blub += 1
        # plt.legend()
        # # plt.xlim((-1, 200))
        # blab = [pdf_poisson_1d(k, labdaBS, labdaU, x) for x in X]
        # # plt.ylim((0, 1.3* max(blab)))
        # plt.xlabel('Degree')
        # plt.ylabel('P$(D_B = x)$')
        #
        # plt.savefig(str(file_name[:-2]) + '.png')
        # plt.show()
        # print(np.std(degrees) / np.mean(degrees), ',')
        # # chi, p_value = chisquare(y, [pdf_poisson_1d(k, labdaBS, labdaU, x) for x in mid_x])
        # # print(chi, p_value)

        if Hexagonal:
            xbs, ybs = initialise_graph_bs_hexagonal(radius)
            xpop, ypop = initialise_graph(int(pointsPop))

        elif Real_Data:
            xbs, ybs = find_real_data_points()
            xpop, ypop = initialise_graph_real(int(pointsPop))
        else:
            xbs, ybs = initialise_graph(pointsBS)
            xpop, ypop = initialise_graph(pointsPop)
        print(pointsPop)
        distance01 = []
        distance1 = []
        distance = []
        for u in range(int(pointsPop)):
            distance01.append(sorted(find_distance_shadowing_m(xpop[u], ypop[u], xbs, ybs, 0.1))[0])
            distance1.append(sorted(find_distance_shadowing_m(xpop[u], ypop[u], xbs, ybs, 1))[0])
            distance.append(sorted(find_distance(xpop[u], ypop[u], xbs, ybs))[0])
        plt.hist(distance, label = 'No shadowing', color = colors[0], alpha = 0.3, density = True)
        plt.hist(distance01, label = '$\sigma = 0.1$', color = colors[1], alpha = 0.3, density = True)
        plt.hist(distance1, label = '$\sigma = 1$', color = colors[2], alpha = 0.3, density = True)

        plt.legend()
        plt.show()
markers = ['o', 's', 'p', 'd', '*']


fig, ax = plt.subplots()
labda = labdaU/labdaBS

kaas = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
kaasbeter = np.arange(1, 51, 1)
hexagonal = [0.32027232763985686,0.14391818111794039, 0.10432556850055005, 0.08428489273123654, 0.07199398768107114, 0.06877198998472604, 0.06111339174195588, 0.054507902573507294, 0.05323739978554481, 0.049504944516272444, 0.04449998870877926]#, 0.04426368501801844, 0.04278020770010588, 0.04169809534244192, 0.03802699734934915, 0.03796037832834752]
poisson = [0.6193631137440029, 0.27658893687203034, 0.1906576221631039, 0.1626767938559746, 0.139910233125863, 0.12157024782221823, 0.11584518830888482, 0.10359300793256757, 0.09348508793741742, 0.09567520960981497, 0.08851057137113309]#, 0.08863435605082048, 0.08448703750382563, 0.0719171310673604, 0.07442125485775669, 0.06719027442850473]
realshadow = [0.4241763708181938, 0.2516635812653643, 0.17565551961709208, 0.13314291661257932, 0.10972262808498774, 0.09721643614593442, 0.09121631920775305, 0.08716618884872558, 0.08472433546522919, 0.08784797769896178, 0.08901381300405566]#, 0.09058078965163466, 0.09296463889868978, 0.09613841867522463, 0.09846784725001817, 0.10067050023926535]
realnoshadow = [1.0701282123900324, 0.7551381692704887, 0.6544866242286561, 0.5860704649087837, 0.539834890258169, 0.5028324761946011, 0.4680382139255717, 0.44012976136347964, 0.41497428098490424, 0.39402372750235737, 0.3699560117901827]#, 0.34933605836789183, 0.3278930824068763, 0.31405780388356347, 0.30233764824340437, 0.29141685427247105]



plt.scatter(kaas, hexagonal, color = colors[0], marker = markers[0], label = 'hexagonal grid')
plt.scatter(kaas, poisson, color = colors[1], marker = markers[1],  label = 'Poisson point process')
plt.scatter(kaas, realshadow, color = colors[3], marker = markers[3], label = 'real data - strong shadowing')
plt.scatter(kaas, realnoshadow, color = colors[2], marker = markers[2], label = 'real data - little shadowing')
plt.plot(kaasbeter, [1/math.sqrt(labdaU/labdaBS * k) for k in kaasbeter], color = colors[0], label = 'Poisson')
plt.plot(kaasbeter, [math.sqrt((1 + 0.5*labda)/(k*labda)) for k in kaasbeter], ':', color = colors[4], label = 'compound Poisson-Erlang')
markers_on = [1, 2, 3, 4, 5]
plt.plot(kaasbeter, [math.sqrt(1/(k*labda) + 1/finda(k)) for k in kaasbeter], '.-', color = colors[5], label = 'compound Poisson-Gamma', markevery=markers_on)


plt.xlabel(str('$k$'))
plt.ylabel('$c_v$')
plt.legend(prop={'size': 12})
plt.xticks([1, 10, 20, 30, 40, 50])
plt.savefig('blub.png')
plt.show()
