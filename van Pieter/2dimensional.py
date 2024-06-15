import networkx as nx
import scipy.stats
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy
import mpmath
import time
import pickle
import os

pointsPop = 10000

xMin, xMax = 0, 1000     # in meters
yMin, yMax = 0, 1000     # in meters

xDelta = xMax - xMin
yDelta = yMax - yMin


def initialise_graph(points):
    xx = xDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + xMin  # x coordinates of Poisson points
    yy = yDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + yMin  # y coordinates of Poisson points

    x = [xx[i][0] for i in range(points)]
    y = [yy[i][0] for i in range(points)]

    return x, y

def initialise_graph_users(radius):
    x, y = list(), list()
    for i in range(int(xDelta/radius) + 1):
        for j in range(int(yDelta/radius)+1):
            x.append(i * radius)
            y.append(j * radius % yDelta)
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

def find_bs(xpop, ypop, xbs, ybs, max_connections):
    indices = find_distance(xpop, ypop, xbs, ybs).argsort()
    return indices[:max_connections]

np.random.seed(49)

k = 5

number_of_iterations = 500
pointsBS = 1000
user_radius = 5

area = np.zeros((k, number_of_iterations * pointsBS))

area_cum = np.zeros((k, number_of_iterations * pointsBS))

Save = True

colors = ['g', 'b', 'r', 'y', 'c', 'm'] * 10
first = True
xpop, ypop = initialise_graph_users(user_radius)

name = str(str(number_of_iterations)+'iterations_user_radius'+str(user_radius) + 'k='+str(k))

if os.path.exists(str('Simulations/' + name + 'area.p')) and Save:
    print('This is already stored in memory')
    area = pickle.load(open(str('Simulations/' + name + 'area.p'), 'rb'))
    area_cum = pickle.load(open(str('Simulations/' + name + 'area_cum.p'), 'rb'))

else:
    for iterations in range(number_of_iterations):
        print(iterations)
        xbs, ybs = initialise_graph(pointsBS)
        pointsBS = len(xbs)
        pointsPop = len(xpop)
        # fig, ax = plt.subplots()
        # G, colorlist, nodesize = make_graph(xbs, ybs, xpop, ypop, pointsBS, pointsPop)
        # draw_graph(G, ax, colorlist, nodesize)
        # plt.show()
        begin = time.time()
        for node in range(pointsPop):
            connections = 0
            base_stations = find_bs(xpop[node], ypop[node], xbs, ybs, k)
            for i in range(k):
                area[i, base_stations[i] + iterations * pointsBS] += 1/40
                for j in range(k):
                    if j <= i:
                        area_cum[i,  base_stations[i] + iterations * pointsBS] += 1/40
        if first:
            total = number_of_iterations * (time.time()-begin)
            print('One iteration takes', time.time() - begin, 'seconds, so the total time will be', total/60, 'minutes')
            time_per_iteration = time.time()-begin
            first = False
        else:
            print((total - iterations * time_per_iteration)/60, 'minutes to go')
    if Save:
        pickle.dump(area, open(str(name + 'area.p'), 'wb'), protocol=4)
        pickle.dump(area_cum, open(str(name + 'area_cum.p'), 'wb'), protocol=4)

def cdf(x):
    return -1/15 * math.exp(-7*x/2)*math.sqrt(x)*(15 + 7*x*(5+7*x))*math.sqrt(14/math.pi) + mpmath.erf(math.sqrt(3.5*x))

def pdf(x):
    return 343/15 * math.sqrt(7/(2*math.pi))*x**(5/2)*math.exp(-7/2 * x)

labdaBS = pointsBS/((1.5 * xDelta)*(1.5*yDelta))
labdaU = len(xpop)/((1.5 * xDelta)*(1.5*yDelta))

pointsBS = number_of_iterations * pointsBS

max = max(area[0])

X = np.arange(0, max, 0.1)

fig, ax = plt.subplots()
for k in [1, 2, 3, 4, 5]:
    plt.hist(sorted([area[k - 1, bs] for bs in range(pointsBS)]), label = str('k = ' + str(k)), bins = 200, cumulative=True, histtype='step', density = True)
plt.plot(X, [cdf(x) for x in X], label = 'cdf for k = 1')
plt.xlabel('Area')
plt.legend()
plt.savefig(str(name + 'area1till5.png'))
plt.show()

# fig, ax = plt.subplots()
# for k in [1, 5, 10]:
#     plt.hist(sorted([area[k - 1, bs] for bs in range(pointsBS)]), label = str('k = ' + str(k)), bins = 100, cumulative=True, histtype='step', density = True)
# # plt.plot(X, [cdf(x) for x in X], label = 'cdf for k = 1')
# plt.xlabel('Area')
# plt.legend()
# plt.show()

# fig, ax = plt.subplots()
# for k in [1, 250, 500]:
#     plt.hist(sorted([area[k - 1, bs] for bs in range(pointsBS)]), label = str('k = ' + str(k)), bins = 100, cumulative=True, histtype='step', density = True)
# # plt.plot(X, [cdf(x) for x in X], label = 'cdf for k = 1')
# plt.xlabel('Area')
# plt.legend()
# plt.show()

def find_error(x0, x1):
    return math.sqrt(x0**2 + x1**2)

def cdf_gamma(a, b, x):
    return 1 - scipy.special.gammaincc(a, b * x)

def pdf_gamma(a, b, x):
    return b**a/scipy.special.gamma(a) * x**(a-1)*math.exp(-b * x)

def find_a_b_gamma(x ,y):
    y = y / y.sum()
    y = y.cumsum()
    best_R = math.inf
    for a in np.arange(1, 20, 0.05):
        # for b in np.arange(1, 100, 1):
        for b in [a]:
            R = 0
            for i in range(len(x) - 1):
                # print(i, x[i], y[i], cdf_gamma(a, b, x[i+1]))
                R += find_error(cdf_gamma(a, b, (x[i] + 0.5*(x[i+1]-x[i]))), y[i])
            if R < best_R:
                best_R = R
                best_a = a
                best_b = b
    return best_a, best_b, best_R

best_a = np.zeros(k)
best_b = np.zeros(k)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 10

fig, ax = plt.subplots()
for k in [1, 2, 3, 4, 5]:
    y, x = np.histogram(sorted([area[k - 1, bs] for bs in range(pointsBS)]), density=True, bins = 500)
    plt.hist(sorted([area[k - 1, bs] for bs in range(pointsBS)]), label = str('k = ' + str(k)), bins = 500, cumulative=True, histtype='step', density = True, color = colors[k-1])
    best_a[k-1], best_b[k-1], best_R = find_a_b_gamma(x, y)
    print('For k =', k, 'the best fit is a =', best_a[k-1], 'and b =', best_b[k-1], 'with error R =', best_R)
    plt.plot(X, [cdf_gamma(best_a[k-1], best_b[k-1], x) for x in X], '--', color = colors[k-1])
# plt.plot(X, [cdf(x) for x in X], label = 'cdf for k = 1')
    plt.xlabel('$A$')
    plt.savefig(str(name + 'k=' + str(k) + 'histogram.png'))
    plt.legend()
    plt.show()

fig, ax = plt.subplots()
x = np.arange(0, np.max(area_cum[-1,:]), 0.1)
for k in [1, 2, 3, 4, 5]:
    # y, x = np.histogram(sorted([area_cum[k - 1, bs] for bs in range(pointsBS)]), density=True, bins = 100)
    plt.hist(sorted([area_cum[k - 1, bs] for bs in range(pointsBS)]), label = str('k = ' + str(k)), bins = 500, cumulative=True, histtype='step', density = True, color = colors[k-1])
    # best_a, best_b, best_R = find_a_b_gamma(x/k, y)
    # print('For k =', k, 'the best fit is a =', best_a, 'and b =', best_b, 'with error R =', best_R)
    plt.plot(x, [cdf_gamma(best_a[k-1], best_b[k-1]/k, X) for X in x],'--', color = colors[k-1])
plt.xlabel('$A_{\leq k}$')
plt.savefig(str(name + '1till5leqk.png'))
plt.legend()
plt.show()

fig, ax = plt.subplots()
x = np.arange(0, np.max(area_cum[-1,:]), 0.1)
for k in [1, 2, 3, 4, 5]:
    # y, x = np.histogram(sorted([area_cum[k - 1, bs] for bs in range(pointsBS)]), density=True, bins = 100)
    plt.hist(sorted([area_cum[k - 1, bs] for bs in range(pointsBS)]), label = str('k = ' + str(k)), bins = 25, alpha = 0.3, cumulative=False, density = True, color = colors[k-1])
    # best_a, best_b, best_R = find_a_b_gamma(x/k, y)
    # print('For k =', k, 'the best fit is a =', best_a, 'and b =', best_b, 'with error R =', best_R)
    plt.plot(x, [pdf_gamma(best_a[k-1], best_b[k-1]/k, X) for X in x],'--', color = colors[k-1])
    plt.xlabel('$A_{\leq k}$')
    plt.savefig(str(name + 'k=' + str(k) + 'with_approxcdf.png'))
    plt.legend()
    plt.show()