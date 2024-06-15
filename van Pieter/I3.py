import matplotlib.pyplot as plt
import mpmath as mp
import scipy.stats
import numpy as np
import networkx as nx
import math
import scipy.special
from scipy.integrate import quad
import seaborn
import matplotlib


matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['lines.markersize'] = 7
matplotlib.rcParams['figure.autolayout'] = True

colors = seaborn.color_palette('rocket')
colors.reverse()

pi = math.pi

mp.dps = 25; mp.pretty = True

def I3real(k, lbs, alpha, c):
    phi = lbs * pi * c**(2/alpha)
    return 1/(2*k**2*math.factorial(k-1)) * phi**k* ( alpha * mp.hyp2f2(k,k,1+k,1+k, -phi) + c**(-2*k/alpha) * ( -alpha * mp.hyp2f2(k,k,k+1,k+1, lbs *pi) + 2*k*(lbs *pi)**(-k)*(-math.factorial(k) + k * math.factorial(k-1)*scipy.special.gammaincc(k, lbs * pi))*math.log(c) ) )

def dgamma(k, x):
    return math.log(x) * scipy.special.gammaincc(k, x)*math.factorial(k-1) + mp.meijerg([[], [1,1]], [[0,0,k],[]],x)

def integrand(x, k, labdabs):
    return math.log(1+x)*x**(-2*k/alpha - 1)*math.exp(-labdabs*pi*c**(2/alpha)*x**(-2/alpha))

def I2real(k, lbs, alpha, c):
    phi = lbs * pi * c**(2/alpha)
    I = quad(integrandI2, 1, c, args = (k, lbs))
    return 2* (phi)**k/(alpha*math.factorial(k-1))* I[0]

def integrandI2(x, k, labdabs):
    return math.log(1+1/x)*x**(-2*k/alpha - 1)*math.exp(-labdabs*pi*c**(2/alpha)*x**(-2/alpha))

def I1real(k, lbs, alpha, c):
    phi = lbs * pi * c**(2/alpha)
    I = quad(integrandI1, 0, 1, args = (k, lbs))
    return 2* (phi)**k/(alpha*math.factorial(k-1))* I[0]

def integrandI1(x, k, labdabs):
    return math.log(1+x)*x**(-2*k/alpha - 1)*math.exp(-labdabs*pi*c**(2/alpha)*x**(-2/alpha))

def R(k, lbs, alpha, c):
    return math.log(1+c)*(1 - scipy.special.gammaincc(k, lbs * pi))

def I3approx_with_R(k, lbs, alpha, c):
    pi = math.pi
    phi = lbs * pi * c**(2/alpha)
    if k == 1:
        return alpha/(2) * (math.log(phi) + mp.euler - lbs *pi)
    else:
        return alpha/2 * (math.log(phi) + mp.euler - mp.harmonic(k-1))

def I3approx_with_R_sum(k, lbs, alpha, c):
    som = 0
    for j in range(k):
        som += I3approx_with_R(j + 1, lbs, alpha, c)
    return wtot / (k * lu) * som


max = 0.1
delta = 50

wtot = 20 * 10**6
lu = 1

x = np.arange(10**(-4), max, max/delta)
x1 = np.arange(10**(-4), max, max/delta)

alpha = 2
c = 10**(3.5)

# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
#           '#17becf']
markers = ['o', 's', 'p', 'd', '*']


fig, ax = plt.subplots()
for k in [1, 2, 3, 4, 5]:
    yyy = [I3real(k, lbs, alpha, c) for lbs in x]
    yy = [I2real(k, lbs, alpha, c) for lbs in x]
    y = [I1real(k, lbs, alpha, c) for lbs in x]
    r = [R(k, lbs, alpha, c) for lbs in x]
    sum = np.array(y) + np.array(yy) + np.array(yyy) + np.array(r)
    y1 = [I3approx_with_R(k, lbs, alpha, c) for lbs in x1]
    # y1sum = [I3approx_with_R_sum(k,lbs, alpha, c) for lbs in x1]

    # plt.plot(x, y, '+',  label = str("$I_1$"), color = colors[0])
    # plt.plot(x, yy, ':', label = str("$I_2$"), color = colors[1])
    # plt.plot(x, yyy, '-.',  label = str("$I_3$"), color = colors[2])
    # plt.plot(x, r, '--', label = str("R"), color = colors[3])
    # plt.plot(x, yyy, label = str('$j =$'+ str(k)))
    # plt.plot(x, sum, label = str('$I_1 + I_2 + I_3 + R$'), color = colors[4])
    plt.plot(x, sum, color = colors[k-1])

    y1[0] = 0
    # y1sum[0] = 0
    plt.scatter(x1, y1, label = str('$j =$ '+ str(k)), facecolor = 'none', color = colors[k-1], marker = markers[k-1])
    print(str('k = ' + str(k)))


# plt.ylabel('$E(\log_2(1+SNR_j))$')
plt.xlabel('$\lambda_{BS}$')
# ax.set_yscale('log')
# ax.set_xscale('log')
# plt.title(str(lu)+str(alpha) + str(c))
plt.legend()
plt.show()


