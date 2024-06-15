import matplotlib.pyplot as plt
import mpmath as mpmath
import scipy.stats
import numpy as np
import networkx as nx
import math
from scipy.special import gamma, loggamma
from scipy.integrate import quad
import mpmath as mp

pi = math.pi

a = [3.527, 7.187, 11.062, 15.212, 21.166]
b = [3.527, 7.187, 11.062, 15.212, 21.166]

# ------------------- SNR and channel capacity ----------------------------------
def find_expectation_log_SNR_numerical(k, alpha, lbs, c):
    I = quad(integrand, 0, c, args = (k, alpha, lbs * pi * c**(2/alpha)))
    return I[0] + math.log2(1+c) * (1 - mpmath.gammainc(k, lbs*pi)/gamma(k))

def integrand(x, k, alpha, phi):
    return math.log2(1+x) * 2 * (phi * x**(-2/alpha))**k/(alpha * x * gamma(k)) * math.exp(-phi * x**(-2/alpha))

def expected_w(lu, wtot, k):
    return wtot / (k * lu)

def channel_capacity(k, lu, lbs, p, c, alpha, wtot):
    som = 0
    for j in range(1, k+1):
        som += (1-p[j-1]) * find_expectation_log_SNR_numerical(j, alpha, lbs, c)
    W = expected_w(lu, wtot, k)
    return som * W

# ------------------- RANDOM FAILURE------------------------------------------
def find_expectation_log_SNR_numerical_random(k, alpha, lbs, c, p):
    som = 0
    for j in range(k):
        I = quad(integrand, 0, c, args = (j+1, alpha, lbs * pi * c**(2/alpha)))
        som += (1 - p) * (I[0] + math.log2(1+c) * (1 - mpmath.gammainc(j+1, lbs*pi)/gamma(j)))
    return som

def find_probability_random(k, p):
    return [p] * k

# ------------------ DETERMINISTIC DISTANCE FAILURE ----------------------------------
def channel_capacity_dist(k, lu, lbs, rmax, c, alpha, wtot): # for deterministic distance failures
    som = 0
    for j in range(1, k+1):
        som += find_expectation_log_SNR_numerical_dist(j, alpha, lbs, c, rmax)
    return expected_w(lu, wtot, k) * som

def find_expectation_log_SNR_numerical_dist(k, alpha, lbs, c, rmax):
    I = quad(integrand, c*rmax**(-alpha), c, args=(k, alpha, lbs * pi * c ** (2 / alpha)))
    return I[0] + math.log2(1 + c) * (1 - mpmath.gammainc(k, lbs * pi) / math.factorial(k - 1))

def integrand_dist(x, lbs, k, c, alpha):
    return math.log2(1+c*x**(-alpha)) * 2 * (lbs * pi * x**2)**k/(x * gamma(k)) * math.exp(-lbs * pi * x**2)

def find_probability_distance_deterministic(k, lbs, rmax):
    p = np.zeros(k)
    for j in range(1, k+1):
        p[j - 1] = scipy.special.gammaincc(j, lbs * pi * rmax**2)
    return p

def outage_probability_deterministic_distance(lbs, rmax):
    return math.exp(-lbs *pi*rmax**2)


# ----------- PROPORTIONAL DISTANCE FAILURE --------------------------------------

def channel_capacity_dist_prop(k, lu, lbs, beta, c, alpha, wtot):
    som = 0
    for j in range(1, k+1):
        som += find_expectation_log_SNR_dist_prop(j, alpha, lbs, c, beta)
    W = expected_w(lu, wtot, k)
    return som * W

def find_expectation_log_SNR_dist_prop(j, alpha, lbs, c, beta):
    I = quad(integrand_dist_prop, 1, math.inf, args = (j, alpha, lbs, c, beta))
    som = I[0] + math.log2(1+c) * (1 - mpmath.gammainc(j, lbs*pi)/gamma(j))
    return som

def integrand_dist_prop(x, k, alpha, lbs, c, beta):
    return math.log2(1+c*x**(-alpha)) * x**(-beta) * 2 * (lbs * pi * x**2)**k/(x * gamma(k)) * math.exp(-lbs * pi * x**2)

def find_probability_distance_proportional(k, lbs, beta):
    p = np.zeros(k)
    for j in range(1, k+1):
        p[j - 1] = 1 - (lbs * pi)**(beta/2) * gamma(j-beta/2)/gamma(j)
    return p

def outage_probability_proportional_distance(lbs, k, beta):
    p = find_probability_distance_proportional(k, lbs, beta)
    prod = 1
    for j in range(k):
        prod = prod * p[j-1]
    return prod

# -------------- LINE OF SIGHT FAILURE -------------------------------------------

def channel_capacity_los(k, lu, lbs, c, alpha, wtot, rlos):
    som = 0
    for j in range(1, k+1):
        som += find_expectation_log_SNR_los(j, alpha, lbs, c, rlos)
    W = expected_w(lu, wtot, k)
    return som * W

def find_expectation_log_SNR_los(j, alpha, lbs, c, rlos):
    I1 = quad(integrand_los1, 1, rlos, args = (j, alpha, lbs, c))
    I2 = quad(integrand_los2, rlos, math.inf, args = (j, alpha, lbs, c, rlos))
    return (I1[0] + math.log2(1+c) * (1 - mpmath.gammainc(j, lbs*pi)/gamma(j))) + I2[0]

def integrand_los1(x, k, alpha, lbs, c):
    return math.log2(1+c*x**(-alpha)) * 2 * (lbs * pi * x**2)**k/(x * gamma(k)) * math.exp(-lbs * pi * x**2)

def integrand_los2(x, k, alpha, lbs, c, rlos):
    return (x**(-1) * (rlos + x * math.exp(-x/(2*rlos)) - rlos * math.exp(-x/(2*rlos)))) * math.log2(1+c*x**(-alpha)) * 2 * (lbs * pi * x**2)**k/(x * gamma(k)) * math.exp(-lbs * pi * x**2)

def find_probability_LoS(k, lbs, rlos):
    p = np.zeros(k)
    for j in range(1, k+1):
        I = quad(integrand_los_outage, rlos, math.inf, args=(j, lbs, rlos))
        # p[j - 1] = mpmath.gammainc(j, lbs * pi * 18**2)/gamma(j) - I[0]
        p[j-1] = I[0]
    return p

def integrand_los_outage(x, j, lbs, rlos):
    return (1- x**(-1) * ( rlos + x*math.exp(-x/(2*rlos)) - rlos * math.exp(-x/(rlos*2)))) * 2 * (lbs * pi * x**2)**j/(x * gamma(j)) * math.exp(-lbs * pi * x**2)

def outage_probability_LoS(lbs, k, rlos):
    prod = 1
    p = find_probability_LoS(k, lbs, rlos)
    for j in range(k):
        prod = prod * p[j]
    return prod

# ------------- DETERMINISTIC OVERLOAD FAILURE -----------------------------------------
def channel_capacity_overload_det(k, lu, lbs, K, c, alpha, wtot):
    p = find_probability_overload_deterministic(k, lu/lbs, K)
    som = 0
    labda = lu/lbs
    ak = a[k-1]
    bk = b[k-1]
    t = (bk/(bk + k*labda))**ak
    s = labda * k /(labda *k + bk)
    for j in range(1, k+1):
        som += find_expectation_log_SNR_numerical(j, alpha, lbs, c)
    expected_DBS = 1/(k*lu)
    return (1-p[k-1]) * wtot * expected_DBS * som

def find_expectation_W_overload_det(wtot, lu, lbs, k, K):
    ak, bk = a[k-1], b[k-1]
    labda = lu/lbs
    t = (bk/(labda * k + bk))**ak
    return wtot/(k*lu) * (1 - K/(k*labda) * (k*labda/(k * labda + bk))**K * t * scipy.special.gamma(K+ak)/(scipy.special.gamma(ak) * scipy.special.gamma(K+1))*mpmath.hyp2f1(1, K+ak, K+1, K*labda/(k*labda + bk)))

def find_probability_overload_deterministic(k, labda, N):
    ak = a[k-1]
    bk = b[k-1]
    t = (bk/(labda * k + bk))**ak
    blub1 = loggamma(N + ak) - loggamma(ak) - loggamma(N)
    blub2 = N * math.log(labda * k/(labda*k + bk))
    p = t * 1/(k*labda) * math.exp(blub1 + blub2) * mpmath.hyp2f1(1, N + ak, N , k*labda/(k*labda + bk))
    return [p] * k

def outage_probability_overload_det(k, labda, N):
    p = find_probability_overload_deterministic(k, labda, N)
    prod = 1
    for j in range(k):
        prod = prod * p[k-1]
    return prod


# -------------- PROPORTIONAL OVERLOAD FAILURE -----------------------------------
def channel_capacity_overload(k, lu, lbs, beta, c, alpha, wtot):
    som = 0
    for j in range(1, k+1):
        som += find_expectation_log_SNR_numerical(j, alpha, lbs, c)
    return find_expectation_W_overload(wtot, lu, lu/lbs, k, beta) * som

def find_expectation_W_overload(wtot, lu, labda, k, beta):
    ak, bk = a[k-1], b[k-1]
    som = 0
    t = (bk/(labda * k + bk))**ak
    oud, nieuw = 0, 10
    n = 0
    while n < 1000000 and abs(oud - nieuw) > 0.000001:
        oud = nieuw
        n += 1
        blub = loggamma(n+ak) - loggamma(ak) - loggamma(n+1)
        som += math.exp(blub) * (labda * k/(labda * k + bk))**n * n**(-beta)
        nieuw = som
    return wtot * 1/(k * lu) * t * som

def find_probability_overload_proportional(k, labda, beta):
    ak = a[k-1]
    bk = b[k-1]
    som = 0
    t = (bk/(labda * k + bk))**ak
    oud = 0
    nieuw = 10
    n = 0
    while n < 1000000 and abs(oud - nieuw) > 0.000001:
        oud = nieuw
        n += 1
        blub1 = loggamma(n+ak) - loggamma(ak) - loggamma(n)
        blub2 = n**(-beta)*(labda *k/(labda *k + bk))**n
        som += math.exp(blub1) * blub2
        nieuw = som
    p = 1 - (t/(k*labda) * som)
    return [p] * k


def outage_probability_overload(k, labda, beta):
    prod = 1
    p = find_probability_overload_proportional(k, labda, beta)
    for j in range(k):
        prod = prod * p[k-1]
    return prod


# ---------------- APPROXIMATIONS ---------------------------
def I3approx_with_R(k, lbs, alpha, c):
    pi = math.pi
    phi = lbs * pi * c**(2/alpha)
    if k == 1:
        return 1/math.log(2)* (math.log(c) + alpha/(2) * (-phi*math.exp(-phi) + math.log(lbs * pi) + mp.euler - lbs *pi))
    else:
        return 1/math.log(2)*(alpha * phi**k/(2 * k**2* math.factorial(k-1)) * mp.hyp2f2(k,k,k+1, k+1, -phi))

def approximate_expected_w(lu, wtot, k):
    return wtot/(k * lu)

def approximate_channel_capacity(k, lu, lbs, p, c, alpha, wtot):
    som = 0
    labda = lu/lbs
    for j in range(1, k+1):
        som += (1 - p[j-1]) * I3approx_with_R(j, lbs, alpha, c)
    W = expected_w(lbs, labda, wtot, k)
    return som * W



# --------------- outage probabilities ------------------------





if __name__ == '__main__':

    # -------------------- initialisation ---------------------------
    lbs = 10**(-4)
    wtot = 20 * 10**6
    lu = 0.5
    labda = lbs/lu
    N = 1
    rmax = 5
    beta = 2
    eta = 1

    alpha = 4
    if alpha == 2:
        c = 12800
    elif alpha == 3:
        c = 2.56*10**6
    elif alpha == 4:
        c = 5.12*10**8

    maxlbs = 1
    delta = 100

    maxlu = 2
    name = str('Pictures/alpha=' + str(alpha) + '_c=' + str(c) + '_wtot=' + str(wtot/(10**6)) + '_lbs=' + str(lbs))


    Approximate = False
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 10

    Plot_Lu_vs_Lbs =        False
    No_Failure =            True
    Random_Failure =        True
    Overload_Failure =      False
    Size_Biased =           False
    Distance_Failure_Det =  False
    Distance_Failure_Prop = False
    LoS_Failure =           False

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * 10

    # ----------------- No failures -------------------------------
    # fig, ax = plt.subplots()
    x = np.arange(10**(-6), maxlbs, maxlbs/delta)
    xlu = np.arange(10**(-6), maxlu, maxlu/delta)

    if Plot_Lu_vs_Lbs:
        First1 = True
        First2 = True
        First3 = True
        First4 = True
        First5 = True

        blub = 0
        for lu in xlu:
            print(blub)
            blub +=1
            for k in [1, 2, 3, 4, 5]:
                EC = [channel_capacity(k, lu, lbs, [0]*k, c, alpha, wtot) for lbs in x]
                if Approximate:
                    ECapproximate = [approximate_channel_capacity(k, lu, lbs, [0]*k, c, alpha, wtot) for lbs in x]
                    plt.scatter(x, ECapproximate, color = 'k', facecolors = 'none')
                if k == 1:
                    EC1 = EC
                elif k == 2:
                    EC2 = EC
                elif k == 3:
                    EC3 = EC
                elif k == 4:
                    EC4 = EC
                elif k == 5:
                    EC5 = EC
            for i in range(len(EC1)):
                maximum = max(EC1[i], EC2[i], EC3[i], EC4[i], EC5[i])
                if EC1[i] == maximum:
                    # print('For', x[i], '1 is highest')
                    if First1:
                        plt.scatter(x[i], lu, color = colors[0], label = 'k = 1')
                        First1 = False
                    else:
                        plt.scatter(x[i], lu, color = colors[0])

                elif EC2[i] == maximum:
                    # print('For', x[i], '2 is highest')
                    if First2:
                        plt.scatter(x[i], lu, color = colors[1], label = 'k = 2')
                        First2 = False
                    else:
                        plt.scatter(x[i], lu, color = colors[1])
                elif EC3[i] == maximum:
                    # print('For', x[i], '3 is highest')
                    if First3:
                        plt.scatter(x[i], lu, color = colors[2], label = 'k = 3')
                        First3 = False
                    else:
                        plt.scatter(x[i], lu, color = colors[2])
                elif EC4[i] == maximum:
                    # print('For', x[i], '4 is highest')
                    if First4:
                        plt.scatter(x[i], lu, color = colors[3], label = 'k = 4')
                        First4 = False
                    else:
                        plt.scatter(x[i], lu, color = colors[3])
                elif EC5[i] == maximum:
                    # print('For', x[i], '5 is highest')
                    if First5:
                        plt.scatter(x[i], lu, color = colors[4], label = 'k = 5')
                        First5 = False
                    else:
                        plt.scatter(x[i], lu, color = colors[4])
        plt.legend()
        plt.xlabel('$\lambda_{BS}$')
        plt.ylabel('$\lambda_U$')
        plt.show()

    # --------------------- No failure
    if No_Failure:
        fig, ax = plt.subplots()
        for k in [1, 2, 3, 4, 5]:
            EC = [channel_capacity(k, lu, lbs, [0]*k, c, alpha, wtot) for lbs in x]
            plt.plot(x, EC, label = str('$k = $' + str(k)))
        # plt.title('Expected channel capacity')
        plt.legend()
        plt.xlabel('$\lambda_{BS}$')
        plt.ylabel('$\mathrm{E}(C)$ in $bps$')
        plt.savefig(str(name + 'channnel_no_failure' + '.png'))
        plt.show()

        fig, ax = plt.subplots()
        for k in [1, 2, 3, 4, 5]:
            EC = [expected_w(lbs, labda, wtot, k) for lbs in x]
            plt.plot(x, EC, label = str('$k = $' + str(k)), color = colors[k-1])
        plt.legend()
        plt.xlabel('$\lambda_{BS}$')
        plt.ylabel('$\mathrm{E}(W)$ in $Hz$')
        plt.savefig(str(name + 'expected_w' + '.png'))
        plt.ylim(0, 10**9)
        plt.show()

        fig, ax = plt.subplots()
        for k in [1, 2, 3, 4, 5]:
            EC = [find_expectation_log_SNR_numerical(k, alpha, lbs, c) for lbs in x]
            plt.plot(x, EC, label = str('$k = $' + str(k)))
        plt.legend()
        plt.xlabel('$\lambda_{BS}$')
        plt.ylabel('$\mathrm{E}(\log_2(1+SNR_k))$')
        plt.savefig(str(name + 'logSNR' + '.png'))
        plt.show()
    min = 0
    max = 2.5
    for lbs in [10**(-4), 0.40]: #, 10**(-3), 10**(-2), 10**(-1), 10**0]:
        name = str('Pictures/alpha=' + str(alpha) + '_c=' + str(c) + '_wtot=' + str(wtot / (10 ** 6)) + '_lbs=' + str(lbs))
        if Random_Failure:
            print('lbs =', lbs)
            name = str('Pictures/alpha=' + str(alpha) + '_c=' + str(c) + '_wtot=' + str(wtot / (10 ** 6)) + '_lbs=' + str(lbs))
            # ---------------------- Random failures
            fig, ax = plt.subplots()
            x = np.arange(0, 1, 1/delta)
            for k in [1, 2, 3, 4, 5]:
                EC = [channel_capacity(k, lu, lbs, [p]*k, c, alpha, wtot) for p in x]
                plt.plot(x, EC, label = str('$k = $' + str(k)), color = colors[k-1])
            plt.legend()
            plt.xlabel('$p$')
            plt.ylabel('$E(C)$ in $bps$')
            plt.savefig(str(name + 'channnel_random_failure' + '.png'))
            plt.show()

            fig, ax = plt.subplots()
            x = np.arange(0, 1, 1/delta)
            for k in [1, 2, 3, 4, 5]:
                outage_prob = [p**k for p in x]
                plt.plot(x, outage_prob,  label = str('$k = $' + str(k)))
            plt.legend()
            plt.xlabel('$p$')
            plt.ylabel('Outage probability')
            plt.savefig(str(name + 'outage_random_failure' + '.png'))
            plt.show()
        if Overload_Failure:
            if Size_Biased:
                # Overload failure proportional
                fig, ax = plt.subplots()
                x = np.arange(min, max, max/delta)
                for k in [1, 2, 3, 4, 5]:
                    EC = [channel_capacity(k, lu, lbs, find_probability_overload_proportional_size_biased(k, lu/lbs, beta, eta), c, alpha, wtot) for beta in x]
                    plt.plot(x, EC, label = str('$k = $' + str(k)))
                # plt.title('Overload failure - proportional')
                plt.legend()
                plt.xlabel('beta')
                plt.ylabel('$E(C)$ in bps')
                plt.savefig(str(name + 'channel_overload_failure_size_biased' + '.png'))
                plt.show()

                fig, ax = plt.subplots()
                x = np.arange(min, max, max/delta)
                for k in [1, 2, 3, 4, 5]:
                    outage_prob = [outage_probability_overload_size_biased(k, lu/lbs, beta, eta) for beta in x]
                    plt.plot(x, outage_prob,  label = str('$k = $' + str(k)))
                plt.legend()
                plt.xlabel('beta')
                plt.ylabel('Outage probability')
                plt.savefig(str(name + 'outage_overload_failure_size_biased' + '.png'))
                plt.show()
            else:
                fig, ax = plt.subplots()
                x = np.arange(min, max, max / delta)
                for k in [1, 2, 3, 4, 5]:
                    EC = [channel_capacity(k, lu, lbs, find_probability_overload_proportional(k, lu / lbs, beta, eta), c,
                                           alpha, wtot) for beta in x]
                    plt.plot(x, EC, label=str('$k = $' + str(k)))
                # plt.title('Overload failure - proportional')
                plt.legend()
                plt.xlabel('beta')
                plt.ylabel('$E(C)$ in bps')
                plt.savefig(str(name + 'channel_overload_failure' + '.png'))
                plt.show()

                fig, ax = plt.subplots()
                x = np.arange(min, max, max / delta)
                for k in [1, 2, 3, 4, 5]:
                    outage_prob = [outage_probability_overload(k, lu / lbs, beta, eta) for beta in x]
                    plt.plot(x, outage_prob, label=str('$k = $' + str(k)))
                plt.legend()
                plt.xlabel('beta')
                plt.ylabel('Outage probability')
                plt.savefig(str(name + 'outage_overload_failure' + '.png'))
                plt.show()
        if Distance_Failure_Det:
            # Distance failure deterministic
            fig, ax = plt.subplots()
            x = np.arange(0, 5, 5/delta)
            for k in [1, 2, 3, 4, 5]:
                EC = [channel_capacity(k, lu, lbs, find_probability_distance_deterministic(k, lbs, rf), c, alpha, wtot) for rf in x]
                plt.plot(x, EC, label = str('$k = $' + str(k)))
            # plt.title('Distance failure - deterministic')
            plt.legend()
            plt.xlabel('$r_{max}$')
            plt.ylabel('$E(C)$ in bps')
            plt.savefig(str(name + 'channel_distance_failure_deterministic' + '.png'))
            plt.show()

            fig, ax = plt.subplots()
            x = np.arange(0, 5, 5/delta)
            for k in [1, 2, 3, 4, 5]:
                outage_prob = [outage_probability_deterministic_distance(lbs, rf) for rf in x]
                plt.plot(x, outage_prob,  label = str('$k = $' + str(k)))
            plt.legend()
            plt.xlabel('$r_{max}$')
            plt.ylabel('Outage probability')
            plt.savefig(str(name + 'outage_distance_failure_deterministic' + '.png'))
            plt.show()
        if Distance_Failure_Prop:
            # Distance failure proportional
            fig, ax = plt.subplots()
            x = np.arange(0, 3, 3/delta)
            for k in [1, 2, 3, 4, 5]:
                EC = [channel_capacity(k, lu, lbs, find_probability_distance_proportional(k, lbs, beta), c, alpha, wtot) for beta in x]
                plt.plot(x, EC, label = str('$k = $' + str(k)))
            # plt.title('Distance failure - proportional')
            plt.legend()
            plt.xlabel('beta')
            plt.ylabel('$E(C)$ in bps')
            plt.savefig(str(name + 'channel_distance_failure_proportional' + '.png'))
            plt.show()

            fig, ax = plt.subplots()
            x = np.arange(0, 3, 3/delta)
            for k in [1, 2, 3, 4, 5]:
                outage_prob = [outage_probability_proportional_distance(lbs, k, beta) for beta in x]
                plt.plot(x, outage_prob,  label = str('$k = $' + str(k)))
            plt.legend()
            plt.xlabel('beta')
            plt.ylabel('Outage probability')
            plt.savefig(str(name + 'outage_distance_failure_proportional' + '.png'))
            plt.show()
    if LoS_Failure:
        # Line of Sight failure
        fig, ax = plt.subplots()
        x = np.arange(10**(-6), 10**(-2), 10**(-2)/delta)
        for k in [1, 2, 3, 4, 5]:
            EC = [channel_capacity(k, lu, lbs, find_probability_LoS(k, lbs), c, alpha, wtot) for lbs in x]
            plt.plot(x, EC, label = str('$k = $' + str(k)))
        plt.legend()
        # plt.title('Line of Sight failure')
        plt.xlabel('$\lambda_{BS}$')
        plt.ylabel('$E(C)$ in bps')
        plt.savefig(str(name + 'channel_LoS_failure' + '.png'))
        plt.show()

        fig, ax = plt.subplots()
        x = np.arange(10**(-4), 6*10**(-4), 6*10**(-4)/delta)
        for k in [1, 2, 3, 4, 5]:
            EC = [channel_capacity(k, lu, lbs, find_probability_LoS(k, lbs), c, alpha, wtot) for lbs in x]
            plt.plot(x, EC, label = str('$k = $' + str(k)))
        plt.legend()
        # plt.title('Line of Sight failure')
        plt.xlabel('$\lambda_{BS}$')
        plt.ylabel('$E(C)$ in bps')
        plt.savefig(str(name + 'channel_LoS_failure-zoomed-in' + '.png'))
        plt.show()

        fig, ax = plt.subplots()
        x = np.arange(10**(-6), 10**(-2), 10**(-2)/delta)
        for k in [1, 2, 3, 4, 5]:
            outage_prob = [outage_probability_LoS(lbs, k) for lbs in x]
            plt.plot(x, outage_prob,  label = str('$k = $' + str(k)))
        plt.legend()
        # plt.title('Line of Sight failure')
        plt.xlabel('$\lambda_{BS}$')
        plt.ylabel('Outage probability')
        plt.savefig(str(name + 'outage_LoS_failure' + '.png'))
        plt.show()