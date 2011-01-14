'''
Created on Nov 30, 2009

@author: apollo
'''

from math import sqrt, factorial #@UnresolvedImport
from numpy import arange, zeros #@UnresolvedImport
from numpy.random import binomial #@UnresolvedImport
from time import clock #@UnresolvedImport
from matplotlib.pyplot import figure  #@UnresolvedImport
from dempster_shafer import MassFunction

def binomial_distribution(c, n, p):
    m = MassFunction()
    for k in range(n / 2 + 1):
        h = (c,) if k == n / 2 else (c + k - n / 2, c + n / 2 - k)
        v = len(h) * factorial(n) / (factorial(k) * factorial(n - k)) * p**k * (1.0 - p)**(n - k)
        m[h] = v
    return m

def vacuous_distribution(c, d):
    return MassFunction([(tuple(range(c - d / 2, c + d / 2 + 1)), 1.0)])

def plot_time(time_exact, time_sampled, sample_counts):
    fig = figure() # figsize = (3, 2)
    fig_gca = fig.gca()
    fig_gca.plot(time_exact, "-k", label = "exact")
    for ci, c in enumerate(sample_counts):
        fig_gca.plot(time_sampled[ci], "--k", label = str(c) + " sampled")
    fig_gca.set_title("computation time")
    fig_gca.set_xlabel("iterations")
    fig_gca.set_ylabel("ms")
    fig_gca.legend(loc=0)
    fig.savefig('plot-time.pdf')

def plot_distribution(m_exact, m_sampled, state, iterations):
    min, max = state - 10, state + 10
    pl_exact = zeros(max + 1 - min)
    pl_sampled = zeros((max + 1 - min, len(m_sampled)))
    pl_sampled_mean, pl_sampled_dev = pl_exact.copy(), pl_exact.copy()
    for s in range(max + 1 - min):
        pl_exact[s] = m_exact.plausibility((s + min,))
        for v, m in enumerate(m_sampled):
            pl_sampled[s, v] = m.plausibility((s + min,))
        pl_sampled_mean[s] = pl_sampled[s].mean()
        pl_sampled_dev[s] = sqrt(pl_sampled[s].var())
    fig = figure() # figsize = (3, 2)
    fig_gca = fig.gca()
    fig_gca.bar(arange(min, max + 1), pl_exact, yerr = pl_sampled_dev, color = "0.5", align="center")
    fig_gca.set_title("distribution after " + str(iterations) + " iterations")
    fig_gca.set_xlabel("state")
    fig_gca.set_ylabel("plausibility")
    fig.savefig('plot-distribution.pdf')

sample_counts = [1000]
state = 0
iterations = 20
variations = 10
dist_n = 30
dist_p = 0.5
time_exact = zeros(iterations + 1)
time_sampled = zeros((len(sample_counts), iterations + 1))
transition_belief = binomial_distribution(0, dist_n, dist_p)
m_exact = binomial_distribution(state, dist_n, dist_p)
m_sampled = [[m_exact.copy() for v in range(variations)] for c in sample_counts]

def fast_transition_model(s):
    global dist_n, dist_p
    x = abs(binomial(dist_n, dist_p) - dist_n / 2)
    if x == 0:
        return (s,)
    else:
        return (s - x, s + x)

for i in range(1, iterations + 1):
    print i
    state += transition_belief.pignistic().sample(1)[0][0]
    m_cor = binomial_distribution(state, dist_n, dist_p)
    t = clock()
    m_exact = m_exact.markov_update(lambda s: binomial_distribution(s, dist_n, dist_p))
    m_exact = m_exact & m_cor
    time_exact[i] = time_exact[i - 1] + clock() - t
    for ci, c in enumerate(sample_counts):
        time_sampled[ci, i] = time_sampled[ci, i - 1]
        for v in range(variations):
            t = clock()
            m = m_sampled[ci][v].markov_update(fast_transition_model, c)
            m = m.combine_conjunctive(m_cor, c)
            time_sampled[ci, i] += (clock() - t) / variations
            m_sampled[ci][v] = m

plot_time(time_exact, time_sampled, sample_counts)
plot_distribution(m_exact, m_sampled[0], state, iterations)