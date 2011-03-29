'''
Created on Mar 29, 2011

@author: reineking
'''

import inspect
import sys
import time
import random
from pyds import MassFunction, gbt_m, gbt_bel, gbt_pl, gbt_q
import numpy


iterations = 10

def measure(f, *args):
    def f_timed(i):
        t = time.clock()
        f(*args)
        return time.clock() - t
    array = numpy.empty((iterations, 1))
    for i, t in enumerate(map(f_timed, range(iterations))):
        array[i] = t
    return array.mean(), array.std()

def random_likelihoods(singleton_count):
    return [(i, random.random()) for i in range(singleton_count)]


def measure_bel():
    return measure(MassFunction.gbt(random_likelihoods(12)).bel, frozenset(range(8)))

def measure_plausibility():
    return measure(MassFunction.gbt(random_likelihoods(12)).pl, frozenset(range(8)))

def measure_commonality():
    return measure(MassFunction.gbt(random_likelihoods(12)).q, frozenset(range(8)))

def measure_gbt():
    return measure(MassFunction.gbt, random_likelihoods(12))

def measure_gbt_m():
    return measure(gbt_m, frozenset(range(8)), random_likelihoods(12))

def measure_gbt_bel():
    return measure(gbt_bel, frozenset(range(8)), random_likelihoods(12))

def measure_gbt_pl():
    return measure(gbt_pl, frozenset(range(8)), random_likelihoods(12))

def measure_gbt_q():
    return measure(gbt_q, frozenset(range(8)), random_likelihoods(12))

def measure_combine_conjunctive():
    m1 = MassFunction.gbt(random_likelihoods(6))
    m2 = MassFunction.gbt(random_likelihoods(6))
    return measure(m1.combine_conjunctive, m2)

def measure_combine_disjunctive():
    m1 = MassFunction.gbt(random_likelihoods(6))
    m2 = MassFunction.gbt(random_likelihoods(6))
    return measure(m1.combine_disjunctive, m2)


if __name__ == '__main__':
    print('Measuring PyDS time performance using %d iterations' % iterations)
    mod = sys.modules[__name__]
    filt = lambda x: inspect.isfunction(x) and inspect.getmodule(x) == mod and x.__name__.startswith('measure_')
    print('\n%-22s%-7s (%4s)' % ('function', 'mean', 'stddev'))
    print('=' * 40)
    for f in sorted(filter(filt, globals().copy().values()), key=str):
        random.seed(0)
        print('%-22s%.4fs (+-%.4f)' % ((f.__name__[8:],) + f()))
