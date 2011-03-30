# Copyright (C) 2010-2011  Thomas Reineking
#
# This file is part of the PyDS library.
# 
# PyDS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PyDS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""
A framework for performing computations in the Dempster-Shafer theory.
"""

from itertools import chain, combinations, product
from functools import partial, reduce
from operator import mul
from math import log, sqrt
from random import random, shuffle, uniform


class MassFunction(dict):
    """
    A Dempster-Shafer mass function (basic probability assignment) based on a dictionary.
    
    Both normalized and unnormalized mass functions are supported.
    The underlying frame of discernment is assumed to be discrete.
    
    Hypotheses and their associated mass values can be added/changed/removed using the standard dictionary methods.
    Each hypothesis can be an arbitrary sequence which is automatically converted to a 'frozenset', meaning its elements must be hashable.
    """
    
    def __init__(self, source=None):
        """
        Creates a new mass function.
        
        If 'source' is not None, it is used to initialize the mass function.
        It can either be a dictionary mapping hypotheses to non-negative mass values
        or an iterable containing tuples consisting of a hypothesis and a corresponding mass value.
        """
        if source != None:
            if isinstance(source, dict):
                source = source.items()
            for h, v in source:
                self[h] += v
    
    @staticmethod
    def _convert(hypothesis):
        """Convert hypothesis to a 'frozenset' in order to make it hashable."""
        if isinstance(hypothesis, frozenset):
            return hypothesis
        else:
            return frozenset(hypothesis)
    
    @staticmethod
    def gbt(likelihoods, normalization=True, sample_count=None):
        """
        Constructs a mass function using the generalized Bayesian theorem.
        For more information, see Ph. Smets, 1993. Belief functions: 
        The disjunctive rule of combination and the generalized Bayesian theorem. International Journal of Approximate Reasoning. 
        
        'likelihoods' specifies the conditional plausibilities for a set of singleton hypotheses.
        It can either be a dictionary mapping singleton hypotheses to plausibilities or an iterable
        containing tuples consisting of a singleton hypothesis and a corresponding plausibility value.
        
        'normalization' determines whether the resulting mass function is normalized, i.e., whether m({}) == 0.
        
        If 'sample_count' is not None, the true mass function is approximated using the specified number of samples.
        """
        m = MassFunction()
        # filter trivial likelihoods 0 and 1
        ones = [h for (h, l) in likelihoods if l >= 1.0]
        likelihoods = [(h, l) for (h, l) in likelihoods if 0.0 < l < 1.0]
        if sample_count == None:   # deterministic
            def traverse(m, likelihoods, ones, index, hyp, mass):
                if index == len(likelihoods):
                    hyp += ones
                    if not normalization or len(hyp) > 0:
                        m[hyp] = mass
                else:
                    traverse(m, likelihoods, ones, index + 1, hyp + [likelihoods[index][0]], mass * likelihoods[index][1])
                    traverse(m, likelihoods, ones, index + 1, hyp, mass * (1.0 - likelihoods[index][1]))
            traverse(m, likelihoods, ones, 0, [], 1.0)
            if normalization:
                m.normalize()
        else:   # Monte-Carlo
            empty_mass = reduce(mul, [1.0 - l[1] for l in likelihoods], 1.0)
            for _ in range(sample_count):
                rv = [random() for _ in range(len(likelihoods))]
                subtree_mass = 1.0
                hyp = set(ones)
                for k in range(len(likelihoods)):
                    l = likelihoods[k][1]
                    p_t = l * subtree_mass
                    p_f = (1.0 - l) * subtree_mass
                    if normalization and not hyp: # avoid empty hypotheses in the normalized case
                        p_f -= empty_mass
                    if p_t > rv[k] * (p_t + p_f):
                        hyp.add(likelihoods[k][0])
                    else:
                        subtree_mass *= 1 - l # only relevant for the normalized empty case
                m[hyp] += 1.0 / sample_count
        return m
    
    @staticmethod
    def from_bel(bel):
        """
        Creates a mass function from a corresponding belief function.
        
        'bel' is a dictionary mapping hypotheses to belief values like to one returned by 'bel()'.
        """
        m = MassFunction()
        for h1 in bel.keys():
            v = sum([bel[h2] * (-1)**(len(h1 - h2)) for h2 in powerset(h1)])
            if v > 0:
                m[h1] = v
        return m
    
    @staticmethod
    def from_pl(pl):
        """
        Creates a mass function from a corresponding plausibility function.
        
        'pl' is a dictionary mapping hypotheses to plausibility values like to one returned by 'pl()'.
        """
        frame = max(pl.keys(), key=len)
        bel_theta = pl[frame]
        bel = {frozenset(frame - h):bel_theta - v for h, v in pl.items()} # follows from bel(-A) = bel(frame) - pl(A)
        return MassFunction.from_bel(bel)
    
    @staticmethod
    def from_q(q):
        """
        Creates a mass function from a corresponding commonality function.
        
        'q' is a dictionary mapping hypotheses to commonality values like to one returned by 'q()'.
        """
        m = MassFunction()
        frame = max(q.keys(), key=len)
        for h1 in q.keys():
            v = sum([q[h1 | h2] * (-1)**(len(h2 - h1)) for h2 in powerset(frame - h1)])
            if v > 0:
                m[h1] = v
        return m
    
    def __missing__(self, key):
        """Return 0 mass for hypotheses that are not contained."""
        return 0.0
    
    def __copy__(self):
        c = MassFunction()
        for k, v in self.items():
            c[k] = v
        return c
    
    def copy(self):
        """Creates a shallow copy of the mass function."""
        return self.__copy__()
    
    def __contains__(self, hypothesis):
        return dict.__contains__(self, MassFunction._convert(hypothesis))
    
    def __getitem__(self, hypothesis):
        return dict.__getitem__(self, MassFunction._convert(hypothesis))
    
    def __setitem__(self, hypothesis, value):
        """
        Adds or updates the mass value of a hypothesis.
        
        'hypothesis' is automatically converted to a 'frozenset' meaning its elements must be hashable.
        In case of a negative mass value, a ValueError is raised.
        """
        if value < 0.0:
            raise ValueError("mass value is negative: %f" % value)
        dict.__setitem__(self, MassFunction._convert(hypothesis), value)
    
    def __delitem__(self, hypothesis):
        return dict.__delitem__(self, MassFunction._convert(hypothesis))
    
    def frame(self):
        """
        Returns the frame of discernment of the mass function as a 'frozenset'.
        
        The frame of discernment is the union of all contained hypotheses.
        In case the mass function does not contain any hypotheses, an empty set is returned.
        """
        if not self:
            return frozenset()
        else:
            return frozenset.union(*self.keys())
    
    def focal(self):
        """
        Returns the set of all focal hypotheses.
        
        A focal hypothesis has a mass value greater than 0.
        """
        return {h for h, v in self.items() if v > 0}
    
    def core(self):
        """
        Returns the core of the mass function as a 'frozenset'.
        
        The core is the union of all focal hypotheses.
        In case the mass function does not contain any focal hypotheses, an empty set is returned.
        """
        focal = self.focal()
        if not focal:
            return frozenset()
        else:
            return frozenset.union(*focal)
    
    def combined_core(self, *mass_functions):
        """Returns the combined core of two or more mass functions as a 'frozenset'."""
        return frozenset.intersection(self.core(), *[m.core() for m in mass_functions])
    
    def bel(self, hypothesis=None):
        """
        Computes either the belief of 'hypothesis' or the entire belief function (hypothesis=None).
        
        If 'hypothesis' is None (default), a dictionary mapping hypotheses to their respective belief values is returned.
        Otherwise, the belief of 'hypothesis' is returned.
        In this case, 'hypothesis' is automatically converted to a 'frozenset' meaning its elements must be hashable.
        """
        if hypothesis is None:
            return {h:self.bel(h) for h in powerset(self.core())}
        else:
            hypothesis = MassFunction._convert(hypothesis)
            if not hypothesis:
                return 0.0
            else:
                return sum([v for h, v in self.items() if hypothesis.issuperset(h)])
    
    def pl(self, hypothesis=None):
        """
        Computes either the plausibility of 'hypothesis' or the entire plausibility function (hypothesis=None).
        
        If 'hypothesis' is None (default), a dictionary mapping hypotheses to their respective plausibility values is returned.
        Otherwise, the plausibility of 'hypothesis' is returned.
        In this case, 'hypothesis' is automatically converted to a 'frozenset' meaning its elements must be hashable.
        """
        if hypothesis is None:
            return {h:self.pl(h) for h in powerset(self.core())}
        else:
            hypothesis = MassFunction._convert(hypothesis)
            if not hypothesis:
                return 0.0
            else:
                return sum([v for h, v in self.items() if hypothesis & h])
    
    def q(self, hypothesis=None):
        """
        Computes either the commonality of 'hypothesis' or the entire commonality function (hypothesis=None).
        
        If 'hypothesis' is None (default), a dictionary mapping hypotheses to their respective commonality values is returned.
        Otherwise, the commonality of 'hypothesis' is returned.
        In this case, 'hypothesis' is automatically converted to a 'frozenset' meaning its elements must be hashable.
        """
        if hypothesis is None:
            return {h:self.q(h) for h in powerset(self.core())}
        else:
            hypothesis = MassFunction._convert(hypothesis)
            return sum([v for h, v in self.items() if h.issuperset(hypothesis)])
    
    def __and__(self, mass_function):
        """Shorthand for 'combine_conjunctive(mass_function)'."""
        return self.combine_conjunctive(mass_function)
    
    def __or__(self, mass_function):
        """Shorthand for 'combine_disjunctive(mass_function)'."""
        return self.combine_disjunctive(mass_function)
    
    def __str__(self):
        hyp = sorted([(v, h) for (h, v) in self.items()], reverse=True)
        return "{" + ";".join([str(tuple(h)) + ":" + str(v) for (v, h) in hyp]) + "}"
    
    def combine_conjunctive(self, mass_function, normalization=True, sample_count=None, importance_sampling=False):
        """
        Conjunctively combines the mass function with another mass function and returns the combination as a new mass function.
        
        The other mass function is assumed to be defined over the same frame of discernment.
        If 'mass_function' is not of type MassFunction, it is assumed to be an iterable containing multiple mass functions that are iteratively combined.
        
        If the mass functions are flatly contracting or if one of the mass functions is empty, an empty mass function is returned.
        
        'normalization' determines whether the resulting mass function is normalized (default is True).
         
        If 'sample_count' is not None, the true combination is approximated using the specified number of samples.
        In this case, 'importance_sampling' determines the method of approximation (only if normalization=True, otherwise 'importance_sampling' is ignored).
        The default method (importance_sampling=False) independently generates samples from both mass functions and computes their intersections.
        If importance_sampling=True, importance sampling is used to avoid empty intersections, which leads to a lower approximation error but is also slower.
        This method should be used if there is significant evidential conflict between the mass functions.
        """
        return self._combine(mass_function, lambda s1, s2: s1 & s2, normalization, sample_count, importance_sampling)
    
    def combine_disjunctive(self, mass_function, sample_count=None):
        """
        Disjunctively combines the mass function with another mass function and returns the combination as a new mass function.
        
        The other mass function is assumed to be defined over the same frame of discernment.
        If 'mass_function' is not of type MassFunction, it is assumed to be an iterable containing multiple mass functions that are iteratively combined.
        
        If 'sample_count' is not None, the true combination is approximated using the specified number of samples.
        """
        return self._combine(mass_function, lambda s1, s2: s1 | s2, False, sample_count, False)
    
    def _combine(self, mass_function, rule, normalization, sample_count, importance_sampling):
        """Helper method for combining two or more mass functions."""
        if not isinstance(mass_function, MassFunction):
            f = partial(MassFunction._combine, rule=rule, normalization=normalization, sample_count=sample_count, importance_sampling=importance_sampling)
            return reduce(f, [self] + mass_function)
        if not isinstance(mass_function, MassFunction):
            raise TypeError("expected type MassFunction")
        if not self or not mass_function:
            return MassFunction()
        if sample_count == None:
            m = self._combine_deterministic(mass_function, rule)
        else:
            if importance_sampling:
                m = self._combine_importance_sampling(mass_function, sample_count)
            else:
                m = self._combine_direct_sampling(mass_function, rule, sample_count)
        if normalization:
            return m.normalize()
        else:
            return m
    
    def _combine_deterministic(self, mass_function, rule):
        """Helper method for deterministically combining two mass functions."""
        combined = MassFunction()
        for h1, v1 in self.items():
            for h2, v2 in mass_function.items():
                h_new = rule(h1, h2)
                if h_new:
                    combined[h_new] += v1 * v2
        return combined
    
    def _combine_direct_sampling(self, mass_function, rule, sample_count):
        """Helper method for approximatively combining two mass functions using direct sampling."""
        combined = MassFunction()
        samples1 = self.sample(sample_count)
        samples2 = mass_function.sample(sample_count)
        for i in range(sample_count):
            s = rule(samples1[i], samples2[i])
            if s:
                combined[s] += 1.0 / sample_count
        return combined
    
    def _combine_importance_sampling(self, mass_function, sample_count):
        """Helper method for approximatively combining two mass functions using importance sampling."""
        combined = MassFunction()
        for s1, n in self.sample(sample_count, as_dict=True).items():
            weight = mass_function.pl(s1)
            for s2 in mass_function.condition(s1).sample(n):
                combined[s2] += weight
        return combined
    
    def combine_gbt(self, likelihoods, sample_count=None, importance_sampling=True):
        """
        Conjunctively combines this mass function with a mass function obtained from a list of likelihoods via the generalized Bayesian theorem.
        
        Arguments:
        mass_function -- A mass function defined over the same frame of discernment.
        sample_count -- The number of samples used for a Monte-Carlo combination. Monte-Carlo is used if sample_count is set to a value different than 'None'.
        sample_method -- The type of Monte-Carlo combination. (Ignored of sample_count is not set.)
            'direct' (default): Independently generate samples from both mass functions and intersect them. Appropriate if the evidential conflict is limited.
            'importance': Sample the second mass function by conditioning it with the samples of the first and use importance re-sampling.
            This method is slower but yields a better approximation of there is significant evidential conflict.
        
        TODO
        Returns:
        The normalized conjunctively combined mass function according to Dempster's combination rule.
        """
        combined = MassFunction()
        # restrict to generally compatible likelihoods
        frame = self.frame()
        likelihoods = [l for l in likelihoods if l[1] > 0 and l[0] in frame]
        if sample_count == None:    # deterministic
            return self.combine_conjunctive(MassFunction.gbt(likelihoods))
        else:   # Monte-Carlo
            for s, n in self.sample(sample_count, as_dict=True).items():
                if importance_sampling:
                    compatible_likelihoods = [l for l in likelihoods if l[0] in s]
                    weight = 1.0 - reduce(mul, [1.0 - l[1] for l in compatible_likelihoods], 1.0)
                else:
                    compatible_likelihoods = likelihoods
                if not compatible_likelihoods:
                    continue
                empty_mass = reduce(mul, [1.0 - l[1] for l in compatible_likelihoods], 1.0)
                for _ in range(n):
                    rv = [random() for _ in range(len(compatible_likelihoods))]
                    subtree_mass = 1.0
                    hyp = set()
                    for k in range(len(compatible_likelihoods)):
                        l = compatible_likelihoods[k][1]
                        norm = 1.0 if hyp else 1.0 - empty_mass / subtree_mass
                        if l / norm > rv[k]:
                            hyp.add(compatible_likelihoods[k][0])
                        else:
                            subtree_mass *= 1.0 - l   # only relevant for negative case
                    if importance_sampling:
                        combined[hyp] += weight
                    else:
                        hyp &= s
                        if hyp:
                            combined[hyp] += 1.0
            return combined.normalize()
    
    def condition(self, hypothesis, normalization=True):
        """
        Conditions the mass function with 'hypothesis' according to Dempster's rule of conditioning.
        
        'normalization' determines whether the resulting conjunctive combination is normalized.
        
        Shorthand for self.combine_conjunctive(MassFunction({hypothesis:1.0}), normalization).
        """
        return self.combine_conjunctive(MassFunction({hypothesis:1.0}), normalization)
    
    def conflict(self, mass_function):
        """
        Calculates the weight of conflict between two mass functions.
        
        The weight of conflict is computed as the (natural) logarithm of the normalization constant in Dempster's rule of combination.
        Returns infinity in case the mass functions are flatly contradicting.
        """ 
        c = 0.0
        for h1, v1 in self.items():
            for h2, v2 in mass_function.items():
                if not h1 & h2:
                    c += v1 * v2
        if c >= 1:
            return float("inf")
        else:
            return -log(1.0 - c, 2)
    
    def normalize(self):
        """
        Normalizes the mass function in-place such that the sum of all mass values equals 1.
        
        It does not set the mass value of the empty set to 0 and only asserts that the mass values sum to 1.
        For convenience, the method returns 'self'.
        """
        mass_sum = sum(self.values())
        if mass_sum != 1.0:
            for h, v in self.items():
                self[h] = v / mass_sum
        return self
    
    def markov_update(self, transition_model, sample_count=None):
        """
        Performs a first-order Markov prediction step using the given transition model.
        
        This mass function expresses the belief about the current state and the transition model describes the state transition belief.
        The transition model is a function that takes a singleton state as input and returns possible successor states either as a MassFunction
        or as a single randomly-sampled state set.
        """
        updated = MassFunction()
        if sample_count == None:
            # deterministic
            for k, v in self.items():
                predicted = None
                for e in k:
                    if predicted == None:
                        predicted = transition_model(e)
                    else:
                        predicted |= transition_model(e)
                for kp, vp in predicted.items():
                    updated[kp] += v * vp
        else:
            # Monte-Carlo
            for s, n in self.sample(sample_count, as_dict=True).items():
                unions = [[] for _ in range(n)]
                for e in s:
                    ts = transition_model(e, n)
                    for i, t in enumerate(ts):
                        unions[i].extend(t)
                for u in unions:
                    updated[u] += 1.0 / sample_count
        return updated
    
    def extend(self, spaces, index):
        """
        Extends the mass function vacuously to an additional dimension.
        
        
        """
        extended = MassFunction()
        for h, v in self.items():
            extended[frozenset(product(*(spaces[:index] + [h] + spaces[index:])))] = v
        return extended
    
    def project(self, dimensions):
        """
        Projects a mass function defined over a multi-dimensional frame of discernment to a subset of these dimensions.
        
        Existing hypotheses are assumed to be sets of tuples, where each tuple is located in the multi-dimensional space.
        'dimensions' is a set of indices determining the dimensions to be preserved.
        """
        projected = MassFunction()
        for h, v in self.items():
            projected[[s[d] for s in h for d in dimensions]] += v
        return projected
    
    def pignistic(self):
        """Computes the pignistic transformation and returns a mass function consisting only of singletons."""
        p = MassFunction()
        for h, v in self.items():
            for s in h:
                p[(s,)] += v / len(h)
        return p
    
    def local_conflict(self):
        """
        Computes the local conflict measure.
        
        See 
        """
        c = 0.0
        for h, v in self.items():
            c += v * log(len(h) / v, 2)
        return c
    
    def distance(self, m):
        """Evidential distance between two mass functions according to Jousselme et al. "A new distance between two bodies of evidence". Information Fusion, 2001."""
        def sp(m1, m2, cache):
            p = 0
            for h1, v1 in m1.items():
                for h2, v2 in m2.items():
                    if (h1, h2) in cache:
                        p += cache[(h1, h2)] * v1 * v2
                    else:
                        w = len(h1 & h2)
                        if w > 0:
                            w /= float(len(h1 | h2))
                            p += w * v1 * v2
                        cache[(h1, h2)] = w
                        cache[(h2, h1)] = w
            return p
        cache = {}
        return sqrt(0.5 * (sp(self, self, cache) + sp(m, m, cache)) - sp(self, m, cache))
    
    def norm(self, m, p=2):
        """
        Computes the p-norm between two mass functions.
        
        Both mass functions are treated as vectors of mass values.
        """
        d = sum([(v - m[h])**p for h, v in self.items()])
        for h, v in m.items():
            if h not in self:
                d += v**p
        return d**(1.0 / p)
    
    def prune(self, max_mass=0.0):
        """
        Removes all hypotheses whose mass value is less than or equal to 'max_mass'.
        
        By default (max_mass=0.0), only non-focal hypotheses are removed.
        Afterwards, the mass function is normalized.
        For convenience, the method returns 'self'.
        """ 
        remove = [h for h, v in self.items() if v <= max_mass]
        for h in remove:
            del self[h]
        return self.normalize()
    
    def is_normalized(self, epsilon=0.0):
        """
        Checks whether the mass values sum to 1.
        
        'epsilon' specifies the tolerance interval and defaults to 0.
        Note that this method only checks the total amount of assigned mass including the mass corresponding to the empty set.
        """
        return abs(sum(self.values()) - 1.0) <= epsilon
    
    def is_compatible(self, m):
        """
        Checks whether another mass function is compatible with this one.
        
        Compatibility means that the mass value of each hypothesis in 'm' is less than
        or equal to the corresponding plausibility given by this mass function.
        """
        return all([self.pl(h) >= v for h, v in m.items()])
    
    def sample(self, n, maximum_likelihood=True, as_dict=False):
        """
        TODO update
        Generates a list of n samples from this distribution.
        
        Arguments:
        n -- The number of samples.
        as_dict -- Return the samples as a list (False) or as a dictionary of samples and their frequencies (True). 
        
        Returns:
        Either a list of hypothesis drawn (with replacement) from the distribution with probability proportional to their mass values
        or a dictionary containing the drawn hypotheses and their frequencies.
        """
        if not isinstance(n, int):
            raise TypeError("n must be int")
        samples = dict() if as_dict else []
        mass_sum = sum(self.values())
        if maximum_likelihood:
            remainders = []
            remaining_sample_count = n
            for h, v in self.items():
                fraction = n * v / mass_sum
                quotient = int(fraction)
                if quotient > 0:
                    if as_dict:
                        samples[h] = quotient
                    else:
                        samples.extend([h] * quotient)
                remainders.append((h, fraction - quotient))
                remaining_sample_count -= quotient
            remainders.sort(reverse=True, key=lambda hv: hv[1])
            for h, _ in remainders[:remaining_sample_count]:
                if as_dict:
                    if h in samples:
                        samples[h] += 1
                    else:
                        samples[h] = 1
                else:
                    samples.append(h)
        else:
            rv = [uniform(0.0, mass_sum) for _ in range(n)]
            hypotheses = sorted(self.items(), reverse=True, key=lambda hv: hv[1])
            for i in range(n):
                mass = 0.0
                for h, v in hypotheses:
                    mass += v
                    if mass >= rv[i]:
                        if as_dict:
                            if h in samples:
                                samples[h] += 1
                            else:
                                samples[h] = 1
                        else:
                            samples.append(h)
                        break
        if not as_dict:
            shuffle(samples)
        return samples
    
    def is_probabilistic(self):
        """
        Checks whether the mass function is a probability function.
        
        Returns true if and only if all hypotheses are singletons (normalization is ignored). 
        """
        return all([len(h) == 1 for h in self.keys()])
    
    def sample_probability_distributions(self, n):
        samples = [MassFunction() for _ in range(n)]
        for i in range(n):
            for h, v in self.items():
                if len(h) == 1:
                    samples[i][h] += v
                else:
                    rv = [random() for _ in range(len(h))]
                    total = sum(rv)
                    for k, s in enumerate(h):
                        samples[i][{s}] += rv[k] * v / total
        return samples


def powerset(set, include_empty=True):
    """
    Returns an iterator over the power set of 'set'.
    
    'set' is an arbitrary iterator over hashable elements.
    All returned subsets are of type 'frozenset'.
    If include_empty is set to False, the empty set will not be returned.
    """
    start = 0 if include_empty else 1
    return map(frozenset, chain.from_iterable(combinations(set, r) for r in range(start, len(set) + 1)))

def gbt_m(hypothesis, likelihoods, normalization=True):
    """
    Computes the mass value of 'hypothesis' using the generalized Bayesian theorem.
    
    Equivalent to MassFunction.gbt(likelihoods, normalization)[hypothesis].
    """
    q = gbt_q(hypothesis, likelihoods, normalization)
    return q * reduce(mul, [1.0 - l[1] for l in likelihoods if l[0] not in hypothesis], 1.0)

def gbt_bel(hypothesis, likelihoods, normalization=True):
    """
    Computes the belief of 'hypothesis' using the generalized Bayesian theorem.
    
    Equivalent to MassFunction.gbt(likelihoods, normalization).bel(hypothesis).
    """
    eta = _gbt_normalization(likelihoods) if normalization else 1.0
    exc = reduce(mul, [1.0 - l[1] for l in likelihoods if l[0] not in hypothesis], 1.0)
    all = reduce(mul, [1.0 - l[1] for l in likelihoods], 1.0)
    return eta * (exc - all)

def gbt_pl(hypothesis, likelihoods, normalization=True):
    """
    Computes the plausibility of 'hypothesis' using the generalized Bayesian theorem.
    
    Equivalent to MassFunction.gbt(likelihoods, normalization).pl(hypothesis).
    """
    eta = _gbt_normalization(likelihoods) if normalization else 1.0
    return eta * (1.0 - reduce(mul, [1.0 - l[1] for l in likelihoods if l[0] in hypothesis], 1.0))
    
def gbt_q(hypothesis, likelihoods, normalization=True):
    """
    Computes the commonality of 'hypothesis' using the generalized Bayesian theorem.
    
    Equivalent to MassFunction.gbt(likelihoods, normalization).q(hypothesis).
    """
    eta = _gbt_normalization(likelihoods) if normalization else 1.0
    return eta * reduce(mul, [l[1] for l in likelihoods if l[0] in hypothesis], 1.0)

def _gbt_normalization(likelihoods):
    """Helper function for computing the GBT normalization constant."""
    return 1.0 / (1.0 - reduce(mul, [1.0 - l[1] for l in likelihoods], 1.0))
