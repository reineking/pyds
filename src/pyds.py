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

from itertools import chain, combinations
from functools import reduce
from operator import mul
from math import log, fsum
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
            for (h, v) in source:
                self[h] += v
    
    @staticmethod
    def _convert(hypothesis):
        """Convert hypothesis to a 'frozenset' in order to make it hashable."""
        if isinstance(hypothesis, frozenset):
            return hypothesis
        else:
            return frozenset(hypothesis)
    
    @staticmethod
    def gbt(likelihoods, *, normalization=True, sample_count=None):
        """
        Constructs a mass function using the generalized Bayesian theorem.
        For more information, see Smets. 1993. Belief functions: 
        The disjunctive rule of combination and the generalized Bayesian theorem. International Journal of Approximate Reasoning. 
        
        'likelihoods' specifies the conditional plausibilities for a set of singleton hypotheses.
        It can either be a dictionary mapping singleton hypotheses to plausibilities or an iterable
        containing tuples consisting of a singleton hypothesis and a corresponding plausibility value.
        
        'normalization' determines whether the resulting mass function is normalized, i.e., whether m({}) == 0.
        
        If 'sample_count' is not None, the true mass function is approximated using the specified number of samples.
        """
        m = MassFunction()
        if isinstance(likelihoods, dict):
            likelihoods = list(likelihoods.items())
        # filter trivial likelihoods 0 and 1
        ones = [h for (h, l) in likelihoods if l >= 1.0]
        likelihoods = [(h, l) for (h, l) in likelihoods if 0.0 < l < 1.0]
        if sample_count == None:   # deterministic
            def traverse(m, likelihoods, ones, index, hyp, mass):
                if index == len(likelihoods):
                    m[hyp + ones] = mass
                else:
                    traverse(m, likelihoods, ones, index + 1, hyp + [likelihoods[index][0]], mass * likelihoods[index][1])
                    traverse(m, likelihoods, ones, index + 1, hyp, mass * (1.0 - likelihoods[index][1]))
            traverse(m, likelihoods, ones, 0, [], 1.0)
            if normalization:
                m.normalize()
        else:   # Monte-Carlo
            if normalization:
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
            v = fsum([bel[h2] * (-1)**(len(h1 - h2)) for h2 in powerset(h1)])
            if v > 0:
                m[h1] = v
        mass_sum = fsum(m.values())
        if mass_sum < 1.0:
            m[frozenset()] = 1.0 - mass_sum
        return m
    
    @staticmethod
    def from_pl(pl):
        """
        Creates a mass function from a corresponding plausibility function.
        
        'pl' is a dictionary mapping hypotheses to plausibility values like to one returned by 'pl()'.
        """
        frame = max(pl.keys(), key=len)
        bel_theta = pl[frame]
        bel = {frozenset(frame - h):bel_theta - v for (h, v) in pl.items()} # follows from bel(-A) = bel(frame) - pl(A)
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
            v = fsum([q[h1 | h2] * (-1)**(len(h2 - h1)) for h2 in powerset(frame - h1)])
            if v > 0:
                m[h1] = v
        mass_sum = fsum(m.values())
        if mass_sum < 1.0:
            m[frozenset()] = 1.0 - mass_sum
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
        return {h for (h, v) in self.items() if v > 0}
    
    def core(self, *mass_functions):
        """
        Returns the core of one or more mass functions as a 'frozenset'.
        
        The core of a single mass function is the union of all its focal hypotheses.
        In case a mass function does not contain any focal hypotheses, its core is an empty set.
        If multiple mass functions are given, their combined core (intersection of all single cores) is returned.
        """
        if mass_functions:
            return frozenset.intersection(self.core(), *[m.core() for m in mass_functions])
        else:
            focal = self.focal()
            if not focal:
                return frozenset()
            else:
                return frozenset.union(*focal)
    
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
                return fsum([v for (h, v) in self.items() if h and hypothesis.issuperset(h)])
    
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
                return fsum([v for (h, v) in self.items() if hypothesis & h])
    
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
            if not hypothesis:
                return 0.0
            else:
                return fsum([v for (h, v) in self.items() if h.issuperset(hypothesis)])
    
    def __and__(self, mass_function):
        """Shorthand for 'combine_conjunctive(mass_function)'."""
        return self.combine_conjunctive(mass_function)
    
    def __or__(self, mass_function):
        """Shorthand for 'combine_disjunctive(mass_function)'."""
        return self.combine_disjunctive(mass_function)
    
    def __str__(self):
        hyp = sorted([(v, h) for (h, v) in self.items()], reverse=True)
        return "{" + "; ".join([str(set(h)) + ":" + str(v) for (v, h) in hyp]) + "}"
    
    def combine_conjunctive(self, *mass_functions, normalization=True, sample_count=None, importance_sampling=False):
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
        return self._combine(*mass_functions, rule=lambda s1, s2: s1 & s2, normalization=normalization, sample_count=sample_count, importance_sampling=importance_sampling)
    
    def combine_disjunctive(self, *mass_functions, sample_count=None):
        """
        Disjunctively combines the mass function with another mass function and returns the combination as a new mass function.
        
        The other mass function is assumed to be defined over the same frame of discernment.
        If 'mass_function' is not of type MassFunction, it is assumed to be an iterable containing multiple mass functions that are iteratively combined.
        
        If 'sample_count' is not None, the true combination is approximated using the specified number of samples.
        """
        return self._combine(*mass_functions, rule=lambda s1, s2: s1 | s2, normalization=False, sample_count=sample_count, importance_sampling=False)
    
    def _combine(self, *mass_functions, rule, normalization, sample_count, importance_sampling):
        """Helper method for combining two or more mass functions."""
        combined = self
        for m in mass_functions:
            if not isinstance(m, MassFunction):
                raise TypeError("expected type MassFunction but got %s; make sure to use keyword arguments for anything other than mass functions" % type(m))
            if sample_count == None:
                combined = combined._combine_deterministic(m, rule)
            else:
                if importance_sampling and normalization:
                    combined = combined._combine_importance_sampling(m, sample_count)
                else:
                    combined = combined._combine_direct_sampling(m, rule, sample_count)
        if normalization:
            return combined.normalize()
        else:
            return combined
    
    def _combine_deterministic(self, mass_function, rule):
        """Helper method for deterministically combining two mass functions."""
        combined = MassFunction()
        for (h1, v1) in self.items():
            for (h2, v2) in mass_function.items():
                combined[rule(h1, h2)] += v1 * v2
        return combined
    
    def _combine_direct_sampling(self, mass_function, rule, sample_count):
        """Helper method for approximatively combining two mass functions using direct sampling."""
        combined = MassFunction()
        samples1 = self.sample(sample_count)
        samples2 = mass_function.sample(sample_count)
        for i in range(sample_count):
            combined[rule(samples1[i], samples2[i])] += 1.0 / sample_count
        return combined
    
    def _combine_importance_sampling(self, mass_function, sample_count):
        """Helper method for approximatively combining two mass functions using importance sampling."""
        combined = MassFunction()
        for (s1, n) in self.sample(sample_count, as_dict=True).items():
            weight = mass_function.pl(s1)
            for s2 in mass_function.condition(s1).sample(n):
                combined[s2] += weight
        return combined
    
    def combine_gbt(self, likelihoods, *, normalization=True, sample_count=None, importance_sampling=True):
        """
        Conjunctively combines the mass function with a mass function obtained from a sequence of
        likelihoods via the generalized Bayesian theorem and returns the combination as a new mass function.
        
        Equivalent to 'combine_conjunctive(MassFunction.gbt(likelihoods))'.
        By ignoring incompatible likelihoods, it is generally faster than the former
        method and yields a better Monte-Carlo approximation in case of normalization.
        
        'likelihoods' specifies the conditional plausibilities for a set of singleton hypotheses.
        It can either be a dictionary mapping singleton hypotheses to plausibilities or an iterable
        containing tuples consisting of a singleton hypothesis and a corresponding plausibility value.
        
        All arguments except for 'likelihoods' must be specified as keyword arguments.
        'normalization' determines whether the resulting mass function is normalized, i.e., whether m({}) == 0.
        If 'sample_count' is not None, the true mass function is approximated using the specified number of samples.
        See 'combine_conjunctive' for details on the effect of setting 'importance_sampling'.
        """
        core = self.core() # restrict to generally compatible likelihoods
        if isinstance(likelihoods, dict):
            likelihoods = list(likelihoods.items())
        likelihoods = [l for l in likelihoods if l[1] > 0 and l[0] in core]
        if sample_count == None: # deterministic
            return self.combine_conjunctive(MassFunction.gbt(likelihoods), normalization=normalization)
        else: # Monte-Carlo
            if not normalization: # only use importance sampling in case of normalization
                importance_sampling = False
            combined = MassFunction()
            for s, n in self.sample(sample_count, as_dict=True).items():
                if importance_sampling:
                    compatible_likelihoods = [l for l in likelihoods if l[0] in s]
                    weight = 1.0 - reduce(mul, [1.0 - l[1] for l in compatible_likelihoods], 1.0)
                else:
                    compatible_likelihoods = likelihoods
                if not compatible_likelihoods:
                    continue
                if normalization:
                    empty_mass = reduce(mul, [1.0 - l[1] for l in compatible_likelihoods], 1.0)
                for _ in range(n):
                    rv = [random() for _ in range(len(compatible_likelihoods))]
                    subtree_mass = 1.0
                    hyp = set()
                    for k in range(len(compatible_likelihoods)):
                        l = compatible_likelihoods[k][1]
                        norm = 1.0 if hyp or not normalization else 1.0 - empty_mass / subtree_mass
                        if l / norm > rv[k]:
                            hyp.add(compatible_likelihoods[k][0])
                        else:
                            subtree_mass *= 1.0 - l   # only relevant for negative case
                    if importance_sampling:
                        combined[hyp] += weight
                    else:
                        combined[hyp & s] += 1.0
            if normalization:
                return combined.normalize()
            else:
                return combined
    
    def condition(self, hypothesis, *, normalization=True):
        """
        Conditions the mass function with 'hypothesis'.
        
        'normalization' determines whether the resulting conjunctive combination is normalized (must be specified as a keyword argument).
        
        Shorthand for self.combine_conjunctive(MassFunction({hypothesis:1.0}), normalization).
        """
        m = MassFunction({MassFunction._convert(hypothesis):1.0})
        return self.combine_conjunctive(m, normalization=normalization)
    
    def conflict(self, *mass_functions, sample_count=None):
        """
        Calculates the weight of conflict between two or more mass functions.
        
        The weight of conflict is computed as the (natural) logarithm of the normalization constant in Dempster's rule of combination.
        Returns infinity in case the mass functions are flatly contradicting.
        """
        # compute full conjunctive combination (could be more efficient)
        c = self.combine_conjunctive(*mass_functions, normalization=False, sample_count=sample_count)[frozenset()]
        if c >= 1.0:
            return float("inf")
        else:
            return -log(1.0 - c, 2)
    
    def normalize(self):
        """
        Normalizes the mass function in-place.
        
        Sets the mass value of the empty set to 0 and scales all other values such that their sum equals 1.
        For convenience, the method returns 'self'.
        """
        if frozenset() in self:
            del self[frozenset()]
        mass_sum = fsum(self.values())
        if mass_sum != 1.0:
            for (h, v) in self.items():
                self[h] = v / mass_sum
        return self
    
    def prune(self):
        """
        Removes all non-focal (0 mass) hypotheses in-place.
        
        For convenience, the method returns 'self'.
        """ 
        remove = [h for (h, v) in self.items() if v == 0.0]
        for h in remove:
            del self[h]
        return self
    
    def markov(self, transition_model, *, sample_count=None):
        """
        Computes the mass function induced by a prior belief (self) and a transition model.
        
        The transition model expresses a joint belief over the frame of this mass function and a new frame.
        The belief over the frame of this mass function is implicitly assumed to be vacuous.
        The transition model is a function returning the conditional belief over the new frame (as a mass function
        if sample_count=None) while taking a singleton hypothesis of the current frame as input.
        The disjunctive rule of combination is then used to construct the mass function over the new frame.
        
        If 'sample_count' is not None, the true mass function is approximated using the specified number of samples.
        In this case, 'transition_model' is expected to take a second argument stating how many samples from the corresponding conditional mass function should be returned.
        The return value in this case is expected to be an iterable over sampled hypotheses from the new frame.
         
        This method can be used to implement the prediction step for estimation in a hidden Markov process (hence the name).
        Under this interpretation, the transition model expresses the mass distribution over successor states given the current state.
        """
        updated = MassFunction()
        if sample_count == None: # deterministic
            for k, v in self.items():
                predicted = None
                for e in k:
                    if predicted == None:
                        predicted = transition_model(e)
                    else:
                        predicted |= transition_model(e)
                for kp, vp in predicted.items():
                    updated[kp] += v * vp
        else: # Monte-Carlo
            for s, n in self.sample(sample_count, as_dict=True).items():
                unions = [[] for _ in range(n)]
                for e in s:
                    ts = transition_model(e, n)
                    for i, t in enumerate(ts):
                        unions[i].extend(t)
                for u in unions:
                    updated[u] += 1.0 / sample_count
        return updated
    
    def map(self, function):
        """
        Maps each hypothesis to a new hypothesis using 'function' and returns the new mass function.
        
        'function' is a function taking a hypothesis as its only input and returning a new hypothesis
        (i.e., a sequence that can be converted to a 'frozenset').
        
        Here are some example use cases:
        
        1. Vacuous extension to a multi-dimensional frame of discernment (m is defined over
        the frame A while the new mass function is defined over the Cartesian product AxB):
            
            B = {'x', 'y', 'z'}
            m.map(lambda h: itertools.product(h, B))
        
        2. Projection to a lower dimensional frame (m is defined over AxBxC such that each hypothesis is
        a set of tuples where each tuple consists of 3 elements; the new mass function is defined over BxC):
        
            m.map(lambda h: (t[1:] for t in h))
        """
        m = MassFunction()
        for (h, v) in self.items():
            m[self._convert(function(h))] += v
        return m
    
    def pignistic(self):
        """Computes the pignistic transformation and returns it as a new mass function consisting only of singletons."""
        p = MassFunction()
        for (h, v) in self.items():
            if v > 0.0:
                size = float(len(h))
                for s in h:
                    p[(s,)] += v / size
        return p.normalize()
    
    def local_conflict(self):
        """
        Computes the local conflict measure.
        
        For more information, see Pal et al. 1993. Uncertainty measures for evidential reasoning II:
        A new measure of total uncertainty. International Journal of Approximate Reasoning.
        
        Only works for normalized mass functions.
        If the mass function is unnormalized, the method returns float('nan')
        
        In case the mass function is a probability function (containing only singleton hypotheses),
        it reduces to the classical entropy measure.
        """
        if self[frozenset()] > 0.0:
            return float('nan')
        c = 0.0
        for (h, v) in self.items():
            if v > 0.0:
                c += v * log(len(h) / v, 2)
        return c
    
    def norm(self, m, p=2):
        """
        Computes the p-norm between two mass functions (default is p=2).
        
        Both mass functions are treated as vectors of mass values.
        """
        d = fsum([(v - m[h])**p for (h, v) in self.items()])
        for (h, v) in m.items():
            if h not in self:
                d += v**p
        return d**(1.0 / p)
    
    def is_compatible(self, m):
        """
        Checks whether another mass function is compatible with this one.
        
        Compatibility means that the mass value of each hypothesis in 'm' is less than
        or equal to the corresponding plausibility given by this mass function.
        """
        return all([self.pl(h) >= v for (h, v) in m.items()])
    
    def sample(self, n, *, quantization=True, as_dict=False):
        """
        Returns n random samples from the mass distribution.
        
        Hypotheses are drawn with a probability proportional to their mass values (with replacement).
         
        If 'quantization' is True (default), the method performs a quantization of the mass values.
        This means the frequency of a hypothesis h in the sample set is at least int(self[h] * n / t) where t is the sum of all mass values.
        The remaining sample slots (if any) are filled up according to the remainders of the fractions computed in the first step.
        
        The parameter 'as_dict' determines the type of the returned value.
        If 'as_dict' is False (default), a list of length n is returned.
        Otherwise, the result is a dictionary specifying the number of samples for each hypothesis.
        """
        if not isinstance(n, int):
            raise TypeError("n must be int")
        samples = dict() if as_dict else []
        mass_sum = fsum(self.values())
        if quantization:
            remainders = []
            remaining_sample_count = n
            for (h, v) in self.items():
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
                for (h, v) in hypotheses:
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
        
        Returns True if and only if all hypotheses are singletons (normalization is ignored). 
        """
        return all([len(h) == 1 for h in self.keys()])
    
    def sample_probability_distributions(self, n):
        """
        Randomly generates n compatible probability distributions from the mass function.
        
        The result is a list of n independently sampled probability distributions expressed as mass functions.
        This can be useful for estimating various statistical measures like the minimum or maximum entropy consistent with the mass distribution.
        """
        samples = [MassFunction() for _ in range(n)]
        for i in range(n):
            for (h, v) in self.items():
                if len(h) == 1:
                    samples[i][h] += v
                else:
                    rv = [random() for _ in range(len(h))]
                    total = fsum(rv)
                    for k, s in enumerate(h):
                        samples[i][{s}] += rv[k] * v / total
        return samples


def powerset(set):
    """
    Returns an iterator over the power set of 'set'.
    
    'set' is an arbitrary iterator over hashable elements.
    All returned subsets are of type 'frozenset'.
    """
    return map(frozenset, chain.from_iterable(combinations(set, r) for r in range(len(set) + 1)))

def gbt_m(hypothesis, likelihoods, normalization=True):
    """
    Computes the mass value of 'hypothesis' using the generalized Bayesian theorem.
    
    Equivalent to MassFunction.gbt(likelihoods, normalization)[hypothesis].
    """
    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    q = gbt_q(hypothesis, likelihoods, normalization)
    return q * reduce(mul, [1.0 - l[1] for l in likelihoods if l[0] not in hypothesis], 1.0)

def gbt_bel(hypothesis, likelihoods, normalization=True):
    """
    Computes the belief of 'hypothesis' using the generalized Bayesian theorem.
    
    Equivalent to MassFunction.gbt(likelihoods, normalization).bel(hypothesis).
    """
    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    eta = _gbt_normalization(likelihoods) if normalization else 1.0
    exc = reduce(mul, [1.0 - l[1] for l in likelihoods if l[0] not in hypothesis], 1.0)
    all = reduce(mul, [1.0 - l[1] for l in likelihoods], 1.0)
    return eta * (exc - all)

def gbt_pl(hypothesis, likelihoods, normalization=True):
    """
    Computes the plausibility of 'hypothesis' using the generalized Bayesian theorem.
    
    Equivalent to MassFunction.gbt(likelihoods, normalization).pl(hypothesis).
    """
    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    eta = _gbt_normalization(likelihoods) if normalization else 1.0
    return eta * (1.0 - reduce(mul, [1.0 - l[1] for l in likelihoods if l[0] in hypothesis], 1.0))
    
def gbt_q(hypothesis, likelihoods, normalization=True):
    """
    Computes the commonality of 'hypothesis' using the generalized Bayesian theorem.
    
    Equivalent to MassFunction.gbt(likelihoods, normalization).q(hypothesis).
    """
    if isinstance(likelihoods, dict):
        likelihoods = list(likelihoods.items())
    eta = _gbt_normalization(likelihoods) if normalization else 1.0
    return eta * reduce(mul, [l[1] for l in likelihoods if l[0] in hypothesis], 1.0)

def _gbt_normalization(likelihoods):
    """Helper function for computing the GBT normalization constant."""
    return 1.0 / (1.0 - reduce(mul, [1.0 - l[1] for l in likelihoods], 1.0))
