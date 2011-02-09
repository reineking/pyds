'''
Created on Nov 27, 2009

@author: reineking
'''

from itertools import product
import operator
from math import log, sqrt
from random import Random
from numpy.random import RandomState # much faster than standard random when generating many random numbers


class MassFunction(dict):
    """
    A Dempster-Shafer mass function (basic probability assignment) based on a dictionary.
    
    TODO normalization
    Hypothesis and their associated mass values can be added/changed/removed using the standard dictionary methods.
    Each hypothesis is represented as a frozenset meaning its elements must be hashable.
    """
    
    def __init__(self, source = None):
        """
        Create a new mass function.
        
        'source' is an iterable containing tuples of hypothesis and their mass values. 
        """
        if source != None:
            for h, v in source:
                self[h] += v
    
    @staticmethod
    def _convert(hypothesis):
        """Converts a hypothesis to a frozenset."""
        if isinstance(hypothesis, frozenset):
            return hypothesis
        else:
            return frozenset(hypothesis)
    
    @staticmethod
    def gbt(likelihoods, sample_count = None, seed = None):
        """
        Constructs a mass function from a list of likelihoods (plausibilities) using the generalized Bayesian theorem.
        
        'likelihoods': list of singleton-plausibility tuples
        """
        if not isinstance(likelihoods, list):
            raise TypeError("expected a list")
        m = MassFunction()
        # filter trivial likelihoods (0, 1)
        ones = [h for (h, l) in likelihoods if l >= 1.0]
        likelihoods = [(h, l) for (h, l) in likelihoods if 0.0 < l < 1.0]
        if sample_count == None:   # deterministic
            def traverse(m, likelihoods, ones, index, hyp, mass):
                if index == len(likelihoods):
                    hyp += ones
                    if len(hyp) > 0:
                        m[hyp] = mass
                else:
                    traverse(m, likelihoods, ones, index + 1, hyp + [likelihoods[index][0]], mass * likelihoods[index][1])
                    traverse(m, likelihoods, ones, index + 1, hyp, mass * (1.0 - likelihoods[index][1]))
            traverse(m, likelihoods, ones, 0, [], 1.0)
            m.normalize()
        else:   # Monte-Carlo
            empty_mass = reduce(operator.mul, [1.0 - l[1] for l in likelihoods], 1.0)
            rs = RandomState(seed)
            for _ in xrange(sample_count):
                rv = rs.random_sample(len(likelihoods))
                subtree_mass = 1.0
                hyp = set(ones)
                for k in xrange(len(likelihoods)):
                    l = likelihoods[k][1]
                    p_t = l * subtree_mass
                    p_f = (1.0 - l) * subtree_mass
                    if not hyp:
                        p_f -= empty_mass
                    if p_t > rv[k] * (p_t + p_f):
                        hyp.add(likelihoods[k][0])
                    else:
                        subtree_mass *= 1 - l # only relevant for negative case
                m[hyp] += 1.0 / sample_count
        return m
    
    def __missing__(self, key):
        """Return 0 mass for hypotheses that are not contained."""
        return 0.0
    
    def __copy__(self):
        c = MassFunction()
        for k, v in self.iteritems():
            c[k] = v
        return c
    
    def copy(self):
        """Creates a copy of the mass function."""
        return self.__copy__()
    
    def __contains__(self, hypothesis):
        return dict.__contains__(self, MassFunction._convert(hypothesis))
    
    def __getitem__(self, hypothesis):
        return dict.__getitem__(self, MassFunction._convert(hypothesis))
    
    def __setitem__(self, hypothesis, value):
        """
        Adds or updates the mass value of a hypothesis.
        
        'hypothesis' is an iterable whose elements must be hashable.
        Empty hypotheses and negative mass values both lead to exceptions.
        """
        if len(hypothesis) == 0:
            raise Exception("hypothesis is empty")
        value = float(value)
        if value < 0.0:
            raise Exception("mass value is negative")
#        if value == 0 and hypothesis in self:
#            del self[hypothesis]
        dict.__setitem__(self, MassFunction._convert(hypothesis), value)
    
    def __delitem__(self, hypothesis):
        return dict.__delitem__(self, MassFunction._convert(hypothesis))
    
    def frame_of_discernment(self):
        """Returns the frame of discernment (union of all contained hypotheses)."""
        if not self:
            return frozenset()
        else:
            return frozenset.union(*self.keys())
    
    def bel(self, hypothesis):
        """
        Computes the belief of 'hypothesis'.
        
        'hypothesis' is an iterable whose elements must be hashable.
        """
        return self._compute(hypothesis, lambda a, b: a.issuperset(b))
    
    def pl(self, hypothesis):
        """
        Computes the plausibility of 'hypothesis'.
        
        'hypothesis' is an iterable whose elements must be hashable.
        """
        return self._compute(hypothesis, lambda a, b: a & b)
    
    def q(self, hypothesis):
        """
        Computes the commonality of 'hypothesis'.
        
        'hypothesis' is an iterable whose elements must be hashable.
        """
        return self._compute(hypothesis, lambda a, b: b.issuperset(a))
    
    def _compute(self, hypothesis, criterion):
        hypothesis = MassFunction._convert(hypothesis)
        c = 0.0
        for h, v in self.iteritems():
            if criterion(hypothesis, h):
                c += v
        return c
    
    def __and__(self, mass_function):
        """Shorthand for 'combine_conjunctive'."""
        return self.combine_conjunctive(mass_function)
    
    def __or__(self, mass_function):
        """Shorthand for 'combine_disjunctive'."""
        return self.combine_disjunctive(mass_function)
    
    def __str__(self):
        hyp = sorted([(v, h) for (h, v) in self.iteritems()], reverse=True)
        return '{' + ';'.join([str(tuple(h)) + ':' + str(v) for (v, h) in hyp]) + '}'
    
    def combine_conjunctive(self, mass_function, sample_count = None, seed = None, sampling_method = "direct"):
        """
        Conjunctively combines this mass function with another mass function.
        
        Arguments:
        mass_function -- A mass function defined over the same frame of discernment.
        sample_count -- The number of samples used for a Monte-Carlo combination. Monte-Carlo is used if sample_count is set to a value different than 'None'.
        seed -- The random seed used during a Monte-Carlo combination. (Ignored of sample_count is not set.)
        sample_method -- The type of Monte-Carlo combination. (Ignored of sample_count is not set.)
            'direct' (default): Independently generate samples from both mass functions and intersect them. Appropriate if the evidential conflict is limited.
            'importance': Sample the second mass function by conditioning it with the samples of the first and use importance re-sampling.
            This method is slower but yields a better approximation of there is significant evidential conflict.
        
        Returns:
        The normalized conjunctively combined mass function according to Dempster's combination rule.
        """
        return self._combine(mass_function, lambda s1, s2: s1 & s2, sample_count, seed, sampling_method)
    
    def combine_disjunctive(self, mass_function, sample_count = None, seed = None):
        """
        Disjunctively combines this mass function with another mass function.
        
        Arguments:
        mass_function -- A mass function defined over the same frame of discernment.
        sample_count -- The number of samples used for a Monte-Carlo combination. Monte-Carlo is used if sample_count is set to a value different than 'None'.
        seed -- The random seed used during a Monte-Carlo combination. (Ignored of sample_count is not set.)
        
        Returns:
        The disjunctively combined mass function.
        """
        return self._combine(mass_function, lambda s1, s2: s1 | s2, sample_count, seed, "direct")
    
    def _combine(self, mass_function, rule, sample_count, seed, sampling_method):
        if isinstance(mass_function, list):
            combined = self
            for m in mass_function:
                combined = combined._combine(m, rule, sample_count, seed, sampling_method)
            return combined
        if not isinstance(mass_function, MassFunction):
            raise Exception("mass_function is not a MassFunction")
        if len(self) == 0 or len(mass_function) == 0:
            return MassFunction()
        if sample_count == None:
            return self._combine_deterministic(mass_function, rule)
        else:
            if sampling_method == "direct":
                return self._combine_direct_sampling(mass_function, rule, sample_count, seed)
            elif sampling_method == "importance":
                return self._combine_importance_sampling(mass_function, sample_count, seed)
            else:
                raise Exception("unknown sampling method")
    
    def _combine_deterministic(self, mass_function, rule):
        combined = MassFunction()
        for h1, v1 in self.iteritems():
            for h2, v2 in mass_function.iteritems():
                h_new = rule(h1, h2)
                if h_new:
                    combined[h_new] += v1 * v2
        return combined.normalize()
    
    def _combine_direct_sampling(self, mass_function, rule, sample_count, seed):
        combined = MassFunction()
        samples1 = self.sample(sample_count, seed=seed)
        samples2 = mass_function.sample(sample_count, seed=seed + 1 if seed != None else None)
        for i in xrange(sample_count):
            s = rule(samples1[i], samples2[i])
            if s:
                combined[s] += 1.0 / sample_count
        return combined.normalize()
    
    def _combine_importance_sampling(self, mass_function, sample_count, seed):
        combined = MassFunction()
        for s1, n in self.sample(sample_count, seed=seed, as_dict=True).iteritems():
            weight = mass_function.pl(s1)
            if seed != None:
                seed += 1
            for s2 in mass_function.condition(s1).sample(n, seed=seed):
                combined[s2] += weight
        return combined.normalize()
    
    def combine_gbt(self, likelihoods, sample_count = None, importance_sampling = True, seed = None):
        """
        Conjunctively combines this mass function with a mass function obtained from a list of likelihoods via the generalized Bayesian theorem.
        
        Arguments:
        mass_function -- A mass function defined over the same frame of discernment.
        sample_count -- The number of samples used for a Monte-Carlo combination. Monte-Carlo is used if sample_count is set to a value different than 'None'.
        seed -- The random seed used during a Monte-Carlo combination. (Ignored of sample_count is not set.)
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
        frame = self.frame_of_discernment()
        likelihoods = [l for l in likelihoods if l[1] > 0 and l[0] in frame]
        if sample_count == None:    # deterministic
            return self.combine_conjunctive(MassFunction.gbt(likelihoods))
        else:   # Monte-Carlo
            random_state = RandomState(seed)
            for s, n in self.sample(sample_count, seed=seed, as_dict=True).iteritems():
                if importance_sampling:
                    compatible_likelihoods = [l for l in likelihoods if l[0] in s]
                    weight = 1.0 - reduce(operator.mul, [1.0 - l[1] for l in compatible_likelihoods], 1.0)
                else:
                    compatible_likelihoods = likelihoods
                if not compatible_likelihoods:
                    continue
                empty_mass = reduce(operator.mul, [1.0 - l[1] for l in compatible_likelihoods], 1.0)
                for _ in xrange(n):
                    rv = random_state.random_sample(len(compatible_likelihoods))
                    subtree_mass = 1.0
                    hyp = set()
                    for k in xrange(len(compatible_likelihoods)):
                        l = compatible_likelihoods[k][1]
                        norm = 1.0 if hyp else 1.0 - empty_mass / subtree_mass
                        if l / norm > rv[k]:
                            hyp.add(compatible_likelihoods[k][0])
                        else:
                            subtree_mass *= 1 - l   # only relevant for negative case
                    if importance_sampling:
                        combined[hyp] += weight
                    else:
                        hyp &= s
                        if hyp:
                            combined[hyp] += 1.0
            return combined.normalize()
    
    def condition(self, hypothesis):
        """Conditions the mass function with hypothesis based on Dempster's rule of conditioning."""
        return self.combine_conjunctive(MassFunction([(hypothesis, 1.0)]))
    
    def conflict(self, mass_function):
        """
        Calculates the weight of conflict between two mass functions.
        
        It is defined as the logarithm of the normalization constant in Dempster's rule of combination.
        """ 
        c = 0.0
        for h1, v1 in self.iteritems():
            for h2, v2 in mass_function.iteritems():
                if not h1 & h2:
                    c += v1 * v2
        if c >= 1:
            return float("inf")
        else:
            return -log(1.0 - c, 2)
    
    def normalize(self):
        """Normalizes the mass function in-place so that the sum of all mass values equals 1."""
        mass_sum = sum(self.values())
        if mass_sum != 1.0:
            for h, v in self.iteritems():
                self[h] = v / mass_sum
        return self
    
    def markov_update(self, transition_model, sample_count = None, seed = None):
        """
        Performs a first-order Markov update.
        
        This mass function expresses the belief about the current state and the 'transition_model' describes the state transition belief.
        The transition model is a function that takes a singleton state as input and returns possible successor states either as a MassFunction
        or as a single randomly-sampled state set.
        """
        updated = MassFunction()
        if sample_count == None:
            # deterministic
            for k, v in self.iteritems():
                predicted = None
                for e in k:
                    if predicted == None:
                        predicted = transition_model(e)
                    else:
                        predicted |= transition_model(e)
                for kp, vp in predicted.iteritems():
                    updated[kp] += v * vp
        else:
            # Monte-Carlo
            for s, n in self.sample(sample_count, seed=seed, as_dict=True).iteritems():
                unions = [[] for _ in xrange(n)]
                for e in s:
                    ts = transition_model(e, n)
                    for i, t in enumerate(ts):
                        unions[i].extend(t)
                for u in unions:
                    updated[u] += 1.0 / sample_count
        return updated
    
    def extend(self, spaces, index):
        extended = MassFunction()
        for k, v in self.iteritems():
            extended[frozenset(product(*(spaces[:index] + [k] + spaces[index:])))] = v
        return extended
    
    def project(self, dims):
        projected = MassFunction()
        for mk, v in self.iteritems():
            projected[[t[d] for t in mk for d in dims]] += v
        return projected
    
    def pignistify(self):
        """Generates the pignistic transformation of this mass function."""
        p = MassFunction()
        for h, v in self.iteritems():
            for s in h:
                p[(s,)] += v / len(h)
        return p
    
    def local_conflict(self):
        c = 0.0
        for h, v in self.iteritems():
            c += v * log(len(h) / v, 2)
        return c
    
    def distance(self, m):
        """Evidential distance between two mass functions according to Jousselme et al. "A new distance between two bodies of evidence". Information Fusion, 2001."""
        def sp(m1, m2, cache):
            p = 0
            for h1, v1 in m1.iteritems():
                for h2, v2 in m2.iteritems():
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
    
    def distance_pnorm(self, m, p = 2):
        d = sum([(v - m[h])**p for h, v in self.iteritems()])
        for h, v in m.iteritems():
            if h not in self:
                d += v**p
        return d**(1.0 / p)
    
    def prune(self, hypotheses_tree):
        """
        Restricts the distribution to a tree-like hypothesis space.
        
        Removes all hypotheses that are not part of the tree and reassigns their
        mass values by preserving the pignistic transform and maximizing the Hartley measure.
        'hypotheses_tree' is a list of tree nodes.
        """ 
        hypotheses_tree = [frozenset(h) for h in sorted(hypotheses_tree, lambda h1, h2: len(h2) - len(h1))]
        pruned = MassFunction()
        for h, v in self.iteritems():
            size = len(h)
            for node in hypotheses_tree:
                if h.issuperset(node):
                    pruned[node] += len(node) * v / size
                    h -= node
                    if len(h) == 0:
                        break
        return pruned
    
    def is_compatible(self, m):
        for h, v in m.iteritems():
            if self.pl(h) < v:
                return False
        return True
    
    def sample(self, n, maximum_likelihood=True, seed=None, as_dict=False):
        """
        TODO update
        Generates a list of n samples from this distribution.
        
        Arguments:
        n -- The number of samples.
        seed -- The random seed used for drawing samples.
        as_dict -- Return the samples as a list (False) or as a dictionary of samples and their frequencies (True). 
        
        Returns:
        Either a list of hypothesis drawn (with replacement) from the distribution with probability proportional to their mass values
        or a dictionary containing the drawn hypotheses and their frequencies.
        """
        if not isinstance(n, int):
            raise TypeError('n must be int')
        samples = dict() if as_dict else []
        mass_sum = sum(self.values())
        if maximum_likelihood:
            remainders = []
            remaining_sample_count = n
            for h, v in self.iteritems():
                fraction = n * v / mass_sum
                quotient = int(fraction)
                if quotient > 0:
                    if as_dict:
                        samples[h] = quotient
                    else:
                        samples.extend([h] * quotient)
                remainders.append((h, fraction - quotient))
                remaining_sample_count -= quotient
            remainders.sort(reverse=True, key=lambda (h, v): v)
            for h, _ in remainders[:remaining_sample_count]:
                if as_dict:
                    if h in samples:
                        samples[h] += 1
                    else:
                        samples[h] = 1
                else:
                    samples.append(h)
        else:
            random_values = RandomState(seed).random_sample(n) * mass_sum
            hypotheses = sorted(self.iteritems(), reverse=True, key=lambda (h, v): v)
            for i in range(n):
                mass = 0.0
                for h, v in hypotheses:
                    mass += v
                    if mass >= random_values[i]:
                        if as_dict:
                            if h in samples:
                                samples[h] += 1
                            else:
                                samples[h] = 1
                        else:
                            samples.append(h)
                        break
        if not as_dict:
            Random(seed).shuffle(samples)
        return samples

def gbt_pl(hypothesis, likelihoods):
    """Computes the plausibility of hypothesis from a list of likelihoods using the generalized Bayesian theorem."""
    eta = 1 - reduce(operator.mul, [1.0 - l[1] for l in likelihoods], 1.0)
    return (1 - reduce(operator.mul, [1.0 - l[1] for l in likelihoods if l[0] in hypothesis], 1.0)) / eta
    
def gbt_q(hypothesis, likelihoods):
    """Computes the commonality of hypothesis from a list of likelihoods using the generalized Bayesian theorem."""
    eta = 1 - reduce(operator.mul, [1.0 - l[1] for l in likelihoods], 1.0)
    return reduce(operator.mul, [l[1] for l in likelihoods if l[0] in hypothesis], 1.0) / eta
