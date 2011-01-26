'''
Created on Nov 27, 2009

@author: reineking
'''

from itertools import product
from math import log, sqrt
from random import Random
from numpy.random import RandomState # much faster than standard random when generating many random numbers


class MassFunction(dict):
    """
        Models a Dempster-Shafer mass function using a dictionary.
    """
    
    def __init__(self, source = None):
        if source != None:
            for h, v in source:
                self[h] += v
    
    @staticmethod
    def _convert(hypothesis):
        if isinstance(hypothesis, frozenset):
            return hypothesis
        else:
            return frozenset(hypothesis)
    
    @staticmethod
    def gbt(likelihoods, sample_count = None, seed = None):
        """
            Constructs a mass function from a list of likelihoods (plausibilities) for a given observation using the Generalized Bayesian Theorem.
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
            empty_mass = reduce(lambda p, l: p * (1.0 - l[1]), likelihoods, 1.0)
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
    
    @staticmethod
    def gbt_plausibility(hypothesis, likelihoods):
        eta = (1 - reduce(lambda p, l: p * (1.0 - l[1]), likelihoods, 1.0))
        likelihoods = [l for l in likelihoods if l[0] in hypothesis]
        return (1 - reduce(lambda p, l: p * (1.0 - l[1]), likelihoods, 1.0)) / eta
    
    def __missing__(self, key):
        return 0.0
    
    def __copy__(self):
        c = MassFunction()
        for k, v in self.iteritems():
            c[k] = v
        return c
    
    def copy(self):
        return self.__copy__()
    
    def __contains__(self, hypothesis):
        return dict.__contains__(self, MassFunction._convert(hypothesis))
    
    def __getitem__(self, hypothesis):
        return dict.__getitem__(self, MassFunction._convert(hypothesis))
    
    def __setitem__(self, hypothesis, value):
        if len(hypothesis) == 0:
            raise Exception("hypothesis is empty")
        value = float(value)
        if value < 0.0:
            raise Exception("mass value is negative")
        if value == 0 and hypothesis in self:
            del self[hypothesis]
        else:
            dict.__setitem__(self, MassFunction._convert(hypothesis), value)
    
    def __delitem__(self, hypothesis):
        return dict.__delitem__(self, MassFunction._convert(hypothesis))
    
    def frame_of_discernment(self):
        return frozenset.union(*self.keys())
    
    def belief(self, hypothesis):
        return self._compute(hypothesis, lambda a, b: a.issuperset(b))
    
    def plausibility(self, hypothesis):
        return self._compute(hypothesis, lambda a, b: a & b)
    
    def commonality(self, hypothesis):
        return self._compute(hypothesis, lambda a, b: b.issuperset(a))
    
    def _compute(self, hypothesis, criterion):
        hypothesis = MassFunction._convert(hypothesis)
        c = 0.0
        for h, v in self.iteritems():
            if criterion(hypothesis, h):
                c += v
        return c
    
    def __and__(self, mass_function):
        return self.combine_conjunctive(mass_function)
    
    def __or__(self, mass_function):
        return self.combine_disjunctive(mass_function)
    
    def combine_conjunctive(self, mass_function, sample_count = None, seed = None, sampling_method = "direct"):
        return self._combine(mass_function, lambda s1, s2: s1 & s2, sample_count, seed, sampling_method)
    
    def combine_disjunctive(self, mass_function, sample_count = None, seed = None):
        return self._combine(mass_function, lambda s1, s2: s1 | s2, sample_count, seed, "direct")
    
    def _combine(self, mass_function, rule, sample_count, seed, sampling_method):
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
        samples1 = self.sample(sample_count, seed)
        samples2 = mass_function.sample(sample_count, seed + 1 if seed != None else None)
        for i in xrange(sample_count):
            s = rule(samples1[i], samples2[i])
            if s:
                combined[s] += 1.0 / sample_count
        return combined.normalize()
    
    def _combine_importance_sampling(self, mass_function, sample_count, seed):
        combined = MassFunction()
        for s1, n in self.sample(sample_count, seed, True).iteritems():
            weight = mass_function.plausibility(s1)
            if seed != None:
                seed += 1
            for s2 in mass_function.condition(s1).sample(n, seed):
                combined[s2] += weight
        return combined.normalize()
    
    def combine_gbt(self, likelihoods, sample_count = None, importance_sampling = True, seed = None):
        """
            likelihoods: list of likelihood tuples
        """
        combined = MassFunction()
        # restrict to generally compatible likelihoods
        frame = self.frame_of_discernment()
        likelihoods = [l for l in likelihoods if l[1] > 0 and l[0] in frame]
        if sample_count == None:    # deterministic
            return self.combine_conjunctive(MassFunction.gbt(likelihoods))
        else:   # Monte-Carlo
            random_state = RandomState(seed)
            for s, n in self.sample(sample_count, seed, True).iteritems():
                if importance_sampling:
                    compatible_likelihoods = [l for l in likelihoods if l[0] in s]
                    weight = 1.0 - reduce(lambda p, l: p * (1.0 - l[1]), compatible_likelihoods, 1.0)
                else:
                    compatible_likelihoods = likelihoods
                if not compatible_likelihoods:
                    continue
                empty_mass = reduce(lambda p, l: p * (1.0 - l[1]), compatible_likelihoods, 1.0)
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
        return self.combine_conjunctive(MassFunction([(hypothesis, 1.0)]))
    
    def conflict(self, mass_function):
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
            for s, n in self.sample(sample_count, seed, True).iteritems():
                unions = [[] for _ in xrange(n)]
                for e in s:
                    for i, t in enumerate(transition_model(e, n)):
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
        """
            Evidential distance between two mass functions according to Jousselme et al. Information Fusion, 2001.
        """
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
    
    def sample(self, n = 1, seed = None, as_dict = False):
        """
            Generates a list of n samples from this distribution.
        """
        if not isinstance(n, int):
            raise TypeError("n must be int")
        samples = dict() if as_dict else []
        remaining_elements = dict()
        mass_sum = sum(self.values())
        remaining_sample_count = n
        for k, v in self.iteritems():
            add = int(n * v / mass_sum)
            if as_dict:
                if add > 0:
                    samples[k] = add
            else:
                samples.extend([k] * add)
            remaining_elements[k] = v - add / n
            remaining_sample_count -= add
        if remaining_sample_count > 0:
            # randomly select the remaining samples from remaining_elements
            remaining_mass_sum = mass_sum - (n - remaining_sample_count) * mass_sum / n
            random_values = sorted(RandomState(seed).random_sample(remaining_sample_count) * remaining_mass_sum)
            sample_index, mass = 0, 0.0
            for k, v in remaining_elements.iteritems():
                mass += v
                while (sample_index < remaining_sample_count and mass >= random_values[sample_index]):
                    if as_dict:
                        if k in samples:
                            samples[k] += 1
                        else:
                            samples[k] = 1
                    else:
                        samples.append(k)
                    sample_index += 1
                if sample_index == remaining_sample_count:
                    break
        if as_dict:
            return samples
        else:
            Random(seed).shuffle(samples)
            return samples
