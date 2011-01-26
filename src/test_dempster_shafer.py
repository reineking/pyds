'''
Created on Nov 28, 2009

@author: reineking
'''

import unittest
from math import log
from itertools import product
from dempster_shafer import MassFunction


class TestDempsterShafer(unittest.TestCase):

    def setUp(self):
        self.m1 = MassFunction([(('a',), 0.4), (('b',), 0.2), (('a', 'd'), 0.1), (('a', 'b', 'c', 'd'), 0.3)])
        self.m2 = MassFunction([(('b',), 0.5), (('c',), 0.2), (('a', 'c'), 0.3)])
        self.seed = 1
    
    def _assert_equal_belief(self, m1, m2, places):
        for h in m1.frame_of_discernment() | m2.frame_of_discernment():
            self.assertAlmostEqual(m1[h], m2[h], places)
    
    def test_items(self):
        self.assertEqual(0.0, self.m1['x'])
        self.m1['a', 'd'] += 0.5
        self.assertEqual(0.6, self.m1[('a', 'd')])
    
    def test_copy(self):
        c = self.m1.copy()
        for k in self.m1.keys():
            self.assertEqual(self.m1.belief(k), c.belief(k))
    
    def test_del(self):
        del self.m1[('a',)]
        self.assertEqual(3, len(self.m1))
        self.assertEqual(0.0, self.m1[('a',)])
    
    def test_belief(self):
        self.assertEqual(0.4, self.m1.belief(('a',)))
        self.assertEqual(0.5, self.m1.belief(('a', 'd')))
        self.assertEqual(1, self.m1.belief(('a', 'b', 'c', 'd')))
    
    def test_commonality(self):
        self.assertEqual(0.8, self.m1.commonality(('a',)))
        self.assertEqual(0.5, self.m1.commonality(('b',)))
        self.assertEqual(0.4, self.m1.commonality(('a', 'd')))
        self.assertEqual(0.3, self.m1.commonality(('a', 'b', 'c', 'd')))
    
    def test_plausibility(self):
        self.assertEqual(0.8, self.m1.plausibility(('a',)))
        self.assertEqual(0.5, self.m1.plausibility(('b',)))
        self.assertEqual(0.8, self.m1.plausibility(('a', 'd')))
        self.assertEqual(1, self.m1.plausibility(('a', 'b', 'c', 'd')))
    
    def test_condition(self):
        m = self.m1.condition(('a', 'd'))
        self.assertEqual(0.5, m[('a',)])
        self.assertEqual(0.5, m[('a', 'd')])
    
    def test_combine_conjunctive(self):
        empty = 0.45
        def test(m, places):
            self.assertAlmostEqual(0.15 / (1.0 - empty), m[('a',)], places)
            self.assertAlmostEqual(0.25 / (1.0 - empty), m[('b',)], places)
            self.assertAlmostEqual(0.06 / (1.0 - empty), m[('c',)], places)
            self.assertAlmostEqual(0.09 / (1.0 - empty), m[('a', 'c')], places)
        test(self.m1 & self.m2, 10)
        test(self.m1.combine_conjunctive(self.m2, 10000, self.seed), 2)
        test(self.m1.combine_conjunctive(self.m2, 1000, self.seed, "importance"), 12)
        # combine multiple mass functions
        m_single = self.m1.combine_conjunctive(self.m1).combine_conjunctive(self.m2)
        m_multi = self.m1.combine_conjunctive([self.m1, self.m2])
        for h, v in m_single.items():
            self.assertAlmostEqual(v, m_multi[h])
    
    def test_combine_disjunctive(self):
        def test(m, places):
            self.assertAlmostEqual(0.2, m[('a', 'b')], places)
            self.assertAlmostEqual(0.2, m[('a', 'c')], places)
            self.assertAlmostEqual(0.1, m[('b',)], places)
            self.assertAlmostEqual(0.04, m[('b', 'c')], places)
            self.assertAlmostEqual(0.06, m[('a', 'b', 'c')], places)
            self.assertAlmostEqual(0.05, m[('a', 'b', 'd')], places)
            self.assertAlmostEqual(0.05, m[('a', 'c', 'd')], places)
            self.assertAlmostEqual(0.3, m[('a', 'b', 'c', 'd')], places)
        test(self.m1 | self.m2, 10)
        test(self.m1.combine_disjunctive(self.m2, 10000, self.seed), 2)
        # combine multiple mass functions
        m_single = self.m1.combine_disjunctive(self.m1).combine_disjunctive(self.m2)
        m_multi = self.m1.combine_disjunctive([self.m1, self.m2])
        for h, v in m_single.items():
            self.assertAlmostEqual(v, m_multi[h])
    
    def test_conflict(self):
        self.assertEqual(-log(0.55, 2), self.m1.conflict(self.m2));
    
    def test_normalize(self):
        v = self.m1[('a',)]
        del self.m1[('a',)]
        self.m1.normalize()
        self.assertAlmostEqual(0.2 / (1 - v), self.m1[('b',)])
        self.assertAlmostEqual(0.1 / (1 - v), self.m1[('a', 'd')])
        self.assertAlmostEqual(0.3 / (1 - v), self.m1[('a', 'b', 'c', 'd')])
        self.assertEqual(0, len(MassFunction().normalize()))
    
    def test_multiple_dimensions(self):
        md1 = MassFunction([((('a', 1), ('b', 2)), 0.8), ((('a', 1),), 0.2)])
        md2 = MassFunction([((('a', 1), ('b', 2), ('c', 1)), 1)])
        md12 = md1 & md2
        self.assertAlmostEqual(0.2, md12[(('a', 1),)])
        self.assertAlmostEqual(0.8, md12[(('a', 1), ('b', 2))])
    
    def test_extend(self):
        extended = self.m2.extend([('x', 'y'), (8, 9)], 1)
        self.assertAlmostEqual(0.3, extended[tuple(product(('x', 'y'), ('a', 'c'), (8, 9)))])
    
    def test_project(self):
        projected = self.m2.extend([('x', 'y'), (8, 9)], 1).project([1])
        for k, v in projected.iteritems():
            self.assertAlmostEqual(self.m2[k], v)
    
    def test_pignistify(self):
        p = self.m1.pignistify()
        self.assertEqual(0.525, p[('a',)])
        self.assertEqual(0.275, p[('b',)])
        self.assertEqual(0.075, p[('c',)])
        self.assertEqual(0.125, p[('d',)])
    
    def test_local_conflict(self):
        c = 0.5 * log(1 / 0.5, 2) + 0.2 * log(1 / 0.2, 2) + 0.3 * log(2 / 0.3, 2)
        self.assertEqual(c, self.m2.local_conflict())
        # pignistic entropy
        h = -0.125 * log(0.125, 2) - 0.075 * log(0.075, 2) - 0.275 * log(0.275, 2) - 0.525 * log(0.525, 2)
        self.assertAlmostEqual(h, self.m1.pignistify().local_conflict())
    
    def test_distance(self):
        m3 = MassFunction([(('b',), 0.7), (('c',), 0.3)])
        self.assertEqual(0, m3.distance(m3))
        self.assertEqual(0, self.m2.distance(self.m2))
        self.assertEqual(1, MassFunction([(('a',), 1.0)]).distance(MassFunction([(('b',), 1.0)])))
#        self.assertEqual(sqrt(0.5*(0.3**2+0.7**2+1)), m3.distance(MassFunction([(('x',), 1.0)])))
#        self.assertEqual(sqrt(0.5*(0.5**2+0.2*0.3+0.3*0.45+1)), self.m2.distance(MassFunction([(('x',), 1.0)])))
    
    def test_prune(self):
        m = MassFunction([(('a', 'b'), 0.8), (('b', 'c'), 0.2)])
        m = m.prune([('a',), ('a', 'b', 'c'), ('c',), ('a', 'b'), ('b',)])
        self.assertEqual(3, len(m))
        self.assertEqual(0.8, m[('a', 'b')])
        self.assertEqual(0.1, m[('b',)])
        self.assertEqual(0.1, m[('c',)])
    
    def test_sample(self):
        sample_count = 1000
        samples = self.m1.sample(sample_count, self.seed)
        self.assertEqual(sample_count, len(samples))
        for k, v in self.m1.iteritems():
            self.assertAlmostEqual(v, float(samples.count(k)) / sample_count)
        self.assertEqual(0, len(MassFunction().sample(sample_count)))
    
    def test_markov_update(self):
        def test(m, places):
            self.assertAlmostEqual(0.4 * 0.8, m[(4, 6)], places)
            self.assertAlmostEqual(0.4 * 0.2, m[(5,)], places)
            self.assertAlmostEqual(0.6 * 0.2 * 0.2, m[(0, 1)], places)
            self.assertAlmostEqual(0.6 * 0.2 * 0.8, m[(-1, 1)], places)
            self.assertAlmostEqual(0.6 * 0.2 * 0.8, m[(0, 2)], places)
            self.assertAlmostEqual(0.6 * 0.8 * 0.8, m[(-1, 0, 1, 2)], places)
        m = MassFunction([((0, 1), 0.6), ((5,), 0.4)])
        def transition(s):
            return MassFunction([((s - 1, s + 1), 0.8), ((s,), 0.2)])
        def transition_sampling(s, n):
            self.seed += 1
            return transition(s).sample(n, self.seed)
        test(m.markov_update(transition), 10)
        test(m.markov_update(transition_sampling, 10000, self.seed), 2)
    
    def test_gbt(self):
        def test(m, places):
            self.assertAlmostEqual(0.3 * 0.8 / (1 - 0.7 * 0.2), m[('a', 'b')], places)
            self.assertAlmostEqual(0.3 * 0.2 / (1 - 0.7 * 0.2), m[('a',)], places)
            self.assertAlmostEqual(0.7 * 0.8 / (1 - 0.7 * 0.2), m[('b',)], places)
        pl = [('a', 0.3), ('b', 0.8), ('c', 0.0)]
        test(MassFunction.gbt(pl), 10)
        test(MassFunction.gbt(pl, 10000, self.seed), 2)
        pl = [('a', 0.3), ('b', 0.8), ('c', 0.0), ('d', 1.0)]
        self._assert_equal_belief(MassFunction.gbt(pl), MassFunction.gbt(pl, 10000, self.seed), 2)
    
    def test_gbt_plausibility(self):
        pl = [('a', 0.3), ('b', 0.8), ('c', 0.0), ('d', 1.0)]
        m = MassFunction.gbt(pl)
        for h in m:
            self.assertAlmostEqual(m.plausibility(h), MassFunction.gbt_plausibility(h, pl), 8)
    
    def test_frame_of_discernment(self):
        self.assertEqual(frozenset(['a', 'b', 'c', 'd']), self.m1.frame_of_discernment())
        self.assertEqual(frozenset(['a', 'b', 'c']), self.m2.frame_of_discernment())
    
    def test_combine_gbt(self):
        pl = [('b', 0.8), ('c', 0.5)]
        correct = self.m1.combine_conjunctive(MassFunction.gbt(pl))
        self._assert_equal_belief(correct, self.m1.combine_gbt(pl), 10)
        self._assert_equal_belief(self.m1.combine_gbt(pl), self.m1.combine_gbt(pl, 10000, self.seed), 1)
        self._assert_equal_belief(self.m2.combine_gbt(pl), self.m2.combine_gbt(pl, 10000, self.seed), 1)
        

if __name__ == "__main__":
    unittest.main()