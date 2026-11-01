"""Tests for the shared machinery of population-based optimizers."""

import unittest

import numpy as np

from sigmaepsilon.math.optimize import Candidate, PopulationOptimizer
from sigmaepsilon.math.optimize.population import (
    normalize,
    hill_valley,
    distinct_optima,
)


def double_well(x):
    # minima at x = -1 and x = 1, separated by a hill at x = 0
    x = np.asarray(x, dtype=float)
    return (x[..., 0] ** 2 - 1) ** 2


class TestCandidate(unittest.TestCase):
    def test_equality_and_hash(self):
        a = Candidate(phenotype=[1.0, 2.0], fitness=1.0)
        b = Candidate(phenotype=[1.0, 2.0], fitness=2.0)
        c = Candidate(phenotype=[1.0, 3.0], fitness=1.0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertFalse(a == 1)
        self.assertEqual(hash(a), hash(b))

    def test_comparison(self):
        a = Candidate(phenotype=[0.0], fitness=1.0)
        b = Candidate(phenotype=[1.0], fitness=2.0)
        self.assertLess(a, b)
        self.assertLessEqual(a, b)
        self.assertGreater(b, a)
        self.assertGreaterEqual(b, a)
        for op in ("__lt__", "__le__", "__gt__", "__ge__"):
            with self.assertRaises(TypeError):
                getattr(a, op)(1)


class TestNiching(unittest.TestCase):
    def test_normalize(self):
        ranges = np.array([[0, 10], [-1, 1], [2, 2]])
        points = np.array([[0, -1, 2], [10, 1, 2]])
        normalized = normalize(points, ranges)
        self.assertTrue(np.allclose(normalized[0], 0))
        self.assertAlmostEqual(np.linalg.norm(normalized[1]), np.sqrt(2 / 3))

    def test_hill_valley(self):
        self.assertTrue(hill_valley(double_well, [-1.0], 0.0, [1.0], 0.0))
        self.assertFalse(hill_valley(double_well, [0.9], double_well([0.9]), [1.0], 0.0))
        negated = lambda x: -double_well(x)
        self.assertTrue(hill_valley(negated, [-1.0], 0.0, [1.0], 0.0, minimize=False))
        self.assertFalse(hill_valley(negated, [0.9], -double_well([0.9]), [1.0], 0.0, minimize=False))

    def test_distinct_optima_radius_only(self):
        points = np.array([[0.0], [0.01], [5.0], [9.9]])
        fitness = np.array([0.0, 0.1, 1.0, 2.0])
        ranges = np.array([[0, 10]])
        indices = distinct_optima(points, fitness, 10, ranges=ranges, radius=0.1)
        self.assertEqual(indices, [0, 2, 3])
        indices = distinct_optima(points, fitness, 2, ranges=ranges, radius=0.1)
        self.assertEqual(indices, [0, 2])
        indices = distinct_optima(points, -fitness, 10, ranges=ranges, radius=0.1, minimize=False)
        self.assertEqual(indices, [0, 2, 3])

    def test_distinct_optima_hill_valley(self):
        # 0.5 is on the slope of the basin of 1.0, -1.0 is another basin
        points = np.array([[1.0], [0.5], [-1.0], [np.nan]])
        fitness = double_well(points)
        fitness[3] = -1.0
        ranges = np.array([[-2, 2]])
        indices = distinct_optima(points, fitness, 10, ranges=ranges, radius=0.01)
        self.assertEqual(sorted(indices), [0, 1, 2])
        indices = distinct_optima(
            points, fitness, 10, ranges=ranges, radius=0.01, evaluate=double_well
        )
        self.assertEqual(sorted(indices), [0, 2])


class TestHillValleyAcrossBasins(unittest.TestCase):
    def test_segment_crossing_better_basins(self):
        # the segment between the two outermost minima crosses three better basins;
        # with only 3 interior points, every sample lands near a minimum and the
        # barriers are missed, so the number of samples must grow with the distance
        def wavy(x):
            x = np.asarray(x, dtype=float)
            return np.sin(3 * x[..., 0]) + 0.1 * x[..., 0] ** 2

        points = np.array([[-0.512], [1.537], [-2.561], [3.585], [-4.608]])
        fitness = wavy(points)
        ranges = np.array([[-5, 5]])
        self.assertFalse(
            hill_valley(wavy, points[4], fitness[4], points[3], fitness[3], samples=3)
        )
        indices = distinct_optima(
            points, fitness, 10, ranges=ranges, radius=0.1, evaluate=wavy
        )
        self.assertEqual(sorted(indices), [0, 1, 2, 3, 4])


class TestPopulationOptimizerBase(unittest.TestCase):
    def test_extension_points_not_implemented(self):
        opt = PopulationOptimizer(lambda x: x[0], [[0, 1]])
        with self.assertRaises(NotImplementedError):
            opt.initialize()
        with self.assertRaises(NotImplementedError):
            opt.step()
        with self.assertRaises(NotImplementedError):
            opt.archive()
        with self.assertRaises(NotImplementedError):
            opt.solve()
        self.assertEqual(opt.state.stage, PopulationOptimizer.Status.ERROR.value)
        self.assertFalse(opt.state.success)


if __name__ == "__main__":
    unittest.main()
