import unittest

import numpy as np

from sigmaepsilon.math.optimize import IntegerGeneticAlgorithm as IGA


VALUES = np.array([60, 100, 120, 80, 30])
WEIGHTS = np.array([10, 20, 30, 15, 5])
CAPACITY = 50


def knapsack(x):
    total_weight = np.dot(WEIGHTS, x)
    total_value = np.dot(VALUES, x)
    penalty = 1000 * max(0, total_weight - CAPACITY)
    return total_value - penalty


class TestIGA(unittest.TestCase):
    def test_IGA(self):
        ranges = [[0, 1]] * len(VALUES)
        iga = IGA(knapsack, ranges, length=1, nPop=50, seed=0)
        iga.evolve()
        iga.genotypes = iga.genotypes
        iga.fitness
        iga.state.to_scipy()
        iga.best_phenotype()
        iga.best_candidate()
        iga.reset()

    def test_knapsack_convergence(self):
        ranges = [[0, 1]] * len(VALUES)
        iga = IGA(knapsack, ranges, length=1, nPop=50, seed=0)
        champion = iga.solve()
        selection = np.array(champion.phenotype)
        total_weight = np.dot(WEIGHTS, selection)
        self.assertLessEqual(total_weight, CAPACITY)
        self.assertEqual(champion.fitness, 270.0)

    def test_reproducibility_with_seed(self):
        ranges = [[0, 1]] * len(VALUES)
        iga1 = IGA(knapsack, ranges, length=1, nPop=20, seed=0)
        iga2 = IGA(knapsack, ranges, length=1, nPop=20, seed=0)
        result1 = iga1.solve()
        result2 = iga2.solve()
        self.assertEqual(result1.phenotype, result2.phenotype)

    def test_no_dict_on_instances(self):
        ranges = [[0, 1]] * len(VALUES)
        iga = IGA(knapsack, ranges, length=1, nPop=20, seed=0)
        with self.assertRaises(AttributeError):
            iga.__dict__

    def test_boolean_domain_stays_binary(self):
        ranges = [[0, 1]] * len(VALUES)
        iga = IGA(knapsack, ranges, length=1, nPop=20, seed=1)
        for _ in range(5):
            iga.evolve(1)
            phenotypes = iga.phenotypes
            self.assertTrue(set(np.unique(phenotypes)).issubset({0, 1}))
            self.assertTrue(np.issubdtype(phenotypes.dtype, np.integer))

    def test_integer_domain_within_bounds(self):
        def f(x):
            return -((x[0] - 4) ** 2)

        iga = IGA(f, [[0, 6]], length=3, nPop=20, seed=2)
        for _ in range(10):
            iga.evolve(1)
            phenotypes = iga.phenotypes
            self.assertTrue((phenotypes >= 0).all())
            self.assertTrue((phenotypes <= 6).all())
            self.assertTrue(np.array_equal(phenotypes, np.round(phenotypes)))

    def test_integer_domain_converges(self):
        def f(x):
            return -((x[0] - 4) ** 2)

        iga = IGA(f, [[0, 6]], length=3, nPop=20, minimize=False, maxage=10, seed=2)
        champion = iga.solve()
        self.assertEqual(champion.phenotype[0], 4.0)


if __name__ == "__main__":
    unittest.main()
