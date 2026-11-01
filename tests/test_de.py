"""Tests for Differential Evolution."""

import unittest

import numpy as np

from sigmaepsilon.math.optimize import DifferentialEvolution as DE

from .population_cases import (
    PopulationOptimizerCases,
    RANGES,
    himmelblau,
    rosenbrock,
    sphere,
)


class TestDE(PopulationOptimizerCases, unittest.TestCase):
    optimizer = DE

    def test_strategies(self):
        for strategy in DE.strategies:
            de = DE(rosenbrock, RANGES, minimize=True, strategy=strategy, maxage=50, seed=0)
            champion = de.solve()
            self.assertLess(champion.fitness, 1e-3, strategy)

    def test_invalid_params(self):
        super().test_invalid_params()
        with self.assertRaises(ValueError):
            DE(sphere, RANGES, strategy="rand/2/exp")
        with self.assertRaises(ValueError):
            DE(sphere, RANGES, CR=1.5)
        with self.assertRaises(ValueError):
            DE(sphere, RANGES, nPop=3)
        with self.assertRaises(ValueError):
            DE(sphere, RANGES, n_neighbors=2)

    def test_greedy_strategy_with_multiple_solutions_warns(self):
        with self.assertWarns(UserWarning):
            de = DE(himmelblau, RANGES, strategy="best/1/bin", n_solutions=4)
        self.assertEqual(de._active_strategy(), "rand/1/bin")

    def test_population_never_worse(self):
        for n_solutions in (1, 4):
            de = DE(himmelblau, RANGES, nPop=20, minimize=True, n_solutions=n_solutions, seed=0)
            de.evolve()
            previous = np.sort(de.fitness)
            for _ in range(5):
                de.evolve()
                current = np.sort(de.fitness)
                self.assertLessEqual(current.sum(), previous.sum())
                previous = current

    def test_crossover_takes_a_gene_from_the_mutant(self):
        de = DE(sphere, RANGES, nPop=20, CR=0.0, seed=0)
        de.evolve()
        mutants = de.mutants()
        trials = de.crossover(mutants)
        clipped = np.clip(mutants, -5, 5)
        self.assertTrue(np.all(np.any(np.isclose(trials, clipped), axis=1)))

    def test_neighborhood_donors(self):
        de = DE(sphere, RANGES, nPop=20, n_solutions=2, n_neighbors=4, seed=0)
        de.evolve()
        donors = de.donors()
        self.assertEqual(len(donors), 20)
        for i, d in enumerate(donors):
            self.assertEqual(len(d), 4)
            self.assertNotIn(i, d)

    def test_set_solution_params(self):
        super().test_set_solution_params()
        de = DE(sphere, RANGES, seed=0)
        de.set_solution_params(F=0.5, CR=0.3, strategy="best/1/bin", n_neighbors=7)
        self.assertEqual((de.F, de.CR, de.strategy, de.n_neighbors), (0.5, 0.3, "best/1/bin", 7))


if __name__ == "__main__":
    unittest.main()
