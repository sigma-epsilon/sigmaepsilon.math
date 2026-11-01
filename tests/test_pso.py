"""Tests for Particle Swarm Optimization."""

import unittest

import numpy as np

from sigmaepsilon.math.optimize import ParticleSwarmOptimization as PSO

from .population_cases import PopulationOptimizerCases, RANGES, himmelblau, sphere


class TestPSO(PopulationOptimizerCases, unittest.TestCase):
    optimizer = PSO

    def test_velocities_bounded(self):
        pso = PSO(sphere, RANGES, nPop=20, v_max=0.1, seed=0)
        for _ in range(5):
            pso.evolve()
            self.assertTrue(np.all(np.abs(pso.velocities) <= 0.1 * 10 + 1e-12))

    def test_personal_bests_never_worse(self):
        pso = PSO(sphere, RANGES, nPop=20, minimize=True, seed=0)
        pso.evolve()
        previous = pso.archive()[1].copy()
        for _ in range(5):
            pso.evolve()
            current = pso.archive()[1]
            self.assertTrue(np.all(current <= previous))
            previous = current.copy()

    def test_global_best_attractor(self):
        pso = PSO(sphere, RANGES, nPop=20, minimize=True, seed=0)
        pso.evolve()
        attractors = pso.attractors()
        self.assertTrue(np.allclose(attractors, pso.best_phenotype()))

    def test_species_attractors(self):
        pso = PSO(himmelblau, RANGES, nPop=40, minimize=True, n_solutions=4, seed=0)
        pso.evolve(10)
        seeds = np.unique(pso.attractors(), axis=0)
        self.assertGreater(len(seeds), 1)

    def test_set_solution_params(self):
        super().test_set_solution_params()
        pso = PSO(sphere, RANGES, seed=0)
        pso.set_solution_params(w=0.5, c1=1.0, c2=2.0, v_max=0.3)
        self.assertEqual((pso.w, pso.c1, pso.c2, pso.v_max), (0.5, 1.0, 2.0, 0.3))


if __name__ == "__main__":
    unittest.main()
