"""Tests for Ant Colony Optimization algorithms."""

import unittest
import operator
import numpy as np

from sigmaepsilon.math.optimize import (
    ContinuousAntColonyOptimization,
    CombinatorialAntColonyOptimization,
    AntSolution,
)
from sigmaepsilon.math.optimize.aco import AntColonyOptimization
from sigmaepsilon.math.function.functions import Rosenbrock as Rosenbrock_sym


def Rosenbrock(a, b, x, y):
    return (a - x) ** 2 + b * (y - x**2) ** 2


class TestAntSolution(unittest.TestCase):
    def test_ant_solution_creation(self):
        sol = AntSolution(phenotype=[1.0, 2.0], fitness=10.0, index=0)
        self.assertEqual(sol.phenotype, [1.0, 2.0])
        self.assertEqual(sol.fitness, 10.0)
        self.assertEqual(sol.index, 0)

    def test_ant_solution_equality(self):
        sol1 = AntSolution(phenotype=[1.0, 2.0], fitness=10.0)
        sol2 = AntSolution(phenotype=[1.0, 2.0], fitness=20.0)
        sol3 = AntSolution(phenotype=[3.0, 4.0], fitness=10.0)
        self.assertEqual(sol1, sol2)
        self.assertNotEqual(sol1, sol3)
        self.assertFalse(sol1 == 1)

    def test_ant_solution_hash(self):
        sol = AntSolution(phenotype=[1.0, 2.0], fitness=10.0)
        hash_val = hash(sol)
        self.assertIsInstance(hash_val, int)

    def test_ant_solution_comparison(self):
        sol1 = AntSolution(phenotype=[1.0, 2.0], fitness=10.0)
        sol2 = AntSolution(phenotype=[3.0, 4.0], fitness=20.0)
        self.assertLess(sol1, sol2)
        self.assertGreater(sol2, sol1)

    def test_ant_solution_comparison_type_error(self):
        sol = AntSolution(phenotype=[1.0, 2.0], fitness=10.0)
        with self.assertRaises(TypeError):
            sol > 1
        with self.assertRaises(TypeError):
            sol < 1


class TestAntColonyOptimizationBase(unittest.TestCase):
    def test_base_class_not_implemented(self):
        """Base class should raise NotImplementedError for construct_solutions."""

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        aco = AntColonyOptimization(f, ranges, nAnts=10, maxiter=5)

        with self.assertRaises(NotImplementedError):
            aco.construct_solutions()

        with self.assertRaises(NotImplementedError):
            aco.update_pheromones(np.zeros((10, 2)), np.zeros(10))

    def test_miniter_gt_maxiter(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        with self.assertRaises(ValueError):
            AntColonyOptimization(f, ranges, miniter=100, maxiter=10)

    def test_reset(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        aco = AntColonyOptimization(f, ranges, nAnts=10)
        aco.reset()
        self.assertIsNone(aco.champion)
        self.assertEqual(aco.state.n_iter, 0)

    def test_properties(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        aco = AntColonyOptimization(f, ranges, nAnts=10, seed=42)
        self.assertIsNotNone(aco.rng)
        self.assertIsNotNone(aco.state)
        self.assertIsNone(aco.champion)

    def test_set_solution_params(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        aco = AntColonyOptimization(f, ranges, nAnts=10)
        aco.set_solution_params(maxiter=500, miniter=10)
        self.assertEqual(aco.maxiter, 500)
        self.assertEqual(aco.miniter, 10)

    def test_celebration_operators(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        aco = AntColonyOptimization(f, ranges, nAnts=10, minimize=False)
        self.assertEqual(aco._celebrate_op, operator.gt)
        aco.set_solution_params(minimize=True)
        self.assertEqual(aco._celebrate_op, operator.lt)


class TestContinuousAntColonyOptimization(unittest.TestCase):
    def test_simple_optimization(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        f.dimension = 2
        ranges = [[-5, 5], [-5, 5]]
        acor = ContinuousAntColonyOptimization(
            f, ranges, archive_size=30, nAnts=20, maxiter=50, minimize=True, seed=42
        )
        result = acor.solve()
        self.assertIsInstance(result, AntSolution)
        self.assertEqual(len(result.phenotype), 2)

    def test_maximize(self):
        def f(x):
            return -(x[0] ** 2 + x[1] ** 2)

        ranges = [[-5, 5], [-5, 5]]
        acor = ContinuousAntColonyOptimization(
            f, ranges, archive_size=20, nAnts=10, maxiter=30, minimize=False, seed=42
        )
        result = acor.solve()
        self.assertIsInstance(result, AntSolution)

    def test_evolve(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        acor = ContinuousAntColonyOptimization(
            f, ranges, archive_size=20, nAnts=10, minimize=True, seed=42
        )
        solutions = acor.evolve(5)
        self.assertEqual(solutions.shape, (10, 2))
        self.assertIsNotNone(acor.champion)

    def test_best_phenotype(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        acor = ContinuousAntColonyOptimization(
            f, ranges, archive_size=20, nAnts=10, minimize=True, seed=42
        )
        acor.evolve(5)
        phenotype = acor.best_phenotype()
        self.assertEqual(len(phenotype), 2)
        self.assertTrue(np.allclose(phenotype, acor.champion.phenotype))

    def test_best_candidate(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        acor = ContinuousAntColonyOptimization(
            f, ranges, archive_size=20, nAnts=10, minimize=True, seed=42
        )
        acor.evolve(5)
        candidate = acor.best_candidate()
        self.assertEqual(candidate, acor.champion)

    def test_reproducibility_with_seed(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        ranges = [[-5, 5], [-5, 5]]
        acor1 = ContinuousAntColonyOptimization(
            f, ranges, archive_size=20, nAnts=10, maxiter=20, minimize=True, seed=42
        )
        acor2 = ContinuousAntColonyOptimization(
            f, ranges, archive_size=20, nAnts=10, maxiter=20, minimize=True, seed=42
        )
        result1 = acor1.solve()
        result2 = acor2.solve()
        self.assertTrue(np.allclose(result1.phenotype, result2.phenotype))
        self.assertAlmostEqual(result1.fitness, result2.fitness)

    def test_symbolic_function(self):
        obj = Rosenbrock_sym()
        ranges = [[-5, 5], [-5, 5]]
        acor = ContinuousAntColonyOptimization(
            obj, ranges, archive_size=20, nAnts=10, maxiter=20, minimize=True, seed=42
        )
        result = acor.solve()
        self.assertIsInstance(result, AntSolution)

    def test_vectorized_evaluation(self):
        def f_vectorized(x):
            return Rosenbrock(1, 100, x[:, 0], x[:, 1])

        f_vectorized.dimension = 2
        ranges = [[-5, 5], [-5, 5]]
        acor = ContinuousAntColonyOptimization(
            f_vectorized,
            ranges,
            archive_size=20,
            nAnts=10,
            maxiter=20,
            minimize=True,
            vectorized=True,
            seed=42,
        )
        result = acor.solve()
        self.assertIsInstance(result, AntSolution)

    def test_reset(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        acor = ContinuousAntColonyOptimization(
            f, ranges, archive_size=20, nAnts=10, minimize=True, seed=42
        )
        acor.evolve(5)
        acor.reset()
        self.assertIsNone(acor.champion)
        self.assertEqual(acor._archive.shape, (20, 2))

    def test_state_tracking(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        acor = ContinuousAntColonyOptimization(
            f, ranges, archive_size=20, nAnts=10, maxiter=10, minimize=True, seed=42
        )
        acor.solve()
        self.assertGreater(acor.state.n_iter, 0)
        self.assertGreater(acor.state.n_fev, 0)
        self.assertIsNotNone(acor.state.x)
        self.assertIsNotNone(acor.state.fun)

    def test_convergence_status(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        acor = ContinuousAntColonyOptimization(
            f,
            ranges,
            archive_size=20,
            nAnts=10,
            maxiter=5,
            maxage=100,
            minimize=True,
            seed=42,
        )
        acor.solve()
        self.assertEqual(
            acor._status, AntColonyOptimization.Status.MAX_ITERATIONS_REACHED
        )

    def test_recycle(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        acor = ContinuousAntColonyOptimization(
            f, ranges, archive_size=20, nAnts=10, maxiter=5, minimize=True, seed=42
        )
        acor.solve()
        initial_fev = acor.state.n_fev
        acor.solve(recycle=True, maxiter=5)
        self.assertGreater(acor.state.n_fev, initial_fev)


class TestCombinatorialAntColonyOptimization(unittest.TestCase):
    def get_tsp_setup(self):
        """Create a simple TSP instance."""
        distance_matrix = np.array(
            [
                [0, 10, 15, 20],
                [10, 0, 35, 25],
                [15, 35, 0, 30],
                [20, 25, 30, 0],
            ]
        )

        def tour_length(tour):
            total = 0
            for i in range(len(tour)):
                total += distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
            return total

        return distance_matrix, tour_length

    def test_simple_tsp(self):
        distance_matrix, tour_length = self.get_tsp_setup()
        aco = CombinatorialAntColonyOptimization(
            tour_length,
            distance_matrix,
            nAnts=10,
            maxiter=30,
            minimize=True,
            seed=42,
        )
        result = aco.solve()
        self.assertIsInstance(result, AntSolution)
        self.assertEqual(len(result.phenotype), 4)
        unique_nodes = set(result.phenotype)
        self.assertEqual(len(unique_nodes), 4)

    def test_maximize_tour(self):
        distance_matrix, tour_length = self.get_tsp_setup()
        aco = CombinatorialAntColonyOptimization(
            tour_length,
            distance_matrix,
            nAnts=10,
            maxiter=20,
            minimize=False,
            seed=42,
        )
        result = aco.solve()
        self.assertIsInstance(result, AntSolution)

    def test_evolve(self):
        distance_matrix, tour_length = self.get_tsp_setup()
        aco = CombinatorialAntColonyOptimization(
            tour_length,
            distance_matrix,
            nAnts=10,
            minimize=True,
            seed=42,
        )
        solutions = aco.evolve(5)
        self.assertEqual(solutions.shape[0], 10)
        self.assertEqual(solutions.shape[1], 4)

    def test_reproducibility_with_seed(self):
        distance_matrix, tour_length = self.get_tsp_setup()
        aco1 = CombinatorialAntColonyOptimization(
            tour_length,
            distance_matrix,
            nAnts=10,
            maxiter=20,
            minimize=True,
            seed=42,
        )
        aco2 = CombinatorialAntColonyOptimization(
            tour_length,
            distance_matrix,
            nAnts=10,
            maxiter=20,
            minimize=True,
            seed=42,
        )
        result1 = aco1.solve()
        result2 = aco2.solve()
        self.assertEqual(result1.phenotype, result2.phenotype)
        self.assertAlmostEqual(result1.fitness, result2.fitness)

    def test_reset(self):
        distance_matrix, tour_length = self.get_tsp_setup()
        aco = CombinatorialAntColonyOptimization(
            tour_length,
            distance_matrix,
            nAnts=10,
            minimize=True,
            seed=42,
        )
        aco.evolve(5)
        aco.reset()
        self.assertIsNone(aco.champion)
        self.assertTrue(np.allclose(aco._tau, aco.tau_init))

    def test_pheromone_parameters(self):
        distance_matrix, tour_length = self.get_tsp_setup()
        aco = CombinatorialAntColonyOptimization(
            tour_length,
            distance_matrix,
            alpha=2.0,
            beta=3.0,
            Q=10.0,
            rho=0.2,
            tau_init=0.5,
            nAnts=10,
            minimize=True,
            seed=42,
        )
        self.assertEqual(aco.alpha, 2.0)
        self.assertEqual(aco.beta, 3.0)
        self.assertEqual(aco.Q, 10.0)
        self.assertEqual(aco.rho, 0.2)
        self.assertEqual(aco.tau_init, 0.5)

    def test_larger_tsp(self):
        rng = np.random.default_rng(42)
        n = 10
        coords = rng.uniform(0, 100, size=(n, 2))
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))

        def tour_length(tour):
            total = 0
            for i in range(len(tour)):
                total += distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
            return total

        aco = CombinatorialAntColonyOptimization(
            tour_length,
            distance_matrix,
            nAnts=20,
            maxiter=50,
            minimize=True,
            seed=42,
        )
        result = aco.solve()
        self.assertEqual(len(result.phenotype), n)
        self.assertEqual(len(set(result.phenotype)), n)


class TestACOStatusCodes(unittest.TestCase):
    def test_status_enum_values(self):
        self.assertEqual(AntColonyOptimization.Status.INITIALIZED.value, 0)
        self.assertEqual(AntColonyOptimization.Status.MAX_ITERATIONS_REACHED.value, 1)
        self.assertEqual(AntColonyOptimization.Status.CONVERGED.value, 2)
        self.assertEqual(AntColonyOptimization.Status.ERROR.value, -1)

    def test_max_iterations_status(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        ranges = [[-5, 5], [-5, 5]]
        acor = ContinuousAntColonyOptimization(
            f,
            ranges,
            archive_size=20,
            nAnts=10,
            maxiter=5,
            maxage=1000,
            minimize=True,
            seed=42,
        )
        acor.solve()
        self.assertEqual(
            acor._status, AntColonyOptimization.Status.MAX_ITERATIONS_REACHED
        )

    def test_converged_status(self):
        def f(x):
            return 0.0

        ranges = [[-5, 5], [-5, 5]]
        acor = ContinuousAntColonyOptimization(
            f,
            ranges,
            archive_size=10,
            nAnts=5,
            maxiter=1000,
            maxage=3,
            minimize=True,
            seed=42,
        )
        acor.solve()
        self.assertEqual(acor._status, AntColonyOptimization.Status.CONVERGED)


if __name__ == "__main__":
    unittest.main()
