"""Test cases shared by the population-based optimizers (PSO, DE)."""

import warnings

import numpy as np

from sigmaepsilon.math.optimize import Candidate, OptimizerState
from sigmaepsilon.math.optimize.population import PopulationOptimizer, normalize
from sigmaepsilon.math.function.functions import Rosenbrock as Rosenbrock_sym

HIMMELBLAU_MINIMA = np.array(
    [
        [3.0, 2.0],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584428, -1.848126],
    ]
)

RANGES = [[-5, 5], [-5, 5]]


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_vectorized(x):
    return (1 - x[:, 0]) ** 2 + 100 * (x[:, 1] - x[:, 0] ** 2) ** 2


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def wavy(x):
    # five interior local minima on [-5, 5] with different values
    return np.sin(3 * x[0]) + 0.1 * x[0] ** 2


WAVY_MINIMA = np.array([-0.512, 1.537, -2.561, 3.585, -4.608])


def sphere(x):
    return x[0] ** 2 + x[1] ** 2


def penalized_sphere(x):
    # minimize x^2 + y^2 subject to x + y >= 1, i.e. g(x) = 1 - x - y <= 0
    return sphere(x) + 1e4 * max(0.0, 1 - x[0] - x[1]) ** 2


class PopulationOptimizerCases:
    """Mixin with the test cases; set `optimizer` in the subclass."""

    optimizer: type[PopulationOptimizer]

    def test_basic_api(self):
        opt = self.optimizer(rosenbrock, RANGES, nPop=20, seed=0)
        self.assertEqual(opt.solutions, [])
        opt.evolve()
        opt.evolve(2)
        self.assertEqual(opt.phenotypes.shape, (20, 2))
        self.assertEqual(opt.fitness.shape, (20,))
        self.assertIsInstance(opt.state, OptimizerState)
        opt.state.to_scipy()
        self.assertIsInstance(opt.best_candidate(), Candidate)
        self.assertEqual(len(opt.best_phenotype()), 2)
        self.assertGreaterEqual(opt.diversity, 0.0)
        self.assertEqual(opt.state.n_fev, 4 * 20)
        opt.reset()
        self.assertIsNone(opt.champion)
        self.assertEqual(opt.diversity, 0.0)

    def test_solve_minimize(self):
        opt = self.optimizer(rosenbrock, RANGES, minimize=True, maxage=50, seed=0)
        champion = opt.solve()
        self.assertIsInstance(champion, Candidate)
        self.assertLess(champion.fitness, 1e-3)
        self.assertTrue(np.allclose(champion.phenotype, [1, 1], atol=0.05))
        self.assertTrue(opt.state.success)
        self.assertEqual(opt.state.x, champion.phenotype)
        self.assertEqual(opt.state.fun, champion.fitness)

    def test_solve_maximize(self):
        opt = self.optimizer(lambda x: -sphere(x), RANGES, seed=0)
        champion = opt.solve()
        self.assertGreater(champion.fitness, -1e-6)

    def test_recycle(self):
        opt = self.optimizer(rosenbrock, RANGES, nPop=20, minimize=True, seed=0)
        opt.solve(maxiter=5)
        n_iter, fitness = opt.state.n_iter, opt.champion.fitness
        opt.solve(recycle=True, maxiter=10)
        self.assertGreater(opt.state.n_iter, n_iter)
        self.assertLessEqual(opt.champion.fitness, fitness)

    def test_status(self):
        opt = self.optimizer(rosenbrock, RANGES, nPop=20, maxiter=3, seed=0)
        opt.solve()
        self.assertEqual(opt.state.stage, PopulationOptimizer.Status.MAX_ITERATIONS_REACHED.value)
        opt = self.optimizer(sphere, RANGES, minimize=True, maxage=2, maxiter=1000, seed=0)
        opt.solve()
        self.assertEqual(opt.state.stage, PopulationOptimizer.Status.CONVERGED.value)
        self.assertLess(opt.state.n_iter, 1000)

    def test_miniter(self):
        opt = self.optimizer(sphere, RANGES, minimize=True, maxage=1, miniter=30, seed=0)
        opt.solve()
        self.assertGreaterEqual(opt.state.n_iter, 30)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            self.optimizer(sphere, RANGES, maxiter=10, miniter=20)
        with self.assertRaises(ValueError):
            self.optimizer(sphere, RANGES, n_solutions=0)
        with self.assertRaises(ValueError):
            self.optimizer(sphere, RANGES, niche_radius=0)
        with self.assertRaises(ValueError):
            self.optimizer(sphere, RANGES, nPop=1)

    def test_reproducibility_with_seed(self):
        results = [
            self.optimizer(rosenbrock, RANGES, nPop=20, maxiter=20, seed=0).solve()
            for _ in range(2)
        ]
        self.assertEqual(results[0].phenotype, results[1].phenotype)
        self.assertEqual(results[0].fitness, results[1].fitness)

    def test_no_dict_on_instances(self):
        opt = self.optimizer(rosenbrock, RANGES, nPop=20, seed=0)
        with self.assertRaises(AttributeError):
            opt.__dict__

    def test_phenotypes_within_ranges(self):
        ranges = [[-1, 2], [0, 0.5]]
        opt = self.optimizer(rosenbrock, ranges, nPop=20, seed=0)
        for _ in range(10):
            opt.evolve(1)
            for d in range(2):
                self.assertTrue((opt.phenotypes[:, d] >= ranges[d][0]).all())
                self.assertTrue((opt.phenotypes[:, d] <= ranges[d][1]).all())

    def test_vectorized_matches_default(self):
        kwargs = dict(nPop=20, maxiter=20, minimize=True, seed=0)
        a = self.optimizer(rosenbrock, RANGES, **kwargs).solve()
        b = self.optimizer(rosenbrock_vectorized, RANGES, vectorized=True, **kwargs).solve()
        self.assertTrue(np.allclose(a.phenotype, b.phenotype))

    def test_parallel_matches_default(self):
        kwargs = dict(nPop=8, maxiter=3, minimize=True, seed=0)
        a = self.optimizer(rosenbrock, RANGES, **kwargs).solve()
        b = self.optimizer(rosenbrock, RANGES, n_jobs=2, **kwargs).solve()
        self.assertTrue(np.allclose(a.phenotype, b.phenotype))

    def test_symbolic_function(self):
        opt = self.optimizer(Rosenbrock_sym(), RANGES, nPop=20, maxiter=20, minimize=True, seed=0)
        self.assertIsInstance(opt.solve(), Candidate)

    def test_single_solution(self):
        opt = self.optimizer(himmelblau, RANGES, minimize=True, seed=0)
        champion = opt.solve()
        self.assertEqual(opt.solutions, [champion])

    def test_multiple_solutions_himmelblau(self):
        for seed in range(3):
            opt = self.optimizer(
                himmelblau, RANGES, nPop=60, minimize=True, n_solutions=4, seed=seed
            )
            champion = opt.solve()
            solutions = opt.solutions
            self.assertEqual(len(solutions), 4)
            self.assertEqual(solutions[0], champion)
            self.assertEqual(solutions[0].fitness, champion.fitness)
            fitness = [s.fitness for s in solutions]
            self.assertEqual(fitness, sorted(fitness))
            found = set()
            for s in solutions:
                distances = np.linalg.norm(HIMMELBLAU_MINIMA - s.phenotype, axis=1)
                self.assertLess(distances.min(), 1e-2)
                found.add(int(np.argmin(distances)))
            self.assertEqual(found, {0, 1, 2, 3})
            points = normalize([s.phenotype for s in solutions], opt.ranges)
            for i in range(4):
                for j in range(i + 1, 4):
                    distance = np.linalg.norm(points[i] - points[j])
                    self.assertGreater(distance, opt.niche_radius)

    def test_multiple_solutions_maximize(self):
        opt = self.optimizer(
            lambda x: -himmelblau(x), RANGES, nPop=60, n_solutions=4, seed=0
        )
        champion = opt.solve()
        self.assertEqual(len(opt.solutions), 4)
        self.assertEqual(opt.solutions[0], champion)
        self.assertTrue(all(s.fitness > -1e-2 for s in opt.solutions))

    def test_multiple_solutions_of_different_quality(self):
        opt = self.optimizer(wavy, [[-5, 5]], minimize=True, n_solutions=10, seed=0)
        opt.solve()
        found = np.array([s.phenotype[0] for s in opt.solutions])
        self.assertEqual(len(found), 5)
        self.assertTrue(np.allclose(found, WAVY_MINIMA, atol=1e-2))

    def test_unimodal_gives_no_duplicates(self):
        opt = self.optimizer(sphere, RANGES, minimize=True, n_solutions=3, seed=0)
        opt.solve()
        self.assertEqual(len(opt.solutions), 1)

    def test_solutions_cached_and_counted(self):
        opt = self.optimizer(himmelblau, RANGES, nPop=60, minimize=True, n_solutions=4, seed=0)
        opt.solve(maxiter=50)
        n_fev = opt.state.n_fev
        solutions = opt.solutions
        self.assertGreater(opt.state.n_fev, n_fev)
        n_fev = opt.state.n_fev
        self.assertIs(opt.solutions, solutions)
        self.assertEqual(opt.state.n_fev, n_fev)
        opt.evolve()
        self.assertIsNot(opt.solutions, solutions)

    def test_penalty(self):
        opt = self.optimizer(penalized_sphere, RANGES, minimize=True, seed=0)
        champion = opt.solve()
        self.assertTrue(np.allclose(champion.phenotype, [0.5, 0.5], atol=1e-2))

    def test_set_solution_params(self):
        opt = self.optimizer(sphere, RANGES, seed=0)
        opt.set_solution_params(n_solutions=3, niche_radius=0.2, maxage=7)
        self.assertEqual(opt.n_solutions, 3)
        self.assertEqual(opt.niche_radius, 0.2)
        self.assertEqual(opt.maxage, 7)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            opt.solve(maxiter=2)
