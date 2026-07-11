import unittest

from sigmaepsilon.math.optimize import RealValuedGeneticAlgorithm as RGA


def Rosenbrock(a, b, x, y):
    return (a - x) ** 2 + b * (y - x**2) ** 2


class TestRGA(unittest.TestCase):
    def test_RGA(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        f.dimension = 2
        ranges = [[-10, 10], [-10, 10]]
        rga = RGA(f, ranges, nPop=100, seed=0)
        rga.evolve()
        rga.genotypes = rga.genotypes
        rga.fitness
        rga.state.to_scipy()
        rga.best_phenotype()
        rga.best_candidate()
        rga.reset()

    def test_solve_minimize(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        f.dimension = 2
        ranges = [[-10, 10], [-10, 10]]
        rga = RGA(f, ranges, nPop=100, minimize=True, maxage=20, seed=0)
        champion = rga.solve()
        self.assertLess(champion.fitness, 1.0)

    def test_reproducibility_with_seed(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        f.dimension = 2
        ranges = [[-10, 10], [-10, 10]]
        rga1 = RGA(f, ranges, nPop=20, minimize=True, seed=0)
        rga2 = RGA(f, ranges, nPop=20, minimize=True, seed=0)
        result1 = rga1.solve()
        result2 = rga2.solve()
        self.assertEqual(result1.phenotype, result2.phenotype)

    def test_no_dict_on_instances(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        f.dimension = 2
        ranges = [[-10, 10], [-10, 10]]
        rga = RGA(f, ranges, nPop=20, seed=0)
        with self.assertRaises(AttributeError):
            rga.__dict__

    def test_phenotypes_within_ranges(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        f.dimension = 2
        ranges = [[-10, 10], [-10, 10]]
        rga = RGA(f, ranges, nPop=20, seed=0)
        for _ in range(5):
            rga.evolve(1)
            for d in range(2):
                self.assertTrue((rga.phenotypes[:, d] >= ranges[d][0]).all())
                self.assertTrue((rga.phenotypes[:, d] <= ranges[d][1]).all())


if __name__ == "__main__":
    unittest.main()
