import unittest

from sigmaepsilon.math.optimize import BinaryGeneticAlgorithm
from sigmaepsilon.math.function.functions import Rosenbrock as Rosenbrock_sym


def Rosenbrock(a, b, x, y):
    return (a - x) ** 2 + b * (y - x**2) ** 2


class TestBGA(unittest.TestCase):
    def test_BGA(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        f.dimension = 2
        ranges = [[-10, 10], [-10, 10]]
        BGA = BinaryGeneticAlgorithm(f, ranges, length=12, nPop=200)
        BGA.evolver()
        BGA.evolve()
        BGA.genotypes = BGA.genotypes
        BGA.fittness
        BGA.best_phenotype()
        BGA.best_candidate()
        BGA.random_parents_generator(BGA.genotypes)
        BGA.reset()

    def test_BGA_elitism_eq_1(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        f.dimension = 2
        ranges = [[-10, 10], [-10, 10]]
        BGA = BinaryGeneticAlgorithm(f, ranges, length=6, nPop=100, elitism=1)
        BGA.evolve()
        BGA.divide()

    def test_BGA_elitism_gt_1(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        f.dimension = 2
        ranges = [[-10, 10], [-10, 10]]
        BGA = BinaryGeneticAlgorithm(f, ranges, length=6, nPop=100, elitism=2)
        BGA.evolve()
        BGA.divide()

    def test_BGA_elitism_lt_1(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        f.dimension = 2
        ranges = [[-10, 10], [-10, 10]]
        BGA = BinaryGeneticAlgorithm(f, ranges, length=6, nPop=100, elitism=0.5)
        BGA.evolve()
        BGA.divide()

    def test_Rosenbrock(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        f.dimension = 2
        ranges = [[-10, 10], [-10, 10]]
        BGA = BinaryGeneticAlgorithm(f, ranges, length=12, nPop=200)
        BGA.solve()

        BGA.genotypes = BGA.genotypes
        BGA.evolver()
        BGA.evolve()
        
    def test_symbolic_function_bulk_eval(self):
        obj = Rosenbrock_sym()
        ranges = [[-10, 10], [-10, 10]]
        bga = BinaryGeneticAlgorithm(obj, ranges, length=12, nPop=200)
        bga.evolve(2)


if __name__ == "__main__":
    unittest.main()
