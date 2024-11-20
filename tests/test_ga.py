import unittest, operator
import numpy as np

from sigmaepsilon.math.optimize import BinaryGeneticAlgorithm
from sigmaepsilon.math.function.functions import Rosenbrock as Rosenbrock_sym
from sigmaepsilon.math.optimize.ga import Genom


def Rosenbrock(a, b, x, y):
    return (a - x) ** 2 + b * (y - x**2) ** 2


class TestGenom(unittest.TestCase):

    def test_genom(self):
        genom1 = Genom(
            phenotype=np.array([1, 2]),
            genotype=np.array([1, 0, 1, 0]),
            fittness=1.0,
        )

        genom2 = Genom(
            phenotype=np.array([1, 2]),
            genotype=np.array([1, 0, 1, 0]),
            fittness=1.0,
        )

        genom3 = Genom(
            phenotype=np.array([1, 2]),
            genotype=np.array([1, 0, 1, 1]),
            fittness=2.0,
        )

        self.assertFalse(genom1 == 1)
        self.assertFalse(genom1 == genom3)
        hash(genom1)
        self.assertEqual(genom1, genom2)
        self.assertGreater(genom3, genom1)
        self.assertLess(genom1, genom3)
        self.assertGreaterEqual(genom3, genom1)
        self.assertLessEqual(genom1, genom3)

    def test_errors(self):
        genom = Genom(
            phenotype=np.array([1, 2]),
            genotype=np.array([1, 0, 1, 0]),
            fittness=1.0,
        )

        with self.assertRaises(TypeError):
            genom > 1

        with self.assertRaises(TypeError):
            genom >= 1

        with self.assertRaises(TypeError):
            genom < 1

        with self.assertRaises(TypeError):
            genom <= 1


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
        BGA.evolver()
        BGA.evolve()

    def test_symbolic_function_bulk_eval(self):
        obj = Rosenbrock_sym()
        ranges = [[-10, 10], [-10, 10]]
        bga = BinaryGeneticAlgorithm(obj, ranges, length=12, nPop=200)
        bga.evolve(2)
        
    def test_celebration_operators(self):
        obj = Rosenbrock_sym()
        ranges = [[-10, 10], [-10, 10]]
        bga = BinaryGeneticAlgorithm(obj, ranges, length=12, nPop=200)
        self.assertEqual(bga._celebrate_op, operator.gt)
        bga.set_solution_params(minimize=True)
        self.assertEqual(bga._celebrate_op, operator.lt)
        bga = BinaryGeneticAlgorithm(obj, ranges, length=12, nPop=200, minimize=True)
        self.assertEqual(bga._celebrate_op, operator.lt)
        bga.set_solution_params(minimize=False)
        self.assertEqual(bga._celebrate_op, operator.gt)
        
    def test_champion_consistency(self):
        obj = Rosenbrock_sym()
        ranges = [[-10, 10], [-10, 10]]
        bga = BinaryGeneticAlgorithm(obj, ranges, length=12, nPop=10, minimize=True)
        for i in range(15):
            bga.evolve(1)
            champion = bga.champion
            best_phenotype = bga.best_phenotype()
            self.assertTrue(np.all(champion.phenotype == best_phenotype))


if __name__ == "__main__":
    unittest.main()
