from sigmaepsilon.math.optimize.ga import GeneticAlgorithm
import numpy as np


class MaxIterGA(GeneticAlgorithm):
    def stopping_criteria(self):
        return False  # Never converges, only maxiter can stop it

    def encode(self, phenotypes=None):
        return np.array([[0, 1], [1, 0]])

    def decode(self, genotypes):
        return np.array([[0.0, 1.0], [1.0, 0.0]])

    def populate(self, genotypes=None):
        return np.array([[0, 1], [1, 0]])

    def crossover(self, parent1, parent2):
        return parent1, parent2

    def mutate(self, child):
        return child

    def select(self, genotypes=None, phenotypes=None):
        return np.array([[0, 1], [1, 0]])


def test_ga_max_iterations_status():
    ga = MaxIterGA(lambda x: sum(x), [(0, 1), (0, 1)], nPop=4, maxiter=2, miniter=2)
    ga.solve()
    assert ga._status == GeneticAlgorithm.Status.MAX_ITERATIONS_REACHED
    assert ga._state.success is True
