import pytest
from sigmaepsilon.math.optimize.ga import GeneticAlgorithm, Genom
import numpy as np


class DummyGA(GeneticAlgorithm):
    def stopping_criteria(self):
        raise RuntimeError("Intentional error for testing error handling.")

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


def test_ga_error_handling():
    ga = DummyGA(lambda x: sum(x), [(0, 1), (0, 1)], nPop=4)
    with pytest.raises(
        RuntimeError, match="Intentional error for testing error handling."
    ):
        ga.solve()
    assert ga._status == GeneticAlgorithm.Status.ERROR
    assert ga._state.success is False
    assert "Intentional error" in ga._state.message
