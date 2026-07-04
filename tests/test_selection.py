import unittest

from sigmaepsilon.math.optimize import (
    BinaryGeneticAlgorithm,
    TournamentSelection,
    RouletteSelection,
    RankSelection,
)


def Rosenbrock(a, b, x, y):
    return (a - x) ** 2 + b * (y - x**2) ** 2


class TestSelectionStrategies(unittest.TestCase):
    def _make_bga(self, strategy):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        f.dimension = 2
        ranges = [[-10, 10], [-10, 10]]
        return BinaryGeneticAlgorithm(
            f, ranges, length=6, nPop=20, seed=0, selection_strategy=strategy
        )

    def test_tournament_selection(self):
        bga = self._make_bga(TournamentSelection())
        bga.solve()

    def test_tournament_selection_invalid_k(self):
        with self.assertRaises(ValueError):
            TournamentSelection(k=1)

    def test_roulette_selection(self):
        bga = self._make_bga(RouletteSelection())
        bga.solve()

    def test_rank_selection(self):
        bga = self._make_bga(RankSelection())
        bga.solve()

    def test_default_strategy_is_tournament(self):
        def f(x):
            return Rosenbrock(1, 100, x[0], x[1])

        f.dimension = 2
        ranges = [[-10, 10], [-10, 10]]
        bga = BinaryGeneticAlgorithm(f, ranges, length=6, nPop=20)
        self.assertIsInstance(bga.selection_strategy, TournamentSelection)


if __name__ == "__main__":
    unittest.main()
