from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from .ga import GeneticAlgorithm

__all__ = [
    "SelectionStrategy",
    "TournamentSelection",
    "RouletteSelection",
    "RankSelection",
]


class SelectionStrategy(ABC):
    """
    Base class for pluggable selection strategies used by the default
    :func:`~sigmaepsilon.math.optimize.ga.GeneticAlgorithm.select` implementation.
    """

    @abstractmethod
    def select(self, ga: "GeneticAlgorithm", fitness: ndarray) -> ndarray:
        """
        Returns an array of indices into `fitness` (repetition allowed) identifying the
        winners of the selection process. The number of returned indices must be at
        least ``ga.nPop // 2``, since the caller pairs them up to breed the rest of the
        next generation.

        Parameters
        ----------
        ga: :class:`~sigmaepsilon.math.optimize.ga.GeneticAlgorithm`
            The genetic algorithm instance, used to access `elitism`, `nPop`,
            `_minimize`, `divide` and the shared `rng`.
        fitness: numpy.ndarray
            Fitness values of the current population.
        """
        ...


class TournamentSelection(SelectionStrategy):
    """
    Classic k-way tournament selection: the elite (per ``ga.elitism``) automatically
    survives, and the rest of the winners are picked by repeatedly drawing `k` random
    candidates from the non-elite pool and keeping the fittest of them.

    Parameters
    ----------
    k: int, Optional
        Tournament size. Default is 3.
    """

    def __init__(self, k: int = 3):
        if k < 2:
            raise ValueError("'k' must be at least 2.")
        self.k = k

    def select(self, ga: "GeneticAlgorithm", fitness: ndarray) -> ndarray:
        winners, others = ga.divide(fitness)
        winners = list(winners)
        others = np.asarray(others)
        k = min(self.k, len(others))
        while len(winners) < int(ga.nPop / 2):
            candidates = ga.rng.choice(others, k, replace=False)
            argsort = np.argsort([fitness[ID] for ID in candidates])
            winner = argsort[0] if ga._minimize else argsort[-1]
            winners.append(candidates[winner])
        return np.array(winners, dtype=int)


class RouletteSelection(SelectionStrategy):
    """
    Fitness-proportionate ("roulette wheel") selection: the probability of an individual
    being picked is proportional to its (shifted, non-negative) fitness.
    """

    def select(self, ga: "GeneticAlgorithm", fitness: ndarray) -> ndarray:
        fitness = np.asarray(fitness, dtype=float)
        elit, others = ga.divide(fitness)
        winners = list(elit)

        n_target = max(int(ga.nPop / 2), 1)
        if len(winners) >= n_target or len(others) == 0:
            return np.array(winners, dtype=int)

        pool_fitness = fitness[others]
        if ga._minimize:
            weights = pool_fitness.max() - pool_fitness + 1e-12
        else:
            weights = pool_fitness - pool_fitness.min() + 1e-12
        probabilities = weights / weights.sum()

        n_remaining = n_target - len(winners)
        picked = ga.rng.choice(others, size=n_remaining, replace=True, p=probabilities)
        winners.extend(picked.tolist())
        return np.array(winners, dtype=int)


class RankSelection(SelectionStrategy):
    """
    Rank-based selection: the probability of an individual being picked depends on its
    rank within the population rather than the raw fitness value, which reduces the
    influence of outlier fitness values compared to :class:`RouletteSelection`.
    """

    def select(self, ga: "GeneticAlgorithm", fitness: ndarray) -> ndarray:
        fitness = np.asarray(fitness, dtype=float)
        elit, others = ga.divide(fitness)
        winners = list(elit)

        n_target = max(int(ga.nPop / 2), 1)
        if len(winners) >= n_target or len(others) == 0:
            return np.array(winners, dtype=int)

        pool_fitness = fitness[others]
        goodness = -pool_fitness if ga._minimize else pool_fitness
        order = np.argsort(goodness)
        n = len(pool_fitness)
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        probabilities = ranks / ranks.sum()

        n_remaining = n_target - len(winners)
        picked = ga.rng.choice(others, size=n_remaining, replace=True, p=probabilities)
        winners.extend(picked.tolist())
        return np.array(winners, dtype=int)
