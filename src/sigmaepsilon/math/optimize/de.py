"""Differential Evolution (DE) implementation."""

import warnings

import numpy as np
from numpy import ndarray

from .population import PopulationOptimizer, normalize

__all__ = ["DifferentialEvolution"]


class DifferentialEvolution(PopulationOptimizer):
    """
    Differential Evolution (DE).

    It finds minimums or maximums of unconstrained problems over box-ranges of
    continuous variables. In every iteration, a trial vector is created for every
    member :math:`x_i` of the population in three steps:

    * **mutation** builds a mutant vector from scaled differences of other members,
      e.g. :math:`v = x_{r_1} + F (x_{r_2} - x_{r_3})` for the ``"rand/1/bin"``
      strategy, where :math:`r_1, r_2, r_3` are distinct random indices other than
      :math:`i`,
    * **crossover** creates the trial vector :math:`u` by taking every coordinate from
      the mutant with probability `CR`, and from :math:`x_i` otherwise (at least one
      coordinate always comes from the mutant),
    * **selection** replaces :math:`x_i` with :math:`u` if the trial is better.

    If `n_solutions` is greater than 1, the neighborhood-based crowding DE of Qu et al.
    (2012) is used: the donors of a mutant are picked from the `n_neighbors` nearest
    members of the population, and the trial vector competes with the member closest to
    it instead of its parent (crowding, Thomsen 2004). This way, the population can
    converge to several optima simultaneously.

    .. note::
       This class is designed for maximizing the objective function. To minimize it, either negate
       the objective function or pass ``minimize=True`` when instantiating the class.

    Parameters
    ----------
    fnc: Callable
        The function to evaluate. It is assumed, that the function expects and N
        number of scalar arguments as a 1d iterable.
    ranges: Iterable
        Ranges for each scalar argument to the objective function.
    F: float, Optional
        The differential weight (mutation factor), typically in [0.4, 1].
        Default is 0.8.
    CR: float, Optional
        The crossover probability, in [0, 1]. Default is 0.9.
    strategy: str, Optional
        The mutation strategy, one of ``"rand/1/bin"``, ``"best/1/bin"`` and
        ``"current-to-best/1/bin"``. The latter two are greedy and converge faster
        to a single optimum, but they are incompatible with multiple solutions, in
        which case ``"rand/1/bin"`` is used instead. Default is ``"rand/1/bin"``.
    n_neighbors: int, Optional
        The size of the neighborhood the donors of a mutant are picked from if
        `n_solutions` is greater than 1, at least 3. Default is 5.
    nPop: int, Optional
        The size of the population, at least 4. Default is 40.
    kwargs: dict, Optional
        The rest of the parameters are passed to
        :class:`~sigmaepsilon.math.optimize.population.PopulationOptimizer`
        (`maxiter`, `miniter`, `maxage`, `ftol`, `minimize`, `seed`, `vectorized`,
        `n_jobs`, `n_solutions`, `niche_radius`).

    See Also
    --------
    :class:`~sigmaepsilon.math.optimize.pso.ParticleSwarmOptimization`

    Examples
    --------
    Find the minimizer of the Rosenbrock function.
    The exact value of the solution is x = [1.0, 1.0].

    >>> from sigmaepsilon.math.optimize import DifferentialEvolution as DE
    >>>
    >>> def rosenbrock(x):
    ...     a, b = 1, 100
    ...     return (a-x[0])**2 + b*(x[1]-x[0]**2)**2
    >>>
    >>> ranges = [[-5, 5], [-5, 5]]
    >>> de = DE(rosenbrock, ranges, minimize=True, seed=0)
    >>> champion = de.solve()
    >>> x = champion.phenotype
    >>> fx = champion.fitness

    Find the four minima of Himmelblau's function.

    >>> def himmelblau(x):
    ...     return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    >>>
    >>> de = DE(himmelblau, ranges, nPop=60, minimize=True, n_solutions=4, seed=0)
    >>> _ = de.solve()
    >>> len(de.solutions)
    4

    """

    strategies = ("rand/1/bin", "best/1/bin", "current-to-best/1/bin")

    __slots__ = ("F", "CR", "strategy", "n_neighbors")

    def __init__(
        self,
        *args,
        F: float = 0.8,
        CR: float = 0.9,
        strategy: str = "rand/1/bin",
        n_neighbors: int = 5,
        **kwargs,
    ):
        self.F = F
        self.CR = CR
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        super().__init__(*args, **kwargs)
        if self.nPop < 4:
            raise ValueError("'nPop' must be at least 4")

    def set_solution_params(self, **kwargs) -> "DifferentialEvolution":
        """
        Set the hyperparameters of the algorithm.

        Besides the parameters of
        :func:`~sigmaepsilon.math.optimize.population.PopulationOptimizer.set_solution_params`,
        it accepts `F`, `CR`, `strategy` and `n_neighbors`.
        """
        for key in ("F", "CR", "strategy", "n_neighbors"):
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
        if self.strategy not in self.strategies:
            raise ValueError(
                f"Invalid strategy '{self.strategy}', use one of {self.strategies}"
            )
        if not 0 <= self.CR <= 1:
            raise ValueError("'CR' must be in the interval [0, 1]")
        if self.n_neighbors < 3:
            raise ValueError("'n_neighbors' must be at least 3")
        super().set_solution_params(**kwargs)
        if self.n_solutions > 1 and self.strategy != "rand/1/bin":
            warnings.warn(
                f"Strategy '{self.strategy}' is incompatible with multiple solutions, "
                "'rand/1/bin' is used instead.",
                stacklevel=2,
            )
        return self

    def archive(self) -> tuple[ndarray, ndarray]:
        """Return the population and its objective values."""
        return self._phenotypes, self._fitness

    def initialize(self) -> None:
        """Sample the initial population uniformly in the box."""
        lower, upper = self.ranges[:, 0], self.ranges[:, 1]
        self._phenotypes = self.rng.uniform(lower, upper, size=(self.nPop, self.dim))
        self._fitness = self.evaluate(self._phenotypes)

    def _active_strategy(self) -> str:
        return "rand/1/bin" if self.n_solutions > 1 else self.strategy

    def donors(self) -> list[ndarray]:
        """Return the indices of the candidate donors of the mutant of every member.

        With a single solution, every other member of the population is a candidate.
        With multiple solutions, only the `n_neighbors` nearest members are, as in the
        neighborhood-based crowding DE of Qu et al. (2012), so that the difference
        vectors stay within the niche of the member.
        """
        n = self.nPop
        if self.n_solutions == 1:
            return [np.delete(np.arange(n), i) for i in range(n)]
        normalized = normalize(self._phenotypes, self.ranges)
        distances = np.linalg.norm(
            normalized[:, None, :] - normalized[None, :, :], axis=-1
        )
        np.fill_diagonal(distances, np.inf)
        m = min(self.n_neighbors, n - 1)
        return list(np.argsort(distances, axis=1, kind="stable")[:, :m])

    def mutants(self) -> ndarray:
        """Return a mutant vector for every member of the population."""
        x, n = self._phenotypes, self.nPop
        strategy = self._active_strategy()
        best = x[
            np.argmin(self._fitness) if self._minimize else np.argmax(self._fitness)
        ]
        donors = self.donors()
        result = np.empty_like(x)
        for i in range(n):
            r1, r2, r3 = self.rng.choice(donors[i], 3, replace=False)
            if strategy == "rand/1/bin":
                result[i] = x[r1] + self.F * (x[r2] - x[r3])
            elif strategy == "best/1/bin":
                result[i] = best + self.F * (x[r1] - x[r2])
            else:
                result[i] = x[i] + self.F * (best - x[i]) + self.F * (x[r1] - x[r2])
        return result

    def crossover(self, mutants: ndarray) -> ndarray:
        """Return the trial vectors from the population and the mutants (binomial crossover)."""
        x = self._phenotypes
        mask = self.rng.random(x.shape) < self.CR
        j_rand = self.rng.integers(self.dim, size=self.nPop)
        mask[np.arange(self.nPop), j_rand] = True
        trials = np.where(mask, mutants, x)
        return np.clip(trials, self.ranges[:, 0], self.ranges[:, 1])

    def step(self) -> None:
        """Create, evaluate and select the trial vectors."""
        trials = self.crossover(self.mutants())
        fitness = self.evaluate(trials)

        if self.n_solutions == 1:
            improved = self.is_better(fitness, self._fitness)
            self._phenotypes[improved] = trials[improved]
            self._fitness[improved] = fitness[improved]
            return

        normalized = normalize(self._phenotypes, self.ranges)
        normalized_trials = normalize(trials, self.ranges)
        for i in range(self.nPop):
            distances = np.linalg.norm(normalized - normalized_trials[i], axis=1)
            j = int(np.argmin(distances))
            if self.is_better(fitness[i], self._fitness[j]):
                self._phenotypes[j] = trials[i]
                self._fitness[j] = fitness[i]
                normalized[j] = normalized_trials[i]
