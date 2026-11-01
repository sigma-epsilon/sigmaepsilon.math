"""Shared machinery for population-based optimizers (PSO, DE) with multi-solution support."""

from typing import Iterable, Callable
from types import NoneType
import operator
from enum import Enum, unique
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy import ndarray
from numpy.random import Generator as RNG
from pydantic import BaseModel, Field

from ..function import Function
from .state import OptimizerState

__all__ = [
    "Candidate",
    "PopulationOptimizer",
    "normalize",
    "hill_valley",
    "distinct_optima",
]


class Candidate(BaseModel):
    """A data class for candidate solutions of population-based optimizers."""

    phenotype: list[float] = Field(default_factory=list)
    fitness: float
    age: int = Field(default=0)
    index: int = Field(default=-1)

    def __eq__(self, other) -> bool:
        """Return whether self and other are at the same point of the search space."""
        if not isinstance(other, Candidate):
            return False
        return np.array_equal(self.phenotype, other.phenotype)

    def __hash__(self):
        """Return a hash of self based on the phenotype."""
        return hash(tuple(float(x) for x in self.phenotype))

    def __gt__(self, other):
        """Return whether self's fitness is greater than other's fitness."""
        if not isinstance(other, Candidate):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return self.fitness > other.fitness

    def __lt__(self, other):
        """Return whether self's fitness is less than other's fitness."""
        if not isinstance(other, Candidate):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return self.fitness < other.fitness

    def __ge__(self, other):
        """Return whether self's fitness is greater than or equal to other's fitness."""
        if not isinstance(other, Candidate):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return self.fitness >= other.fitness

    def __le__(self, other):
        """Return whether self's fitness is less than or equal to other's fitness."""
        if not isinstance(other, Candidate):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return self.fitness <= other.fitness


def normalize(points: ndarray, ranges: ndarray) -> ndarray:
    """
    Map points of the search space into a normalized space.

    Every coordinate is scaled to [0, 1] using its range, and the result is divided by
    the square root of the dimension, so that Euclidean distances between normalized
    points of the box also fall into [0, 1], independently of the dimension and the
    units of the variables.

    Parameters
    ----------
    points: numpy.ndarray
        A 1d point or a 2d array of points (one per row).
    ranges: numpy.ndarray
        The ``(dim, 2)`` array of lower and upper bounds.
    """
    ranges = np.asarray(ranges, dtype=float)
    lower = ranges[:, 0]
    span = ranges[:, 1] - lower
    span = np.where(span > 0, span, 1.0)
    return (np.asarray(points, dtype=float) - lower) / span / np.sqrt(len(span))


def hill_valley(
    evaluate: Callable,
    x_a: ndarray,
    f_a: float,
    x_b: ndarray,
    f_b: float,
    *,
    minimize: bool = True,
    samples: int = 3,
) -> bool:
    """
    Return True if the hill-valley test finds a barrier between two points.

    The objective is evaluated at `samples` equally spaced interior points of the segment
    between `x_a` and `x_b`. If any of them is worse than both endpoints, the two points
    are separated by a hill (minimization) or a valley (maximization), hence they belong
    to different basins of attraction (Ursem, 1999).

    Parameters
    ----------
    evaluate: Callable
        A function that takes a 2d array of points (one per row) and returns
        one objective value per row.
    x_a, x_b: numpy.ndarray
        The two points.
    f_a, f_b: float
        The objective values at the two points.
    minimize: bool, Optional
        Whether the objective is minimized. Default is True.
    samples: int, Optional
        The number of interior points to test. Default is 3.
    """
    x_a = np.asarray(x_a, dtype=float)
    x_b = np.asarray(x_b, dtype=float)
    t = np.linspace(0, 1, samples + 2)[1:-1]
    points = x_a + t[:, None] * (x_b - x_a)
    values = np.asarray(evaluate(points), dtype=float)
    if minimize:
        return bool(np.any(values > max(f_a, f_b)))
    return bool(np.any(values < min(f_a, f_b)))


def distinct_optima(
    points: ndarray,
    fitness: ndarray,
    k: int,
    *,
    ranges: ndarray,
    radius: float,
    minimize: bool = True,
    evaluate: Callable | NoneType = None,
    samples: int = 3,
) -> list[int]:
    """
    Pick the indices of at most `k` points that represent different optima.

    The points are visited from best to worst. A point is accepted if it is farther than
    `radius` (measured in the space of :func:`normalize`) from every point accepted so
    far, and, if `evaluate` is provided, the :func:`hill_valley` test finds a barrier
    between it and every accepted point. Points that fail are considered duplicates of
    an optimum that is already represented by a better point.

    Parameters
    ----------
    points: numpy.ndarray
        The ``(n, dim)`` array of points.
    fitness: numpy.ndarray
        The objective values of the points.
    k: int
        The maximum number of points to return.
    ranges: numpy.ndarray
        The ``(dim, 2)`` array of lower and upper bounds.
    radius: float
        The minimum normalized distance between accepted points.
    minimize: bool, Optional
        Whether the objective is minimized. Default is True.
    evaluate: Callable, Optional
        A function evaluating the objective for a 2d array of points. If not
        provided, only the distance criterion is used. Default is None.
    samples: int, Optional
        The minimum number of interior points for the hill-valley test. Longer
        segments get more points, so that consecutive points are at most
        ``radius / 2`` apart, and hills narrower than a niche aren't missed when the
        segment crosses other basins. Default is 3.

    Returns
    -------
    list[int]
        The indices of the accepted points, best first.
    """
    points = np.asarray(points, dtype=float)
    fitness = np.asarray(fitness, dtype=float)
    order = np.argsort(fitness if minimize else -fitness, kind="stable")
    normalized = normalize(points, ranges)
    accepted: list[int] = []
    for i in order:
        if len(accepted) >= k:
            break
        if not np.all(np.isfinite(points[i])) or not np.isfinite(fitness[i]):
            continue
        if accepted:
            distances = np.linalg.norm(normalized[accepted] - normalized[i], axis=1)
            if np.any(distances <= radius):
                continue
            if evaluate is not None:
                is_distinct = True
                for n in np.argsort(distances):
                    j = accepted[n]
                    n_samples = max(samples, int(np.ceil(2 * distances[n] / radius)))
                    if not hill_valley(
                        evaluate,
                        points[i],
                        fitness[i],
                        points[j],
                        fitness[j],
                        minimize=minimize,
                        samples=n_samples,
                    ):
                        is_distinct = False
                        break
                if not is_distinct:
                    continue
        accepted.append(int(i))
    return accepted


class PopulationOptimizer:
    """
    Base class for population-based optimizers with multi-solution support.

    It provides the driver shared by
    :class:`~sigmaepsilon.math.optimize.pso.ParticleSwarmOptimization` and
    :class:`~sigmaepsilon.math.optimize.de.DifferentialEvolution`, with the same usage
    pattern as :class:`~sigmaepsilon.math.optimize.ga.GeneticAlgorithm`
    (:func:`solve`, :func:`evolve`, :attr:`champion`, :attr:`state`, ...). A subclass
    must implement three extension points: :func:`initialize`, :func:`step` and
    :func:`archive`.

    Besides the champion, the optimizer can return several solutions that represent
    different optima of the objective (see :attr:`solutions`). Set `n_solutions` to the
    number of optima you are interested in; subclasses then switch to a niching variant
    of the algorithm that keeps several basins of attraction populated.

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
    nPop: int, Optional
        The size of the population. Default is 40.
    maxiter: int, Optional
        The maximum number of iterations. Default is 200.
    miniter: int, Optional
        The minimum number of iterations. Default is 0.
    maxage: int, Optional
        The maximum number of iterations the champion can spend at the top without
        a significant improvement before termination. If `n_solutions` is greater
        than 1, it is the maximum number of iterations without a significant
        improvement of any member of the archive. Default is 20.
    ftol: float, Optional
        An improvement of the champion smaller than ``ftol * max(1, |fitness|)`` doesn't
        reset its age. Default is 1e-10.
    minimize: bool, Optional
        If True, the objective function is minimized. Default is False.
    seed: int | numpy.random.SeedSequence | numpy.random.Generator | None, Optional
        A seed for a per-instance :func:`numpy.random.default_rng`, used for every
        stochastic operation instead of the global :mod:`numpy.random` state. Default is
        None (nondeterministic).
    vectorized: bool, Optional
        If True, :func:`evaluate` calls the objective function once with the whole
        ``(n, dim)`` array of points and expects it to return one value per row.
        Default is False.
    n_jobs: int, Optional
        If different from 1 and ``vectorized`` is False, :func:`evaluate` calls the
        objective function in parallel worker processes; -1 means "use all available
        CPUs". The objective function must be picklable. Default is 1.
    n_solutions: int, Optional
        The maximum number of distinct solutions returned by :attr:`solutions`. If
        greater than 1, the niching variant of the algorithm is used. Default is 1.
    niche_radius: float, Optional
        The radius of a niche, measured in a normalized space where the whole box of
        `ranges` has a diameter of 1 (see :func:`normalize`). Solutions closer to each
        other than this are considered to be the same. Default is 0.1.

    See Also
    --------
    :class:`~sigmaepsilon.math.optimize.pso.ParticleSwarmOptimization`
    :class:`~sigmaepsilon.math.optimize.de.DifferentialEvolution`
    """

    @unique
    class Status(Enum):
        """Status codes for population-based optimizers."""

        INITIALIZED = 0
        MAX_ITERATIONS_REACHED = 1
        CONVERGED = 2
        ERROR = -1

    __slots__ = [
        "fnc",
        "ranges",
        "dim",
        "nPop",
        "maxiter",
        "miniter",
        "maxage",
        "ftol",
        "n_solutions",
        "niche_radius",
        "vectorized",
        "n_jobs",
        "_is_symbolic_Function",
        "_celebrate_op",
        "_minimize",
        "_state",
        "_status",
        "_rng",
        "_champion",
        "_phenotypes",
        "_fitness",
        "_solutions",
        "_stagnation",
    ]

    def __init__(
        self,
        fnc: Callable,
        ranges: Iterable,
        *,
        nPop: int = 40,
        maxiter: int = 200,
        miniter: int = 0,
        maxage: int = 20,
        ftol: float = 1e-10,
        minimize: bool = False,
        seed: int | np.random.SeedSequence | RNG | NoneType = None,
        vectorized: bool = False,
        n_jobs: int = 1,
        n_solutions: int = 1,
        niche_radius: float = 0.1,
    ):
        super().__init__()
        self.fnc = fnc
        self.ranges = np.array(ranges, dtype=float)
        self.dim = getattr(fnc, "dimension", self.ranges.shape[0])
        self._rng = seed if isinstance(seed, RNG) else np.random.default_rng(seed)
        self.vectorized = vectorized
        self.n_jobs = n_jobs
        self._is_symbolic_Function = isinstance(fnc, Function) and fnc.is_symbolic

        if nPop < 2:
            raise ValueError("'nPop' must be at least 2")
        self.nPop = nPop
        self.ftol = ftol
        self._champion: Candidate | NoneType = None
        self._celebrate_op = None
        self._minimize = False
        self._state = None
        self._status = None
        self._phenotypes = None
        self._fitness = None
        self._solutions = None
        self.n_solutions = 1
        self.niche_radius = 0.1
        self.set_solution_params(
            maxiter=maxiter,
            miniter=miniter,
            maxage=maxage,
            minimize=minimize,
            n_solutions=n_solutions,
            niche_radius=niche_radius,
        )
        self.reset()

    @property
    def state(self) -> OptimizerState:
        """Return the state of the optimizer."""
        return self._state

    @property
    def rng(self) -> RNG:
        """
        Return the random number generator of the instance.

        All stochastic operations must draw from this generator rather than the global
        :mod:`numpy.random` state, so that runs are reproducible (via the `seed`
        constructor argument).
        """
        return self._rng

    @property
    def champion(self) -> Candidate:
        """Return the best solution found so far."""
        return self._champion

    @property
    def phenotypes(self) -> ndarray:
        """Return the points of the current population."""
        return self._phenotypes

    @property
    def fitness(self) -> ndarray:
        """Return the objective values of the current population."""
        return self._fitness

    @property
    def diversity(self) -> float:
        """
        Return a simple measure of the diversity of the current population.

        Computed as the mean, over all dimensions, of the per-dimension standard
        deviation of the points. A value close to zero indicates a converged,
        homogeneous population.
        """
        if self._phenotypes is None:
            return 0.0
        return float(np.mean(np.std(self._phenotypes, axis=0)))

    @property
    def solutions(self) -> list[Candidate]:
        """
        Return at most `n_solutions` solutions representing different optima, best first.

        The solutions are picked from :func:`archive` with :func:`distinct_optima`: a
        point is only returned if it is farther than `niche_radius` from every better
        solution, and the hill-valley test finds a barrier between them. Hence, the same
        optimum is never returned twice, and the list can be shorter than `n_solutions`
        if fewer distinct optima were found. The first solution is always the champion.

        .. note::
           The hill-valley tests evaluate the objective function, which is accounted
           for in :attr:`state`. The result is cached until the next call to
           :func:`evolve` or :func:`reset`.
        """
        if self._champion is None:
            return []
        if self._solutions is None:
            points, fitness = self.archive()
            evaluate = self.evaluate if self.n_solutions > 1 else None
            indices = distinct_optima(
                points,
                fitness,
                self.n_solutions,
                ranges=self.ranges,
                radius=self.niche_radius,
                minimize=self._minimize,
                evaluate=evaluate,
            )
            self._solutions = [
                Candidate(
                    phenotype=list(map(float, points[i])),
                    fitness=float(fitness[i]),
                    index=i,
                )
                for i in indices
            ]
        return self._solutions

    def reset(self) -> "PopulationOptimizer":
        """
        Reset the solver and return the object.

        Only use it if you want to have a completely clean sheet. Also, the function
        is called for every object at instantiation.
        """
        self._champion = None
        self._phenotypes = None
        self._fitness = None
        self._solutions = None
        self._stagnation = 0
        self._state = OptimizerState()
        self._status = PopulationOptimizer.Status.INITIALIZED
        return self

    def set_solution_params(self, **kwargs) -> "PopulationOptimizer":
        """
        Set the hyperparameters of the algorithm.

        Parameters
        ----------
        maxiter: int, Optional
            Maximum number of iterations.
        miniter: int, Optional
            Minimum number of iterations.
        maxage: int, Optional
            Maximum age of the champion.
        minimize: bool, Optional
            If True, the objective function is minimized.
        n_solutions: int, Optional
            The maximum number of distinct solutions to return.
        niche_radius: float, Optional
            The normalized radius of a niche.
        """
        if "maxiter" in kwargs:
            self.maxiter = kwargs["maxiter"]
        if "miniter" in kwargs:
            self.miniter = kwargs["miniter"]
        if "maxage" in kwargs:
            self.maxage = kwargs["maxage"]
        if "minimize" in kwargs:
            self._minimize = kwargs["minimize"]
        if "n_solutions" in kwargs:
            self.n_solutions = kwargs["n_solutions"]
        if "niche_radius" in kwargs:
            self.niche_radius = kwargs["niche_radius"]

        if self.miniter > self.maxiter:
            raise ValueError("'maxiter' must be greater than 'miniter'")

        if not isinstance(self.n_solutions, int) or self.n_solutions < 1:
            raise ValueError("'n_solutions' must be a positive integer")

        if not 0 < self.niche_radius <= 1:
            raise ValueError("'niche_radius' must be in the interval (0, 1]")

        self._celebrate_op = operator.lt if self._minimize else operator.gt
        self._solutions = None
        return self

    def evolve(self, cycles: int = 1) -> ndarray:
        """Perform a certain number of iterations and return the current population."""
        for _ in range(cycles):
            if self._phenotypes is None:
                self.initialize()
            previous = np.array(self.archive()[1], dtype=float)
            self.step()
            self._solutions = None
            self._track_stagnation(previous, self.archive()[1])
            self._celebrate(self.best_candidate())
            self._state.diversity = self.diversity
        return self.phenotypes

    def solve(self, recycle: bool = False, **kwargs) -> Candidate:
        """
        Solve the problem and return the champion.

        Use :attr:`solutions` afterwards to get several distinct solutions.

        Parameters
        ----------
        recycle: bool, Optional
            If True, the leftover resources of the previous calculation are the starting
            point of the new solution.
        kwargs: dict
            Additional parameters to be passed to the :func:`set_solution_params`.

        Returns
        -------
        :class:`~sigmaepsilon.math.optimize.population.Candidate`
            The best candidate.
        """
        if not recycle:
            self.reset()
        self.set_solution_params(**kwargs)

        finished = False
        self._status = PopulationOptimizer.Status.INITIALIZED

        try:
            while not finished:
                self.evolve(1)
                self.state.n_iter += 1
                min_iter_reached = self.state.n_iter >= self.miniter
                max_iter_reached = self.state.n_iter >= self.maxiter
                finished = (
                    self.stopping_criteria() or max_iter_reached
                ) and min_iter_reached

            if finished:
                if max_iter_reached:
                    self._status = PopulationOptimizer.Status.MAX_ITERATIONS_REACHED
                elif self.stopping_criteria():
                    self._status = PopulationOptimizer.Status.CONVERGED
                self._state.success = True
            else:  # pragma: no cover
                self._state.success = False
        except Exception as e:
            self._state.success = False
            self._state.message = str(e)
            self._status = PopulationOptimizer.Status.ERROR
            raise e
        finally:
            self._state.stage = self._status.value

        return self.champion

    def evaluate(self, phenotypes: ndarray | None = None) -> ndarray:
        """
        Evaluate the objective for a 2d array of points (one per row).

        If the points are not explicitly specified, the current population is evaluated.

        Note
        ----
        By default, the objective function is called once per point in a plain Python
        loop. Set ``vectorized=True`` at construction if the objective function can
        consume the whole ``(n, dim)`` array at once, or set ``n_jobs`` to a value other
        than 1 to evaluate points in parallel worker processes.
        """
        try:
            phenotypes = self.phenotypes if phenotypes is None else phenotypes
            phenotypes = np.asarray(phenotypes, dtype=float)
            if self._is_symbolic_Function:
                result = np.asarray(self.fnc(phenotypes.T), dtype=float)
            elif self.vectorized:
                result = np.asarray(self.fnc(phenotypes), dtype=float)
            elif self.n_jobs != 1:
                max_workers = None if self.n_jobs < 0 else self.n_jobs
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    result = np.array(
                        list(executor.map(self.fnc, phenotypes)), dtype=float
                    )
            else:
                result = np.array([self.fnc(x) for x in phenotypes], dtype=float)
            self.state.n_fev += len(phenotypes)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Error during evaluation of the objective function."
            ) from e
        return result

    def is_better(self, a: ndarray | float, b: ndarray | float) -> ndarray | bool:
        """Return whether objective value(s) `a` are strictly better than `b`."""
        return self._celebrate_op(a, b)

    def best_phenotype(self) -> ndarray:
        """Return the phenotype of the best candidate of the archive."""
        return np.array(self.best_candidate().phenotype)

    def best_candidate(self) -> Candidate:
        """
        Return the best candidate of the archive.

        .. note::
           The archive of a population-based optimizer only ever improves, hence
           this is also the best known solution after an iteration. The reigning
           champion, with its age, is available through the :attr:`champion` property.
        """
        points, fitness = self.archive()
        argfunc = np.argmin if self._minimize else np.argmax
        index = int(argfunc(fitness))
        return Candidate(
            phenotype=list(map(float, points[index])),
            fitness=float(fitness[index]),
            index=index,
        )

    def _celebrate(self, candidate: Candidate) -> None:
        """Update the champion and its age."""
        champion = self._champion
        if champion is None:
            self._champion = candidate
        elif self._celebrate_op(candidate.fitness, champion.fitness):
            improvement = abs(candidate.fitness - champion.fitness)
            significant = improvement > self.ftol * max(1.0, abs(champion.fitness))
            candidate.age = 0 if significant else champion.age
            self._champion = candidate
        self.state.x = self._champion.phenotype
        self.state.fun = self._champion.fitness
        self._champion.age += 1

    def _track_stagnation(self, previous: ndarray, current: ndarray) -> None:
        """Count the iterations without a significant improvement of the archive."""
        improvement = np.abs(np.asarray(current, dtype=float) - previous)
        threshold = self.ftol * np.maximum(1.0, np.abs(previous))
        if np.any(improvement > threshold):
            self._stagnation = 0
        else:
            self._stagnation += 1

    def stopping_criteria(self) -> bool:
        """Implement a simple stopping criterion.

        The default implementation stops if the champion didn't improve significantly
        for more than `maxage` iterations. If `n_solutions` is greater than 1, it stops
        if no member of the archive improved significantly for more than `maxage`
        iterations, so that the search goes on while any of the niches is improving.
        """
        if self._champion is None:
            return False
        if self.n_solutions > 1:
            return self._stagnation > self.maxage
        return self._champion.age > self.maxage

    def initialize(self) -> None:
        """Create and evaluate the initial population.

        This is an algorithm-specific extension point; override it in a subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement 'initialize'."
        )

    def step(self) -> None:
        """Perform one iteration of the algorithm.

        This is an algorithm-specific extension point; override it in a subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement 'step'.")

    def archive(self) -> tuple[ndarray, ndarray]:
        """Return the best points found so far and their objective values.

        These are the points :attr:`solutions` are picked from, e.g. the personal
        bests of a particle swarm. This is an algorithm-specific extension point;
        override it in a subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement 'archive'."
        )
