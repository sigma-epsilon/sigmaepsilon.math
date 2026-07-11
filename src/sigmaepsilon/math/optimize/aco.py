"""Ant Colony Optimization (ACO) algorithm base classes and implementations."""

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
    "AntSolution",
    "AntColonyOptimization",
    "ContinuousAntColonyOptimization",
    "CombinatorialAntColonyOptimization",
]


class AntSolution(BaseModel):
    """A data class for ant solutions."""

    phenotype: list[float] = Field(default_factory=list)
    fitness: float
    index: int = Field(default=-1)

    def __eq__(self, other) -> bool:
        if not isinstance(other, AntSolution):
            return False
        return np.allclose(self.phenotype, other.phenotype)

    def __hash__(self):
        return hash(tuple(float(x) for x in self.phenotype))

    def __gt__(self, other):
        if not isinstance(other, AntSolution):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return self.fitness > other.fitness

    def __lt__(self, other):
        if not isinstance(other, AntSolution):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return self.fitness < other.fitness


class AntColonyOptimization:
    """
    Base class for Ant Colony Optimization (ACO) algorithms.

    This class provides the common framework for ACO algorithms, with two extension
    points that subclasses must implement: :func:`construct_solutions` and
    :func:`update_pheromones`.

    Parameters
    ----------
    fnc : Callable
        The objective function to optimize. It should accept an N-dimensional
        input and return a scalar fitness value.
    ranges : Iterable
        Ranges for each dimension of the search space, as a list of [min, max] pairs.
    nAnts : int, Optional
        Number of ants in the colony. Default is 50.
    maxiter : int, Optional
        Maximum number of iterations. Default is 200.
    miniter : int, Optional
        Minimum number of iterations before stopping criteria can trigger. Default is 0.
    rho : float, Optional
        Pheromone evaporation rate. Default is 0.1.
    minimize : bool, Optional
        If True, minimize the objective function. Default is False (maximize).
    seed : int | numpy.random.SeedSequence | numpy.random.Generator | None, Optional
        Random seed for reproducibility. Default is None.
    vectorized : bool, Optional
        If True, the objective function accepts batched inputs. Default is False.
    n_jobs : int, Optional
        Number of parallel jobs for evaluation. -1 uses all CPUs. Default is 1.
    maxage : int, Optional
        Maximum number of iterations without improvement before convergence. Default is 10.

    Note
    ----
    This is an abstract base class. Use :class:`ContinuousAntColonyOptimization` for
    real-valued optimization or :class:`CombinatorialAntColonyOptimization` for
    discrete/graph-based problems.
    """

    @unique
    class Status(Enum):
        """Status codes for the ACO algorithm."""

        INITIALIZED = 0
        MAX_ITERATIONS_REACHED = 1
        CONVERGED = 2
        ERROR = -1

    __slots__ = [
        "fnc",
        "ranges",
        "dim",
        "nAnts",
        "maxiter",
        "miniter",
        "rho",
        "maxage",
        "_is_symbolic_Function",
        "_celebrate_op",
        "_minimize",
        "_state",
        "_status",
        "_rng",
        "_champion",
        "vectorized",
        "n_jobs",
    ]

    def __init__(
        self,
        fnc: Callable,
        ranges: Iterable,
        *,
        nAnts: int = 50,
        maxiter: int = 200,
        miniter: int = 0,
        rho: float = 0.1,
        minimize: bool = False,
        seed: int | np.random.SeedSequence | RNG | NoneType = None,
        vectorized: bool = False,
        n_jobs: int = 1,
        maxage: int = 10,
    ):
        super().__init__()
        self.fnc = fnc
        self.ranges = np.array(ranges)
        self.dim = getattr(fnc, "dimension", self.ranges.shape[0])
        self._rng = seed if isinstance(seed, RNG) else np.random.default_rng(seed)
        self.vectorized = vectorized
        self.n_jobs = n_jobs

        self._is_symbolic_Function = isinstance(fnc, Function) and fnc.is_symbolic

        self.nAnts = nAnts
        self.rho = rho
        self.maxage = maxage
        self._champion: AntSolution | NoneType = None
        self._celebrate_op = None
        self._minimize = False
        self._state = None
        self._status = None
        self.set_solution_params(
            maxiter=maxiter,
            miniter=miniter,
            minimize=minimize,
        )
        self.reset()

    @property
    def state(self) -> OptimizerState:
        """Return the state of the optimizer."""
        return self._state

    @property
    def rng(self) -> RNG:
        """Return the random number generator of the instance."""
        return self._rng

    @property
    def champion(self) -> AntSolution:
        """Return the best solution found so far."""
        return self._champion

    def reset(self) -> "AntColonyOptimization":
        """Reset the solver and return the object."""
        self._champion = None
        self._state = OptimizerState()
        self._status = AntColonyOptimization.Status.INITIALIZED
        return self

    def set_solution_params(self, **kwargs) -> "AntColonyOptimization":
        """Set the hyperparameters of the algorithm."""
        if "maxiter" in kwargs:
            self.maxiter = kwargs["maxiter"]
        if "miniter" in kwargs:
            self.miniter = kwargs["miniter"]
        if "minimize" in kwargs:
            self._minimize = kwargs["minimize"]

        if self.miniter > self.maxiter:
            raise ValueError("'maxiter' must be greater than 'miniter'")

        self._celebrate_op = operator.lt if self._minimize else operator.gt
        return self

    def evolve(self, cycles: int = 1) -> ndarray:
        """Perform a number of evolution cycles."""
        for _ in range(cycles):
            solutions = self.construct_solutions()
            fitness = self.evaluate(solutions)
            self.update_pheromones(solutions, fitness)

            argfunc = np.argmin if self._minimize else np.argmax
            best_idx = argfunc(fitness)
            candidate = AntSolution(
                phenotype=list(solutions[best_idx]),
                fitness=float(fitness[best_idx]),
                index=int(best_idx),
            )
            self._celebrate(candidate)
        return solutions

    def solve(self, recycle: bool = False, **kwargs) -> AntSolution:
        """Solve the optimization problem and return the best solution."""
        if not recycle:
            self.reset()
        self.set_solution_params(**kwargs)

        finished = False
        self._status = AntColonyOptimization.Status.INITIALIZED

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
                    self._status = AntColonyOptimization.Status.MAX_ITERATIONS_REACHED
                elif self.stopping_criteria():
                    self._status = AntColonyOptimization.Status.CONVERGED
                self._state.success = True
            else:  # pragma: no cover
                self._state.success = False
        except Exception as e:
            self._state.success = False
            self._state.message = str(e)
            self._status = AntColonyOptimization.Status.ERROR
            raise e
        finally:
            self._state.stage = self._status.value

        return self.champion

    def evaluate(self, phenotypes: ndarray | None = None) -> ndarray:
        """Evaluate the objective function for a set of solutions."""
        try:
            if self._is_symbolic_Function:
                result = self.fnc(phenotypes.T)
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
            result = None
            raise RuntimeError(
                "Error during evaluation of the objective function."
            ) from e
        return result

    def best_phenotype(self) -> ndarray:
        """Return the phenotype of the best solution found."""
        return np.array(self.champion.phenotype) if self.champion else None

    def best_candidate(self) -> AntSolution:
        """Return the best solution found (alias for champion)."""
        return self.champion

    def _celebrate(self, solution: AntSolution) -> None:
        """Update the champion if a better solution is found."""
        if self.champion is None:
            self._champion = solution
            self._champion.index = 0
        else:
            is_better = self._celebrate_op(solution.fitness, self.champion.fitness)
            if is_better:
                self._champion = solution
                self._champion.index = 0
        self.state.x = self.champion.phenotype
        self.state.fun = self.champion.fitness
        self._champion.index += 1

    def stopping_criteria(self) -> bool:
        """Check if the algorithm should stop (stagnation-based)."""
        if self.champion is None:
            return False
        return self.champion.index > self.maxage

    def construct_solutions(self) -> ndarray:
        """Construct solutions using the ACO mechanism. Must be implemented by subclass."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement 'construct_solutions'."
        )

    def update_pheromones(self, solutions: ndarray, fitness: ndarray) -> None:
        """Update pheromone information. Must be implemented by subclass."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement 'update_pheromones'."
        )


class ContinuousAntColonyOptimization(AntColonyOptimization):
    """
    Continuous Ant Colony Optimization (ACOR) for real-valued problems.

    Implements the ACOR algorithm (Socha & Dorigo, 2008) which maintains a solution
    archive and samples new solutions using Gaussian kernels.

    Parameters
    ----------
    fnc : Callable
        The objective function to optimize.
    ranges : Iterable
        Ranges for each dimension as [min, max] pairs.
    archive_size : int, Optional
        Size of the solution archive. Default is 50.
    q : float, Optional
        Locality of the search (smaller = more focused). Default is 0.5.
    xi : float, Optional
        Convergence speed parameter. Default is 0.85.
    nAnts : int, Optional
        Number of ants constructing solutions per iteration. Default is 50.
    maxiter : int, Optional
        Maximum number of iterations. Default is 200.
    miniter : int, Optional
        Minimum iterations before stopping. Default is 0.
    rho : float, Optional
        Not used in ACOR but kept for API compatibility. Default is 0.1.
    minimize : bool, Optional
        If True, minimize the objective. Default is False.
    seed : int | numpy.random.SeedSequence | numpy.random.Generator | None, Optional
        Random seed for reproducibility. Default is None.
    vectorized : bool, Optional
        If True, objective accepts batched inputs. Default is False.
    n_jobs : int, Optional
        Number of parallel jobs. Default is 1.
    maxage : int, Optional
        Stagnation threshold. Default is 10.
    """

    __slots__ = ["archive_size", "q", "xi", "_archive", "_archive_fitness"]

    def __init__(
        self,
        fnc: Callable,
        ranges: Iterable,
        *,
        archive_size: int = 50,
        q: float = 0.5,
        xi: float = 0.85,
        nAnts: int = 50,
        maxiter: int = 200,
        miniter: int = 0,
        rho: float = 0.1,
        minimize: bool = False,
        seed: int | np.random.SeedSequence | RNG | NoneType = None,
        vectorized: bool = False,
        n_jobs: int = 1,
        maxage: int = 10,
    ):
        self.archive_size = archive_size
        self.q = q
        self.xi = xi
        self._archive = None
        self._archive_fitness = None
        super().__init__(
            fnc,
            ranges,
            nAnts=nAnts,
            maxiter=maxiter,
            miniter=miniter,
            rho=rho,
            minimize=minimize,
            seed=seed,
            vectorized=vectorized,
            n_jobs=n_jobs,
            maxage=maxage,
        )

    def reset(self) -> "ContinuousAntColonyOptimization":
        """Reset the solver and initialize the archive with random solutions."""
        super().reset()
        lower = self.ranges[:, 0]
        upper = self.ranges[:, 1]
        self._archive = self.rng.uniform(
            lower, upper, size=(self.archive_size, self.dim)
        )
        self._archive_fitness = self.evaluate(self._archive)
        self._sort_archive()
        return self

    def _sort_archive(self) -> None:
        """Sort archive by fitness (best first based on minimize flag)."""
        if self._minimize:
            order = np.argsort(self._archive_fitness)
        else:
            order = np.argsort(self._archive_fitness)[::-1]
        self._archive = self._archive[order]
        self._archive_fitness = self._archive_fitness[order]

    def construct_solutions(self) -> ndarray:
        """Construct solutions using ACOR Gaussian kernel sampling."""
        k = self.archive_size
        solutions = np.zeros((self.nAnts, self.dim))

        weights = np.zeros(k)
        for rank in range(k):
            weights[rank] = (1.0 / (self.q * k * np.sqrt(2 * np.pi))) * np.exp(
                -((rank) ** 2) / (2 * (self.q * k) ** 2)
            )
        weights /= weights.sum()

        for ant in range(self.nAnts):
            chosen_idx = self.rng.choice(k, p=weights)
            chosen_solution = self._archive[chosen_idx]

            for d in range(self.dim):
                distances = np.abs(self._archive[:, d] - chosen_solution[d])
                sigma = self.xi * np.sum(distances) / (k - 1) if k > 1 else 1.0
                sigma = max(sigma, 1e-10)
                solutions[ant, d] = self.rng.normal(chosen_solution[d], sigma)

        lower = self.ranges[:, 0]
        upper = self.ranges[:, 1]
        solutions = np.clip(solutions, lower, upper)
        return solutions

    def update_pheromones(self, solutions: ndarray, fitness: ndarray) -> None:
        """Update archive by merging new solutions and keeping the best."""
        combined = np.vstack([self._archive, solutions])
        combined_fitness = np.concatenate([self._archive_fitness, fitness])

        if self._minimize:
            order = np.argsort(combined_fitness)
        else:
            order = np.argsort(combined_fitness)[::-1]

        self._archive = combined[order[: self.archive_size]]
        self._archive_fitness = combined_fitness[order[: self.archive_size]]


class CombinatorialAntColonyOptimization(AntColonyOptimization):
    """
    Combinatorial Ant Colony Optimization (Ant System) for TSP-like problems.

    Implements the classic Ant System algorithm (Dorigo et al., 1996) for
    solving traveling salesman-type problems on a distance matrix.

    Parameters
    ----------
    fnc : Callable
        Tour evaluation function. Takes a tour (array of node indices) and
        returns the tour cost/length.
    distance_matrix : ndarray
        Symmetric distance matrix where distance_matrix[i,j] is the cost
        from node i to node j.
    alpha : float, Optional
        Pheromone importance factor. Default is 1.0.
    beta : float, Optional
        Heuristic importance factor. Default is 2.0.
    Q : float, Optional
        Pheromone deposit factor. Default is 1.0.
    nAnts : int, Optional
        Number of ants. Default is 50.
    maxiter : int, Optional
        Maximum iterations. Default is 200.
    miniter : int, Optional
        Minimum iterations. Default is 0.
    rho : float, Optional
        Pheromone evaporation rate. Default is 0.1.
    minimize : bool, Optional
        If True, minimize tour length. Default is True.
    seed : int | numpy.random.SeedSequence | numpy.random.Generator | None, Optional
        Random seed. Default is None.
    maxage : int, Optional
        Stagnation threshold. Default is 10.
    tau_init : float, Optional
        Initial pheromone value. Default is 0.1.
    """

    __slots__ = [
        "distance_matrix",
        "alpha",
        "beta",
        "Q",
        "tau_init",
        "_tau",
        "_eta",
        "nNodes",
    ]

    def __init__(
        self,
        fnc: Callable,
        distance_matrix: ndarray,
        *,
        alpha: float = 1.0,
        beta: float = 2.0,
        Q: float = 1.0,
        nAnts: int = 50,
        maxiter: int = 200,
        miniter: int = 0,
        rho: float = 0.1,
        minimize: bool = True,
        seed: int | np.random.SeedSequence | RNG | NoneType = None,
        maxage: int = 10,
        tau_init: float = 0.1,
    ):
        self.distance_matrix = np.array(distance_matrix)
        self.nNodes = self.distance_matrix.shape[0]
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.tau_init = tau_init
        self._tau = None
        self._eta = None

        dummy_ranges = np.array([[0, self.nNodes - 1]] * self.nNodes)
        super().__init__(
            fnc,
            dummy_ranges,
            nAnts=nAnts,
            maxiter=maxiter,
            miniter=miniter,
            rho=rho,
            minimize=minimize,
            seed=seed,
            vectorized=False,
            n_jobs=1,
            maxage=maxage,
        )

    def reset(self) -> "CombinatorialAntColonyOptimization":
        """Reset the solver and initialize pheromone matrix."""
        super().reset()
        self._tau = np.full((self.nNodes, self.nNodes), self.tau_init)
        with np.errstate(divide="ignore"):
            self._eta = 1.0 / self.distance_matrix
        self._eta[~np.isfinite(self._eta)] = 0.0
        return self

    def construct_solutions(self) -> ndarray:
        """Construct tours using random-proportional rule."""
        tours = np.zeros((self.nAnts, self.nNodes), dtype=int)

        for ant in range(self.nAnts):
            visited = np.zeros(self.nNodes, dtype=bool)
            current = self.rng.integers(0, self.nNodes)
            tours[ant, 0] = current
            visited[current] = True

            for step in range(1, self.nNodes):
                unvisited = np.where(~visited)[0]
                probs = (
                    (self._tau[current, unvisited] ** self.alpha)
                    * (self._eta[current, unvisited] ** self.beta)
                )
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs /= prob_sum
                else:
                    probs = np.ones(len(unvisited)) / len(unvisited)

                next_node = self.rng.choice(unvisited, p=probs)
                tours[ant, step] = next_node
                visited[next_node] = True
                current = next_node

        return tours

    def update_pheromones(self, solutions: ndarray, fitness: ndarray) -> None:
        """Evaporate pheromones and deposit based on tour quality."""
        self._tau *= 1 - self.rho

        for ant in range(len(solutions)):
            tour = solutions[ant]
            tour_length = fitness[ant]
            if tour_length > 0:
                deposit = self.Q / tour_length
                for i in range(self.nNodes):
                    from_node = tour[i]
                    to_node = tour[(i + 1) % self.nNodes]
                    self._tau[from_node, to_node] += deposit
                    self._tau[to_node, from_node] += deposit

    def evaluate(self, tours: ndarray) -> ndarray:
        """Evaluate tour lengths."""
        fitness = np.zeros(len(tours))
        for i, tour in enumerate(tours):
            fitness[i] = self.fnc(tour)
        self.state.n_fev += len(tours)
        return fitness
