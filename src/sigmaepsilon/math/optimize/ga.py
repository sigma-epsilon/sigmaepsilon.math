"""Genetic algorithm base classes and population data structures."""

from typing import Iterable, Callable, Tuple, Generator
from types import NoneType
from numbers import Number
import operator
from enum import Enum, unique
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy import ndarray
from numpy.random import Generator as RNG
from pydantic import BaseModel, Field

from ..function import Function
from .state import OptimizerState
from .selection import SelectionStrategy, TournamentSelection

__all__ = ["GeneticAlgorithm", "Genom"]


def even(n: Number) -> bool:
    return n % 2 == 0


def odd(n: Number) -> bool:
    return not even(n)


class Genom(BaseModel):
    """A data class for members of a population."""

    phenotype: list[float] = Field(default_factory=list)
    genotype: list[int | float] = Field(default_factory=list)
    fitness: float
    age: int = Field(default=0)
    index: int = Field(default=-1)

    def __eq__(self, other) -> bool:
        """Return whether self and other have the same genotype."""
        if not isinstance(other, Genom):
            return False
        return np.all(self.genotype == other.genotype)

    def __hash__(self):
        """Return a hash of self based on the genotype."""
        return hash(tuple(float(x) for x in self.genotype))

    def __gt__(self, other):
        """Return whether self's fitness is greater than other's fitness."""
        if not isinstance(other, Genom):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return np.all(self.fitness > other.fitness)

    def __lt__(self, other):
        """Return whether self's fitness is less than other's fitness."""
        if not isinstance(other, Genom):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return np.all(self.fitness < other.fitness)

    def __ge__(self, other):
        """Return whether self's fitness is greater than or equal to other's fitness."""
        if not isinstance(other, Genom):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return np.all(self.fitness >= other.fitness)

    def __le__(self, other):
        """Return whether self's fitness is less than or equal to other's fitness."""
        if not isinstance(other, Genom):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return np.all(self.fitness <= other.fitness)

    @property
    def fittness(self) -> float:  # pragma: no cover
        """Return the fitness value (deprecated alias for :attr:`fitness`)."""
        import warnings

        warnings.warn(
            "'fittness' was a typo and is deprecated; it will be removed in a future version. Use 'fitness' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fitness


class GeneticAlgorithm:
    """
    Base class for Genetic Algorithms (GA).

    Use this as a base class to your custom implementation of a GA.

    The class has 3 representation-specific extension points that a subclass must
    implement to yield a working genetic algorithm: :func:`populate`, :func:`crossover`
    and :func:`mutate` (their base implementations raise ``NotImplementedError``).
    :func:`encode`/:func:`decode` default to the identity mapping (suitable for
    representations where genotype and phenotype coincide, e.g. real-valued encodings)
    and :func:`select` already comes with a working, pluggable default (see
    :attr:`selection_strategy` /
    :class:`~sigmaepsilon.math.optimize.selection.SelectionStrategy`); override them only
    if you need representation-specific behavior. It is also possible to use a custom
    stopping criteria by overriding :func:`stopping_criteria`. See the
    :class:`~sigmaepsilon.math.optimize.bitchromosome.BitChromosomeGeneticAlgorithm`
    (shared bit-chromosome machinery, with
    :class:`~sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm` and
    :class:`~sigmaepsilon.math.optimize.iga.IntegerGeneticAlgorithm` as its concrete,
    continuous/integer subclasses) and
    :class:`~sigmaepsilon.math.optimize.rga.RealValuedGeneticAlgorithm` classes for
    examples.

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
    length: int, Optional
        Chromosome length. The higher the value, the more precision. Default is 5.
    p_c: float, Optional
        Probability of crossover. Default is 1.
    p_m: float, Optional
        Probability of mutation. Default is 0.2.
    nPop: int, Optional
        The size of the population. Default is 100.
    maxiter: int, Optional
        The maximum number of iterations. Default is 200.
    miniter: int, Optional
        The minimum number of iterations. Default is 100.
    elitism: float | int | None, Optional
        Determines the portion of the population designated as elite, which automatically survives
        to the next generation. If less than or equal to 1, it specifies a fraction of the population.
        If greater than 1, it indicates the exact number of individuals to be selected as elite.
        The default value of 1 assures that the reigning champion is always preserved. To turn this off,
        det the value to None. Default is 1.
    ftol: float, Optional
        Torelance for floating point operations. Default is 1e-12.
    maxage: int, Optional
        The age is the maximum number of generations a candidate spends at the top
        (being the best candidate). Default is 5.
    minimize: bool, Optional
        If True, the objective function is minimized. Default is False.
    seed: int | numpy.random.SeedSequence | numpy.random.Generator | None, Optional
        A seed for a per-instance :func:`numpy.random.default_rng`, used for every
        stochastic operation instead of the global :mod:`numpy.random` state. Passing the
        same seed makes runs reproducible, and lets multiple instances draw independent
        random streams in the same process. Default is None (nondeterministic).
    selection_strategy: :class:`~sigmaepsilon.math.optimize.selection.SelectionStrategy`, Optional
        The strategy used by the default :func:`select` implementation to pick the
        survivors of a generation. Default is
        :class:`~sigmaepsilon.math.optimize.selection.TournamentSelection`.
    vectorized: bool, Optional
        If True, :func:`evaluate` calls the objective function once with the whole
        ``(nPop, dim)`` array of phenotypes and expects it to return one fitness value
        per row, instead of calling it once per individual. Default is False.
    n_jobs: int, Optional
        If different from 1 and ``vectorized`` is False, :func:`evaluate` calls the
        objective function once per individual in parallel worker processes (using
        :class:`concurrent.futures.ProcessPoolExecutor`); -1 means "use all available
        CPUs". The objective function must be picklable (e.g. a module-level function,
        not a lambda or a closure). Default is 1 (sequential evaluation).

    Note
    ----
    Be cautious what you use a genetic algorithm for. Like all metahauristic methods, a
    genetic algorithm can be wery demanding on the computational side. If the objective
    function takes a lot of time to evaluate, it is probably not a good idea to use a heuristic
    approach, unless you have a dedicated evaluator that is able to run efficiently for a large
    number of problems or if the long running time is not an issue. If you want to customize the way the
    objective is evaluated, override the :func:`evaluate` method.

    See Also
    --------
    :class:`~sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm`
    :class:`~sigmaepsilon.math.optimize.iga.IntegerGeneticAlgorithm`
    :class:`~sigmaepsilon.math.optimize.rga.RealValuedGeneticAlgorithm`
    """

    @unique
    class Status(Enum):
        """Status codes for the genetic algorithm."""

        INITIALIZED = 0
        MAX_ITERATIONS_REACHED = 1
        CONVERGED = 2
        ERROR = -1

    __slots__ = [
        "fnc",
        "ranges",
        "dim",
        "length",
        "p_c",
        "p_m",
        "nPop",
        "_genotypes",
        "_phenotypes",
        "_fitness",
        "_champion",
        "_evolver",
        "maxiter",
        "miniter",
        "elitism",
        "maxage",
        "_is_symbolic_Function",
        "_celebrate_op",
        "_minimize",
        "_state",
        "_status",
        "_rng",
        "selection_strategy",
        "vectorized",
        "n_jobs",
    ]

    def __init__(
        self,
        fnc: Callable,
        ranges: Iterable,
        *,
        length: int = 5,
        p_c: float = 1,
        p_m: float = 0.2,
        nPop: int = 100,
        maxiter: int = 200,
        miniter: int = 0,
        elitism: int | float | NoneType = 1,
        maxage: int = 5,
        minimize: bool = False,
        seed: int | np.random.SeedSequence | RNG | NoneType = None,
        selection_strategy: SelectionStrategy | NoneType = None,
        vectorized: bool = False,
        n_jobs: int = 1,
    ):
        super().__init__()
        self.fnc = fnc
        self.ranges = np.array(ranges)
        self.dim = getattr(fnc, "dimension", self.ranges.shape[0])
        self.length = length
        self._rng = seed if isinstance(seed, RNG) else np.random.default_rng(seed)
        self.selection_strategy = (
            selection_strategy
            if selection_strategy is not None
            else TournamentSelection()
        )
        self.vectorized = vectorized
        self.n_jobs = n_jobs

        if odd(nPop):
            nPop += 1
        if odd(int(nPop / 2)):
            nPop += 2
        assert nPop % 4 == 0
        assert nPop >= 4

        self._is_symbolic_Function = isinstance(fnc, Function) and fnc.is_symbolic

        self.nPop = nPop
        self.p_c = None
        self.p_m = None
        self._genotypes = None
        self._phenotypes = None
        self._fitness = None
        self._champion: Genom | NoneType = None
        self._celebrate_op = None
        self._minimize = False
        self.elitism = None
        self._state = None
        self._status = None
        self.set_solution_params(
            p_c=p_c,
            p_m=p_m,
            maxiter=maxiter,
            miniter=miniter,
            elitism=elitism,
            maxage=maxage,
            minimize=minimize,
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

        All stochastic operations
        (population initialization, crossover, mutation, selection) must draw from this
        generator rather than the global :mod:`numpy.random` state, so that runs are
        reproducible (via the `seed` constructor argument) and independent instances
        don't interfere with each other's randomness.
        """
        return self._rng

    @property
    def diversity(self) -> float:
        """
        Return a simple measure of the phenotypic diversity of the current population.

        Computed as the mean, over all dimensions, of the per-dimension standard
        deviation of the phenotypes. A value close to zero indicates a converged,
        homogeneous population; this can be used, in addition to champion age, as a
        signal for premature convergence.
        """
        phenotypes = np.asarray(self.phenotypes, dtype=float)
        if phenotypes.size == 0:
            return 0.0
        return float(np.mean(np.std(phenotypes, axis=0)))

    @property
    def nIter(self) -> int:  # pragma: no cover
        """For backwards compatibility. Returns the number of iterations performed."""
        return self.state.n_iter

    @property
    def champion(self) -> Genom:
        """Return the genom of the champion."""
        return self._champion

    @property
    def genotypes(self) -> Iterable:
        """Return the genotypes of the population."""
        return self._genotypes

    @genotypes.setter
    def genotypes(self, value: Iterable) -> None:
        """Set the genotypes of the population."""
        self._genotypes = value
        self._phenotypes = None
        self._fitness = None

    @property
    def phenotypes(self) -> Iterable:
        """Return the phenotypes of the population."""
        if self._phenotypes is None:
            genotypes = self.genotypes
            if genotypes is not None:
                self._phenotypes = self.decode(genotypes)
        return self._phenotypes

    @property
    def fitness(self) -> ndarray:
        """
        Return the actual fitness values of the population.

        Or the fitness of the population described by the argument `phenotypes`.
        """
        if self._fitness is not None:
            return self._fitness

        self._fitness = self.evaluate(self.phenotypes)
        return self._fitness

    @property
    def fittness(self) -> ndarray:  # pragma: no cover
        """Return the fitness values (deprecated alias for :attr:`fitness`)."""
        import warnings

        warnings.warn(
            "'fittness' was a typo and is deprecated; it will be removed in a future version. Use 'fitness' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fitness

    def reset(self) -> "GeneticAlgorithm":
        """
        Reset the solver and return the object.

        Only use it if you want to have a completely clean sheet. Also, the function
        is called for every object at instantiation.

        Note
        ----
        This method resets the internal state of the genetic algorithm, including
        the population and champion. Use this method when you want to start a new
        optimization process from scratch.
        """
        self._evolver = self.evolver()
        self._evolver.send(None)
        self._champion = None
        self._state = OptimizerState()
        self._status = GeneticAlgorithm.Status.INITIALIZED
        return self

    def set_solution_params(self, **kwargs) -> "GeneticAlgorithm":
        """
        Set the hyperparameters of the algorithm.

        Parameters
        ----------
        p_c: float, Optional
            Probability of crossover.
        p_m: float, Optional
            Probability of mutation.
        maxiter: int, Optional
            Maximum number of iterations.
        miniter: int, Optional
            Minimum number of iterations.
        elitism: float or int, Optional
            Determines the portion of the population designated as elite, which automatically survives
            to the next generation. If less than or equal to 1, it specifies a fraction of the population.
            If greater than 1, it indicates the exact number of individuals to be selected as elite.
            A value of 1 assures that the reigning champion is always preserved. To turn this off,
            set the value to None.
        maxage: int, Optional
            Maximum age of the champion.
        minimize: bool, Optional
            If True, the objective function is minimized. Default is False.
        """
        if "p_c" in kwargs:
            self.p_c = kwargs["p_c"]
        if "p_m" in kwargs:
            self.p_m = kwargs["p_m"]
        if "maxiter" in kwargs:
            self.maxiter = kwargs["maxiter"]
        if "miniter" in kwargs:
            self.miniter = kwargs["miniter"]
        if "elitism" in kwargs:
            self.elitism = kwargs["elitism"]
        if "maxage" in kwargs:
            self.maxage = kwargs["maxage"]
        if "minimize" in kwargs:
            self._minimize = kwargs["minimize"]

        if isinstance(self.elitism, (int, float)):
            if self.elitism <= 0:
                raise ValueError("'elitism' must be greater than 0")

            if self.elitism > 1:
                if not isinstance(self.elitism, int):
                    raise ValueError("'elitism' must be an integer if greater than 1")

                if self.elitism >= self.nPop:
                    raise ValueError("'elitism' must be less than 'nPop'")

        if self.miniter > self.maxiter:
            raise ValueError("'maxiter' must be greater than 'miniter'")

        self._celebrate_op = operator.lt if self._minimize else operator.gt

        return self

    def evolver(self) -> Iterable:
        """Return a generator that can be used to manually control evolutions."""
        self.genotypes = self.populate()
        _ = yield
        yield self.genotypes
        while True:
            genotypes = self.select()
            self.genotypes = self.populate(genotypes)
            yield self.genotypes

    def evolve(self, cycles: int = 1) -> Iterable:
        """Perform a certain number of cycles of evolution and return the genotypes."""
        for _ in range(cycles):
            next(self._evolver)
            candidate: Genom = self.best_candidate()
            self._celebrate(candidate)
            self._state.diversity = self.diversity
        return self.genotypes

    def solve(self, recycle: bool = False, **kwargs) -> Genom:
        """
        Solves the problem and returns the champion.

        .. note::
           This class is designed for maximizing the objective function. To minimize it,
           either negate the objective function or pass ``minimize=True`` when instantiating
           the class.

        Parameters
        ----------
        recycle: bool, Optional
            If True, the leftover resources of the previous calculation are the starting
            point of the new solution.
        kwargs: dict
            Additional parameters to be passed to the :func:`set_solution_params`.

        Returns
        -------
        :class:`~sigmaepsilon.math.optimize.ga.Genom`
            The best candidate.

        """
        if not recycle:
            self.reset()
        self.set_solution_params(**kwargs)

        finished = False
        self._status = GeneticAlgorithm.Status.INITIALIZED

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
                    self._status = GeneticAlgorithm.Status.MAX_ITERATIONS_REACHED
                elif self.stopping_criteria():
                    self._status = GeneticAlgorithm.Status.CONVERGED
                self._state.success = True
            else:  # pragma: no cover
                self._state.success = False
        except Exception as e:
            self._state.success = False
            self._state.message = str(e)
            self._status = GeneticAlgorithm.Status.ERROR
            raise e
        finally:
            self._state.stage = self._status.value

        return self.champion

    def evaluate(self, phenotypes: Iterable | None = None) -> ndarray:
        """
        Evaluate the objective for a list of phenotypes.

        If the phenotypes are not explicitly specified, the population at hand
        is evaluated.

        Parameters
        ----------
        phenotypes: Iterable, Optional
            The phenotypes the objective function is to be evaluated for.
            Default is None.

        Note
        ----
        By default, the objective function is called once per individual in a plain
        Python loop. Set ``vectorized=True`` at construction if the objective function
        can consume the whole ``(nPop, dim)`` array of phenotypes at once and return one
        fitness value per row. Alternatively, set ``n_jobs`` to a value other than 1 to
        evaluate individuals in parallel worker processes (requires a picklable
        objective function).
        """
        try:
            phenotypes = self.phenotypes if phenotypes is None else phenotypes
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
        """
        Return the best phenotype from the active population.

        .. note::
           The value returned by this method is the phenotype of the best candidate
           from the active population, but this is not necessarily the best known solution
           to the optimization problem at hand. If you want to get the reignng champion, use
           the :func:`champion` property.
        """
        return self.best_candidate().phenotype

    def best_candidate(self) -> Genom:
        """Return the Genom of the best candidate in the active population.

        .. note::
           The value returned by this method is the Genom of the best candidate
           from the active population, but this is not necessarily the best known solution
           to the optimization problem at hand. If you want to get the reignng champion, use
           the :func:`champion` property.

        """
        fitness = self.fitness
        argfunc = np.argmin if self._minimize else np.argmax
        index = argfunc(fitness)
        return Genom(
            phenotype=self.phenotypes[index],
            genotype=self.genotypes[index],
            fitness=fitness[index],
            index=index,
        )

    def _celebrate(self, genom: Genom) -> None:
        """Celebrate the winner.

        Currently, this means that the best candidate is added to a history
        to keep track of the improvements across evolutions.
        """
        if self.champion is None:
            self._champion = genom
            self._champion.age = 0
        else:
            has_new_champion = self._celebrate_op(genom, self.champion)
            if has_new_champion:
                self._champion = genom
                self._champion.age = 0
        self.state.x = self.champion.phenotype
        self.state.fun = self.champion.fitness
        self._champion.age += 1

    def divide(self, fitness: ndarray | None = None) -> tuple[ndarray, ndarray]:
        """Divide population to elit and others.

        Returns the corresponding index arrays.

        Parameters
        ----------
        fitness: numpy.ndarray, Optional
            Fitness values. If not provided, values from the latest
            evaluation are used. Default is None.

        Returns
        -------
        list
            Indices of the members of the elite.
        list
            Indices of the members of the others.
        """
        fitness = self.fitness if fitness is None else fitness
        assert fitness is not None, "No available fitness data detected."

        if self.elitism is None:
            return np.array([], dtype=int), np.arange(self.nPop)

        if self.elitism is not None:
            argsort = np.argsort(fitness)
            if not self._minimize:
                argsort = argsort[::-1]

        if self.elitism < 1:
            elit = argsort[: int(self.nPop * self.elitism)]
            others = argsort[int(self.nPop * self.elitism) :]
        else:
            elit = argsort[: self.elitism]
            others = argsort[self.elitism :]

        return elit, others

    def random_parents_generator(self, genotypes: ndarray) -> Generator:
        """Yield random pairs from a list of genotypes.

        The implemantation assumes that the length of the input array
        is a multiple of 2.

        Parameters
        ----------
        genotypes: numpy.ndarray
            Genotypes of the parents as a 2d array.

        Yields
        ------
        numpy.ndarray
            The first parent.
        numpy.ndarray
            The second parent.
        """
        n = len(genotypes)
        assert n % 2 == 0, "'n' must be a multiple of 2"
        pool = np.full(n, True)
        while True:
            where = np.argwhere(pool == True).flatten()
            if len(where) < 2:
                break
            pair = self.rng.choice(where, 2, replace=False)
            parent1 = genotypes[pair[0]]
            parent2 = genotypes[pair[1]]
            pool[pair] = False
            yield parent1, parent2

    def stopping_criteria(self) -> bool:
        """Implement a simple stopping criterion.

        Return True if the current champion is thought to be the best solution
        and no further progress can be made, or only at a slow rate.

        The default implementation considers a champion as the winner, if it is the champion
        for for at least 5 times in a row. This can be dontrolled with the `maxage` parameter
        when instantiating an instance.
        """
        return self.champion.age > self.maxage

    def encode(self, phenotypes: ndarray | None = None) -> ndarray:
        """Turn phenotypes into genotypes.

        The default implementation is the identity mapping (genotype ==
        phenotype), suitable for representations that don't need a separate
        encoding, e.g. real-valued genotypes. Override for representations
        that do, e.g. binary encoding.
        """
        return phenotypes

    def decode(self, genotypes: ndarray) -> ndarray:
        """Turn genotypes into phenotypes.

        The default implementation is the identity mapping (phenotype ==
        genotype), suitable for representations that don't need a separate
        decoding, e.g. real-valued genotypes. Override for representations
        that do, e.g. binary encoding.
        """
        return genotypes

    def populate(self, genotypes: ndarray | None = None) -> ndarray:
        """Produce a pool of genotypes.

        This is a representation-specific extension point with no
        meaningful generic default; override it in a subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement 'populate'."
        )

    def crossover(self, parent1: ndarray, parent2: ndarray) -> Tuple[ndarray]:
        """Take in two parents, return two offspring.

        You'd probably want to use it inside the populator. This is a
        representation-specific extension point with no meaningful generic
        default; override it in a subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement 'crossover'."
        )

    def mutate(self, child: ndarray) -> ndarray:
        """Take a child in, return a mutant.

        This is a representation-specific extension point with no
        meaningful generic default; override it in a subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement 'mutate'.")

    def select(
        self, genotypes: ndarray | None = None, phenotypes: ndarray | None = None
    ) -> ndarray:
        """Run :attr:`selection_strategy` over the current population's fitness values.

        Returns the genotypes of the winners.

        .. note::
           Providing either ``genotypes`` or ``phenotypes`` (or both) explicitly is not
           currently supported and raises a ``NotImplementedError``; only the default
           case (both None, operating on the current population) is implemented.
        """
        if (genotypes is not None) or (phenotypes is not None):
            raise NotImplementedError(
                "Selection with either 'genotypes' or 'phenotypes' (or both) provided is "
                "not implemented. This branch is reached when at least one of these "
                "arguments is given to 'select', but only the default case (both None) "
                "is currently supported."
            )
        fitness = self.fitness
        genotypes = self.genotypes
        winner_indices = self.selection_strategy.select(self, fitness)
        return np.asarray([genotypes[w] for w in winner_indices])
