from abc import abstractmethod
from typing import Iterable, Callable, Tuple, Generator
import numpy as np
from numpy import ndarray
from pydantic import BaseModel, Field

from ..function import Function

__all__ = ["GeneticAlgorithm", "Genom"]


def even(n):
    return n % 2 == 0


def odd(n):
    return not even(n)


class Genom(BaseModel):
    """
    A data class for members of a population.
    """

    phenotype: list[float] = Field(default_factory=list)
    genotype: list[int] = Field(default_factory=list)
    fittness: float
    age: int = Field(default=0)
    index: int = Field(default=-1)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Genom):
            return False
        return np.all(self.genotype == other.genotype)

    def __hash__(self):
        arr_string = "".join(str(i) for i in self.genotype)
        return hash(arr_string)

    def __gt__(self, other):
        if not isinstance(other, Genom):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return np.all(self.fittness > other.fittness)

    def __lt__(self, other):
        if not isinstance(other, Genom):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return np.all(self.fittness < other.fittness)

    def __ge__(self, other):
        if not isinstance(other, Genom):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return np.all(self.fittness >= other.fittness)

    def __le__(self, other):
        if not isinstance(other, Genom):
            raise TypeError(
                f"This operation is not supported between instances of {type(other)} and {type(self)}."
            )
        return np.all(self.fittness <= other.fittness)


class GeneticAlgorithm:
    """
    Base class for Genetic Algorithms (GA). Use this as a base
    class to your custom implementation of a GA.

    The class has 4 abstract methods wich upon being implemented, yield
    a working genetic algorithm. These are :func:`populate`, :func:`decode`,
    :func:`crossover`, :func:`mutate` and :func:`select`. It is also possible to
    use a custom stopping criteria by implementing :func:`stopping_criteria`.
    See the :class:`~sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm` class for an example.

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
    elitism: float or int, Optional
        Default is 1
    ftol: float, Optional
        Torelance for floating point operations. Default is 1e-12.
    maxage: int, Optional
        The age is the number of generations a candidate spends at the top
        (being the best candidate). Setting an upper limit to this value is a kind
        of stopping criterion. Default is 5.

    Note
    ----
    Be cautious what you use a genetic algorithm for. Like all metahauristic methods, a
    genetic algorithm can be wery demanding on the computational side. If the objective
    function takes a lot of time to evaluate, it is probably not a good idea to use a heuristic
    approach, unless you have a dedicated evaluator that is able to run efficiently for a large
    number of problems. If you want to customize the way the objective is evaluated, override
    the :func:`evaluate` method.

    See also
    --------
    :class:`~sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm`
    """

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
        "_fittness",
        "_champion",
        "_evolver",
        "maxiter",
        "miniter",
        "elitism",
        "maxage",
        "nIter",
        "_is_symbolic_Function",
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
        elitism: int = 1,
        maxage: int = 5,
    ):
        super().__init__()
        self.fnc = fnc
        self.ranges = np.array(ranges)
        self.dim = getattr(fnc, "dimension", self.ranges.shape[0])
        self.length = length

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
        self._pnenotypes = None
        self._fittness = None
        self._champion: Genom | None = None
        self.set_solution_params(
            p_c=p_c,
            p_m=p_m,
            maxiter=maxiter,
            miniter=miniter,
            elitism=elitism,
            maxage=maxage,
        )
        self.reset()

    @property
    def champion(self) -> Genom:
        """
        Returnes the genotypes of the population.
        """
        return self._champion

    @property
    def genotypes(self) -> Iterable:
        """
        Returnes the genotypes of the population.
        """
        return self._genotypes

    @genotypes.setter
    def genotypes(self, value: Iterable):
        """
        Sets the genotypes of the population.
        """
        self._genotypes = value
        self._phenotypes = None
        self._fittness = None

    @property
    def phenotypes(self) -> Iterable:
        """
        Returnes the phenotypes of the population.
        """
        if self._phenotypes is None:
            genotypes = self.genotypes
            if genotypes is not None:
                self._phenotypes = self.decode(genotypes)
        return self._phenotypes

    @property
    def fittness(self) -> ndarray:
        """
        Returns the actual fittness values of the population, or the fittness
        of the population described by the argument `phenotypes`.
        """
        if self._fittness is None:
            self._fittness = self.evaluate(self.phenotypes)
        return self._fittness

    def reset(self) -> "GeneticAlgorithm":
        """
        Resets the solver and returns the object. Only use it if you want to have
        a completely clean sheet. Also, the function is called for every object at
        instantiation.
        """
        self._evolver = self.evolver()
        self._evolver.send(None)
        self._champion = None
        return self

    def set_solution_params(
        self,
        *,
        p_c: float | None = None,
        p_m: float | None = None,
        maxiter: int | None = None,
        miniter: int | None = None,
        elitism: int | None = None,
        maxage: int | None = None,
    ) -> "GeneticAlgorithm":
        """
        Sets the hyperparameters of the algorithm.

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
        elitism: int, Optional
            Elitism.
        maxage: int, Optional
            Maximum age of the champion.
        """
        if p_c is not None:
            self.p_c = p_c
        if p_m is not None:
            self.p_m = p_m
        if maxiter is not None:
            self.maxiter = maxiter
        if miniter is not None:
            self.miniter = miniter
        if elitism is not None:
            self.elitism = elitism
        if maxage is not None:
            self.maxage = maxage

        if self.miniter > self.maxiter:
            raise ValueError("'maxiter' must be greater than 'miniter'")

        return self

    def evolver(self) -> Iterable:
        """
        Returns a generator that can be used to manually control evolutions.
        """
        self.genotypes = self.populate()
        _ = yield
        yield self.genotypes
        while True:
            genotypes = self.select(self.genotypes, self.phenotypes)
            self.genotypes = self.populate(genotypes)
            yield self.genotypes

    def evolve(self, cycles: int = 1) -> Iterable:
        """
        Performs a certain number of cycles of evolution and returns the
        genotypes.
        """
        for _ in range(cycles):
            next(self._evolver)
        return self.genotypes

    def solve(self, recycle: bool = False, **kwargs) -> Genom:
        """
        Solves the problem and returns the best phenotype.

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
        self.reset() if not recycle else None
        self.set_solution_params(**kwargs)

        nIter, finished = 0, False

        while (not finished and nIter < self.maxiter) or (nIter < self.miniter):
            self.evolve()
            candidate: Genom = self.best_candidate()
            self._celebrate(candidate)
            finished = self.stopping_criteria()
            nIter += 1

        self.nIter = nIter
        return self.champion

    def evaluate(self, phenotypes: Iterable | None = None) -> ndarray:
        """
        Evaluates the objective for a list of phenotypes.

        If the phenotypes are not explicitly specified, the population at hand
        is evaluated.

        Parameters
        ----------
        phenotypes: Iterable, Optional
            The phenotypes the objective function is to be evaluated for.
            Default is None.
        """
        phenotypes = self.phenotypes if phenotypes is None else phenotypes
        if self._is_symbolic_Function:
            return self.fnc(phenotypes.T)
        else:
            return np.array([self.fnc(x) for x in phenotypes], dtype=float)

    def best_phenotype(self) -> ndarray:
        """
        Returns the best phenotype.

        Parameters
        ----------
        lastknown: bool, Optional
            If True, the last evaluation is used. If False, the phenotypes
            are evaluated before selecting the best. In this case the results
            are not stored. Default is True.
        """
        return self.best_candidate().phenotype

    def best_candidate(self) -> Genom:
        """
        Returns data about the best candidate in the population like index,
        phenotype, genotype and fittness value.

        Parameters
        ----------
        lastknown: bool, Optional
            If True, the last evaluation is used. If False, the phenotypes
            are evaluated before selecting the best. In this case the results
            are not stored. Default is True.
        """
        fittness = self.fittness
        index = np.argmin(fittness)
        return Genom(
            phenotype=self.phenotypes[index],
            genotype=self.genotypes[index],
            fittness=fittness[index],
            index=index,
        )

    def _celebrate(self, genom: Genom) -> None:
        """
        Celebration of the winner. Curretly this means that the beast candidate is added
        to a history to keep track of the improvements across evolutions.
        """
        if self.champion is None:
            self._champion = genom
            self._champion.age = 0
        else:
            if genom > self.champion:
                self._champion = genom
                self._champion.age = 0
        self._champion.age += 1

    def divide(self, fittness: ndarray | None = None) -> Tuple[list]:
        """
        Divides population to elit and others, and returns the corresponding
        index arrays.

        Parameters
        ----------
        fittness: numpy.ndarray, Optional
            Fittness values. If not provided, values from the latest
            evaluation are used. Default is None.

        Returns
        -------
        list
            Indices of the members of the elite.
        list
            Indices of the members of the others.
        """
        fittness = self.fittness if fittness is None else fittness
        assert fittness is not None, "No available fittness data detected."
        if self.elitism < 1:
            argsort = np.argsort(fittness)
            elit = argsort[: int(self.nPop * self.elitism)]
            others = argsort[int(self.nPop * self.elitism) :]
        elif self.elitism > 1:
            argsort = np.argsort(fittness)
            elit = argsort[: self.elitism]
            others = argsort[self.elitism :]
        else:
            elit = []
            others = list(range(self.nPop))
        return list(elit), others

    @classmethod
    def random_parents_generator(cls, genotypes: ndarray) -> Generator:
        """
        Yields random pairs from a list of genotypes.

        The implemantation assumes that the length of the input array
        is a multiple of 2.

        Parameters
        ----------
        genotypes: numpy.ndarray
            Genotypes of the parents as a 2d integer array.

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
        nPool = n
        while nPool > 2:
            where = np.argwhere(pool == True).flatten()
            nPool = len(where)
            pair = np.random.choice(where, 2, replace=False)
            parent1 = genotypes[pair[0]]
            parent2 = genotypes[pair[1]]
            pool[pair] = False
            yield parent1, parent2

    @abstractmethod
    def stopping_criteria(self) -> bool:
        """
        Implements a simple stopping criteria that evaluates to `True` if the
        current chanpion is thought ti bee the best solution and no further progress
        can be made, or at lest with a bad rate.

        The default implementation considers a champion as the winner, if it is the champion
        for for at least 5 times in a row. This can be dontrolled with the `maxage` parameter
        when instantiating an instance.
        """
        return self.champion.age > self.maxage

    @abstractmethod
    def encode(self, phenotypes: ndarray | None = None) -> ndarray:
        """
        Turns phenotypes into genotypes.
        """
        return phenotypes

    @abstractmethod
    def decode(self, genotypes: ndarray) -> ndarray:
        """
        Turns genotypes into phenotypes.
        """
        return genotypes

    @abstractmethod
    def populate(self, genotypes: ndarray | None = None) -> ndarray:
        """
        Ought to produce a pool of phenotypes.
        """
        ...

    @abstractmethod
    def crossover(self, parent1: ndarray, parent2: ndarray) -> Tuple[ndarray]:
        """
        Takes in two parents, returns two offspring. You'd probably want to use it inside
        the populator.
        """
        ...

    @abstractmethod
    def mutate(self, child: ndarray) -> ndarray:
        """
        Takes a child in, returns a mutant.
        """
        ...

    @abstractmethod
    def select(self, genotypes: ndarray, phenotypes: ndarray | None = None) -> ndarray:
        """
        Ought to implement dome kind of selection mechanism, eg. a roulette wheel,
        tournament or other.
        """
        ...
