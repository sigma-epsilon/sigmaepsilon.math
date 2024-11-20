from abc import abstractmethod
from typing import Iterable, Callable, Tuple, Generator
from types import NoneType
import operator

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
        The age is the number of generations a candidate spends at the top
        (being the best candidate). Setting an upper limit to this value is a kind
        of stopping criterion. Default is 5.
    minimize: bool, Optional
        If True, the objective function is minimized. Default is False.

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
        "_celebrate_op",
        "_minimize",
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
        self._champion: Genom | NoneType = None
        self._celebrate_op = None
        self._minimize = False
        self.elitism = None
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

    def set_solution_params(self, **kwargs) -> "GeneticAlgorithm":
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
            candidate: Genom = self.best_candidate()
            self._celebrate(candidate)
        return self.genotypes

    def solve(self, recycle: bool = False, **kwargs) -> Genom:
        """
        Solves the problem and returns the best phenotype.

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
        self.reset() if not recycle else None
        self.set_solution_params(**kwargs)

        nIter, finished = 0, False

        while (not finished and nIter < self.maxiter) or (nIter < self.miniter):
            self.evolve(1)
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
        Returns the best phenotype from the active population.

        .. note::
           The value returned by this method is the phenotype of the best candidate
           from the active population, but this is not necessarily the best known solution
           to the optimization problem at hand. If you want to get the reignng champion, use
           the :func:`champion` property.
        """
        return self.best_candidate().phenotype

    def best_candidate(self) -> Genom:
        """
        Returns the Genom of the best candidate in the active population.

        .. note::
           The value returned by this method is the Genom of the best candidate
           from the active population, but this is not necessarily the best known solution
           to the optimization problem at hand. If you want to get the reignng champion, use
           the :func:`champion` property.

        """
        fittness = self.fittness
        argfunc = np.argmin if self._minimize else np.argmax
        index = argfunc(fittness)
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
            has_new_champion = self._celebrate_op(genom, self.champion)
            if has_new_champion:
                self._champion = genom
                self._champion.age = 0
        self._champion.age += 1

    def divide(self, fittness: ndarray | None = None) -> tuple[ndarray, ndarray]:
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

        if self.elitism is None:
            return [], list(range(self.nPop))

        if self.elitism is not None:
            argsort = np.argsort(fittness)
            if not self._minimize:
                argsort = argsort[::-1]

        if self.elitism < 1:
            elit = argsort[: int(self.nPop * self.elitism)]
            others = argsort[int(self.nPop * self.elitism) :]
        else:
            elit = argsort[: self.elitism]
            others = argsort[self.elitism :]

        return elit, others

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
