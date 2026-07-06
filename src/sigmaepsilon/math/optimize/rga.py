import numpy as np
from numpy import ndarray

from .ga import GeneticAlgorithm

__all__ = ["RealValuedGeneticAlgorithm"]


class RealValuedGeneticAlgorithm(GeneticAlgorithm):
    """
    A real-valued (continuous) Genetic Algorithm (GA) for finding minimums or maximums
    of unconstrained problems over box-ranges of continuous variables, without
    binary encoding.

    Unlike :class:`~sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm`, individuals
    are represented directly as real-valued vectors (genotype == phenotype), using
    arithmetic (blend) crossover and Gaussian mutation. This avoids the precision/length
    trade-off inherent to binary encoding, at the cost of the different exploration
    behavior of a continuous representation.

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
    p_c: float, Optional
        Probability of crossover. Default is 1.
    p_m: float, Optional
        Probability of mutation (per gene). Default is 0.2.
    mutation_scale: float, Optional
        Standard deviation of the Gaussian mutation noise, expressed as a fraction of
        each variable's range. Default is 0.1.
    nPop: int, Optional
        The size of the population. Default is 100.
    maxiter: int, Optional
        The maximum number of iterations. Default is 200.
    miniter: int, Optional
        The minimum number of iterations. Default is 0.
    elitism: float or int, Optional
        Determines the portion of the population designated as elite, which automatically survives
        to the next generation. If less than or equal to 1, it specifies a fraction of the population.
        If greater than 1, it indicates the exact number of individuals to be selected as elite.
        The default value of 1 assures that the reigning champion is always preserved. To turn this off,
        set the value to None. Default is 1.
    maxage: int, Optional
        The age is the maximum number of generations a candidate spends at the top
        (being the best candidate) before termination. Default is 5.
    minimize: bool, Optional
        If True, the objective function is minimized. Default is False.
    seed: int | numpy.random.SeedSequence | numpy.random.Generator | None, Optional
        A seed for a per-instance random number generator. Default is None.
    selection_strategy: :class:`~sigmaepsilon.math.optimize.selection.SelectionStrategy`, Optional
        The selection strategy used by :func:`select`. Default is
        :class:`~sigmaepsilon.math.optimize.selection.TournamentSelection`.
    vectorized: bool, Optional
        See :func:`~sigmaepsilon.math.optimize.ga.GeneticAlgorithm.evaluate`. Default is False.
    n_jobs: int, Optional
        See :func:`~sigmaepsilon.math.optimize.ga.GeneticAlgorithm.evaluate`. Default is 1.

    See Also
    --------
    :class:`~sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm`
    :class:`~sigmaepsilon.math.optimize.iga.IntegerGeneticAlgorithm`

    Examples
    --------
    Find the minimizer of the Rosenbrock function.
    The exact value of the solution is x = [1.0, 1.0].

    >>> from sigmaepsilon.math.optimize.rga import RealValuedGeneticAlgorithm as RGA
    >>>
    >>> def rosenbrock(x):
    ...     a, b = 1, 100
    ...     return (a-x[0])**2 + b*(x[1]-x[0]**2)**2
    >>>
    >>>
    >>> ranges = [[-10, 10], [-10, 10]]
    >>> rga = RGA(rosenbrock, ranges, nPop=100, minimize=True)
    >>> _ = rga.solve()
    >>> champion = rga.champion
    >>> x = champion.phenotype
    >>> fx = champion.fitness

    """

    __slots__ = ("mutation_scale",)

    def __init__(self, *args, mutation_scale: float = 0.1, **kwargs):
        self.mutation_scale = mutation_scale
        super().__init__(*args, **kwargs)

    def populate(self, genotypes: ndarray | None = None) -> ndarray:
        """
        Populates the model and returns the array of genotypes.
        """
        nPop = self.nPop

        if genotypes is None:
            low = self.ranges[:, 0]
            high = self.ranges[:, 1]
            genotypes = self.rng.uniform(low, high, size=(int(nPop / 2), self.dim))

        nParent = genotypes.shape[0]

        if nParent < nPop:
            offspring = []
            g = self.random_parents_generator(genotypes)
            try:
                while (len(offspring) + nParent) < nPop:
                    parent1, parent2 = next(g)
                    offspring.extend(self.crossover(parent1, parent2))
                genotypes = np.vstack([genotypes, offspring])
            except Exception:  # pragma: no cover
                raise RuntimeError

        return genotypes

    def crossover(self, parent1: ndarray, parent2: ndarray) -> tuple[ndarray, ndarray]:
        """
        Performs arithmetic (blend) crossover on the parents `parent1` and `parent2`
        and returns two children.
        """
        if self.rng.random() > self.p_c:  # pragma: no cover
            return parent1, parent2

        alpha = self.rng.random(self.dim)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1

        return self.mutate(child1), self.mutate(child2)

    def mutate(self, child: ndarray) -> ndarray:
        """
        Returns a mutated genotype. Children come in, mutants go out.
        """
        mask = self.rng.random(self.dim) < self.p_m
        if not np.any(mask):
            return child

        span = self.ranges[:, 1] - self.ranges[:, 0]
        noise = self.rng.normal(scale=self.mutation_scale * span, size=self.dim)
        mutant = np.where(mask, child + noise, child)
        return np.clip(mutant, self.ranges[:, 0], self.ranges[:, 1])
