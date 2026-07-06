import numpy as np
from numpy import ndarray

from .bitchromosome import BitChromosomeGeneticAlgorithm

__all__ = ["IntegerGeneticAlgorithm"]


class IntegerGeneticAlgorithm(BitChromosomeGeneticAlgorithm):
    """
    An implementation of a Genetic Algorithm (GA) for problems whose decision variables
    are natively boolean or small bounded integers, e.g. a 0/1 knapsack indicator vector,
    or a handful of discrete levels per variable.

    Like :class:`~sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm`, individuals are
    represented as flat 0/1 bit chromosomes and decoded with the same linear scaling into
    ``ranges``. The only difference is that the decoded value is rounded to the nearest
    integer instead of kept as a continuous float, which is both the semantically correct
    representation for discrete decision variables and cheaper than retaining
    floating-point precision that the problem doesn't need.

    .. note::
       This class is designed for maximizing the objective function. To minimize it, either negate
       the objective function or pass ``minimize=True`` when instantiating the class.

    .. note::
       Choose ``length`` so that ``2 ** length - 1`` comfortably covers the width of the
       widest range. For pure boolean variables (``ranges=[[0, 1]] * dim``), ``length=1``
       is enough and is the most efficient choice: each gene already encodes exactly the
       one bit of information that is needed. For a wider integer range, e.g. ``[0, 6]``
       (7 distinct values), pick ``length`` so that ``2 ** length - 1 >= 6``, e.g.
       ``length=3`` (8 raw levels, rounded down to the 7 needed).

    Parameters
    ----------
    fnc: Callable
        The function to evaluate. It is assumed, that the function expects and N
        number of scalar arguments as a 1d iterable.
    ranges: Iterable
        Integer bounds for each scalar argument to the objective function, e.g.
        ``[[0, 1], [0, 1]]`` for two boolean variables, or ``[[0, 6]]`` for a single
        7-valued discrete variable.
    length: int, Optional
        Chromosome length (bits) per variable. See the note above for sizing guidance.
        Default is 5.
    p_c: float, Optional
        Probability of crossover. Default is 1.
    p_m: float, Optional
        Probability of mutation. Default is 0.2.
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
    :class:`~sigmaepsilon.math.optimize.ga.Genom`
    :class:`~sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm`
    :class:`~sigmaepsilon.math.optimize.bitchromosome.BitChromosomeGeneticAlgorithm`

    Examples
    --------
    Solve a small 0/1 knapsack problem: pick a subset of items (each either included or
    excluded) to maximize total value without exceeding a weight budget.

    >>> import numpy as np
    >>> from sigmaepsilon.math.optimize import IntegerGeneticAlgorithm as IGA
    >>>
    >>> values = np.array([60, 100, 120, 80, 30])
    >>> weights = np.array([10, 20, 30, 15, 5])
    >>> capacity = 50
    >>>
    >>> def knapsack(x):
    ...     total_weight = np.dot(weights, x)
    ...     total_value = np.dot(values, x)
    ...     penalty = 1000 * max(0, total_weight - capacity)
    ...     return total_value - penalty
    >>>
    >>> ranges = [[0, 1]] * len(values)
    >>> iga = IGA(knapsack, ranges, length=1, nPop=50, seed=0)
    >>> champion = iga.solve()
    >>> selection = champion.phenotype

    """

    __slots__ = ()

    def _postprocess_phenotypes(self, phenotypes: ndarray) -> ndarray:
        """
        Rounds the linearly-decoded phenotypes to the nearest integer.
        """
        return np.round(phenotypes).astype(int)
