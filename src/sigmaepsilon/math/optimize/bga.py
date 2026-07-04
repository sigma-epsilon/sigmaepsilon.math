from .bitchromosome import BitChromosomeGeneticAlgorithm

__all__ = ["BinaryGeneticAlgorithm"]


class BinaryGeneticAlgorithm(BitChromosomeGeneticAlgorithm):
    """
    An implementation of a Binary Genetic Algorithm (BGA) for finding
    minimums of real valued unconstrained problems of continuous variables
    in n-dimensional vector spaces.

    The class is able to solve unconstrained optimization problems of the form:

    .. math::

        \\begin{eqnarray}
            & maximize&  \\quad  f(\\mathbf{x}) \\quad in \\quad \\mathbf{x} \\in \\mathbf{R}^n.
        \\end{eqnarray}

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
    elitism: float or int, Optional
        Determines the portion of the population designated as elite, which automatically survives
        to the next generation. If less than or equal to 1, it specifies a fraction of the population.
        If greater than 1, it indicates the exact number of individuals to be selected as elite.
        The default value of 1 assures that the reigning champion is always preserved. To turn this off,
        det the value to None. Default is 1.
    ftol: float, Optional
        Torelance for floating point operations. Default is 1e-12.
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
    :class:`~sigmaepsilon.math.optimize.rga.RealValuedGeneticAlgorithm`
    :class:`~sigmaepsilon.math.optimize.iga.IntegerGeneticAlgorithm`
    :class:`~sigmaepsilon.math.optimize.bitchromosome.BitChromosomeGeneticAlgorithm`

    Examples
    --------
    Find the minimizer of the Rosenbrock function.
    The exact value of the solution is x = [1.0, 1.0].

    >>> from sigmaepsilon.math.optimize import BinaryGeneticAlgorithm as BGA
    >>>
    >>> def rosenbrock(x):
    ...     a, b = 1, 100
    ...     return (a-x[0])**2 + b*(x[1]-x[0]**2)**2
    >>>
    >>>
    >>> ranges = [[-10, 10], [-10, 10]]
    >>> bga = BGA(rosenbrock, ranges, length=12, nPop=100, minimize=True)
    >>> _ = bga.solve()
    >>> champion = bga.champion
    >>> x = champion.phenotype
    >>> fx = champion.fitness

    """

    __slots__ = ()
