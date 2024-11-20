import numpy as np
from numpy import ndarray

from .ga import GeneticAlgorithm

__all__ = ["BinaryGeneticAlgorithm"]


class BinaryGeneticAlgorithm(GeneticAlgorithm):
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
        The age is the number of generations a candidate spends at the top
        (being the best candidate). Setting an upper limit to this value is a kind
        of stopping criterion. Default is 5.
    minimize: bool, Optional
        If True, the objective function is minimized. Default is False.

    See Also
    --------
    :class:`~sigmaepsilon.math.optimize.ga.GeneticAlgorithm`
    :class:`~sigmaepsilon.math.optimize.ga.Genom`

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
    >>> fx = champion.fittness

    """

    def populate(self, genotypes: ndarray | None = None) -> ndarray:
        """
        Populates the model and returns the array of genotypes.
        """
        nPop = self.nPop

        if genotypes is None:
            poolshape = (int(nPop / 2), self.dim * self.length)
            genotypes = np.random.randint(2, size=poolshape)
        else:
            poolshape = genotypes.shape

        nParent = poolshape[0]

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

    def decode(self, genotypes: ndarray) -> ndarray:
        """
        Decodes the genotypes to phenotypes and returns them as an array.
        """
        span = 2**self.length - 2**0
        genotypes = genotypes.reshape((self.nPop, self.dim, self.length))
        precisions = [
            (self.ranges[d, -1] - self.ranges[d, 0]) / span for d in range(self.dim)
        ]
        phenotypes = np.sum(
            [genotypes[:, :, i] * 2**i for i in range(self.length)], axis=0
        ).astype(float)
        for d in range(self.dim):
            phenotypes[:, d] *= precisions[d]
            phenotypes[:, d] += self.ranges[d, 0]
        return phenotypes

    def crossover(
        self, parent1: ndarray, parent2: ndarray, nCut: int | None = None
    ) -> tuple[ndarray, ndarray]:
        """
        Performs crossover on the parents `parent1` and `parent2`,
        using an `nCut` number of cuts and returns two childs.
        """
        if np.random.rand() > self.p_c:  # pragma: no cover
            return parent1, parent2

        if nCut is None:
            nCut = np.random.randint(1, self.dim * self.length - 1)

        cuts = [0, self.dim * self.length]
        p = np.random.choice(range(1, self.length * self.dim - 1), nCut, replace=False)
        cuts.extend(p)
        cuts = np.sort(cuts)

        child1 = np.zeros(self.dim * self.length, dtype=int)
        child2 = np.zeros(self.dim * self.length, dtype=int)

        randBool = np.random.rand() > 0.5
        for i in range(nCut + 1):
            if (i % 2 == 0) == randBool:
                child1[cuts[i] : cuts[i + 1]] = parent1[cuts[i] : cuts[i + 1]]
                child2[cuts[i] : cuts[i + 1]] = parent2[cuts[i] : cuts[i + 1]]
            else:
                child1[cuts[i] : cuts[i + 1]] = parent2[cuts[i] : cuts[i + 1]]
                child2[cuts[i] : cuts[i + 1]] = parent1[cuts[i] : cuts[i + 1]]

        return self.mutate(child1), self.mutate(child2)

    def mutate(self, child: ndarray) -> ndarray:
        """
        Returns a mutated genotype. Children come in, mutants go out.
        """
        p = np.random.rand(self.dim * self.length)
        return np.where(p > self.p_m, child, 1 - child)

    def select(self, genotypes: ndarray, phenotypes: ndarray | None = None) -> ndarray:
        """
        Organizes a tournament and returns the genotypes of the winners.
        """
        fittness = self.evaluate(phenotypes)
        winners, others = self.divide(fittness)
        winners = winners.tolist()
        while len(winners) < int(self.nPop / 2):
            candidates = np.random.choice(others, 3, replace=False)
            argsort = np.argsort([fittness[ID] for ID in candidates])
            winner = argsort[0] if self._minimize else argsort[-1]
            winners.append(candidates[winner])
        return np.array([genotypes[w] for w in winners], dtype=float)
