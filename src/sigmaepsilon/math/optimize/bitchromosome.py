"""Shared base class for bit-chromosome-encoded genetic algorithms."""

import numpy as np
from numpy import ndarray

from .ga import GeneticAlgorithm

__all__ = ["BitChromosomeGeneticAlgorithm"]


class BitChromosomeGeneticAlgorithm(GeneticAlgorithm):
    """Shared base for genetic algorithms whose genotype is a flat 0/1 bit chromosome.

    The chromosome has length ``dim * length``, decoded into ``dim`` scalar phenotypes
    by linearly scaling each variable's ``length``-bit segment into its ``ranges[d]``
    bounds.

    This class provides the representation-specific machinery (:func:`populate`,
    :func:`crossover`, :func:`mutate`, :func:`decode`) for any bit-chromosome GA. What
    varies between concrete subclasses is only how the linearly-decoded scalar is
    interpreted, which is controlled by the :func:`_postprocess_phenotypes` hook:

    - :class:`~sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm` keeps the
      continuous, floating point value (suitable for real-valued, box-constrained
      optimization).
    - :class:`~sigmaepsilon.math.optimize.iga.IntegerGeneticAlgorithm` rounds it to the
      nearest integer (suitable for boolean or small-integer-domain decision variables).

    .. note::
       This class is not meant to be instantiated directly; use one of its concrete
       subclasses instead.
    """

    __slots__ = ()

    def populate(self, genotypes: ndarray | None = None) -> ndarray:
        """Populate the model and return the array of genotypes."""
        nPop = self.nPop

        if genotypes is None:
            poolshape = (int(nPop / 2), self.dim * self.length)
            genotypes = self.rng.integers(2, size=poolshape)
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
        """Decode the genotypes to phenotypes and return them as an array."""
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
        return self._postprocess_phenotypes(phenotypes)

    def _postprocess_phenotypes(self, phenotypes: ndarray) -> ndarray:
        """Transform the linearly-decoded phenotypes for subclasses.

        This is a hook that transforms the continuous phenotypes into the
        representation subclasses actually want to expose (e.g. rounded to the
        nearest integer). The default implementation is the identity (continuous).
        """
        return phenotypes

    def crossover(
        self, parent1: ndarray, parent2: ndarray, nCut: int | None = None
    ) -> tuple[ndarray, ndarray]:
        """Perform crossover on the parents.

        Crosses `parent1` and `parent2` using an `nCut` number of cuts and returns
        two children.
        """
        if self.rng.random() > self.p_c:  # pragma: no cover
            return parent1, parent2

        if nCut is None:
            nCut = self.rng.integers(1, self.dim * self.length - 1)

        cuts = [0, self.dim * self.length]
        p = self.rng.choice(
            np.arange(1, self.length * self.dim - 1), nCut, replace=False
        )
        cuts.extend(p)
        cuts = np.sort(cuts)

        child1 = np.zeros(self.dim * self.length, dtype=int)
        child2 = np.zeros(self.dim * self.length, dtype=int)

        randBool = self.rng.random() > 0.5
        for i in range(nCut + 1):
            if (i % 2 == 0) == randBool:
                child1[cuts[i] : cuts[i + 1]] = parent1[cuts[i] : cuts[i + 1]]
                child2[cuts[i] : cuts[i + 1]] = parent2[cuts[i] : cuts[i + 1]]
            else:
                child1[cuts[i] : cuts[i + 1]] = parent2[cuts[i] : cuts[i + 1]]
                child2[cuts[i] : cuts[i + 1]] = parent1[cuts[i] : cuts[i + 1]]

        return self.mutate(child1), self.mutate(child2)

    def mutate(self, child: ndarray) -> ndarray:
        """Return a mutated genotype. Children come in, mutants go out."""
        p = self.rng.random(self.dim * self.length)
        return np.where(p > self.p_m, child, 1 - child)
