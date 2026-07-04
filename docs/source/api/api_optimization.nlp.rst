.. _api_optimization_NLP:

===========================
Nonlinear Programming (NLP)
===========================

**Nonlinear programming (NLP)** is a subset of optimization where the objective 
function or constraints are nonlinear. Unlike linear programming, where relationships 
between variables are linear, NLP deals with more complex systems where variables may 
interact in intricate ways, resulting in non-straightforward solutions. NLP is used 
in a variety of fields such as economics, engineering, machine learning, and 
operations research, where real-world problems often exhibit nonlinear behaviors. 
The goal of NLP is to find the best possible solution (such as maximum profit or 
minimum cost) subject to given constraints.

.. _api_optimization_GA:

Genetic Algorithms (GA)
=======================

**Genetic algorithms (GAs)** are a type of optimization algorithm inspired by the 
principles of natural selection and genetics. GAs work by iteratively evolving a 
population of potential solutions to a problem through processes like selection, 
crossover (recombination), and mutation. Each individual solution is represented 
as a "chromosome," and better solutions are evolved over generations by selecting 
and breeding the fittest individuals. GAs are particularly useful for solving complex, 
nonlinear, or discrete optimization problems where traditional methods may struggle. 
They are widely applied in fields such as artificial intelligence, engineering, and 
economics.

For a good explanation of how Genetic Algorithms work, read 
`this <https://www.mathworks.com/help/gads/how-the-genetic-algorithm-works.html>`_
from 
`MathWorks <https://www.mathworks.com/?s_tid=gn_logo>`_.

.. autoclass:: sigmaepsilon.math.optimize.ga.GeneticAlgorithm

.. autoclass:: sigmaepsilon.math.optimize.ga.Genom
    :members:

Selection strategies
---------------------

The winners of each generation (beyond the automatically-surviving elite) are picked by
a pluggable :class:`~sigmaepsilon.math.optimize.selection.SelectionStrategy`, injected
via the `selection_strategy` constructor argument of
:class:`~sigmaepsilon.math.optimize.ga.GeneticAlgorithm` and its subclasses.

.. autoclass:: sigmaepsilon.math.optimize.selection.SelectionStrategy
    :members:

.. autoclass:: sigmaepsilon.math.optimize.selection.TournamentSelection
    :members:

.. autoclass:: sigmaepsilon.math.optimize.selection.RouletteSelection
    :members:

.. autoclass:: sigmaepsilon.math.optimize.selection.RankSelection
    :members:

.. _api_optimization_BGA:

Binary Genetic Algorithm (BGA)
------------------------------

**Binary genetic algorithms (BGA)** are a specific type of genetic algorithm where
each solution is encoded as a string of binary digits (0s and 1s). These binary
strings, known as genotypes (or chromosomes), represent the decision variables in the problem.
Through genetic operations like selection, crossover, and mutation, BGAs evolve a
population of solutions over time to find the best possible outcome. This approach
is particularly well-suited for optimization problems where variables naturally lend
themselves to binary encoding, such as combinatorial optimization and certain
engineering design tasks.

BGA and :class:`~sigmaepsilon.math.optimize.iga.IntegerGeneticAlgorithm` (below) share
the same underlying bit-chromosome machinery, provided by
:class:`~sigmaepsilon.math.optimize.bitchromosome.BitChromosomeGeneticAlgorithm`; they
only differ in whether the decoded value is kept as a continuous float (BGA) or rounded
to the nearest integer (IGA).

.. autoclass:: sigmaepsilon.math.optimize.bitchromosome.BitChromosomeGeneticAlgorithm

.. autoclass:: sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm

.. _api_optimization_IGA:

Integer Genetic Algorithm (IGA)
--------------------------------

**Integer genetic algorithms (IGA)** are for problems whose decision variables are
natively boolean or small bounded integers (e.g. a 0/1 knapsack indicator vector, or a
handful of discrete levels per variable), instead of continuous, box-constrained
real numbers. IGA reuses BGA's bit-chromosome representation and genetic operators, but
rounds the decoded value to the nearest integer, which is both the semantically correct
representation for discrete variables and cheaper than retaining unneeded floating-point
precision.

.. autoclass:: sigmaepsilon.math.optimize.iga.IntegerGeneticAlgorithm

.. _api_optimization_RGA:

Real-Valued Genetic Algorithm (RGA)
------------------------------------

**Real-valued genetic algorithms (RGA)** operate directly on continuous, real-valued
chromosomes instead of a binary encoding, using arithmetic (blend) crossover and
Gaussian mutation. This avoids the precision/chromosome-length trade-off of BGA, at the
cost of a different exploration behavior.

.. autoclass:: sigmaepsilon.math.optimize.rga.RealValuedGeneticAlgorithm
