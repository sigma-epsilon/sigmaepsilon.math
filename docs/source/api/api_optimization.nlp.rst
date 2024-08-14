===========================
Nonlinear Programming (NLP)
===========================

Genetic Algorithms (GA)
=======================

The :class:`~sigmaepsilon.math.optimize.ga.GeneticAlgorithm` class provides a skeleton for the implementation of
custom GAs, and the :class:`~sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm` is the standard implementation of it.

For a good explanation of how Genetic Algorithms work, read 
`this <https://www.mathworks.com/help/gads/how-the-genetic-algorithm-works.html>`_
from 
`MathWorks <https://www.mathworks.com/?s_tid=gn_logo>`_.

.. autoclass:: sigmaepsilon.math.optimize.ga.Genom
    :members:

.. autoclass:: sigmaepsilon.math.optimize.ga.GeneticAlgorithm
    :members: solve, populate, decode, mutate, crossover, select

Binary Genetic Algorithm (BGA)
------------------------------

.. autoclass:: sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm
    :members: populate, decode, mutate, crossover, select
