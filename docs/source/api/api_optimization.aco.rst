.. _api_optimization_ACO:

=========================
Ant Colony Optimization
=========================

**Ant Colony Optimization (ACO)** is a family of metaheuristics inspired by the
foraging behavior of real ants, which lay down pheromone trails that bias the
paths taken by other ants, reinforcing shorter or better paths over time. In
the library, this idea is split into a shared driver,
:class:`~sigmaepsilon.math.optimize.aco.AntColonyOptimization`, and two concrete
strategies that differ in how solutions are represented and how "pheromones"
are stored and used:

* :class:`~sigmaepsilon.math.optimize.aco.ContinuousAntColonyOptimization` (ACOR)
  for continuous, box-constrained real-valued problems.
* :class:`~sigmaepsilon.math.optimize.aco.CombinatorialAntColonyOptimization`
  (Ant System) for permutation problems defined over a distance matrix, such as
  the Traveling Salesman Problem (TSP).

See :doc:`../user_guide/optimization/aco` for an explanation of how the two
algorithms work, and worked examples for both.

.. autoclass:: sigmaepsilon.math.optimize.aco.AntSolution
    :members:

.. autoclass:: sigmaepsilon.math.optimize.aco.AntColonyOptimization
    :members:

.. autoclass:: sigmaepsilon.math.optimize.aco.ContinuousAntColonyOptimization
    :members:

.. autoclass:: sigmaepsilon.math.optimize.aco.CombinatorialAntColonyOptimization
    :members:
