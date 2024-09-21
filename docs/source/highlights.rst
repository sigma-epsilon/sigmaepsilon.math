* :ref:`Linear Algebra <user_guide_linalg>`

  * A :class:`~sigmaepsilon.math.linalg.ReferenceFrame` class for all kinds of frames, and dedicated :class:`~sigmaepsilon.math.linalg.RectangularFrame` and :class:`~sigmaepsilon.math.linalg.CartesianFrame` 
    classes as special cases, all NumPy compliant.
  * NumPy compliant classes like :class:`~sigmaepsilon.math.linalg.Tensor` and :class:`~sigmaepsilon.math.linalg.Vector` to handle various kinds of tensorial 
    quantities efficiently with a built-in mechanism that guarantees to maintain the property of objectivity.
  * A :class:`~sigmaepsilon.math.linalg.sparse.JaggedArray` and a Numba-jittable :class:`~sigmaepsilon.math.linalg.sparse.csr_matrix` to handle sparse data.

* :ref:`Optimization <user_guide_optimization>`

  * Classes to define and solve linear and nonlinear optimization problems.
    
    * A :class:`~sigmaepsilon.math.optimize.LinearProgrammingProblem` class to define and solve continuous, integer or mixed-integer linear optimization problems.
    * A :class:`~sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm` class to tackle more complicated optimization problems.

* :ref:`Appriximation <user_guide_approximation>`

  * Several methods and classes to approximate functions and data, including a :class:`~sigmaepsilon.math.approx.mls.MLSApproximator` for multilinear regression
    using the moving least squares method.

* :ref:`Graph Theory <user_guide_graph>`

  * Algorithms to calculate rooted level structures and pseudo peripheral nodes of a graph, and a :class:`~sigmaepsilon.math.graph.graph.Graph` class
    that extends ``networkx.Graph``.