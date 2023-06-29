# Features

* Linear Algebra
  * A mechanism that guarantees to maintain the property of objectivity of tensorial quantities.
  * A `ReferenceFrame` class for all kinds of frames, and dedicated `RectangularFrame` and `CartesianFrame` classes as special cases, all NumPy compliant.
  * NumPy compliant classes like `Tensor` and `Vector` to handle various kinds of tensorial quantities efficiently.
  * A `JaggedArray` and a Numba-jittable `csr_matrix` to handle sparse data.

* Operations Research
  * Classes to define and solve linear and nonlinear optimization problems.
    * A `LinearProgrammingProblem` class to define and solve any kind of linear optimization problem.
    * A `BinaryGeneticAlgorithm` class to tackle more complicated optimization problems.

* Graph Theory
  * Algorithms to calculate rooted level structures and pseudo peripheral nodes of a `networkx` graph, which are useful if you want to minimize the bandwidth of sparse symmetrix matrices.
