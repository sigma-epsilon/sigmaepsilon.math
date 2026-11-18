# Changelog

All notable changes to this project will be documented in this file. If you are interested in bug fixes, enhancements etc., the best is to follow the project on GitHub.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.9.0]

### Changed

- `orient` and `orient_new` of `Vector`, `Tensor` and their subclasses rotate about the axes of the ambient frame, the same way `ReferenceFrame.orient` rotates frames. Before, they rotated about the axes of the frame of the object, which is not a rigid rotation in an oblique frame. The results only differ if the object is in a frame other than the ambient one. `Tensor.orient` now also accepts `dcm=`, like `Vector.orient`. The docstrings of the frame methods now state that rotations are always about the ambient axes, and that `'Body'` and `'Space'` only set the order of the angles within one call.
- `ReferenceFrame.volume()` and `CartesianFrame.volume()` return the signed volume `det(axes)`, which is negative for left-handed frames. Before, they returned `sqrt(det(Gram))` and `1.0`, which are never negative.
- `Tensor4` transforms in any dimension, for single tensors and collections alike. 3x3x3x3 tensors keep using the compiled kernels, which also serve `Tensor4x3`.
- `linsolve` with the methods `'Jordan'` and `'Gauss-Jordan'` uses partial pivoting, so matrices with zeros on the diagonal are solved. A matrix is considered singular if no pivot is larger than `1e-12` times the largest absolute value of the matrix, instead of an absolute `1e-12`. For a 1d `B` the solution and the residuals are 1d as well, they had the shape `(n, 1)` before.
- `is_hermitian` is deprecated in favour of `has_equal_dimensions`. It never checked if an array is Hermitian, only that all its dimensions are equal.
- `show_vector` uses NumPy instead of Numba, which is simpler and at least as fast.

### Fixed

- `dot(vector, Tensor2)` computed `T·a` instead of `a·T`, and gave the result in the wrong frame for oblique frames.
- `cross(vector, vector)` was only correct in right-handed orthonormal frames. It is computed from the ambient components now, and works for bulk vectors as well.
- `cross(Tensor2, vector)` returned `-(b × A)` instead of `A × b`. It follows the dyadic convention `(a ⊗ c) × b = a ⊗ (c × b)` of Gibbs, as set out in Wilson's *Vector Analysis* (1901). `cross(vector, Tensor2)` was correct, and now also works for collections.
- `ReferenceFrame.show(target)` returned the inverse transpose of the components of the frame in the target frame.
- `Vector.copy()` and `Tensor.copy()` dropped the frame, so the copy of a vector in a rotated frame was a different vector.
- `Vector.orient_new()`, `Vector.dual()` and `dot(vector, vector)` failed or returned wrong values for bulk vectors.
- Shape mismatches between components and transformation matrices raise an error. Before, the parallel Numba kernels behind `Tensor2`, `Tensor4`, `show_vector` and `show_frame` returned arrays of zeros.
- `linsolve` checks the shapes of its inputs. Before, a right-hand side with fewer rows than `A` silently solved a block of `A`. Only singular matrices raise `LinAlgError` now, other errors are no longer reported as a singular matrix.
- `csr_matrix` takes the number of columns from the largest column index if `shape` is not given, instead of the widest row, which dropped entries.

### Removed

- The unused helper `_tr_3333_sym` in `sigmaepsilon.math.linalg.tr`.

## [2.8.0]

### Added

- `ParticleSwarmOptimization` (PSO) and `DifferentialEvolution` (DE) in `sigmaepsilon.math.optimize`, for continuous, box-bounded problems. They are used the same way as the genetic algorithms (`solve`, `evolve`, `champion`, `state`, `minimize`, `seed`, `vectorized`, `n_jobs`, ...). DE supports the `rand/1/bin`, `best/1/bin` and `current-to-best/1/bin` mutation strategies.
- Both optimizers can return several distinct solutions in one run, one per local optimum. Set `n_solutions` and read the `solutions` property. During the search, PSO switches to species-based PSO and DE to neighbourhood-based crowding DE, which keeps several basins populated. The returned solutions are filtered by distance (`niche_radius`) and by a hill-valley test, so the same optimum is never returned twice.
- New base class `PopulationOptimizer` and result class `Candidate` shared by the two optimizers.
- User guide pages on PSO and DE, and a page on solving constrained problems with penalty functions.

## [2.7.1]

### Fixed

- With Numba installed, `det2x2` and `det3x3` in `sigmaepsilon.math.linalg` returned uninitialized memory instead of the determinants (often the result of a previous computation), because the kernels never wrote their output. They now return the correct determinants for single matrices and stacks of matrices alike.
- Two documentation pages included `global_refs.rst` with a Windows-style path, which broke the build of their links on other platforms (including ReadTheDocs).

## [2.7.0]

### Changed

- Numba is now an optional dependency. Install it with `pip install "sigmaepsilon.math[numba]"` for the best performance. Without Numba, the performance-critical parts of the library run as plain Python with the same results, but they can be orders of magnitude slower. The optimizers, the `function` module, numerical integration and k-nearest neighbour search don't use Numba at all. See the installation guide in the documentation for details.
- `sigmaepsilon.core>=1.4.0` is required.

## [2.2.0]

### Changed

- The `BinaryGeneticAlgorithm` class was updated with a new optional `minimize` parameter to be pprovided at object instantiation. With this, users can signal if they want their objective to be minimized or maximized.
- The docstring of the `BinaryGeneticAlgorithm` class is updated to clarify that by default, the class performs optimization, unless otherwise specified by the user.

## [2.0.0] - 2024-09-05

### Changed

- The `LinearProgrammingProblem` class has changed. Now it uses SciPy as a solver and the class only serves as a preprocessor. Usage of the modified class should be straightforward from the documentation.
- The `MLSApproximator` class got refactored.

### Fixed

- Fixed the `has_full_rank` function in `sigmaepsilon.math.linalg`
- Fixed initialization isssues related to relation classes in `sigmaepsilon.math.function.relation`.

### Removed

- The `VariableManager` class was removed, although it was not for the public, it only served as a helper for the `LinearProgrammingProblem` class.

## [1.1.0] - 2024-02-17

### Added

- Protocols for PointData, CellData, PolyData and PolyCell classes
- Cell interpolators (#7)

### Fixed

- Copy and deepcopy operations (#29).
- Cell class generation (#36).

### Changed

- Class architecture of data structures. Now the geometry is a nested class.

### Removed

- Intermediate, unnecessary geometry classes.
