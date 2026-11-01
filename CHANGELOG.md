# Changelog

All notable changes to this project will be documented in this file. If you are interested in bug fixes, enhancements etc., the best is to follow the project on GitHub.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The documentation is written in MyST Markdown instead of reStructuredText, and the notebooks are rendered with MyST-NB. The landing page of the user guide and its introduction are separate pages now, the example notebooks are listed in the user guide, and the Theory Guide and the API page of the arrays are part of the navigation.
- The deploy jobs on CircleCI end successfully without publishing anything if the version in `pyproject.toml` is already on (Test)PyPI, so merges that don't bump the version no longer fail.

### Added

- LLM-friendly documentation with `sphinx-llm`: `llms.txt`, `llms-full.txt` and a Markdown version of every page, with a page on using the library with LLMs. To keep LLM contexts small, the index and the bundle only contain the user guide, the examples and the API reference. Every page has a description in its front matter, used in `llms.txt`.
- `TensorLike`, `FunctionLike`, `OptimizerState` and the base classes of the test functions are part of the API reference.

### Fixed

- The documentation builds without warnings. Fixed the docstrings with broken markup or cross-references, among them the description of `MLSApproximator`, which referred to a nonexistent `fit` method, and the return type of `Graph.adjacency_matrix`.

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
