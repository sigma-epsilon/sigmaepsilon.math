# Changelog

All notable changes to this project will be documented in this file. If you are interested in bug fixes, enhancements etc., the best is to follow the project on GitHub.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
