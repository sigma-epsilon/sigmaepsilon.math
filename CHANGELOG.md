# Changelog

All notable changes to this project will be documented in this file. If you are interested in bug fixes, enhancements etc., best follow the project on GitHub.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.2.0] - 2024-09-05

### Changed

- The `LinearProgrammingProblem` class has changed. Usage of the modified class should be straightforward.

### Fixed

- Several fixes to the `LinearProgrammingProblem` class has been made.
- Fixed the `has_full_rank` function in `sigmaepsilon.math.linalg`

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
