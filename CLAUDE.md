# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`sigmaepsilon.math` is a Python library of general-purpose applied-math tools (vector/tensor algebra,
sparse arrays, function/relation modeling, least-squares & moving-least-squares approximation,
optimization: linear programming, genetic algorithms) used as a foundation package in the SigmaEpsilon
computational-solid-mechanics ecosystem. It is distributed as the namespace package `sigmaepsilon.math`
(source under `src/sigmaepsilon/math`), and depends on `sigmaepsilon-core` / `sigmaepsilon-deepdict` from
the same ecosystem, plus NumPy, SciPy, Numba, Awkward and SymPy.

Package/dependency management is Poetry. Packaging metadata, dependencies and tool config live in
`pyproject.toml`.

## Common commands

Install dependencies (uses `poetry.toml`, which configures an in-project virtualenv):

```
poetry install --with dev,test,docs
```

Run the full test suite:

```
poetry run pytest
```

Run a single test file / test:

```
poetry run pytest tests/test_lpp.py
poetry run pytest tests/test_lpp.py::test_name -v
```

Run with coverage (two configs exist — the default `.coveragerc` excludes numba-jitted code from
coverage; `.coveragerc_nojit` does not):

```
poetry run pytest --cov=sigmaepsilon --cov-config=.coveragerc
```

Format code (pre-commit runs Black on `src/` only):

```
poetry run black src/
poetry run pre-commit run --all-files
```

Build docs (Sphinx, config under `docs/source`):

```
cd docs && make html
```

Numba JIT: several modules (e.g. `linalg/sparse/*`) are `@njit`-compiled; on first import/test-run
expect JIT compilation overhead and `__pycache__`/`.nbi`/`.nbc` cache artifacts to appear alongside
those files.

## Architecture

The package exposes several largely independent subpackages under `src/sigmaepsilon/math/`:

- **`linalg/`** — the core vector/tensor algebra layer.
  - `meta.py` defines the abstract base types (`FrameLike`, `TensorLike`, `ArrayWrapper`) that the
    rest of the linalg layer builds on.
  - `frame.py` (`ReferenceFrame`, `RectangularFrame`, `CartesianFrame`) implements reference frames;
    frames keep weak references to the tensorial objects registered to them (`_weakrefs`) so that
    in-place frame operations (e.g. rotation) can transform all dependent tensors/vectors together.
  - `vector.py` / `tensor.py` implement vector and tensor objects that carry a reference to a frame
    and transform when the frame changes (`tr.py` holds transformation-rule helpers).
  - `sparse/` is a separate sparse-array layer: `csr.py` (custom CSR matrix), `jaggedarray.py`
    (ragged/jagged arrays, backed by Awkward), `utils.py` (Numba-jitted helper kernels). These are
    re-exported at the top of `linalg/__init__.py` as `JaggedArray`, `csr_matrix`.
  - `logical.py` / `imap.py` / `solve.py` / `exceptions.py` hold predicates (e.g. is-orthonormal
    checks), index mapping, linear solve routines, and linalg-specific exceptions respectively.
  - Every submodule defines its own `__all__`; `linalg/__init__.py` aggregates them manually — when
    adding a new public symbol, add it to the submodule's `__all__`, not just to the class body.

- **`function/`** — symbolic/numeric function modeling used by the optimizers.
  - `Function`/`FunctionLike` (`function.py`) wrap a callable or SymPy expression uniformly, exposing
    value/gradient/Hessian access (`metafunction.py`, `symutils.py` handle the SymPy side).
  - `relation.py` defines `Relation`, `Equality`, `InEquality` (with a `Relations` enum) used to
    express optimization constraints in terms of `Function`s.
  - `testfunction.py` provides standard benchmark test functions for optimizers.

- **`optimize/`** — optimization algorithms, all built on top of `function/`.
  - `lp.py`: `LinearProgrammingProblem` — thin, well-documented wrapper around
    `scipy.optimize.linprog` (HIGHS backend). Problems are assembled from `Function`/`Relation`/
    `Equality`/`InEquality` objects (symbolic or string expressions), which get translated into the
    matrix form `linprog` expects; supports sparse constraint matrices (`scipy.sparse`) and mixed
    integer problems via `integrality`.
  - `ga.py`: `GeneticAlgorithm` — general real-valued GA. Population members are `Genom` (pydantic
    `BaseModel`: phenotype, genotype, fitness, age, index). Supports both minimize and maximize via a
    `minimize` flag.
  - `bga.py`: `BinaryGeneticAlgorithm` — binary-encoded GA subclassing `GeneticAlgorithm`, for
    unconstrained real-valued problems over box ranges; encodes each scalar argument into a fixed-length
    chromosome (`length` controls precision).
  - `state.py`: `OptimizerState` — shared state/result object threaded through GA iterations.

- **`approx/`** — least-squares style approximation.
  - `ls.py` (`least_squares`, `weighted_least_squares`, `moving_least_squares`) and `mls.py`
    (`MLSApproximator`) implement (moving) least squares fits; `ls_poly.py`/`ls_preproc.py` build
    polynomial bases and preprocess sample data; `functions.py` provides the weight kernels
    (`CubicWeightFunction`, `ConstantWeightFunction`, `SingularWeightFunction`) consumed by MLS.
  - `lagrange.py` provides 1D Lagrange interpolation/generation helpers, independent of the MLS code.

- **`graph/`** — small graph utilities (`graph.py`, `utils.py`), used together with optional
  `networkx` (a test/docs-only dependency, not a core runtime dependency).

- Top-level modules (`utils.py`, `arraysetops.py`, `logical.py`, `numint.py`, `knn.py`, `hist.py`,
  `decorate.py`, `mathtypes.py`, `metautils.py`, `downloads.py`) are standalone helpers (numerical
  integration rules, set operations on arrays, k-NN, histogramming, shared type aliases, decorators)
  imported piecemeal by the subpackages above rather than forming their own layer.

Tests under `tests/` largely mirror this subpackage structure (`tests/linalg/`, plus
`test_ga*.py`, `test_lpp.py`, `test_lp_*constraints.py`, `test_mls_*.py`, `test_approx.py`,
`test_function*.py`, `test_graph.py`, `test_knn.py`, `test_numint.py`, `test_utils.py`,
`test_docstrings.py` which exercises doctest-style examples in docstrings). `tests/conftest.py`
forces the Matplotlib backend to `agg` for all tests.

## Notes for making changes

- Public API surface is controlled by explicit `__all__` lists in nearly every module — when adding
  a new public class/function, update the relevant `__all__` (and the subpackage `__init__.py` if it
  should be importable from the subpackage root).
- Code is Black-formatted (`src/` only, per `.pre-commit-config.yaml`); run Black before committing.
- Numba (`@njit`) is used for performance-critical numeric kernels (notably in `linalg/sparse/`);
  coverage reporting intentionally excludes these (`.coveragerc` excludes `@njit`/`@jit`/`@guv`
  decorated code from coverage).
