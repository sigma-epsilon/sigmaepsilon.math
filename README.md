# **SigmaEpsilon.Math** - A Python Library for Applied Mathematics in Physical Sciences

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/sigma-epsilon/sigmaepsilon.math/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/sigma-epsilon/sigmaepsilon.math/tree/main)
[![Documentation Status](https://readthedocs.org/projects/sigmaepsilonmath/badge/?version=latest)](https://sigmaepsilonmath.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://badge.fury.io/py/sigmaepsilon.math.svg)](https://pypi.org/project/sigmaepsilon.math)
[![codecov](https://codecov.io/gh/dewloosh/sigmaepsilon.math/branch/main/graph/badge.svg?token=TBI6GG4ECG)](https://codecov.io/gh/dewloosh/sigmaepsilon.math)
[![Python 3.7-3.10](https://img.shields.io/badge/python-3.7%E2%80%923.10-blue)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`SigmaEpsilon.Math` is a Python library that provides tools to formulate and solve problems related to all kinds of scientific disciplines. It is a part of the SigmaEpsilon ecosystem, which is designed mainly to solve problems related to computational solid mechanics, but if something is general enough, it ends up here. A good example is the included vector and tensor algebra modules, or the various optimizers, which are applicable in a much broader context than they were originally designed for.

The most important features:

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

> **Note**
> Be aware, that the library uses JIT-compilation through Numba, and as a result,
> first calls to these functions may take longer, but it pays off big time in the long run.

## **Documentation**

The [documentation](https://sigmaepsilonmath.readthedocs.io/en/latest/) is hosted on ReadTheDocs. You can find examples there.

## **Installation**

sigmaepsilon.math can be installed from PyPI using `pip` on Python >= 3.7:

```console
>>> pip install sigmaepsilon.math
```

or chechkout with the following command using GitHub CLI

```console
gh repo clone sigma-epsilon/sigmaepsilon.math
```

and install from source by typing

```console
>>> pip install .
```

If you want to run the tests, you can install the package along with the necessary optional dependencies like this

```console
>>> pip install ".[test]"
```

### Development mode

If you are a developer and want to install the library in development mode, the suggested way is by using this command:

```console
>>> pip install "-e .[test, dev]"
```

## How to contribute?

Contributions are currently expected in any the following ways:

* finding bugs
  If you run into trouble when using the library and you think it is a bug, feel free to raise an issue.
* feedback
  All kinds of ideas are welcome. For instance if you feel like something is still shady (after reading the user guide), we want to know. Be gentle though, the development of the library is financially not supported yet.
* feature requests
  Tell us what you think is missing (with realistic expectations).
* examples
  If you've done something with the library and you think that it would make for a good example, get in touch with the developers and we will happily inlude it in the documention.
* sharing is caring
  If you like the library, share it with your friends or colleagues so they can like it too.

## Acknowledgements

Although `sigmaepsilon.math` heavily builds on `NumPy`, `Scipy`, `Numba` and `Awkward` and it also has functionality related to `networkx` and other third party libraries. Whithout these libraries the concept of writing performant, yet elegant Python code would be much more difficult.

**A lot of the packages mentioned on this document here and the introduction have a citable research paper. If you use them in your work through sigmaepsilon.mesh, take a moment to check out their documentations and cite their papers.**

Also, funding of these libraries is partly based on the size of the community they are able to support. If what you are doing strongly relies on these libraries, don't forget to press the :star: button to show your support.

## **License**

This package is licensed under the [MIT license](https://opensource.org/license/mit/).
