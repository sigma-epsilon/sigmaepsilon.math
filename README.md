# **SigmaEpsilon.Math** - A Python Library for Applied Mathematics in Physical Sciences

[![CircleCI](https://circleci.com/gh/dewloosh/sigmaepsilon.math.svg?style=shield)](https://circleci.com/gh/dewloosh/sigmaepsilon.math)
[![Documentation Status](https://readthedocs.org/projects/sigmaepsilon.math/badge/?version=latest)](https://sigmaepsilon.math.readthedocs.io/en/latest/?badge=latest)
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

The documentation is hosted on [ReadTheDocs](https://sigmaepsilon.math.readthedocs.io/en/latest/).

## **Installation**

`sigmaepsilon.math` can be installed (either in a virtual enviroment or globally) from PyPI using `pip` on Python >= 3.7:

```console
>>> pip install sigmaepsilon.math
```

or chechkout with the following command using GitHub CLI

```console
gh repo clone sigma-epsilon/sigmaepsilon.math
```

and install from source by typing

```console
>>> python install setup.py
```

## **Motivating Examples**

### Linear Algebra

Define a reference frame $\mathbf{B}$ relative to the frame $\mathbf{A}$:

```python
>>> from sigmaepsilon.math.linalg import ReferenceFrame, Vector, Tensor
>>> A = ReferenceFrame(name='A', axes=np.eye(3))
>>> B = A.orient_new('Body', [0, 0, 90*np.pi/180], 'XYZ', name='B')
```

Get the *DCM matrix* of the transformation between two frames:

```python
>>> B.dcm(target=A)
```

Define a vector $\mathbf{v}$ in frame $\mathbf{A}$ and show the components of it in frame $\mathbf{B}$:

```python
>>> v = Vector([0.0, 1.0, 0.0], frame=A)
>>> v.show(B)
```

Define the same vector in frame $\mathbf{B}$:

```python
>>> v = Vector(v.show(B), frame=B)
>>> v.show(A)
```

### Linear Programming

Solve the following Linear Programming Problem (LPP) with one unique solution:

```python
>>> from sigmaepsilon.math.optimize import LinearProgrammingProblem as LPP
>>> from sigmaepsilon.math.function import Function, Equality
>>> import sympy as sy
>>> variables = ['x1', 'x2', 'x3', 'x4']
>>> x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
>>> obj1 = Function(3*x1 + 9*x3 + x2 + x4, variables=syms)
>>> eq11 = Equality(x1 + 2*x3 + x4 - 4, variables=syms)
>>> eq12 = Equality(x2 + x3 - x4 - 2, variables=syms)
>>> problem = LPP(cost=obj1, constraints=[eq11, eq12], variables=syms)
>>> problem.solve()['x']
array([0., 6., 0., 4.])
```

### NonLinear Programming

Find the minimizer of the Rosenbrock function:

```python
>>> from sigmaepsilon.math.optimize import BinaryGeneticAlgorithm
>>> def Rosenbrock(x):
...     a, b = 1, 100
...     return (a-x[0])**2 + b*(x[1]-x[0]**2)**2
>>> ranges = [[-10, 10], [-10, 10]]
>>> BGA = BinaryGeneticAlgorithm(Rosenbrock, ranges, length=12, nPop=200)
>>> BGA.solve()
...
```

## **License**

This package is licensed under the MIT license.
