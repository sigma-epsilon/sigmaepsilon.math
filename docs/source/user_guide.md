# A Quick Guide

## **Installation**

`sigmaepsilon.math` can be installed (either in a virtual enviroment or globally) from PyPI using `pip` on Python >= 3.7:

```console
>>> pip install sigmaepsilon.math
```

or chechkout with the following command using GitHub CLI

```console
>>> gh repo clone dewloosh/sigmaepsilon.math
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
