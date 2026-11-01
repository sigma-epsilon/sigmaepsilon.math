---
myst:
  html_meta:
    description: "API reference of sigmaepsilon.math.function: the Function class, the Relation, Equality and InEquality classes for constraints, and test functions for optimization."
---

# Functions and Relations

## Functions

```{eval-rst}
.. autoclass:: sigmaepsilon.math.function.Function
   :members:

.. autodata:: sigmaepsilon.math.function.function.FunctionLike
```

## Relations

```{eval-rst}
.. autoclass:: sigmaepsilon.math.function.Relation
   :members:

.. autoclass:: sigmaepsilon.math.function.Equality
   :members:

.. autoclass:: sigmaepsilon.math.function.InEquality
   :members:
```

## Test Functions for Optimization

The test functions share a few base classes, that you can also use to define your own
benchmark functions with known optima.

```{eval-rst}
.. automodule:: sigmaepsilon.math.function.testfunction
   :members:

.. automodule:: sigmaepsilon.math.function.functions
   :members:
```
