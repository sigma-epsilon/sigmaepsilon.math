---
myst:
  html_meta:
    description: "API reference of sigmaepsilon.math.function: the Function class, the Relation, Equality and InEquality classes for constraints, and test functions for optimization."
---

# Functions and Relations

## Functions

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   ~sigmaepsilon.math.function.Function
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-data-template.rst

   ~sigmaepsilon.math.function.function.FunctionLike
```

## Relations

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   ~sigmaepsilon.math.function.Relation
   ~sigmaepsilon.math.function.Equality
   ~sigmaepsilon.math.function.InEquality
```

## Test Functions for Optimization

The test functions share a few base classes, that you can also use to define your own
benchmark functions with known optima.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst

   ~sigmaepsilon.math.function.testfunction
   ~sigmaepsilon.math.function.functions
```
