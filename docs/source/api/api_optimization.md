---
myst:
  html_meta:
    description: "API reference of sigmaepsilon.math.optimize: linear programming and metaheuristics (genetic algorithms, ant colony optimization, particle swarm optimization and differential evolution)."
---

(api_optimization)=

# Optimization

```{toctree}
:maxdepth: 1

api_optimization.lp
api_optimization.metaheuristics
```

## Optimizer state

The iterative optimizers keep track of their progress in an
{class}`~sigmaepsilon.math.optimize.state.OptimizerState` instance.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   ~sigmaepsilon.math.optimize.state.OptimizerState
```
