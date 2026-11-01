---
myst:
  html_meta:
    description: "API reference of sigmaepsilon.math.optimize: linear programming, genetic algorithms, particle swarm optimization, differential evolution and ant colony optimization."
---

(api_optimization)=

# Optimization

```{toctree}
:maxdepth: 1

api_optimization.lp
api_optimization.nlp
api_optimization.aco
```

## Optimizer state

The iterative optimizers keep track of their progress in an
{class}`~sigmaepsilon.math.optimize.state.OptimizerState` instance.

```{eval-rst}
.. autoclass:: sigmaepsilon.math.optimize.state.OptimizerState
    :members:
```
