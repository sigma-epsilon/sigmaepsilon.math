---
myst:
  html_meta:
    description: "API reference of sigmaepsilon.math: linear algebra, functions and relations, optimization, graphs and approximation."
---

(api_reference)=

# API Reference

The reference guide contains a detailed description of the functions,
modules, and objects included in the library. The reference describes how the
methods work and which parameters can be used. It assumes that you have an
understanding of the key concepts.

```{toctree}
:hidden:

api/api_linalg
api/api_function
api/api_optimization
api/api_graph
api/api_approx
```

## Linear Algebra

### Reference Frames

See {doc}`api/api_linalg.frame` for the details.

```{eval-rst}
.. autosummary::

   ~sigmaepsilon.math.linalg.meta.FrameLike
   ~sigmaepsilon.math.linalg.ReferenceFrame
   ~sigmaepsilon.math.linalg.RectangularFrame
   ~sigmaepsilon.math.linalg.CartesianFrame
```

### Vectors

See {doc}`api/api_linalg.vector` for the details.

```{eval-rst}
.. autosummary::

   ~sigmaepsilon.math.linalg.Vector
```

### Tensors

See {doc}`api/api_linalg.tensor` for the details.

```{eval-rst}
.. autosummary::

   ~sigmaepsilon.math.linalg.meta.TensorLike
   ~sigmaepsilon.math.linalg.Tensor
   ~sigmaepsilon.math.linalg.Tensor2
   ~sigmaepsilon.math.linalg.Tensor2x3
   ~sigmaepsilon.math.linalg.Tensor4
   ~sigmaepsilon.math.linalg.Tensor4x3
```

### Arrays

See {doc}`api/api_linalg.array` for the details.

```{eval-rst}
.. autosummary::

   ~sigmaepsilon.math.linalg.meta.Array
   ~sigmaepsilon.math.linalg.meta.ArrayWrapper
   ~sigmaepsilon.math.linalg.sparse.JaggedArray
   ~sigmaepsilon.math.linalg.sparse.csr_matrix
```

### Routines

See {doc}`api/api_linalg.routines` for the details.

```{eval-rst}
.. autosummary::

   ~sigmaepsilon.math.linalg.logical
   ~sigmaepsilon.math.linalg.utils
```

## Functions and Relations

See {doc}`api/api_function` for the details.

```{eval-rst}
.. autosummary::

   ~sigmaepsilon.math.function.Function
   ~sigmaepsilon.math.function.function.FunctionLike
   ~sigmaepsilon.math.function.Relation
   ~sigmaepsilon.math.function.Equality
   ~sigmaepsilon.math.function.InEquality
   ~sigmaepsilon.math.function.testfunction
   ~sigmaepsilon.math.function.functions
```

## Optimization

### Linear Programming (LP)

See {doc}`api/api_optimization.lp` for the details.

```{eval-rst}
.. autosummary::

   ~sigmaepsilon.math.optimize.LinearProgrammingProblem
```

### Metaheuristics

See {doc}`api/api_optimization.metaheuristics` for the details.

```{eval-rst}
.. autosummary::

   ~sigmaepsilon.math.optimize.ga.GeneticAlgorithm
   ~sigmaepsilon.math.optimize.ga.Genom
   ~sigmaepsilon.math.optimize.selection.SelectionStrategy
   ~sigmaepsilon.math.optimize.selection.TournamentSelection
   ~sigmaepsilon.math.optimize.selection.RouletteSelection
   ~sigmaepsilon.math.optimize.selection.RankSelection
   ~sigmaepsilon.math.optimize.bitchromosome.BitChromosomeGeneticAlgorithm
   ~sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm
   ~sigmaepsilon.math.optimize.iga.IntegerGeneticAlgorithm
   ~sigmaepsilon.math.optimize.rga.RealValuedGeneticAlgorithm
   ~sigmaepsilon.math.optimize.aco.AntSolution
   ~sigmaepsilon.math.optimize.aco.AntColonyOptimization
   ~sigmaepsilon.math.optimize.aco.ContinuousAntColonyOptimization
   ~sigmaepsilon.math.optimize.aco.CombinatorialAntColonyOptimization
   ~sigmaepsilon.math.optimize.pso.ParticleSwarmOptimization
   ~sigmaepsilon.math.optimize.de.DifferentialEvolution
   ~sigmaepsilon.math.optimize.population.Candidate
   ~sigmaepsilon.math.optimize.population.PopulationOptimizer
   ~sigmaepsilon.math.optimize.population.normalize
   ~sigmaepsilon.math.optimize.population.hill_valley
   ~sigmaepsilon.math.optimize.population.distinct_optima
```

### Optimizer state

```{eval-rst}
.. autosummary::

   ~sigmaepsilon.math.optimize.state.OptimizerState
```

## Graphs and Graph Routines

### Graph

See {doc}`api/api_graph.graph` for the details.

```{eval-rst}
.. autosummary::

   ~sigmaepsilon.math.graph.graph.Graph
```

### Graph Routines

See {doc}`api/api_graph.routines` for the details.

```{eval-rst}
.. autosummary::

   ~sigmaepsilon.math.graph.utils
```

## Approximation

### Interpolation

See {doc}`api/api_approx.interpolation` for the details.

```{eval-rst}
.. autosummary::

   ~sigmaepsilon.math.approx.lagrange
```

### Regression

See {doc}`api/api_approx.regression` for the details.

```{eval-rst}
.. autosummary::

   ~sigmaepsilon.math.approx.functions
   ~sigmaepsilon.math.approx.ls
   ~sigmaepsilon.math.approx.mls
```
