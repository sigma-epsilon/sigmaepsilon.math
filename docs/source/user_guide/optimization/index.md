---
myst:
  html_meta:
    description: "Overview of the optimization tools in sigmaepsilon.math: linear programming, genetic algorithms, ant colony optimization, particle swarm optimization and differential evolution."
---

(user_guide_optimization)=

# Optimization

📖 {ref}`Optimization API Reference <api_optimization>`

Given that the main purpose of the SigmaEpsilon ecosystem is to facilitate structural optimization, this submodule needs no further elaboration, as having a capable optimization toolkit is essential for the project. The overarching approach is to utilize third-party solutions whenever feasible, though we also provide essential solutions when necessary or fill in the gaps wherever required.

```{note}
Most of the classes for optimization rely on other classes like {class}`~sigmaepsilon.math.function.function.Function` or {class}`~sigmaepsilon.math.function.relation.Relation`. Before you proceed forward, it might be a good idea to check them out first {ref}`here <user_guide_function>`.
```

## Linear Programming (LP)

📖 {ref}`Linear Programming API Reference <api_optimization_LP>`

The library offers a solver for a wide range of linear optimization problems, including continuous, integer and mixed-integer problems, built around [SciPy](https://scipy.org/)'s `linprog` and [SymPy](https://www.sympy.org/en/index.html). See {doc}`lp` for the basic API, a 2d example visualized with Matplotlib, and mixed-integer programs.

## Nonlinear Programming (NLP)

📖 {ref}`Nonlinear Programming API Reference <api_optimization_NLP>`

Nonlinear programming deals with problems where the objective function, the constraints, or both are nonlinear. The library provides a small family of genetic algorithms (GAs) for this purpose:

* {doc}`bga` -- an introduction to genetic algorithms in general, and a full walkthrough of the Binary Genetic Algorithm (BGA).
* {doc}`ga_variants` -- the Real-Valued (RGA) and Integer (IGA) genetic algorithms.
* {doc}`ga_tuning` -- pluggable selection strategies and monitoring population diversity.
* {doc}`ga_performance` -- vectorized and parallel objective evaluation, choosing a representation, and posing objectives symbolically.
* {doc}`custom_ga` -- a full, worked example of subclassing the framework for a custom representation.

## Ant Colony Optimization (ACO)

📖 {ref}`Ant Colony Optimization API Reference <api_optimization_ACO>`

**Ant Colony Optimization (ACO)** is another nature-inspired metaheuristic, this time based
on the pheromone-trail-following behavior of foraging ants. Unlike the GA family, which
evolves a population of candidate solutions directly, ACO builds solutions incrementally
(step by step) and steers the construction process with a shared memory (the "pheromones")
that gets reinforced along good solutions and decays over time. See {doc}`aco` for how the
continuous (ACOR) and combinatorial (Ant System / TSP) variants work, when to reach for them
instead of a GA or `linprog`, and worked examples of both.

## Particle Swarm Optimization (PSO) and Differential Evolution (DE)

📖 {ref}`Particle Swarm Optimization API Reference <api_optimization_PSO>` · {ref}`Differential Evolution API Reference <api_optimization_DE>`

**Particle Swarm Optimization** moves a swarm of particles through the search space. Each particle is drawn towards the best point it has seen and towards good points found by the others. **Differential Evolution** evolves a population by perturbing its members with scaled differences of other members. Both are robust, derivative-free optimizers for continuous, box-bounded problems, with the same usage pattern as the GAs and ACO. Both can also return **several distinct solutions in a single run**, one per basin of attraction, without ever returning the same optimum twice.

* {doc}`pso` -- how PSO works, how to use it, and how it finds multiple optima (species-based PSO).
* {doc}`de` -- how DE works, its mutation strategies, and how it finds multiple optima (neighborhood-based crowding DE).
* {doc}`penalty` -- solving problems with general equality and inequality constraints using penalty functions, including constrained problems with multiple optima.

```{toctree}
:maxdepth: 1
:hidden:

lp
bga
ga_variants
ga_tuning
ga_performance
custom_ga
aco
pso
de
penalty
```
