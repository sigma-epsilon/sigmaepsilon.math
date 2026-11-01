---
myst:
  html_meta:
    description: "Overview of the optimization tools in sigmaepsilon.math: linear programming, and metaheuristics like genetic algorithms, ant colony optimization, particle swarm optimization and differential evolution."
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

## Metaheuristics

For nonlinear problems, the library provides a family of **metaheuristics**: nature-inspired, derivative-free search methods that only need to evaluate the objective function. They work with objectives that are non-smooth, noisy, multimodal or only available as a black box, and many of them handle discrete or combinatorial variables too. In exchange, they come with *no optimality guarantee*: you get a good solution found within a finite budget of iterations and function evaluations, which is why it pays to compare a few of them on your problem.

All of them follow the same usage pattern: you pass the objective function and the definition of the search space to the constructor, call `solve`, and get back the best solution found (the *champion*). Iterations can also be run one by one with `evolve`, the runs are reproducible with `seed`, and objectives can be evaluated in batches (`vectorized`) or in parallel (`n_jobs`).

### Genetic Algorithms (GA)

📖 {ref}`Genetic Algorithms API Reference <api_optimization_GA>`

**Genetic algorithms** evolve a population of candidate solutions through selection, crossover and mutation. The library has binary, integer and real-valued variants, built on a framework that is easy to extend with your own representation:

* {doc}`bga` -- an introduction to genetic algorithms in general, and a full walkthrough of the Binary Genetic Algorithm (BGA).
* {doc}`ga_variants` -- the Real-Valued (RGA) and Integer (IGA) genetic algorithms.
* {doc}`ga_tuning` -- pluggable selection strategies and monitoring population diversity.
* {doc}`ga_performance` -- vectorized and parallel objective evaluation, choosing a representation, and posing objectives symbolically.
* {doc}`custom_ga` -- a full, worked example of subclassing the framework for a custom representation.

### Ant Colony Optimization (ACO)

📖 {ref}`Ant Colony Optimization API Reference <api_optimization_ACO>`

**Ant Colony Optimization** is based on the pheromone-trail-following behavior of foraging ants. Unlike the GA family, which
evolves a population of candidate solutions directly, ACO builds solutions incrementally
(step by step) and steers the construction process with a shared memory (the "pheromones")
that gets reinforced along good solutions and decays over time. See {doc}`aco` for how the
continuous (ACOR) and combinatorial (Ant System / TSP) variants work, when to reach for them
instead of a GA or `linprog`, and worked examples of both.

### Particle Swarm Optimization (PSO)

📖 {ref}`Particle Swarm Optimization API Reference <api_optimization_PSO>`

**Particle Swarm Optimization** moves a swarm of particles through a continuous, box-bounded search space. Each particle is drawn towards the best point it has seen and towards good points found by the others. It has few parameters, and the defaults rarely need tuning. It can also return **several distinct solutions in a single run**, one per basin of attraction. See {doc}`pso` for how it works, how to use it, and how it finds multiple optima (species-based PSO).

### Differential Evolution (DE)

📖 {ref}`Differential Evolution API Reference <api_optimization_DE>`

**Differential Evolution** evolves a population by perturbing its members with scaled differences of other members. It solves the same class of continuous, box-bounded problems as PSO, and tends to be more robust on rugged and ill-conditioned landscapes. Like PSO, it can return several distinct solutions in a single run. See {doc}`de` for how it works, its mutation strategies, and how it finds multiple optima (neighborhood-based crowding DE).

### Constrained problems

The metaheuristics above search a box given by the bounds of the variables. To solve problems with general equality and inequality constraints, you can add the constraints to the objective as penalties. See {doc}`penalty` for how to do this with static exterior penalty functions, including constrained problems with multiple optima.

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
