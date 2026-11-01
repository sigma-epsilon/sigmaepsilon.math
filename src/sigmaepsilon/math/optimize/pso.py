"""Particle Swarm Optimization (PSO) implementation."""

import numpy as np
from numpy import ndarray

from .population import PopulationOptimizer, normalize

__all__ = ["ParticleSwarmOptimization"]


class ParticleSwarmOptimization(PopulationOptimizer):
    """
    Particle Swarm Optimization (PSO).

    It finds minimums or maximums of unconstrained problems over box-ranges of
    continuous variables. Every particle of the swarm has a position and a velocity,
    and remembers the best position it has visited (its personal best). In every
    iteration, the velocities are updated as

    .. math::

        v \\leftarrow w v + c_1 r_1 (p - x) + c_2 r_2 (g - x),

    where :math:`p` is the personal best of the particle, :math:`g` is its social
    attractor and :math:`r_1, r_2` are uniform random numbers in [0, 1], drawn
    independently for every coordinate. Then the particles move: :math:`x
    \\leftarrow x + v`.

    If `n_solutions` is 1, the social attractor of every particle is the best position
    found by the swarm (global best PSO). Otherwise, the species-based PSO of Li (2004)
    is used: in every iteration, the personal bests are visited from best to worst, and
    a personal best becomes the seed of a new species if it is farther than
    `niche_radius` from all the existing seeds, or joins the species of the first seed
    within that radius. Every particle is attracted by the seed of its species, so
    the swarm can converge to several optima simultaneously.

    .. note::
       This class is designed for maximizing the objective function. To minimize it, either negate
       the objective function or pass ``minimize=True`` when instantiating the class.

    Parameters
    ----------
    fnc: Callable
        The function to evaluate. It is assumed, that the function expects and N
        number of scalar arguments as a 1d iterable.
    ranges: Iterable
        Ranges for each scalar argument to the objective function.
    w: float, Optional
        The inertia weight. Default is 0.7298.
    c1: float, Optional
        The cognitive coefficient (attraction towards the personal best).
        Default is 1.49618.
    c2: float, Optional
        The social coefficient (attraction towards the social attractor).
        Default is 1.49618.
    v_max: float, Optional
        The maximum speed of a particle along each axis, as a fraction of the range of
        the corresponding variable. Default is 0.2.
    nPop: int, Optional
        The number of particles. Default is 40.
    kwargs: dict, Optional
        The rest of the parameters are passed to
        :class:`~sigmaepsilon.math.optimize.population.PopulationOptimizer`
        (`maxiter`, `miniter`, `maxage`, `ftol`, `minimize`, `seed`, `vectorized`,
        `n_jobs`, `n_solutions`, `niche_radius`).

    See Also
    --------
    :class:`~sigmaepsilon.math.optimize.de.DifferentialEvolution`

    Examples
    --------
    Find the minimizer of the Rosenbrock function.
    The exact value of the solution is x = [1.0, 1.0].

    >>> from sigmaepsilon.math.optimize import ParticleSwarmOptimization as PSO
    >>>
    >>> def rosenbrock(x):
    ...     a, b = 1, 100
    ...     return (a-x[0])**2 + b*(x[1]-x[0]**2)**2
    >>>
    >>> ranges = [[-5, 5], [-5, 5]]
    >>> pso = PSO(rosenbrock, ranges, minimize=True, seed=0)
    >>> champion = pso.solve()
    >>> x = champion.phenotype
    >>> fx = champion.fitness

    Find the four minima of Himmelblau's function.

    >>> def himmelblau(x):
    ...     return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    >>>
    >>> pso = PSO(himmelblau, ranges, nPop=60, minimize=True, n_solutions=4, seed=0)
    >>> _ = pso.solve()
    >>> len(pso.solutions)
    4

    """

    __slots__ = (
        "w",
        "c1",
        "c2",
        "v_max",
        "_velocities",
        "_pbest_x",
        "_pbest_f",
    )

    def __init__(
        self,
        *args,
        w: float = 0.7298,
        c1: float = 1.49618,
        c2: float = 1.49618,
        v_max: float = 0.2,
        **kwargs,
    ):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        super().__init__(*args, **kwargs)

    def reset(self) -> "ParticleSwarmOptimization":
        """Reset the solver and return the object."""
        self._velocities = None
        self._pbest_x = None
        self._pbest_f = None
        return super().reset()

    def set_solution_params(self, **kwargs) -> "ParticleSwarmOptimization":
        """
        Set the hyperparameters of the algorithm.

        Besides the parameters of
        :func:`~sigmaepsilon.math.optimize.population.PopulationOptimizer.set_solution_params`,
        it accepts `w`, `c1`, `c2` and `v_max`.
        """
        for key in ("w", "c1", "c2", "v_max"):
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
        return super().set_solution_params(**kwargs)

    @property
    def velocities(self) -> ndarray:
        """Return the velocities of the particles."""
        return self._velocities

    def archive(self) -> tuple[ndarray, ndarray]:
        """Return the personal bests of the particles and their objective values."""
        return self._pbest_x, self._pbest_f

    def initialize(self) -> None:
        """Place the particles randomly in the box, with random velocities."""
        lower, upper = self.ranges[:, 0], self.ranges[:, 1]
        v_max = self.v_max * (upper - lower)
        size = (self.nPop, self.dim)
        self._phenotypes = self.rng.uniform(lower, upper, size=size)
        self._velocities = self.rng.uniform(-v_max, v_max, size=size)
        self._fitness = self.evaluate(self._phenotypes)
        self._pbest_x = self._phenotypes.copy()
        self._pbest_f = self._fitness.copy()

    def attractors(self) -> ndarray:
        """Return the social attractor of every particle as a 2d array."""
        pbest_x, pbest_f = self._pbest_x, self._pbest_f
        order = np.argsort(pbest_f if self._minimize else -pbest_f, kind="stable")

        if self.n_solutions == 1:
            return np.tile(pbest_x[order[0]], (self.nPop, 1))

        normalized = normalize(pbest_x, self.ranges)
        seeds: list[int] = []
        species = np.empty(self.nPop, dtype=int)
        for i in order:
            if seeds:
                distances = np.linalg.norm(normalized[seeds] - normalized[i], axis=1)
                within = np.flatnonzero(distances <= self.niche_radius)
                if len(within) > 0:
                    species[i] = seeds[within[0]]
                    continue
            seeds.append(int(i))
            species[i] = i
        return pbest_x[species]

    def step(self) -> None:
        """Move the particles and update their personal bests."""
        x, v = self._phenotypes, self._velocities
        lower, upper = self.ranges[:, 0], self.ranges[:, 1]
        v_max = self.v_max * (upper - lower)

        r1 = self.rng.random(x.shape)
        r2 = self.rng.random(x.shape)
        cognitive = self.c1 * r1 * (self._pbest_x - x)
        social = self.c2 * r2 * (self.attractors() - x)
        v = np.clip(self.w * v + cognitive + social, -v_max, v_max)
        x = x + v

        out_of_bounds = (x < lower) | (x > upper)
        x = np.clip(x, lower, upper)
        v = np.where(out_of_bounds, 0.0, v)

        fitness = self.evaluate(x)
        improved = self.is_better(fitness, self._pbest_f)
        self._pbest_x[improved] = x[improved]
        self._pbest_f[improved] = fitness[improved]

        self._phenotypes, self._velocities, self._fitness = x, v, fitness
