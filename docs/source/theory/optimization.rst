=============
Optimization
=============

Optimization is the branch of mathematics and computer science concerned with finding the
"best" solution to a problem, typically subject to a set of constraints. In broad terms, an
optimization problem seeks to:

.. math::

   \min_{\mathbf{x}} \; f(\mathbf{x})

(or maximize, depending on the setup), where :math:`\mathbf{x}` is a set of variables and
:math:`f(\mathbf{x})` is a cost function or objective function. Many areas of science,
engineering, economics, and operations research rely on optimization methods to find
optimal or near-optimal solutions to various problems.

Approaches to Optimization
==========================

There are several numerical and analytical approaches to solving optimization problems.

Gradient-Based Methods
----------------------

**Gradient-based** (or derivative-based) methods require first-order (and sometimes higher-order)
derivatives of the objective function. Examples include:

- **Gradient Descent**:
  
  .. math::

     \mathbf{x}_{k+1} = \mathbf{x}_{k} - \alpha \nabla f(\mathbf{x}_{k}),

  where :math:`\alpha` is the step size or learning rate, and :math:`\nabla f(\mathbf{x}_{k})`
  is the gradient of :math:`f` evaluated at :math:`\mathbf{x}_{k}`.

- **Newton's Method**:
  
  .. math::

     \mathbf{x}_{k+1} = \mathbf{x}_{k} 
                       - \bigl(\nabla^2 f(\mathbf{x}_{k})\bigr)^{-1}
                       \nabla f(\mathbf{x}_{k}),

  which uses second-order derivatives (the Hessian matrix :math:`\nabla^2 f(\mathbf{x}_k)`).

- **Quasi-Newton Methods** (e.g., BFGS), which approximate the Hessian to reduce computation.

Derivative-Free Methods
-----------------------

**Derivative-free** (or gradient-free) methods do not require explicit gradients or Hessians.
They are often used when:

- The objective function is noisy or discontinuous.
- Analytical gradients are difficult or impossible to compute.
- Simulation-based or black-box models are involved.

Examples include:
- **Genetic Algorithms**
- **Simulated Annealing**
- **Particle Swarm Optimization**

Categories of Optimization Problems
===================================

Optimization problems can be classified in various ways depending on the nature of the objective
function and constraints.

Unconstrained vs. Constrained
-----------------------------

- **Unconstrained Optimization**:
  
  .. math::

     \min_{\mathbf{x} \in \mathbb{R}^n} \; f(\mathbf{x}).

  Here, the domain is all of :math:`\mathbb{R}^n`. Local minima can be found by setting the gradient 
  to zero (if :math:`f` is differentiable).

- **Constrained Optimization**:
  
  .. math::

     \min_{\mathbf{x}} \quad f(\mathbf{x})
     \quad \text{subject to} \quad 
     g_i(\mathbf{x}) = 0, \quad h_j(\mathbf{x}) \ge 0,

  where :math:`g_i` and :math:`h_j` represent equality and inequality constraints, respectively.

Convex vs. Non-Convex
---------------------

- **Convex Optimization**: The objective function :math:`f` is convex, and the feasible region 
  (defined by the constraints) is a convex set. Convex problems have the property that any local 
  minimum is also a global minimum. These problems are generally easier to solve efficiently.

- **Non-Convex Optimization**: The objective or constraints form a non-convex problem. Such 
  problems can have multiple local minima, and finding the global minimum is often challenging.

Important Highlights
====================

Below are some key concepts in optimization that often prove critical to understanding more
advanced techniques and theory.

Lagrange Multipliers
--------------------

For a problem

.. math::

   \min_{\mathbf{x}} \; f(\mathbf{x}) 
   \quad \text{subject to} \quad g(\mathbf{x}) = 0,

we introduce a multiplier :math:`\lambda` and form the **Lagrangian**:

.. math::

   \mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda \, g(\mathbf{x}).

Stationary points of the Lagrangian (in terms of both :math:`\mathbf{x}` and :math:`\lambda`) 
can yield candidates for global or local minima.

Karush-Kuhn-Tucker (KKT) Conditions
-----------------------------------

When dealing with inequality constraints, the **KKT conditions** generalize the method of Lagrange 
multipliers. For a problem with inequality constraints :math:`h_j(\mathbf{x}) \ge 0`, the KKT 
conditions introduce multipliers :math:`\mu_j \ge 0` for each inequality constraint and require:

.. math::

   \mu_j \, h_j(\mathbf{x}) = 0,

which imposes *complementary slackness*. Together with stationarity, primal feasibility, and dual 
feasibility, the KKT conditions characterize optimal solutions for many constrained problems (especially 
convex ones).

Local vs. Global Optima
-----------------------

- **Local Minimum**: A point :math:`\mathbf{x}^*` such that :math:`f(\mathbf{x}^*) \le f(\mathbf{x})`
  for all :math:`\mathbf{x}` in a small neighborhood around :math:`\mathbf{x}^*`.
- **Global Minimum**: A point :math:`\mathbf{x}^*` such that :math:`f(\mathbf{x}^*) \le f(\mathbf{x})`
  for **all** :math:`\mathbf{x}` in the feasible domain.

In non-convex optimization, local minima may not be global. Identifying or guaranteeing global 
optimality often requires specialized techniques (e.g., branch-and-bound, global solvers,
or other heuristics).

Suggested Readings
==================

- `Wikipedia: Mathematical Optimization <https://en.wikipedia.org/wiki/Mathematical_optimization>`_
- `Nocedal, J. & Wright, S. (2006). Numerical Optimization. <https://www.springer.com/gp/book/9780387303031>`_
- `Boyd, S. & Vandenberghe, L. (2004). Convex Optimization. <http://stanford.edu/~boyd/cvxbook/>`_
- `Kirkpatrick, S. et al. (1983). "Optimization by Simulated Annealing." Science. <https://science.sciencemag.org/content/220/4598/671>`_
