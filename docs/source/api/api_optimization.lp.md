---
myst:
  html_meta:
    description: "API reference of sigmaepsilon.math.optimize.LinearProgrammingProblem, for defining and solving continuous, integer and mixed-integer linear programming problems."
---

(api_optimization_LP)=

# Linear Programming (LP)

The main feature of a linear programming problem (LPP) is that all functions involved,
the objective function and those expressing the constraints, must be linear.
The appearance of a single nonlinear function anywhere suffices to reject the problem
as an LPP.

The definition of an LPP is expected in **General Form**:

$$
\begin{aligned}
& \text{minimize} \quad & c x = \sum_i c_i x_i & \\
& \text{subject to} \quad & \sum_i a_{ji} \, x_i \leq b_j, & \qquad j = 1, \ldots, p, \\
& & \sum_i a_{ji} \, x_i \geq b_j, & \qquad j = p+1, \ldots, q, \\
& & \sum_i a_{ji} \, x_i = b_j, & \qquad j = q+1, \ldots, m,
\end{aligned}
$$

where $c_i, b_i$, and $a_{ji}$ are the data of the problem.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   ~sigmaepsilon.math.optimize.LinearProgrammingProblem
```
