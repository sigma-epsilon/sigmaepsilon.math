=======================
Linear Programming (LP)
=======================

The main feature of a linear programming problem (LPP) is that all functions involved,
the objective function and those expressing the constraints, must be linear. 
The appereance of a single nonlinear function anywhere, suffices to reject the problem
as an LPP.

The definition of an LPP is expected in **General Form**:

.. math::
   :nowrap:

   \begin{eqnarray}
      & minimize  \quad  &cx = \, \sum_i c_i x_i \\
      & subject \, to \quad & \nonumber \\
      && \sum_i a_{ji} \,x_i \,\leq\, b_j, \qquad j = 1, \ldots, p, \\
      && \sum_i a_{ji} \,x_i \,\geq\, b_j, \qquad j = p+1, \ldots, q, \\
      && \sum_i a_{ji} \,x_i \,=\, b_j, \qquad j = q+1, \ldots, m,
   \end{eqnarray}

where :math:`c_i, b_i`, and :math:`a_{ji}` are the data of the problem.

.. autoclass:: sigmaepsilon.math.optimize.LinearProgrammingProblem
   :members:
