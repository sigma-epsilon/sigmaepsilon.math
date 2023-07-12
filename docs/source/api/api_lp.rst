=======================
Linear Programming (LP)
=======================

The main feature of a linear programming problem (LPP) is that all functions involved,
the objective function and those expressing the constraints, must be linear. 
The appereance of a single nonlinear function anywhere, suffices to reject the problem
as an LPP.

.. tip::
   For problems starting from medium size, it is suggested to use the 
   :func:`solve_standard_form <sigmaepsilon.math.optimize.LinearProgrammingProblem.solve_standard_form>` 
   to solve linear problems.

The definition of an LPP is expected in **General Form**:

.. math::
   :nowrap:

   \begin{eqnarray}
      minimize  \quad  cx = \, & \sum_i c_i x_i  &\\
      \sum_i a_{j,i} \,x_i \,\leq\, & b_j, \qquad j &= 1, \ldots, p, \\
      \sum_i a_{j,i} \,x_i \,\geq\, & b_j, \qquad j &= p+1, \ldots, q, \\
      \sum_i a_{j,i} \,x_i \,=\, & b_j, \qquad j &= q+1, \ldots, m,
   \end{eqnarray}

where :math:`c_i, b_i`, and :math:`a_{j,i}` are the data of the problem. It can be shown, 
that all problems that admit the general form can be simplified to a 
**Standard Form**:

.. math::
   :nowrap:

   \begin{eqnarray}
      minimize  \quad  \mathbf{c}\mathbf{x} \quad under \quad 
      \mathbf{A}\mathbf{x}=\mathbf{b}, \quad \mathbf{x} \, \geq \, 
      \mathbf{0}.
   \end{eqnarray}

where :math:`\mathbf{b} \in \mathbf{R}^m, \mathbf{c} \in \mathbf{R}^n` and :math:`\mathbf{A}` is
an :math:`m \times n` matrix with :math:`n>m` and typically :math:`n` much
greater than :math:`m`.

.. autoclass:: sigmaepsilon.math.optimize.LinearProgrammingProblem
    :members:

.. autoclass:: sigmaepsilon.math.optimize.DegenerateProblemError
    :members:

.. autoclass:: sigmaepsilon.math.optimize.NoSolutionError
    :members:
