import itertools
from copy import copy

import numpy as np
from numpy import ndarray
from numpy.linalg import LinAlgError

from ..linalg.utils import unit_basis_vector
from .errors import DegenerateProblemError, NoSolutionError, OverDeterminedError


__all__ = ["SimplexSolverLP"]


class SimplexSolverLP:
    """
    Solves a real valued linear programming problem in standard form using the simplex method.

    .. math::
        :nowrap:

        \\begin{eqnarray}
            minimize  \quad  \mathbf{c}\mathbf{x} \quad under \quad
            \mathbf{A}\mathbf{x}=\mathbf{b}, \quad \mathbf{x} \, \geq \,
            \mathbf{0}.
        \\end{eqnarray}

    Parameters
    ----------
    c: numpy.ndarray
        Coefficients of the objective function.
    A: numpy.ndarray
        Coefficient matrix of the constraints.
    b: numpy.ndarray
        Right-hand sides of the constraints.
    """

    def __init__(self, c: ndarray, A: ndarray, b: ndarray):
        self.c = c
        self.A = A
        self.b = b
        self.x = None

        # intermediate variables
        self.B = None
        self.B_inv = None
        self.N = None
        self.xB = None
        self.xN = None
        self.cB = None
        self.cN = None
        self.order = None
        self.no_constraints = None
        self.no_variables = None
        self.no_freedom = None

    def _preprocess(self) -> None:
        self.no_constraints, self.no_variables = self.A.shape
        self.no_freedom = self.no_variables - self.no_constraints
        self.SMALLNUM = np.nextafter(0, 1)

    def _initialize_basic_solution(self) -> tuple[ndarray] | None:
        """
        Calculates a basic solution.
        """
        assert self.no_freedom > 0

        stop = False
        basic_order = None

        try:
            column_indices = list(range(self.no_variables))
            combinations = itertools.combinations(column_indices, self.no_constraints)

            while not stop:
                basic_column_indices = next(combinations)
                remaining_column_indices = np.setdiff1d(
                    column_indices, basic_column_indices
                )
                basic_order = np.concatenate(
                    (basic_column_indices, remaining_column_indices)
                )

                A_ = self.A[:, basic_order]
                B_ = A_[:, : self.no_constraints]
                B_det = np.linalg.det(B_)
                if abs(B_det) < self.SMALLNUM:
                    continue

                B_inv = np.linalg.inv(B_)
                xB = B_inv @ self.b
                stop = all(xB >= 0)
        except StopIteration:  # pragma: no cover
            raise NoSolutionError("Failed to find basic solution!")
        finally:
            if stop:
                self.B = B_
                self.B_inv = B_inv
                self.N = A_[:, self.no_constraints :]
                self.xB = xB
                self.xN = np.zeros(self.no_freedom, dtype=float)
                self.order = basic_order

    def _enter(self, i_enter: int) -> None:
        v = unit_basis_vector(self.no_freedom, i_enter, 1.0)

        # w = vector of decrements of the current solution xB
        # Only positive values are a threat to feasibility, and we
        # need to tell which of the components of xB vanishes first,
        # which, since all components of xB are posotive,
        # has to do with the positive components only.
        w_enter = self.W @ v
        i_leaving = np.argwhere(w_enter > 0)
        if len(i_leaving) == 0:
            # step size could be indefinitely increased in this
            # direction without violating feasibility, there is
            # no solution to the problem
            raise NoSolutionError("There is no solution to this problem!")

        vanishing_ratios = self.xB[i_leaving] / w_enter[i_leaving]
        # the variable that vanishes first is the one with the smallest
        # vanishing ratio
        i_leave = i_leaving.flatten()[np.argmin(vanishing_ratios)]

        # step size in the direction of current basis vector
        t = self.xB[i_leave] / w_enter[i_leave]

        # update solution
        if abs(t) < self.SMALLNUM:
            # Smallest vanishing ratio is zero, any step would
            # result in an infeasible situation.
            # -> go for the next entering variable
            return False

        self.xB -= t * w_enter
        self.xN = t * v

        self.order[self.no_constraints + i_enter], self.order[i_leave] = (
            self.order[i_leave],
            self.order[self.no_constraints + i_enter],
        )
        self.B[:, i_leave], self.N[:, i_enter] = self.N[:, i_enter], copy(
            self.B[:, i_leave]
        )
        self.B_inv = np.linalg.inv(self.B)
        self.cB[i_leave], self.cN[i_enter] = self.cN[i_enter], self.cB[i_leave]
        self.xB[i_leave], self.xN[i_enter] = self.xN[i_enter], self.xB[i_leave]
        return True

    def _unique_result(self) -> ndarray:
        self.x = np.concatenate((self.xB, self.xN))[np.argsort(self.order)]
        return self.x

    def _multiple_results(self) -> ndarray:
        assert np.all(self.reduced_costs >= 0)
        assert self.reduced_costs.min() <= self.SMALLNUM
        res = []
        res.append(self._unique_result())
        inds = np.where(np.abs(self.reduced_costs) < self.SMALLNUM)[0]
        for i in inds:
            assert self._enter(i)
            res.append(self._unique_result())
        self.x = np.stack(res)
        return self.x

    def _calculate_reduced_costs(self) -> ndarray:
        self.W = self.B_inv @ self.N
        self.reduced_costs = self.cN - self.cB @ self.W

    def _process(self) -> ndarray:
        if self.no_freedom == 0:
            try:
                self.x = np.linalg.inv(self.A) @ self.b
                return self.x
            except LinAlgError:
                raise NoSolutionError("There is no solution to this problem!")

        if not self.no_freedom > 0:
            raise OverDeterminedError(
                "There are more constraints than variables. "
                "Since the system is overdetermined, the problem might not have "
                "a feasible solution at all. Consider using a different approach."
            )

        self._initialize_basic_solution()

        c_ = self.c[self.order]
        self.cB = c_[: self.no_constraints]
        self.cN = c_[self.no_constraints :]
        del c_

        degenerate = False
        while True:
            if degenerate:
                # The objective could be decreased, but only on the expense
                # of violating positivity of the standard variables.
                # Hence, the solution is degenerate.
                raise DegenerateProblemError("The problem is ill posed!")

            # calculate reduced costs
            self._calculate_reduced_costs()
            nEntering = np.count_nonzero(self.reduced_costs < 0)
            if nEntering == 0:
                # The objective can not be further reduced.
                # There was only one basic solution, which is
                # a unique optimizer.
                d = np.count_nonzero(self.reduced_costs >= self.SMALLNUM)
                if d < len(self.reduced_costs):
                    # There is at least one component of the vector of
                    # reduced costs that is zero. This means that there
                    # are multiple solutions to the problem.
                    return self._multiple_results()
                else:
                    return self._unique_result()

            # If we reach this line, reduction of the objective is possible,
            # although maybe indefinitely. If the objective can be decreased,
            # but only on the expense of violating feasibility, the
            # solution is degenerate.
            degenerate = True

            # Candidates for entering index are the indices of the negative
            # components of the vector of reduced costs.
            i_entering = np.argsort(self.reduced_costs)[:nEntering]
            for i_enter in i_entering:
                if not self._enter(i_enter):
                    # Smallest vanishing ratio is zero, any step would
                    # result in an infeasible situation.
                    # -> go for the next entering variable
                    continue

                # break loop at the first meaningful (t != 0) decrease and
                # force recalculation of the vector of reduced costs
                degenerate = False
                break

    def solve(self) -> ndarray:
        """
        Solves the linear programming problem and returns the results as
        a NumPy array. The returned array might be
        """
        self._preprocess()
        self.x = self._process()
        return self.x
