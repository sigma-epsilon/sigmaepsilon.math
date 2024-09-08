from contextlib import suppress
from typing import Iterable
from copy import copy

import numpy as np
from numpy import ndarray
from numpy.linalg import LinAlgError
from sympy.utilities.iterables import multiset_permutations

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

    def _preprocess(self) -> None:
        self.M, self.N = self.A.shape
        self.R = self.N - self.M
        self.SMALLNUM = np.nextafter(0, 1)

    @staticmethod
    def basic_solution(
        A: ndarray, b: ndarray, order: Iterable[int] | None = None
    ) -> tuple[ndarray] | None:
        """
        Returns a basic (aka. extremal) solution to a problem in the form

        .. math::
            :nowrap:

            \\begin{eqnarray}
                minimize  \quad  \mathbf{c}\mathbf{x} \quad under \quad
                \mathbf{A}\mathbf{x}=\mathbf{b}, \quad \mathbf{x} \, \geq \,
                \mathbf{0}.
            \\end{eqnarray}

        where :math:`\mathbf{b} \in \mathbf{R}^m, \mathbf{c} \in \mathbf{R}^n` and :math:`\mathbf{A}` is
        an :math:`m \\times n` matrix with :math:`n>m`.

        If the function is unable to find a basic solution, it returns `None`.

        Parameters
        ----------
        A: numpy.ndarray
            An :math:`m \times n` matrix with :math:`n>m`
        b: numpy.ndarray
            Right-hand sides. :math:`\mathbf{b} \in \mathbf{R}^m`
        order: Iterable[int], Optional
            An arbitrary permutation of the indices to start with.

        Returns
        -------
        numpy.ndarray
            Coefficient matrix :math:`\mathbf{B}`
        numpy.ndarray
            Inverse of coefficient matrix :math:`\mathbf{B}^{-1}`
        numpy.ndarray
            Coefficient matrix :math:`\mathbf{N}`
        numpy.ndarray
            Basic solution :math:`\mathbf{x}_{B}`
        numpy.ndarray
            Remaining solution :math:`\mathbf{x}_{N}`
        """
        m, n = A.shape
        r = n - m
        assert r > 0

        stop = False
        basic_order = None
        try:
            with suppress(StopIteration):
                """
                StopIteration:

                There is no permutation of columns that would produce a regular
                mxm submatrix
                    -> there is no feasible basic solution
                        -> there is no feasible solution
                """
                if order is not None:
                    if isinstance(order, Iterable):
                        permutations = iter([order])
                else:
                    order = [i for i in range(n)]
                    permutations = multiset_permutations(order)

                while not stop:
                    order = next(permutations)

                    A_ = A[:, order]
                    B_ = A_[:, :m]

                    with suppress(LinAlgError):
                        B_inv = np.linalg.inv(B_)
                        xB = np.matmul(B_inv, b)
                        stop = all(xB >= 0)
                        basic_order = order if stop else None
                        """
                        LinAlgError:
                        
                        If there is no error, it means that calculation
                        of xB was succesful, which is only possible if the
                        current permutation defines a positive definite submatrix.
                        Note that this is cheaper than checking the eigenvalues,
                        since it only requires the eigenvalues to be all positive,
                        and does not involve calculating their actual values.
                        """
        finally:
            if stop:
                N_ = A_[:, m:]
                xN = np.zeros(r, dtype=float)
                assert basic_order is not None
                return B_, B_inv, N_, xB, xN, basic_order
            else:
                return None

    def _enter(self, i_enter: int) -> None:
        v = unit_basis_vector(self.R, i_enter, 1.0)

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

        self.order[self.M + i_enter], self.order[i_leave] = (
            self.order[i_leave],
            self.order[self.M + i_enter],
        )
        self.B[:, i_leave], self.N[:, i_enter] = self.N[:, i_enter], copy(
            self.B[:, i_leave]
        )
        self.B_inv = np.linalg.inv(self.B)
        self.cB[i_leave], self.cN[i_enter] = self.cN[i_enter], self.cB[i_leave]
        self.xB[i_leave], self.xN[i_enter] = self.xN[i_enter], self.xB[i_leave]
        return True

    def _unique_result(self) -> ndarray:
        return np.concatenate((self.xB, self.xN))[np.argsort(self.order)]

    def _multiple_results(self) -> ndarray:
        assert np.all(self.reduced_costs >= 0)
        assert self.reduced_costs.min() <= self.SMALLNUM
        res = []
        res.append(self._unique_result())
        inds = np.where(np.abs(self.reduced_costs) < self.SMALLNUM)[0]
        for i in inds:
            assert self._enter(i)
            res.append(self._unique_result())
        return np.stack(res)

    def _calculate_reduced_costs(self) -> ndarray:
        self.W = self.B_inv @ self.N
        self.reduced_costs = self.cN - self.cB @ self.W

    def _process(self) -> ndarray:
        if self.R == 0:
            try:
                self.x = np.linalg.inv(self.A) @ self.b
                return
            except LinAlgError:
                raise NoSolutionError("There is no solution to this problem!")

        if not self.R > 0:
            raise OverDeterminedError(
                "There are more constraints than variables. "
                "Since the system is overdetermined, the problem might not have "
                "a feasible solution at all. Consider using a different approach."
            )

        basic = self.basic_solution(self.A, self.b)
        if basic:
            self.B, self.B_inv, self.N, self.xB, self.xN, self.order = basic
            c_ = self.c[self.order]
            self.cB = c_[: self.M]
            self.cN = c_[self.M :]
            del c_
            self.t = None
        else:
            raise NoSolutionError("Failed to find basic solution!")

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
