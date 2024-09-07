from typing import Iterable
from collections import defaultdict
from enum import Enum, auto, unique
from copy import copy, deepcopy
from contextlib import suppress

import numpy as np
from numpy import ndarray
from numpy.linalg import LinAlgError
import sympy as sy
from sympy.utilities.iterables import multiset_permutations
from sympy import Symbol
from pydantic import BaseModel

from ..function import Function, InEquality, Equality
from ..function.relation import Relations, Relation
from ..function.symutils import coefficients, generate_symbols, substitute
from .errors import DegenerateProblemError, NoSolutionError, OverDeterminedError
from ..utils import atleast2d


__all__ = [
    "LinearProgrammingProblem",
    "LinearProgrammingStatus",
    "LinearProgrammingResult",
]


@unique
class LinearProgrammingStatus(Enum):
    """
    An enumeration of the possible statuses of a linear programming problem.
    The explanation of each status is as follows:

    * UNIQUE: The problem has a unique solution.
    * MULTIPLE: The problem has multiple solutions.
    * NOSOLUTION: The problem has no solution.
    * DEGENERATE: The problem is degenerate.
    * OVERDETERMINED: The problem is overdetermined.
    * FAILED: The problem might have a solution, but the solver failed to get it.
    """

    UNIQUE = auto()
    MULTIPLE = auto()
    NOSOLUTION = auto()
    DEGENERATE = auto()
    OVERDETERMINED = auto()
    FAILED = auto()


class LinearProgrammingResult(BaseModel):
    status: LinearProgrammingStatus = LinearProgrammingStatus.FAILED
    x: list[float | int] | None = None
    errors: list[str] = []
    success: bool = False


class LinearProgrammingProblem:
    """
    A class to handle real valued linear programming problems.
    
    To define the objective and the constraints of the problem,
    you can use the `Function`, `Relation`, `Equality` and `InEquality` 
    classes, or `SymPy` expressions directly.

    Parameters
    ----------
    obj: Function
        The objective function.
    constraints: Iterable[Relation]
        The constraints.
        
    See Also
    --------
    :class:`~sigmaepsilon.math.function.function.Function`
    :class:`~sigmaepsilon.math.function.relation.Relation`
    :class:`~sigmaepsilon.math.function.relation.Equality`
    :class:`~sigmaepsilon.math.function.relation.InEquality`

    Examples
    --------
    The following example solves a problem with a unique solution.

    .. math::
        :nowrap:

        \\begin{eqnarray}
            & minimize&  \quad  3 x_1 + x_2 + 9 x_3 + x_4  \\\\
            & subject \, to& & \\nonumber\\\\
            & & x_1 + 2 x_3 + x_4 \,=\, 4, \\\\
            & & x_2 + x_3 - x_4 \,=\, 2, \\\\
            & & x_i \,\geq\, \, 0, \qquad i=1, \ldots, 4.
        \\end{eqnarray}

    >>> from sigmaepsilon.math.optimize import LinearProgrammingProblem as LPP
    >>> import sympy as sy
    >>> x1, x2, x3, x4 = variables = sy.symbols('x1:5', nonnegative=True)
    >>> obj = Function(3*x1 + 9*x3 + x2 + x4, variables=variables)
    >>> eq1 = Equality(x1 + 2*x3 + x4 - 4, variables=variables)
    >>> eq2 = Equality(x2 + x3 - x4 - 2, variables=variables)
    >>> problem = LPP(obj, [eq1, eq2])
    >>> problem.solve().x
    [0., 6., 0., 4.]
    
    """

    __slots__ = ["obj", "constraints", "_vmap", "variables", "_original_variables"]

    __tmpl_surplus__ = r"\beta_{}"
    __tmpl_slack__ = r"\gamma_{}"
    __tmpl_standard__ = r"\alpha_{}"
    __tmpl_coeff__ = "c_{}"

    def __init__(
        self,
        obj: Function,
        constraints: Iterable[Relation],
    ):
        super().__init__()
        self.obj = None
        self.constraints = []
        self.variables = []
        self._vmap = dict()

        if isinstance(obj, Function):
            if not obj.is_symbolic:
                raise ValueError("The objective function must be symbolic!")

            self.obj = obj
            self.variables += self.obj.variables
        else:
            raise TypeError("The objective function must be an intance of `Function`!")

        for constraint in constraints:
            if isinstance(constraint, Relation):
                if not constraint.is_symbolic:
                    raise ValueError("The constraints must be symbolic!")

                self.constraints.append(constraint)
                self.variables += constraint.variables
            else:
                raise TypeError("The constraints must be instances of `Relation`!")

        self.variables = list(set(self.variables))
        self._original_variables = deepcopy(self.variables)

    def _substitute(
        self, vmap: dict, inverse: bool = False, inplace: bool = True
    ) -> None:
        """
        Substitute the variables with the corresponding expressions.
        """
        if not inverse:
            old = list(vmap.values())
            new = list(vmap.keys())
        else:
            old = list(vmap.keys())
            new = list(vmap.values())

        result = self._vmap if inplace else dict()
        for v, expr in self._vmap.items():
            result[v] = substitute(expr, old, new)

        return result

    def _transform_variables(self) -> None:
        """
        Handle variables not in the form `x >= 0`.
        """
        vmap = dict()
        tmpl = self.__class__.__tmpl_surplus__
        counter = 1

        for v in self._vmap.keys():
            if v.is_nonnegative:
                pass  # there is nothing to do, variable is already nonnegative
            elif v.is_positive:
                sym = sy.symbols(tmpl.format(counter), nonnegative=True)
                vmap[v] = sym + np.nextafter(0, 1)
                counter += 1
            elif v.is_negative:
                sym = sy.symbols(tmpl.format(counter), nonnegative=True)
                vmap[v] = -sym - np.nextafter(0, 1)
                counter += 1
            elif v.is_nonpositive:
                sym = sy.symbols(tmpl.format(counter), nonnegative=True)
                vmap[v] = -sym
                counter += 1
            else:  # unrestricted
                sym = [
                    tmpl.format(counter),
                    tmpl.format(counter + 1),
                ]
                si, sj = sy.symbols(sym, nonnegative=True)
                vmap[v] = si - sj
                counter += 2

        self._substitute(vmap, inverse=False)

    def _has_integer_variables(self) -> bool:
        return any([v.is_integer for v in self.variables])

    def _get_slack_variables(self, template: str | None = None) -> list[Symbol]:
        tmpl = self.__class__.__tmpl_slack__ if template is None else template
        c = self.constraints
        inequalities = list(filter(lambda c: isinstance(c, InEquality), c))
        n = len(inequalities)
        symbols_str = list(map(tmpl.format, range(1, n + 1)))
        slack_variables = sy.symbols(symbols_str, nonnegative=True)
        for i in range(n):
            inequalities[i].slack = slack_variables[i]
        return slack_variables

    def _has_standardform(self) -> bool:
        all_eq = all([isinstance(c, Equality) for c in self.constraints])
        all_pos = all([v.is_nonnegative for v in self.variables])
        return all_pos and all_eq

    def _get_target_variables(self) -> list[Symbol]:
        """
        Returns the target variables of the problem.
        """
        s = set()
        for expr in self._vmap.values():
            s.update(expr.free_symbols)
        return list(s)

    def _to_standard_form(
        self, maximize: bool = False, inplace: bool = False
    ) -> "LinearProgrammingProblem":
        P = self if inplace else deepcopy(self)

        if P._has_integer_variables():
            raise NotImplementedError("Integer variables are not supported yet!")

        # initialize the variable mapper
        P._vmap = {v: v for v in self.variables}

        # handle variables not restricted in sign
        P._transform_variables()

        # gather and add slack variables to the variable manager
        slack_variables = P._get_slack_variables()
        P._vmap.update({v: v for v in slack_variables})

        # standardize variables
        general_variables = P._get_target_variables()
        number_of_general_variables = len(general_variables)
        variable_indices = range(1, number_of_general_variables + 1)
        standard_variables = generate_symbols(
            self.__tmpl_standard__, variable_indices, nonnegative=True
        )
        standard_coefficients = generate_symbols(self.__tmpl_coeff__, variable_indices)
        standard_to_general = {
            k: v for k, v in zip(standard_variables, general_variables)
        }
        P._substitute(vmap=standard_to_general, inverse=True)

        # create template for a general linear function
        general_variables.append(1)
        standard_variables.append(1)
        standard_coefficients.append(sy.symbols("c"))
        standard_to_coeff = {
            k: v for k, v in zip(standard_variables, standard_coefficients)
        }
        template = np.inner(standard_coefficients, standard_variables)
        standard_variables.pop(-1)
        general_variables.pop(-1)

        original_to_standard = {v: P._vmap[v] for v in self.variables}
        slack_to_standard = {s: P._vmap[s] for s in slack_variables}

        def expr_to_standard(fnc: Function):
            expr = fnc.expr.subs(
                [(v, expr) for v, expr in original_to_standard.items()]
            )
            expr_coeffs = coefficients(expr=expr, normalize=True)
            coeff_map = defaultdict(lambda: 0)
            coeff_map.update({standard_to_coeff[x]: c for x, c in expr_coeffs.items()})
            return template.subs([(c, coeff_map[c]) for c in standard_coefficients])

        def tr_obj(fnc: Function) -> Function:
            minmax = -1 if maximize else 1
            expr = minmax * expr_to_standard(fnc)
            return Function(
                expr, variables=standard_variables, vmap=standard_to_general
            )

        def tr_eq(fnc: Equality) -> Equality:
            expr = expr_to_standard(fnc)
            return Equality(
                expr, variables=standard_variables, vmap=standard_to_general
            )

        def tr_ieq(fnc: InEquality) -> Equality:
            expr = expr_to_standard(fnc)

            # bring inequality to the form expr >= 0
            if fnc.op == Relations.le:
                expr *= -1
            elif fnc.op == Relations.gt:
                expr -= np.nextafter(0, 1)
            elif fnc.op == Relations.lt:
                expr *= -1
                expr -= np.nextafter(0, 1)

            # handle inequality in the form of expr >= 0 with slack variable
            expr -= slack_to_standard[fnc.slack]
            eq = Equality(expr, variables=standard_variables, vmap=standard_to_general)
            eq.slack = fnc.slack
            fnc.slack = None

            return eq

        # transform the objective function and the constraints
        obj = tr_obj(P.obj)
        constraints = []
        constraints += list(
            map(tr_eq, filter(lambda c: isinstance(c, Equality), P.constraints))
        )
        if len(slack_variables) > 0:
            constraints += list(
                map(tr_ieq, filter(lambda c: isinstance(c, InEquality), P.constraints))
            )

        P.obj = obj
        P.constraints = constraints
        P.variables = standard_variables
        return P

    def _eval_constraints(self, x: Iterable) -> ndarray:
        return np.array([c(x) for c in self.constraints], dtype=float)

    def is_feasible(self, x: Iterable) -> bool:
        """
        Returns `True` if `x` is a feasible candidate to the current problem,
        `False` othwerise. This function can be used for instance if you want to
        transform the problem to an unconstrained optimization problem and solve it
        with a nonlinear solver like the genetic algorithm.

        Parameters
        ----------
        x: Iterable
            The candidate solution. The length of the iterable must be equal to the
            number of variables in the problem.

        Notes
        -----
        The order of variables in the argument `x` must follow the order of the variables
        in the functions that make up the problem, which is assumed to be uniform across
        all functions.
        """
        c = [c.relate(x) for c in self.constraints]
        if self._has_standardform():
            x = np.array(x, dtype=float)
            return all(c) and all(x >= 0)
        else:
            return all(c)

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
                return B_, B_inv, N_, xB, xN, order
            else:
                return None

    @staticmethod
    def solve_standard_form(
        A: ndarray,
        b: ndarray,
        c: ndarray,
        order: Iterable[int] | None = None,
        tol: float = 1e-10,
    ) -> ndarray:
        """
        Solves a linear problem in standard form:

        .. math::
           :nowrap:

           \\begin{eqnarray}
               minimize  \\quad  \\mathbf{c} \\mathbf{x} \\quad under \\quad
               \\mathbf{A} \\mathbf{x} = \\mathbf{b}, \\quad \\mathbf{x} \\, \\geq \\,
               \\mathbf{0}.
           \\end{eqnarray}

        See the notes section for the behaviour and the possible gotchas.

        Parameters
        ----------
        A: numpy.ndarray
            2d float array.
        b: numpy.ndarray
            1d float array.
        c: numpy.ndarray
            1d float array.
        order: Iterable, Optional
            The order of the variables.
        tol: float, Optional
            Floating point tolerance. Default is 1e-10.

        Returns
        -------
        numpy.ndarray
            A 1d (unique solution) or a 2d (multiple solutions) numpy array.

        Notes
        -----
        1) The value of the parameter `tol` is used to make judgements on the vanishing ratios
        of entering variables, therfore effects the detection of degenerate situations. The higher
        the value, the more tolerant the system is to violations.

        2) The line between the unique, the degenerate situation and having no solution at all may
        be very thin in some settings. In such a scenario, repeated solutions might return a
        a solution, a `NoSolutionError` or a `DegenerateProblemError`. Problems
        with this behaviour are all considered degenerate, and suggest an ill-posed setup.

        Raises
        ------
        NoSolutionError
            If there is no solution to the problem.
        DegenerateProblemError
            If the problem is degenerate.
        """
        m, n = A.shape
        r = n - m

        if r == 0:
            try:
                return np.linalg.inv(A) @ b
            except LinAlgError:
                raise NoSolutionError("There is no solution to this problem!")

        if not r > 0:  # pragma: no cover
            raise OverDeterminedError(
                "There are more constraints than variables. "
                "Since the system is overdetermined, the problem might not have "
                "a feasible solution at all. Consider using a different approach."
            )

        basic = LinearProgrammingProblem.basic_solution(A, b, order=order)
        if basic:
            B, B_inv, N, xB, xN, order = basic
            c_ = c[order]
            cB = c_[:m]
            cN = c_[m:]
            t = None
        else:
            raise NoSolutionError("Failed to find basic solution!")

        def unit_basis_vector(
            length: int, index: int = 0, value: float = 1.0
        ) -> ndarray:
            return value * np.bincount([index], None, length)

        def enter(i_enter):
            nonlocal B, B_inv, N, xB, xN, order, cB, cN, t
            v = unit_basis_vector(r, i_enter, 1.0)

            # w = vector of decrements of the current solution xB
            # Only positive values are a threat to feasibility, and we
            # need to tell which of the components of xB vanishes first,
            # which, since all components of xB are posotive,
            # has to do with the positive components only.
            w_enter = np.matmul(W, v)
            i_leaving = np.argwhere(w_enter > 0)
            if len(i_leaving) == 0:
                # step size could be indefinitely increased in this
                # direction without violating feasibility, there is
                # no solution to the problem
                raise NoSolutionError("There is no solution to this problem!")

            vanishing_ratios = xB[i_leaving] / w_enter[i_leaving]
            # the variable that vanishes first is the one with the smallest
            # vanishing ratio
            i_leave = i_leaving.flatten()[np.argmin(vanishing_ratios)]

            # step size in the direction of current basis vector
            t = xB[i_leave] / w_enter[i_leave]

            # update solution
            if abs(t) <= tol:
                # Smallest vanishing ratio is zero, any step would
                # result in an infeasible situation.
                # -> go for the next entering variable
                return False
            xB -= t * w_enter
            xN = t * v

            order[m + i_enter], order[i_leave] = order[i_leave], order[m + i_enter]
            B[:, i_leave], N[:, i_enter] = N[:, i_enter], copy(B[:, i_leave])
            B_inv = np.linalg.inv(B)
            cB[i_leave], cN[i_enter] = cN[i_enter], cB[i_leave]
            xB[i_leave], xN[i_enter] = xN[i_enter], xB[i_leave]
            return True

        def unique_result() -> ndarray:
            return np.concatenate((xB, xN))[np.argsort(order)]

        def multiple_results() -> ndarray:
            assert np.all(reduced_costs >= 0)
            assert reduced_costs.min() <= tol
            inds = np.where(reduced_costs <= tol)[0]
            res = [
                unique_result(),
            ]
            for i in inds:
                assert enter(i)
                res.append(unique_result())
            return np.stack(res)

        degenerate = False
        while True:
            if degenerate:
                # The objective could be decreased, but only on the expense
                # of violating positivity of the standard variables.
                # Hence, the solution is degenerate.
                raise DegenerateProblemError("The problem is ill posed!")

            # calculate reduced costs
            W = np.matmul(B_inv, N)
            reduced_costs = cN - np.matmul(cB, W)
            nEntering = np.count_nonzero(reduced_costs < 0)
            if nEntering == 0:
                # The objective can not be further reduced.
                # There was only one basic solution, which is
                # a unique optimizer.
                d = np.count_nonzero(reduced_costs >= tol)
                if d < len(reduced_costs):
                    # there are edges along with the objective does
                    # not increase
                    # dc = np.abs(reduced_costs - reduced_costs.min())
                    # inds = np.where(dc <= tol)[0]
                    return multiple_results()
                else:
                    return unique_result()

            # If we reach this line, reduction of the objective is possible,
            # although maybe indefinitely. If the objective can be decreased,
            # but only on the expense of violating feasibility, the
            # solution is degenerate.
            degenerate = True

            # Candidates for entering index are the indices of the negative
            # components of the vector of reduced costs.
            i_entering = np.argsort(reduced_costs)[:nEntering]
            for i_enter in i_entering:
                if not enter(i_enter):
                    # Smallest vanishing ratio is zero, any step would
                    # result in an infeasible situation.
                    # -> go for the next entering variable
                    continue

                # break loop at the first meaningful (t != 0) decrease and
                # force recalculation of the vector of reduced costs
                degenerate = False
                break

    def _to_numpy(
        self,
        maximize: bool = False,
        assume_standard: bool = False,
    ) -> tuple[ndarray, ndarray, ndarray]:
        """
        Returns the arrays A, b and c.
        """
        if not (assume_standard or self._has_standardform()):
            P = self._to_standard_form(maximize=maximize, inplace=False)
        else:
            P = self

        x = P.variables

        zeros = np.zeros((len(x),), dtype=float)
        b = -P._eval_constraints(zeros)

        A = []
        for c in P.constraints:
            coeffs = c.linear_coefficients(normalize=True)
            A.append(np.array([coeffs[x_] for x_ in x], dtype=float))
        A = np.vstack(A)

        coeffs = P.obj.linear_coefficients(normalize=True)
        c = np.array([coeffs[x_] for x_ in x], dtype=float)

        return A, b, c

    def solve(
        self,
        *,
        return_all: bool = True,
        maximize: bool = False,
        raise_errors: bool = False,
        tol: float = 1e-10,
        _order: Iterable[int] | None = None,
    ) -> LinearProgrammingResult:
        """
        Solves the problem and returns the solution(s) if there are any.

        Parameters
        ----------
        raise_errors: bool
            If `True`, the solution raises the errors during solution,
            otherwise they get returned within the result, under key `e`
            without being explicitly raised.
        tol: float, Optional
            Floating point tolerance. Default is `1e-10`.
        maximize: bool
            Set this to `True` if the problem is a maximization. Default is `False`.
        return_all: bool
            If `True`, and there are multiple solutions, all of them are returned.
            Default is `True`.

        Returns
        -------
        LinearProgrammingResult

        Raises
        ------
        NoSolutionError
            If there is no solution to the problem.
        DegenerateProblemError
            If the problem is degenerate.
        OverDeterminedError
            If the problem is overdetermined.
        NotimplementedError
            If the problem is not yet supported, for instance if it contains integer variables.
        """
        result = LinearProgrammingResult()
        errors = []
        try:
            P = self._to_standard_form(maximize=maximize, inplace=False)
            A, b, c = P._to_numpy(maximize=False, assume_standard=True)
            x = self.solve_standard_form(A, b, c, order=_order, tol=tol)

            res = None

            standard_variables = P.variables
            original_variables = P._original_variables

            if len(x.shape) == 1:
                result.status
                result.status = LinearProgrammingStatus.UNIQUE
            elif len(x.shape) == 2:
                result.status = LinearProgrammingStatus.MULTIPLE

            if not return_all:
                x = x[0]
            x = atleast2d(x)

            original_values = {var: [] for var in original_variables}

            for i in range(x.shape[0]):
                smap = {s: sx for s, sx in zip(standard_variables, x[i])}
                vmap = P._substitute(smap, inplace=False)
                [original_values[g].append(vmap[g]) for g in original_variables]

            res = {
                g: np.squeeze(np.array(original_values[g], dtype=float))
                for g in original_variables
            }

            res = np.array([res[v] for v in original_variables]).T

            result.x = res.tolist()
            result.success = True
        except Exception as e:
            errors.append(e)
            result.success = False
            if isinstance(e, NoSolutionError):
                result.status = LinearProgrammingStatus.NOSOLUTION
            elif isinstance(e, DegenerateProblemError):
                result.status = LinearProgrammingStatus.DEGENERATE
            elif isinstance(e, OverDeterminedError):
                result.status = LinearProgrammingStatus.OVERDETERMINED
            else:
                result.status = LinearProgrammingStatus.FAILED
        finally:
            if len(errors) > 0:
                result.x = None
                result.errors = [str(e) for e in errors]

                if raise_errors:
                    raise errors[0]

            return result
