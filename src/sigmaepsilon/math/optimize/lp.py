from typing import List, Any, Tuple, Optional, Union, Iterable
from collections import defaultdict
from enum import Enum, auto, unique
from time import time
from copy import copy, deepcopy
from contextlib import suppress

import numpy as np
from numpy import ndarray
from numpy.linalg import LinAlgError
import sympy as sy
from sympy.utilities.iterables import multiset_permutations

from sigmaepsilon.core.kwargtools import getasany

from ..function import Function, InEquality, Equality, VariableManager
from ..function.relation import Relations, Relation
from ..function.metafunction import coefficients
from .errors import DegenerateProblemError, NoSolutionError
from ..utils import atleast2d


__all__ = ["LinearProgrammingProblem"]


@unique
class LinearProgrammingResult(Enum):
    UNIQUE = auto()
    MULTIPLE = auto()
    NOSOLUTION = auto()
    DEGENERATE = auto()


class LinearProgrammingProblem:
    """
    A lightweight class to handle general linear programming problems. It gaps the 
    bridge between the general form and the standard form. The class accepts
    symbolic expressions, but this should not be expected to be too fast. 
    For problems starting from medium size, it is suggested to use the 
    `solve_standard_form` method of the class.

    Parameters
    ----------
    constraints: Iterable[Function]
        List of constraint functions.
    variables: Iterable
        List of variables of the system.
    positive: bool, Optional
        A `True` value indicated that all variables are expected to take
        only positive values. Default is `True`.
    'obj', 'cost', 'payoff', 'fittness', 'f' : Function
        The objective function.

    Examples
    --------
    The examples require `sympy` to be installed.

    (1) Example for unique solution.

    .. math::
        :nowrap:

        \\begin{eqnarray}
            & minimize&  \quad  3 x_1 + x_2 + 9 x_3 + x_4  \\\\
            & subject \, to& & \\\\
            & & x_1 + 2 x_3 + x_4 \,=\, 4, \\\\
            & & x_2 + x_3 - x_4 \,=\, 2, \\\\
            & & x_i \,\geq\, \, 0, \qquad i=1, \ldots, 4.
        \\end{eqnarray}

    >>> from sigmaepsilon.math.optimize import LinearProgrammingProblem as LPP
    >>> import sympy as sy
    >>> variables = ['x1', 'x2', 'x3', 'x4']
    >>> x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
    >>> obj1 = Function(3*x1 + 9*x3 + x2 + x4, variables=syms)
    >>> eq11 = Equality(x1 + 2*x3 + x4 - 4, variables=syms)
    >>> eq12 = Equality(x2 + x3 - x4 - 2, variables=syms)
    >>> problem = LPP(cost=obj1, constraints=[eq11, eq12], variables=syms)
    >>> problem.solve()['x']
    array([0., 6., 0., 4.])

    (2) Example for degenerate solution. The problem is on the edge of being
    ill posed, it may or may not raise an error, or return a solution.

    >>> variables = ['x1', 'x2', 'x3', 'x4']
    >>> x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
    >>> obj2 = Function(3*x1 + x2 + 9*x3 + x4, variables=syms)
    >>> eq21 = Equality(x1 + 2*x3 + x4, variables=syms)
    >>> eq22 = Equality(x2 + x3 - x4 - 2, variables=syms)
    >>> P2 = LPP(cost=obj2, constraints=[eq21, eq22], variables=syms)
    >>> # P2.solve()

    (3) Example for no solution. Solution returns None and the error
    message can be inspected.

    >>> variables = ['x1', 'x2', 'x3', 'x4']
    >>> x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
    >>> obj3 = Function(-3*x1 + x2 + 9*x3 + x4, variables=syms)
    >>> eq31 = Equality(x1 - 2*x3 - x4 + 2, variables=syms)
    >>> eq32 = Equality(x2 + x3 - x4 - 2, variables=syms)
    >>> P3 = LPP(cost=obj3, constraints=[eq31, eq32], variables=syms)
    >>> P3.solve()['e']
    [NoSolutionError('There is not solution to this problem!')]
    
    If you want the actual error to be raised, call `solve` with the option
    `raise_errors=True`.
    
    >>> P3.solve(raise_errors=True)
    Traceback (most recent call last):
        ...
    sigmaepsilon.math.optimize.errors.NoSolutionError: There is not solution to this problem!

    (4) Example for multiple solutions. We can ask for all the results
    with the option `return_all=True`. Note that the order of the solutions in the
    result are not deterministic. If you rerun the solution multiple times, you
    will see the solutions in different order.

    >>> variables = ['x1', 'x2', 'x3', 'x4']
    >>> x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
    >>> obj4 = Function(3*x1 + 2*x2 + 8*x3 + x4, variables=syms)
    >>> eq41 = Equality(x1 - 2*x3 - x4 + 2, variables=syms)
    >>> eq42 = Equality(x2 + x3 - x4 - 2, variables=syms)
    >>> P4 = LPP(cost=obj4, constraints=[eq41, eq42], variables=syms)
    >>> len(P4.solve(return_all=True)['x'])
    2
    """

    __tmpl_shift__ = "y_{}"
    __tmpl_slack__ = "z_{}"

    def __init__(
        self,
        *args,
        constraints=None,
        variables=None,
        positive=None,
        standardform=False,
        **kwargs,
    ):
        super().__init__()
        self.obj = None
        self.constraints = None
        self.standardform = None
        self.vmanager = None
        self._hook = None
        self._update(
            *args,
            constraints=constraints,
            variables=variables,
            positive=positive,
            standardform=standardform,
            **kwargs,
        )

    def _update(self, *args, variables=None, positive=None, **kwargs):
        self.standardform = kwargs.get("standardform", self.standardform)
        if len(args) > 0:
            obj = None
            if isinstance(args[0], Function):
                obj = args[0]
            if obj is not None:
                self.obj = obj
        if self.obj is None:
            self.obj = getasany(
                ["obj", "cost", "payoff", "fittness", "f"], None, **kwargs
            )
        assert self.obj is not None, "Objective must be set on instance creation!"
        self.constraints = kwargs.get("constraints", self.constraints)
        if self.constraints is None:
            self.constraints = []
        if variables is not None:
            if isinstance(positive, bool):
                self.vmanager = VariableManager(variables, positive=positive)
            else:
                self.vmanager = VariableManager(variables)
        self._hook = kwargs.get("_hook", self._hook)

    @property
    def vm(self):
        return self.vmanager

    @staticmethod
    def example_unique() -> "LinearProgrammingProblem":
        """
        Returns the following LPP:

        .. math::
            :nowrap:

            \\begin{eqnarray}
                & minimize&  \quad  3 x_1 + x_2 + 9 x_3 + x_4  \\\\
                & subject \, to& & \\\\
                & & x_1 + 2 x_3 + x_4 \,=\, 4, \\\\
                & & x_2 + x_3 - x_4 \,=\, 2, \\\\
                & & x_i \,\geq\, \, 0, \qquad i=1, \ldots, 4.
            \\end{eqnarray}

        The returned LPP has a unique solution:

        :math:`\quad \mathbf{x} = (0., 6., 0., 4.), \quad f(\mathbf{x}) = 10.`

        Example
        -------
        >>> from sigmaepsilon.math.optimize import LinearProgrammingProblem as LPP
        >>> problem = LPP.example_unique()
        >>> problem.solve()['x']
        array([0., 6., 0., 4.])
        """
        variables = ["x1", "x2", "x3", "x4"]
        x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
        obj1 = Function(3 * x1 + 9 * x3 + x2 + x4, variables=syms)
        eq11 = Equality(x1 + 2 * x3 + x4 - 4, variables=syms)
        eq12 = Equality(x2 + x3 - x4 - 2, variables=syms)
        P = LinearProgrammingProblem(
            cost=obj1, constraints=[eq11, eq12], variables=syms, positive=True
        )
        return P

    def add_constraint(self, *args, **kwargs) -> None:
        """
        Adds a new constraint to the system.
        """
        if isinstance(args[0], Function):
            if isinstance(args[0], InEquality):
                assert args[0].op in [
                    Relations.ge,
                    Relations.le,
                ], "Only '>=' and '<=' arre allowed!"
            c = args[0]
        else:
            c = Relation(*args, **kwargs)
        self.constraints.append(c)

    @property
    def variables(self) -> List[Any]:
        return self.vmanager.target()

    def _sync_variables(self) -> None:
        s = set()
        s.update(self.obj.variables)
        for c in self.constraints:
            s.update(c.variables)
        self.vmanager.add_variables(s)

    def _shift_variables(self) -> None:
        """
        Handle variables not restricted in sign.
        """
        vmap = dict()
        tmpl = self.__class__.__tmpl_shift__
        count = 1
        for v in self.vmanager.source():
            if not v.is_positive:
                sym = [tmpl.format(count), tmpl.format(count + 1)]
                si, sj = sy.symbols(sym, positive=True)
                vmap[v] = si - sj
        self.vmanager.substitute(vmap)

    def get_system_variables(self) -> list:
        """
        Returns all variables of the system.
        """
        s = set()
        s.update(self.obj.variables)
        for c in self.constraints:
            s.update(c.variables)
        return list(s)

    def get_slack_variables(self, template=None) -> list:
        """
        Returns the slack variables of the inequality constraints of the problem.
        """
        tmpl = self.__class__.__tmpl_slack__ if template is None else template
        c = self.constraints
        inequalities = list(filter(lambda c: isinstance(c, InEquality), c))
        n = len(inequalities)
        strlist = list(map(tmpl.format, range(1, n + 1)))
        y = sy.symbols(strlist, positive=True)
        for i in range(n):
            inequalities[i].slack = y[i]
        return y

    def has_standardform(self) -> bool:
        """
        Tells if the object admits the standard form of a linear program or not.

        Notes
        -----
        An LPP admits standard form if
            * only equality constraints are present
            * all variables are positive

        Returns
        -------
        bool
            `True` if the problem admits the standard form of an LPP.
        """
        all_eq = all([isinstance(c, Equality) for c in self.constraints])
        all_pos = all([v.is_positive for v in self.get_system_variables()])
        return all_pos and all_eq

    def simplify(self, maximize=False, inplace=False) -> "LinearProgrammingProblem":
        """
        Simplifies the problem so that it admits the standard form.

        This is done in 3 steps:

        1) Decomposing variables not restricted in sign according to

            :math:`x = x^{+} - x^{-}, \quad \mid x \mid = x^{+} + x^{-}`,

            where

            :math:`x^{+} = max(0, x) \geq 0, \quad x^{-} = max(0, -x) \geq 0`.

            Then the variable :math:`x` can be substituted by

            :math:`x = x^{+} - x^{-}` and the conditions :math:`\quad x^{+}, x^{-} \geq 0`.

        2) Transforming inequalities into equalities using slack variables.

            We introduce new variables

            :math:`\mathbf{y} = \mathbf{b} - \mathbf{A} \mathbf{x} \geq \mathbf{0}`,

            and formulate the extended problem

            :math:`\hat{\mathbf{A}} \mathbf{X} = \mathbf{b}`,

            where

            :math:`\mathbf{X} = (\mathbf{x} \, \, \mathbf{y}), \quad \hat{\mathbf{A}} = (\mathbf{A} \, \, \mathbf{1})`.

        3) Transform to minimization problem if necessary using the simple rule

            :math:`max(f) = -min(-f)`.

        Parameters
        ----------
        maximize: bool
            Set this true if the problem is a maximization. Default is False.
        inplace: bool
            If `True`, transformation happend in place, changing the internal structure
            ob the instance it is invoked by. In this case, `self` gets returned for
            possible continuation.

        Notes
        -----
        Steps 1) and 2) both increase the number of variables of the standard system.

        Returns
        -------
        LinearProgrammingProblem
            An LPP that admits a standard form.
        dict, Optional
            A mapping between variables of the standard and the general form.
            Only if `return_inverse` is `True`.

        Raises
        ------
        NotImplementedError
            On invalid input.
        """
        P = self if inplace else deepcopy(self)
        # handle variables
        P._sync_variables()
        P._shift_variables()
        s = P.get_slack_variables()
        P.vmanager.add_variables(s)
        v = P.vmanager.source()  # can be in arbitrary order
        n = len(v)
        x = list(
            sy.symbols(
                " ".join(["X_{}".format(i) for i in range(1, n + 1)]), positive=True
            )
        )
        c = list(sy.symbols(" ".join(["c_{}".format(i) for i in range(1, n + 1)])))
        x_v = {x_: v_ for x_, v_ in zip(x, v)}
        P.vmanager.substitute(vmap=x_v, inverse=True)
        v.append(1)
        x.append(1)
        c.append(sy.symbols("c"))
        x_c = {x_: c_ for x_, c_ in zip(x, c)}
        template = np.inner(c, x)
        x.pop(-1)
        v.pop(-1)
        vmap = P.vmanager.vmap  # == v_x, inverse of x_v
        smap = {x_v[vmap[s]]: 1 for s in s}

        def redefine_expr(fnc, aux: dict = None):
            expr = fnc.expr.subs([(v, expr) for v, expr in vmap.items()])
            fnc_coeffs = coefficients(expr=expr, normalize=True)
            coeffs = defaultdict(lambda: 0)
            if isinstance(aux, dict):
                coeffs.update(aux)
            coeffs.update({x_c[x]: c for x, c in fnc_coeffs.items()})
            return template.subs([(c_, coeffs[c_]) for c_ in c])

        def redefine_function(fnc):
            minmax = -1 if maximize else 1
            expr = minmax * redefine_expr(fnc)
            return Function(expr, variables=x, vmap=x_v)

        def redefine_equality(fnc):
            expr = redefine_expr(fnc)
            return Equality(expr, variables=x, vmap=x_v)

        def redefine_inequality(fnc):
            expr = redefine_expr(fnc, smap)
            if fnc.op == Relations.ge:
                pass
            elif fnc.op == Relations.le:
                expr *= -1
            else:
                raise NotImplementedError("Only >= and <= are allowed!")
            expr -= vmap[fnc.slack]
            eq = Equality(expr, variables=x, vmap=x_v)
            eq.slack = fnc.slack
            return eq

        obj = redefine_function(P.obj)
        _c = P.constraints
        constraints = []
        eq = list(map(redefine_equality, filter(lambda c: isinstance(c, Equality), _c)))
        constraints += eq
        if len(smap) > 0:
            ieq = list(
                map(
                    redefine_inequality, filter(lambda c: isinstance(c, InEquality), _c)
                )
            )
            constraints += ieq
        if inplace:
            self._update(constraints=constraints, variables=x, positive=True, _hook=P)
        else:
            lpp = LinearProgrammingProblem(
                obj=obj, constraints=constraints, variables=x, positive=True, _hook=P
            )
            lpp.vmanager.inv = x_v
            return lpp

    def eval_constraints(self, x: Iterable) -> Iterable:
        """
        Evaluates the constraints at `x`.
        """
        return np.array([c.f0(x) for c in self.constraints], dtype=float)

    def feasible(self, x: Iterable = None) -> bool:
        """
        Returns `True` if `x` is a feasible candidate to the current problem,
        `False` othwerise.
        """
        c = [c.relate(x) for c in self.constraints]
        if self.has_standardform():
            return all(c) and all(x >= 0)
        else:
            return all(c)

    @staticmethod
    def basic_solution(
        A: ndarray, b: ndarray, order: Optional[Union[Iterable[int], None]] = None
    ) -> Tuple[ndarray]:
        """
        Returns a basic solution to a problem the form

        .. math::
            :nowrap:

            \\begin{eqnarray}
                minimize  \quad  \mathbf{c}\mathbf{x} \quad under \quad
                \mathbf{A}\mathbf{x}=\mathbf{b}, \quad \mathbf{x} \, \geq \,
                \mathbf{0}.
            \\end{eqnarray}

        where :math:`\mathbf{b} \in \mathbf{R}^m, \mathbf{c} \in \mathbf{R}^n` and :math:`\mathbf{A}` is
        an :math:`m \\times n` matrix with :math:`n>m`.

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
        order: Optional[Union[Iterable[int], None]] = None,
        tol: Optional[float] = 1e-10,
    ):
        """Solves a linear problem in standard form:

        :math:`minimize \quad \mathbf{c} \mathbf{x} \quad under \quad \mathbf{A}\mathbf{x} = \mathbf{b}`.

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
            Results as a 1d (unique solution) or a 2d (multiple solutions)
            numpy array.

        Notes
        -----
        1) The value of the parameter `tol` is used to make judgements on the vanishing ratios
        of entering variables, therfore effects the detection of degenerate situations. The higher
        the value, the more tolerant the system is to violations.

        2) The line between the unique, the degenerate situation and having no solution at all may
        be very thin in some settings. In such a scenario, repeated solutions might return a
        a solution, a `NoSolutionError` or a `DegenerateProblemError` as well. Problems
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
        assert r > 0
        basic = LinearProgrammingProblem.basic_solution(A, b, order=order)
        if basic:
            B, B_inv, N, xB, xN, order = basic
            c_ = c[order]
            cB = c_[:m]
            cN = c_[m:]
            t = None
        else:
            raise NoSolutionError("Failed to find basic solution!")

        def unit_basis_vector(length, index=0, value=1.0):
            return value * np.bincount([index], None, length)

        def enter(i_enter):
            nonlocal B, B_inv, N, xB, xN, order, cB, cN, t
            b_enter = unit_basis_vector(r, i_enter, 1.0)

            # w = vector of decrements of the current solution xB
            # Only positive values are a threat to feasibility, and we
            # need to tell which of the components of xB vanishes first,
            # which, since all components of xB are posotive,
            # has to do with the positive components only.
            w_enter = np.matmul(W, b_enter)
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
            xN = t * b_enter

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

    def to_numpy(
        self, maximize: Optional[bool] = False, return_coeffs: Optional[bool] = False
    ) -> Tuple[ndarray, ndarray, Optional[List]]:
        """
        Returns the complete numerical representation of the standard
        form of the problem:

            :math:`minimize \quad \mathbf{c} \mathbf{x} \quad under \quad \mathbf{A}\mathbf{x} = \mathbf{b}`.

        Parameters
        ----------
        maximize : bool
            Set this true if the problem is a maximization. Default is False.
        return_coeffs : bool
            If `True`, linear coefficients of the design variables are returned as
            a sequence of `SymPy` symbols.

        Returns
        -------
        numpy.ndarray
            2d NumPy array 'A'
        numpy.ndarray
            1d NumPy array 'b'
        list, Optional
            A list of `SymPy` symbols. Only if `return_coeffs` is `True`.
        """
        if not self.has_standardform():
            P = self.simplify(maximize=maximize)
        else:
            P = self
        x = P.variables
        n = len(x)
        zeros = np.zeros((n,), dtype=float)
        b = -P.eval_constraints(zeros)
        A = []
        for c in P.constraints:
            coeffs = c.linear_coefficients(normalize=True)
            A.append(np.array([coeffs[x_] for x_ in x], dtype=float))
        A = np.vstack(A)
        coeffs = P.obj.linear_coefficients(normalize=True)
        if return_coeffs:
            coeffs = P.obj.linear_coefficients(normalize=True)
            c = np.array([coeffs[x_] for x_ in x], dtype=float)
            return A, b, c
        return A, b

    def maximize(self, *args, **kwargs) -> dict:
        """
        Solves the LPP as a maximization. For the possible arguments, and
        return types see `solve`.
        """
        kwargs["maximize"] = True
        return self.solve(*args, **kwargs)

    def minimize(self, *args, **kwargs) -> dict:
        """
        Solves the LPP as a minimization. For the possible arguments, and
        return types see `solve`.

        .. versionadded:: 1.1.0
        """
        return self.solve(*args, **kwargs)

    def solve(
        self,
        order: Optional[Union[Iterable[int], None]] = None,
        return_all: Optional[bool] = True,
        maximize: Optional[bool] = False,
        as_dict: Optional[bool] = False,
        raise_errors: Optional[bool] = False,
        tol: Optional[float] = 1e-10,
    ) -> dict:
        """
        Solves the problem as a minimization and returns the solution(s) if there are any.

        Parameters
        ----------
        order: Iterable, Optional
            The order of the variables. This might speed up finding the
            basic solution. Default is None.
        as_dict: bool
            If `True`, the results are returned as a dictionary, where the
            keys are sympy symbols of the variables of the problem.
        raise_errors: bool
            If `True`, the solution raises the actual errors on exception events,
            otherwise they get returned within the result, under key `e`.
        tol: float, Optional
            Floating point tolerance. Default is 1e-10.

        Notes
        -----
        It is assumed that this function gets invoked on relatively small problems.
        For large-scale situations, we suggest to use `solve_standard_form` function
        of this class instead. Contrary to this one, it rases errors, while the more
        high-level `solve` method of the instance catches those errors, and returns
        them as part of the returned dictionary.

        Returns
        -------
        dict
            A dictionary with the following items:
            x: numpy.ndarray or dict
                The solution as an array or a dictionary, depending on your input.
            e: Iterable
                A list of errors that occured during solution.
            time: dict
                A dictionary with information of execution times of the main stages
                of the calculation.
        """
        summary = {"time": {}, "x": None, "e": []}
        try:
            # general form -> standard form
            _t0 = time()
            P = self.simplify(maximize=maximize, inplace=False)
            A, b, c = P.to_numpy(maximize=False, return_coeffs=True)
            summary["time"]["preproc"] = time() - _t0

            # calculate solution
            _cls = self.__class__
            _t0 = time()
            x = _cls.solve_standard_form(A, b, c, order=order, tol=tol)
            summary["time"]["solution"] = time() - _t0

            # standard form -> general form
            _t0 = time()
            res = None
            if x is not None:
                svars = P.variables  # standard variables
                gvars = self.variables  # general variables
                if not return_all:
                    x = x[0]
                x = atleast2d(x)
                if as_dict:
                    arr = []
                    gdata = {g: [] for g in gvars}
                    for i in range(x.shape[0]):
                        smap = {s: sx for s, sx in zip(svars, x[i])}
                        vm = P._hook.vm.substitute(smap, inplace=False)
                        [gdata[g].append(vm[g]) for g in gvars]
                    res = {
                        g: np.squeeze(np.array(gdata[g], dtype=float)) for g in gvars
                    }
                else:
                    arr = []
                    for i in range(x.shape[0]):
                        smap = {s: sx for s, sx in zip(svars, x[i])}
                        vm = P._hook.vm.substitute(smap, inplace=False)
                        vals = [vm[g] for g in gvars]
                        arr.append(np.array(list(vals), dtype=float))
                    res = np.squeeze(np.stack(arr))
            summary["x"] = res
            summary["time"]["postproc"] = time() - _t0
        except Exception as e:
            summary["e"].append(e)
        finally:
            if len(summary["e"]) > 0:
                summary["time"]["solution"] = None
                if raise_errors:
                    raise summary["e"][0]
            return summary
