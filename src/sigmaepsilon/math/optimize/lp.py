from typing import Iterable, Sequence

import numpy as np
from numpy import ndarray
from sympy import Symbol
from scipy.optimize import linprog, OptimizeResult

from ..function import Function, InEquality, Equality
from ..function.relation import Relations, Relation
from ..mathtypes import BoundsLike

__all__ = ["LinearProgrammingProblem"]


class LinearProgrammingProblem:
    """
    A class to solve linear programming problems [1]_. It uses the :func:`scipy.optimize.linprog`
    function as a solver [2]_, which eventually calls into the HIGHS solver [3]_.
    
    To define the objective and the constraints,
    you can use the `Function`, `Relation`, `Equality` and `InEquality` 
    classes. These are able to understand SymPy expressions and strings, giving you
    the flexibility in defining the problem.

    Parameters
    ----------
    obj : Function
        The objective function.
    constraints : Sequence[Relation]
        The constraints.
    variables : Sequence[Symbol], Optional
        The variables of the problem. The variables must be provided for symbolic functions
        to guarantee that the order of the variables is consistent with the symbolic functions
        that make up the problem.
    bounds : BoundsLike, Optional
        See the `bounds` parameter in `scipy.optimize.linprog` for more details.
    integrality : Iterable[int] | int, Optional
        See the `integrality` parameter in `scipy.optimize.linprog` for more details.
        
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

        \\begin{eqnarray}
            & minimize&  \\quad  3 x_1 + x_2 + 9 x_3 + x_4  \\\\
            & subject \\, to& & \\nonumber\\\\
            & & x_1 + 2 x_3 + x_4 \\,=\\, 4, \\\\
            & & x_2 + x_3 - x_4 \\,=\\, 2, \\\\
            & & x_i \\,\\geq\\, \\, 0, \\qquad i=1, \\ldots, 4.
        \\end{eqnarray}

    >>> from sigmaepsilon.math.optimize import LinearProgrammingProblem as LPP
    >>> from sigmaepsilon.math.function import Function, Relation
    >>> import sympy as sy
    >>> 
    >>> x1, x2, x3, x4 = variables = sy.symbols('x1:5')
    >>> obj = Function(3*x1 + 9*x3 + x2 + x4, variables=variables)
    >>> eq1 = Relation(x1 + 2*x3 + x4 - 4, variables=variables)
    >>> eq2 = Relation(x2 + x3 - x4 - 2, variables=variables)
    >>>
    >>> bounds = [(0, None), (0, None), (0, None), (0, None)]
    >>> problem = LPP(obj, [eq1, eq2], variables=variables, bounds=bounds)
    >>> problem.solve().x
    array([0., 6., 0., 4.])
    
    In this example, bounds could have been specified as `bounds=(0, None)` as
    well, since all variables have the same bound.
    
    Now let see how to solve the following mixed integer linear programming problem.
    
    .. math::

        \\begin{eqnarray}
            & minimize&  \\quad  3 x_2 + 2 x_3  \\\\
            & subject \\, to& & \\nonumber\\\\
            & & 2 x_1 + 2 x_2 - 4 x_3 \\,=\\, 5, \\\\
            & & x_i \\,\\geq\\, \\, 0, \\qquad i=1, \\ldots, 4. \\\\
            & & x_1, x_3 \\,\\in\\, \\mathbb{Z}.
        \\end{eqnarray}
    
    >>> variables = x1, x2, x3 = sy.symbols(["x1", "x2", "x3"])
    >>> f = Function(3 * x2 + 2 * x3, variables=variables)
    >>> eq = Relation(2 * x1 + 2 * x2 - 4 * x3 - 5, op="=", variables=variables)
    >>> bounds = (0, None)
    >>> integrality = [1, 0, 1]
    >>> problem = LPP(f, [eq], variables=variables, bounds=bounds, integrality=integrality)
    
    These integrality constraints can also be specified using assumptions on the
    symbolic variables.
    
    >>> x1, x3 = sy.symbols(["x1", "x3"], integer=True)
    >>> x2 = sy.symbols("x2")
    >>> variables = x1, x2, x3
    >>> f = Function(3 * x2 + 2 * x3, variables=variables)
    >>> eq = Relation("2 * x1 + 2 * x2 - 4 * x3 = 5", variables=variables)
    >>> problem = LPP(f, [eq], variables=variables, bounds=(0, None))
    
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Linear_programming
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    .. [3] https://highs.dev/
    """

    __slots__ = [
        "obj",
        "constraints",
        "_variables",
        "bounds",
        "integrality",
    ]

    def __init__(
        self,
        obj: Function,
        constraints: Sequence[Relation],
        *,
        variables: Sequence[Symbol] | None = None,
        bounds: BoundsLike = (0, None),
        integrality: Iterable[int] | int = None,
    ):
        super().__init__()
        self.obj = None
        self.constraints = []
        self._variables = []
        self.bounds = bounds
        self.integrality = integrality

        _variables = []

        if variables is None:
            raise ValueError("The variables must be provided for symbolic problems!")

        if isinstance(obj, Function):
            if not obj.is_symbolic:
                raise ValueError("The objective function must be symbolic!")

            self.obj = obj
            _variables += self.obj.variables
        else:
            raise TypeError("The objective function must be an intance of `Function`!")

        for constraint in constraints:
            if isinstance(constraint, Relation):
                if not constraint.is_symbolic:
                    raise ValueError("The constraints must be symbolic!")

                self.constraints.append(constraint)
                _variables += constraint.variables
            else:
                raise TypeError("The constraints must be instances of `Relation`!")

        _variables = set(_variables)

        if variables is not None:
            if not all([isinstance(v, Symbol) for v in variables]):
                raise TypeError("All variables must be instances of `Symbol`!")

            if not all([v in _variables for v in variables]):
                raise ValueError("Inconsistent variables provided!")

            self._variables = variables

    @property
    def variables(self) -> Iterable[Symbol]:
        """
        Returns the variables of the problem.
        """
        return self._variables

    def _to_scipy(self, *, maximize: bool = False) -> tuple[ndarray, dict]:
        """
        Returns values for the parameters `A_ub`, `b_ub`, `A_eq`, `b_eq`, `bounds` and `integrality`
        for the `scipy.optimize.linprog` function.
        """
        n_eq = len(list(filter(lambda c: isinstance(c, Equality), self.constraints)))
        n_ieq = len(list(filter(lambda c: isinstance(c, InEquality), self.constraints)))
        n_x = len(self.variables)

        coeffs = self.obj.linear_coefficients(normalize=True)
        c = np.array([coeffs[x_] for x_ in self.variables]).astype(float)
        if maximize:
            c *= -1

        A_ub, b_ub, A_eq, b_eq = 4 * [None]

        x_zero = np.zeros((n_x,))
        SMALLNUM = np.nextafter(0, 1)

        if n_eq > 0:
            A_eq = np.zeros((n_eq, n_x))
            b_eq = np.zeros((n_eq,))
            equalities: Iterable[Equality] = filter(
                lambda c: isinstance(c, Equality), self.constraints
            )
            for i, eq in enumerate(equalities):
                coeffs = eq.linear_coefficients(normalize=True)
                A_eq[i] = [coeffs[x_] for x_ in self.variables]
                b_eq[i] = -eq(x_zero)

        if n_ieq > 0:
            A_ub = np.zeros((n_ieq, n_x))
            b_ub = np.zeros((n_ieq,))
            inequalities: Iterable[InEquality] = filter(
                lambda c: isinstance(c, InEquality), self.constraints
            )
            for i, ieq in enumerate(inequalities):
                coeffs = ieq.linear_coefficients(normalize=True)
                A_ub[i] = [coeffs[x_] for x_ in self.variables]
                b_ub[i] = -ieq(x_zero)

                if ieq.op == Relations.ge:
                    A_ub[i] *= -1
                    b_ub[i] *= -1
                elif ieq.op == Relations.gt:
                    A_ub[i] *= -1
                    b_ub[i] *= -1
                    b_ub[i] -= SMALLNUM
                elif ieq.op == Relations.lt:
                    b_ub[i] -= SMALLNUM

        integrality = self.integrality
        if integrality is None:
            is_int = [v.is_integer for v in self.variables]
            if any(is_int):
                integrality = np.array(is_int, dtype=bool).astype(int)

        kwargs = dict(
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=self.bounds,
            integrality=integrality,
        )

        return c, kwargs

    def solve(
        self, *, maximize: bool = False, method: str = "highs", **kwargs
    ) -> OptimizeResult:
        """
        Solves the linear programming problem using `scipy.optimize.linprog`
        and returns an instance of `scipy.optimize.OptimizeResult`.

        Parameters
        ----------
        maximize : bool, Optional
            If `True`, the problem is a maximization problem.
        method : str, Optional
            The solver to use. The default is "highs". For more options see the
            `method` parameter in `scipy.optimize.linprog`.
        **kwargs
            Additional keyword arguments to pass to `scipy.optimize.linprog`.

        Returns
        -------
        :class:`scipy.optimize.OptimizeResult`
            The result of the optimization.
        """
        c, _kwargs = self._to_scipy(maximize=maximize)
        _kwargs.update(kwargs)
        _kwargs["method"] = method
        res = linprog(c, **_kwargs)
        if res.success:
            res.fun = self.obj(res.x)
        return res
