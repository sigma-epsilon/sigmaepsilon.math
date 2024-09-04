from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify, derive_by_array, symbols, Expr, Symbol
from sympy.core.numbers import One
from typing import Iterable
import sympy as sy


def decode(
    *_,
    expr: Expr | None = None,
    str_expr: str | None = None,
    variables: Iterable | None = None,
    **__,
) -> tuple[Expr, Iterable[Symbol]]:
    """
    Takes an expression as either a string or a `SymPy` expression and returns
    the expression and the variables in the expression.
    """
    if str_expr is not None:
        expr = parse_expr(str_expr, evaluate=False)

    if not variables:
        variables = []
        variables = tuple(expr.free_symbols)
    else:
        if not all([isinstance(v, sy.Expr) for v in variables]):
            variables = list(symbols(variables))

    return expr, variables


def symbolize(*args, **kwargs):
    expr, variables = decode(*args, **kwargs)
    f0 = lambdify([variables], expr, "numpy")
    g = derive_by_array(expr, variables)
    f1 = lambdify([variables], g, "numpy")
    G = derive_by_array(g, variables)
    f2 = lambdify([variables], G, "numpy")
    return {
        "value": f0,
        "gradient": f1,
        "Hessian": f2,
        "d": len(variables),
        "variables": variables,
        "expr": expr,
    }


def substitute(expr, values, variables=None, as_string=False):
    if variables is None:
        variables = tuple(expr.free_symbols)
    if not as_string:
        return expr.subs([(v, val) for v, val in zip(variables, values)])
    else:
        return expr.subs([(str(v), val) for v, val in zip(variables, values)])


def coefficients(
    expr: Expr, variables: Iterable | None = None, normalize: bool = False
) -> dict:
    """
    Returns the coefficients of the expression.

    Parameters
    ----------
    expr : sympy.Expr
        The expression to extract the coefficients from.
    variables : list, optional
        The variables to extract the coefficients from. If not provided, the function
        will attempt to extract the variables from the expression.
    normalize : bool, optional
        If True, the constant term (if present) in the expression will be normalized.
        See the examples for more details.

    Examples
    --------
    The following example illustrated the effect ot the `normalize` parameter.

    >>> g = Function("3*x + 4*y - 2", variables=["x", "y", "z"])
    >>> coefficients(g.expr)
    {x: 3, y: 4, 2: -1}

    >>> coefficients(g.expr, normalize=True)
    {x: 3, y: 4, 1: -2}

    We can see, that the constant term in the expression is handled differently.
    In the first case, the constant term is viewed as the number 2 with coefficient -1,
    while in the second case, the constant term is viewed as the number 1 with coefficient -2.
    """
    if variables is None:
        variables = tuple(expr.free_symbols)

    d = dict({x: 0 for x in variables})
    d.update(expr.as_coefficients_dict())

    if not normalize:
        return d
    else:
        res = dict()
        for key, value in d.items():
            if len(key.free_symbols) == 0:
                res[One()] = value * key
            else:
                res[key] = value
        return res
