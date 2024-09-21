from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify, derive_by_array, symbols, Expr, Symbol
from sympy.core.numbers import One
from typing import Iterable
import sympy as sy


def generate_symbols(
    template: str, indices: Iterable[int], **assumptions
) -> list[Symbol]:
    """
    Generates a list of symbols.
    """
    result = list(
        sy.symbols(" ".join([template.format(i) for i in indices]), **assumptions)
    )
    return result


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
        if not all([isinstance(v, (sy.Expr, sy.Symbol)) for v in variables]):
            variables = list(symbols(variables))

        expr = substitute(expr, variables, variables, as_string=True)

    return expr, variables


def symbolize(*args, simplify: bool = True, **kwargs) -> dict:
    expr, variables = decode(*args, **kwargs)
    if simplify:
        expr = expr.simplify()
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


def substitute(
    expr: Expr,
    values: Iterable,
    variables: Iterable[str | Symbol] | None = None,
    as_string: bool = False,
) -> Expr:
    """
    Substitutes the variables in the expression with the corresponding values.

    Parameters
    ----------
    expr : sympy.Expr
        The expression to substitute the values in.
    values : list
        The values to substitute the variables with.
    variables : list, optional
        The variables to substitute. If not provided, the function will attempt to
        extract the variables from the expression.
    as_string : bool, optional
        If True, the variables will be substituted as strings. Default is `False`.
        This might be important if let say you want to instantiate a Function with
        the string 'x + y' and you want the expression to have variables that you
        prepared with `sympy.symbols`. In that case, two SymPy variables with the same
        name will be treated as different variables. Substituting the variables as strings
        will help you avoid this issue.
    """
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

    >>> from sigmaepsilon.math.function import Function
    >>> from sigmaepsilon.math.function.symutils import coefficients
    >>> g = Function("3*x + 4*y - 2", variables=["x", "y", "z"])
    >>> coefficients(g.expr)  # doctest: +SKIP
    {x: 3, y: 4, 2: -1}

    >>> coefficients(g.expr, normalize=True)  # doctest: +SKIP
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
