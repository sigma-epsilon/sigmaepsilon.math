from typing import Iterable, Callable

import sympy as sy
from sympy import latex
import numpy as np

from sigmaepsilon.deepdict import DeepDict


__all__ = ["gen_Lagrange_1d", "approx_Lagrange_1d"]


def _var_tmpl(i: int):
    return r"\Delta_{}".format(i)


def _var_str(inds, i: int):
    return _var_tmpl(inds[i])


def _diff(xvar, fnc: sy.Expr):
    return fnc.diff(xvar).expand().simplify().factor().simplify()


def gen_Lagrange_1d(
    *_,
    x: Iterable | None = None,
    i: Iterable[int] = None,
    xsym: str | None = None,
    fsym: str | None = None,
    sym: bool = False,
    N: int | None = None,
    lambdify: bool = False,
    out: dict | None = None
) -> dict:
    """
    Generates Lagrange polynomials and their derivatives up to 3rd, for approximation
    in 1d space, based on N input pairs of position and value. Geometrical parameters
    can be numeric or symbolic.

    Parameters
    ----------
    x: Iterable, Optional
        The locations of the data points. If not specified and `sym=False`, a range
        of [-1, 1] is assumed and the locations are generated as `np.linspace(-1, 1, N)`,
        where N is the number of data points. If `sym=True`, the calculation is entirely
        symbolic. Default is None.
    i: Iterable[int]
        If not specified, indices are assumed as [1, ..., N], but this is only relevant
        for symbolic representation, not the calculation itself, which only cares about
        the number of data points, regardless of their actual indices.
    xsym: str, Optional
        Symbol of the variable in the symbolic representation of the generated functions.
        Default is :math:`x`.
    fsym: str, Optional
        Symbol of the function in the symbolic representation of the generated functions.
        Default is 'f'.
    sym: bool, Optional.
        If True, locations of the data points are left in a symbolic state. This requires
        the inversion of a symbolic matrix, which has some reasonable limitations.
        Default is False.
    N: int, Optional
        If neither 'x' nor 'i' is specified, this controls the number of functions to
        generate. Default is None.
    lambdify: bool, Optional
        If True, the functions are turned into `NumPy` functions via `sympy.lambdify`
        and stored in the output for each index with keyword 'fnc'. Default is False.
    out: dict, Optional
        A dictionary to store the values in. Default is None.

    Returns
    -------
    dict
        A dictionary containing the generated functions for the reuested nodes.
        The keys of the dictionary are the indices of the points, the values are
        dictionaries with the following keys and values:

            'symbol' : the `SymPy` symbol of the function

            0 : the function

            1 : the first derivative as a `SymPy` expression

            2 : the second derivative as a `SymPy` expression

            3 : the third derivative as a `SymPy` expression

    Examples
    --------
    >>> from sigmaepsilon.math.approx import gen_Lagrange_1d

    To generate approximation functions for a 2-noded line:

    >>> functions = gen_Lagrange_1d(x=[-1, 1])

    or equivalently

    >>> functions = gen_Lagrange_1d(N=2)

    To generate the same functions in symbolic form:

    >>> functions = gen_Lagrange_1d(i=[1, 2], sym=True)

    Notes
    -----
    Inversion of a heavily symbolic matrix may take quite some time, and is not suggested
    for N > 3. Fixing the locations as constant real numbers symplifies the process and
    makes the solution much faster.
    """
    xsym = xsym if xsym is not None else r"x"
    fsym = fsym if fsym is not None else r"\phi"
    module_data = DeepDict() if not isinstance(out, dict) else out

    xvar = sy.symbols(xsym)
    if not isinstance(N, int):
        assert (x is not None) or (i is not None), "'N', 'x' or 'i' must be provided!"
        N = len(x) if x is not None else len(i)
    inds = list(range(1, N + 1)) if i is None else i

    coeffs = sy.symbols(", ".join(["c_{}".format(i + 1) for i in range(N)]))
    variables = sy.symbols(", ".join([_var_str(inds, i) for i in range(N)]))
    if x is None:
        if not sym:
            x = np.linspace(-1, 1, N)
        else:
            symbols = [xsym + "_{}".format(i + 1) for i in range(N)]
            x = sy.symbols(", ".join(symbols))
    poly = sum([c * xvar**i for i, c in enumerate(coeffs)])

    evals = [poly.subs({xsym: x[i]}) for i in range(N)]
    A = sy.zeros(N, N)
    for i in range(N):
        A[i, :] = sy.Matrix([evals[i].coeff(c) for c in coeffs]).T
    coeffs_new = A.inv() * sy.Matrix(variables)
    subs = {coeffs[i]: coeffs_new[i] for i in range(N)}
    approx = poly.subs(subs).simplify().expand()

    shp = [approx.coeff(v).factor().simplify() for v in variables]

    dshp1 = [_diff(xvar, fnc) for fnc in shp]
    dshp2 = [_diff(xvar, fnc) for fnc in dshp1]
    dshp3 = [_diff(xvar, fnc) for fnc in dshp2]

    for i, ind in enumerate(inds):
        fnc_str = latex(sy.symbols(fsym + "_{}".format(ind)))
        module_data[ind]["symbol"] = fnc_str
        module_data[ind][0] = shp[i]
        module_data[ind][1] = dshp1[i]
        module_data[ind][2] = dshp2[i]
        module_data[ind][3] = dshp3[i]

    if lambdify:
        assert not sym, "If 'lambdify' is True, 'sym' must be False"
        for ind in inds:
            for j in range(4):
                fnc = module_data[ind][j]
                module_data[ind]["fnc"][j] = sy.lambdify(xvar, fnc, "numpy")

    if isinstance(module_data, DeepDict):
        module_data.lock()

    return module_data


def approx_Lagrange_1d(
    points: Iterable, values: Iterable, lambdify: bool = False
) -> Callable:
    """
    Returns a callable that maps from 'source' to 'target' in 1d.

    Parameters
    ----------
    points: Iterable[float]
        The locations of the data points.
    values: Iterable[float]
        The values at the data points.
    lambdify: bool, Optional
        If `True`, the returned function is turned into a `NumPy` function via
        `sympy.lambdify`, otherwise it is left as a `SymPy` expression. Default is `False`.

    Returns
    -------
    Callable
        A vectorized function if 'lambdify' is True, or a SymPy expression otherwise.

    Example
    -------
    >>> from sigmaepsilon.math.approx import approx_Lagrange_1d
    >>> approx = approx_Lagrange_1d([-1, 1], [0, 10], lambdify=True)
    >>> approx(-1), approx(1), approx(0)
    (0, 10, 5)

    A symbolic example:

    >>> import sympy as sy
    >>> L = sy.symbols('L', real=True, positive=True)
    >>> fnc = approx_Lagrange_1d([-1, 1], [0, L])
    >>> str(fnc)
    'L*(x/2 + 1/2)'

    To get the Jacobian of the transformation [0, L] -> [-1, 1]:

    >>> L = sy.symbols('L', real=True, positive=True)
    >>> fnc = approx_Lagrange_1d([0, L], [-1, 1])
    >>> dfnc = fnc.diff('x')
    >>> str(dfnc)
    '2/L'

    Or the other way:

    >>> L = sy.symbols('L', real=True, positive=True)
    >>> fnc = approx_Lagrange_1d([-1, 1], [0, L])
    >>> dfnc = fnc.diff('x')
    >>> str(dfnc)
    'L/2'

    """
    xsym = "x"
    assert len(points) == len(values), "'source' and 'target' must have the same length"
    indices = list(range(len(points)))
    basis = gen_Lagrange_1d(x=points, i=indices, xsym=xsym, lambdify=False)
    shp0 = [basis[i, 0] for i in indices]
    fnc0 = sum([a * f for a, f in zip(values, shp0)])
    if lambdify:
        return sy.lambdify(xsym, fnc0, "numpy")
    else:
        return fnc0
