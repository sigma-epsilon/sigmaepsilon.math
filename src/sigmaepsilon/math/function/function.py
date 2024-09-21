from typing import TypeVar, Callable, Iterable
from collections import OrderedDict
import warnings

from sympy import Expr, degree, latex
from sympy.core.numbers import One
import numpy as np

from sigmaepsilon.core.kwargtools import getasany

from .metafunction import MetaFunction
from .symutils import substitute


__all__ = ["Function", "FunctionLike"]


FunctionLike = TypeVar("FunctionLike", str, Callable, Expr)


class Function(MetaFunction):
    """
    Base class for all kinds of functions. It can be used to represent
    symbolic and numerical functions. The class is designed to be as
    flexible as possible, so it can be used in a wide range of applications.

    As you can see from the examples, the class can be used in a variety of ways.
    You can initialize a function object with a string expression, a `SymPy` expression,
    or with numerical functions. If you initialize the object with a string or a `SymPy`
    expression, first and second derivatives can be calculated automatically.

    Parameters
    ----------
    f0: FunctionLike or None, Optional
        Positional parameter. A callable object that returns function evaluations.
    f1: Callable or None, Optional
        Positional parameter.
        A callable object that returns evaluations of the gradient of the function.
        If the first positional argument is a string or a `SymPy` expression, this
        parameter can be ignored and derivative functions are calculated automatically.
    f2: Callable or None, Optional
        Positional parameter.
        A callable object that returns evaluations of the Hessian of the function.
        If the first positional argument is a string or a `SymPy` expression, this
        parameter can be ignored and derivative functions are calculated automatically.
    variables: List, Optional
        Symbolic variables. Only required if the function is defined by
        a string or `SymPy` expression.
    value: Callable, Optional
        Keyword only parameter. Same as `f0`.
    gradient: Callable, Optional
        Keyword only parameter.Same as `f1`.
    Hessian: Callable, Optional
        Keyword only parameter.Same as `f2`.
    dimension or dim or d : int, Optional
        Keyword only parameter. The number of dimensions of the domain of the function. This property
        is only required if the function is defined by numerical functions.

    See Also
    --------
    :class:`~sigmaepsilon.math.function.relation.Relation`
    :class:`~sigmaepsilon.math.function.relation.Equality`
    :class:`~sigmaepsilon.math.function.relation.InEquality`

    Examples
    --------
    >>> from sigmaepsilon.math.function import Function
    >>> import sympy as sy
    >>> import numpy as np

    Define a symbolic function with positive variables. Note here that if it was not relevant
    from the aspect of the application to indicate that the variables are positive, it
    wouldn't be necessary to explicity specify them using the parameter `variables`.

    >>> variables = ['x1', 'x2', 'x3', 'x4']
    >>> x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
    >>> f = Function(3*x1 + 9*x3 + x2 + x4, variables=syms)
    >>> f([0, 6, 0, 4])
    10

    An equivalent definition can be given without using SymPy explicitly:

    >>> f = Function("3*x1 + 9*x3 + x2 + x4", variables=['x1', 'x2', 'x3', 'x4'])
    >>> f([0, 6, 0, 4])
    10

    In both cases, providing the argument 'variables' is optional, but it determines the
    order of the arguments. If the function is defined without the variables being provided,
    it is derived from the input, but the order of the arguments may differ from what you would
    expect. Also defining the variables with SymPy might be important if you want to indicate
    something about their nature, eg. that they are positive, as in the example above. For instance
    if you execute the following code, the result my change from execution to execution.

    >>> f = Function("3*x1 + 9*x3 + x2 + x4")
    >>> f.variables  # doctest: +SKIP
    (x3, x1, x4, x2)

    Define a numerical function. In this case the dimension of the input must be specified
    explicitly.

    >>> def f0(x, y): return x**2 + y
    >>> def f1(x, y): return np.array([2*x, 1])
    >>> def f2(x, y): return np.array([[0, 0], [0, 0]])
    >>> f = Function(f0, f1, f2, d=2)
    >>> f.is_linear
    False

    To call the function, call it like you would call the function `f0`:

    >>> f(1, 1)
    2
    >>> f.g(1, 1)
    array([2, 1])

    You can mix different kinds of input signatures:

    >>> def f0(x): return x[0]**2 + x[1]
    >>> def f1(x, y): return np.array([2*x, 1])
    >>> def f2(x, y): return np.array([[0, 0], [0, 0]])
    >>> f = Function(f0, f1, f2, d=2)

    The point is that you always call the resulting `Function` object
    according to your definition. Now your `f0` expects an iterable,
    therefore you can call it like

    >>> f([1, 1])
    2

    but the gradient function expects the same values as two scalars, so
    you call it like this

    >>> f.g(1, 1)
    array([2, 1])

    Explicity defining the variables for a symbolic function is important
    if not all variables appear in the string expression you feed the object with:

    >>> g = Function('3*x + 4*y - 2', variables=['x', 'y', 'z'])
    >>> g.is_linear
    True

    If you do not specify 'z' as a variable here, the resulting object expects
    two values

    >>> h = Function('3*x + 4*y - 2')
    >>> h([1, 1])
    5

    The variables can be `SymPy` variables as well:

    >>> m = Function('3*x + 4*y - 2', variables=sy.symbols('x y z'))
    >>> m.is_linear
    True
    >>> m([1, 2, -30])
    9

    """

    __slots__ = ("f0", "f1", "f2", "dimension", "domain", "vmap", "from_str")

    # NOTE domain is missing from the possible parameters
    # NOTE investigate if dimensions could be derived

    def __init__(
        self,
        f0: FunctionLike | None = None,
        f1: Callable | None = None,
        f2: Callable | None = None,
        *args,
        variables: Iterable | None = None,
        **kwargs
    ):
        super().__init__()
        self.update(f0, f1, f2, *args, variables=variables, **kwargs)

    def update(
        self,
        f0: FunctionLike | None = None,
        f1: Callable | None = None,
        f2: Callable | None = None,
        *_,
        variables: Iterable | None = None,
        **kwargs
    ):
        self.from_str = None

        if f0 is not None:
            if isinstance(f0, str):
                kwargs.update(self._str_to_func(f0, variables=variables, **kwargs))
                self.from_str = True
            elif isinstance(f0, Expr):
                kwargs.update(self._sympy_to_func(f0, variables=variables, **kwargs))

        self.expr = kwargs.get("expr", None)
        self.variables = kwargs.get("variables", variables)
        self.f0 = kwargs.get("value", f0)
        self.f1 = kwargs.get("gradient", f1)
        self.f2 = kwargs.get("Hessian", f2)
        self.dimension = getasany(["d", "dimension", "dim"], None, **kwargs)
        self.domain = kwargs.get("domain", None)
        self.vmap = kwargs.get("vmap", None)

    @property
    def symbolic(self) -> bool:
        """
        Returns `True` if the function is a fit subject of symbolic manipulation.
        This is probably only true if the object was created from a string or
        `sympy` expression.
        """
        warnings.warn(
            "The property `symbolic` is deprecated and will be removed in a future version. "
            "Use `is_symbolic` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.is_symbolic

    @property
    def linear(self) -> bool:
        """
        Returns True if the function is at most linear in all of its variables.
        """
        warnings.warn(
            "The property `linear` is deprecated and will be removed in a future version. "
            "Use `is_linear` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.is_linear

    @property
    def is_linear(self) -> bool:
        """
        Returns `True` if the function is at most linear in all of its variables.
        """
        if self.is_symbolic:
            return all(
                np.array([degree(self.expr, v) for v in self.variables], dtype=int) <= 1
            )
        else:
            return self.f2 is None

    @property
    def is_symbolic(self) -> bool:
        """
        Returns `True` if the function is a fit subject of symbolic manipulation.
        This is probably only true if the object was created from a string or
        `SymPy` expression.
        """
        return self.expr is not None

    def simplify(self) -> None:
        """
        Simplifies the symbolic expression of the instance.
        """
        if self.is_symbolic:
            self.expr = self.expr.simplify()
        else:
            raise TypeError("This is exclusive to symbolic functions.")

    def linear_coefficients(self, normalize: bool = False) -> dict | None:
        """
        Returns the linear coeffiecients, if the function is symbolic and linear.

        Parameters
        ----------
        normalize : bool, Optional
            If True, the coefficients are normalized. Default is False.

        Examples
        --------
        >>> from sigmaepsilon.math.function import Function
        >>> from sigmaepsilon.math.approx.lagrange import gen_Lagrange_1d
        >>> f = gen_Lagrange_1d(N=2)
        >>> f1 = Function(f[1][0], f[1][1], f[1][2])
        >>> linear_coefficients = f1.linear_coefficients()

        """
        d = self.coefficients(normalize)
        if d:
            return {
                key: value for key, value in d.items() if len(key.free_symbols) <= 1
            }
        return None

    def coefficients(self, normalize: bool = False) -> dict | None:
        """
        Returns the coefficients if the function is symbolic.

        Parameters
        ----------
        normalize : bool, Optional
            If True, the coefficients are normalized. Default is False.

        Examples
        --------
        >>> from sigmaepsilon.math.function import Function
        >>> from sigmaepsilon.math.approx.lagrange import gen_Lagrange_1d
        >>> f = gen_Lagrange_1d(N=2)
        >>> f1 = Function(f[1][0], f[1][1], f[1][2])
        >>> coefficients = f1.coefficients()

        """
        if not self.is_symbolic:
            raise TypeError("This is exclusive to symbolic functions.")

        try:
            d = OrderedDict({x: 0 for x in self.variables})
            expr = self.expr.simplify()
            d.update(expr.as_coefficients_dict())
            if not normalize:
                return d
            else:
                res = OrderedDict()
                for key, value in d.items():
                    if len(key.free_symbols) == 0:
                        res[One()] = value * key
                    else:
                        res[key] = value
                return res
        except Exception:  # pragma: no cover
            return None

    def to_latex(self) -> str:
        """
        Returns the LaTeX code of the symbolic expression of the instance.
        Only for symbolic functions.

        Examples
        --------
        >>> from sigmaepsilon.math.function import Function
        >>> from sigmaepsilon.math.approx.lagrange import gen_Lagrange_1d
        >>> f = gen_Lagrange_1d(N=2)
        >>> f1 = Function(f[1][0], f[1][1], f[1][2])
        >>> latex_string = f1.to_latex()

        """
        if self.is_symbolic:
            return latex(self.expr)
        else:
            raise TypeError("This is exclusive to symbolic functions.")

    def subs(
        self, values: Iterable, variables: Iterable | None = None, inplace: bool = False
    ) -> "Function":
        """
        Substitites values for variables. Only for symbolic functions.

        Examples
        --------
        >>> from sigmaepsilon.math.function import Function
        >>> g = Function("3*x + 4*y - 2", variables=["x", "y", "z"])
        >>> g = g.subs([0, 0, 0], ["x", "y", "z"], inplace=True)

        """
        if not self.is_symbolic:
            raise TypeError("This is exclusive to symbolic functions.")

        expr = substitute(self.expr, values, variables, as_string=self.from_str)
        kwargs = self._sympy_to_func(expr=expr, variables=variables)

        if not inplace:
            return Function(None, None, None, **kwargs)
        else:
            self.update(None, None, None, **kwargs)
            return self
