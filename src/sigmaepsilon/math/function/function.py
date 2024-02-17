from typing import TypeVar, Callable, Iterable, Optional, Union
from collections import OrderedDict

import sympy as sy
from sympy import Expr, degree, latex, lambdify
from sympy.core.numbers import One
import numpy as np

from sigmaepsilon.core.kwargtools import getasany

from .metafunction import MetaFunction, substitute


__all__ = ["Function", "VariableManager", "FuncionLike"]


FuncionLike = TypeVar("FuncionLike", str, Callable, Expr)


class Function(MetaFunction):
    """
    Base class for all kinds of functions.

    Parameters
    ----------
    f0: Callable
        A callable object that returns function evaluations.
    f1: Callable
        A callable object that returns evaluations of the
        gradient of the function.
    f2: Callable
        A callable object that returns evaluations of the
        Hessian of the function.
    variables: List, Optional
        Symbolic variables. Only required if the function is defined by
        a string or `SymPy` expression.
    value: Callable, Optional
        Same as `f0`.
    gradient: Callable, Optional
        Same as `f1`.
    Hessian: Callable, Optional
        Same as `f2`.
    dimension or dim or d : int, Optional
        The number of dimensions of the domain of the function. Required only when
        going full blind, in most of the cases it can be inferred from other properties.

    Examples
    --------
    >>> from sigmaepsilon.math.function import Function
    >>> import sympy as sy

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
    expect.

    >>> f = Function("3*x1 + 9*x3 + x2 + x4")
    >>> f.variables
    (x3, x1, x4, x2)

    Define a numerical function. In this case the dimension of the input must be specified
    explicitly.

    >>> def f0(x, y): return x**2 + y
    >>> def f1(x, y): return np.array([2*x, 1])
    >>> def f2(x, y): return np.array([[0, 0], [0, 0]])
    >>> f = Function(f0, f1, f2, d=2)
    >>> f.linear
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
    >>> g.linear
    True

    If you do not specify 'z' as a variable here, the resulting object expects
    two values

    >>> h = Function('3*x + 4*y - 2')
    >>> h([1, 1])
    5

    The variables can be `SymPy` variables as well:

    >>> m = Function('3*x + 4*y - 2', variables=sy.symbols('x y z'))
    >>> m.linear
    True
    >>> m([1, 2, -30])
    9
    """

    # FIXME domain is missing from the possible parameters
    # NOTE investigate if dimensions should be derived

    def __init__(
        self,
        f0: FuncionLike = None,
        f1: Callable = None,
        f2: Callable = None,
        *args,
        variables: Optional[Union[Iterable, None]] = None,
        **kwargs
    ):
        super().__init__()
        self.update(f0, f1, f2, *args, variables=variables, **kwargs)

    def update(
        self,
        f0: FuncionLike = None,
        f1: Callable = None,
        f2: Callable = None,
        *_,
        variables: Iterable = None,
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
        Returns True if the function is a fit subject of symbolic manipulation.
        This is probably only true if the object was created from a string or
        `sympy` expression.
        """
        return self.expr is not None

    @property
    def linear(self) -> bool:
        """
        Returns True if the function is at most linear in all of its variables.
        """
        if self.symbolic:
            return all(
                np.array([degree(self.expr, v) for v in self.variables], dtype=int) <= 1
            )
        else:
            return self.f2 is None

    def linear_coefficients(
        self, normalize: Optional[bool] = False
    ) -> Union[Iterable, None]:
        """
        Returns the linear coeffiecients, if the function is symbolic.
        """
        d = self.coefficients(normalize)
        if d:
            return {
                key: value for key, value in d.items() if len(key.free_symbols) <= 1
            }
        return None

    def coefficients(self, normalize: bool = False):
        """
        Returns the coefficients if the function is symbolic.
        """
        try:
            d = OrderedDict({x: 0 for x in self.variables})
            d.update(self.expr.as_coefficients_dict())
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
        except Exception:
            return None

    def to_latex(self) -> str:
        """
        Returns the LaTeX code of the symbolic expression of the object.
        Only for simbolic functions.
        """
        if self.symbolic:
            return latex(self.expr)
        else:
            raise TypeError("This is exclusive to symbolic functions.")

    def subs(self, values, variables=None, inplace=False) -> "Function":
        """
        Substitites values for variables.
        """
        if not self.symbolic:
            raise TypeError("This is exclusive to symbolic functions.")

        expr = substitute(self.expr, values, variables, as_string=self.from_str)
        kwargs = self._sympy_to_func(expr=expr, variables=variables)

        if not inplace:
            return Function(None, None, None, **kwargs)
        else:
            self.update(None, None, None, **kwargs)
            return self


class VariableManager(object):
    def __init__(self, variables=None, vmap=None, **kwargs):
        try:
            variables = list(sy.symbols(variables, **kwargs))
        except Exception:
            variables = variables
        try:
            self.vmap = (
                vmap if vmap is not None else OrderedDict({v: v for v in variables})
            )
        except Exception:
            self.vmap = OrderedDict()
        self.variables = variables  # this may be unnecessary

    def substitute(self, vmap: dict = None, inverse=False, inplace=True):
        if not inverse:
            sval = list(vmap.values())
            svar = list(vmap.keys())
        else:
            sval = list(vmap.keys())
            svar = list(vmap.values())
        if inplace:
            for v, expr in self.vmap.items():
                self.vmap[v] = substitute(expr, sval, svar)
            return self
        else:
            vmap = OrderedDict()
            for v, expr in self.vmap.items():
                vmap[v] = substitute(expr, sval, svar)
            return vmap

    def lambdify(self, variables=None):
        assert variables is not None
        for v, expr in self.vmap.items():
            self.vmap[v] = lambdify([variables], expr, "numpy")

    def __call__(self, v):
        return self.vmap[v]

    def target(self):
        return list(self.vmap.keys())

    def source(self):
        s = set()
        for expr in self.vmap.values():
            s.update(expr.free_symbols)
        return list(s)

    def add_variables(self, variables, overwrite=True):
        if overwrite:
            self.vmap.update({v: v for v in variables})
        else:
            for v in variables:
                if v not in self.vmap:
                    self.vmap[v] = v
