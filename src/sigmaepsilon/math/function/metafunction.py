from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify, derive_by_array, symbols, Expr
from sympy.core.numbers import One
from collections import OrderedDict
from typing import Iterable
import sympy as sy

from sigmaepsilon.core.abstract import ABCMeta_Weak


class ABCMeta_MetaFunction(ABCMeta_Weak):
    """
    Metaclass for defining ABCs for algebraic structures.
    """

    def __new__(metaclass, name, bases, namespace, /, **kwargs):
        cls = super().__new__(metaclass, name, bases, namespace, **kwargs)
        if "value" in namespace:
            cls.f = namespace["value"]

        if "gradient" in namespace:
            cls.g = namespace["gradient"]

        if "Hessian" in namespace:
            cls.G = namespace["Hessian"]

        return cls


class MetaFunction(metaclass=ABCMeta_MetaFunction):
    __slots__ = ("f0", "f1", "f2", "dimension", "domain", "expr", "variables", "vmap")

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def f(self, *args, **kwargs):
        """
        Returns the function value.

        For this operation the object must have an implementation of

        value(self, *args, **kwargs):
            <...>
            return <...>
        """
        return self.f0(*args, **kwargs)

    def g(self, *args, **kwargs):
        """
        Returns the gradient vector if available.

        For this operation the object must have an implementation of

        gradient(self, *args, **kwargs):
            <...>
            return <...>
        """
        return self.f1(*args, **kwargs)

    def G(self, *args, **kwargs):
        """
        Returns the Hessian matrix if available.

        For this operation the object must have an implementation of

        Hessian(self,*args,**kwargs):
            <...>
            return <...>
        """
        return self.f2(*args, **kwargs)

    @classmethod
    def _str_to_func(cls, str_expr: str, *args, **kwargs):
        return symbolize(*args, str_expr=str_expr, **kwargs)

    @classmethod
    def _sympy_to_func(cls, expr: Expr, *args, **kwargs):
        return symbolize(*args, expr=expr, **kwargs)


def decode(*_, expr=None, str_expr: str = None, variables: Iterable = None, **__):
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


def coefficients(expr=None, variables=None, normalize=False):
    if variables is None:
        variables = tuple(expr.free_symbols)
    d = OrderedDict({x: 0 for x in variables})
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
