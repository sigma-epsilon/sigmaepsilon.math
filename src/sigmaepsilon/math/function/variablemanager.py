from collections import OrderedDict
from typing import Iterable

import sympy as sy
from sympy import lambdify

from .metafunction import substitute


class VariableManager(object):
    """
    A class to manage variables.
    """

    def __init__(
        self, variables: Iterable | None = None, vmap: dict | None = None, **kwargs
    ):
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

    def substitute(
        self, vmap: dict | None = None, inverse: bool = False, inplace: bool = True
    ):
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

    def lambdify(self, variables: Iterable | None = None):
        assert variables is not None
        for v, expr in self.vmap.items():
            self.vmap[v] = lambdify([variables], expr, "numpy")

    def __call__(self, v):
        return self.vmap[v]

    def target(self) -> list:
        return list(self.vmap.keys())

    def source(self) -> list:
        s = set()
        for expr in self.vmap.values():
            s.update(expr.free_symbols)
        return list(s)

    def add_variables(self, variables: Iterable, overwrite:bool=True) -> None:
        if overwrite:
            self.vmap.update({v: v for v in variables})
        else:
            for v in variables:
                if v not in self.vmap:
                    self.vmap[v] = v
