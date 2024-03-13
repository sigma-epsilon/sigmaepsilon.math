from collections import OrderedDict

import sympy as sy
from sympy import lambdify

from .metafunction import substitute


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