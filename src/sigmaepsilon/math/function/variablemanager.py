from typing import Iterable
import contextlib

import sympy as sy
from sympy import lambdify, Symbol

from .symutils import substitute


class VariableManager:
    """
    A class to manage symbolic variables. It is heavily used in optimization
    problems to keep track of the mapping of variables. For instance, during the
    solution of a linear programming problem, the problem is transformed from a
    general form to a standard form. This transformation involves the introduction
    of new variables (eg. slack variables), and later all the variables of the system
    are generalized to a standard form. The `VariableManager` class helps to map the
    variables of the original problem to the variables of the transformed problem, since
    after solving the problem in standard form, we wish to retain the values of the original
    variables.

    Parameters
    ----------
    variables : Iterable[str | sympy.Symbol], optional
        A list of symbolic variables or strings. Default is `None`.
    vmap : dict, optional
        A dictionary containing the mapping of the variables. Default is `None`.
    **kwargs
        Additional keyword arguments to pass to `sympy.symbols`, if the `variables` argument
        is a list of strings.
    """

    def __init__(
        self,
        variables: Iterable[str | Symbol] | None = None,
        vmap: dict | None = None,
        **kwargs
    ):
        self._vmap = dict()

        with contextlib.suppress(Exception):
            variables = list(sy.symbols(variables, **kwargs))

        with contextlib.suppress(Exception):
            self._vmap = vmap if vmap is not None else dict({v: v for v in variables})

        self.variables = variables  # this may be unnecessary

    @property
    def vmap(self) -> dict:
        """
        Returns the variable map.
        """
        return self._vmap

    def substitute(
        self, vmap: dict, inverse: bool = False, inplace: bool = True
    ) -> dict:
        """
        Substitute the variables in the variable manager with the corresponding expressions.

        Parameters
        ----------
        vmap : dict
            A dictionary containing the variables to subtitute and their the values
            to substitute them with. The values can be either variables or other expressions.
        inverse : bool, optional
            If True, the substitution is done in the opposite direction, not how it would be
            indicated by the argument `vmap`. Default is `False`.
        inplace : bool, optional
            If True, the substitution is done in place. Default is `True`.
        """
        if not inverse:
            sval = list(vmap.values())
            svar = list(vmap.keys())
        else:
            sval = list(vmap.keys())
            svar = list(vmap.values())

        if inplace:
            for v, expr in self._vmap.items():
                self._vmap[v] = substitute(expr, sval, svar)
            return self._vmap
        else:
            vmap = dict()
            for v, expr in self._vmap.items():
                vmap[v] = substitute(expr, sval, svar)
            return vmap

    def lambdify(self, variables: Iterable[str | Symbol]) -> None:
        for v, expr in self._vmap.items():
            self._vmap[v] = lambdify([variables], expr, "numpy")

    def __call__(self, v):
        return self._vmap[v]

    def source(self) -> list:
        """
        Return the source variables of the variable manager.
        """
        return list(self._vmap.keys())

    def target(self) -> list:
        """
        Returns the target variables of the variable manager.
        """
        s = set()
        for expr in self._vmap.values():
            s.update(expr.free_symbols)
        return list(s)

    def add_variables(
        self, variables: Iterable[str | Symbol], overwrite: bool = True
    ) -> None:
        """
        Adds new variables to the variable manager.
        """
        if overwrite:
            self._vmap.update({v: v for v in variables})
        else:
            for v in variables:
                if v not in self._vmap:
                    self._vmap[v] = v
