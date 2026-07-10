"""Metaclass and base class shared by symbolic and numerical functions."""

from sympy import Expr

from sigmaepsilon.core.abstract import ABCMeta_Weak
from .symutils import symbolize


class ABCMeta_MetaFunction(ABCMeta_Weak):
    """Metaclass for defining ABCs for algebraic structures."""

    def __new__(metaclass, name, bases, namespace, /, **kwargs):
        """Create a new class, promoting `value`/`gradient`/`Hessian` to `f`/`g`/`G`."""
        cls = super().__new__(metaclass, name, bases, namespace, **kwargs)
        if "value" in namespace:
            cls.f = namespace["value"]

        if "gradient" in namespace:
            cls.g = namespace["gradient"]

        if "Hessian" in namespace:
            cls.G = namespace["Hessian"]

        return cls


class MetaFunction(metaclass=ABCMeta_MetaFunction):
    """Base class providing the `f`/`g`/`G` interface for functions.

    Subclasses that define `value`, `gradient` and/or `Hessian` methods
    automatically get those promoted to `f`, `g` and `G` respectively, via
    :class:`ABCMeta_MetaFunction`.
    """

    __slots__ = ("f0", "f1", "f2", "dimension", "domain", "expr", "variables", "vmap")

    def __call__(self, *args, **kwargs):
        """Return the function value, equivalent to calling :meth:`f`."""
        return self.f(*args, **kwargs)

    def f(self, *args, **kwargs):
        """Return the function value.

        For this operation the object must have an implementation of

        value(self, *args, **kwargs):
            <...>
            return <...>
        """
        return self.f0(*args, **kwargs)

    def g(self, *args, **kwargs):
        """Return the gradient vector if available.

        For this operation the object must have an implementation of

        gradient(self, *args, **kwargs):
            <...>
            return <...>
        """
        return self.f1(*args, **kwargs)

    def G(self, *args, **kwargs):
        """Return the Hessian matrix if available.

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
