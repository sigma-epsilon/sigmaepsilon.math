from enum import Enum
import operator as op
from typing import TypeVar, Callable

from sigmaepsilon.core.kwargtools import getasany

from .function import Function


__all__ = ["Equality", "InEquality", "Relation"]


class Relations(Enum):
    eq = "="
    gt = ">"
    ge = ">="
    lt = "<"
    le = "<="

    def to_op(self):
        return _rel_to_op[self]


_rel_to_op = {
    Relations.eq: op.eq,
    Relations.gt: op.gt,
    Relations.ge: op.ge,
    Relations.lt: op.lt,
    Relations.le: op.le,
}

RelationType = TypeVar("RelationType", str, Relations, Callable)


class Relation(Function):
    """
    Base class for relations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = None
        self.opfunc = None
        op = getasany(["op", "operator"], None, **kwargs)
        if op:
            if isinstance(op, str):
                self.op = Relations(op)
            elif isinstance(op, Relations):
                self.op = op
            elif isinstance(op, Callable):
                self.opfunc = op
                self.op = None
        else:
            self.op = Relations.eq
        if op and isinstance(self.op, Relations):
            self.opfunc = self.op.to_op()
        self.slack = 0

    @property
    def operator(self) -> Callable:
        """
        Returns the associated operator.
        """
        return self.op

    def to_eq(self):
        raise NotImplementedError

    def relate(self, *args, **kwargs):
        """
        Relates an input and returns True if it is feasible.
        """
        return self.opfunc(self.f0(*args, **kwargs), 0)


class Equality(Relation):
    """
    Class for equality constraints.

    Example
    -------
    >>> import sympy as sy
    >>> variables = ['x1', 'x2', 'x3', 'x4']
    >>> x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
    >>> eq1 = Equality(x1 + 2*x3 + x4 - 4, variables=syms)
    >>> eq2 = Equality(x2 + x3 - x4 - 2, variables=syms)
    """

    def __init__(self, *args, **kwargs):
        kwargs["op"] = Relations.eq
        super().__init__(*args, **kwargs)

    def to_eq(self):
        return self


class InEquality(Relation):
    """
    Class for inequality constraints.

    Examples
    --------
    >>> gt = InEquality('x + y', op='>')
    >>> gt([0.0, 0.0])
    0.0

    >>> ge = InEquality('x + y', op='>=')
    >>> ge([0.0, 0.0])
    0.0

    >>> le = InEquality('x + y', op=lambda x, y: x <= y)
    >>> le([0.0, 0.0])
    0.0

    >>> lt = InEquality('x + y', op=lambda x, y: x < y)
    >>> lt([0.0, 0.0])
    0.0
    """

    def __init__(self, *args, **kwargs):
        op = getasany(["op", "operator"], None, **kwargs)
        if not op:
            raise ValueError("An operator must be defined.")
        super().__init__(*args, **kwargs)

    def to_eq(self):
        raise NotImplementedError
