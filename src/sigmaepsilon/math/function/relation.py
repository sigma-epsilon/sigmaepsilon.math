from enum import Enum
import operator as op
from typing import Callable

from sigmaepsilon.core.kwargtools import getasany

from .function import Function
from .utils import parse_expression, has_operator, valid_operators


__all__ = ["Equality", "InEquality", "Relation"]


class Relations(Enum):
    eq = "="
    gt = ">"
    ge = ">="
    lt = "<"
    le = "<="

    def to_opfunc(self) -> Callable:
        """
        Returns the operator associated with the relation.
        """
        if self == Relations.eq:
            return op.eq
        elif self == Relations.gt:
            return op.gt
        elif self == Relations.ge:
            return op.ge
        elif self == Relations.lt:
            return op.lt
        elif self == Relations.le:
            return op.le
        else:  # pragma: no cover
            raise ValueError(f"Unsupported relation: {self}")

    def to_latex(self) -> str:
        """
        Returns the LaTeX representation of the relation.
        """
        if self == Relations.eq:
            return "="
        elif self == Relations.gt:
            return ">"
        elif self == Relations.ge:
            return r"\geq"
        elif self == Relations.lt:
            return "<"
        elif self == Relations.le:
            return r"\leq"
        else:  # pragma: no cover
            raise ValueError(f"Unsupported relation: {self}")


class Relation(Function):
    """
    Class to express relations. You can use this class to express equalities
    and inequalities. The class is a subclass of :class:`~sigmaepsilon.math.function.function.Function`
    and can be instantiated similatly, with an additional parameter `op` to specify the operator.

    Parameters
    ----------
    op or operator : str or callable
        The operator to be used in the relation. If a string is provided, it should be one of
        =, >, >=, <, <=. If a callable is provided, it should implement a binary operator.
    op_str: {'=', '>', '>=', '<', '<='}, Optional
        If the operator is defined with a callable, you can define the symbol for the operator as
        a string. This is optional and only used in some cases, for instance when generating
        the LaTeX representation of the relation. Also, providing this parameter will help the
        constructor to tetermine the class of the instance when calling the `__new__` method.

    See Also
    --------
    :class:`~sigmaepsilon.math.function.function.Function`
    :class:`~sigmaepsilon.math.function.relation.Equality`
    :class:`~sigmaepsilon.math.function.relation.InEquality`
    """

    __slots__ = ["op", "opfunc", "slack", "op_str"]

    def __new__(cls, *args, op_str: str | None = None, **kwargs):
        # Determine the operator from the arguments
        op = getasany(["op", "operator"], None, **kwargs)

        string_based_input = isinstance(args[0], str) if len(args) == 1 else False
        has_op_in_input = has_operator(args[0]) if string_based_input else False

        if isinstance(op, str):
            if op not in valid_operators:
                raise ValueError(f"Invalid operator: {op}")

            if not op_str:
                op_str = op

            op = Relations(op)

        if not op and has_op_in_input:
            _, operator, _ = parse_expression(args[0])
            op_str = op
            op = Relations(operator)

        if not op and not has_op_in_input:
            if op_str in valid_operators:
                op = Relations(op_str)
            else:
                op = Relations.eq
                op_str = "="

        if op:
            if isinstance(op, Relations):
                op_str = op.value
            elif op_str in valid_operators:
                op = Relations(op_str)

        if cls is not Relation:
            instance = super().__new__(cls)
        elif op == Relations.eq:
            instance = super().__new__(Equality)
        elif op in {Relations.gt, Relations.ge, Relations.lt, Relations.le}:
            instance = super().__new__(InEquality)
        else:
            instance = super().__new__(cls)

        return instance

    def __init__(self, *args, op_str: str | None = None, **kwargs):
        self.op = None
        self.opfunc = None
        self.slack = 0
        self.op_str = op_str

        string_based_input = isinstance(args[0], str) if len(args) == 1 else False
        has_op_in_input = has_operator(args[0]) if string_based_input else False

        op = getasany(["op", "operator"], None, **kwargs)

        if op:
            if isinstance(op, str):
                if op not in valid_operators:
                    raise ValueError(f"Invalid operator: {op}")
                self.op = op = Relations(op)
            elif isinstance(op, Relations):
                self.op = op
            elif isinstance(op, Callable):
                self.opfunc = op
                self.op = None

        if not op and not has_op_in_input:
            if op_str in valid_operators:
                self.op = op = Relations(op_str)
            else:
                self.op = op = Relations.eq

        if not op and has_op_in_input:
            lhs, operator, rhs = parse_expression(args[0])
            rhs = "(" + rhs + ")"
            args = (" - ".join([lhs, rhs]),)
            self.op = op = Relations(operator)

        if op and isinstance(self.op, Relations):
            self.opfunc = self.op.to_opfunc()
            self.op_str = self.op.value

        super().__init__(*args, **kwargs)

    @property
    def operator(self) -> Callable:
        """
        Returns the associated operator.
        """
        return self.op

    def to_latex(self) -> str:
        """
        Returns the LaTeX code of the symbolic expression of the instance.
        Only for symbolic relations.
        """
        expr_str = super().to_latex()
        if self.op in Relations:
            return f"{expr_str} {self.op.to_latex()} 0"
        elif self.op_str:
            op = Relations(self.op_str)
            return f"{expr_str} {op.to_latex()} 0"
        else:  # pragma: no cover
            raise NotImplementedError(
                "LaTeX code generation not available for this kind of function definition."
            )

    def relate(self, *args, **kwargs):
        """
        Relates an input and returns True if it is feasible.
        """
        return self.opfunc(self.f0(*args, **kwargs), 0)


class Equality(Relation):
    """
    Class for equalities, mostly used for expressing
    constraints in optimization problems.

    Examples
    --------
    >>> from sigmaepsilon.math.function import Equality
    >>> import sympy as sy
    >>> variables = ['x1', 'x2', 'x3', 'x4']
    >>> x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
    >>> eq1 = Equality(x1 + 2*x3 + x4 - 4, variables=syms)
    >>> eq2 = Equality(x2 + x3 - x4 - 2, variables=syms)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self, "op")
        if self.op:
            assert self.op == Relations.eq


class InEquality(Relation):
    """
    Class for inequalities, mostly used for expressing
    constraints in optimization problems.

    Examples
    --------
    >>> from sigmaepsilon.math.function import InEquality
    >>> gt = InEquality('x + y', op='>')
    >>> ge = InEquality('x + y', op='>=')
    >>> le = InEquality('x + y', op=lambda x, y: x <= y)
    >>> lt = InEquality('x + y', op=lambda x, y: x < y)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self, "op")
        if self.op:
            assert self.op in {Relations.gt, Relations.ge, Relations.lt, Relations.le}
