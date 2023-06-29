from enum import Enum

import numpy as np

from .function import Function


class ConstraintType(Enum):
    nan = 0
    equality = 1
    inequality = 2


class ConstraintFunction(Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.constraintType = ConstraintType.nan
        for arg in args:
            if arg in ["eq", "equality"]:
                self.constraintType = ConstraintType.equality
                break
            elif arg in ["iq", "inequality"]:
                self.constraintType = ConstraintType.inequality
                break
            elif isinstance(arg, ConstraintType):
                self.constraintType = arg
                break
        assert self.constraintType != ConstraintType.nan


class EqualityConstraint(ConstraintFunction):
    def __init__(self, *args, **kwargs):
        super().__init__("eq", *args, **kwargs)


class InequalityConstraint(ConstraintFunction):
    def __init__(self, *args, **kwargs):
        super().__init__("iq", *args, **kwargs)


class PenaltyType(Enum):
    courant = 1


class PenaltyFunction(EqualityConstraint):
    """
    Penalty function class for equality constraints.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.PenaltyType = PenaltyType.courant

        if "penalty" in kwargs:
            assert isinstance(kwargs["penalty"], float)
            self.penalty = kwargs["penalty"]
        else:
            self.penalty = np.array(1e12)

    def f(self, *args, **kwargs):
        try:
            if self.PenaltyType == PenaltyType.courant:
                f = self.f0(*args, **kwargs)
                p = self.penalty
                return np.multiply(0.5 * p, np.power(f, 2))
        except:
            return None

    def g(self, *args, **kwargs):
        try:
            if self.PenaltyType == PenaltyType.courant:
                f = self.f0(*args, **kwargs)
                g = self.f1(*args, **kwargs)
                p = self.penalty
                return np.multiply(np.multiply(p, f), g)
        except:
            return None

    def G(self, *args, **kwargs):
        try:
            if self.PenaltyType == PenaltyType.courant:
                f = self.f0(*args, **kwargs)
                g = self.f1(*args, **kwargs)
                G = self.f2(*args, **kwargs)
                p = self.penalty
                return np.multiply(p, np.add(np.multiply(f, G), np.outer(g, g)))
        except:
            return None
