# from collections.abc import Iterable
from typing import Tuple, Union, Optional, Any, Iterable
from numbers import Number

import numpy as np
from numpy.linalg import norm
from numpy import ndarray

from ..function import Function


class MLSWeightFunction(Function):
    """
    Base class for weight functions for the moving least squares method.
    """

    def __init__(self, core: Union[int, Iterable[Number], ndarray]):
        if not isinstance(core, np.ndarray):
            if isinstance(core, Iterable):
                core = np.array(core)
                dim = core.shape[0]
            elif isinstance(core, int):
                dim = core
                core = np.zeros([dim])
            else:
                raise ValueError(f"Type {type(core)} is invalid for 'core'.")
        else:
            dim = core.shape[0]
        super().__init__(d=dim)
        self.core = core

    def value(self, x: Iterable[Number]) -> float:
        """
        Evaluates the function.
        """
        if not isinstance(x, np.ndarray):
            if isinstance(x, Iterable):
                x = np.array(x)
            else:
                raise ValueError(
                    f"Expected a NumPy ndarray, or an Iterable, got {type(x)}"
                )


def isMLSWeightFunction(f: Any) -> bool:
    """
    Returns `True` if the argument is a valid weight function for the
    moving least squares method.
    """
    c1 = isinstance(f, MLSWeightFunction)
    c2 = MLSWeightFunction in list(type(f).__bases__)
    return any([c1, c2])


class ConstantWeightFunction(MLSWeightFunction):
    """
    A constant weight function for the moving least squares method.
    """

    def __init__(self, dimension: int, val: Optional[Number] = 1.0):
        super().__init__(np.zeros([dimension]))
        self.val = val
        return

    def value(self, x: Iterable[Number]) -> float:
        return self.val

    def gradient(self, x: Iterable[Number]) -> ndarray:
        return np.zeros([self.dimension])

    def Hessian(self, x: Iterable[Number]) -> ndarray:
        return np.zeros([self.dimension, self.dimension])


class SingularWeightFunction(MLSWeightFunction):
    """
    A singular weight function for the moving least squares method.
    """

    def __init__(
        self, core: Union[int, Iterable[Number], ndarray], eps: Optional[float] = 1e-5
    ):
        super().__init__(core)
        self.eps = eps
        return

    def value(self, x: Iterable[Number]):
        super().value(x)
        return 1 / (norm(np.subtract(self.core, x)) ** 2 + self.eps**2)


class CubicWeightFunction(MLSWeightFunction):
    """
    A cubic weight function for the moving least squares method.

    Example
    -------
    >>> from sigmaepsilon.math.approx import CubicWeightFunction
    >>> w = CubicWeightFunction([0.0, 0.0], [0.5, 0.5])
    >>> w([0.0, 0.0])
    0.4444444444444444
    """

    def __init__(
        self,
        core: Union[int, Iterable[Number], ndarray],
        supportdomain: Iterable[Number],
    ):
        super().__init__(core)
        assert self.dimension == 2
        self.supportdomain = supportdomain
        return

    def evaluator(self, x: Iterable[Number]) -> Tuple[float, ndarray, ndarray]:
        d = np.subtract(self.core, x)
        difX = d[0]
        difY = d[1]
        dmX = self.supportdomain[0]
        dmY = self.supportdomain[1]
        rX = abs(difX) / dmX
        rY = abs(difY) / dmY

        if abs(difX) < 1e-12:
            drdX = 0
        else:
            drdX = (difX / abs(difX)) / dmX

        if abs(difY) < 1e-12:
            drdY = 0
        else:
            drdY = (difY / abs(difY)) / dmY

        if rX <= 0.5:
            wX = 2 / 3 - 4 * rX**2 + 4 * rX**3
            dwXdX = (-8 * rX + 12 * rX**2) * drdX
            dwXdXX = (-8 + 24 * rX) * drdX * drdX
        elif rX > 0.5 and rX <= 1:
            wX = 4 / 3 - 4 * rX + 4 * rX**2 - (4 / 3) * rX**3
            dwXdX = (-4 + 8 * rX - 4 * rX**2) * drdX
            dwXdXX = (8 - 8 * rX) * drdX * drdX
        else:
            wX = 0
            dwXdX = 0
            dwXdXX = 0

        if rY <= 0.5:
            wY = 2 / 3 - 4 * rY**2 + 4 * rY**3
            dwYdY = (-8 * rY + 12 * rY**2) * drdY
            dwYdYY = (-8 + 24 * rY) * drdY * drdY
        elif rY > 0.5 and rY <= 1:
            wY = 4 / 3 - 4 * rY + 4 * rY**2 - (4 / 3) * rY**3
            dwYdY = (-4 + 8 * rY - 4 * rY**2) * drdY
            dwYdYY = (8 - 8 * rY) * drdY * drdY
        else:
            wY = 0
            dwYdY = 0
            dwYdYY = 0

        val = wX * wY
        grad = np.array([wY * dwXdX, wX * dwYdY])
        Hessian = np.array([[wY * dwXdXX, dwXdX * dwYdY], [dwXdX * dwYdY, wX * dwYdYY]])

        return val, grad, Hessian

    def value(self, x: Iterable[Number]) -> float:
        super().value(x)
        res, _, _ = self.evaluator(x)
        return res

    def gradient(self, x: Iterable[Number]) -> ndarray:
        super().value(x)
        _, res, _ = self.evaluator(x)
        return res

    def Hessian(self, x: Iterable[Number]) -> ndarray:
        super().value(x)
        _, _, res = self.evaluator(x)
        return res
