from typing import Tuple, Any, Iterable
from numbers import Number

import numpy as np
from numpy.linalg import norm
from numpy import ndarray

from ..function import Function


class MLSWeightFunction(Function):
    """
    Base class for weight functions for the moving least squares method.
    """

    def __init__(
        self,
        *,
        core: int | Iterable[Number] | ndarray | None = None,
        supportdomain: Iterable[Number] | None = None,
        sd: Iterable[Number] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        dim = None
        if not isinstance(core, ndarray):
            if isinstance(core, Iterable):
                core = np.array(core)
                dim = core.shape[0]
            elif isinstance(core, int):
                dim = 1
                core = np.zeros([dim])
        else:
            dim = core.shape[0]

        if dim is not None:
            self.dimension = dim

        if not any([sd is None, supportdomain is None]):
            raise ValueError(
                "`supportdomain` and `sd` cannot be both specified at the same time."
            )
        sd = sd if sd is not None else supportdomain

        self._core = core
        self._supportdomain = sd

    @property
    def core(self) -> ndarray | Iterable[Number] | None:
        return self._core

    @core.setter
    def core(self, val: ndarray | Iterable[Number] | Number | None):
        if not isinstance(val, ndarray):
            if isinstance(val, Iterable):
                val = np.array(val)
            elif isinstance(val, Number):
                val = np.array([val])
            else:
                raise ValueError(
                    f"Expected a NumPy ndarray, or an Iterable, got {type(val)}"
                )
        self._core = val
        return

    @property
    def supportdomain(self) -> ndarray | Iterable[Number] | None:
        return self._supportdomain

    @supportdomain.setter
    def supportdomain(self, val: ndarray | Iterable[Number] | None):
        if not isinstance(val, ndarray):
            if isinstance(val, Iterable):
                val = np.array(val)
            else:
                raise ValueError(
                    f"Expected a NumPy ndarray, or an Iterable, got {type(val)}"
                )
        self._supportdomain = val
        return

    def preproc_evaluation(self, x: Iterable[Number]):
        if not isinstance(x, (ndarray, float)):
            if isinstance(x, Iterable):
                x = np.array(x)
            else:
                raise ValueError(
                    f"Expected a NumPy ndarray, or an Iterable, got {type(x)}"
                )

    def value(self, x: Iterable[Number]) -> float:
        """
        Evaluates the function.
        """
        raise NotImplementedError


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

    def __init__(self, *, value: Number = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._value = value
        return

    def value(self, x: Iterable[Number]) -> float:
        if self.supportdomain is None:
            return self._value

        d = np.subtract(self.core, x)
        r = np.abs(d / np.array(self.supportdomain))

        if any(r > 1):
            return 0

        return self._value

    def gradient(self, x: Iterable[Number]) -> ndarray:
        return np.zeros(self.dimension, dtype=float)

    def Hessian(self, x: Iterable[Number]) -> ndarray:
        return np.zeros((self.dimension, self.dimension), dtype=float)


class SingularWeightFunction(MLSWeightFunction):
    """
    A singular weight function for the moving least squares method.
    """

    def __init__(self, *, eps: Number = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        return

    def value(self, x: Iterable[Number]):
        self.preproc_evaluation(x)
        return 1 / (norm(np.subtract(self.core, x)) ** 2 + self.eps**2)

    def gradient(self, x: Iterable[Number]) -> ndarray:
        return np.zeros(self.dimension, dtype=float)

    def Hessian(self, x: Iterable[Number]) -> ndarray:
        return np.zeros((self.dimension, self.dimension), dtype=float)


class CubicWeightFunction(MLSWeightFunction):
    """
    A cubic weight function for the moving least squares method.

    Example
    -------
    >>> from sigmaepsilon.math.approx import CubicWeightFunction
    >>> w = CubicWeightFunction(core=[0.0, 0.0], sd=[0.5, 0.5])
    >>> w([0.0, 0.0])
    0.4444444444444444

    """

    def evaluate(self, x: Iterable[Number]) -> Tuple[float, ndarray, ndarray]:
        if self.dimension == 1:
            return self._evaluate_1d(x)
        elif self.dimension == 2:
            return self._evaluate_2d(x)

        raise NotImplementedError

    def _evaluate_1d(self, x: Iterable[Number]) -> Tuple[float, ndarray, ndarray]:
        d = np.subtract(self.core, x)
        difX = d[0]
        dmX = self.supportdomain[0]
        rX = abs(difX) / dmX

        if abs(difX) < 1e-12:
            drdX = 0
        else:
            drdX = (difX / abs(difX)) / dmX

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

        val = wX
        grad = np.array([dwXdX])
        Hessian = np.array([[dwXdXX]])

        return val, grad, Hessian

    def _evaluate_2d(self, x: Iterable[Number]) -> Tuple[float, ndarray, ndarray]:
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
        self.preproc_evaluation(x)
        res, _, _ = self.evaluate(x)
        return res

    def gradient(self, x: Iterable[Number]) -> ndarray:
        self.preproc_evaluation(x)
        _, res, _ = self.evaluate(x)
        return res

    def Hessian(self, x: Iterable[Number]) -> ndarray:
        self.preproc_evaluation(x)
        _, _, res = self.evaluate(x)
        return res
