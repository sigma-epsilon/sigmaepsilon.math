from typing import Tuple, Any, Iterable
from numbers import Number

import numpy as np
from numpy.linalg import norm
from numpy import ndarray

from ..function import Function


class ConstantWeightFunction(Function):
    """
    A constant weight function for the moving least squares method.
    """

    def __init__(self, *, value: Number = 1.0, **kwargs):
        super().__init__(**kwargs)
        
        if not isinstance(value, (int, float)):
            raise TypeError(f"Value must be a number, got {type(value)}")
        
        self._value = value
    
    def _check_input_dimensions(self, x: Number | Iterable[Number]) -> None:
        pass
    
    def _return_value(self, x: Iterable[Number]) -> Number | ndarray[Number]:
        if isinstance(x, (int, float)):
            return self._value
        elif isinstance(x, (ndarray, list)):
            return np.full_like(x, self._value)

    def value(self, x: Iterable[Number]) -> float:
        if self.supportdomain is None:
            return self._return_value(x)

        d = np.subtract(self.core, x)
        r = np.abs(d / np.array(self.supportdomain))

        if any(r > 1):
            return 0

        return self._return_value(x)

    def gradient(self, x: Iterable[Number]) -> ndarray:
        return np.zeros(self.dimension, dtype=float)

    def Hessian(self, x: Iterable[Number]) -> ndarray:
        return np.zeros((self.dimension, self.dimension), dtype=float)


class SingularWeightFunction(Function):
    """
    A singular weight function for the moving least squares method.
    """

    def __init__(self, *, eps: float = 1e-5, **kwargs):
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
