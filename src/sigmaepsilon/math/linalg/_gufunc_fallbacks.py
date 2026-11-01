"""Pure NumPy stand-ins for the gufuncs in :mod:`sigmaepsilon.math.linalg.utils`.

They are used if Numba is not installed. Like the gufuncs, they accept a single
matrix or a stack of matrices, return the result if called as ``f(A)`` and write
it into ``out`` if called as ``f(A, out)``.
"""

import numpy as np
from numpy import ndarray

__all__ = ["guvectorize"]


def guvectorize(*args, **kwargs):
    """Return the NumPy implementation of the decorated gufunc, looked up by name."""
    return lambda kernel: _IMPLEMENTATIONS[kernel.__name__]


def _result(res: ndarray, out: ndarray | None) -> ndarray:
    if out is None:
        return res
    out[...] = res
    return out


def _det2x2(A: ndarray) -> ndarray:
    return A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]


def _det3x3(A: ndarray) -> ndarray:
    return (
        A[..., 0, 0] * A[..., 1, 1] * A[..., 2, 2]
        - A[..., 0, 0] * A[..., 1, 2] * A[..., 2, 1]
        - A[..., 0, 1] * A[..., 1, 0] * A[..., 2, 2]
        + A[..., 0, 1] * A[..., 1, 2] * A[..., 2, 0]
        + A[..., 0, 2] * A[..., 1, 0] * A[..., 2, 1]
        - A[..., 0, 2] * A[..., 1, 1] * A[..., 2, 0]
    )


def _adj3x3(A: ndarray) -> ndarray:
    res = np.empty_like(A)
    res[..., 0, 0] = A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1]
    res[..., 0, 1] = -A[..., 0, 1] * A[..., 2, 2] + A[..., 0, 2] * A[..., 2, 1]
    res[..., 0, 2] = A[..., 0, 1] * A[..., 1, 2] - A[..., 0, 2] * A[..., 1, 1]
    res[..., 1, 0] = -A[..., 1, 0] * A[..., 2, 2] + A[..., 1, 2] * A[..., 2, 0]
    res[..., 1, 1] = A[..., 0, 0] * A[..., 2, 2] - A[..., 0, 2] * A[..., 2, 0]
    res[..., 1, 2] = -A[..., 0, 0] * A[..., 1, 2] + A[..., 0, 2] * A[..., 1, 0]
    res[..., 2, 0] = A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0]
    res[..., 2, 1] = -A[..., 0, 0] * A[..., 2, 1] + A[..., 0, 1] * A[..., 2, 0]
    res[..., 2, 2] = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    return res


def det3x3(A, out=None):
    """Compute the determinant of a 3x3 matrix, or a stack of such matrices."""
    return _result(_det3x3(np.asarray(A, dtype=float)), out)


def det2x2(A, out=None):
    """Compute the determinant of a 2x2 matrix, or a stack of such matrices."""
    return _result(_det2x2(np.asarray(A, dtype=float)), out)


def inv2x2u(A, out=None):
    """Compute the inverse of a 2x2 matrix, or a stack of such matrices."""
    A = np.asarray(A, dtype=float)
    res = np.empty_like(A)
    res[..., 0, 0] = A[..., 1, 1]
    res[..., 1, 1] = A[..., 0, 0]
    res[..., 0, 1] = -A[..., 0, 1]
    res[..., 1, 0] = -A[..., 1, 0]
    return _result(res / _det2x2(A)[..., None, None], out)


def adj3x3(A, out=None):
    """Compute the adjugate of a 3x3 matrix, or a stack of such matrices."""
    return _result(_adj3x3(np.asarray(A, dtype=float)), out)


def inv3x3u(A, out=None):
    """Compute the inverse of a 3x3 matrix, or a stack of such matrices."""
    A = np.asarray(A, dtype=float)
    return _result(_adj3x3(A) / _det3x3(A)[..., None, None], out)


_IMPLEMENTATIONS = {f.__name__: f for f in (det2x2, det3x3, inv2x2u, adj3x3, inv3x3u)}
