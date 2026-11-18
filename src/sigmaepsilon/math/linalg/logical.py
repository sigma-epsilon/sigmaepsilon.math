"""Logical predicates for arrays, matrices and reference frames."""

import warnings

import numpy as np
from numpy import ndarray
import sympy as sy

from .utils import Gram

__all__ = [
    "has_full_row_rank",
    "has_full_column_rank",
    "has_full_rank",
    "is_rectangular_frame",
    "is_normal_frame",
    "is_orthonormal_frame",
    "is_independent_frame",
    "has_equal_dimensions",
    "is_hermitian",
    "is_pos_def",
    "is_pos_semidef",
]


def is_pos_def(arr) -> bool:
    """Return True if the input is positive definite."""
    return bool(np.all(np.linalg.eigvals(arr) > 0))


def is_pos_semidef(arr) -> bool:
    """Return True if the input is positive semi definite."""
    return bool(np.all(np.linalg.eigvals(arr) >= 0))


def is_rectangular_frame(axes: ndarray) -> bool:
    """Return True if a frame is Cartesian.

    Parameters
    ----------
    axes: numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    assert len(axes.shape) == 2, "Input is not a matrix!"
    assert axes.shape[0] == axes.shape[1], "Input is not a square matrix!"
    agram = np.abs(axes @ axes.T)
    return np.isclose(np.trace(agram), np.sum(agram))


def is_normal_frame(axes: ndarray) -> bool:
    """Return True if a frame is normal.

    This means that its base vectors are all of unit length.

    Parameters
    ----------
    axes: numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return np.allclose(np.linalg.norm(axes, axis=1), 1.0)


def is_orthonormal_frame(axes: ndarray) -> bool:
    """Return True if a frame is orthonormal.

    Parameters
    ----------
    axes: numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return is_rectangular_frame(axes) and is_normal_frame(axes)


def is_independent_frame(axes: ndarray, tol: float = 0) -> bool:
    """Return True if the base vectors of a frame are linearly independent.

    Parameters
    ----------
    axes: numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return np.linalg.det(Gram(axes)) > tol


def has_equal_dimensions(arr: ndarray) -> bool:
    """Return True if all the dimensions of the input array are equal.

    This is the shape of the components of a tensor of any order, like
    ``(3, 3)`` or ``(2, 2, 2, 2)``.
    """
    shp = arr.shape
    return all(s == shp[0] for s in shp[1:])


def is_hermitian(arr: ndarray) -> bool:
    """Return True if all the dimensions of the input array are equal.

    .. deprecated:: 2.9.0
        The name is misleading, as the function doesn't check if the array is
        Hermitian. Use :func:`has_equal_dimensions` instead.
    """
    warnings.warn(
        "'is_hermitian' is deprecated, use 'has_equal_dimensions' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return has_equal_dimensions(arr)


def has_full_row_rank(matrix: ndarray) -> bool:
    """Return `True` if the input matrix has full row rank.

    This means that all its rows are linearly independent.

    See Also
    --------
    :func:`numpy.linalg.matrix_rank`
    """
    num_rows, num_columns = matrix.shape
    if num_rows > num_columns:
        return False
    rank = np.linalg.matrix_rank(matrix)
    return rank == num_rows


def has_full_column_rank(matrix: ndarray) -> bool:
    """Return `True` if the input matrix has full column rank.

    This means that all its columns are linearly independent.

    See Also
    --------
    :func:`numpy.linalg.matrix_rank`
    """
    num_rows, num_columns = matrix.shape
    if num_columns > num_rows:
        return False
    rank = np.linalg.matrix_rank(matrix)
    return rank == num_columns


def has_full_rank(matrix: ndarray | sy.Matrix) -> bool:
    """Return `True` if the input matrix has full rank, `False` otherwise.

    Parameters
    ----------
    matrix : numpy.ndarray | sympy.Matrix
        The input matrix.
    """
    if isinstance(matrix, sy.Matrix):
        return matrix.rank() == min(matrix.shape)
    else:
        return np.linalg.matrix_rank(matrix) == min(matrix.shape)
