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
    "is_hermitian",
    "is_pos_def",
    "is_pos_semidef",
]


def is_pos_def(arr) -> bool:
    """
    Returns True if the input is positive definite.
    """
    return np.all(np.linalg.eigvals(arr) > 0)


def is_pos_semidef(arr) -> bool:
    """
    Returns True if the input is positive semi definite.
    """
    return np.all(np.linalg.eigvals(arr) >= 0)


def is_rectangular_frame(axes: ndarray) -> bool:
    """
    Returns True if a frame is Cartesian.

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
    """
    Returns True if a frame is normal, meaning, that it's base vectors
    are all of unit length.

    Parameters
    ----------
    axes: numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return np.allclose(np.linalg.norm(axes, axis=1), 1.0)


def is_orthonormal_frame(axes: ndarray) -> bool:
    """
    Returns True if a frame is orthonormal.

    Parameters
    ----------
    axes: numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return is_rectangular_frame(axes) and is_normal_frame(axes)


def is_independent_frame(axes: ndarray, tol: float = 0) -> bool:
    """
    Returns True if a the base vectors of a frame are linearly independent.

    Parameters
    ----------
    axes: numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return np.linalg.det(Gram(axes)) > tol


def is_hermitian(arr: ndarray) -> bool:
    """
    Returns True if the input is a hermitian array.
    """
    shp = arr.shape
    s0 = shp[0]
    return all([s == s0 for s in shp[1:]])


def has_full_row_rank(matrix: ndarray) -> bool:
    """
    Returns `True` if the input matrix has full row rank, ie
    if all its rows are linearly independent.

    See also
    --------
    :func:`numpy.linalg.matrix_rank`
    """
    num_rows, num_columns = matrix.shape
    if num_rows > num_columns:
        return False
    rank = np.linalg.matrix_rank(matrix)
    return rank == num_rows


def has_full_column_rank(matrix: ndarray) -> bool:
    """
    Returns `True` if the input matrix has full column rank, ie
    if all its columns are linearly independent.

    See also
    --------
    :func:`numpy.linalg.matrix_rank`
    """
    num_rows, num_columns = matrix.shape
    if num_columns > num_rows:
        return False
    rank = np.linalg.matrix_rank(matrix)
    return rank == num_columns


def has_full_rank(matrix: ndarray | sy.Matrix) -> bool:
    """
    Returns `True` if the input matrix has full rank, `False` otherwise.

    Parameters
    ----------
    matrix : numpy.ndarray | sympy.Matrix
        The input matrix.
    """
    if isinstance(matrix, sy.Matrix):
        return matrix.rank() == min(matrix.shape)
    else:
        return np.linalg.matrix_rank(matrix) == min(matrix.shape)
