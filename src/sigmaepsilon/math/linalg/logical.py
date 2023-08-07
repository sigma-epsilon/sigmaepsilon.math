import numpy as np
from numpy import ndarray


__all__ = ["has_full_row_rank", "has_full_column_rank", "has_full_rank"]


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


def has_full_rank(matrix: ndarray) -> bool:
    """
    Returns `True` if the input matrix has full rank.
    
    See also
    --------
    :func:`numpy.linalg.matrix_rank`
    """
    num_rows, num_columns = matrix.shape
    rank = np.linalg.matrix_rank(matrix)
    return rank == num_columns == num_rows