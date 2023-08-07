import numpy as np
from numpy import ndarray


__all__ = ["has_full_row_rank", "has_full_column_rank"]


def has_full_row_rank(matrix: ndarray) -> bool:
    """
    Returns `True` if the input matrix has full row rank, ie
    if all its rows are linearly independent. 
    """
    num_rows = matrix.shape[0]
    rank = np.linalg.matrix_rank(matrix)
    return rank == num_rows


def has_full_column_rank(matrix: ndarray) -> bool:
    """
    Returns `True` if the input matrix has full column rank, ie
    if all its columns are linearly independent.
    """
    num_columns = matrix.shape[1]
    rank = np.linalg.matrix_rank(matrix)
    return rank == num_columns