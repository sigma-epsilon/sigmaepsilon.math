import unittest

import numpy as np
import sympy as sy

from sigmaepsilon.math.linalg.testing import LinalgTestCase
from sigmaepsilon.math.linalg.logical import (
    has_full_column_rank,
    has_full_row_rank,
    has_full_rank,
)


class TestMatrixRank(LinalgTestCase):
    def test_row_rank(self):
        # (Rank = 3, Num rows = 3)
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertFalse(has_full_row_rank(matrix))

        # (Rank = 2, Num rows = 3)
        matrix = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertFalse(has_full_row_rank(matrix))

        # (Rank = 2, Num rows = 2)
        matrix = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(has_full_row_rank(matrix.T))

    def test_column_rank(self):
        # (Rank = 3, Num cols = 3)
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertFalse(has_full_column_rank(matrix))

        # (Rank = 2, Num cols = 2)
        matrix = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(has_full_column_rank(matrix))

        # (Rank = 2, Num cols = 2)
        matrix = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertFalse(has_full_column_rank(matrix.T))

    def test_rank(self):
        # (Rank = 3)
        matrix = np.eye(3)
        self.assertTrue(has_full_rank(matrix))

        # (Rank = 2)
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertFalse(has_full_rank(matrix))
        
        # Example usage with SymPy
        matrix = sy.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Example matrix
        self.assertFalse(has_full_rank(matrix))

        matrix = sy.eye(3)
        self.assertTrue(has_full_rank(matrix))


if __name__ == "__main__":
    unittest.main()
