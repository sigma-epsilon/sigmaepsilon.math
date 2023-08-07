import unittest

import numpy as np

from sigmaepsilon.math.linalg.testing import LinalgTestCase
from sigmaepsilon.math.linalg.utils import (
    generalized_inverse,
    generalized_left_inverse,
    generalized_right_inverse,
    random_posdef_matrix
)
from sigmaepsilon.math.linalg.exceptions import LinalgError


class TestMatrixInverse(LinalgTestCase):
    def test_generalized_inverse(self):
        matrix = random_posdef_matrix(3)
        generalized_inverse(matrix)
        
        matrix = random_posdef_matrix(3)
        generalized_inverse(matrix[:, :2])
        
        matrix = random_posdef_matrix(3)
        generalized_inverse(matrix[:2, :])
        
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertFailsProperly(LinalgError, generalized_inverse, matrix)
        
    def test_generalized_left_inverse(self):
        matrix = random_posdef_matrix(3)
        generalized_left_inverse(matrix[:, :2])
        
    def test_generalized_right_inverse(self):
        matrix = random_posdef_matrix(3)
        generalized_right_inverse(matrix[:2, :])


if __name__ == "__main__":
    unittest.main()
