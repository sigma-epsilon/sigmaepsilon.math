import unittest

import numpy as np

from sigmaepsilon.math.linalg.testing import LinalgTestCase
from sigmaepsilon.math.linalg.utils import (
    random_pos_semidef_matrix,
    random_posdef_matrix,
)
from sigmaepsilon.math.linalg.logical import (
    is_pos_def,
    is_pos_semidef,
    is_rectangular_frame,
    is_normal_frame,
    is_orthonormal_frame,
    is_independent_frame,
    is_hermitian,
)

class TestLinalgLogical(LinalgTestCase):
    def test_frame(self):
        A = np.eye(3)
        is_independent_frame(A)
        is_orthonormal_frame(A)
        is_normal_frame(A)
        is_rectangular_frame(A)
        
    def test_random_pos_semidef(self, N=5):
        """
        Tests the creation of random, positive semidefinite matrices.
        """
        self.assertTrue(is_pos_semidef(random_pos_semidef_matrix(N)))
        
    def test_random_posdef(self, N=2):
        """
        Tests the creation of random, positive definite matrices.
        """
        self.assertTrue(is_pos_def(random_posdef_matrix(N)))
        
    def test_hermitian(self):
        self.assertTrue(is_hermitian(random_posdef_matrix(3)))
        self.assertFalse(is_hermitian(random_posdef_matrix(3)[:, :2]))


if __name__ == "__main__":
    unittest.main()
