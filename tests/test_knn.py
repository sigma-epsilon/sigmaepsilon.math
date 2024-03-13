import unittest

import numpy as np

from sigmaepsilon.math.knn import k_nearest_neighbours


class TestKNN(unittest.TestCase):

    def test_knn(self):
        X = 100*np.random.rand(10, 3)
        i = k_nearest_neighbours(X, X, k=3, max_distance=10.0)

if __name__ == "__main__":
    unittest.main()
