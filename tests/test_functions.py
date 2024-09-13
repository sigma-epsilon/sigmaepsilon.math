import unittest

import numpy as np

from sigmaepsilon.math.function import Function
from sigmaepsilon.math.function.functions import (
    Rosenbrock,
    Himmelblau,
    GoldsteinPrice,
    Beale,
    Matyas,
)


class TestFunctions(unittest.TestCase):

    def test_smoke(self):
        functions = (
            Rosenbrock,
            Himmelblau,
            GoldsteinPrice,
            Beale,
            Matyas,
        )

        x = np.zeros((2))
        x_bulk = np.zeros((10, 2))

        for f_cls in functions:
            f: Function = f_cls()
            f(x)
            f([x_bulk[:, 0], x_bulk[:, 1]])
            f(x_bulk.T)


if __name__ == "__main__":
    unittest.main()
