import unittest

import numpy as np
import sympy as sy

from sigmaepsilon.math.approx.lagrange import gen_Lagrange_1d, approx_Lagrange_1d
from sigmaepsilon.math.approx.func import CubicWeightFunction
from sigmaepsilon.math.approx import least_squares


class TestLagrange(unittest.TestCase):
    def test_1d(self):
        gen_Lagrange_1d(x=[-1, 0, 1])
        gen_Lagrange_1d(i=[1, 2], sym=True)
        gen_Lagrange_1d(i=[1, 2], sym=False, lambdify=True)
        gen_Lagrange_1d(i=[1, 2, 3], sym=False)
        gen_Lagrange_1d(N=3)
        approx_Lagrange_1d([-1, 1], [0, 10], lambdify=True)
        approx_Lagrange_1d([-1, 1], [0, 10], lambdify=False)

    def test_approx_Lagrange_1d(self):
        source, target = [-1, 1], [0, 10]
        approx = approx_Lagrange_1d(source, target, lambdify=True)
        for s, t in zip(source, target):
            self.assertTrue(np.isclose(approx(s), t))

        L = sy.symbols("L", real=True, positive=True)
        source, target = [-1, 1], [0, L]
        approx = approx_Lagrange_1d(source, target)
        v = float(approx.subs([("x", -1), (L, 10)]))
        self.assertTrue(np.isclose(v, 0))
        v = float(approx.subs([("x", 1), (L, 10)]))
        self.assertTrue(np.isclose(v, 10))


class TestMLS(unittest.TestCase):
    def test_cubic_weight_function(self):
        w = CubicWeightFunction([0.0, 0.0], [0.5, 0.5])
        w([0.0, 0.0])


class TestLS(unittest.TestCase):
    def test_least_squares_approximation_multiset(self):
        # the coordinates of the known values
        points = [
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]
        points = np.array(points)

        # two sets of known values
        values = [[1, -0.5, 1, 1, -1, 0, 0, 0, 0], [1, -1, 0, 0, 1, 0, -1, -1, 1]]

        # transpose the values because the function expext datasets to be columns
        values = np.array(values).T

        approx = least_squares(points, values, deg=2, order=1)
        approx([0, 0])
        
    def test_least_squares_approximation(self):
        # the coordinates of the known values
        points = [
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]
        points = np.array(points)

        # two sets of known values
        values = [1, -0.5, 1, 1, -1, 0, 0, 0, 0]

        # transpose the values because the function expext datasets to be columns
        values = np.array(values).T

        approx = least_squares(points, values, deg=2, order=1)
        approx([0, 0])


if __name__ == "__main__":
    unittest.main()
