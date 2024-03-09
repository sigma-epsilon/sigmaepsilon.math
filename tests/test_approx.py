import unittest

import numpy as np
from numpy import ndarray
import sympy as sy

from sigmaepsilon.math.approx import moving_least_squares
from sigmaepsilon.math.approx.lagrange import gen_Lagrange_1d, approx_Lagrange_1d
from sigmaepsilon.math.approx.functions import (
    CubicWeightFunction,
    SingularWeightFunction,
    ConstantWeightFunction,
    isMLSWeightFunction,
)
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


class TestMLSWeightFunctions(unittest.TestCase):
    def test_cubic_weight_function(self):
        w = CubicWeightFunction([0.0, 0.0], [0.5, 0.5])
        self.assertTrue(isMLSWeightFunction(w))
        self.assertTrue(isinstance(w([0.0, 0.0]), float))
        self.assertTrue(np.isclose(w([0.0, 0.0]), 0.444444444444))

    def test_singular_weight_function(self):
        w = SingularWeightFunction([0.0, 0.0])
        self.assertTrue(isMLSWeightFunction(w))
        self.assertTrue(isinstance(w([0.0, 0.0]), float))

    def test_constant_weight_function(self):
        w = ConstantWeightFunction(2, 1.0)
        self.assertTrue(isMLSWeightFunction(w))
        self.assertTrue(isinstance(w([0.0, 0.0]), float))
        self.assertTrue(np.isclose(w([0.0, 0.0]), 1.0))

        w = ConstantWeightFunction(2, 10.0)
        self.assertTrue(isMLSWeightFunction(w))
        self.assertTrue(isinstance(w([0.0, 0.0]), float))
        self.assertTrue(np.isclose(w([0.0, 0.0]), 10.0))


class TestMLSApprox2d(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        points = np.zeros((len(X), 2), dtype=float)
        points[:, 0] = X
        points[:, 1] = Y
        values = np.random.rand(len(X), 2)
        self.points = points
        self.values = values

    def test_mls_2d_deg1_order0(self):
        w = CubicWeightFunction([5.0, 5.0], [0.5, 0.5])
        approx = moving_least_squares(self.points, self.values, deg=1, order=0, w=w)
        f, fdx, fdy, fdxx, fdyy, fdxy = approx([0, 0])
        assert isinstance(f, ndarray)
        assert fdx is None
        assert fdy is None
        assert fdxx is None
        assert fdyy is None
        assert fdxy is None

    def test_mls_2d_deg1_order1(self):
        w = CubicWeightFunction([5.0, 5.0], [0.5, 0.5])
        approx = moving_least_squares(self.points, self.values, deg=1, order=1, w=w)
        f, fdx, fdy, fdxx, fdyy, fdxy = approx([0, 0])
        assert isinstance(f, ndarray)
        assert isinstance(fdx, ndarray)
        assert isinstance(fdy, ndarray)
        assert fdxx is None
        assert fdyy is None
        assert fdxy is None

    def test_mls_2d_deg1_order2(self):
        w = CubicWeightFunction([5.0, 5.0], [0.5, 0.5])
        approx = moving_least_squares(self.points, self.values, deg=1, order=2, w=w)
        f, fdx, fdy, fdxx, fdyy, fdxy = approx([0, 0])
        assert isinstance(f, ndarray)
        assert isinstance(fdx, ndarray)
        assert isinstance(fdy, ndarray)
        assert isinstance(fdxx, ndarray)
        assert isinstance(fdyy, ndarray)
        assert isinstance(fdxy, ndarray)

    def test_mls_2d_deg2_order0(self):
        w = CubicWeightFunction([5.0, 5.0], [0.5, 0.5])
        approx = moving_least_squares(self.points, self.values, deg=2, order=0, w=w)
        f, fdx, fdy, fdxx, fdyy, fdxy = approx([0, 0])
        assert isinstance(f, ndarray)
        assert fdx is None
        assert fdy is None
        assert fdxx is None
        assert fdyy is None
        assert fdxy is None

    def test_mls_2d_deg2_order1(self):
        w = CubicWeightFunction([5.0, 5.0], [0.5, 0.5])
        approx = moving_least_squares(self.points, self.values, deg=2, order=1, w=w)
        f, fdx, fdy, fdxx, fdyy, fdxy = approx([0, 0])
        assert isinstance(f, ndarray)
        assert isinstance(fdx, ndarray)
        assert isinstance(fdy, ndarray)
        assert fdxx is None
        assert fdyy is None
        assert fdxy is None

    def test_mls_2d_deg2_order2(self):
        w = CubicWeightFunction([5.0, 5.0], [0.5, 0.5])
        approx = moving_least_squares(self.points, self.values, deg=2, order=2, w=w)
        f, fdx, fdy, fdxx, fdyy, fdxy = approx([0, 0])
        assert isinstance(f, ndarray)
        assert isinstance(fdx, ndarray)
        assert isinstance(fdy, ndarray)
        assert isinstance(fdxx, ndarray)
        assert isinstance(fdyy, ndarray)
        assert isinstance(fdxy, ndarray)


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
