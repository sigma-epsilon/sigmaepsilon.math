import unittest
from typing import Callable
import numpy as np

from sigmaepsilon.math.approx import moving_least_squares
from sigmaepsilon.math.approx.functions import (
    CubicWeightFunction,
    SingularWeightFunction,
    ConstantWeightFunction,
)
from sigmaepsilon.math.approx import least_squares


class TestMLSApprox3dRandomPointCloudData(unittest.TestCase):

    def setUp(self):
        nx, ny, nz = (10, 10, 10)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        z = np.linspace(0, 1, nz)
        xv, yv, zv = np.meshgrid(x, y, z)
        points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)
        values = np.random.rand(nx * ny * nz)
        self.coords = points
        self.data = values
        
    def _test_mls_3d(self, w: Callable, *args, **kwargs):
        approx = moving_least_squares(*args, w=w, **kwargs)
        f, fdx, fdy, fdz, fdxx, fdyy, fdzz, fdxy, fdxz, fdyz = approx([0, 0, 0])
        
    def test_least_squares(self):
        approx = least_squares(self.coords, self.data, deg=1, order=2)
        f, fdx, fdy, fdz, fdxx, fdyy, fdzz, fdxy, fdxz, fdyz = approx([0, 0, 0])

    def test_mls_3d_constant_nosupport(self):
        w = ConstantWeightFunction(value=1, dim=3)
        self._test_mls_3d(w, self.coords, self.data, deg=1, order=2)
        self._test_mls_3d(w, self.coords, self.data, deg=2, order=2)

    def test_mls_3d_constant_supportdomain(self):
        w = ConstantWeightFunction(core=[0.0, 0.0, 0.0], supportdomain=[0.5, 0.5, 0.5])
        self._test_mls_3d(w, self.coords, self.data, deg=1, order=2)
        self._test_mls_3d(w, self.coords, self.data, deg=2, order=2)
        
    def test_mls_3d_singular_nosupport(self):
        w = SingularWeightFunction(core=[0.0, 0.0, 0.0], supportdomain=[0.5, 0.5, 0.5])
        self._test_mls_3d(w, self.coords, self.data, deg=1, order=2)
        self._test_mls_3d(w, self.coords, self.data, deg=2, order=2)

    def test_mls_3d_singular_supportdomain(self):
        w = SingularWeightFunction(core=[0.0, 0.0, 0.0], supportdomain=[0.5, 0.5, 0.5])
        self._test_mls_3d(w, self.coords, self.data, deg=1, order=2)
        self._test_mls_3d(w, self.coords, self.data, deg=2, order=2)


if __name__ == "__main__":
    unittest.main()
