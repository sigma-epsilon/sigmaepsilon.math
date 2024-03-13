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


class TestMLSApprox1d(unittest.TestCase):

    def setUp(self):
        coords = np.linspace(0, 1, 100)
        random_noise = np.random.normal(0, 0.1, coords.shape)
        data = np.sin(2 * np.pi * coords) + random_noise
        Ddata_Dx = 2 * np.pi * np.cos(2 * np.pi * coords) + random_noise
        Ddata_Dxx = -((2 * np.pi) ** 2) * np.sin(2 * np.pi * coords) + random_noise

        # normmalize data
        ratio_dx = max(np.abs(data.max()), np.abs(data.min())) / max(
            np.abs(Ddata_Dx.max()), np.abs(Ddata_Dx.min())
        )
        Ddata_Dx *= ratio_dx
        ratio_dxx = max(np.abs(data.max()), np.abs(data.min())) / max(
            np.abs(Ddata_Dxx.max()), np.abs(Ddata_Dxx.min())
        )
        Ddata_Dxx *= ratio_dxx

        self.coords = coords
        self.data = data
        self.Ddata_Dx = Ddata_Dx
        self.Ddata_Dxx = Ddata_Dxx
        self.ratio_dx = ratio_dx
        self.ratio_dxx = ratio_dxx

        self.coords_approx = np.linspace(0, 1, 16)

    def _approximate(self, approximator: Callable):
        coords_approx = self.coords_approx
        data_approx = [approximator(x)[0][0] for x in coords_approx]
        Ddata_Dx_approx = [approximator(x)[1][0] for x in coords_approx]
        Ddata_Dxx_approx = [approximator(x)[2][0] for x in coords_approx]
        _control_data = [approximator(x)[0][0] for x in self.coords]
        _control_data_dx = [approximator(x)[1][0] for x in self.coords]
        _control_data_dxx = [approximator(x)[2][0] for x in self.coords]
        error = np.sum((_control_data - self.data) ** 2)
        error_dx = np.sum((_control_data_dx - self.Ddata_Dx) ** 2)
        error_dxx = np.sum((_control_data_dxx - self.Ddata_Dxx) ** 2)
        Ddata_Dx_approx = [d * self.ratio_dx for d in Ddata_Dx_approx]
        Ddata_Dxx_approx = [d * self.ratio_dxx for d in Ddata_Dxx_approx]
        return (
            data_approx,
            Ddata_Dx_approx,
            Ddata_Dxx_approx,
            error,
            error_dx,
            error_dxx,
        )

    def _test_mls_1d(self, w: Callable, *args, **kwargs):
        approx = moving_least_squares(*args, w=w, **kwargs)
        self._approximate(approx)

    def test_mls_1d_constant_nosupport(self):
        w = ConstantWeightFunction(value=1, dim=1)
        self._test_mls_1d(w, self.coords, self.data, deg=1, order=2)
        self._test_mls_1d(w, self.coords, self.data, deg=2, order=2)
    
    def test_mls_1d_constant_supportdomain(self):
        w = ConstantWeightFunction(value=1, dim=1, supportdomain=[0.2])
        self._test_mls_1d(w, self.coords, self.data, deg=1, order=2)
        self._test_mls_1d(w, self.coords, self.data, deg=2, order=2)
        w = ConstantWeightFunction(value=1, dim=1, supportdomain=[0.1])
        self._test_mls_1d(w, self.coords, self.data, deg=1, order=2)
        self._test_mls_1d(w, self.coords, self.data, deg=2, order=2)
        
    def test_mls_1d_cubic_nosupport(self):
        w = CubicWeightFunction(core=[0.0], supportdomain=[1000])
        self._test_mls_1d(w, self.coords, self.data, deg=1, order=2)
        self._test_mls_1d(w, self.coords, self.data, deg=2, order=2)
    
    def test_mls_1d_cubic_supportdomain(self):
        w = CubicWeightFunction(core=[0.0], supportdomain=[0.2])
        self._test_mls_1d(w, self.coords, self.data, deg=1, order=2)
        self._test_mls_1d(w, self.coords, self.data, deg=2, order=2)
        w = CubicWeightFunction(core=[0.0], supportdomain=[0.1])
        self._test_mls_1d(w, self.coords, self.data, deg=1, order=2)
        self._test_mls_1d(w, self.coords, self.data, deg=2, order=2)
        
    def test_mls_1d_singular_nosupport(self):
        w = SingularWeightFunction(core=[0.0], supportdomain=[1000])
        self._test_mls_1d(w, self.coords, self.data, deg=1, order=2)
        self._test_mls_1d(w, self.coords, self.data, deg=2, order=2)
    
    def test_mls_1d_singular_supportdomain(self):
        w = SingularWeightFunction(core=[0.0], supportdomain=[0.2])
        self._test_mls_1d(w, self.coords, self.data, deg=1, order=2)
        self._test_mls_1d(w, self.coords, self.data, deg=2, order=2)
        w = SingularWeightFunction(core=[0.0], supportdomain=[0.1])
        self._test_mls_1d(w, self.coords, self.data, deg=1, order=2)
        self._test_mls_1d(w, self.coords, self.data, deg=2, order=2)

if __name__ == "__main__":
    unittest.main()
