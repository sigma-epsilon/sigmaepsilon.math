import unittest
import doctest

import sigmaepsilon.math.logical as logical
from sigmaepsilon.math.logical import (
    isclose,
    allclose,
    isintarray,
    is1dintarray,
    isfloatarray,
    is1dfloatarray,
    isboolarray,
    issymmetric,
    is_none_or_false,
)
from sigmaepsilon.math.linalg.utils import random_posdef_matrix
import numpy as np


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(logical))
    return tests


class TestLogical(unittest.TestCase):
    def test_istype(self):
        # 0d
        self.assertTrue(is_none_or_false(False))
        self.assertTrue(is_none_or_false(None))
        self.assertFalse(is_none_or_false(True))
        self.assertFalse(is_none_or_false("a"))
        # 1d
        arr = np.zeros(2, dtype=int)
        self.assertTrue(isintarray(arr))
        self.assertTrue(is1dintarray(arr))
        self.assertTrue(not isfloatarray(arr))
        self.assertTrue(not isboolarray(arr))
        self.assertTrue(not is1dfloatarray(arr))
        arr = arr.astype(float)
        self.assertTrue(not isintarray(arr))
        self.assertTrue(not is1dintarray(arr))
        self.assertTrue(isfloatarray(arr))
        self.assertTrue(is1dfloatarray(arr))
        self.assertTrue(not isboolarray(arr))
        arr = arr.astype(bool)
        self.assertTrue(not isintarray(arr))
        self.assertTrue(not is1dintarray(arr))
        self.assertTrue(not isfloatarray(arr))
        self.assertTrue(not is1dfloatarray(arr))
        self.assertTrue(isboolarray(arr))
        # 2d
        arr = np.zeros((2, 2), dtype=int)
        self.assertTrue(isintarray(arr))
        self.assertTrue(not is1dintarray(arr))
        self.assertTrue(not isfloatarray(arr))
        self.assertTrue(not is1dfloatarray(arr))
        self.assertTrue(not isboolarray(arr))
        arr = arr.astype(float)
        self.assertTrue(not isintarray(arr))
        self.assertTrue(not is1dintarray(arr))
        self.assertTrue(isfloatarray(arr))
        self.assertTrue(not is1dfloatarray(arr))
        self.assertTrue(not isboolarray(arr))
        arr = arr.astype(bool)
        self.assertTrue(not isintarray(arr))
        self.assertTrue(not is1dintarray(arr))
        self.assertTrue(not isfloatarray(arr))
        self.assertTrue(not is1dfloatarray(arr))
        self.assertTrue(isboolarray(arr))

    def test_issymmetric(self):
        self.assertTrue(issymmetric(random_posdef_matrix(2)))
        a = np.zeros((2, 2))
        a[0, 0] = 1.0
        self.assertTrue(issymmetric(a))
        a[0, 1] = 1.0
        self.assertFalse(issymmetric(a))
        a[1, 0] = 1.0
        self.assertTrue(issymmetric(a))

    def test_isclose(self):
        def assertAssert(fnc, *args, **kwargs):
            try:
                fnc(*args, **kwargs)
                self.assertTrue(False)
            except AssertionError as e:
                pass

        # 0d
        self.assertTrue(isclose(0, 0))
        self.assertTrue(not isclose(0, 1))
        self.assertTrue(not isclose(1, 0))
        # 1d
        x1 = np.zeros(10, dtype=float)
        x2 = x1 + 0.01
        self.assertTrue(allclose(x1, x1))
        self.assertTrue(not allclose(x1, x2))
        self.assertTrue(not allclose(x2, x1))
        # 2d
        x1 = np.zeros((2, 2), dtype=float)
        x2 = x1 + 0.01
        self.assertTrue(allclose(x1, x1))
        self.assertTrue(not allclose(x1, x2))
        self.assertTrue(not allclose(x2, x1))
        # purposeful assertion errors
        assertAssert(isclose, 0, 0, atol=None, rtol=None)
        assertAssert(isclose, 0, 0, atol=-1)
        assertAssert(isclose, 0, 0, rtol=-1)


if __name__ == "__main__":
    unittest.main()
