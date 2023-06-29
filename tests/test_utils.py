import unittest
import doctest

import numpy as np
import awkward as ak

import sigmaepsilon.math
import sigmaepsilon.math.arraysetops as arraysetops
from sigmaepsilon.math.linalg.sparse import JaggedArray
from sigmaepsilon.math import squeeze
from sigmaepsilon.math import utils as nu
from sigmaepsilon.math.hist import histogram


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(sigmaepsilon.math.decorate))
    tests.addTests(doctest.DocTestSuite(arraysetops))
    tests.addTests(doctest.DocTestSuite(sigmaepsilon.math.utils))
    return tests


class TestUtils(unittest.TestCase):
    def test_arraysetops(self):
        arr = np.array([[1, 2, 3], [1, 2, 4]], dtype=int)
        arraysetops.unique2d(arr)
        arraysetops.unique2d(arr, return_index=True)
        arraysetops.unique2d(arr, return_inverse=True)
        arraysetops.unique2d(arr, return_counts=True)

        arr = ak.Array([[1, 2], [1, 2, 3]])
        arraysetops.unique2d(arr)
        arraysetops.unique2d(arr, return_index=True)
        arraysetops.unique2d(arr, return_inverse=True)
        arraysetops.unique2d(arr, return_counts=True)

        data = np.array([1, 2, 1, 2, 3])
        arr = JaggedArray(data, cuts=[2, 3])
        arraysetops.unique2d(arr)
        arraysetops.unique2d(arr, return_index=True)
        arraysetops.unique2d(arr, return_inverse=True)
        arraysetops.unique2d(arr, return_counts=True)

        data = np.array([1, 2, 1, 2, 3, 5])
        arr = JaggedArray(data, cuts=[3, 3], force_numpy=True)
        arraysetops.unique2d(arr)

        failed_properly = False
        try:
            arraysetops.unique2d("aaa")
        except TypeError:
            failed_properly = True
        finally:
            self.assertTrue(failed_properly)

    def test_decorators(self):
        @squeeze(default=True)
        def foo1(arr) -> np.ndarray:
            return arr

        self.assertTrue(len(foo1(np.array([[1, 2]])).shape), 1)
        self.assertTrue(len(foo1(np.array([[1, 2]]), squeeze=False).shape), 2)

        @squeeze(default=True)
        def foo2(arr) -> np.ndarray:
            return arr, arr

        arr = np.array([[1, 2]])
        foo2(arr)
        foo2(arr, squeeze=False)

        @squeeze(default=True)
        def foo3(arr) -> np.ndarray:
            return {"1": arr, "2": arr}

        foo3(arr)
        foo3(arr, squeeze=False)

    def test_utils(self):
        self.assertEqual(nu.itype_of_ftype(np.float32), np.int32)
        self.assertEqual(nu.itype_of_ftype(np.float64), np.int64)

        nu.atleastnd(1, 2, front=True)
        nu.atleastnd(1, 2, back=True)
        nu.atleast1d(1)
        nu.atleast2d(1)
        nu.atleast3d(1)
        nu.atleast4d(1)
        nu.matrixform(1)
        nu.flatten2d(np.eye(3), order="C")
        nu.flatten2d(np.eye(3), order="F")
        nu.bool_to_float([True, False])
        nu.choice([False, True], (2, 2), [0.2, 0.8])
        nu.choice([False, True], (2, 2))
        nu.indices_of_equal_rows(np.eye(3), np.eye(3))
        nu.indices_of_equal_rows(np.eye(3, dtype=int), np.eye(3, dtype=int))
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        nu.indices_of_equal_rows(arr, arr)
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        nu.indices_of_equal_rows(arr, arr)
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        nu.indices_of_equal_rows(x, y)
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        nu.indices_of_equal_rows(x, y)
        nu.to_range_1d([0.3, 0.5], source=[0, 1], target=[-1, 1])

        histogram([1, 2, 1], bins=[0, 1, 2, 3])
        histogram([1, 2, 1], bins=[0, 1, 2, 3], return_edges=True)


if __name__ == "__main__":
    unittest.main()
