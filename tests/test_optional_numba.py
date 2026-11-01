import importlib
import importlib.util
import subprocess
import sys
import textwrap
import unittest
from contextlib import contextmanager
from types import ModuleType
from unittest import mock

import numpy as np

import sigmaepsilon.core.jit as corejit
from sigmaepsilon.core.jit import NUMBA_AVAILABLE
from sigmaepsilon.math.linalg import utils as lautils
from sigmaepsilon.math.linalg.utils import random_posdef_matrix


@contextmanager
def numba_hidden():
    """Hide Numba, as if it was not installed."""
    try:
        with mock.patch.dict(sys.modules, {"numba": None}):
            importlib.reload(corejit)
            yield
    finally:
        importlib.reload(corejit)


def load_without_numba(name: str) -> ModuleType:
    """Execute the source of a module with Numba hidden.

    The module is created as a separate module object, leaving the already
    imported one untouched.
    """
    with numba_hidden():
        spec = importlib.util.spec_from_file_location(
            name + "_nonumba", importlib.util.find_spec(name).origin
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class TestNbTypesFallback(unittest.TestCase):
    def test_fallback(self):
        nbtypes = load_without_numba("sigmaepsilon.math._nbtypes")
        self.assertIsNone(nbtypes.int64)
        self.assertIsNone(nbtypes.int64_1d)
        d = nbtypes.Dict.empty(key_type=nbtypes.int64, value_type=nbtypes.int64_1d)
        self.assertEqual(d, {})
        d[1] = np.array([1, 2])
        self.assertIsInstance(d, dict)


class TestGufuncFallbacks(unittest.TestCase):
    """The pure NumPy versions of the gufuncs in `linalg.utils`."""

    @classmethod
    def setUpClass(cls):
        cls.fb = load_without_numba("sigmaepsilon.math.linalg.utils")
        np.random.seed(0)
        cls.A2 = np.stack([random_posdef_matrix(2) for _ in range(4)])
        cls.A3 = np.stack([random_posdef_matrix(3) for _ in range(4)])

    def test_is_fallback(self):
        self.assertFalse(self.fb.NUMBA_AVAILABLE)

    def test_det(self):
        for A, det in ((self.A2, self.fb.det2x2), (self.A3, self.fb.det3x3)):
            self.assertTrue(np.allclose(det(A), np.linalg.det(A)))
            self.assertTrue(np.isclose(det(A[0]), np.linalg.det(A[0])))

    def test_inv(self):
        for A, inv in ((self.A2, self.fb.inv2x2u), (self.A3, self.fb.inv3x3u)):
            self.assertTrue(np.allclose(inv(A), np.linalg.inv(A)))
            out = np.zeros_like(A[0])
            res = inv(A[0], out)
            self.assertIs(res, out)
            self.assertTrue(np.allclose(out, np.linalg.inv(A[0])))

    def test_adj(self):
        A = self.A3
        expected = np.linalg.inv(A) * np.linalg.det(A)[:, None, None]
        self.assertTrue(np.allclose(self.fb.adj3x3(A), expected))

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba is not installed")
    def test_same_as_numba(self):
        # det2x2 and det3x3 are left out, since their Numba versions don't
        # write their output (they rebind `res` instead of assigning to it)
        for name, A in (
            ("inv2x2u", self.A2),
            ("inv3x3u", self.A3),
            ("adj3x3", self.A3),
        ):
            expected = getattr(lautils, name)(A)
            self.assertTrue(np.allclose(getattr(self.fb, name)(A), expected), name)


class TestWithoutNumba(unittest.TestCase):
    """Runs the library in a separate process where Numba can't be imported."""

    def test_import_and_run(self):
        script = textwrap.dedent(
            """
            import sys
            sys.modules["numba"] = None

            import numpy as np
            from scipy.sparse import csr_matrix as csr_scipy
            import sigmaepsilon.math
            from sigmaepsilon.core.jit import NUMBA_AVAILABLE
            from sigmaepsilon.math import linalg, graph, approx, optimize
            from sigmaepsilon.math.arraysetops import unique2d
            from sigmaepsilon.math.graph.utils import rooted_level_structure
            from sigmaepsilon.math.linalg import csr_matrix

            assert not NUMBA_AVAILABLE

            unique, indices = unique2d(np.array([[1, 2], [2, 3]]), return_index=True)
            assert list(unique) == [1, 2, 3]
            assert np.array_equal(indices[2], [[0, 1], [1, 0]])

            adj = csr_matrix(csr_scipy(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])))
            rls = rooted_level_structure(adj, 0)
            assert [list(rls[i]) for i in range(len(rls))] == [[0], [1], [2]]

            def f(x):
                return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

            f.dimension = 2
            rga = optimize.RealValuedGeneticAlgorithm(
                f, [[-10, 10], [-10, 10]], nPop=100, minimize=True, maxage=20, seed=0
            )
            assert rga.solve().fitness < 1.0
            print("OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "OK")


if __name__ == "__main__":
    unittest.main()
