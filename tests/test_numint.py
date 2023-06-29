import unittest
import doctest

from sigmaepsilon.math import numint
from sigmaepsilon.math.numint import gauss_points


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(numint))
    return tests


class TestGaussNumInt(unittest.TestCase):
    def test_gauss_numint(self):
        gauss_points(2)
        gauss_points(2, 2)
        gauss_points(2, 2, 2)

        failed_properly = False
        try:
            gauss_points(2, 2, 2, 2)
        except NotImplementedError:
            failed_properly = True
        finally:
            self.assertTrue(failed_properly)


if __name__ == "__main__":
    unittest.main()
