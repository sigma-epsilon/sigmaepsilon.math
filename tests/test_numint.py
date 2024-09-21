import unittest

from sigmaepsilon.math.numint import gauss_points


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
