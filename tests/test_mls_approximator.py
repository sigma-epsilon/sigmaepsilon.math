import unittest
import numpy as np

from sigmaepsilon.math.approx import MLSApproximator
from sigmaepsilon.math import atleast2d


class TestMLSApproximator(unittest.TestCase):

    def setUp(self):
        nx, ny, nz = (10, 10, 10)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        z = np.linspace(0, 1, nz)
        xv, yv, zv = np.meshgrid(x, y, z)
        self.points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)
        self.values = np.random.rand(nx * ny * nz)

    def test_mls_3d(self):
        nx, ny, nz = (10, 10, 10)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        z = np.linspace(0, 1, nz)
        xv, yv, zv = np.meshgrid(x, y, z)
        points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)
        values = np.random.rand(nx * ny * nz)
        values = np.ones((len(values), 3))

        approximator = MLSApproximator(points, values)
        
        values_approx = approximator.approximate(points)
        self.assertIsInstance(values_approx, np.ndarray)
        self.assertEqual(len(values_approx), len(points))

    def test_constant_field_bulk(self):
        values = np.ones_like(self.values)
        values = np.ones((len(values), 3))  # bulkify
        points = self.points
        approximator = MLSApproximator(points, values)
        values_approx = approximator.approximate(points)
        self.assertTrue(np.allclose(np.ones_like(values_approx), values_approx))

    def test_constant_field_nd_bulk(self):
        values = np.ones_like(self.values)
        values = np.ones((len(values), 3, 3))  # bulkify
        points = self.points
        approximator = MLSApproximator(points, values)
        values_approx = approximator.approximate(points)
        self.assertTrue(np.allclose(np.ones_like(values_approx), values_approx))

    def test_constant_field_1d(self):
        values = np.ones_like(self.values)
        points = self.points
        approximator = MLSApproximator(points, values)
        values_approx = approximator.approximate(points)
        self.assertTrue(np.allclose(np.ones_like(values_approx), values_approx))


class TestMLSApproximatorConfig(unittest.TestCase):

    def setUp(self):
        self.points = np.zeros((10, 3))
        self.values = np.zeros((10,))

    def test_config__knn_backend(self):
        self.assertRaises(ValueError, MLSApproximator, self.points, self.values, knn_backend=1)

    def test_config__k(self):
        self.assertRaises(ValueError, MLSApproximator, self.points, self.values, k=0.5)

    def test_config__max_distance(self):
        self.assertRaises(ValueError, MLSApproximator, self.points, self.values, max_distance="5")


def test_mls_1d_sine_with_noise():
    number_of_data_points = 100
    number_of_sampling_points = 16

    coords = np.linspace(0, 1, number_of_data_points)
    random_noise = np.random.normal(0, 0.1, coords.shape)
    data = np.sin(2 * np.pi * coords) + random_noise
    data = atleast2d(data, back=True)
    coords = np.zeros((number_of_data_points, 3))
    coords[:, 0] = np.linspace(0, 1, number_of_data_points)

    coords_approx = np.zeros((number_of_sampling_points, 3))
    coords_approx[:, 0] = np.linspace(0, 1, number_of_sampling_points)
    approximator = MLSApproximator(coords, data)
    data_approx = approximator.approximate(coords_approx)
    assert isinstance(data_approx, np.ndarray)
    assert len(data_approx) == len(coords_approx)


if __name__ == "__main__":
    unittest.main()
