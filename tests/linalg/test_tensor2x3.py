import unittest

import numpy as np

from sigmaepsilon.math.linalg import ReferenceFrame, Tensor2x3
from sigmaepsilon.math.linalg.exceptions import TensorShapeMismatchError
from sigmaepsilon.math.linalg.testing import LinalgTestCase
from sigmaepsilon.math import repeat


class TestTensor2x3Create(LinalgTestCase):
    def test_create_failing_property(self):
        frame = ReferenceFrame(dim=3)

        array = np.ones((3, 3), dtype=float)
        self.assertFailsProperly(ValueError, Tensor2x3, array, frame=frame, bulk=True)

        array = np.ones((3,), dtype=float)
        self.assertFailsProperly(
            TensorShapeMismatchError, Tensor2x3, array, frame=frame, bulk=True
        )

        array = np.ones((4,), dtype=float)
        self.assertFailsProperly(
            TensorShapeMismatchError, Tensor2x3, array, frame=frame, bulk=True
        )

        array = np.ones((3, 4), dtype=float)
        self.assertFailsProperly(
            TensorShapeMismatchError, Tensor2x3, array, frame=frame, bulk=True
        )

        array = np.ones((9), dtype=float)
        self.assertFailsProperly(
            TensorShapeMismatchError, Tensor2x3, array, frame=frame, bulk=True
        )

        array = np.ones((2, 3, 3), dtype=float)
        self.assertFailsProperly(ValueError, Tensor2x3, array, frame=frame, bulk=False)

        array = np.ones((2, 9), dtype=float)
        self.assertFailsProperly(
            TensorShapeMismatchError, Tensor2x3, array, frame=frame, bulk=False
        )


class TestTensor2x3(LinalgTestCase):
    def setUp(self) -> None:
        self.frame = ReferenceFrame(dim=3)

        array = np.ones((3, 3), dtype=float)
        frame = ReferenceFrame(dim=3)
        self.tensor_single = Tensor2x3(array, frame=frame)

        array = np.ones((2, 3, 3), dtype=float)
        frame = ReferenceFrame(dim=3)
        self.tensor_bulk = Tensor2x3(array, frame=frame)

        axes = repeat(np.eye(3), 2)
        frame = ReferenceFrame(axes)
        array = np.ones((2, 3, 3), dtype=float)
        self.tensor_bulk_multi = Tensor2x3(array, frame=frame)

        axes = repeat(np.eye(3), 2)
        frame = ReferenceFrame(axes)
        array = np.ones((2, 1, 3, 3), dtype=float)
        self.tensor_bulk_multi2 = Tensor2x3(array, frame=frame)

        axes = repeat(np.eye(3), 2)
        frame = ReferenceFrame(axes)
        array = np.ones((2, 1, 1, 3, 3), dtype=float)
        self.tensor_bulk_multi3 = Tensor2x3(array, frame=frame)

        self.tensors = [
            self.tensor_single,
            self.tensor_bulk,
            self.tensor_bulk_multi,
            self.tensor_bulk_multi2,
            self.tensor_bulk_multi3,
        ]

        return super().setUp()

    def test_orient(self):
        for t in self.tensors:
            t.orient("Space", [0, 0, np.pi / 2], "123")

    def test_orient_new(self):
        for t in self.tensors:
            t.orient_new("Space", [0, 0, np.pi / 2], "123")

    def test_show(self):
        target = self.frame.rotate("Space", [0, 0, np.pi / 2], "123", inplace=False)
        for t in self.tensors:
            t.show(target)

    def test_transform_fail(self):
        axes = repeat(np.eye(3, 3), 2)
        target = ReferenceFrame(axes)
        self.assertFailsProperly(NotImplementedError, self.tensor_single.show, target)
        self.tensor_bulk.show(target)

    def test_dual(self):
        for t in self.tensors:
            self.assertTensorDualityRules(t)

    def test_transposition_rules(self):
        for t in self.tensors:
            self.assertTensorTranspositionRules(t)


if __name__ == "__main__":
    unittest.main()
