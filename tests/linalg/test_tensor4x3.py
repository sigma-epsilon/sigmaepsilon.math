import unittest

import numpy as np

from sigmaepsilon.math.linalg import ReferenceFrame, Tensor4x3
from sigmaepsilon.math.linalg.exceptions import TensorShapeMismatchError
from sigmaepsilon.math.linalg.testing import LinalgTestCase
from sigmaepsilon.math import repeat


class TestTensor4x3Create(LinalgTestCase):
    def test_create_failing_property(self):
        frame = ReferenceFrame(dim=3)

        array = np.ones((3, 3, 3, 3), dtype=float)
        self.assertFailsProperly(ValueError, Tensor4x3, array, frame=frame, bulk=True)

        array = np.ones((3,), dtype=float)
        self.assertFailsProperly(
            TensorShapeMismatchError, Tensor4x3, array, frame=frame, bulk=True
        )

        array = np.ones((4, 4, 4, 4), dtype=float)
        self.assertFailsProperly(
            TensorShapeMismatchError, Tensor4x3, array, frame=frame, bulk=True
        )

        array = np.ones((3, 4), dtype=float)
        self.assertFailsProperly(
            TensorShapeMismatchError, Tensor4x3, array, frame=frame, bulk=True
        )

        array = np.ones((9, 9), dtype=float)
        self.assertFailsProperly(
            TensorShapeMismatchError, Tensor4x3, array, frame=frame, bulk=True
        )

        array = np.ones((2, 3, 3, 3, 3), dtype=float)
        self.assertFailsProperly(ValueError, Tensor4x3, array, frame=frame, bulk=False)

        array = np.ones((2, 9, 9), dtype=float)
        self.assertFailsProperly(
            TensorShapeMismatchError, Tensor4x3, array, frame=frame, bulk=False
        )


class TestTensor4x3(LinalgTestCase):
    def setUp(self) -> None:
        self.frame = ReferenceFrame(dim=3)

        frame = ReferenceFrame(dim=3)
        array = np.ones((3, 3, 3, 3), dtype=float)
        self.tensor_single = Tensor4x3(array, frame=frame)

        frame = ReferenceFrame(dim=3)
        array = np.ones((2, 3, 3, 3, 3), dtype=float)
        self.tensor_bulk = Tensor4x3(array, frame=frame)

        axes = repeat(np.eye(3), 2)
        frame = ReferenceFrame(axes)
        array = np.ones((2, 3, 3, 3, 3), dtype=float)
        self.tensor_bulk = Tensor4x3(array, frame=frame)

        self.tensors = [
            self.tensor_single,
            self.tensor_bulk,
        ]

        return super().setUp()

    def test_show(self):
        target = self.frame.rotate("Space", [0, 0, np.pi / 2], "123", inplace=False)
        for t in self.tensors:
            t.show(target)

    def test_transform(self):
        axes = np.eye(3, 3)
        target = ReferenceFrame(axes)
        for t in self.tensors:
            t.show(target)

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
