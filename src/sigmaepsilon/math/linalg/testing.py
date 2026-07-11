"""Test case helpers for asserting properties of tensor-like objects."""

from copy import deepcopy

import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase

from .meta import TensorLike


class LinalgTestCase(SigmaEpsilonTestCase):
    """A test case with assertions specific to tensorial objects."""

    def assertTensorDualityRules(self, t: TensorLike) -> None:
        """Assert that the dual of the dual of a tensor is itself."""
        # test if the dual of the dual is self
        self.assertTrue(np.allclose(t.dual().dual().show(), t.show()))

    def assertTensorTranspositionRules(self, t: TensorLike) -> None:
        """Assert that transposition of a tensor is self-consistent and linear."""
        # test if A.T.T == A
        self.assertTrue(np.allclose(t.show(), t.T.T.show()))

        # test if (A + B).T == A.T + B.T
        t_ = deepcopy(t)
        t_.array *= 2
        t__ = t + t_
        self.assertTrue(np.allclose(t__.T.show(), t.T.show() + t_.T.show()))

        # test if (A + B).T == A.T + B.T
        t__ = t - t_
        self.assertTrue(np.allclose(t__.T.show(), t.T.show() - t_.T.show()))

    def assertSymmetricTensor(self, t: TensorLike) -> None:
        """Assert that a tensor is symmetric, i.e. A - A.T == 0."""
        # test if A - A.T == 0
        self.assertTrue(np.allclose(np.zeros_like(t.array), t.show() - t.T.show()))
