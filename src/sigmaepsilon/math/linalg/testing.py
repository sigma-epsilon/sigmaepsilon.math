from copy import deepcopy

import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase

from .meta import TensorLike


class LinalgTestCase(SigmaEpsilonTestCase):
    def assertTensorDualityRules(self, t: TensorLike) -> None:
        # test if the dual of the dual is self
        self.assertTrue(np.allclose(t.dual().dual().show(), t.show()))

    def assertTensorTranspositionRules(self, t: TensorLike) -> None:
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
        # test if A - A.T == 0
        self.assertTrue(np.allclose(np.zeros_like(t.array), t.show() - t.T.show()))
