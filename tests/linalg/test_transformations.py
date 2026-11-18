"""Regression tests for the vector and tensor algebra of `sigmaepsilon.math.linalg`.

Every result is checked against an independent reference: plain NumPy (mostly
`numpy.einsum` on the ambient components), SymPy for chained rotations, and a
partitioned solve for linear systems with prescribed unknowns. The frames used
are oblique, rotated and left-handed, because the identity frame hides most
errors.

References
----------
- The convention of the cross product of a second-order tensor and a vector,
  :math:`(\\mathbf{a} \\otimes \\mathbf{c}) \\times \\mathbf{b} = \\mathbf{a} \\otimes
  (\\mathbf{c} \\times \\mathbf{b})` and :math:`\\mathbf{b} \\times (\\mathbf{a}
  \\otimes \\mathbf{c}) = (\\mathbf{b} \\times \\mathbf{a}) \\otimes \\mathbf{c}`,
  is the one of E. B. Wilson, *Vector Analysis*, founded upon the lectures of
  J. W. Gibbs, Charles Scribner's Sons, New York, 1901, §104.
- Chained rotations are compared to
  :meth:`sympy.physics.vector.ReferenceFrame.orientnew`.
"""

import unittest
import warnings

import numpy as np
from numpy.linalg import LinAlgError
from sympy.physics.vector import ReferenceFrame as SymPyFrame

from sigmaepsilon.math.linalg import (
    ReferenceFrame,
    CartesianFrame,
    Vector,
    Tensor,
    Tensor2,
    Tensor4,
    Tensor4x3,
    dot,
    cross,
    rotation_matrix,
    permutation_tensor,
    show_vector,
    show_frame,
    csr_matrix,
    has_equal_dimensions,
    is_hermitian,
)
from sigmaepsilon.math.linalg.exceptions import LinalgOperationInputError
from sigmaepsilon.math.linalg.solve import linsolve, reduce
from sigmaepsilon.math.linalg.tr import _tr_tensors2, _tr_tensors4x3

EPS = permutation_tensor(3)
ANGLES = [0.3, -0.7, 1.1]


def oblique_frame(rng) -> ReferenceFrame:
    return ReferenceFrame(rng.random((3, 3)) + np.eye(3))


def left_handed_frame() -> ReferenceFrame:
    return ReferenceFrame(np.diag([1.0, 1.0, -1.0]))


def sympy_dcm(*rotations) -> np.ndarray:
    """Chain rotations with SymPy, each one about the axes of the previous frame."""
    frame = SymPyFrame("N")
    N = frame
    for i, (rot_type, amounts, rot_order) in enumerate(rotations):
        frame = frame.orientnew(f"F{i}", rot_type, amounts, rot_order)
    return np.array(frame.dcm(N).evalf()).astype(float)


class TestDot(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_vector_tensor_cartesian(self):
        I3 = ReferenceFrame(np.eye(3))
        T, b = self.rng.random((3, 3)), self.rng.random(3)
        res = dot(Vector(b, frame=I3), Tensor2(T, frame=I3))
        self.assertTrue(np.allclose(res.show(), b @ T))
        self.assertFalse(np.allclose(res.show(), T @ b))

    def test_vector_tensor_oblique(self):
        for frame in (oblique_frame(self.rng), left_handed_frame()):
            a = Vector(self.rng.random(3), frame=frame)
            T = Tensor2(self.rng.random((3, 3)), frame=frame)
            res = dot(a, T)
            self.assertIs(res.frame, frame)
            self.assertTrue(np.allclose(res.show(), a.show() @ T.show()))

    def test_tensor_vector_oblique(self):
        frame = oblique_frame(self.rng)
        a = Vector(self.rng.random(3), frame=frame)
        T = Tensor2(self.rng.random((3, 3)), frame=frame)
        self.assertTrue(np.allclose(dot(T, a).show(), T.show() @ a.show()))

    def test_bulk_vectors(self):
        frame = oblique_frame(self.rng)
        u = Vector(self.rng.random((5, 3)), frame=frame, bulk=True)
        v = Vector(self.rng.random((5, 3)), frame=frame, bulk=True)
        expected = np.einsum("ni,ni->n", u.show(), v.show())
        self.assertTrue(np.allclose(dot(u, v), expected))


class TestCross(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_vector_vector(self):
        for frame in (oblique_frame(self.rng), left_handed_frame()):
            a = Vector(self.rng.random(3), frame=frame)
            b = Vector(self.rng.random(3), frame=frame)
            res = cross(a, b)
            self.assertIs(res.frame, frame)
            self.assertTrue(np.allclose(res.show(), np.cross(a.show(), b.show())))

    def test_vector_vector_bulk(self):
        frame = oblique_frame(self.rng)
        a = Vector(self.rng.random((5, 3)), frame=frame, bulk=True)
        b = Vector(self.rng.random((5, 3)), frame=frame, bulk=True)
        res = cross(a, b)
        self.assertTrue(np.allclose(res.show(), np.cross(a.show(), b.show())))

    def test_tensor_vector(self):
        # (A x b)_ij = e_jkl A_ik b_l
        for frame in (ReferenceFrame(np.eye(3)), oblique_frame(self.rng)):
            A = Tensor2(self.rng.random((3, 3)), frame=frame)
            b = Vector(self.rng.random(3), frame=frame)
            expected = np.einsum("jkl,ik,l->ij", EPS, A.show(), b.show())
            self.assertTrue(np.allclose(cross(A, b).show(), expected))

    def test_vector_tensor(self):
        # (b x A)_ij = e_ikl b_k A_lj
        for frame in (ReferenceFrame(np.eye(3)), oblique_frame(self.rng)):
            A = Tensor2(self.rng.random((3, 3)), frame=frame)
            b = Vector(self.rng.random(3), frame=frame)
            expected = np.einsum("ikl,k,lj->ij", EPS, b.show(), A.show())
            self.assertTrue(np.allclose(cross(b, A).show(), expected))

    def test_dyad_definition(self):
        # (a c) x b = a (c x b) and b x (a c) = (b x a) c
        I3 = ReferenceFrame(np.eye(3))
        a, b, c = self.rng.random((3, 3))
        A = Tensor2(np.outer(a, c), frame=I3)
        B = Vector(b, frame=I3)
        self.assertTrue(np.allclose(cross(A, B).show(), np.outer(a, np.cross(c, b))))
        self.assertTrue(np.allclose(cross(B, A).show(), np.outer(np.cross(b, a), c)))

    def test_identity_example(self):
        I3 = ReferenceFrame(np.eye(3))
        res = cross(Tensor2(np.eye(3), frame=I3), Vector([0.0, 1.0, 0.0], frame=I3))
        self.assertEqual(res.show()[0, 2], 1.0)
        self.assertEqual(res.show()[2, 0], -1.0)


class TestFrame(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)

    def test_show_target(self):
        A, B = oblique_frame(self.rng), oblique_frame(self.rng)
        C = A.show(B)
        # the rows of C are the base vectors of A in B: G_A = C @ G_B
        self.assertTrue(np.allclose(C @ B.show(), A.show()))
        self.assertTrue(np.allclose(A.show(), A.show(None)))

    def test_volume_is_signed(self):
        G = self.rng.random((3, 3)) + np.eye(3)
        self.assertTrue(np.isclose(ReferenceFrame(G).volume(), np.linalg.det(G)))
        self.assertTrue(np.isclose(left_handed_frame().volume(), -1.0))
        L = CartesianFrame(np.diag([1.0, 1.0, -1.0]))
        self.assertTrue(np.isclose(L.volume(), -1.0))
        self.assertTrue(np.isclose(CartesianFrame(dim=3).volume(), 1.0))

    def test_chained_rotations_about_ambient_axes(self):
        # 90 degrees about z, then 90 degrees 'Body' about x
        frame = ReferenceFrame(dim=3)
        frame.orient("Body", [0, 0, np.pi / 2], "XYZ")
        frame.orient("Body", [np.pi / 2, 0, 0], "XYZ")
        expected = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
        self.assertTrue(np.allclose(frame.show(), expected))

    def test_chained_rotations_against_sympy(self):
        # Rotating about the ambient axes means G' = G @ R, while SymPy rotates
        # about the axes of the frame (G' = R @ G). Hence the same rotations
        # applied in reverse order with SymPy must give the same frame.
        r1 = ("Body", [0.2, 0.5, -0.4], "XYZ")
        r2 = ("Space", [-0.3, 0.1, 0.9], "ZXY")
        frame = ReferenceFrame(dim=3).orient_new(*r1).orient_new(*r2)
        self.assertTrue(np.allclose(frame.show(), sympy_dcm(r2, r1)))
        R1, R2 = rotation_matrix(*r1), rotation_matrix(*r2)
        self.assertTrue(np.allclose(frame.show(), R1 @ R2))

    def test_rotation_is_rigid_for_oblique_frames(self):
        frame = oblique_frame(self.rng)
        rotated = frame.orient_new("Body", ANGLES, "XYZ")
        self.assertTrue(np.allclose(rotated.Gram(), frame.Gram()))


class TestOrient(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.R = rotation_matrix("Body", ANGLES, "XYZ")

    def test_vector_rotates_about_ambient_axes(self):
        for frame in (oblique_frame(self.rng), left_handed_frame()):
            v = Vector(self.rng.random(3), frame=frame)
            x = v.show()
            res = v.orient_new("Body", ANGLES, "XYZ")
            self.assertIs(res.frame, frame)
            self.assertTrue(np.allclose(res.show(), self.R.T @ x))
            self.assertTrue(np.allclose(v.show(), x))  # the original is untouched
            v.orient("Body", ANGLES, "XYZ")
            self.assertTrue(np.allclose(v.show(), self.R.T @ x))

    def test_vector_with_dcm(self):
        frame = oblique_frame(self.rng)
        v = Vector(self.rng.random(3), frame=frame)
        w = v.orient_new("Body", ANGLES, "XYZ")
        v.orient(dcm=self.R)
        self.assertTrue(np.allclose(v.show(), w.show()))

    def test_vector_and_frame_rotated_together(self):
        frame = oblique_frame(self.rng)
        arr = self.rng.random(3)
        v = Vector(arr, frame=frame).orient_new("Body", ANGLES, "XYZ")
        w = Vector(arr, frame=frame.orient_new("Body", ANGLES, "XYZ"))
        self.assertTrue(np.allclose(v.show(), w.show()))

    def test_bulk_vectors(self):
        frame = oblique_frame(self.rng)
        for N in (3, 5):
            v = Vector(self.rng.random((N, 3)), frame=frame, bulk=True)
            x = v.show()
            res = v.orient_new("Body", ANGLES, "XYZ")
            self.assertTrue(np.allclose(res.show(), x @ self.R))
            v.orient("Body", ANGLES, "XYZ")
            self.assertTrue(np.allclose(v.show(), x @ self.R))

    def test_chained_vector_rotations(self):
        frame = oblique_frame(self.rng)
        r1 = ("Body", [0.2, 0.5, -0.4], "XYZ")
        r2 = ("Space", [-0.3, 0.1, 0.9], "ZXY")
        v = Vector(self.rng.random((4, 3)), frame=frame, bulk=True)
        x = v.show()
        v.orient(*r1).orient(*r2)
        R1, R2 = rotation_matrix(*r1), rotation_matrix(*r2)
        self.assertTrue(np.allclose(v.show(), x @ R1 @ R2))

    def test_tensors(self):
        frame = oblique_frame(self.rng)
        T2 = Tensor2(self.rng.random((3, 3)), frame=frame)
        X = T2.show()
        res = T2.orient_new("Body", ANGLES, "XYZ")
        self.assertTrue(np.allclose(res.show(), self.R.T @ X @ self.R))
        T2.orient("Body", ANGLES, "XYZ")
        self.assertTrue(np.allclose(T2.show(), self.R.T @ X @ self.R))

        T4 = Tensor4(self.rng.random((3, 3, 3, 3)), frame=frame)
        X = T4.show()
        Q = self.R.T
        expected = np.einsum("ip,jq,kr,ls,pqrs->ijkl", Q, Q, Q, Q, X)
        res = T4.orient_new("Body", ANGLES, "XYZ")
        self.assertTrue(np.allclose(res.show(), expected))

        T3 = Tensor(self.rng.random((3, 3, 3)), frame=frame)
        X = T3.show()
        expected = np.einsum("ip,jq,kr,pqr->ijk", Q, Q, Q, X)
        self.assertTrue(
            np.allclose(T3.orient_new("Body", ANGLES, "XYZ").show(), expected)
        )


class TestVector(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)

    def test_copy_keeps_frame(self):
        frame = ReferenceFrame(dim=3).orient_new("Body", ANGLES, "XYZ")
        for obj in (
            Vector(self.rng.random(3), frame=frame),
            Vector(self.rng.random((5, 3)), frame=frame, bulk=True),
            Tensor2(self.rng.random((3, 3)), frame=frame),
            Tensor4(self.rng.random((3, 3, 3, 3)), frame=frame),
        ):
            shallow, deep = obj.copy(), obj.deepcopy()
            self.assertIs(shallow.frame, frame)
            self.assertIsNot(deep.frame, frame)
            for c in (shallow, deep):
                self.assertIs(type(c), type(obj))
                self.assertTrue(np.allclose(c.show(), obj.show()))
                self.assertFalse(np.shares_memory(c.array, obj.array))

    def test_dual_bulk(self):
        frame = oblique_frame(self.rng)
        v = Vector(self.rng.random((5, 3)), frame=frame, bulk=True)
        d = v.dual()
        self.assertTrue(np.allclose(d.array, v.array @ frame.Gram()))
        self.assertTrue(np.allclose(d.show(), v.show()))


class TestTensor4(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(5)

    def test_any_dimension(self):
        frame = ReferenceFrame(self.rng.random((2, 2)) + np.eye(2))
        arr = self.rng.random((2, 2, 2, 2))
        T = Tensor4(arr, frame=frame)
        Q = frame.dcm(target=ReferenceFrame(dim=2))
        expected = np.einsum("ip,jq,kr,ls,pqrs->ijkl", Q, Q, Q, Q, arr)
        self.assertTrue(np.allclose(T.show(), expected))
        self.assertTrue(np.allclose(T.dual().show(), T.show()))

    def test_any_dimension_bulk(self):
        arr = self.rng.random((4, 5, 2, 2, 2, 2))
        Q = self.rng.random((4, 2, 2))
        T = Tensor4(arr, bulk=True)
        expected = np.einsum("nip,njq,nkr,nls,nxpqrs->nxijkl", Q, Q, Q, Q, arr)
        self.assertTrue(np.allclose(T.transform_components(Q), expected))

    def test_3x3x3x3_cases(self):
        Q = rotation_matrix("Body", ANGLES, "XYZ")
        Qs = self.rng.random((2, 3, 3))
        cases = [
            (self.rng.random((3, 3, 3, 3)), Q, "ip,jq,kr,ls,pqrs->ijkl"),
            (self.rng.random((2, 3, 3, 3, 3)), Q, "ip,jq,kr,ls,npqrs->nijkl"),
            (self.rng.random((2, 4, 3, 3, 3, 3)), Q, "ip,jq,kr,ls,nxpqrs->nxijkl"),
            (self.rng.random((2, 3, 3, 3, 3)), Qs, "nip,njq,nkr,nls,npqrs->nijkl"),
            (
                self.rng.random((2, 4, 3, 3, 3, 3)),
                Qs,
                "nip,njq,nkr,nls,nxpqrs->nxijkl",
            ),
        ]
        for arr, Q_, command in cases:
            expected = np.einsum(command, Q_, Q_, Q_, Q_, arr)
            self.assertTrue(np.allclose(_tr_tensors4x3(arr, Q_), expected))
            bulk = len(arr.shape) > 4
            T = Tensor4(arr, bulk=bulk) if bulk else Tensor4(arr)
            self.assertTrue(np.allclose(T.transform_components(Q_), expected))
            T = Tensor4x3(arr, bulk=bulk) if bulk else Tensor4x3(arr)
            self.assertTrue(np.allclose(T.transform_components(Q_), expected))


class TestShapeChecks(unittest.TestCase):
    """Mismatching DCMs raise instead of returning zeros."""

    def setUp(self):
        self.rng = np.random.default_rng(6)

    def test_tensors2(self):
        arr = self.rng.random((5, 3, 3))
        for Q in (np.eye(2), self.rng.random((5, 2, 2)), self.rng.random((4, 3, 3))):
            with self.assertRaises(ValueError):
                _tr_tensors2(arr, Q)

    def test_tensors4(self):
        arr = self.rng.random((5, 3, 3, 3, 3))
        for Q in (np.eye(2), self.rng.random((5, 2, 2)), self.rng.random((4, 3, 3))):
            with self.assertRaises(ValueError):
                _tr_tensors4x3(arr, Q)
            with self.assertRaises(ValueError):
                Tensor4(arr, bulk=True).transform_components(Q)

    def test_show_vector(self):
        arr = self.rng.random((5, 3))
        for Q in (np.eye(2), self.rng.random((5, 2, 2)), self.rng.random((4, 3, 3))):
            with self.assertRaises(LinalgOperationInputError):
                show_vector(Q, arr)

    def test_show_frame(self):
        arr = self.rng.random((5, 3, 3))
        for Q in (np.eye(2), self.rng.random((5, 2, 2)), self.rng.random((4, 3, 3))):
            with self.assertRaises(LinalgOperationInputError):
                show_frame(Q, arr)


class TestLinsolve(unittest.TestCase):
    METHODS = ("Jordan", "Gauss-Jordan")

    def setUp(self):
        self.rng = np.random.default_rng(7)

    def reference(self, A, B, pb, pv):
        """Solve the partitioned system and return the solution and the reactions."""
        free = ~pb
        X = np.zeros(B.shape)
        X[pb] = pv[pb, None]
        rhs = B[free] - A[np.ix_(free, pb)] @ X[pb]
        X[free] = np.linalg.solve(A[np.ix_(free, free)], rhs)
        R = A @ X - B
        R[free] = 0
        return X, R

    def test_prescribed_values(self):
        A = self.rng.random((8, 8)) + 8 * np.eye(8)
        B = self.rng.random((8, 3))
        for method in self.METHODS:
            for pattern in ([0], [7], [1, 4, 6]):
                pb = np.zeros(8, dtype=bool)
                pb[pattern] = True
                pv = self.rng.random(8) * pb
                X, R = linsolve(A, B, presc_bool=pb, presc_val=pv, method=method)
                X_ref, R_ref = self.reference(A, B, pb, pv)
                self.assertTrue(np.allclose(X, X_ref))
                self.assertTrue(np.allclose(R, R_ref))

    def test_pivoting(self):
        P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        b = np.array([1.0, 2.0, 3.0])
        A = self.rng.random((6, 6))
        A[[2, 3], [2, 3]] = 0
        pb = np.zeros(6, dtype=bool)
        pb[4] = True
        pv = np.zeros(6)
        pv[4] = 0.5
        B = np.ones((6, 1))
        for method in self.METHODS:
            self.assertTrue(np.allclose(linsolve(P, b, method=method), [2, 1, 3]))
            X, R = linsolve(A, B, presc_bool=pb, presc_val=pv, method=method)
            X_ref, R_ref = self.reference(A, B, pb, pv)
            self.assertTrue(np.allclose(X, X_ref))
            self.assertTrue(np.allclose(R, R_ref))

    def test_relative_tolerance(self):
        A = (self.rng.random((4, 4)) + 4 * np.eye(4)) * 1e-13
        b = self.rng.random(4)
        for method in self.METHODS:
            X = linsolve(A, b, method=method)
            self.assertTrue(np.allclose(X, np.linalg.solve(A, b)))

    def test_output_shape(self):
        A = self.rng.random((4, 4)) + 4 * np.eye(4)
        b = self.rng.random(4)
        pb = np.array([False, True, False, False])
        pv = np.array([0, 1.0, 0, 0])
        for method in self.METHODS:
            self.assertEqual(linsolve(A, b, method=method).shape, (4,))
            X, R = linsolve(A, b, presc_bool=pb, presc_val=pv, method=method)
            self.assertEqual(X.shape, (4,))
            self.assertEqual(R.shape, (4,))
            self.assertEqual(linsolve(A, b[:, None], method=method).shape, (4, 1))
            A_, b_ = reduce(A, b, method=method)
            self.assertEqual(b_.shape, (4,))

    def test_invalid_input(self):
        A = self.rng.random((8, 8)) + 8 * np.eye(8)
        pb = np.zeros(8, dtype=bool)
        for method in self.METHODS:
            bad_inputs = [
                dict(B=self.rng.random(3)),
                dict(B=self.rng.random(8), presc_bool=pb),
                dict(B=self.rng.random(8), presc_bool=pb[:3], presc_val=np.zeros(3)),
                dict(
                    B=self.rng.random(8), presc_bool=np.ones(8), presc_val=np.zeros(8)
                ),
            ]
            for kwargs in bad_inputs:
                with self.assertRaises(ValueError):
                    linsolve(A, method=method, **kwargs)
            with self.assertRaises(LinAlgError):
                linsolve(np.ones((3, 3)), np.ones(3), method=method)


class TestCSR(unittest.TestCase):
    def test_shape_from_column_indices(self):
        c = csr_matrix(np.array([1.0, 2.0]), np.array([4, 0]), np.array([0, 1, 2]))
        self.assertEqual(c.shape, (2, 5))
        expected = np.array([[0, 0, 0, 0, 1], [2, 0, 0, 0, 0]], dtype=float)
        self.assertTrue(np.allclose(c.to_numpy(), expected))
        empty = csr_matrix(np.array([]), np.array([], dtype=int), np.array([0, 0]))
        self.assertEqual(empty.shape, (1, 0))


class TestLogical(unittest.TestCase):
    def test_is_hermitian_is_deprecated(self):
        arr = np.ones((3, 3, 3))
        self.assertTrue(has_equal_dimensions(arr))
        self.assertFalse(has_equal_dimensions(np.ones((3, 2))))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertTrue(is_hermitian(arr))
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))


if __name__ == "__main__":
    unittest.main()
