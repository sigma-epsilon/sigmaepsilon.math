from typing import Union, Iterable
import numbers
import itertools

import numpy as np
from numpy import ndarray
from numba import njit, prange, guvectorize as guv
import sympy as sy
from sympy import symbols, Matrix
from sympy.physics.vector import ReferenceFrame as SymPyFrame

from sigmaepsilon.core.alphabet import latinrange

from .meta import TensorLike, ArrayWrapper, FrameLike
from .exceptions import LinalgOperationInputError, LinalgMissingInputError, LinalgError

__cache = True


__all__ = [
    "permutation_tensor",
    "dot",
    "cross",
    "normalize_frame",
    "Gram",
    "dual_frame",
    "random_pos_semidef_matrix",
    "random_posdef_matrix",
    "inv_sym_3x3",
    "vpath",
    "det3x3",
    "det2x2",
    "inv2x2",
    "inv2x2u",
    "adj3x3",
    "inv3x3u",
    "inv3x3",
    "inv3x3_bulk",
    "inv3x3_bulk2",
    "normalize",
    "normalize2d",
    "norm",
    "norm2d",
    "linspace",
    "linspace1d",
    "inv",
    "show_vector",
    "show_frame",
    "rotation_matrix",
    "generalized_left_inverse",
    "generalized_right_inverse",
    "generalized_inverse",
    "unit_basis_vector",
]


def rotation_matrix(
    rot_type: str, amounts: Iterable, rot_order: Union[str, int] = ""
) -> ndarray:
    """
    Returns a rotation matrix using the mechanism provided by
    `sympy.physics.vector.ReferenceFrame.orientnew`.

    Parameters
    ----------
    rot_type: str
        The method used to generate the direction cosine matrix. Supported
        methods are:

        - ``'Axis'``: simple rotations about a single common axis
        - ``'DCM'``: for setting the direction cosine matrix directly
        - ``'Body'``: three successive rotations about new intermediate
            axes, also called "Euler and Tait-Bryan angles"
        - ``'Space'``: three successive rotations about the parent
            frames' unit vectors
        - ``'Quaternion'``: rotations defined by four parameters which
            result in a singularity free direction cosine matrix

    amounts: Iterable
        Expressions defining the rotation angles or direction cosine
        matrix. These must match the ``rot_type``. See examples below for
        details. The input types are:

        - ``'Axis'``: 2-tuple (expr/sym/func, Vector)
        - ``'DCM'``: Matrix, shape(3, 3)
        - ``'Body'``: 3-tuple of expressions, symbols, or functions
        - ``'Space'``: 3-tuple of expressions, symbols, or functions
        - ``'Quaternion'``: 4-tuple of expressions, symbols, or
            functions

    rot_order: str or int, Optional
        If applicable, the order of the successive of rotations. The string
        ``'123'`` and integer ``123`` are equivalent, for example. Required
        for ``'Body'`` and ``'Space'``.

    Returns
    -------
    ReferenceFrame
        A new ReferenceFrame object.

    See Also
    --------
    :func:`sympy.physics.vector.ReferenceFrame.orientnew`

    Example
    -------
    Define a standard Cartesian frame and rotate it around axis 'Z'
    with 180 degrees:

    >>> from sigmaepsilon.math.linalg.utils import rotation_matrix
    >>> import numpy as np
    >>> R = rotation_matrix('Space', [0, 0, np.pi], 'XYZ')

    """
    source = SymPyFrame("S")
    target = source.orientnew("T", rot_type, amounts, rot_order)
    dcm = np.array(target.dcm(source).evalf()).astype(float)
    return dcm


def permutation_tensor(dim: int = 3) -> ndarray:
    """
    Returns the Levi-Civita pseudotensor for N dimensions as a NumPy array.

    Parameters
    ----------
    N : int, Optional
        The number of dimensions. Default is 3.
    """
    arr = np.zeros(tuple([dim for _ in range(dim)]))
    mat = np.zeros((dim, dim), dtype=np.int32)
    for x in itertools.permutations(tuple(range(dim))):
        mat[:, :] = 0
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x] = int(np.linalg.det(mat))
    return arr


def dot(
    a: Union[TensorLike, ArrayWrapper],
    b: Union[TensorLike, ArrayWrapper],
    out: Union[TensorLike, ArrayWrapper] = None,
    frame: FrameLike = None,
    axes: Union[list, tuple] = None,
) -> Union[TensorLike, ndarray, numbers.Number]:
    """
    Returns the dot product (without complex conjugation) of two quantities. The behaviour
    coincides with NumPy when all inputs are arrays and generalizes when they are not,
    but all inputs must be either all arrays or all tensors of some kind. The operation for
    tensors of order 1 and 2 have dedicated implementations, for higher order tensors
    it generalizes to tensor contraction along specified axes.

    Parameters
    ----------
    a: :class:`~sigmaepsilon.math.linalg.meta.TensorLike` or ArrayLike
       A tensor or an array.
    b: :class:`~sigmaepsilon.math.linalg.meta.TensorLike` or ArrayLike
       A tensor or an array.
    out: ArrayLike, Optional
        Output argument. This must have the exact kind that would be returned if it was
        not used. See `numpy.dot` for the details. Only if all inputs are ArrayLike.
        Default is None.
    frame: FrameLike, Optional
        The target frame of the output. Only if all inputs are TensorLike. If not specified,
        the returned tensor migh be returned in an arbitrary frame, depending on the inputs.
        Default is None.
    axes: tuple or list, Optional
        The indices along which contraction happens if any of the input tensors have a rank
        higher than 2. Default is None.

    Returns
    -------
    :class:`~sigmaepsilon.math.linalg.meta.TensorLike` or numpy.ndarray or scalar
        An array or a tensor, depending on the inputs.

    Notes
    -----
    For general tensors, the current implementation has an upper limit considering the rank
    of the input tensors. The sum of the ranks of the input tensors plus the sum of contraction
    indices must be at most 26.

    References
    ----------
    https://mathworld.wolfram.com/DotProduct.html

    Examples
    --------
    When working with NumPy arrays, the behaviour coincides with `numpy.dot`. To take the dot
    product of a 2nd order tensor and a vector, use it like this:

    >>> from sigmaepsilon.math.linalg import ReferenceFrame, Vector, Tensor2
    >>> from sigmaepsilon.math.linalg import dot
    >>> import numpy as np
    >>> frame = ReferenceFrame(np.eye(3))
    >>> A = Tensor2(np.eye(3), frame=frame)
    >>> v = Vector(np.array([1., 0, 0]), frame=frame)
    >>> dot(A, v)
    Array([1., 0., 0.])

    For general tensors, you have to specify the axes along which contraction happens:

    >>> from sigmaepsilon.math.linalg import Tensor
    >>> A = Tensor(np.ones((3, 3, 3, 3)), frame=frame)  # a tensor of order 4
    >>> B = Tensor(np.ones((3, 3, 3)), frame=frame)  # a tensor of order 3
    >>> dot(A, B, axes=(0, 0)).rank
    5

    """
    if isinstance(a, TensorLike) and isinstance(b, TensorLike):
        ra, rb = a.rank, b.rank
        result = None
        if ra == 1 and rb == 1:
            if out is not None:
                raise LinalgOperationInputError(
                    "Parameter 'out' is not allowed with tensors."
                )
            return np.dot(a.show(), b.show())
        elif ra == 2 and rb == 1:
            arr = (a.array @ b.show(a.frame.dual()).T).T
            result = b.__class__(arr, frame=a.frame)
        elif ra == 1 and rb == 2:
            arr = (a.array.T @ b.show(a.frame.dual()).T).T
            result = a.__class__(arr, frame=a.frame)
        elif ra == 2 and rb == 2:
            g = a.frame.Gram()
            result = a.__class__(a.array @ g @ b.show(a.frame), frame=a.frame)
        else:
            if not axes:
                msg = "The parameter 'axes' is required for tensor contraction of general tensors."
                raise LinalgMissingInputError(msg)
            ia = latinrange(ra, start=ord("a"))
            ib = latinrange(rb, start=ord("a") + ra)
            ax_a, ax_b = axes
            ic = latinrange(1, start=ord("a") + ra + rb)[0]
            ia[ax_a] = ic
            ib[ax_b] = ic
            command = "..." + "".join(ia) + "," + "..." + "".join(ib)
            arr = np.einsum(command, a.show(), b.show(), optimize="greedy")
            result = a.__class__._from_any_input(arr)
        if frame:
            result.frame = frame
        return result
    if frame:
        raise LinalgOperationInputError(
            "Parameter 'frame' is exclusive for tensorial inputs."
        )
    if not all([isinstance(x, (ndarray, ArrayWrapper, list)) for x in [a, b]]):
        raise TypeError("Invalid types encountered for dot product.")
    inputs = [x._array if isinstance(x, ArrayWrapper) else x for x in [a, b]]
    return np.dot(*inputs, out=out)


def cross(
    a: Union[TensorLike, ArrayWrapper],
    b: Union[TensorLike, ArrayWrapper],
    *args,
    frame: FrameLike = None,
    **kwargs,
) -> Union[TensorLike, ndarray]:
    """
    Calculates the cross product of two vectors or one vector and a second order
    tensor. The behaviour coincides with NumPy when all inputs are arrays and generalizes
    when they are not, but all inputs must be either all arrays or all tensors of some kind.

    Parameters
    ----------
    *args : Tuple, Optional
        Positional arguments forwarded to NumPy, if all input objects are arrays.
    a: :class:`~sigmaepsilon.math.linalg.meta.TensorLike` or ArrayLike
        A tensor or an array.
    b: :class:`~sigmaepsilon.math.linalg.meta.TensorLike` or ArrayLike
        A tensor or an array.
    frame: FrameLike, Optional
        The target frame of the output. Only if all inputs are TensorLike. If not specified,
        the returned tensor migh be returned in an arbitrary frame, depending on the inputs.
        Default is None.
    **kwargs: dict, Optional
        Keyword arguments forwarded to `numpy.cross`. As of NumPy version '1.22.4', there
        are no keyword arguments for `numpy.cross`, this is to assure compliance with
        all future versions of numpy.

    Returns
    -------
    numpy.ndarray or :class:`~sigmaepsilon.math.linalg.meta.TensorLike`
        An 1d or 2d array, or an 1d or 2d tensor, depending on the inputs.

    References
    ----------
    https://mathworld.wolfram.com/CrossProduct.html

    Examples
    --------
    The cross product of two vectors results in a vector:

    >>> from sigmaepsilon.math.linalg import ReferenceFrame, Vector, Tensor2
    >>> from sigmaepsilon.math.linalg import cross
    >>> import numpy as np
    >>> frame = ReferenceFrame(np.eye(3))
    >>> a = Vector(np.array([1., 0, 0]), frame=frame)
    >>> b = Vector(np.array([0, 1., 0]), frame=frame)
    >>> cross(a, b)
    Array([0., 0., 1.])

    The cross product of a second order tensor and a vector result a second order tensor:

    >>> A = Tensor2(np.eye(3), frame=frame)
    >>> cross(A, b)
    Array([[ 0.,  0., -1.],
           [ 0.,  0.,  0.],
           [ 1.,  0.,  0.]])

    """
    if isinstance(a, TensorLike) and isinstance(b, TensorLike):
        ra, rb = a.rank, b.rank
        result = None
        if ra == 1 and rb == 1:
            arr = np.cross(a.array, b.show(a.frame), axis=0)
            result = a.__class__(arr, frame=a.frame)
        elif ra == 2 and rb == 1:
            arr = np.cross(a.show(), b.show(), axis=0)
            result = a.__class__(arr)
        elif ra == 1 and rb == 2:
            arr = np.cross(a.show(), b.show(), axis=0)
            result = b.__class__(arr)
        else:
            msg = (
                "The cross product is not implemented",
                f"for tensors of rank {ra} and {rb}",
            )
            raise NotImplementedError(msg)
        if frame:
            result.frame = frame
        return result
    if frame:
        raise LinalgOperationInputError(
            "Parameter 'frame' is exclusive for tensorial inputs."
        )
    if any([isinstance(x, TensorLike) for x in [a, b]]):
        raise TypeError("Invalid types encountered for dot product.")
    if not all([isinstance(x, (ndarray, ArrayWrapper, list)) for x in [a, b]]):
        raise TypeError("Invalid types encountered for dot product.")
    inputs = [x._array if isinstance(x, ArrayWrapper) else x for x in [a, b]]
    return np.cross(*inputs, *args, **kwargs)


def show_vector(dcm: ndarray, arr: ndarray) -> ndarray:
    """
    Returns the coordinates of a single or multiple vectors in a frame specified
    by one or several DCM matrices. The function can handle the following scenarios:

        - a single (1d) vector and a single (2d) dcm matrix (trivial case)
        - a stack of vectors (2d) and a single (2d) dcm matrix
        - a stack of fectors (2d) and dcm matrices for each vector in the stack (3d)

    .. versionadded:: 1.0.5

    Parameters
    ----------
    dcm: numpy.ndarray
        The dcm matrix of the transformation as a 2d or 3d float array.
    arr: numpy.ndarray
        1d or 2d float array of coordinates of a single vector. If it is 2d, then
        it is assumed that the coordinates of the i-th vector are accessible as
        `arr[i]`.

    Returns
    -------
    numpy.ndarray
        The new coordinates with the same shape as `arr`.
    """
    if len(arr.shape) == 1 and len(dcm.shape) == 2:
        return _show_vector(dcm, arr)  # dcm @ arr
    elif len(arr.shape) == 2 and len(dcm.shape) == 2:
        return _show_vectors(dcm, arr)  # dcm @ arr
    elif len(arr.shape) == 2 and len(dcm.shape) == 3:
        return _show_vectors_multi(dcm, arr)  # dcm @ arr
    else:
        msg = (
            "Mismatch in shapes!"
            f"Input one has shape {dcm.shape} and input two has shape {arr.shape}."
            "See the docs for the correct input shapes."
        )
        raise LinalgOperationInputError(msg)


def show_frame(dcm: ndarray, arr: ndarray) -> ndarray:
    if len(arr.shape) == 2 and len(dcm.shape) == 2:
        return _show_frame(dcm, arr)  # dcm @ arr
    elif len(arr.shape) == 3 and len(dcm.shape) == 2:
        return _show_frames(dcm, arr)  # dcm @ arr
    elif len(arr.shape) == 3 and len(dcm.shape) == 3:
        return _show_frames_multi(dcm, arr)  # dcm @ arr
    else:
        msg = (
            "Mismatch in shapes!"
            f"Input one has shape {dcm.shape} and input two has shape {arr.shape}."
            "See the docs for the correct input shapes."
        )
        raise LinalgOperationInputError(msg)


@njit(nogil=True, cache=__cache)
def _show_vector(dcm: ndarray, arr: ndarray) -> ndarray:
    """
    Returns the coordinates of a single vector in a frame specified
    by a DCM matrix.

    Parameters
    ----------
    dcm: numpy.ndarray
        The dcm matrix of the transformation as a 2d float array.
    arr: numpy.ndarray
        1d float array of coordinates of a single vector.

    Returns
    -------
    numpy.ndarray
        The new coordinates of the vector with the same shape as `arr`.
    """
    return dcm @ arr


@njit(nogil=True, parallel=True, cache=__cache)
def _show_vectors(dcm: ndarray, arr: ndarray) -> ndarray:
    """
    Returns the coordinates of multiple vectors in a frame specified
    by a DCM matrix.

    Parameters
    ----------
    dcm: numpy.ndarray
        The dcm matrix of the transformation as a 2d float array.
    arr: numpy.ndarray
        2d float array of coordinates of multiple vectors.

    Returns
    -------
    numpy.ndarray
        The new coordinates of the vectors with the same shape as `arr`.
    """
    res = np.zeros_like(arr)
    for i in prange(arr.shape[0]):
        res[i] = dcm @ arr[i, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _show_vectors_multi(dcm: ndarray, arr: ndarray) -> ndarray:
    """
    Returns the coordinates of multiple vectors and multiple DCM matrices.

    Parameters
    ----------
    dcm: numpy.ndarray
        The dcm matrix of the transformation as a 3d float array.
    arr: numpy.ndarray
        2d float array of coordinates of multiple vectors.

    Returns
    -------
    numpy.ndarray
        The new coordinates of the vectors with the same shape as `arr`.
    """
    res = np.zeros_like(arr)
    for i in prange(arr.shape[0]):
        res[i] = dcm[i] @ arr[i, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _show_frame(dcm: ndarray, arr: ndarray) -> ndarray:
    """
    Returns the coordinates of a single frame in a target frame specified
    by a DCM matrix.

    Parameters
    ----------
    dcm: numpy.ndarray
        The dcm matrix of the transformation as a 2d float array.
    arr: numpy.ndarray
        2d float array of coordinates of a single frame.

    Returns
    -------
    numpy.ndarray
        The new coordinates of the frame with the same shape as `arr`.
    """
    res = np.zeros_like(arr)
    for i in prange(arr.shape[-1]):
        res[i, :] = dcm @ arr[i, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _show_frames(dcm: ndarray, arr: ndarray) -> ndarray:
    """
    Returns the coordinates of multiple frames in a target frame specified
    by a DCM matrix.

    Parameters
    ----------
    dcm: numpy.ndarray
        The dcm matrix of the transformation as a 2d float array.
    arr: numpy.ndarray
        3d float array of coordinates of multiple frames.

    Returns
    -------
    numpy.ndarray
        The new coordinates of the frames with the same shape as `arr`.
    """
    res = np.zeros_like(arr)
    for i in prange(arr.shape[0]):
        for j in prange(arr.shape[-1]):
            res[i, j, :] = dcm @ arr[i, j, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _show_frames_multi(dcm: ndarray, arr: ndarray) -> ndarray:
    """
    Returns the coordinates of multiple frames and multiple DCM matrices.

    Parameters
    ----------
    dcm: numpy.ndarray
        The dcm matrix of the transformation as a 3d float array.
    arr: numpy.ndarray
        3d float array of coordinates of multiple frames.

    Returns
    -------
    numpy.ndarray
        The new coordinates of the frames with the same shape as `arr`.
    """
    res = np.zeros_like(arr)
    for i in prange(arr.shape[0]):
        for j in prange(arr.shape[-1]):
            res[i, j, :] = dcm[i] @ arr[i, j, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _transpose_multi(dcm: ndarray) -> ndarray:
    N = dcm.shape[0]
    res = np.zeros_like(dcm)
    for i in prange(N):
        res[i, :, :] = dcm[i].T
    return res


def transpose_axes(arr: ndarray) -> ndarray:
    if len(arr.shape) == 2:
        return arr.T
    elif len(arr.shape) == 3:
        # FIXME this might be unnecessary
        return _transpose_multi(arr)
    else:
        shape = arr.shape
        indices = tuple(range(len(shape)))
        data_indices = indices[:-2]
        tensor_indices = indices[len(shape) - 2 :]
        indices = data_indices + tensor_indices[::-1]
        return np.transpose(arr, indices)


def normalize_frame(axes: ndarray) -> ndarray:
    """
    Returns the frame with normalized base vectors.

    Parameters
    ----------
    axes: numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return np.array([normalize(a) for a in axes], dtype=axes.dtype)


def Gram(axes: ndarray) -> ndarray:
    """
    Returns the Gram matrix of a frame.

    Parameters
    ----------
    axes: numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return axes @ transpose_axes(axes)


def dual_frame(axes: ndarray) -> ndarray:
    """
    Returns the dual frame of the input.

    Parameters
    ----------
    axes: numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return transpose_axes(np.linalg.inv(axes))


def random_pos_semidef_matrix(N) -> ndarray:
    """
    Returns a random positive semidefinite matrix of shape (N, N).

    Example
    -------
    >>> from sigmaepsilon.math.linalg import random_pos_semidef_matrix, is_pos_semidef
    >>> arr = random_pos_semidef_matrix(2)
    >>> is_pos_semidef(arr)
    True

    """
    A = np.random.rand(N, N)
    return A.T @ A


def random_posdef_matrix(N, alpha: float = 1e-12) -> ndarray:
    """
    Returns a random positive definite matrix of shape (N, N).

    All eigenvalues of this matrix are >= alpha.

    Example
    -------
    >>> from sigmaepsilon.math.linalg import random_posdef_matrix, is_pos_def
    >>> arr = random_posdef_matrix(2)
    >>> is_pos_def(arr)
    True

    """
    A = np.random.rand(N, N)
    return A @ A.T + alpha * np.eye(N)


def inv_sym_3x3(m: Matrix, as_adj_det=False) -> Matrix:
    P11, P12, P13, P21, P22, P23, P31, P32, P33 = symbols(
        "P_{11} P_{12} P_{13} P_{21} P_{22} P_{23} P_{31} \
                P_{32} P_{33}",
        real=True,
    )
    Pij = [[P11, P12, P13], [P21, P22, P23], [P31, P32, P33]]
    P = sy.Matrix(Pij)
    detP = P.det()
    adjP = P.adjugate()
    invP = adjP / detP
    subs = {s: r for s, r in zip(sy.flatten(P), sy.flatten(m))}
    if as_adj_det:
        return detP.subs(subs), adjP.subs(subs)
    else:
        return invP.subs(subs)


@njit(nogil=True, parallel=True, cache=__cache)
def vpath(p1: ndarray, p2: ndarray, n: int) -> ndarray:
    nD = len(p1)
    dist = p2 - p1
    length = np.linalg.norm(dist)
    s = np.linspace(0, length, n)
    res = np.zeros((n, nD), dtype=p1.dtype)
    d = dist / length
    for i in prange(n):
        res[i] = p1 + s[i] * d
    return res


@njit(nogil=True, cache=__cache)
def linsolve(A, b) -> ndarray:
    return np.linalg.solve(A, b)


@njit(nogil=True, cache=__cache)
def inv(A: ndarray) -> ndarray:
    return np.linalg.inv(A)


@njit(nogil=True, cache=__cache)
def matmul(A: ndarray, B: ndarray) -> ndarray:
    return A @ B


@njit(nogil=True, cache=__cache)
def ATB(A: ndarray, B: ndarray) -> ndarray:
    return A.T @ B


@njit(nogil=True, cache=__cache)
def matmulw(A: ndarray, B: ndarray, w: float = 1.0) -> ndarray:
    return w * (A @ B)


@njit(nogil=True, cache=__cache)
def ATBA(A: ndarray, B: ndarray) -> ndarray:
    return A.T @ B @ A


@njit(nogil=True, cache=__cache)
def ATBAw(A: ndarray, B: ndarray, w: float = 1.0) -> ndarray:
    return w * (A.T @ B @ A)


@guv(["(f8[:, :], f8)"], "(n, n) -> ()", nopython=True, cache=__cache)
def det3x3(A, res):
    res = (
        A[0, 0] * A[1, 1] * A[2, 2]
        - A[0, 0] * A[1, 2] * A[2, 1]
        - A[0, 1] * A[1, 0] * A[2, 2]
        + A[0, 1] * A[1, 2] * A[2, 0]
        + A[0, 2] * A[1, 0] * A[2, 1]
        - A[0, 2] * A[1, 1] * A[2, 0]
    )


@guv(["(f8[:, :], f8)"], "(n, n) -> ()", nopython=True, cache=__cache)
def det2x2(A, res):
    res = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


@njit(nogil=True, cache=__cache)
def inv2x2(A) -> ndarray:
    res = np.zeros_like(A)
    d = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    res[0, 0] = A[1, 1] / d
    res[1, 1] = A[0, 0] / d
    res[0, 1] = -A[0, 1] / d
    res[1, 0] = -A[1, 0] / d
    return res


@guv(["(f8[:, :], f8[:, :])"], "(n, n) -> (n, n)", nopython=True, cache=__cache)
def inv2x2u(A, res):
    d = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    res[0, 0] = A[1, 1] / d
    res[1, 1] = A[0, 0] / d
    res[0, 1] = -A[0, 1] / d
    res[1, 0] = -A[1, 0] / d


@guv(["(f8[:, :], f8[:, :])"], "(n, n) -> (n, n)", nopython=True, cache=__cache)
def adj3x3(A, res):
    res[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    res[0, 1] = -A[0, 1] * A[2, 2] + A[0, 2] * A[2, 1]
    res[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    res[1, 0] = -A[1, 0] * A[2, 2] + A[1, 2] * A[2, 0]
    res[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    res[1, 2] = -A[0, 0] * A[1, 2] + A[0, 2] * A[1, 0]
    res[2, 0] = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    res[2, 1] = -A[0, 0] * A[2, 1] + A[0, 1] * A[2, 0]
    res[2, 2] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


@guv(["(f8[:, :], f8[:, :])"], "(n, n) -> (n, n)", nopython=True, cache=__cache)
def inv3x3u(A, res):
    d = (
        A[0, 0] * A[1, 1] * A[2, 2]
        - A[0, 0] * A[1, 2] * A[2, 1]
        - A[0, 1] * A[1, 0] * A[2, 2]
        + A[0, 1] * A[1, 2] * A[2, 0]
        + A[0, 2] * A[1, 0] * A[2, 1]
        - A[0, 2] * A[1, 1] * A[2, 0]
    )
    res[0, 0] = A[1, 1] * A[2, 2] / d - A[1, 2] * A[2, 1] / d
    res[0, 1] = -A[0, 1] * A[2, 2] / d + A[0, 2] * A[2, 1] / d
    res[0, 2] = A[0, 1] * A[1, 2] / d - A[0, 2] * A[1, 1] / d
    res[1, 0] = -A[1, 0] * A[2, 2] / d + A[1, 2] * A[2, 0] / d
    res[1, 1] = A[0, 0] * A[2, 2] / d - A[0, 2] * A[2, 0] / d
    res[1, 2] = -A[0, 0] * A[1, 2] / d + A[0, 2] * A[1, 0] / d
    res[2, 0] = A[1, 0] * A[2, 1] / d - A[1, 1] * A[2, 0] / d
    res[2, 1] = -A[0, 0] * A[2, 1] / d + A[0, 1] * A[2, 0] / d
    res[2, 2] = A[0, 0] * A[1, 1] / d - A[0, 1] * A[1, 0] / d


@njit(nogil=True, cache=__cache)
def inv3x3(A):
    res = np.zeros_like(A)
    det = (
        A[0, 0] * A[1, 1] * A[2, 2]
        - A[0, 0] * A[1, 2] * A[2, 1]
        - A[0, 1] * A[1, 0] * A[2, 2]
        + A[0, 1] * A[1, 2] * A[2, 0]
        + A[0, 2] * A[1, 0] * A[2, 1]
        - A[0, 2] * A[1, 1] * A[2, 0]
    )
    res[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    res[0, 1] = -A[0, 1] * A[2, 2] + A[0, 2] * A[2, 1]
    res[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    res[1, 0] = -A[1, 0] * A[2, 2] + A[1, 2] * A[2, 0]
    res[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    res[1, 2] = -A[0, 0] * A[1, 2] + A[0, 2] * A[1, 0]
    res[2, 0] = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    res[2, 1] = -A[0, 0] * A[2, 1] + A[0, 1] * A[2, 0]
    res[2, 2] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    res /= det
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def inv3x3_bulk(A) -> ndarray:
    res = np.zeros_like(A)
    for i in prange(A.shape[0]):
        det = (
            A[i, 0, 0] * A[i, 1, 1] * A[i, 2, 2]
            - A[i, 0, 0] * A[i, 1, 2] * A[i, 2, 1]
            - A[i, 0, 1] * A[i, 1, 0] * A[i, 2, 2]
            + A[i, 0, 1] * A[i, 1, 2] * A[i, 2, 0]
            + A[i, 0, 2] * A[i, 1, 0] * A[i, 2, 1]
            - A[i, 0, 2] * A[i, 1, 1] * A[i, 2, 0]
        )
        res[i, 0, 0] = A[i, 1, 1] * A[i, 2, 2] - A[i, 1, 2] * A[i, 2, 1]
        res[i, 0, 1] = -A[i, 0, 1] * A[i, 2, 2] + A[i, 0, 2] * A[i, 2, 1]
        res[i, 0, 2] = A[i, 0, 1] * A[i, 1, 2] - A[i, 0, 2] * A[i, 1, 1]
        res[i, 1, 0] = -A[i, 1, 0] * A[i, 2, 2] + A[i, 1, 2] * A[i, 2, 0]
        res[i, 1, 1] = A[i, 0, 0] * A[i, 2, 2] - A[i, 0, 2] * A[i, 2, 0]
        res[i, 1, 2] = -A[i, 0, 0] * A[i, 1, 2] + A[i, 0, 2] * A[i, 1, 0]
        res[i, 2, 0] = A[i, 1, 0] * A[i, 2, 1] - A[i, 1, 1] * A[i, 2, 0]
        res[i, 2, 1] = -A[i, 0, 0] * A[i, 2, 1] + A[i, 0, 1] * A[i, 2, 0]
        res[i, 2, 2] = A[i, 0, 0] * A[i, 1, 1] - A[i, 0, 1] * A[i, 1, 0]
        res[i] /= det
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def inv3x3_bulk2(A) -> ndarray:
    res = np.zeros_like(A)
    for i in prange(A.shape[0]):
        res[i] = inv3x3(A[i])
    return res


@njit(nogil=True, cache=__cache)
def normalize(A) -> ndarray:
    return A / np.linalg.norm(A)


@njit(nogil=True, parallel=True, cache=__cache)
def normalize2d(A) -> ndarray:
    res = np.zeros_like(A)
    for i in prange(A.shape[0]):
        res[i] = normalize(A[i])
    return res


@njit(nogil=True, cache=__cache)
def norm(A) -> float:
    return np.linalg.norm(A)


@njit(nogil=True, parallel=True, cache=__cache)
def norm2d(A) -> ndarray:
    res = np.zeros(A.shape[0])
    for i in prange(A.shape[0]):
        res[i] = norm(A[i, :])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _linspace(p0: ndarray, p1: ndarray, N):
    s = p1 - p0
    L = np.linalg.norm(s)
    n = s / L
    djac = L / (N - 1)
    step = n * djac
    res = np.zeros((N, p0.shape[0]))
    res[0] = p0
    for i in prange(1, N - 1):
        res[i] = p0 + i * step
    res[-1] = p1
    return res


def linspace(start, stop, N) -> ndarray:
    if isinstance(start, ndarray):
        return _linspace(start, stop, N)
    else:
        return np.linspace(start, stop, N)


@njit(nogil=True, parallel=True, cache=__cache)
def linspace1d(start, stop, N) -> ndarray:
    res = np.zeros(N)
    di = (stop - start) / (N - 1)
    for i in prange(N):
        res[i] = start + i * di
    return res


def generalized_left_inverse(matrix: ndarray) -> ndarray:
    """Returns the generalized left inverse

    .. math::
        :nowrap:

        \\begin{equation}
            \left( \mathbf{A}^{T} \mathbf{A} \\right)^{-1} \mathbf{A}^{T}
        \\end{equation}

    """
    return np.linalg.inv(matrix.T @ matrix) @ matrix.T


def generalized_right_inverse(matrix: ndarray) -> ndarray:
    """Returns the generalized right inverse

    .. math::
        :nowrap:

        \\begin{equation}
            \mathbf{A}^{T} \left( \mathbf{A} \mathbf{A}^{T} \\right)^{-1}
        \\end{equation}

    """
    return matrix.T @ np.linalg.inv(matrix @ matrix.T)


def generalized_inverse(matrix: ndarray) -> ndarray:
    """
    Returns the generalized inverse of the input matrix, in any of the following
    cases:

    1. The matrix is square and has full rank. In this case the returned matrix
       is the usual inverse.

    2. The matrix has more columns than rows and has full row rank. In this case
       the generalized right inverse is returned.

    3. The matrix has more rows than columns and has full column rank. In this case
       the generalized left inverse is returned.

    """
    if not len(matrix.shape) == 2:
        raise LinalgOperationInputError("The input must be a matrix")

    num_rows, num_columns = matrix.shape
    rank = np.linalg.matrix_rank(matrix)
    if (num_rows == num_columns) and rank == num_columns == num_rows:
        return np.linalg.inv(matrix)
    elif (num_rows > num_columns) and rank == num_columns:
        return generalized_left_inverse(matrix)
    elif (num_rows < num_columns) and rank == num_rows:
        return generalized_right_inverse(matrix)
    else:
        raise LinalgError("The matrix has no inverse")


def unit_basis_vector(length: int, index: int = 0, value: float = 1.0) -> ndarray:
    """
    Returns a unit basis vector of length `length` with a value of `value` at
    the index `index`.
    """
    return value * np.bincount([index], None, length)
