import numpy as np
import numbers
from numpy import ndarray
from numba import njit, prange
from typing import Union, Tuple, Iterable

try:
    from collections.abc import Iterable  # pragma: no cover
except ImportError:  # pragma: no cover
    from collections import Iterable  # pragma: no cover

__cache = True

ArrayOrFloat = Union[float, ndarray, list]


def itype_of_ftype(dtype):
    """
    Returns a matching NumPy integer type to a float type.
    """
    name = np.dtype(dtype).name
    if "32" in name:
        return np.int32
    elif "64" in name:
        return np.int64
    else:
        raise TypeError("Unrecognized float type.")  # pragma: no cover


@njit(nogil=True, cache=__cache)
def minmax(a: ndarray) -> Tuple[float]:
    """
    Returns the minimum and maximum values of an array.
    """
    return a.min(), a.max()


def ascont(array: ndarray) -> ndarray:
    """
    Returns the input as contiguous array.
    It is basically a shortcut to `numpy.ascontiguousarray`.
    """
    return np.ascontiguousarray(array)


@njit(nogil=True, cache=__cache)
def clip1d(a: ndarray, a_min: float, a_max: float) -> ndarray:
    """
    Clips the values outside the interval [a_min, a_max] to
    either a_min or a_max.

    Parameters
    ----------
    a : numpy.ndarray
        An 1d array.
    a_min : float
        The lower limit.
    a_max : float
        The upper limit.

    Examples
    --------
    >>> from sigmaepsilon.math import clip1d
    >>> import numpy as np
    ...
    >>> clip1d(np.array([0.9, 2.5]), 1.0, 1.5)
    array([1. , 1.5])

    """
    a[a < a_min] = a_min
    a[a > a_max] = a_max
    return a


def atleastnd(
    a: Union[numbers.Number, Iterable],
    n: int = 2,
    front: bool = True,
    back: bool = False,
) -> ndarray:
    """
    Returns an array that is at least 'n' dimensional.
    The required shape is obtained by inserting new axes either
    before or after existing ones. This behaviour can be controlled
    using the parameters 'front' and 'back'. If front is True and back
    is False, new axes are crated before the first existing data index,
    and the opposite happens in every other case.

    Examples
    --------
    >>> from sigmaepsilon.math import atleastnd
    >>> import numpy as np
    ...
    >>> atleastnd(np.array([1, 1]), 3, front=True).shape
    (1, 1, 2)

    >>> atleastnd(np.array([1, 1]), 3, back=True).shape
    (2, 1, 1)

    """
    if not isinstance(a, Iterable):
        a = [
            a,
        ]
    if not isinstance(a, ndarray):
        a = np.array(a)
    shp = a.shape
    nD = len(shp)
    if nD >= n:
        return a
    else:
        if front and not back:
            newshape = (n - nD) * (1,) + shp
        else:
            newshape = shp + (n - nD) * (1,)
        return np.reshape(a, newshape)


def atleast1d(a: Union[numbers.Number, Iterable]) -> ndarray:
    """
    Returns an array that is at least 1 dimensional.

    Examples
    --------
    >>> from sigmaepsilon.math import atleast1d
    >>> atleast1d(1)
    array([1])

    """
    return np.atleast_1d(a)


def atleast2d(a: Union[numbers.Number, Iterable], **kwargs) -> ndarray:
    """
    Returns an array that is at least 2 dimensional.

    Examples
    --------
    >>> from sigmaepsilon.math import atleast2d
    >>> atleast2d(1)
    array([[1]])

    """
    return atleastnd(a, 2, **kwargs)


def matrixform(a: Union[numbers.Number, Iterable]) -> ndarray:
    """
    Returns an array that is at least 2 dimensional.

    Examples
    --------
    >>> from sigmaepsilon.math import matrixform
    >>> matrixform(1)
    array([[1]])

    """
    if not isinstance(a, Iterable):
        a = [
            a,
        ]
    if not isinstance(a, ndarray):
        a = np.array(a)
    size = len(a.shape)
    assert size <= 2, "Input array must be at most 2 dimensional."
    if size == 1:
        nV, nC = len(a), 1
        return a.reshape(nV, nC)
    return a


def atleast3d(a: Union[numbers.Number, Iterable], **kwargs) -> ndarray:
    """
    Returns an array that is at least 3 dimensional.

    Examples
    --------
    >>> from sigmaepsilon.math import atleast3d
    >>> atleast3d(1)
    array([[[1]]])

    """
    return atleastnd(a, 3, **kwargs)


def atleast4d(a: ndarray, **kwargs) -> ndarray:
    """
    Returns an array that is at least 4 dimensional.

    Examples
    --------
    >>> from sigmaepsilon.math import atleast4d
    >>> atleast4d(1)
    array([[[[1]]]])

    """
    return atleastnd(a, 4, **kwargs)


@njit(nogil=True, cache=__cache)
def flatten2dC(a: ndarray) -> ndarray:
    I, J = a.shape
    res = np.zeros(I * J, dtype=a.dtype)
    ind = 0
    for i in range(I):
        for j in range(J):
            res[ind] = a[i, j]
            ind += 1
    return res


@njit(nogil=True, cache=__cache)
def flatten2dF(a: ndarray) -> ndarray:
    I, J = a.shape
    res = np.zeros(I * J, dtype=a.dtype)
    ind = 0
    for j in range(J):
        for i in range(I):
            res[ind] = a[i, j]
            ind += 1
    return res


def flatten2d(a: ndarray, order: str = "C") -> ndarray:
    """
    Returns a flattened view of `a`.
    """
    if order == "C":
        return flatten2dC(a)
    elif order == "F":
        return flatten2dF(a)


def bool_to_float(a: Iterable, true: float = 1.0, false: float = 0.0) -> ndarray:
    """
    Transforms a boolean array to a float array using the specified
    values for `True` and `False`.

    Example
    -------
    >>> from sigmaepsilon.math import bool_to_float
    >>> bool_to_float([True, False], 1.0, -2.0)
    array([ 1., -2.])

    """
    if not isinstance(a, ndarray):
        a = np.array(a)
    res = np.full(a.shape, false, dtype=float)
    res[a] = true
    return res


def choice(choices: Iterable, size: Tuple, probs: Iterable = None) -> ndarray:
    """
    Returns a NumPy array, whose elements are selected from
    'choices' under probabilities provided with 'probs' (optionally).

    Parameters
    ----------
    choices : Iterable
        The choices.
    size : tuple
        The size of the output array.
    probs : Iterable, Optional
        The probabilities of each item in 'choices'. Default is an array
        of uniform probabilities.

    Example
    -------
    ```python
    N, p = 2, 0.2
    choice([False, True], (N, N), [p, 1-p])
    array([[ True,  True],
           [ True,  True]])
    ```

    """
    if probs is None:
        probs = np.full((len(choices),), 1 / len(choices))
    return np.random.choice(a=choices, size=size, p=probs)


@njit(nogil=True, parallel=True, cache=__cache)
def repeat(a: ndarray, N: int = 1) -> ndarray:
    """
    Repeats an array N-times.

    Parameters
    ----------
    a: numpy.ndarray
        Input array.
    N: int, Optional
        Number of repetitions. Default is 1.

    Returns
    -------
    numpy.ndarray
        A NumPy array with shape (N, a.shape[0], a.shape[1]).

    Example
    -------
    For example, to generate basis vectors for 10 vectors embedded
    in the same coordinate frame, we need to stack up 10 identical
    identity matrices. This can be done the quickest by:

    >>> from sigmaepsilon.math import repeat
    >>> import numpy as np
    ...
    >>> repeat(np.eye(2), 3)
    array([[[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [0., 1.]]])

    >>> repeat(np.eye(2), 3).shape
    (3, 2, 2)

    """
    res = np.zeros((N, a.shape[0], a.shape[1]), dtype=a.dtype)
    for i in prange(N):
        res[i, :, :] = a
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def repeat1d(a: ndarray, N=1) -> ndarray:
    """
    Repeats a 1d array N times.

    Example
    -------
    >>> from sigmaepsilon.math import repeat1d
    >>> import numpy as np
    ...
    >>> repeat1d(np.array([1, 2]), 3)
    array([1, 2, 1, 2, 1, 2])

    """
    M = a.shape[0]
    res = np.zeros(N * M, dtype=a.dtype)
    for i in prange(N):
        res[i * M : (i + 1) * M] = a
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def tile(a: ndarray, da: ndarray, N: int = 1) -> ndarray:
    """
    Tiles a 1d array N times.

    Example
    -------
    >>> from sigmaepsilon.math import tile
    >>> import numpy as np
    ...
    >>> arr = tile(np.array([[0, 0]]), np.array([[1, -1]]), 3)
    >>> arr
    array([[[ 0,  0]],
    <BLANKLINE>
           [[ 1, -1]],
    <BLANKLINE>
           [[ 2, -2]]])

    """
    res = np.zeros((N, a.shape[0], a.shape[1]), dtype=a.dtype)
    for i in prange(N):
        res[i, :, :] = a + i * da
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def tile1d(a: ndarray, da: ndarray, N=1) -> ndarray:
    M = a.shape[0]
    res = np.zeros(N * M, dtype=a.dtype)
    for i in prange(N):
        res[i * M : (i + 1) * M] = a + i * da
    return res


def indices_of_equal_rows(x: ndarray, y: ndarray, tol=1e-12):
    from .logical import isintegerarray

    nX, dX = x.shape
    nY, dY = y.shape
    assert dX == dY, "Input arrays must have identical second dimensions."
    square = nX == nY
    integer = isintegerarray(x) and isintegerarray(y)
    if integer:
        if square:
            inds = np.flatnonzero((x == y).all(1))
            return inds, np.copy(inds)
        else:
            return indices_of_equal_rows_njit(x, y, 0)
    else:
        if square:
            return indices_of_equal_rows_square_njit(x, y, tol)
        else:
            return indices_of_equal_rows_njit(x, y, tol)


@njit(nogil=True, parallel=True, cache=__cache)
def indices_of_equal_rows_njit(x: ndarray, y: ndarray, tol: float = 1e-12):
    R = np.zeros((x.shape[0], y.shape[0]), dtype=x.dtype)
    for i in prange(R.shape[0]):
        for j in prange(R.shape[1]):
            R[i, j] = np.sum(np.abs(x[i] - y[j]))
    return np.where(R <= tol)


@njit(nogil=True, parallel=True, cache=__cache)
def indices_of_equal_rows_square_njit(x: ndarray, y: ndarray, tol=1e-12):
    nx, ny = x.shape[0], y.shape[0]
    if nx < ny:
        n = nx
    else:
        n = ny
    R = np.zeros((n, n), dtype=x.dtype)
    for i in prange(n):
        for j in prange(n):
            R[i, j] = np.sum(np.abs(x[i] - y[j]))
    return np.where(R <= tol)


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def count_cols(arr: ndarray) -> ndarray:
    """
    Count and return the number of columns for each row in
    the input array.
    """
    n = len(arr)
    res = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        res[i] = len(arr[i])
    return res


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def count_cols_csr(indptr: ndarray, indices: ndarray) -> ndarray:
    """
    Count and return the number of columns for each row in
    a CSR matrix.

    Parameters
    ----------
    indptr : numpy.ndarray
        Index pointers of a CSR matrix.
    indices : numpy.ndarray
        Column indices of the data in a CSR matrix.

    Returns
    -------
    numpy.ndarray
        1d integer array.

    See Also
    --------
    :class:`~sigmaepsilon.math.linalg.sparse.csr.csr_matrix`
    """
    n = len(indptr)
    res = np.zeros(n, dtype=indptr.dtype)
    for i in prange(n):
        res[i] = indices[indptr[i + 1]] - indices[indptr[i]]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def repeat_diagonal_2d(a: ndarray, N: int = 2) -> ndarray:
    """
    Assembles a 2d dense block-diagonal matrix by repeating.

    Parameters
    ----------
    a : numpy.ndarray
        A 2d array. This is the block that gets repeated.
    N : int, Optional
        The number of copies.

    Returns
    -------
    numpy.ndarray
        A 2d numpy array.
    """
    nR, nC = a.shape[:2]
    res = np.zeros((nR * N, nC * N), dtype=a.dtype)
    for i in prange(N):
        _iR = i * nR
        iR_ = _iR + nR
        _iC = i * nC
        iC_ = _iC + nC
        res[_iR:iR_, _iC:iC_] = a
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _to_range_1d_(vals: ndarray, source: ndarray, target: ndarray):
    res = np.zeros_like(vals)
    s0, s1 = source
    t0, t1 = target
    b = (t1 - t0) / (s1 - s0)
    a = (t0 + t1) / 2 - b * (s0 + s1) / 2
    for i in prange(res.shape[0]):
        res[i] = a + b * vals[i]
    return res


def to_range_1d(
    vals: ndarray | Iterable,
    *_,
    source: ndarray,
    target: ndarray = None,
):
    if not isinstance(vals, ndarray):
        vals = np.array(
            [
                vals,
            ]
        )
    source = np.array([0.0, 1.0]) if source is None else np.array(source)
    target = np.array([-1.0, 1.0]) if target is None else np.array(target)
    return _to_range_1d_(vals, source, target)


# !FIXME : assumes a unique input array
@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def find1d(arr, space):
    res = np.zeros(arr.shape, dtype=np.uint64)
    for i in prange(arr.shape[0]):
        res[i] = np.where(arr[i] == space)[0][0]
    return res


# !TODO generalize this up to n dimensions
# !FIXME : assumes a unique input array
@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def find2d(arr, space):
    res = np.zeros(arr.shape, dtype=np.uint64)
    for i in prange(arr.shape[0]):
        res[i] = find1d(arr[i], space)
    return res
