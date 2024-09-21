from typing import Union
import numpy as np
from numpy import ndarray
from .utils import atleast1d


def is_none_or_false(a):
    if isinstance(a, bool):
        return not a
    elif a is None:
        return True
    return False


def isfloatarray(a: ndarray) -> bool:
    """
    Returns `True` if `a` is a float array.
    """
    return np.issubdtype(a.dtype, float)


def isintegerarray(a: ndarray) -> bool:
    """
    Returns `True` if `a` is an integer array.
    """
    return np.issubdtype(a.dtype, int)


def isintarray(a: ndarray) -> bool:
    """
    Returns `True` if `a` is a integer array.
    """
    return isintegerarray(a)


def isboolarray(a: ndarray) -> bool:
    """
    Returns `True` if `a` is a boolean array.
    """
    return np.issubdtype(a.dtype, bool)


def is1dfloatarray(a: ndarray) -> bool:
    """
    Returns `True` if `a` is a 1d float array.
    """
    return isfloatarray(a) and len(a.shape) == 1


def is1dintarray(a: ndarray) -> bool:
    """
    Returns `True` if `a` is a 1d integer array.
    """
    return isintarray(a) and len(a.shape) == 1


def issymmetric(a: ndarray, tol: float = 1e-8) -> bool:
    """
    Returns `True` if `a` is symmetric with a given tolerance
    prescribed by `tol`.
    """
    return np.linalg.norm(a - a.T) < tol


def isposdef(A: ndarray, tol=0) -> bool:
    """
    Returns `True` if `A` is positive definite.

    Examples
    --------
    >>> from sigmaepsilon.math.linalg import random_posdef_matrix
    >>> from sigmaepsilon.math.logical import isposdef
    ...
    >>> A = random_posdef_matrix(3, 0.1)
    >>> isposdef(A)
    True

    >>> A[0, 0] = 0
    >>> isposdef(A)
    False

    """
    return np.all(np.linalg.eigvals(A) > tol)


def ispossemidef(A: ndarray) -> bool:
    """
    Returns `True` if `A` is positive semidefinite.

    Example
    -------
    >>> from sigmaepsilon.math.linalg import random_pos_semidef_matrix
    >>> from sigmaepsilon.math.logical import ispossemidef
    >>> A = random_pos_semidef_matrix(3)
    >>> ispossemidef(A)
    True

    """
    return np.all(np.linalg.eigvals(A) >= 0)


def isclose(
    x1: ndarray, x2: ndarray, *, atol: float = 1e-8, rtol: float = 1e-5
) -> Union[bool, ndarray]:
    """
    Returns a boolean array where two arrays are element-wise equal
    in absolute and optionally in relative sense. In the latter case
    the relative difference is measured agains `x2`.

    Absolute difference is measured as abs(`x1` - `x2`) <= `atol`, relative
    difference is measured as abs(`x1` - `x2`) <= abs(`rtol` * `x2`).

    .. versionadded:: 0.0.6

    .. warning:: Default values might fail for very small numbers.

    Parameters
    ----------
    x1 : ndarray or float
        Values of the first array.
    x2 : ndarray or float
        Values of the second array.
    atol : float, Optional
        Absolute tolerance. It must be a positive number, or None to
        turn this check off. Default is 1e-8.
    rtol : float, Optional
        Relative tolerance. It must be a positive number or None to
        turn this check off. In the latter case, the input arrays are only
        compared in an absolute sense. Default is 1e-5.

    Returns
    -------
    numpy.ndarray or float
        A boolean array or a float, depending on the input.

    See Also
    --------
    numpy.isclose
    numpy.allclose
    math.isclose
    """
    if not isinstance(x1, ndarray):
        x1 = atleast1d(x1)
    if not isinstance(x2, ndarray):
        x2 = atleast1d(x2)
    c = np.ones(x1.shape, dtype=bool)
    assert (atol is not None) or (
        rtol is not None
    ), "At least one from `atol` or `rtol` must be other than `None`."
    if atol is not None:
        assert atol > 0, "The absolute tolerance must be a positive number."
        c_abs = np.abs(x1 - x2) <= atol
        c = c & c_abs
    if rtol is not None:
        assert rtol > 0, "The relative tolerance must be a positive number."
        c_rel = np.abs(x1 - x2) <= np.abs(rtol * x2)
        c = c & c_rel
    if len(c.shape) == 1 and c.shape[0] == 1:
        return c[0]
    return c


def allclose(*args, **kwargs) -> bool:
    """
    Same as `isclose`, but it returns a single boolean that is `True` or `False`
    if all the values returned by `isclose` are `True` or `False`.

    See Also
    --------
    isclose
    """
    return np.all(isclose(*args, **kwargs))
