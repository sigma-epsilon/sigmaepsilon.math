"""Linear solve routines with support for prescribed values."""

from typing import Union, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from sigmaepsilon.core.jit import njit

__cache = True


__all__ = ["linsolve", "reduce", "backsub", "npsolve"]


def linsolve(
    A: np.ndarray,
    B: np.ndarray,
    *,
    presc_bool: np.ndarray = None,
    presc_val: np.ndarray = None,
    method: str = "numpy",
    inplace: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """
    Solves a linear system of equations using various methods. It supports prescription of values.

    Parameters
    ----------
    A : numpy.ndarray
        The coefficient matrix.
    B : numpy.ndarray
        One more right hand sides.
    presc_bool : numpy.ndarray, Optional
        An 1d NumPy array of booleans to indicate which unknowns are prescribed.
        Default is None.
    presc_val : numpy.ndarray, Optional
        An 1d NumPy array of floats of prescribed values. It should be used together
        with 'presc_bool'. Default is None.
    method : str, Optional
        The method to use. Valid options are:
        * 'numpy' :  uses NumPy's solver for dense scenarios. In this case prescribed
           values are not available. This is the same as to call `numpy.linalg.solve` directly.
        * 'Jordan' : Jordan elimination
        * 'Gauss-Jordan' : Gauss-Jordan elimination
    inplace : bool, Optional
        To allow for in-place modifications of the inputs 'A' and 'B' not. If False, a
        copy is created. Default is False.

    Note
    ----
    Methods 'Jordan' and 'Gauss-Jordan' use elimination to account for constraints on the unknown
    variables, hence yield exact results. These algorithms doesn't scale too well and are not suggested
    in a production enviroment, but they are handy to measure the error of a penalty function approach
    or something else. They use partial pivoting among the equations of the unknowns that are not
    prescribed, and they consider the matrix singular if no pivot is larger than `1e-12` times the
    largest absolute value in 'A'. Their implementation relies on JIT-compiled code using Numba, and
    for small and medium sized problems, they should do the job in a reasonable amount of time.

    Returns
    -------
    numpy.ndarray
        The solution with a shape matching input 'B'.
    numpy.ndarray, Optional
        The residual vector, if the solution is constrained. It has the same shape as 'B'.

    Raises
    ------
    ValueError
        If the shapes of the inputs don't match.
    numpy.linalg.LinAlgError
        If the matrix is singular.

    Example
    -------
    To solve a simple system of equation:

    >>> from sigmaepsilon.math.linalg.solve import linsolve
    >>> import numpy as np
    >>> A = np.array([[3, 1, 2], [1, 1, 1], [2, 1, 2]], dtype=float)
    >>> B = np.array([11, 6, 10], dtype=float)
    >>> linsolve(A, B, method='Gauss-Jordan')
    array([1., 2., 3.])

    Of course, in this case you could have used NumPy's solver, but if you want to
    have a few unknowns prescribed, you can make use of the direct solvers presented
    here.

    >>> pb = np.array([False, True, False], dtype=bool)
    >>> pv = np.array([0, 3, 0], dtype=float)
    >>> X, R = linsolve(A, B, method='Gauss-Jordan', presc_bool=pb, presc_val=pv)

    In this case, the first argument is the constrained solution vector

    >>> X
    array([1. , 3. , 2.5])

    and the second one is the vector of resudials due to the constraints

    >>> R
    array([0. , 0.5, 0. ])

    the meaning of which becomes clear if you compare the image of the constrained
    solution you've just calculated against the input B

    >>> A @ X
    array([11. ,  6.5, 10. ])

    >>> B
    array([11.,  6., 10.])

    """
    if method == "numpy":
        if presc_bool is not None or presc_val is not None:
            raise ValueError(f"Method '{method}' does not support prescribed values.")
        return npsolve(A, B)
    elif method in ["Jordan", "Gauss-Jordan"]:
        fnc = _Jordan if method == "Jordan" else _GaussJordan
        A, B, presc_bool, presc_val, pre = _check_linsolve_input(
            A, B, presc_bool, presc_val
        )
        is_1d = len(B.shape) == 1
        if is_1d:
            B = B.reshape((len(B), 1))
        if not inplace:
            A, B = A.copy(), B.copy()
        tol = _singularity_tolerance(A)
        A, B, info = fnc(A, B, presc_bool, presc_val, tol)
        if info >= 0:
            raise LinAlgError(f"The matrix is singular! (no pivot for unknown {info})")
        X, R = backsub(A, B, presc_bool, presc_val)
        if is_1d:
            X, R = X[:, 0], R[:, 0]
        return (X, R) if pre else X
    else:
        raise AttributeError("Invalid method name '{}'.".format(method))


def _singularity_tolerance(A: np.ndarray) -> float:
    """Return the absolute pivot tolerance, relative to the scale of 'A'."""
    scale = np.abs(A).max() if A.size > 0 else 0.0
    return 1e-12 * scale


def _check_linsolve_input(
    A: np.ndarray,
    B: np.ndarray,
    presc_bool: np.ndarray | None,
    presc_val: np.ndarray | None,
) -> tuple:
    """Validate the inputs of the elimination methods and fill in the defaults."""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    if len(B.shape) not in (1, 2):
        raise ValueError(f"'B' must be a 1d or 2d array, got shape {B.shape}.")
    nEQ = B.shape[0]
    if A.shape != (nEQ, nEQ):
        raise ValueError(
            f"'A' must be a square matrix matching 'B', got shapes {A.shape} "
            f"and {B.shape}."
        )
    pre = presc_bool is not None or presc_val is not None
    if pre:
        if presc_bool is None or presc_val is None:
            raise ValueError("'presc_bool' and 'presc_val' must be used together.")
        presc_bool, presc_val = np.asarray(presc_bool), np.asarray(presc_val)
        if presc_bool.dtype != np.bool_:
            # integer masks of zeros and ones are accepted
            is_mask = np.issubdtype(presc_bool.dtype, np.integer) and np.all(
                (presc_bool == 0) | (presc_bool == 1)
            )
            if not is_mask:
                raise ValueError("'presc_bool' must be a boolean array.")
            presc_bool = presc_bool.astype(np.bool_)
        if presc_bool.shape != (nEQ,) or presc_val.shape != (nEQ,):
            raise ValueError(
                f"'presc_bool' and 'presc_val' must be 1d arrays of length {nEQ}, "
                f"got shapes {presc_bool.shape} and {presc_val.shape}."
            )
        presc_val = presc_val.astype(np.float64)
    else:
        presc_bool = np.zeros((nEQ,), dtype=np.bool_)
        presc_val = np.zeros((nEQ,), dtype=np.float64)
    return A, B, presc_bool, presc_val, pre


def reduce(
    A: np.ndarray,
    B: np.ndarray,
    presc_bool: np.ndarray = None,
    presc_val: np.ndarray = None,
    method="Gauss-Jordan",
    inplace=False,
):
    """Reduce a linear system of equations, with support for prescribed values.

    The reduced system can be solved with :func:`backsub`. See :func:`linsolve`
    for the meaning of the arguments.
    """
    fnc = None
    if method == "Gauss-Jordan":
        fnc = _GaussJordan
    elif method == "Jordan":
        fnc = _Jordan
    if fnc is None:
        raise AttributeError("Invalid method name '{}'.".format(method))
    A_, B_, presc_bool, presc_val, _ = _check_linsolve_input(
        A, B, presc_bool, presc_val
    )
    if inplace and (A_ is not A or B_ is not B):
        raise ValueError("Inplace reduction requires float64 NumPy arrays.")
    if not inplace:
        A_, B_ = A_.copy(), B_.copy()
    shape_B = B_.shape
    B_ = B_.reshape((shape_B[0], -1))  # a view, inplace operation is kept
    A_, B_, info = fnc(A_, B_, presc_bool, presc_val, _singularity_tolerance(A_))
    if info >= 0:
        raise LinAlgError(f"The matrix is singular! (no pivot for unknown {info})")
    return A_, B_.reshape(shape_B)


@njit(nogil=True, cache=__cache)
def npsolve(A, b):
    """Solve a linear system of equations using `numpy.linalg.solve`."""
    return np.linalg.solve(A, b)


@njit(nogil=True, cache=__cache)
def _pivot(A, B, presc_bool, iEQ, tol):
    """Swap the row with the largest pivot into row `iEQ`.

    Only the equations of unknowns that are not prescribed are candidates, the
    equations of the prescribed ones stay in place to give the residuals. Returns
    False if all candidates are smaller than `tol` in absolute value.
    """
    nEQ = A.shape[0]
    iMAX = iEQ
    vMAX = abs(A[iEQ, iEQ])
    for iROW in range(iEQ + 1, nEQ):
        if presc_bool[iROW]:
            continue
        v = abs(A[iROW, iEQ])
        if v > vMAX:
            iMAX, vMAX = iROW, v
    if not vMAX > tol:
        return False
    if iMAX != iEQ:
        for iCOL in range(nEQ):
            tmp = A[iEQ, iCOL]
            A[iEQ, iCOL] = A[iMAX, iCOL]
            A[iMAX, iCOL] = tmp
        for iRHS in range(B.shape[1]):
            tmp = B[iEQ, iRHS]
            B[iEQ, iRHS] = B[iMAX, iRHS]
            B[iMAX, iRHS] = tmp
    return True


@njit(nogil=True, cache=__cache)
def _Jordan(
    A: np.ndarray,
    B: np.ndarray,
    presc_bool: np.ndarray,
    presc_val: np.ndarray,
    tol: float,
):
    nEQ = len(B)
    for iEQ in range(nEQ):
        if presc_bool[iEQ]:
            for iROW in range(iEQ, nEQ):
                B[iROW] -= A[iROW, iEQ] * presc_val[iEQ]
                A[iROW, iEQ] = 0.0
        else:
            if not _pivot(A, B, presc_bool, iEQ, tol):
                return A, B, iEQ
            pivot = A[iEQ, iEQ]
            for iROW in range(nEQ):
                factor = A[iROW, iEQ] / pivot
                if iROW == iEQ or factor == 0.0:
                    continue
                for iCOL in range(nEQ):
                    A[iROW, iCOL] -= factor * A[iEQ, iCOL]
                B[iROW] -= factor * B[iEQ]
    return A, B, -1


@njit(nogil=True, cache=__cache)
def _GaussJordan(
    A: np.ndarray,
    B: np.ndarray,
    presc_bool: np.ndarray,
    presc_val: np.ndarray,
    tol: float,
):
    nEQ = len(B)
    for iEQ in range(nEQ):
        if presc_bool[iEQ]:
            for iROW in range(iEQ, nEQ):
                B[iROW] -= A[iROW, iEQ] * presc_val[iEQ]
                A[iROW, iEQ] = 0.0
        else:
            if not _pivot(A, B, presc_bool, iEQ, tol):
                return A, B, iEQ
            pivot = A[iEQ, iEQ]
            for iROW in range(iEQ + 1, nEQ):
                factor = A[iROW, iEQ] / pivot
                if factor == 0.0:
                    continue
                for iCOL in range(nEQ):
                    A[iROW, iCOL] -= factor * A[iEQ, iCOL]
                B[iROW] -= factor * B[iEQ]
    return A, B, -1


@njit(nogil=True, cache=__cache)
def backsub(
    A: np.ndarray, B: np.ndarray, presc_bool: np.ndarray, presc_val: np.ndarray
):
    """Perform back-substitution to solve a reduced linear system of equations."""
    nEQ, nRHS = B.shape
    X = np.zeros((nEQ, nRHS), dtype=np.float64)
    R = np.zeros((nEQ, nRHS), dtype=np.float64)
    resid = np.zeros(nRHS, dtype=np.float64)
    for iEQ in range(nEQ - 1, -1, -1):
        pivot = A[iEQ, iEQ]
        resid[:] = B[iEQ]
        if iEQ < nEQ - 1:
            for iCOL in range(iEQ + 1, nEQ):
                resid -= A[iEQ, iCOL] * X[iCOL]
        if presc_bool[iEQ] == True:
            X[iEQ] = presc_val[iEQ]
            R[iEQ] = -resid
        else:
            X[iEQ] = resid / pivot
    return X, R
