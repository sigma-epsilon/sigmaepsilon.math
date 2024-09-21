from typing import Union, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from numba import njit

__cache = True


__all__ = ["linsolve", "reduce", "backsub", "npsolve"]


def linsolve(
    A: np.ndarray,
    B: np.ndarray,
    *,
    presc_bool: np.ndarray = None,
    presc_val: np.ndarray = None,
    method: str = "numpy",
    inplace: bool = False
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
    or something else. However, their implementation relies on JIT-compiled code using Numba, and
    for small and medium sized problems, they should do the job in a reasonable amount of time.

    Returns
    -------
    numpy.ndarray
        The solution with a shape matching input 'B'.
    numpy.ndarray, Optional
        The residual vector, if the solution is constrained.

    Example
    -------
    To solve a simple system of equation:

    >>> from sigmaepsilon.math.linalg.solve import linsolve
    >>> import numpy as np
    >>> A = np.array([[3, 1, 2], [1, 1, 1], [2, 1, 2]], dtype=float)
    >>> B = np.array([11, 6, 10], dtype=float)
    >>> X = linsolve(A, B, method='Gauss-Jordan')
    >>> X.flatten()
    array([1., 2., 3.])

    Of course, in this case you could have used NumPy's solver, but if you want to
    have a few unknowns prescribed, you can make use of the direct solvers presented
    here.

    >>> pb = np.array([False, True, False], dtype=bool)
    >>> pv = np.array([0, 3, 0], dtype=float)
    >>> X, R = linsolve(A, B, method='Gauss-Jordan', presc_bool=pb, presc_val=pv)

    In this case, the first argument is the constrained solution vector

    >>> X.flatten()
    array([1. , 3. , 2.5])

    and the second one is the vector of resudials due to the constraints

    >>> R.flatten()
    array([0. , 0.5, 0. ])

    the meaning of which becomes clear if you compare the image of the constrained
    solution you've just calculated against the input B

    >>> np.ravel(A @ X)  # ravel is used to flatten the column array
    array([11. ,  6.5, 10. ])

    >>> B
    array([11.,  6., 10.])

    """
    # FIXME : check if presc_bool and presc_val have matching shape to A and B
    # FIXME : check if presc_bool is a boolean array
    if method == "numpy":
        assert (
            presc_bool is None
        ), "Method '{}' does not support prescribed values.".format(method)
        return npsolve(A, B)
    elif method in ["Jordan", "Gauss-Jordan"]:
        fnc = _Jordan if method == "Jordan" else _GaussJordan
        try:
            nEQ = len(B)
            if len(B.shape) == 1:
                B = B.reshape((nEQ, 1))
            pre = presc_val is not None
            if not pre:
                presc_bool = np.zeros((nEQ,), dtype=bool)
                presc_val = np.zeros((nEQ,), dtype=np.float64)
            if inplace:
                res = backsub(*fnc(A, B, presc_bool, presc_val), presc_bool, presc_val)
            else:
                res = backsub(
                    *fnc(A.copy(), B.copy(), presc_bool, presc_val),
                    presc_bool,
                    presc_val
                )
            if pre:
                return res
            else:
                return res[0]
        except Exception:
            raise LinAlgError("The matrix is singular!")
    else:
        raise AttributeError("Invalid method name '{}'.".format(method))


def reduce(
    A: np.ndarray,
    B: np.ndarray,
    presc_bool: np.ndarray = None,
    presc_val: np.ndarray = None,
    method="Gauss-Jordan",
    inplace=False,
):
    fnc = None
    if method == "Gauss-Jordan":
        fnc = _GaussJordan
    elif method == "Jordan":
        fnc = _Jordan
    if fnc is not None:
        try:
            if presc_bool is None:
                nEQ = len(B)
                presc_bool = np.zeros((nEQ,), dtype=np.int64)
                presc_val = np.zeros((nEQ,), dtype=np.float64)
            if inplace:
                return fnc(A, B, presc_bool, presc_val)
            else:
                return fnc(A.copy(), B.copy(), presc_bool, presc_val)
        except Exception:
            raise LinAlgError("The matrix is singular!")
    else:
        raise AttributeError("Invalid method name '{}'.".format(method))


@njit(nogil=True, cache=__cache)
def npsolve(A, b):
    return np.linalg.solve(A, b)


@njit(nogil=True, cache=__cache)
def _Jordan(
    A: np.ndarray, B: np.ndarray, presc_bool: np.ndarray, presc_val: np.ndarray
):
    nEQ = len(B)
    for iEQ in range(nEQ):
        if presc_bool[iEQ] == True:
            for iROW in range(iEQ, nEQ):
                B[iROW] -= A[iROW, iEQ] * presc_val[iEQ]
                A[iROW, iEQ] = 0.0
        else:
            pivot = A[iEQ, iEQ]
            if abs(pivot) < 1e-12:
                raise Exception("The matrix is singular!")
            for iROW in range(nEQ):
                factor = A[iROW, iEQ] / pivot
                if iROW == iEQ or abs(factor) < 1e-12:
                    continue
                for iCOL in range(nEQ):
                    A[iROW, iCOL] -= factor * A[iEQ, iCOL]
                B[iROW] -= factor * B[iEQ]
    return A, B


@njit(nogil=True, cache=__cache)
def _GaussJordan(
    A: np.ndarray, B: np.ndarray, presc_bool: np.ndarray, presc_val: np.ndarray
):
    nEQ = len(B)
    for iEQ in range(nEQ):
        if presc_bool[iEQ] == True:
            for iROW in range(iEQ, nEQ):
                B[iROW] -= A[iROW, iEQ] * presc_val[iEQ]
                A[iROW, iEQ] = 0.0
        else:
            if iEQ < nEQ - 1:
                pivot = A[iEQ, iEQ]
                if abs(pivot) < 1e-12:
                    raise Exception("The matrix is singular!")
                for iROW in range(iEQ + 1, nEQ):
                    factor = A[iROW, iEQ] / pivot
                    if abs(factor) < 1e-12:
                        continue
                    for iCOL in range(nEQ):
                        A[iROW, iCOL] -= factor * A[iEQ, iCOL]
                    B[iROW] -= factor * B[iEQ]
    return A, B


@njit(nogil=True, cache=__cache)
def backsub(
    A: np.ndarray, B: np.ndarray, presc_bool: np.ndarray, presc_val: np.ndarray
):
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
