# -*- coding: utf-8 -*-
import numpy as np
from numba import jit


@jit(nopython=True, nogil=True, cache=True)
def backsub_fr(
    A: np.ndarray,
    B: np.ndarray,
    presc_bool: np.ndarray,
    presc_val: np.ndarray,
    eqpath: np.ndarray,
    glob_to_front: np.ndarray,
    glob_to_width: np.ndarray,
):
    nEQ, maxwidth = A.shape
    nEQ, nRHS = B.shape
    X = np.zeros((nEQ, nRHS), dtype=A.dtype)
    R = np.zeros((nEQ, nRHS), dtype=A.dtype)
    resid = np.zeros(nRHS, dtype=A.dtype)
    X_fr = np.zeros((maxwidth, nRHS), dtype=A.dtype)
    for iEQ in range(nEQ - 1, -1, -1):
        gind = eqpath[iEQ]
        ifront = glob_to_front[gind]
        frwidth = glob_to_width[gind]
        pivot = A[gind, ifront]
        A[gind, ifront] = 0.0
        resid[:] = B[gind]
        for jfront in range(frwidth):
            resid -= A[gind, jfront] * X_fr[jfront]
        if presc_bool[gind] == True:
            X_fr[ifront] = presc_val[gind]
            R[gind] = -resid + pivot * presc_val[gind]
        else:
            X_fr[ifront] = resid / pivot
        X[gind] = X_fr[ifront]
    return X, R
