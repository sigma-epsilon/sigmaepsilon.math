from typing import Iterable

import numpy as np
from numpy import ndarray
from numba import njit, prange

__cache = True


def _tr_tensors2(arr: ndarray, Q: ndarray) -> ndarray:
    shape_arr = arr.shape
    shape_Q = Q.shape
    n_shape_arr = len(shape_arr)
    n_shape_Q = len(shape_Q)
    if n_shape_arr == 2 and n_shape_Q == 2:
        return Q @ arr @ Q.T
    elif n_shape_arr == 3 and n_shape_Q == 2:
        return _tr_tensors2_single(arr, Q)
    elif n_shape_arr >= 4 and n_shape_Q == 2:
        nX = np.prod(shape_arr[1:-2], dtype=int)
        arr = arr.reshape(shape_arr[0], nX, shape_arr[-2], shape_arr[-1])
        res = _tr_tensors2_single2(arr, Q)
        res = res.reshape(shape_arr)
        return res
    elif n_shape_arr == 3 and n_shape_Q == 3:
        return _tr_tensors2_multi(arr, Q)
    elif n_shape_arr == 4 and n_shape_Q == 3:
        return _tr_tensors2_multi2(arr, Q)
    elif n_shape_arr > 4 and n_shape_Q == 3:
        nX = np.prod(shape_arr[1:-2], dtype=int)
        arr = arr.reshape(shape_arr[0], nX, shape_arr[-2], shape_arr[-1])
        res = _tr_tensors2_multi2(arr, Q)
        res = res.reshape(shape_arr)
        return res
    else:
        raise NotImplementedError


@njit(nogil=True, parallel=True, cache=__cache)
def _tr_tensors2_single(arr: ndarray, Q: ndarray) -> ndarray:
    nE = arr.shape[0]
    res = np.zeros_like(arr)
    Q_T = Q.T
    for iE in prange(nE):
        res[iE, :, :] = Q @ arr[iE] @ Q_T
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _tr_tensors2_single2(arr: ndarray, Q: ndarray) -> ndarray:
    nE, nX = arr.shape[:2]
    res = np.zeros_like(arr)
    Q_T = Q.T
    for iE in prange(nE):
        for iX in prange(nX):
            res[iE, iX, :, :] = Q @ arr[iE, iX] @ Q_T
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _tr_tensors2_multi(arr: ndarray, Q: ndarray) -> ndarray:
    nE = arr.shape[0]
    res = np.zeros_like(arr)
    for iE in prange(nE):
        res[iE, :, :] = Q[iE] @ arr[iE] @ Q[iE].T
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _tr_tensors2_multi2(arr: ndarray, Q: ndarray) -> ndarray:
    nE, nX = arr.shape[:2]
    res = np.zeros_like(arr)
    for iE in prange(nE):
        Qi = Q[iE]
        QiT = Qi.T
        for iX in prange(nX):
            res[iE, iX, :, :] = Qi @ arr[iE, iX] @ QiT
    return res


def _tr_tensors4x3(arr: ndarray, Q: ndarray) -> ndarray:
    shape_arr = arr.shape
    shape_Q = Q.shape
    n_shape_arr = len(shape_arr)
    n_shape_Q = len(shape_Q)
    if n_shape_arr == 4 and n_shape_Q == 2:
        return _tr_3333(arr, Q)
    elif n_shape_arr == 5 and n_shape_Q == 2:
        return _tr_3333_multi3(arr, Q)
    elif n_shape_arr == 5 and n_shape_Q == 3:
        return _tr_3333_multi(arr, Q)
    elif n_shape_arr == 6 and n_shape_Q == 3:
        return _tr_3333_multi2(arr, Q)
    elif n_shape_arr > 6 and n_shape_Q == 3:
        nX = np.prod(shape_arr[1:-4], dtype=int)
        arr = arr.reshape(
            shape_arr[0], nX, shape_arr[-4], shape_arr[-3], shape_arr[-2], shape_arr[-1]
        )
        res = _tr_3333_multi2(arr, Q)
        res = res.reshape(shape_arr)
        return res
    else:
        raise NotImplementedError


def _tr_3333_sym(array: Iterable, Q: ndarray) -> ndarray:
    res = np.zeros((3, 3, 3, 3), dtype=array.dtype)
    for p in prange(3):
        for q in prange(3):
            for r in prange(3):
                for s in prange(3):
                    for i in prange(3):
                        for j in prange(3):
                            for k in prange(3):
                                for m in prange(3):
                                    res[i, j, k, m] += (
                                        Q[i, p]
                                        * Q[j, q]
                                        * Q[k, r]
                                        * Q[m, s]
                                        * array[p, q, r, s]
                                    )
    return res


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _tr_3333(array: ndarray, Q: ndarray) -> ndarray:
    res = np.zeros((3, 3, 3, 3), dtype=array.dtype)
    for p in prange(3):
        for q in prange(3):
            for r in prange(3):
                for s in prange(3):
                    for i in prange(3):
                        for j in prange(3):
                            for k in prange(3):
                                for m in prange(3):
                                    res[i, j, k, m] += (
                                        Q[i, p]
                                        * Q[j, q]
                                        * Q[k, r]
                                        * Q[m, s]
                                        * array[p, q, r, s]
                                    )
    return res


@njit(nogil=True, parallel=False, fastmath=True, cache=__cache)
def _tr_3333_out(array: ndarray, Q: ndarray, out: ndarray) -> ndarray:
    for p in prange(3):
        for q in prange(3):
            for r in prange(3):
                for s in prange(3):
                    for i in prange(3):
                        for j in prange(3):
                            for k in prange(3):
                                for m in prange(3):
                                    out[i, j, k, m] += (
                                        Q[i, p]
                                        * Q[j, q]
                                        * Q[k, r]
                                        * Q[m, s]
                                        * array[p, q, r, s]
                                    )


@njit(nogil=True, parallel=True, cache=__cache)
def _tr_3333_multi(arr: ndarray, Q: ndarray) -> ndarray:
    nE = arr.shape[0]
    res = np.zeros_like(arr)
    for iE in prange(nE):
        _tr_3333_out(arr[iE, :, :, :, :], Q[iE], res[iE, :, :, :, :])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _tr_3333_multi2(arr: ndarray, Q: ndarray) -> ndarray:
    nE, nX = arr.shape[:2]
    res = np.zeros_like(arr)
    for iE in prange(nE):
        Qi = Q[iE]
        for iX in prange(nX):
            _tr_3333_out(arr[iE, iX, :, :, :, :], Qi, res[iE, iX, :, :, :, :])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _tr_3333_multi3(arr: ndarray, Q: ndarray) -> ndarray:
    nE = arr.shape[0]
    res = np.zeros_like(arr)
    for iE in prange(nE):
        _tr_3333_out(arr[iE, :, :, :, :], Q, res[iE, :, :, :, :])
    return res
