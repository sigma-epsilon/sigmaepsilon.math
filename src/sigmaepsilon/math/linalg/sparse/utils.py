import numpy as np
from numpy import ndarray
from numba import njit, prange

__cache = True


def lower_spdata(data: ndarray, rows: ndarray, cols: ndarray):
    inds = np.where(rows >= cols)[0]
    return data[inds], rows[inds], cols[inds]


def upper_spdata(data: ndarray, rows: ndarray, cols: ndarray):
    inds = np.where(cols >= rows)[0]
    return data[inds], rows[inds], cols[inds]


@njit(nogil=True, parallel=True, cache=__cache)
def get_shape_sp(indptr: np.ndarray):
    nE = len(indptr) - 1
    widths = np.zeros(nE, dtype=indptr.dtype)
    for iE in prange(nE):
        widths[iE] = indptr[iE + 1] - indptr[iE]
    return nE, widths.max()


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def count_cols(arr: np.ndarray):
    n = len(arr)
    res = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        res[i] = len(arr[i])
    return res


@njit(nogil=True)
def _jagged_to_csr_data_(indices, indptr, counts):
    indptr.begin_list()
    indptr.integer(0)
    c = 0
    for i in range(len(counts)):
        c += counts[i]
        indptr.integer(c)
        indices.begin_list()
        for j in range(counts[i]):
            indices.integer(j)
        indices.end_list()
    indptr.end_list()
