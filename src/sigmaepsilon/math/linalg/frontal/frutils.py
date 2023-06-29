# -*- coding: utf-8 -*-
import numpy as np
from numba import jit


@jit(nopython=True, nogil=True)
def max_frontwidth_bulk(topo: np.ndarray, path: np.ndarray, nDOF: int = 1):
    nV = np.max(np.abs(topo))
    nE, nN = topo.shape
    ndest = np.zeros(nV, dtype=topo.dtype)
    maxwidth = 0
    width = 0
    for i in range(nE):
        iE = path[i]
        for j in range(nN):
            if ndest[abs(topo[iE, j]) - 1] == 0:
                width += nDOF
                ndest[abs(topo[iE, j]) - 1] = 1
        maxwidth = max(maxwidth, width)
        for j in range(nN):
            if topo[iE, j] < 0:
                width -= nDOF
    return maxwidth


@jit(nopython=True, nogil=True)
def max_frontwidth_flattened(topof):
    nV = np.max(np.abs(topof))
    ndest = np.zeros(nV, dtype=topof.dtype)
    maxwidth = 0
    width = 0
    for i in range(len(topof)):
        if ndest[abs(topof[i]) - 1] == 0:
            width += 1
            ndest[abs(topof[i]) - 1] = 1
        maxwidth = max(maxwidth, width)
        if topof[i] < 0:
            width -= 1
    return maxwidth


@jit(nopython=True, nogil=True)
def flatind(row: int, col: int, ncols: int, base: int = 0):
    return base + ncols * row + col


@jit(nopython=True, nogil=True)
def flatsize_sym(n: int):
    return np.int64((n * n - n) / 2 + n + 0.1)


@jit(nopython=True, nogil=True)
def flatind_sym(row: int, col: int, base: int = 0):
    return flatsize_sym(max(row, col) - base) + min(row, col)


@jit(nopython=True, nogil=True)
def flatind_bulk(rows: np.ndarray, cols: np.ndarray, base: int = 0):
    ncols = np.max(cols) - base
    ncase, ndata = cols.shape
    res = np.zeros((ncase, ndata), 0, dtype=rows.dtype)
    for i in range(ncase):
        for j in range(ndata):
            res[i, j] = flatind(rows[i, j], cols[i, j], ncols, base)
    return res


@jit(nopython=True, nogil=True)
def flatind_flattened(rows: np.ndarray, cols: np.ndarray, base: int = 0):
    ncols = np.max(cols) - base
    ndata = len(cols)
    res = np.zeros(ndata, 0, dtype=rows.dtype)
    for i in range(ndata):
        res[i] = flatind(rows[i], cols[i], ncols, base)
    return res
