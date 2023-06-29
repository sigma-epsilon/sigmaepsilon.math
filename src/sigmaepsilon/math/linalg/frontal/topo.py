# -*- coding: utf-8 -*-
import numpy as np
from numba import jit

from ..sparse.csr import CSR


@jit(nopython=True, nogil=True, fastmath=False, cache=True)
def signed_topo_CSR(conn: CSR) -> CSR:
    return None


@jit(nopython=True, nogil=True)
def signed_topo_flattened(a: np.ndarray, path: np.ndarray):
    a = np.abs(a)
    a_max = np.max(a)
    a_min = np.min(a)
    res = a + 1 - a_min
    found = np.full(a_max + 1 - a_min, 0, dtype=a.dtype)
    for i in range(len(res) - 1, -1, -1):
        if found[a[path[i]] - a_min] == 0:
            found[a[path[i]] - a_min] = 1
            res[path[i]] *= -1
    return res


@jit(nopython=True, nogil=True, cache=True)
def signed_topo_bulk(a: np.ndarray, path: np.ndarray):
    a = np.abs(a)
    a_max = np.max(a)
    a_min = np.min(a)
    res = a + 1 - a_min
    I, J = a.shape
    found = np.full(a_max + 1 - a_min, 0, dtype=a.dtype)
    for i in range(I - 1, -1, -1):
        for j in range(J):
            if found[a[path[i], j] - a_min] == 0:
                found[a[path[i], j] - a_min] = 1
                res[path[i], j] *= -1
    return res


@jit(nopython=True, nogil=True, cache=True)
def unsign_topo(signed: np.ndarray):
    return np.abs(signed) - 1


@jit(nopython=True, nogil=True, cache=True)
def topo_to_adj(topo: np.ndarray):
    nN = np.max(topo) + 1
    res = np.zeros((nN, nN), dtype=topo.dtype)
    nE, nNE = topo.shape
    for iE in range(nE):
        for iNE in range(nNE):
            for jNE in range(nNE):
                res[topo[iE, iNE], topo[iE, jNE]] = 1
                res[topo[iE, jNE], topo[iE, iNE]] = 1
    return res
