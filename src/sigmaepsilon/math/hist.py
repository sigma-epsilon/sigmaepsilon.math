# -*- coding: utf-8 -*-
from numba import njit, prange
import numpy as np


__all__ = ["histogram"]


def histogram(data, bins, *args, return_edges=False, **kwargs):
    hist, bin_edges = np.histogram(data, bins, *args, **kwargs)
    if return_edges:
        return hist, bin_edges
    else:
        bc = bin_centers(bin_edges)
        return hist, bc


@njit(nogil=True, parallel=True, cache=True)
def bin_centers(bin_edges: np.ndarray):
    res = np.zeros(bin_edges.shape[0] - 1, dtype=bin_edges.dtype)
    for i in prange(res.shape[0]):
        res[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
    return res
