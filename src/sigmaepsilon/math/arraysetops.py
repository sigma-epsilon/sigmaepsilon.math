import numpy as np
from numpy import ndarray
from awkward import Array as akarray
from numba import njit
from numba.core import types
from numba.typed import Dict
from typing import Union, Any
import awkward as ak

from .utils import flatten2dC
from .utils import count_cols

__cache = True


__all__ = ["unique2d"]


ArrayLike = Union[ndarray, akarray]
i64 = types.int64
i64A = types.int64[:]
i64A2 = types.int64[:, :]


def unique2d(
    arr: ArrayLike | Any,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    """
    Find the unique elements of a 2d array.

    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:

    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array

    Parameters
    ----------
    arr : numpy.ndarray
        Input array.
    return_index : bool, optional
        If True, also return the indices of `ara` that result in the unique array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array that can be used to
        reconstruct `arr`.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `arr`.

    Example
    -------
    >>> from sigmaepsilon.math.arraysetops import unique2d
    >>> import numpy as np
    >>> arr = np.array([[1, 2, 3], [1, 2, 4]], dtype=int)
    >>> unique2d(arr)
    array([1, 2, 3, 4])

    >>> unique2d(arr, return_index=True)
    [array([1, 2, 3, 4]), DictType[int64,array(int64, 2d, A)]<iv=None>({1: [[0 0]
     [1 0]], 2: [[0 1]
     [1 1]], 3: [[0 2]], 4: [[1 2]]})]

    >>> unique2d(arr, return_inverse=True)  # doctest: +SKIP
    [array([1, 2, 3, 4]), array([[0, 1, 2],
           [0, 1, 3]])]

    >>> unique2d(arr, return_counts=True)  # doctest: +SKIP
    [array([1, 2, 3, 4]), array([2, 2, 1, 1])]

    >>> from sigmaepsilon.math.linalg.sparse import JaggedArray
    >>> arr = JaggedArray(np.array([1, 2, 1, 2, 3]), cuts=[2, 3])
    >>> unique2d(arr)
    array([1, 2, 3])

    """
    if isinstance(arr, ndarray):
        unique, counts, inverse, indices = _unique2d_njit(arr)
        res = [
            unique,
        ]
        if return_index:
            res.append(indices)
        if return_inverse:
            res.append(inverse)
        if return_counts:
            res.append(counts)
        return res if len(res) > 1 else res[0]
    elif isinstance(arr, akarray):
        assert arr.ndim == 2, "Only 2 dimensional awkward arrays are supported!"
        flatarr = ak.flatten(arr).to_numpy()
        unique, counts, inverse, indices = _unique1d_njit(flatarr)
        res = [
            unique,
        ]
        if return_index:
            res.append(indices)
        if return_inverse:
            res.append(ak.unflatten(inverse, count_cols(arr)))
        if return_counts:
            res.append(counts)
        return res if len(res) > 1 else res[0]
    else:
        if hasattr(arr, "is_jagged"):
            if not arr.is_jagged():
                return unique2d(
                    arr.to_numpy(),
                    return_index=return_index,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            else:
                return unique2d(
                    arr.to_ak(),
                    return_index=return_index,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
    raise TypeError(f"Invalid input type {type(arr)}")


@njit(nogil=True, parallel=False, fastmath=False, cache=__cache)
def _unique1d_njit(flatdata: ndarray):
    unique = np.unique(flatdata)
    nF = len(flatdata)
    nU = len(unique)
    counts = np.zeros(nU, dtype=np.int64)
    inverse = np.zeros(nF, dtype=np.int64)
    imap = Dict.empty(key_type=i64, value_type=i64)
    for i in range(nU):
        imap[unique[i]] = i
    for i in range(nF):
        counts[imap[flatdata[i]]] += 1
        inverse[i] = imap[flatdata[i]]
    indices = Dict.empty(key_type=i64, value_type=i64A)
    for i in range(nU):
        indices[unique[i]] = np.zeros((counts[i]), dtype=np.int64)
    counts[:] = 0
    for iF in range(nF):
        i = inverse[iF]
        indices[unique[i]][counts[i]] = iF
        counts[i] += 1
    return unique, counts, inverse, indices


@njit(nogil=True, parallel=False, fastmath=False, cache=__cache)
def _unique2d_njit(data: ndarray):
    flatdata = flatten2dC(data)
    unique = np.unique(flatdata)
    nF = len(flatdata)
    nU = len(unique)
    counts = np.zeros(nU, dtype=np.int64)
    inverse = np.zeros(nF, dtype=np.int64)
    imap = Dict.empty(key_type=i64, value_type=i64)
    for i in range(nU):
        imap[unique[i]] = i
    for i in range(nF):
        counts[imap[flatdata[i]]] += 1
        inverse[i] = imap[flatdata[i]]
    inverse = inverse.reshape(data.shape)
    indices = Dict.empty(key_type=i64, value_type=i64A2)
    for i in range(nU):
        indices[unique[i]] = np.zeros((counts[i], 2), dtype=np.int64)
    counts[:] = 0
    nR, nC = data.shape
    for r in range(nR):
        for c in range(nC):
            i = inverse[r, c]
            indices[unique[i]][counts[i], 0] = r
            indices[unique[i]][counts[i], 1] = c
            counts[i] += 1
    return unique, counts, inverse, indices
