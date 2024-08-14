from typing import Callable, Tuple

import numpy as np
from numpy import outer, ndarray
from scipy.special import factorial as fact

from .functions import MLSWeightFunction


def _mls_preproc(
    points: ndarray,
    values: ndarray,
    deg: int,
    dim: int,
    w: MLSWeightFunction,
    b: Callable,
    g: bool = True,
    H: bool = True,
):
    if dim == 1:
        return _mls_preproc_1d(points, values, deg, w, b, g, H)
    elif dim == 2:
        return _mls_preproc_2d(points, values, deg, w, b, g, H)
    elif dim == 3:
        return _mls_preproc_3d(points, values, deg, w, b, g, H)
    else:  # pragma: no cover
        raise NotImplementedError(
            "Only 1, 2 and 3 dimensional pointclouds are supported."
        )


def _mls_preproc_1d(
    points: ndarray,
    values: ndarray,
    deg: int,
    w: MLSWeightFunction,
    b: Callable,
    g: bool = True,
    H: bool = True,
) -> Tuple[ndarray]:
    nData = points.shape[0]
    nDim = 1 if len(points.shape) == 1 else points.shape[1]
    nRec = 1 if len(values.shape) == 1 else values.shape[1]

    k = int(fact(deg + nDim) / fact(deg) / fact(nDim))
    A = np.zeros([k, k])
    B = np.zeros([k, nData])
    Adx, Bdx = None, None
    Adxx, Bdxx = None, None

    if g:
        Adx = np.zeros([k, k])
        Bdx = np.zeros([k, nData])

    if H:
        Adxx = np.zeros([k, k])
        Bdxx = np.zeros([k, nData])

    V = np.zeros([nData, nRec])
    for i in range(nData):
        xi = points[i]
        fi = values[i]
        bi = b(xi)
        wi = w.f(xi)
        Mi = outer(bi, bi)
        A += Mi * wi
        B[:, i] += bi * wi
        if g:
            gwi = w.g(xi)
            Adx += Mi * gwi[0]
            Bdx[:, i] += bi * gwi[0]
        if H:
            Gwi = w.G(xi)
            Adxx += Mi * Gwi[0, 0]
            Bdxx[:, i] += bi * Gwi[0, 0]
        for r in range(nRec):
            V[i, r] = fi[r]

    invA = np.linalg.inv(A)

    return invA, V, B, Adx, Adxx, Bdx, Bdxx


def _mls_preproc_2d(
    points: ndarray,
    values: ndarray,
    deg: int,
    w: MLSWeightFunction,
    b: Callable,
    g: bool = True,
    H: bool = True,
) -> Tuple[ndarray]:
    nData = points.shape[0]
    nDim = points.shape[1]
    nRec = values.shape[1]

    k = int(fact(deg + nDim) / fact(deg) / fact(nDim))
    A = np.zeros([k, k])
    B = np.zeros([k, nData])
    Adx, Ady, Bdx, Bdy = 4 * (None,)
    Adxx, Adyy, Adxy, Bdxx, Bdyy, Bdxy = 6 * (None,)

    if g:
        Adx = np.zeros([k, k])
        Ady = np.zeros([k, k])
        Bdx = np.zeros([k, nData])
        Bdy = np.zeros([k, nData])

    if H:
        Adxx = np.zeros([k, k])
        Adyy = np.zeros([k, k])
        Adxy = np.zeros([k, k])
        Bdxx = np.zeros([k, nData])
        Bdyy = np.zeros([k, nData])
        Bdxy = np.zeros([k, nData])

    V = np.zeros([nData, nRec])
    for i in range(nData):
        xi = points[i]
        fi = values[i]
        bi = b(xi)
        wi = w.f(xi)
        Mi = outer(bi, bi)
        A += Mi * wi
        B[:, i] += bi * wi
        if g:
            gwi = w.g(xi)
            Adx += Mi * gwi[0]
            Ady += Mi * gwi[1]
            Bdx[:, i] += bi * gwi[0]
            Bdy[:, i] += bi * gwi[1]
        if H:
            Gwi = w.G(xi)
            Adxx += Mi * Gwi[0, 0]
            Adyy += Mi * Gwi[1, 1]
            Adxy += Mi * Gwi[0, 1]
            Bdxx[:, i] += bi * Gwi[0, 0]
            Bdyy[:, i] += bi * Gwi[1, 1]
            Bdxy[:, i] += bi * Gwi[0, 1]
        for r in range(nRec):
            V[i, r] = fi[r]

    invA = np.linalg.inv(A)

    return invA, V, B, Adx, Ady, Adxx, Adyy, Adxy, Bdx, Bdy, Bdxx, Bdyy, Bdxy


def _mls_preproc_3d(
    points: ndarray,
    values: ndarray,
    deg: int,
    w: MLSWeightFunction,
    b: Callable,
    g: bool = True,
    H: bool = True,
) -> Tuple[ndarray]:
    nData = points.shape[0]
    nDim = points.shape[1]
    nRec = values.shape[1]

    k = int(fact(deg + nDim) / fact(deg) / fact(nDim))
    A = np.zeros([k, k])
    B = np.zeros([k, nData])
    Adx, Ady, Adz, Bdx, Bdy, Bdz = 6 * (None,)
    Adxx, Adyy, Adzz, Adxy, Adxz, Adyz, Bdxx, Bdyy, Bdzz, Bdxy, Bdxz, Bdyz = 12 * (
        None,
    )

    if g:
        Adx = np.zeros([k, k])
        Ady = np.zeros([k, k])
        Adz = np.zeros([k, k])
        Bdx = np.zeros([k, nData])
        Bdy = np.zeros([k, nData])
        Bdz = np.zeros([k, nData])

    if H:
        Adxx = np.zeros([k, k])
        Adyy = np.zeros([k, k])
        Adzz = np.zeros([k, k])
        Adxy = np.zeros([k, k])
        Adxz = np.zeros([k, k])
        Adyz = np.zeros([k, k])
        Bdxx = np.zeros([k, nData])
        Bdyy = np.zeros([k, nData])
        Bdzz = np.zeros([k, nData])
        Bdxy = np.zeros([k, nData])
        Bdxz = np.zeros([k, nData])
        Bdyz = np.zeros([k, nData])

    V = np.zeros([nData, nRec])
    for i in range(nData):
        xi = points[i]
        fi = values[i]
        bi = b(xi)
        wi = w.f(xi)
        Mi = outer(bi, bi)
        A += Mi * wi
        B[:, i] += bi * wi
        if g:
            gwi = w.g(xi)
            Adx += Mi * gwi[0]
            Ady += Mi * gwi[1]
            Adz += Mi * gwi[2]
            Bdx[:, i] += bi * gwi[0]
            Bdy[:, i] += bi * gwi[1]
            Bdz[:, i] += bi * gwi[2]
        if H:
            Gwi = w.G(xi)
            Adxx += Mi * Gwi[0, 0]
            Adyy += Mi * Gwi[1, 1]
            Adzz += Mi * Gwi[2, 2]
            Adxy += Mi * Gwi[0, 1]
            Adxz += Mi * Gwi[0, 2]
            Adyz += Mi * Gwi[1, 2]
            Bdxx[:, i] += bi * Gwi[0, 0]
            Bdyy[:, i] += bi * Gwi[1, 1]
            Bdzz[:, i] += bi * Gwi[2, 2]
            Bdxy[:, i] += bi * Gwi[0, 1]
            Bdxz[:, i] += bi * Gwi[0, 2]
            Bdyz[:, i] += bi * Gwi[1, 2]
        for r in range(nRec):
            V[i, r] = fi[r]

    invA = np.linalg.inv(A)

    return (
        invA,
        V,
        B,
        Adx,
        Ady,
        Adz,
        Adxx,
        Adyy,
        Adzz,
        Adxy,
        Adxz,
        Adyz,
        Bdx,
        Bdy,
        Bdz,
        Bdxx,
        Bdyy,
        Bdzz,
        Bdxy,
        Bdxz,
        Bdyz,
    )
