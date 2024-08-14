from typing import Tuple

from numpy import ndarray
from numba import njit


def _mls_approx(*args, dim: int = None, **kwargs):
    if dim == 1:
        return _mls_approx_1d(*args, **kwargs)
    elif dim == 2:
        return _mls_approx_2d(*args, **kwargs)
    elif dim == 3:
        return _mls_approx_3d(*args, **kwargs)
    else:  # pragma: no cover
        raise NotImplementedError(
            "Only 1, 2 and 3 dimensional pointclouds are supported."
        )


def _mls_approx_1d(
    invA: ndarray,
    B: ndarray,
    b: ndarray,
    V: ndarray,
    Adx: ndarray,
    Adxx: ndarray,
    Bdx: ndarray,
    Bdxx: ndarray,
    bdx: ndarray,
    bdxx: ndarray,
    g: bool = True,
    H: bool = True,
) -> Tuple[ndarray, ndarray, ndarray]:
    gamma = invA @ b
    f = _mls_f(gamma, B, V)

    fdx, fdxx = (None,) * 2

    if g:
        gammadx = invA @ (bdx - Adx @ gamma)
        fdx = _mls_g_1d(gamma, gammadx, B, Bdx, V)

    if H:
        fdxx = _mls_H_1d(
            invA,
            bdxx,
            Adx,
            Adxx,
            gamma,
            gammadx,
            B,
            Bdx,
            Bdxx,
            V,
        )

    return f, fdx, fdxx


def _mls_approx_2d(
    invA: ndarray,
    B: ndarray,
    b: ndarray,
    V: ndarray,
    Adx: ndarray,
    Ady: ndarray,
    Adxx: ndarray,
    Adyy: ndarray,
    Adxy: ndarray,
    Bdx: ndarray,
    Bdy: ndarray,
    Bdxx: ndarray,
    Bdyy: ndarray,
    Bdxy: ndarray,
    bdx: ndarray,
    bdy: ndarray,
    bdxx: ndarray,
    bdyy: ndarray,
    bdxy: ndarray,
    g: bool = True,
    H: bool = True,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    gamma = invA @ b
    f = _mls_f(gamma, B, V)

    fdx, fdy, fdxx, fdyy, fdxy = (None,) * 5

    if g:
        gammadx = invA @ (bdx - Adx @ gamma)
        gammady = invA @ (bdy - Ady @ gamma)
        fdx, fdy = _mls_g_2d(gamma, gammadx, gammady, B, Bdx, Bdy, V)

    if H:
        fdxx, fdyy, fdxy = _mls_H_2d(
            invA,
            bdxx,
            bdyy,
            bdxy,
            Adx,
            Ady,
            Adxx,
            Adyy,
            Adxy,
            gamma,
            gammadx,
            gammady,
            B,
            Bdx,
            Bdy,
            Bdxx,
            Bdyy,
            Bdxy,
            V,
        )

    return f, fdx, fdy, fdxx, fdyy, fdxy


def _mls_approx_3d(
    invA: ndarray,
    B: ndarray,
    b: ndarray,
    V: ndarray,
    Adx: ndarray,
    Ady: ndarray,
    Adz: ndarray,
    Adxx: ndarray,
    Adyy: ndarray,
    Adzz: ndarray,
    Adxy: ndarray,
    Adxz: ndarray,
    Adyz: ndarray,
    Bdx: ndarray,
    Bdy: ndarray,
    Bdz: ndarray,
    Bdxx: ndarray,
    Bdyy: ndarray,
    Bdzz: ndarray,
    Bdxy: ndarray,
    Bdxz: ndarray,
    Bdyz: ndarray,
    bdx: ndarray,
    bdy: ndarray,
    bdz: ndarray,
    bdxx: ndarray,
    bdyy: ndarray,
    bdzz: ndarray,
    bdxy: ndarray,
    bdxz: ndarray,
    bdyz: ndarray,
    g: bool = True,
    H: bool = True,
) -> Tuple[
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
]:
    gamma = invA @ b
    f = _mls_f(gamma, B, V)

    fdx, fdy, fdz, fdxx, fdyy, fdzz, fdxy, fdxz, fdyz = (None,) * 9

    if g:
        gammadx = invA @ (bdx - Adx @ gamma)
        gammady = invA @ (bdy - Ady @ gamma)
        gammadz = invA @ (bdz - Adz @ gamma)
        fdx, fdy, fdz = _mls_g_3d(gamma, gammadx, gammady, gammadz, B, Bdx, Bdy, Bdz, V)

    if H:
        fdxx, fdyy, fdzz, fdxy, fdxz, fdyz = _mls_H_3d(
            invA,
            bdxx,
            bdyy,
            bdzz,
            bdxy,
            bdxz,
            bdyz,
            Adx,
            Ady,
            Adz,
            Adxx,
            Adyy,
            Adzz,
            Adxy,
            Adxz,
            Adyz,
            gamma,
            gammadx,
            gammady,
            gammadz,
            B,
            Bdx,
            Bdy,
            Bdz,
            Bdxx,
            Bdyy,
            Bdzz,
            Bdxy,
            Bdxz,
            Bdyz,
            V,
        )

    return f, fdx, fdy, fdz, fdxx, fdyy, fdzz, fdxy, fdxz, fdyz


@njit(nogil=True, cache=True)
def _mls_f(
    gamma: ndarray,
    B: ndarray,
    V: ndarray,
) -> ndarray:
    SHP = gamma @ B
    f = SHP.T @ V
    return f


@njit(nogil=True, cache=True)
def _mls_g_1d(
    gamma: ndarray,
    gammadx: ndarray,
    B: ndarray,
    Bdx: ndarray,
    V: ndarray,
) -> ndarray:
    SHPdx = gammadx @ B + gamma @ Bdx
    fdx = SHPdx.T @ V
    return fdx


@njit(nogil=True, cache=True)
def _mls_g_2d(
    gamma: ndarray,
    gammadx: ndarray,
    gammady: ndarray,
    B: ndarray,
    Bdx: ndarray,
    Bdy: ndarray,
    V: ndarray,
) -> Tuple[ndarray, ndarray]:
    SHPdx = gammadx @ B + gamma @ Bdx
    SHPdy = gammady @ B + gamma @ Bdy
    fdx = SHPdx.T @ V
    fdy = SHPdy.T @ V
    return fdx, fdy


@njit(nogil=True, cache=True)
def _mls_g_3d(
    gamma: ndarray,
    gammadx: ndarray,
    gammady: ndarray,
    gammadz: ndarray,
    B: ndarray,
    Bdx: ndarray,
    Bdy: ndarray,
    Bdz: ndarray,
    V: ndarray,
) -> Tuple[ndarray, ndarray, ndarray]:
    SHPdx = gammadx @ B + gamma @ Bdx
    SHPdy = gammady @ B + gamma @ Bdy
    SHPdz = gammadz @ B + gamma @ Bdz
    fdx = SHPdx.T @ V
    fdy = SHPdy.T @ V
    fdz = SHPdz.T @ V
    return fdx, fdy, fdz


@njit(nogil=True, cache=True)
def _mls_H_1d(
    invA: ndarray,
    bdxx: ndarray,
    Adx: ndarray,
    Adxx: ndarray,
    gamma: ndarray,
    gammadx: ndarray,
    B: ndarray,
    Bdx: ndarray,
    Bdxx: ndarray,
    V: ndarray,
) -> ndarray:
    gammadxx = invA @ (bdxx - Adxx @ gamma - 2 * Adx @ gammadx)
    SHPdxx = gammadxx @ B + 2 * gammadx @ Bdx + gamma @ Bdxx
    fdxx = SHPdxx.T @ V
    return fdxx


@njit(nogil=True, cache=True)
def _mls_H_2d(
    invA: ndarray,
    bdxx: ndarray,
    bdyy: ndarray,
    bdxy: ndarray,
    Adx: ndarray,
    Ady: ndarray,
    Adxx: ndarray,
    Adyy: ndarray,
    Adxy: ndarray,
    gamma: ndarray,
    gammadx: ndarray,
    gammady: ndarray,
    B: ndarray,
    Bdx: ndarray,
    Bdy: ndarray,
    Bdxx: ndarray,
    Bdyy: ndarray,
    Bdxy: ndarray,
    V: ndarray,
) -> Tuple[ndarray, ndarray, ndarray]:
    gammadxx = invA @ (bdxx - Adxx @ gamma - 2 * Adx @ gammadx)
    gammadyy = invA @ (bdyy - Adyy @ gamma - 2 * Ady @ gammady)
    gammadxy = invA @ (bdxy - Adxy @ gamma - Adx @ gammady - Ady @ gammadx)
    SHPdxx = gammadxx @ B + 2 * gammadx @ Bdx + gamma @ Bdxx
    SHPdyy = gammadyy @ B + 2 * gammady @ Bdy + gamma @ Bdyy
    SHPdxy = gammadxy @ B + gammadx @ Bdy + gammady @ Bdx + gamma @ Bdxy
    fdxx = SHPdxx.T @ V
    fdyy = SHPdyy.T @ V
    fdxy = SHPdxy.T @ V
    return fdxx, fdyy, fdxy


@njit(nogil=True, cache=True)
def _mls_H_3d(
    invA: ndarray,
    bdxx: ndarray,
    bdyy: ndarray,
    bdzz: ndarray,
    bdxy: ndarray,
    bdxz: ndarray,
    bdyz: ndarray,
    Adx: ndarray,
    Ady: ndarray,
    Adz: ndarray,
    Adxx: ndarray,
    Adyy: ndarray,
    Adzz: ndarray,
    Adxy: ndarray,
    Adxz: ndarray,
    Adyz: ndarray,
    gamma: ndarray,
    gammadx: ndarray,
    gammady: ndarray,
    gammadz: ndarray,
    B: ndarray,
    Bdx: ndarray,
    Bdy: ndarray,
    Bdz: ndarray,
    Bdxx: ndarray,
    Bdyy: ndarray,
    Bdzz: ndarray,
    Bdxy: ndarray,
    Bdxz: ndarray,
    Bdyz: ndarray,
    V: ndarray,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    gammadxx = invA @ (bdxx - Adxx @ gamma - 2 * Adx @ gammadx)
    gammadyy = invA @ (bdyy - Adyy @ gamma - 2 * Ady @ gammady)
    gammadzz = invA @ (bdzz - Adzz @ gamma - 2 * Adz @ gammadz)
    gammadxy = invA @ (bdxy - Adxy @ gamma - Adx @ gammady - Ady @ gammadx)
    gammadxz = invA @ (bdxz - Adxz @ gamma - Adx @ gammadz - Adz @ gammadx)
    gammadyz = invA @ (bdyz - Adyz @ gamma - Ady @ gammadz - Adz @ gammady)
    SHPdxx = gammadxx @ B + 2 * gammadx @ Bdx + gamma @ Bdxx
    SHPdyy = gammadyy @ B + 2 * gammady @ Bdy + gamma @ Bdyy
    SHPdzz = gammadzz @ B + 2 * gammadz @ Bdz + gamma @ Bdzz
    SHPdxy = gammadxy @ B + gammadx @ Bdy + gammady @ Bdx + gamma @ Bdxy
    SHPdxz = gammadxz @ B + gammadx @ Bdz + gammadz @ Bdx + gamma @ Bdxz
    SHPdyz = gammadyz @ B + gammady @ Bdz + gammadz @ Bdy + gamma @ Bdyz
    fdxx = SHPdxx.T @ V
    fdyy = SHPdyy.T @ V
    fdzz = SHPdzz.T @ V
    fdxy = SHPdxy.T @ V
    fdxz = SHPdxz.T @ V
    fdyz = SHPdyz.T @ V
    return fdxx, fdyy, fdzz, fdxy, fdxz, fdyz
