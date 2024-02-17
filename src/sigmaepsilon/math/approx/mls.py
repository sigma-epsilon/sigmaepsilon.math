from typing import Callable, Tuple, Optional, Union
from typing import Iterable

import numpy as np
from numpy import outer, ndarray
from scipy.special import factorial as fact
from numba import njit

from .func import isMLSWeightFunction, ConstantWeightFunction, MLSWeightFunction


def moving_least_squares(
    points: ndarray,
    values: ndarray,
    *,
    w: Optional[Union[Callable, None]] = None,
    **kwargs,
) -> Callable:
    if not isMLSWeightFunction(w):
        dim = points.shape[1]
        w = ConstantWeightFunction(dim)

    def inner(x):
        if not isinstance(x, np.ndarray):
            if isinstance(x, Iterable):
                x = np.array(x)
            else:
                raise TypeError
        w.core = x
        f = weighted_least_squares(points, values, w=w, **kwargs)
        return f(x)

    return inner


def least_squares(
    points: ndarray,
    values: ndarray,
    *,
    deg: Optional[int] = 1,
    order: Optional[int] = 2,
) -> Callable:
    """
    Given :math:`N` points located at :math:`\mathbf{x}_i` in :math:`\mathbb{R}^d`
    where :math:`i \in [1 \dots N]`. The returned fit function approximates the given
    values :math:`f_i` at :math:`\mathbf{x}_i` in the least-squares sence with the
    error functional

    .. math::
        \sum_{i} \\biggr[ || f \left( \mathbf{x}_i \\right) - f_i || \\biggr] ^2

    where :math:`f` is taken from :math:`\Pi_{m}^d`, the space of polynomials of
    total degree :math:`m` in :math:`d` spatial dimensions.

    Parameters
    ----------
    points: Iterable
        [[X11, X12, ..., X1d], ..., [Xn1, Xn2, ..., Xnd]]
    values: Iterable
        [[f11, f12, ..., f1r], ..., [fn1, fn2, ..., fnr]]
    deg: int, Optional
        The degree of the fit function. Default is 1.
    order: int, Optional.
        The order of the approximation. Default is 2.

    Returns
    -------
    Callable
        Fit function r(x) -> f(x), fdx(x), fdy(x), fdxx(x), fdyy(x), fdxy(x)
        fi([X1, X2, ..., Xd]) = [fi1, fi2,..., fir]

    Note
    ----
    The resulting fit function can have an approximation or regression behaviour,
    depending on the dataset and the degree of the polynomial.
    """
    return weighted_least_squares(points, values, deg=deg, order=order)


def weighted_least_squares(
    points: ndarray,
    values: ndarray,
    *,
    deg: Optional[int] = 1,
    order: Optional[int] = 2,
    w: Optional[Union[Callable, None]] = None,
) -> Callable:
    """
    Returns a Callable that can be used to approximate over datasets.

    Parameters
    ----------
    points: Iterable
        [[X11, X12, ..., X1d], ..., [Xn1, Xn2, ..., Xnd]]
    values: Iterable
        [[f11, f12, ..., f1r], ..., [fn1, fn2, ..., fnr]]
    deg: int, Optional
        The degree of the fit function. Default is 1.
    w: MLSWeightFunction, Optional
        A proper weight function. Default is a `ConstantWeightFunction`.
    order: int, Optional.
        The order of the approximation. Default is 2.

    Returns
    -------
    Callable
        Fit function r(x) -> f(x), fdx(x), fdy(x), fdxx(x), fdyy(x), fdxy(x)
        fi([X1, X2, ..., Xd]) = [fi1, fi2,..., fir]

    Note
    ----
    The resulting fit function can have an approximation or regression behaviour,
    depending on the dataset and the degree of the polynomial.
    """
    if not isinstance(points, ndarray):
        points = np.array(points)

    if not isinstance(values, ndarray):
        values = np.array(values)

    assert isinstance(points, np.ndarray)
    assert isinstance(values, np.ndarray)
    assert points.shape[0] == values.shape[0]

    if len(values.shape) == 1:
        values = values.reshape(len(values), 1)

    dim = points.shape[1]

    if isMLSWeightFunction(w):
        assert dim == w.dimension
    else:
        w = ConstantWeightFunction(dim)

    grad = True if order > 0 else False
    hess = True if order > 1 else False

    if hess:
        grad = True
        assert hasattr(w, "Hessian")

    if grad:
        assert hasattr(w, "gradient")

    b, bdx, bdy, bdxx, bdyy, bdxy = _get_polynomial(deg, dim)

    (invA, V, B, Adx, Ady, Adxx, Adyy, Adxy, Bdx, Bdy, Bdxx, Bdyy, Bdxy) = _mls_preproc(
        points, values, deg, w, b, grad, hess
    )

    def inner(x):
        return _mls_approx(
            invA,
            B,
            b(x),
            V,
            Adx,
            Ady,
            Adxx,
            Adyy,
            Adxy,
            Bdx,
            Bdy,
            Bdxx,
            Bdyy,
            Bdxy,
            bdx(x),
            bdy(x),
            bdxx(x),
            bdyy(x),
            bdxy(x),
            g=grad,
            H=hess,
        )

    return inner


def _get_polynomial(deg: int, dim: int):
    if deg == 1:
        if dim == 1:

            def b(x):
                return np.array([1, x])

            def bdx(x):
                return np.array([0, 1])

            def bdy(x):
                return None

            def bdxx(x):
                return np.array([0, 0])

            def bdyy(x):
                return None

            def bdxy(x):
                return None

        elif dim == 2:

            def b(x):
                return np.array([1, x[0], x[1]])

            def bdx(x):
                return np.array([0, 1, 0])

            def bdy(x):
                return np.array([0, 0, 1])

            def bdxx(x):
                return np.array([0, 0, 0])

            def bdyy(x):
                return np.array([0, 0, 0])

            def bdxy(x):
                return np.array([0, 0, 0])

        else:
            raise NotImplementedError
    elif deg == 2:
        if dim == 1:

            def b(x):
                return np.array([1, x, x**2])

            def bdx(x):
                return np.array([0, 1, 2 * x[0]])

            def bdy(x):
                return None

            def bdxx(x):
                return np.array([0, 0, 2])

            def bdyy(x):
                return None

            def bdxy(x):
                return None

        elif dim == 2:

            def b(x):
                return np.array([1, x[0], x[1], x[0] ** 2, x[1] ** 2, x[0] * x[1]])

            def bdx(x):
                return np.array([0, 1, 0, 2 * x[0], 0, x[1]])

            def bdy(x):
                return np.array([0, 0, 1, 0, 2 * x[1], x[0]])

            def bdxx(x):
                return np.array([0, 0, 0, 2, 0, 0])

            def bdyy(x):
                return np.array([0, 0, 0, 0, 2, 0])

            def bdxy(x):
                return np.array([0, 0, 0, 0, 0, 1])

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return b, bdx, bdy, bdxx, bdyy, bdxy


def _mls_preproc(
    points: ndarray,
    values: ndarray,
    deg: int,
    w: MLSWeightFunction,
    b: Callable,
    g: Optional[bool] = True,
    H: Optional[bool] = True,
) -> Tuple[ndarray]:
    nData = points.shape[0]
    nDim = points.shape[1]
    nRec = values.shape[1]

    # moment matrix
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


def _mls_approx(
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
    g: Optional[bool] = True,
    H: Optional[bool] = True,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    gamma = invA @ b
    f = _mls_f(gamma, B, V)

    fdx, fdy, fdxx, fdyy, fdxy = (None,) * 5

    if g:
        gammadx = invA @ (bdx - Adx @ gamma)
        gammady = invA @ (bdy - Ady @ gamma)
        fdx, fdy = _mls_g(gamma, gammadx, gammady, B, Bdx, Bdy, V)

    if H:
        fdxx, fdyy, fdxy = _mls_H(
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


@njit(nogil=True, cache=True)
def _mls_f(
    gamma: ndarray,
    B: ndarray,
    V: ndarray,
) -> Tuple[ndarray, ndarray]:
    SHP = gamma @ B
    f = SHP.T @ V
    return f


@njit(nogil=True, cache=True)
def _mls_g(
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
def _mls_H(
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
