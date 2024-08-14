from typing import Callable, Iterable

import numpy as np
from numpy import ndarray

from .functions import isMLSWeightFunction, ConstantWeightFunction
from .ls_poly import _get_polynomial
from .ls_approx import _mls_approx
from .ls_preproc import _mls_preproc

__all__ = ["moving_least_squares", "least_squares", "weighted_least_squares"]


def moving_least_squares(
    points: ndarray | Iterable,
    values: ndarray | Iterable,
    *,
    w: Callable | None = None,
    **kwargs,
) -> Callable:
    """
    Moving least squares approximation. The usage is the same as for the
    :func:`~sigmaepsilon.math.approx.ls.weighted_least_squares` function.
    """
    dim = 1 if len(points.shape) == 1 else points.shape[1]

    if not isMLSWeightFunction(w):
        w = ConstantWeightFunction(dim=dim)

    def inner(x):
        if not isinstance(x, np.ndarray):
            if isinstance(x, Iterable):
                x = np.array(x)
            else:
                if dim > 1:
                    raise TypeError(
                        f"Invalid input type. Expected a numpy array or an iterable, got {type(x)}"
                    )
        w.core = x
        f = weighted_least_squares(points, values, w=w, **kwargs)
        return f(x)

    return inner


def least_squares(
    points: ndarray | Iterable,
    values: ndarray | Iterable,
    *,
    deg: int = 1,
    order: int = 2,
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
    points: ndarray | Iterable,
    values: ndarray | Iterable,
    *,
    deg: int = 1,
    order: int = 2,
    w: Callable | None = None,
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

    assert isinstance(points, ndarray)
    assert isinstance(values, ndarray)

    if not points.shape[0] == values.shape[0]:
        raise ValueError(
            (
                f"The number of points ({len(points)}) does "
                "not match the number of values ({len(values)})."
            )
        )

    if len(values.shape) == 1:
        values = values.reshape(len(values), 1)

    dim = 1 if len(points.shape) == 1 else points.shape[1]

    if isMLSWeightFunction(w):
        assert dim == w.dimension
    else:
        w = ConstantWeightFunction(dim=dim)

    grad = True if order > 0 else False
    hess = True if order > 1 else False

    if hess:
        grad = True
        assert hasattr(w, "Hessian")

    if grad:
        assert hasattr(w, "gradient")

    if dim == 1:
        return _wls_1d(points, values, deg, dim, w, grad, hess)
    elif dim == 2:
        return _wls_2d(points, values, deg, dim, w, grad, hess)
    elif dim == 3:
        return _wls_3d(points, values, deg, dim, w, grad, hess)
    else:
        raise NotImplementedError(
            "Only 1, 2 and 3 dimensional pointclouds are supported"
        )


def _wls_1d(points, values, deg, dim, w, grad, hess):
    b, bdx, bdxx = _get_polynomial(deg, dim)

    (invA, V, B, Adx, Adxx, Bdx, Bdxx) = _mls_preproc(
        points, values, deg, dim, w, b, grad, hess
    )

    def inner(x):
        return _mls_approx(
            invA,
            B,
            b(x),
            V,
            Adx,
            Adxx,
            Bdx,
            Bdxx,
            bdx(x),
            bdxx(x),
            g=grad,
            H=hess,
            dim=dim,
        )

    return inner


def _wls_2d(points, values, deg, dim, w, grad, hess):
    b, bdx, bdy, bdxx, bdyy, bdxy = _get_polynomial(deg, dim)

    (invA, V, B, Adx, Ady, Adxx, Adyy, Adxy, Bdx, Bdy, Bdxx, Bdyy, Bdxy) = _mls_preproc(
        points, values, deg, dim, w, b, grad, hess
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
            dim=dim,
        )

    return inner


def _wls_3d(points, values, deg, dim, w, grad, hess):
    b, bdx, bdy, bdz, bdxx, bdyy, bdzz, bdxy, bdxz, bdyz = _get_polynomial(deg, dim)

    (
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
    ) = _mls_preproc(points, values, deg, dim, w, b, grad, hess)

    def inner(x):
        return _mls_approx(
            invA,
            B,
            b(x),
            V,
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
            bdx(x),
            bdy(x),
            bdz(x),
            bdxx(x),
            bdyy(x),
            bdzz(x),
            bdxy(x),
            bdxz(x),
            bdyz(x),
            g=grad,
            H=hess,
            dim=dim,
        )

    return inner
