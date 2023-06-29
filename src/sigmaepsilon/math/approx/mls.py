import numpy as np
from scipy.special import factorial as fact
from numpy import outer
from collections.abc import Iterable
from numba import njit


from .func import isMLSWeightFunction, ConstantWeightFunction


def moving_least_squares(points, *args, w=None, **kwargs):
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
        f = weighted_least_squares(points, *args, w=w, **kwargs)
        return f(x)

    return inner


def least_squares(*args, **kwargs):
    return weighted_least_squares(*args, **kwargs)


def weighted_least_squares(points, values, *args, deg=1, order=2, w=None, **kwargs):
    """
    Returns a Callable that can be used to approximate over datasets.

    Parameters
    ----------
    points : Iterable
        [[X11, X12, ..., X1d], ..., [Xn1, Xn2, ..., Xnd]]
    values : Iterable
        [[f11, f12, ..., f1r], ..., [fn1, fn2, ..., fnr]]
    deg : int, Optional
        The degree of the approximation. Default is 1.
    dim : int, Optional
        The dimension of the dataset. Default is 1.
    n : int, Optional
        Number of data points per dimension. Default is 20.
    w : MLSWeightFunction, Optional
        A proper weight function. Default is a ConstantWeightFunction.
    order : int, Optional.
        The order of the approximation. Default is 2.

    Returns
    -------
    Callable
        Regression function r(x) = f(x), fdx(x), fdy(x), fdxx(x), fdyy(x), fdxy(x)
        fi([X1, X2, ..., Xd]) = [fi1, fi2,..., fir]

    Note
    ----
    The resulting approximation can have an approximation or regression behaviour,
    depending on the data set and the degree of the polynomial.
    """

    assert isinstance(points, np.ndarray)
    assert isinstance(values, np.ndarray)
    assert points.shape[0] == values.shape[0]
    dim = points.shape[1]
    try:
        rec = values.shape[1]
    except Exception:
        rec = 1

    if isMLSWeightFunction(w):
        assert dim == w.dimension
    else:
        w = ConstantWeightFunction(dim)

    grad = True if order > 0 else False
    hess = True if order > 1 else False
    if grad:
        assert hasattr(w, "gradient")
    if hess:
        grad = True
        assert hasattr(w, "gradient")
        assert hasattr(w, "Hessian")

    def bdx(x):
        return None

    def bdy(x):
        return None

    def bdxx(x):
        return None

    def bdyy(x):
        return None

    def bdxy(x):
        return None

    if deg == 1:
        if dim == 1:

            def b(x):
                return np.array([1, x])

            if grad:

                def bdx(x):
                    return np.array([0, 1])

            if hess:

                def bdxx(x):
                    return np.array([0, 0])

        elif dim == 2:

            def b(x):
                return np.array([1, x[0], x[1]])

            if grad:

                def bdx(x):
                    return np.array([0, 1, 0])

                def bdy(x):
                    return np.array([0, 0, 1])

            if hess:

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

            if grad:

                def bdx(x):
                    return np.array([0, 1, 2 * x[0]])

            if hess:

                def bdxx(x):
                    return np.array([0, 0, 2])

        elif dim == 2:

            def b(x):
                return np.array([1, x[0], x[1], x[0] ** 2, x[1] ** 2, x[0] * x[1]])

            if grad:

                def bdx(x):
                    return np.array([0, 1, 0, 2 * x[0], 0, x[1]])

                def bdy(x):
                    return np.array([0, 0, 1, 0, 2 * x[1], x[0]])

            if hess:

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

    # moment matrix
    nData = points.shape[0]
    k = int(fact(deg + dim) / fact(deg) / fact(dim))
    A = np.zeros([k, k])
    B = np.zeros([k, nData])
    Adx, Ady, Bdx, Bdy = 4 * (None,)
    Adxx, Adyy, Adxy, Bdxx, Bdyy, Bdxy = 6 * (None,)
    if grad:
        Adx = np.zeros([k, k])
        Ady = np.zeros([k, k])
        Bdx = np.zeros([k, nData])
        Bdy = np.zeros([k, nData])
    if hess:
        Adxx = np.zeros([k, k])
        Adyy = np.zeros([k, k])
        Adxy = np.zeros([k, k])
        Bdxx = np.zeros([k, nData])
        Bdyy = np.zeros([k, nData])
        Bdxy = np.zeros([k, nData])

    V = np.zeros([nData, rec])
    for i, (xi, fi) in enumerate(zip(points, values)):
        bi = b(xi)
        wi = w.f(xi)
        Mi = outer(bi, bi)
        A += Mi * wi
        B[:, i] += bi * wi
        if grad:
            gwi = w.g(xi)
            Adx += Mi * gwi[0]
            Ady += Mi * gwi[1]
            Bdx[:, i] += bi * gwi[0]
            Bdy[:, i] += bi * gwi[1]
        if hess:
            Gwi = w.G(xi)
            Adxx += Mi * Gwi[0, 0]
            Adyy += Mi * Gwi[1, 1]
            Adxy += Mi * Gwi[0, 1]
            Bdxx[:, i] += bi * Gwi[0, 0]
            Bdyy[:, i] += bi * Gwi[1, 1]
            Bdxy[:, i] += bi * Gwi[0, 1]
        for r in range(rec):
            V[i, r] = fi[r]

    invA = np.linalg.inv(A)

    def inner(x):
        return mls_approx(
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


def mls_preproc(points, values, deg, w, b, g=True, H=True):
    nData = points.shape[0]
    nDim = points.shape[1]
    nRec = values.shape[1]

    # moment matrix
    k = int(fact(deg + nDim) / fact(deg) / fact(nDim))
    A = np.zeros([k, k])
    B = np.zeros([k, nData])
    Adx, Ady, Bdx, Bdy = None, None, None, None
    Adxx, Adyy, Adxy, Bdxx, Bdyy, Bdxy = None, None, None, None, None, None
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
        bi = b[i]
        wi = w[i]
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


def mls_approx(
    invA,
    B,
    b,
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
    bdx,
    bdy,
    bdxx,
    bdyy,
    bdxy,
    g=True,
    H=True,
):
    gamma = invA @ b
    SHP = gamma @ B
    f = SHP.T @ V
    gammadx = invA @ (bdx - Adx @ gamma)
    gammady = invA @ (bdy - Ady @ gamma)
    fdxx = None
    fdyy = None
    fdxy = None
    if g:
        fdx, fdy = mls_g(gamma, gammadx, gammady, B, Bdx, Bdy, V)
    if H:
        fdxx, fdyy, fdxy = mls_H(
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


@njit(nogil=True, parallel=True, fastmath=True, cache=True)
def mls_g(gamma, gammadx, gammady, B, Bdx, Bdy, V):
    SHPdx = gammadx @ B + gamma @ Bdx
    SHPdy = gammady @ B + gamma @ Bdy
    fdx = SHPdx.T @ V
    fdy = SHPdy.T @ V
    return fdx, fdy


@njit(nogil=True, parallel=True, fastmath=True, cache=True)
def mls_H(
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
):
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
