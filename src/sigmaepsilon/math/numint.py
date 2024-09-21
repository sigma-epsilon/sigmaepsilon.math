import numpy as np
from collections import namedtuple


__all__ = [
    "Quadrature",
    "gauss_points",
    "gauss_points_1d",
    "gauss_points_2d",
    "gauss_points_3d",
]


Quadrature = namedtuple("QuadratureRule", ["inds", "pos", "weight"])


def gauss_points(*args):
    """
    Returns data for numerical integration on an N-dimensional unit square using
    the Gauss-Legendre rule for 1d, 2d and 3d scenarios. The implementation using
    `numpy.polynomial.legendre.leggauss` repeatedly for higher dimensions.

    Parameters
    ----------
    *args : tuple
        An integer for each dimension (see the examples).

    Returns
    -------
    numpy.ndarray
        A NumPy array. For one dimension, this output includes the locations and
        the weights as well, for higher dimensions this only contains locations,
        and the weights are returned as a separate array.
    numpy.ndarray
        The weights as a separate output. Only for dimensions greater than 1.

    Examples
    --------
    For a one dimensional case, there is only one output. The first row of the returned array
    are the locations, the second one are the weights.

    >>> from sigmaepsilon.math.numint import gauss_points
    ...
    >>> gauss_points(2)
    array([[-0.57735027,  0.57735027],
           [ 1.        ,  1.        ]])

    For 2d:

    >>> gauss_points(2, 2)
    (array([[-0.57735027, -0.57735027],
           [-0.57735027,  0.57735027],
           [ 0.57735027, -0.57735027],
           [ 0.57735027,  0.57735027]]), array([1., 1., 1., 1.]))

    For 3d:

    >>> gauss_points(2, 2, 2)
    (array([[-0.57735027, -0.57735027, -0.57735027],
           [-0.57735027, -0.57735027,  0.57735027],
           [-0.57735027,  0.57735027, -0.57735027],
           [-0.57735027,  0.57735027,  0.57735027],
           [ 0.57735027, -0.57735027, -0.57735027],
           [ 0.57735027, -0.57735027,  0.57735027],
           [ 0.57735027,  0.57735027, -0.57735027],
           [ 0.57735027,  0.57735027,  0.57735027]]), array([1., 1., 1., 1., 1., 1., 1., 1.]))

    """
    nD = len(args)
    if nD == 1:
        return gauss_points_1d(args[0])
    elif nD == 2:
        return gauss_points_2d(args)
    elif nD == 3:
        return gauss_points_3d(args)
    else:
        raise NotImplementedError(
            "Currently there is no implementation for dimensions higher than 3."
        )


def gauss_points_1d(NumPoints):
    x, w = np.polynomial.legendre.leggauss(NumPoints)
    v = np.zeros([2, NumPoints])
    v[0, :] = x
    v[1, :] = w
    return v


def gauss_points_2d(NumPoints):
    nGaus = NumPoints[0] * NumPoints[1]
    QuadraturePos = np.zeros((nGaus, 2))
    QuadratureWeight = np.zeros((nGaus))
    quad1 = gauss_points_1d(NumPoints[0])
    quad2 = gauss_points_1d(NumPoints[1])
    g = 0
    for gi in np.nditer(quad1, flags=["external_loop"], order="F"):
        for gj in np.nditer(quad2, flags=["external_loop"], order="F"):
            QuadraturePos[g] = np.array([gi[0], gj[0]])
            QuadratureWeight[g] = gi[1] * gj[1]
            g += 1
    return QuadraturePos, QuadratureWeight


def gauss_points_3d(NumPoints):
    nGaus = NumPoints[0] * NumPoints[1] * NumPoints[2]
    QuadraturePos = np.zeros((nGaus, 3))
    QuadratureWeight = np.zeros((nGaus))
    quad1 = gauss_points_1d(NumPoints[0])
    quad2 = gauss_points_1d(NumPoints[1])
    quad3 = gauss_points_1d(NumPoints[2])
    g = 0
    for gi in np.nditer(quad1, flags=["external_loop"], order="F"):
        for gj in np.nditer(quad2, flags=["external_loop"], order="F"):
            for gk in np.nditer(quad3, flags=["external_loop"], order="F"):
                QuadraturePos[g] = np.array([gi[0], gj[0], gk[0]])
                QuadratureWeight[g] = gi[1] * gj[1] * gk[1]
                g += 1
    return QuadraturePos, QuadratureWeight
