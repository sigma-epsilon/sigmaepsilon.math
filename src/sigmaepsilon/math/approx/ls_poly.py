import numpy as np


def _get_polynomial(deg: int, dim: int):
    if deg == 1:
        if dim == 1:
            return _get_poly_1_1()
        elif dim == 2:
            return _get_poly_1_2()
        elif dim == 3:
            return _get_poly_1_3()
        else:  # pragma: no cover
            raise NotImplementedError
    elif deg == 2:
        if dim == 1:
            return _get_poly_2_1()
        elif dim == 2:
            return _get_poly_2_2()
        elif dim == 3:
            return _get_poly_2_3()
        else:  # pragma: no cover
            raise NotImplementedError
    else:  # pragma: no cover
        raise NotImplementedError


def _get_poly_1_1():
    def b(x):
        return np.array([1, x])

    def bdx(x):
        return np.array([0, 1])

    def bdxx(x):
        return np.array([0, 0])

    return b, bdx, bdxx


def _get_poly_1_2():
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

    return b, bdx, bdy, bdxx, bdyy, bdxy


def _get_poly_1_3():
    def b(x):
        return np.array([1, x[0], x[1], x[2]])

    def bdx(x):
        return np.array([0, 1, 0, 0])

    def bdy(x):
        return np.array([0, 0, 1, 0])

    def bdz(x):
        return np.array([0, 0, 0, 1])

    def bdxx(x):
        return np.array([0, 0, 0, 0])

    def bdyy(x):
        return np.array([0, 0, 0, 0])

    def bdzz(x):
        return np.array([0, 0, 0, 0])

    def bdxy(x):
        return np.array([0, 0, 0, 0])

    def bdxz(x):
        return np.array([0, 0, 0, 0])

    def bdyz(x):
        return np.array([0, 0, 0, 0])

    return b, bdx, bdy, bdz, bdxx, bdyy, bdzz, bdxy, bdxz, bdyz


def _get_poly_2_1():
    def b(x):
        return np.array([1, x, x**2])

    def bdx(x):
        return np.array([0, 1, 2 * x])

    def bdxx(x):
        return np.array([0, 0, 2])

    return b, bdx, bdxx


def _get_poly_2_2():
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

    return b, bdx, bdy, bdxx, bdyy, bdxy


def _get_poly_2_3():
    def b(x):
        return np.array(
            [
                1,
                x[0],
                x[1],
                x[2],
                x[0] ** 2,
                x[1] ** 2,
                x[2] ** 2,
                x[0] * x[1],
                x[0] * x[2],
                x[1] * x[2],
            ]
        )

    def bdx(x):
        return np.array([0, 1, 0, 0, 2 * x[0], 0, 0, x[1], x[2], 0])

    def bdy(x):
        return np.array([0, 0, 1, 0, 0, 2 * x[1], 0, x[0], 0, x[2]])

    def bdz(x):
        return np.array([0, 0, 0, 1, 0, 0, 2 * x[2], 0, x[0], x[1]])

    def bdxx(x):
        return np.array([0, 0, 0, 0, 2, 0, 0, 0, 0, 0])

    def bdyy(x):
        return np.array([0, 0, 0, 0, 0, 2, 0, 0, 0, 0])

    def bdzz(x):
        return np.array([0, 0, 0, 0, 0, 0, 2, 0, 0, 0])

    def bdxy(x):
        return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    def bdxz(x):
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

    def bdyz(x):
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    return b, bdx, bdy, bdz, bdxx, bdyy, bdzz, bdxy, bdxz, bdyz
