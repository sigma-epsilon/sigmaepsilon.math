from numpy import ndarray


__all__ = ["LSApproximator"]


class LSApproximator:
    """
    A class to perform least-squares approximations on known data.

    Given :math:`N` points located at :math:`\mathbf{x}_i` in :math:`\mathbb{R}^d`
    where :math:`i \in [1 \dots N]`. The returned fit function approximates the given
    values :math:`f_i` at :math:`\mathbf{x}_i` in the least-squares sence with the
    error functional

    .. math::
        \sum_{i} \\biggr[ || f \left( \mathbf{x}_i \\right) - f_i || \\biggr] ^2

    where :math:`f` is taken from :math:`\Pi_{m}^d`, the space of polynomials of
    total degree :math:`m` in :math:`d` spatial dimensions.

    Note
    ----
    The fit function can have an approximation or regression behaviour,
    depending on the dataset and the degree of the polynomial.
    """

    def fit(self, coords: ndarray, values: ndarray, deg: int, order: int) -> None:
        """
        Creates the fitting function.

        Parameters
        ----------
        coords: numpy.ndarray
            The coordinates of the points where data is known as a 2d NumPy array.
        values: numpy.ndarray
            The known data.
        """
        pass

    def approximate(self, coords: ndarray) -> ndarray:
        """
        Approcimates the field variables onto the desired coordinates.

        Parameters
        ----------
        coords: numpy.ndarray
            The coordinates of the target points of approximation.
        """
        pass
