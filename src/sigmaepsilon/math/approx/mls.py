from numbers import Number

from numpy import ndarray
import numpy as np
from numba import njit, prange

from ..knn import k_nearest_neighbours
from .. import atleast2d

__all__ = ["MLSApproximator"]


@njit(parallel=True, cache=True)
def _approximate_2d(values: ndarray, neighbours: ndarray, factors: ndarray) -> ndarray:
    nT, nN = neighbours.shape
    nD = values.shape[-1]
    res = np.zeros((nT, nD), dtype=values.dtype)
    for i in prange(nT):
        for j in prange(nD):
            for k in range(nN):
                res[i, j] += values[neighbours[i, k], j] * factors[i, k]
    return res


@njit(parallel=True, cache=True)
def _approximate_nd(
    values: ndarray, neighbours: ndarray, factors: ndarray, out: ndarray
) -> ndarray:
    nT, nN = neighbours.shape
    for i in prange(nT):
        for j in range(nN):
            out[i] += values[neighbours[i, j]] * factors[i, j]


class MLSApproximator:
    """
    Object oriented, high performance implementation of a specific version of the
    moving least squares method. This implementation is less flexible than the others,
    but performes well for extremely large datasets too. If you want to experiment
    with the hyperparameters of the MLS as a method, it is suggested to use the other
    solutions offered by the library.

    Parameters
    ----------
    X_S: ndarray
        The source points.
    Y_S: ndarray, Optional
        The source data.
    knn_backend: {"scipy", "sklearn"}, Optional
        The backend to use for the KNN calculation. If None, the default backend
        of the library is used. Default is None.
    k: int, Optional
        The number of neighbours to consider. Default is None.
    max_distance: Number, Optional
        The maximum distance to consider for the neighbours. Default is None.

    Notes
    -----
    1) There is a KNN calculation involved in the process, which is done using either
    scipy or sklearn.
    2) The target points can be provided when calling the :func:`fit` method to precalculate
    the neighbours and factors. If not provided, the neighbours and factors are calculated
    when the :func:`approximate` method is called.
    3) We use Numba to speed up the approximation, hence the first call might be slower.

    Examples
    --------
    >>> import numpy as np
    >>> from sigmaepsilon.math.approx import MLSApproximator
    >>>
    >>> # prepare the source points
    >>> nx, ny, nz = (10, 10, 10)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> z = np.linspace(0, 1, nz)
    >>> xv, yv, zv = np.meshgrid(x, y, z)
    >>> source_points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)
    >>>
    >>> # prepare the source values
    >>> source_values = np.ones((nx*ny*nz, 3))  # could be anything with a shape of (nx*ny*nz, ...)
    >>>
    >>> # instantiate the approximator
    >>> approximator = MLSApproximator(source_points, source_values, knn_backend="scipy")
    >>>
    >>> # approximate to the target points
    >>> target_points = source_points[:10]  # could be anything with a shape of (..., source_points.shape[-1])
    >>> target_values = approximator.approximate(target_points)
    >>>
    >>> # check the results
    >>> assert np.allclose(target_values, np.ones_like(target_values))

    """

    __slots__ = ["X_S", "Y_S", "_neighbours", "_factors", "_knn_config"]

    def __init__(
        self,
        X_S: ndarray,
        Y_S: ndarray,
        knn_backend: str | None = None,
        k: int | None = None,
        max_distance: Number | None = None,
    ) -> None:
        self.X_S = X_S
        self.Y_S = Y_S
        self._neighbours = None
        self._factors = None
        self._knn_config = dict()

        if isinstance(knn_backend, str):
            self._knn_config["knn_backend"] = knn_backend
        elif knn_backend is not None:  # pragma: no cover
            raise ValueError(
                f"Expected a string for parameter 'knn_backend', got {type(knn_backend)} instead."
            )

        if isinstance(k, int):
            self._knn_config["k"] = k
        elif k is not None:  # pragma: no cover
            raise ValueError(
                f"Expected an integer for parameter 'k', got {type(k)} instead."
            )

        if isinstance(max_distance, Number):
            self._knn_config["max_distance"] = float(max_distance)
        elif max_distance is not None:  # pragma: no cover
            raise ValueError(
                f"Expected a number for parameter 'max_distance', got {type(max_distance)} instead."
            )

    @property
    def neighbours(self) -> ndarray | None:
        """
        Returns the neighbours of the target points. If the neighbours are not
        calculated yet, it returns None.
        """
        return self._neighbours

    @neighbours.setter
    def neighbours(self, val: ndarray) -> None:
        """
        Sets the neighbours of the target points.
        """
        self._neighbours = val

    @property
    def factors(self) -> ndarray | None:
        """
        Returns the factors of the target points. If the factors are not
        calculated yet, it returns None.
        """
        return self._factors

    @factors.setter
    def factors(self, val: ndarray) -> None:
        """
        Sets the factors of the target points.
        """
        self._factors = val

    def _calc_factors_and_neighbours(self, X_S: ndarray, X_T: ndarray) -> None:
        self.neighbours = self._get_neighbours(X_S, X_T, **self._knn_config)
        self.factors = np.ones_like(self.neighbours) / self.neighbours.shape[-1]

    @staticmethod
    def _get_neighbours(
        X_S: ndarray,
        X_T,
        *,
        k: int = 4,
        max_distance: float | None = None,
        knn_backend: str = "scipy",
    ) -> ndarray:
        k = 4 if not k else k
        return k_nearest_neighbours(
            X_S, X_T, k=k, max_distance=max_distance, backend=knn_backend
        )

    def approximate(self, X_T: ndarray) -> ndarray:
        """
        Estimates the value of the function at the given points.

        Parameters
        ----------
        X_T: ndarray
            The target points.
        """
        X_S = atleast2d(self.X_S, back=True)
        X_T = atleast2d(X_T, back=True)
        self._calc_factors_and_neighbours(X_S, X_T)
        neighbours = self.neighbours
        factors = self.factors

        data = self.Y_S
        if len(self.Y_S.shape) == 1:
            data = atleast2d(data, back=True)

        if len(data.shape) == 2:
            res = _approximate_2d(data, neighbours, factors)
        else:
            res = np.zeros((len(neighbours),) + data.shape[1:], dtype=data.dtype)
            _approximate_nd(data, neighbours, factors, out=res)

        return np.squeeze(res)
