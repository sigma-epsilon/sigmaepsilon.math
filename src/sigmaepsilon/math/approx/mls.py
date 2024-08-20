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
    **config: dict, Optional
        Keyword arguments that configure the instance.
        For the possible options, see the :func:`config` method.

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
    >>> # instantiate the approximator
    >>> approximator = MLSApproximator(knn_backend="scipy")
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
    >>> # fit the approximator
    >>> approximator.fit(source_points)
    >>>
    >>> # approximate to the target points
    >>> target_points = source_points[:10]  # could be anything with a shape of (..., source_points.shape[-1])
    >>> target_values = approximator.approximate(target_points, source_values)
    >>>
    >>> # check the results
    >>> assert np.allclose(target_values, np.ones_like(target_values))
    """

    __slots__ = ["X_S", "X_T", "_neighbours", "_factors", "_config"]

    def __init__(self, **config) -> None:
        self._config = dict()
        self.clean()
        self.config(**config)

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

    def clean(self) -> None:
        """
        Sets the instance to default state.
        """
        self.X_S = None
        self.X_T = None
        self._neighbours = None
        self._factors = None

    def config(
        self,
        knn_backend: str | None = None,
        k: int | None = None,
        max_distance: Number | None = None,
    ) -> None:
        """
        Updates the configuration of the instance.

        Parameters
        ----------
        knn_backend: {"scipy", "sklearn"}, Optional
            The backend to use for the KNN calculation. If None, the default backend
            of the library is used. Default is None.
        k: int, Optional
            The number of neighbours to consider. Default is None.
        max_distance: Number, Optional
            The maximum distance to consider for the neighbours. Default is None.
        """
        if isinstance(knn_backend, str):
            self._config["knn_backend"] = knn_backend
        elif knn_backend is not None:  # pragma: no cover
            raise ValueError(
                f"Expected a string for parameter 'knn_backend', got {type(knn_backend)} instead."
            )

        if isinstance(k, int):
            self._config["k"] = k
        elif k is not None:  # pragma: no cover
            raise ValueError(
                f"Expected an integer for parameter 'k', got {type(k)} instead."
            )

        if isinstance(max_distance, Number):
            self._config["max_distance"] = float(max_distance)
        elif max_distance is not None:  # pragma: no cover
            raise ValueError(
                f"Expected a number for parameter 'max_distance', got {type(max_distance)} instead."
            )

    def fit(
        self,
        X_S: ndarray,
        X_T: ndarray | None = None,
    ) -> None:
        """
        Records and preprocesses the data if necessary.

        Parameters
        ----------
        X_S: ndarray
            The source points.
        X_T: ndarray, Optional
            The target points. Default is None.
        """
        self.X_S = X_S
        self.X_T = X_T
        self.neighbours = None

        if X_T is not None:
            self._calc_factors_and_neighbours(X_S, X_T)

    def _calc_factors_and_neighbours(self, X_S: ndarray, X_T: ndarray) -> None:
        self.neighbours = MLSApproximator._get_neighbours(X_S, X_T, **self._config)
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

    def approximate(self, X: ndarray, data: ndarray) -> ndarray:
        """
        Estimates the value of the function at the given points.

        Parameters
        ----------
        X: ndarray
            The target points.
        data: ndarray
            The data to approximate. The shape of this array must match
            the same of the source points.
        """
        neighbours = self.neighbours
        factors = self.factors

        if neighbours is None or factors is None:
            self._calc_factors_and_neighbours(self.X_S, X)

        neighbours = self.neighbours
        factors = self.factors

        if len(data.shape) == 1:
            data = atleast2d(data, back=True)

        if len(data.shape) == 2:
            res = _approximate_2d(data, neighbours, factors)
        else:
            res = np.zeros((len(neighbours),) + data.shape[1:], dtype=data.dtype)
            _approximate_nd(data, neighbours, factors, out=res)

        return np.squeeze(res)
