from numpy import ndarray
import numpy as np
from numba import njit, prange

from ..knn import k_nearest_neighbours
from .. import atleast2d

__all__ = ["MLSApproximator"]


@njit(parallel=True, cache=True)
def _approximate(values: ndarray, neighbours: ndarray, factors: ndarray) -> ndarray:
    nT, nN = neighbours.shape
    nD = values.shape[-1]
    res = np.zeros((nT, nD), dtype=values.dtype)
    for i in prange(nT):
        for j in prange(nD):
            for k in range(nN):
                res[i, j] += values[neighbours[i, k], j] * factors[i, k]
    return res


class MLSApproximator:
    """
    Object oriented, high performance implementation of a specific version of the
    moving least squares method. This implementation is less flexible than the others,
    but performes well for extremely large datasets as well. If you want to experiment
    with the hyperparameters of the MLS as a method, it is suggested to use the other
    ways offered by the library.
    
    Parameters
    ----------
    knn_backend: {"scipy", "sklearn"}, Optional
        The backend to use for the KNN calculation. If None, the default backend
        of the library is used. Default is None.

    Notes
    -----
    1) There is a KNN calculation involved in the process, which is done using either
    scipy or sklearn.

    Examples
    --------
    >>> import numpy as np
    >>> from sigmaepsilon.math.approx import MLSApproximator
    >>> approximator = MLSApproximator(knn_backend="scipy")
    >>> nx, ny, nz = (10, 10, 10)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> z = np.linspace(0, 1, nz)
    >>> xv, yv, zv = np.meshgrid(x, y, z)
    >>> source_points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)
    >>> source_values = np.ones((nx*ny*nz, 3))  # could be anything with a shape of (nx*ny*nz, ...)
    >>> approximator.fit(source_points, source_values, source_points)
    >>> target_points = source_points[:10]  # could be anything with a shape of (..., 3)
    >>> target_values = approximator.approximate(target_points)
    >>> assert np.allclose(target_values, np.ones_like(target_values))
    """

    def __init__(self, **config) -> None:
        self.clean()
        self._config = dict(
            knn_backend=None,
        )
        self._config.update(config)

    def clean(self) -> None:
        """
        Sets the instance to default state.
        """
        self.X_S: ndarray = None
        self.Y: ndarray = None
        self.X_T: ndarray = None
        self.neighbours: ndarray = None
        self.factors: ndarray = None

    def config(self, **kwargs) -> None:
        """
        Updates the configuration of the instance.
        """
        self._config.update(kwargs)

    def fit(
        self,
        X_S: ndarray,
        Y: ndarray,
        X_T: ndarray | None = None,
        **kwargs,
    ) -> None:
        """
        Records and preprocesses the data if necessary.
        """
        self.X_S = X_S
        self.Y = atleast2d(Y, back=True)
        self.X_T = X_T

        self.neighbours: ndarray = None
        if X_T is not None:
            self.neighbours = MLSApproximator._get_neighbours(X_S, X_T, **kwargs)
            self.factors = np.ones_like(self.neighbours) / self.neighbours.shape[-1]

    @staticmethod
    def _get_neighbours(
        X_S: ndarray, X_T, *, k: int | None = None, max_distance: float | None = None
    ) -> ndarray:
        k = 4 if not k else k
        return k_nearest_neighbours(X_S, X_T, k=k, max_distance=max_distance)

    def approximate(self, X: ndarray, **kwargs) -> ndarray:
        """
        Estimates the value of the function at the given points.
        """
        neighbours: ndarray = self.neighbours
        factors: ndarray = self.factors
        if neighbours is None or factors is None:
            neighbours = MLSApproximator._get_neighbours(self.X_S, X, **kwargs)
            factors = np.ones_like(neighbours) / neighbours.shape[-1]
        
        return np.squeeze(_approximate(self.Y, neighbours, factors))
