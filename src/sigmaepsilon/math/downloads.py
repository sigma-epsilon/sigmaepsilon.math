import numpy as np

from sigmaepsilon.core.downloads import download_file, delete_downloads


__all__ = [
    "download_mls_testdata",
    "delete_downloads",
]


def download_mls_testdata() -> np.ndarray:
    """
    Downloads the data of a cloud of points.

    Returns
    -------
    numpy.ndarray
        2 dimensional array of shape (102, 6), which means 102
        points with 6 data per each.

    Example
    --------
    >>> from sigmaepsilon.math.downloads import download_mls_testdata
    >>> download_mls_testdata().shape  # doctest: +SKIP
    (102, 6)

    """
    return np.loadtxt(download_file("mls_testdata.out"))  # pragma: no cover
