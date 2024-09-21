from typing import Union
import numpy as np
import awkward as ak
from numba.core import types as nbtypes, cgutils
from numba.extending import (
    typeof_impl,
    models,
    make_attribute_wrapper,
    register_model,
    box,
    unbox,
    NativeValue,
    overload_method,
)
from scipy.sparse import issparse
from scipy.sparse import csr_matrix as csr_scipy, spmatrix

from .utils import get_shape_sp, _jagged_to_csr_data_, count_cols


__all__ = ["csr_matrix"]


SparseLike = Union[spmatrix, np.ndarray, ak.Array]


class csr_matrix:
    """
    Numba-jittable Python class for a sparse matrices in CSR format.
    The meaning of the input variables is the same as in SciPy, and
    object creation follows the same pattern.

    Parameters
    ----------
    data : SparseLike
        Contains the non-zero values of the matrix, in the order in which
        they would be encountered if we walked along the rows left to
        right and top to bottom. If this is a CSC matrix, the walk
        happens along the columns. From version 0.0.8, `Awkward` arrays
        are also accepted.
        .. versionmodified:: 0.0.8
    indices : numpy.ndarray, Optional
        The indices of the columns (rows) during the walk.
        Default is None.
    indptr : numpy.ndarray, Optional
        Stores row (column) boundaries. Default is None.
    shape : Tuple, Optional
        Default is None.

    Note
    ----
    1) At the moment, this class does not support `NumPy`'s array protocoll.
    If you want this to be the argument to a numpy function, use the
    :func:`to_scipy` method of this class.
    2) The attributed 'data', 'indices', 'indptr' and 'shape' are all
    accessible inside Numba-jitted functions.

    Examples
    --------
    Create from a JaggedArray

    >>> import numpy as np
    >>> from sigmaepsilon.math.linalg import JaggedArray, csr_matrix
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> csr = JaggedArray(data, cuts=[3, 3, 4]).to_csr()
    >>> csr
    3x4 CSR matrix of 10 values.

    You can watch it as a NumPy array

    >>> csr.to_numpy()
    array([[ 1.,  2.,  3.,  0.],
           [ 4.,  5.,  6.,  0.],
           [ 7.,  8.,  9., 10.]])

    Create from a SciPy sparse matrix

    >>> from scipy.sparse import csr_matrix as csr_scipy
    >>> scipy_matrix = csr_scipy((3, 4), dtype=np.int8).toarray()
    >>> csr_matrix(scipy_matrix)
    3x4 CSR matrix of 12 values.

    To create the 10 by 10 identity matrix, do this:

    >>> csr_matrix.eye(10)
    10x10 CSR matrix of 10 values.

    You can access rows and row indices of a CSR matrix in Numba
    jitted code, even in 'nopython' mode:

    >>> from numba import jit
    >>> @jit(nopython=True)
    ... def numba_nopython(csr: csr_matrix, i: int):
    ...     return csr.row(i), csr.irow(i)
    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> matrix = csr_scipy((data, (row, col)), shape=(3, 3))
    >>> matrix.toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])
    >>> csr = csr_matrix(matrix)
    >>> numba_nopython(csr, 0)  # doctest: +SKIP
    (array([1., 2.]), array([0, 2]))

    See also
    --------
    :class:`~sigmaepsilon.math.linalg.sparse.jaggedarray.JaggedArray`
    :class:`scipy.sparse.csr_matrix`
    """

    def __init__(
        self,
        data: SparseLike,
        indices: np.ndarray = None,
        indptr: np.ndarray = None,
        shape: tuple = None,
    ):
        if issparse(data):
            data = data.tocsr()
            self.data = data.data.astype(np.float64)
            self.indices = data.indices.astype(np.int32)
            self.indptr = data.indptr.astype(np.int32)
            self.shape = data.shape
        elif isinstance(data, np.ndarray) and indices is None:
            assert (
                len(data.shape) == 2
            ), "If 'data' is a NumPy array, it must be 2 dimensional."
            self.data = data.flatten()
            self.data = self.data.astype(np.float64)
            self.indices = np.tile(np.arange(data.shape[1]), data.shape[0])
            self.indices = self.indices.astype(np.int32)
            self.indptr = np.arange(data.shape[0] + 1) * data.shape[1]
            self.indptr = self.indptr.astype(np.int32)
            self.shape = data.shape
        elif isinstance(data, ak.Array):
            self.data = ak.flatten(data).to_numpy()
            self.data = self.data.astype("float64")
            cc = count_cols(data)
            bi = ak.ArrayBuilder()
            bptr = ak.ArrayBuilder()
            _jagged_to_csr_data_(bi, bptr, cc)
            self.indices = ak.flatten(bi.snapshot()).to_numpy().astype(np.int32)
            self.indptr = ak.flatten(bptr.snapshot()).to_numpy().astype(np.int32)
            self.shape = get_shape_sp(self.indptr)
        else:
            self.data = np.array(data).astype(np.float64)
            self.indices = np.array(indices).astype(np.int32)
            self.indptr = np.array(indptr).astype(np.int32)
            if shape is None:
                shape = get_shape_sp(indptr)
            self.shape = shape

    def to_numpy(self) -> np.ndarray:
        """
        Returns the matrix as a NumPy array.
        .. versionadded:: 0.0.8
        """
        return self.to_scipy().toarray()

    def to_scipy(self) -> csr_scipy:
        """
        Returns data as a `SciPy` object.
        """
        return csr_scipy((self.data, self.indices, self.indptr), shape=self.shape)

    @staticmethod
    def eye(N: int) -> "csr_matrix":
        """
        Returns the NxN identity matrix as a CSR matrix.
        """
        indices = np.arange(N)
        indptr = np.arange(N + 1)
        data = np.ones(N, dtype=float)
        return csr_matrix(data=data, indices=indices, indptr=indptr, shape=(N, N))

    def row(self, i: int = 0) -> np.ndarray:
        """
        Returns the values of the i-th row.

        .. versionmodified:: 0.0.8

        The behavior was changed in version 0.0.8. After that, the
        call only returns the data related to the i-th row. For the
        indices see :func:`irow`.

        .. note::
            This method is available inside Numba-jitted functions,
            even in nopython mode.
        """
        return self.data[self.indptr[i] : self.indptr[i + 1]]

    def irow(self, i: int = 0) -> np.ndarray:
        """
        Returns the colum indices of the values of the i-th row.

        .. versionadded:: 0.0.8

        .. note::
            This method is available inside Numba-jitted functions,
            even in nopython mode.
        """
        return self.indices[self.indptr[i] : self.indptr[i + 1]]

    def __repr__(self):
        N = len(self.data)
        n, m = self.shape
        return f"{n}x{m} CSR matrix of {N} values."


class csr_matrix_nb(nbtypes.Type):  # pragma: no cover
    """Numba type for a sparse matrix."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.data = nbtypes.Array(self.dtype, 1, "C")
        self.indices = nbtypes.Array(nbtypes.int32, 1, "C")
        self.indptr = nbtypes.Array(nbtypes.int32, 1, "C")
        self.shape = nbtypes.UniTuple(nbtypes.int64, 2)
        super(csr_matrix_nb, self).__init__("csr_matrix")


@overload_method(csr_matrix_nb, "row")
def row(csr, i: int):  # pragma: no cover
    if isinstance(csr, csr_matrix_nb):

        def row_impl(csr, i: int):
            return csr.data[csr.indptr[i] : csr.indptr[i + 1]]

        return row_impl


@overload_method(csr_matrix_nb, "irow")
def irow(csr, i: int):  # pragma: no cover
    if isinstance(csr, csr_matrix_nb):

        def irow_impl(csr, i: int):
            return csr.indices[csr.indptr[i] : csr.indptr[i + 1]]

        return irow_impl


@typeof_impl.register(csr_matrix)
def typeof_csr(val, c):  # pragma: no cover
    data = typeof_impl(val.data, c)
    return csr_matrix_nb(data.dtype)


make_attribute_wrapper(csr_matrix_nb, "data", "data")
make_attribute_wrapper(csr_matrix_nb, "indices", "indices")
make_attribute_wrapper(csr_matrix_nb, "indptr", "indptr")
make_attribute_wrapper(csr_matrix_nb, "shape", "shape")


@register_model(csr_matrix_nb)
class csr_model(models.StructModel):  # pragma: no cover
    """Data model for nopython mode."""

    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data),
            ("indices", fe_type.indices),
            ("indptr", fe_type.indptr),
            ("shape", fe_type.shape),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@unbox(csr_matrix_nb)
def unbox_csr(typ, obj, c):  # pragma: no cover
    """Convert a python object to a numba-native structure."""
    data = c.pyapi.object_getattr_string(obj, "data")
    indices = c.pyapi.object_getattr_string(obj, "indices")
    indptr = c.pyapi.object_getattr_string(obj, "indptr")
    shape = c.pyapi.object_getattr_string(obj, "shape")
    matrix = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    matrix.data = c.unbox(typ.data, data).value
    matrix.indices = c.unbox(typ.indices, indices).value
    matrix.indptr = c.unbox(typ.indptr, indptr).value
    matrix.shape = c.unbox(typ.shape, shape).value
    for att in [data, indices, indptr, shape]:
        c.pyapi.decref(att)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(matrix._getvalue(), is_error=is_error)


@box(csr_matrix_nb)
def box_csr(typ, val, c):  # pragma: no cover
    """Convert a numba-native structure to a python object."""
    matrix = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    classobj = c.pyapi.unserialize(c.pyapi.serialize_object(csr_matrix))
    data_obj = c.box(typ.data, matrix.data)
    indices_obj = c.box(typ.indices, matrix.indices)
    indptr_obj = c.box(typ.indptr, matrix.indptr)
    shape_obj = c.box(typ.shape, matrix.shape)
    matrix_obj = c.pyapi.call_function_objargs(
        classobj, (data_obj, indices_obj, indptr_obj, shape_obj)
    )
    return matrix_obj
