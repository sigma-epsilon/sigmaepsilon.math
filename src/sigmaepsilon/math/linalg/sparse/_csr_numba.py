"""Numba extension for :class:`~sigmaepsilon.math.linalg.sparse.csr.csr_matrix`.

It makes the class usable inside Numba-jitted functions, and is only
imported if Numba is installed.
"""

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

from .csr import csr_matrix

__all__ = ["csr_matrix_nb"]


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
