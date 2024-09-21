from typing import Iterable, Callable, Any
from copy import deepcopy as dcopy

import numpy as np
from numpy import ndarray

from sigmaepsilon.core.typing import issequence

from .utils import (
    normalize_frame,
    Gram,
    dual_frame,
    transpose_axes,
    show_frame,
    rotation_matrix,
)
from .logical import (
    is_rectangular_frame,
    is_orthonormal_frame,
    is_normal_frame,
)
from ..utils import repeat
from .meta import FrameLike, TensorLike, ArrayWrapper


__all__ = ["ReferenceFrame", "RectangularFrame", "CartesianFrame"]


def inplace_binary(
    obj: FrameLike, other: Any, bop: Callable, rtype: FrameLike | None = None
) -> FrameLike:
    """
    Performs a binary operation inplace. Components of registered tensorial entities
    are transformed accordingly and registered to the output frame. This is important because
    the operation takes effect inplace, the resulting instance can be new in the case of
    having to change the type of the object (eg. from CartesianFrame to RectangularFrame).
    """
    axes = np.copy(obj.show())
    new_axes = np.zeros_like(axes)
    bop(axes, other, out=(new_axes,))
    if rtype:
        result = rtype(axes)
        tensors = list(obj._weakrefs.values())
        for t in tensors:
            t.frame = result
        result.axes = new_axes
    else:
        obj.axes = new_axes
        result = obj
    return result


def out_of_place_binary(
    obj: FrameLike, other: Any, bop: Callable, rtype: FrameLike | None = None
) -> FrameLike:
    """
    Performs a binary operation inplace. Components of registered tensorial entities
    are transformed accordingly and registered to the output frame. This is important because
    the operation takes effect inplace, the resulting instance can be new in the case of
    having to change the type of the object (eg. from CartesianFrame to RectangularFrame).
    """
    axes = np.copy(obj.show())
    bop(axes, other, out=(axes,))
    rtype = rtype if rtype else obj.__class__
    result = rtype(axes)
    return result


class ReferenceFrame(FrameLike):
    """
    A class for arbitrary reference frames, that facilitates transformation of tensor-like
    quantities. Instances of this class support NumPy's functions, universal
    functions and other standard features of NumPy (see the notes below).

    An important feature of the class is that it maintains the property of objectivity
    of the tensorial quantities that use instances of this class in their representation.
    Upon transformation of a frame, the components of the associated tensorial quantities
    transform correspondingly.

    Parameters
    ----------
    axes: numpy.ndarray, Optional
        2d numpy array of floats specifying cartesian reference frames.
        Default is None.
    dim: int, Optional
        Dimension of the mesh. Default is 3.
    name: str, Optional
        The name of the frame. Default is None.

    Notes
    -----
    1) The object is able to handle any reference frame, not just cartesian ones.
    The only restriction is that the base vectors should be linearly independent.
    2) All NumPy universal functions return ReferenceFrame instances. If for a NumPy
    universal function a ReferenceFrame instance is provided by the parameter 'out'
    (eg. numpy.sqrt(A, out=A)), the instance is modified inplace and all associated
    tensorial quantities change aggordingly. This also true for operators like +=,
    -=, @=, etc. that naturally cause inplace modification of the called instance.

    Examples
    --------
    Define a frame and rotate it around axis 'Z' with an amount of 180 degrees:

    >>> from sigmaepsilon.math.linalg import ReferenceFrame
    >>> import numpy as np
    >>> A = ReferenceFrame(dim=3)
    >>> B = A.orient_new('Space', [0, 0, np.pi], 'XYZ')

    If we define a tensorial quantity with components in a frame, the
    components of the quantity change if the frame changes:

    >>> from sigmaepsilon.math.linalg import Vector
    >>> v = Vector([1., 0, 0], frame=A)
    >>> A *= 2.0
    >>> v
    Array([0.5, 0. , 0. ])

    A ReferenceFrame instance can be an argument to a NumPy universal function, but the
    result is always a simple array object. The following call results in a simple array,
    leaving the frame (and hence the associated tensors, if there are any) unchanged

    >>> A = ReferenceFrame(dim=3)
    >>> np.sqrt(A)
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    but this one changes the frame (and hence the associated tensors if there are any):

    >>> A = ReferenceFrame(dim=3)
    >>> np.sqrt(A, out=A)
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Of course, if you want your result to be a ReferenceFrame instance, you can always
    do this:

    >>> A = ReferenceFrame(np.sqrt(ReferenceFrame(dim=3)))

    The basis vectors that make up a frame are accessible through the property 'axes':

    >>> A = ReferenceFrame(dim=3)
    >>> A.axes
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    If the basis vectors change, so do the tensors associated with the frame:

    >>> import numpy as np
    >>> v = Vector([1.0, 0, 0], frame=A)
    >>> A.axes = np.eye(3) * 2.0
    >>> v
    Array([0.5, 0. , 0. ])

    The same happens if we change the frame of a tensorial quantity:

    >>> A = ReferenceFrame(dim=3)
    >>> v = Vector([1., 0, 0], frame=A)
    >>> v.frame = A * 2.0
    >>> v
    Array([0.5, 0. , 0. ])

    To get the matrix that transforms quantities from frame A to frame B,
    use the `dcm` method:

    >>> source = ReferenceFrame(dim=3)
    >>> target = source.orient_new('Space', [0, 0, 90*np.pi/180],  'XYZ')
    >>> DCM = source.dcm(target=target)

    or equivalenty

    >>> DCM = target.dcm(source=source)

    See Also
    --------
    :class:`~sigmaepsilon.math.linalg.RectangularFrame`
    :class:`~sigmaepsilon.math.linalg.CartesianFrame`
    """

    def __init__(self, axes: ndarray = None, *args, name: str = None, dim: int = None):
        try:
            if not isinstance(axes, ndarray):
                if isinstance(dim, Iterable):
                    if len(dim) == 1:
                        axes = np.eye(dim[0])
                    elif len(dim) == 2:
                        axes = repeat(np.eye(dim[-1]), dim[0])
                    else:
                        raise NotImplementedError
                elif isinstance(dim, int):
                    axes = np.eye(dim)
        except Exception as e:  # pragma: no cover
            raise e
        super().__init__(axes, *args)
        self._name = name

    @property
    def is_rectangular(self) -> bool:
        """
        Returns True if the frame is a rectangular one.
        """
        return bool(is_rectangular_frame(self.axes))

    @property
    def is_cartesian(self) -> bool:
        """
        Returns True if the frame is a cartesian (orthonormal) one.
        """
        return bool(is_orthonormal_frame(self.axes))

    @property
    def is_independent(self) -> bool:
        """
        Returns True if the base vectors that make up the frame are linearly
        independent.
        """
        return np.linalg.det(self.Gram()) > 0

    @classmethod
    def eye(cls, *args, dim=3, **kwargs) -> "ReferenceFrame":
        """
        Returns a standard orthonormal frame.

        Returns
        -------
        ReferenceFrame
        """
        if len(args) > 0 and isinstance(args[0], int):
            dim = args[0]
        return cls(np.eye(dim), *args, **kwargs)

    def Gram(self) -> ndarray:
        """
        Returns the Gram-matrix of the frame.
        """
        return Gram(self.show())

    def metric_tensor(self) -> TensorLike:
        """
        Returns the metric tensor of the frame.
        """
        from .tensor import Tensor

        return Tensor(self.Gram(), frame=self)

    def volume(self) -> float:
        """
        Returns the signed volume of the general parallelepiped described by the
        base vectors that make up the frame.
        """
        return np.sqrt(np.linalg.det(self.Gram()))

    def dual(self) -> "ReferenceFrame":
        """
        Returns the dual (or reciprocal) frame.
        """
        return self.__class__(dual_frame(self.show()))

    @property
    def name(self) -> str:
        """
        Returns the name of the frame.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        """
        Returns the name of the frame.
        """
        if isinstance(value, str):
            self._name = value
        else:
            raise TypeError("Name must be a string.")

    @property
    def axes(self) -> ndarray:
        """
        Returns a matrix, where each row (or column) is the component array
        of a basis vector with respect to the ambient frame.

        Returns
        -------
        numpy.ndarray
        """
        return self._array

    @axes.setter
    def axes(self, value: Iterable):
        """
        Sets the array of the frame.
        """
        if isinstance(value, np.ndarray):
            buf = value
        else:
            if not issequence(value):
                raise TypeError("'value' must be some kind of iterable")
            buf = np.array(value).astype(float)
        value = self._array_cls_(shape=buf.shape, buffer=buf, dtype=buf.dtype)
        if value.shape == self._array.shape:
            if self._weakrefs and len(self._weakrefs) > 0:
                target = ReferenceFrame(value)
                dcm = self.dcm(target=target)
                for v in self._weakrefs.values():
                    arr = v.show(dcm=dcm)
                    v.array = arr
            self._array = value
        else:
            raise ValueError("Mismatch in data dimensinons!")

    def show(self, target: "ReferenceFrame" = None) -> ndarray:
        """
        Returns the components of the current frame in a target frame.
        If the target is None, the componants are returned in the ambient frame.

        Returns
        -------
        numpy.ndarray
        """
        return self.dcm(source=target)

    def dcm(
        self, *, target: "ReferenceFrame" = None, source: "ReferenceFrame" = None
    ) -> ndarray:
        """
        Returns the direction cosine matrix (DCM) of a transformation
        from a source (S) to a target (T) frame. The current frame can be
        the source or the target, depending on the arguments.

        If called without arguments, it returns the DCM matrix from the
        ambient frame to the current frame (S=None, T=self).

        If `source` is not `None`, then T=self.

        If `target` is not `None`, then S=self.

        Parameters
        ----------
        source: 'ReferenceFrame', Optional
            Source frame. Default is None.
        target: 'ReferenceFrame', Optional
            Target frame. Default is None.

        Returns
        -------
        numpy.ndarray
            DCM matrix from S to T.

        Example
        -------
        >>> from sigmaepsilon.math.linalg import ReferenceFrame
        >>> import numpy as np
        >>> source = ReferenceFrame(dim=3)
        >>> target = source.orient_new('Space', [0, 0, 90*np.pi/180],  'XYZ')
        >>> DCM = source.dcm(target=target)
        >>> arr_source = np.array([3 ** 0.5 / 2, 0.5, 0])
        >>> arr_target = DCM @ arr_source

        """
        if source is not None:
            S, T = source.dcm(), self.dual().dcm()
            return T @ transpose_axes(S)
        elif target is not None:
            S, T = self.dcm(), target.dual().dcm()
            if len(S.shape) == 3:
                return T @ transpose_axes(S)
            elif len(S.shape) == 2:
                return T @ transpose_axes(S)
            else:  # pragma: no cover
                msg = (
                    "There is no transformation rule implemented for"
                    " source shape {} and target shape {}"
                )
                raise NotImplementedError(msg.format(S.shape, T.shape))
        # We only get here if the function is called without arguments.
        # The DCM from the ambient frame to the current frame is returned.
        return self.axes

    def transpose(self, inplace: bool = False) -> "ReferenceFrame":
        """
        Either transposes the array of the frame, or returns a copy
        of it with the components transposed.

        Parameters
        ----------
        inplace: bool, Optional
            If ``True``, the operation is performed on the instance the call
            is made upon. Default is False.

        Note
        ----
        The rule of transposition differs from the one implemented in NumPy, as
        only tensorial axes are being transposed.
        """
        if inplace:
            self.axes = transpose_axes(self.axes)
            return self
        else:
            return self.__class__(transpose_axes(self.axes))

    def orient(self, *args, **kwargs) -> "ReferenceFrame":
        """
        Orients the current frame inplace. All arguments are forwarded
        to :func:`~sigmaepsilon.math.linalg.utils.rotation_matrix`, see there for the
        details.

        Returns
        -------
        ReferenceFrame

        Example
        -------
        Define a standard Cartesian frame and rotate it around axis 'Z'
        with 180 degrees:

        >>> from sigmaepsilon.math.linalg import ReferenceFrame
        >>> import numpy as np
        >>> A = ReferenceFrame(dim=3)
        >>> A.orient('Space', [0, 0, np.pi], 'XYZ')
        Array([[-1.0000000e+00,  1.2246468e-16,  0.0000000e+00],
               [-1.2246468e-16, -1.0000000e+00,  0.0000000e+00],
               [ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])

        See Also
        --------
        :func:`orient_new`
        :func:`~sigmaepsilon.math.linalg.utils.rotation_matrix`
        """
        dcm = rotation_matrix(*args, **kwargs)
        self.axes = show_frame(transpose_axes(dcm), self.axes)
        return self

    def orient_new(self, *args, name="", **kwargs) -> "ReferenceFrame":
        """
        Returns a new frame, oriented relative to the called object.
        All extra positional and keyword arguments are forwarded to
        :func:`~sigmaepsilon.math.linalg.utils.rotation_matrix`, see there for the
        details.

        Parameters
        ----------
        name: str
            Name for the new reference frame.

        Returns
        -------
        ReferenceFrame
            A new ReferenceFrame object.

        See Also
        --------
        :func:`orient`
        :func:`~sigmaepsilon.math.linalg.utils.rotation_matrix`

        Example
        -------
        Define a standard Cartesian frame and rotate it around axis 'Z'
        with 180 degrees:

        >>> A = ReferenceFrame(dim=3)
        >>> B = A.orient_new('Space', [0, 0, np.pi], 'XYZ')

        """
        result = self.deepcopy(name=name)
        if (len(args) + len(kwargs)) == 0:
            return result
        else:
            result.orient(*args, **kwargs)
            return result

    def rotate(self, *args, inplace: bool = True, **kwargs) -> "ReferenceFrame":
        """
        Alias for `orient` and `orient_new`, all extra arguments are forwarded.

        .. versionadded:: 0.0.4

        Parameters
        ----------
        inplace: bool, Optional
            If True, transformation is carried out on the instance the function call
            is made upon, otherwise a new frame is returned. Default is True.

        Returns
        -------
        ReferenceFrame
            An exisitng (if inplace is `True`) or a new ReferenceFrame object.

        Example
        -------
        Define a standard Cartesian frame and rotate it around axis 'Z'
        with 180 degrees:

        >>> from sigmaepsilon.math.linalg import ReferenceFrame
        >>> import numpy as np
        >>> A = ReferenceFrame(dim=3)
        >>> A.rotate('Space', [0, 0, np.pi], 'XYZ')
        Array([[-1.0000000e+00,  1.2246468e-16,  0.0000000e+00],
               [-1.2246468e-16, -1.0000000e+00,  0.0000000e+00],
               [ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])

        See Also
        --------
        :func:`orient`
        :func:`orient_new`
        """
        if inplace:
            return self.orient(*args, **kwargs)
        else:
            return self.orient_new(*args, **kwargs)

    def rotate_new(self, *args, **kwargs) -> "ReferenceFrame":
        """
        Alias for `orient_new`, all extra arguments are forwarded.

        .. versionadded:: 0.0.4

        Returns
        -------
        ReferenceFrame
            A new ReferenceFrame object.

        See Also
        --------
        :func:`orient_new`
        """
        return self.orient_new(*args, **kwargs)

    def __imul__(self, other) -> "ReferenceFrame":
        return inplace_binary(self, other, np.multiply)

    def __imatmul__(self, other) -> "ReferenceFrame":
        return inplace_binary(self, other, np.matmul)

    def __iadd__(self, other) -> "ReferenceFrame":
        return inplace_binary(self, other, np.add)

    def __isub__(self, other) -> "ReferenceFrame":
        return inplace_binary(self, other, np.subtract)

    def __itruediv__(self, other) -> "ReferenceFrame":
        return inplace_binary(self, other, np.divide)

    def __ipow__(self, other) -> "ReferenceFrame":
        return inplace_binary(self, other, np.power)

    def __array_function__(self, func, types, args, kwargs):
        arrs = [arg._array if isinstance(arg, ArrayWrapper) else arg for arg in args]
        return func(*arrs, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Reimplemented to ensure that the result is always a general reference frame.
        This is necessary, because it cannot be guaranteed, that the resulting
        object will admit any regularities after manipulation.
        """
        out = kwargs.get("out", ())

        handled_types = self._HANDLED_TYPES_ + (ReferenceFrame,)
        for x in inputs + out:
            if not isinstance(x, handled_types):
                msg = f"Invalid type {type(x)} encountered at {ufunc}."
                raise TypeError(msg)

        # mark reference frames among outputs and create backup
        if out:
            refs = []
            for x in out:
                if isinstance(x, ReferenceFrame):
                    if not type(x) == ReferenceFrame:
                        msg = (
                            f"Only instances of {ReferenceFrame} can be modified inplace,"
                            + " as no regularity can be guaranteed to hold."
                        )
                        raise NotImplementedError(msg)
                    else:
                        refs.append(True)
                else:
                    refs.append(False)

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x._array if isinstance(x, ArrayWrapper) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(
                np.copy(x._array) if isinstance(x, ArrayWrapper) else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # modify registered tensorial instances of refrence frames
        if out:
            out_ = kwargs["out"]
            for i, x in enumerate(out):
                if refs[i]:
                    out[i].axes = out_[i]
                else:
                    out[i][...] = out_[i]

        if isinstance(result, tuple):
            # multiple return values
            return (ReferenceFrame(x) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            return ReferenceFrame(result)

    def copy(self, deep: bool = False, name: str = None) -> "ReferenceFrame":
        """
        Returns a shallow or deep copy of this object, depending of the
        argument `deepcopy` (default is False).
        """
        if deep:
            return self.__class__(dcopy(self.axes), name=name)
        else:
            return self.__class__(self.axes, name=name)

    def deepcopy(self, name: str = None) -> "ReferenceFrame":
        """
        Returns a deep copy of the frame.
        """
        return self.copy(deep=True, name=name)


class RectangularFrame(ReferenceFrame):
    """
    A class for rectangular reference frames. The behaviour of a RectanguarFrame
    instance is similar to that of a ReferenceFrame, with minor differences. One is
    that the rectangular property is utilized wherever possible, resulting in slightly
    better performance for some operations. The downside is that operations causing
    inplace modifications are only permitted if it can be guaranteed, that the result is
    a valid ReferenceFrame object. This is generally not true for function calls like
    `numpy.sqrt(A, out=A)` and an exception is raised. However, operators like +=, -=
    are fine.

    Note
    ----
    A frame being rectangular only implies that the base vectors that make up the
    frame are mutually perpendicular, but it has no say on the length of the base
    vectors. That is, a rectangular frame is not necessarily orthonormal.

    Examples
    --------
    For operators *= and /= the type of the instance is unchanged

    >>> from sigmaepsilon.math.linalg import RectangularFrame
    >>> A = RectangularFrame(dim=3)
    >>> A *= 2
    >>> type(A)
    <class 'sigmaepsilon.math.linalg.frame.RectangularFrame'>

    but for other operations the type of the object is cast to a more general
    frame

    >>> A += 2
    >>> type(A)
    <class 'sigmaepsilon.math.linalg.frame.ReferenceFrame'>

    See also
    --------
    :class:`~sigmaepsilon.math.linalg.ReferenceFrame`
    :class:`~sigmaepsilon.math.linalg.CartesianFrame`
    """

    def __init__(self, *args, assume_rectangular: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if not assume_rectangular:
            assert is_rectangular_frame(
                self.axes
            ), "This frame is not rectangular, check your input!"

    @property
    def is_rectangular(self) -> bool:
        """
        Returns True if the frame is a rectangular one.
        """
        return True

    @property
    def is_independent(self) -> bool:
        """
        Returns True if the base vectors that make up the frame are linearly
        independent.
        """
        return True

    def __mul__(self, other) -> ReferenceFrame:
        rtype = RectangularFrame if isinstance(other, (float, int)) else ReferenceFrame
        return out_of_place_binary(self, other, np.multiply, rtype)

    def __imul__(self, other) -> ReferenceFrame:
        rtype = None if isinstance(other, (float, int)) else ReferenceFrame
        return inplace_binary(self, other, np.multiply, rtype)

    def __truediv__(self, other) -> ReferenceFrame:
        rtype = RectangularFrame if isinstance(other, (float, int)) else ReferenceFrame
        return out_of_place_binary(self, other, np.divide, rtype)

    def __itruediv__(self, other) -> ReferenceFrame:
        rtype = None if isinstance(other, (float, int)) else ReferenceFrame
        return inplace_binary(self, other, np.divide, rtype)

    def __imatmul__(self, other) -> ReferenceFrame:
        return inplace_binary(self, other, np.matmul, ReferenceFrame)

    def __iadd__(self, other) -> ReferenceFrame:
        return inplace_binary(self, other, np.add, ReferenceFrame)

    def __isub__(self, other) -> ReferenceFrame:
        return inplace_binary(self, other, np.subtract, ReferenceFrame)

    def __ipow__(self, other) -> ReferenceFrame:
        return inplace_binary(self, other, np.power, ReferenceFrame)


class CartesianFrame(RectangularFrame):
    """
    A class for cartesian (orthonormal) reference frames. Just like the RectangularFrame
    class, this is similar to the more general ReferenceFrame, but with increased
    performance and even more limitations.

    Examples
    --------
    For operators *= and /= the type of the instance is cast to RectangularFrame

    >>> from sigmaepsilon.math.linalg import CartesianFrame
    >>> import numpy as np
    >>> A = CartesianFrame(dim=3)
    >>> A *= 2
    >>> type(A)
    <class 'sigmaepsilon.math.linalg.frame.RectangularFrame'>

    For other operations, the type of the object is cast to a general one

    >>> A += 2
    >>> type(A)
    <class 'sigmaepsilon.math.linalg.frame.ReferenceFrame'>

    The only operation that results in a CartesianFrame object is a rotation:

    >>> A = CartesianFrame(dim=3)
    >>> A.orient('Space', [0, 0, np.pi], 'XYZ')
    Array([[-1.0000000e+00,  1.2246468e-16,  0.0000000e+00],
           [-1.2246468e-16, -1.0000000e+00,  0.0000000e+00],
           [ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])

    >>> type(A)
    <class 'sigmaepsilon.math.linalg.frame.CartesianFrame'>

    See also
    --------
    :class:`~sigmaepsilon.math.linalg.ReferenceFrame`
    :class:`~sigmaepsilon.math.linalg.RectangularFrame`
    """

    def __init__(
        self, *args, normalize: bool = False, assume_cartesian: bool = False, **kwargs
    ):
        super().__init__(*args, assume_rectangular=assume_cartesian, **kwargs)
        if normalize:
            self.axes = normalize_frame(self.axes)
        else:
            if not assume_cartesian:
                assert is_normal_frame(
                    self.axes
                ), "This frame is not cartesian, check your input!"

    @property
    def is_rectangular(self) -> bool:
        """
        Returns True if the frame is a rectangular one.
        """
        return True

    @property
    def is_cartesian(self) -> bool:
        """
        Returns True if the frame is a cartesian (orthonormal) one.
        """
        return True

    def Gram(self) -> ndarray:
        """
        Returns the Gram-matrix of the frame.
        """
        return np.eye(self.axes.shape[0])

    def volume(self) -> float:
        """
        Returns the signed volume of the general parallelepiped described by the
        base vectors that make up the frame.
        """
        return 1.0

    def dual(self) -> ReferenceFrame:
        """
        Returns the dual (or reciprocal) frame.
        """
        return self

    def __imul__(self, other) -> ReferenceFrame | RectangularFrame:
        rtype = RectangularFrame if isinstance(other, (float, int)) else ReferenceFrame
        return inplace_binary(self, other, np.multiply, rtype)

    def __itruediv__(self, other) -> ReferenceFrame | RectangularFrame:
        rtype = RectangularFrame if isinstance(other, (float, int)) else ReferenceFrame
        return inplace_binary(self, other, np.divide, rtype)

    def __imatmul__(self, other) -> ReferenceFrame:
        return inplace_binary(self, other, np.matmul, ReferenceFrame)

    def __iadd__(self, other) -> ReferenceFrame:
        return inplace_binary(self, other, np.add, ReferenceFrame)

    def __isub__(self, other) -> ReferenceFrame:
        return inplace_binary(self, other, np.subtract, ReferenceFrame)

    def __ipow__(self, other) -> ReferenceFrame:
        return inplace_binary(self, other, np.power, ReferenceFrame)
