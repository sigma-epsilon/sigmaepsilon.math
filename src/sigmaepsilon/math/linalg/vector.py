"""Vector class built on top of the linalg reference frame and abstract tensor layer."""

from copy import copy, deepcopy

from numpy import ndarray
import numbers

from .utils import show_vector
from .frame import ReferenceFrame as Frame
from .abstract import AbstractTensor
from .meta import FrameLike


__all__ = ["Vector"]


class Vector(AbstractTensor):
    """Extend `NumPy`'s ``ndarray`` class to handle arrays with associated reference frames.

    The class also provides a mechanism to transform vectors between
    different frames. Use it like if it was a ``numpy.ndarray`` instance.

    All parameters are identical to those of ``numpy.ndarray``, except that
    this class allows to specify an embedding frame.

    Parameters
    ----------
    args: tuple, Optional
        Positional arguments forwarded to `numpy.ndarray`.
    frame: FrameLike, Optional
        The reference frame the vector is represented by its coordinates.
    kwargs: dict, Optional
        Keyword arguments forwarded to `numpy.ndarray`.

    Examples
    --------
    Import the necessary classes:

    >>> import numpy as np
    >>> from sigmaepsilon.math.linalg import Vector, ReferenceFrame

    Create a default frame in 3d space, and create 2 others, each
    being rotated with 30 degrees around the third axis.

    >>> A = ReferenceFrame(dim=3)
    >>> B = A.orient_new('Body', [0, 0, 30*np.pi/180], 'XYZ')
    >>> C = B.orient_new('Body', [0, 0, 30*np.pi/180], 'XYZ')

    To create a vector in a frame:

    >>> vA = Vector([1.0, 1.0, 0.0], frame=A)

    To create a vector with a relative transformation to another one:

    >>> vB = vA.orient_new('Body', [0, 0, -30*np.pi/180], 'XYZ')

    Use the `array` property to get the componets of a `Vector`:

    >>> vB.array
    Array([1.3660254, 0.3660254, 0.       ])

    If you want to obtain the components of a vector in a specific
    target frame C, do this:

    >>> vB.show(C)
    array([ 1., -1.,  0.])

    The reason why the result is represented now as 'array' insted of 'Array'
    as in the previous case is that the Vector class is an array container. When
    you type `vB.array`, what is returned is a wrapped object, an instance of `Array`,
    which is also a class of this library. When you say `vB.show(C)`, a NumPy array
    is returned. Since the `Array` class is a direct subclass of NumPy's `ndarray` class,
    it doesn't really matter and the only difference is the capital first letter.

    To create a vector in a target frame C, knowing the components in a
    source frame A:

    >>> vC = Vector(vA.show(C), frame=C)

    See Also
    --------
    :class:`~sigmaepsilon.math.linalg.tensor.Tensor`
    :class:`~sigmaepsilon.math.linalg.frame.ReferenceFrame`
    """

    _frame_cls_ = Frame
    _HANDLED_TYPES_ = (numbers.Number,)

    def __init__(
        self,
        *args,
        frame: FrameLike | None = None,
        **kwargs,
    ):
        super().__init__(*args, frame=frame, **kwargs)

    @classmethod
    def _verify_input(cls, arr: ndarray, *_, **kwargs) -> bool:
        """Verify if an array input is acceptable for the current class.

        If not a general Tensor class is returned upon calling the creator.
        """
        return True

    @property
    def rank(self) -> int:
        """Return the tensor rank (or order)."""
        return 1

    def dual(self) -> "Vector":
        """Return the vector described in the dual (or reciprocal) frame."""
        # NOTE Strictly this should be the transpose of the Gram matrix,
        # but it is symmetric. `show_vector` handles bulk vectors and frames.
        a = show_vector(self.frame.Gram(), self.array)
        return self.__class__(a, frame=self.frame.dual())

    def show(self, target: Frame = None, *, dcm: ndarray = None) -> ndarray:
        """Return the components in a target frame.

        If the target is `None`, the components are returned in the ambient
        frame. The transformation can also be specified with a proper DCM matrix.

        Parameters
        ----------
        target: numpy.ndarray, Optional
            Target frame.
        dcm: numpy.ndarray, Optional
            The DCM matrix of the transformation.

        Returns
        -------
        numpy.ndarray
            The components of the vector in a specified frame, or
            the ambient frame, depending on the arguments.
        """
        if not isinstance(dcm, ndarray):
            if target is None:
                target = self._frame_cls_(dim=self._array.shape[-1])
            dcm = self.frame.dcm(target=target)
        return show_vector(dcm, self.array)  # dcm @ arr

    def orient(self, *args, dcm: ndarray = None, **kwargs) -> "Vector":
        """Rotate the vector inplace about the axes of the ambient frame.

        The vector is rotated the same way
        :meth:`~sigmaepsilon.math.linalg.frame.ReferenceFrame.orient` rotates a
        frame with the same arguments, regardless of the frame the vector is
        expressed in. The frame of the vector is left unchanged, only the
        components change. If the frame is rotated together with the vector,
        the components don't change.

        Parameters
        ----------
        *args: tuple, Optional
            Positional arguments forwarded to
            :func:`~sigmaepsilon.math.linalg.utils.rotation_matrix`.
        dcm: numpy.ndarray, Optional
            The DCM of the rotation, as returned by
            :func:`~sigmaepsilon.math.linalg.utils.rotation_matrix`. If provided,
            the other arguments are ignored.
        **kwargs: dict, Optional
            Keyword arguments forwarded to
            :func:`~sigmaepsilon.math.linalg.utils.rotation_matrix`.

        Returns
        -------
        Vector
            The same vector the function is called upon.

        See Also
        --------
        :func:`orient_new`
        """
        # The DCM of a rotation is orthogonal, so its transpose is its inverse.
        Q = self._orientation_matrix(*args, dcm=dcm, **kwargs)
        self.array = show_vector(Q, self.array)
        return self

    def orient_new(self, *args, **kwargs) -> "Vector":
        """Return a copy of the vector, rotated about the axes of the ambient frame.

        All arguments are forwarded to :func:`orient`.

        Returns
        -------
        Vector
            A new vector, in the same frame as the instance.

        See Also
        --------
        :func:`orient`
        """
        return self.copy().orient(*args, **kwargs)

    def copy(self, deep: bool = False, name: str = None) -> "Vector":
        """Return a shallow or deep copy of the vector, keeping its frame.

        A shallow copy shares the frame of the instance, a deep copy gets a
        copy of it. The argument `name` is not used, it is kept for backward
        compatibility.
        """
        return deepcopy(self) if deep else copy(self)

    def deepcopy(self, name: str = None) -> "Vector":
        """Return a deep copy of the vector."""
        return self.copy(deep=True, name=name)
