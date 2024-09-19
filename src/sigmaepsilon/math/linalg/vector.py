from copy import deepcopy as dcopy

from numpy import ndarray
import numbers

from .utils import show_vector
from .frame import ReferenceFrame as Frame
from .abstract import AbstractTensor
from .meta import FrameLike


__all__ = ["Vector"]


class Vector(AbstractTensor):
    """
    Extends `NumPy`'s ``ndarray`` class to handle arrays with associated
    reference frames. The class also provides a mechanism to transform
    vectors between different frames. Use it like if it was a ``numpy.ndarray``
    instance.

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
        """
        Ought to verify if an array input is acceptable for the current class.
        If not a general Tensor class is returned upon calling the creator.
        """
        return True

    @property
    def rank(self) -> int:
        """
        Returns the tensor rank (or order).
        """
        return 1

    def dual(self) -> "Vector":
        """
        Returns the vector described in the dual (or reciprocal) frame.
        """
        # NOTE Strictly this should be self.frame.Gram().T @ self.array,
        # but since the Gram matrix is symmetric, it's cheaper like this
        a = self.frame.Gram() @ self.array
        return self.__class__(a, frame=self.frame.dual())

    def show(self, target: Frame = None, *, dcm: ndarray = None) -> ndarray:
        """
        Returns the components in a target frame. If the target is
        `None`, the components are returned in the ambient frame.

        The transformation can also be specified with a proper DCM matrix.

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
        """
        Orients the vector inplace. If the transformation is not specified by 'dcm',
        all arguments are forwarded to `orient_new`.

        Parameters
        ----------
        dcm: numpy.ndarray, Optional
            The DCM matrix of the transformation.

        Returns
        -------
        Vector
            The same vector the function is called upon.

        See Also
        --------
        :func:`orient_new`
        """
        if not isinstance(dcm, ndarray):
            fcls = self.__class__._frame_cls_
            dcm = fcls.eye(dim=len(self)).orient_new(*args, **kwargs).dcm()
            # self.array = dcm.T @ self._array
            self.array = show_vector(dcm.T, self.array)
            # self.array = np.linalg.inv(dcm) @ self._array
            # FIXME check this
        else:
            self.array = show_vector(dcm.T, self.array)
            # self.array = dcm.T @ self._array
            # FIXME check if inversion is necessary here
            # inversion might be necessary here because it is uncertain if the
            # dcm matrix was fabricated properly.
            # self.array = np.linalg.inv(dcm) @ self._array
        return self

    def orient_new(self, *args, **kwargs) -> "Vector":
        """
        Returns a transformed version of the instance.

        Returns
        -------
        Vector
            A new vector.

        See Also
        --------
        :func:`orient`
        """
        fcls = self.__class__._frame_cls_
        dcm = fcls.eye(dim=len(self)).orient_new(*args, **kwargs).dcm()
        array = dcm.T @ self._array
        # FIXME check if inversion is necessary or not
        # array = np.linalg.inv(dcm) @ self._array
        return Vector(array, frame=self.frame)

    def copy(self, deep: bool = False, name: str = None) -> "Vector":
        """
        Returns a shallow or deep copy of this object, depending of the
        argument `deepcopy` (default is False).
        """
        if deep:
            return self.__class__(dcopy(self.array), name=name)
        else:
            return self.__class__(self.array, name=name)

    def deepcopy(self, name: str = None) -> "Vector":
        """
        Returns a deep copy of the frame.
        """
        return self.copy(deep=True, name=name)
