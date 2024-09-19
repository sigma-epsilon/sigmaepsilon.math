from copy import deepcopy as dcopy

import numpy as np
from numpy import ndarray

from sigmaepsilon.core.alphabet import latinrange

from .frame import ReferenceFrame as Frame
from .abstract import AbstractTensor
from .tr import _tr_tensors2, _tr_tensors4x3
from .exceptions import TensorShapeMismatchError
from .utils import transpose_axes
from .logical import is_hermitian

__all__ = [
    "Tensor",
    "Tensor2",
    "Tensor4",
    "Tensor2x3",
    "Tensor4x3",
]


class Tensor(AbstractTensor):
    """
    A class to handle tensors.

    Parameters
    ----------
    args: tuple, Optional
        Positional arguments forwarded to `numpy.ndarray`.
    frame: numpy.ndarray, Optional
        The reference frame the vector is represented by its coordinates.
    kwargs: dict, Optional
        Keyword arguments forwarded to `numpy.ndarray`.

    Examples
    --------
    Import the necessary classes:

    >>> from sigmaepsilon.math.linalg import Tensor, ReferenceFrame
    >>> from numpy.random import rand

    Create a Tensor of order 6 in a frame with random components

    >>> frame = ReferenceFrame(dim=3)
    >>> array = rand(3, 3, 3, 3, 3, 3)
    >>> A = Tensor(array, frame=frame)

    Get the tensor in the dual frame:

    >>> A_dual = A.dual()

    Create an other tensor, in this case a 5th-order one, and calculate their
    generalized dot product, which is a 9th-order tensor:

    >>> from sigmaepsilon.math.linalg import dot
    >>> array = rand(3, 3, 3, 3, 3)
    >>> B = Tensor(array, frame=frame)
    >>> C = dot(A, B, axes=[0, 0])
    >>> assert C.rank == (A.rank + B.rank - 2)

    See Also
    --------
    :class:`~sigmaepsilon.math.linalg.vector.Vector`
    :class:`~sigmaepsilon.math.linalg.frame.ReferenceFrame`
    """

    _frame_cls_ = Frame
    _einsum_params_ = {}

    @classmethod
    def _verify_input(cls, arr: ndarray, *_, **kwargs) -> bool:
        return is_hermitian(arr)

    @classmethod
    def _from_any_input(cls, *args, **kwargs) -> AbstractTensor:
        if cls._verify_input(*args, **kwargs):
            return cls(*args, **kwargs)
        else:
            if not Tensor._verify_input(*args, **kwargs):
                raise ValueError("Invalid input to Tensor class.")
            else:
                return Tensor(*args, **kwargs)

    def dual(self) -> "Tensor2":
        """
        Returns the tensor described in the dual (or reciprocal) frame.
        """
        a = self.transform_components(self.frame.Gram())
        return self.__class__(a, frame=self.frame.dual())

    def transform_components(self, Q: ndarray) -> ndarray:
        """
        Returns the components of the tensor transformed by the matrix Q.
        """
        r = self.rank
        arr = self.array
        args = [Q for _ in range(r)]
        if r not in self.__class__._einsum_params_:
            target = latinrange(r, start=ord("a"))
            source = latinrange(r, start=ord("a") + r)
            terms = [t + s for s, t in zip(source, target)]
            command = ",".join(terms) + "," + "".join(source)
            einsum_path = np.einsum_path(command, *args, arr, optimize="greedy")[0]
            self.__class__._einsum_params_[r] = (command, einsum_path)
        else:
            command, einsum_path = self.__class__._einsum_params_[r]
        return np.einsum(command, *args, arr, optimize=einsum_path)

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
            The components of the tensor in a specified frame, or
            the ambient frame, depending on the arguments.
        """
        if not isinstance(dcm, ndarray):
            if target is None:
                target = Frame(dim=self.frame.axes.shape[-1])
            dcm = self.frame.dcm(target=target)
        return self.transform_components(dcm)

    def orient(self, *args, **kwargs) -> "Tensor":
        """
        Orients the vector inplace. All arguments are forwarded to
        `orient_new`.

        Returns
        -------
        Vector
            The same vector the function is called upon.

        See Also
        --------
        :func:`orient_new`
        """
        fcls: Frame = self.__class__._frame_cls_
        dcm = fcls.eye(dim=self.frame.shape[-1]).orient_new(*args, **kwargs).dcm()
        self.array = self.transform_components(transpose_axes(dcm))
        return self

    def orient_new(self, *args, **kwargs) -> "Tensor":
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
        fcls: Frame = self.__class__._frame_cls_
        dcm = fcls.eye(dim=self.frame.shape[-1]).orient_new(*args, **kwargs).dcm()
        array = self.transform_components(dcm.T)
        return self.__class__(array, frame=self.frame)

    def copy(self, deep: bool = False, name: str = None) -> "Tensor":
        """
        Returns a shallow or deep copy of this object, depending of the
        argument `deepcopy` (default is False).
        """
        if deep:
            return self.__class__(dcopy(self.array), name=name)
        else:
            return self.__class__(self.array, name=name)

    def deepcopy(self, name: str = None) -> "Tensor":
        """
        Returns a deep copy of the frame.
        """
        return self.copy(deep=True, name=name)


class Tensor2(Tensor):
    """
    A class to handle second-order tensors. Some operations have dedicated implementations
    that provide higher performence utilizing implicit parallelization. Examples
    for tensors of this class include the metric tensor, or the stress and strain tensors
    of elasticity.

    See also
    --------
    :class:`~sigmaepsilon.math.linalg.tensor.Tensor2x3`
    """

    _rank_ = 2

    @classmethod
    def _verify_input(cls, arr: ndarray, *_, bulk: bool = False, **kwargs) -> bool:
        if bulk:
            return len(arr.shape) == 3 and arr.shape[-1] == arr.shape[-2]
        else:
            return len(arr.shape) == 2 and arr.shape[-1] == arr.shape[-2]

    def transform_components(self, Q: ndarray) -> ndarray:
        return _tr_tensors2(self.array, Q)


class Tensor2x3(Tensor2):
    """
    Dedicated class for second-order tensors, with 3 indices per axis.
    Since the shape of the tensor is known, instances are able to automatically detect
    if the provided components resemble a single item or a collection.
    """

    def __init__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], ndarray):
            arr = args[0]
            shape = arr.shape
            if shape[-2:] == (3, 3):
                if len(shape) >= 3:
                    is_bulk = kwargs.get("bulk", True)
                    if not is_bulk:
                        raise ValueError("Incorrect input!")
                    kwargs["bulk"] = is_bulk
                else:
                    if not len(shape) == 2:
                        raise TensorShapeMismatchError("Invalid shape!")
                    is_bulk = kwargs.get("bulk", False)
                    if is_bulk:
                        raise ValueError("Incorrect input!")
            else:
                raise TensorShapeMismatchError("Invalid shape!")

        super().__init__(*args, **kwargs)

    @classmethod
    def _verify_input(cls, arr: ndarray, *_, bulk: bool = False, **kwargs) -> bool:
        if bulk:
            return len(arr.shape) >= 3 and arr.shape[-2:] == (3, 3)
        else:
            return len(arr.shape) == 2 and arr.shape[-2:] == (3, 3)


class Tensor4(Tensor):
    """
    A class to handle fourth-order tensors. Some operations have dedicated implementations
    that provide higher performence utilizing implicit parallelization. Examples of this class
    include the piezo-optical tensor, the elasto-optical tensor, the flexoelectric tensor or the
    elasticity tensor.

    See also
    --------
    :class:`~sigmaepsilon.math.linalg.tensor.Tensor4x3`
    """

    _rank_ = 4

    @classmethod
    def _verify_input(cls, arr: ndarray, *_, bulk: bool = False, **kwargs) -> bool:
        shape = arr.shape
        is_hermitian = (shape[-1],) * 4 == shape[-4:]
        if bulk:
            return len(shape) == 5 and is_hermitian
        else:
            return len(shape) == 4 and is_hermitian

    def transform_components(self, dcm: ndarray) -> ndarray:
        """
        Returns the components of the transformed numerical tensor, based on
        the provided direction cosine matrix.
        """
        return _tr_tensors4x3(self._array, dcm)


class Tensor4x3(Tensor4):
    """
    Dedicated class for fourth order tensors, with 3 indices per axis.
    Since the shape of the tensor is known, instances are able to automatically detect
    if the provided components resemble a single item or a collection.
    """

    def __init__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], ndarray):
            arr = args[0]
            shape = arr.shape
            if shape[-4:] == (3, 3, 3, 3):
                if len(shape) >= 5:
                    is_bulk = kwargs.get("bulk", True)
                    if not is_bulk:
                        raise ValueError("Incorrect input!")
                    kwargs["bulk"] = is_bulk
                else:
                    if not len(shape) == 4:
                        raise TensorShapeMismatchError("Invalid shape!")
                    is_bulk = kwargs.get("bulk", False)
                    if is_bulk:
                        raise ValueError("Incorrect input!")
            else:
                raise TensorShapeMismatchError("Invalid shape!")

        super().__init__(*args, **kwargs)

    @classmethod
    def _verify_input(cls, arr: ndarray, *_, bulk: bool = False, **kwargs) -> bool:
        is_hermitian = arr.shape[-4:] == (3, 3, 3, 3)
        if bulk:
            return len(arr.shape) >= 5 and is_hermitian
        else:
            return len(arr.shape) == 4 and is_hermitian
