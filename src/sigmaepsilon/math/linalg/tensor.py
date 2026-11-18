"""Tensor classes built on top of the linalg reference frame and abstract tensor layer."""

from copy import copy, deepcopy

import numpy as np
from numpy import ndarray

from sigmaepsilon.core.alphabet import latinrange

from .frame import ReferenceFrame as Frame
from .abstract import AbstractTensor
from .tr import _tr_tensors2, _tr_tensors4, _tr_tensors4x3
from .exceptions import TensorShapeMismatchError
from .logical import has_equal_dimensions

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
        return has_equal_dimensions(arr)

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
        """Return the tensor described in the dual (or reciprocal) frame."""
        a = self.transform_components(self.frame.Gram())
        return self.__class__(a, frame=self.frame.dual())

    def transform_components(self, Q: ndarray) -> ndarray:
        """Return the components of the tensor transformed by the matrix Q."""
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
        """Return the components in a target frame.

        If the target is `None`, the components are returned in the ambient frame.
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

    def orient(self, *args, dcm: ndarray = None, **kwargs) -> "Tensor":
        """Rotate the tensor inplace about the axes of the ambient frame.

        The tensor is rotated the same way
        :meth:`~sigmaepsilon.math.linalg.frame.ReferenceFrame.orient` rotates a
        frame with the same arguments, regardless of the frame the tensor is
        expressed in. The frame of the tensor is left unchanged, only the
        components change.

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
        Tensor
            The same tensor the function is called upon.

        See Also
        --------
        :func:`orient_new`
        """
        Q = self._orientation_matrix(*args, dcm=dcm, **kwargs)
        self.array = self.transform_components(Q)
        return self

    def orient_new(self, *args, **kwargs) -> "Tensor":
        """Return a copy of the tensor, rotated about the axes of the ambient frame.

        All arguments are forwarded to :func:`orient`.

        Returns
        -------
        Tensor
            A new tensor, in the same frame as the instance.

        See Also
        --------
        :func:`orient`
        """
        return self.copy().orient(*args, **kwargs)

    def copy(self, deep: bool = False, name: str = None) -> "Tensor":
        """Return a shallow or deep copy of the tensor, keeping its frame.

        A shallow copy shares the frame of the instance, a deep copy gets a
        copy of it. The argument `name` is not used, it is kept for backward
        compatibility.
        """
        return deepcopy(self) if deep else copy(self)

    def deepcopy(self, name: str = None) -> "Tensor":
        """Return a deep copy of the tensor."""
        return self.copy(deep=True, name=name)


class Tensor2(Tensor):
    """A class to handle second-order tensors.

    Some operations have dedicated implementations that provide higher
    performence utilizing implicit parallelization. Examples for tensors
    of this class include the metric tensor, or the stress and strain
    tensors of elasticity.

    See Also
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
        """Return the components of the tensor transformed by the matrix Q."""
        return _tr_tensors2(self.array, Q)


class Tensor2x3(Tensor2):
    """Dedicated class for second-order tensors, with 3 indices per axis.

    Since the shape of the tensor is known, instances are able to
    automatically detect if the provided components resemble a single
    item or a collection.
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
    """A class to handle fourth-order tensors.

    Some operations have dedicated implementations that provide higher
    performence utilizing implicit parallelization. Examples of this class
    include the piezo-optical tensor, the elasto-optical tensor, the
    flexoelectric tensor or the elasticity tensor.

    See Also
    --------
    :class:`~sigmaepsilon.math.linalg.tensor.Tensor4x3`
    """

    _rank_ = 4

    @classmethod
    def _verify_input(cls, arr: ndarray, *_, bulk: bool = False, **kwargs) -> bool:
        shape = arr.shape
        if len(shape) < 4:
            return False
        equal_dims = (shape[-1],) * 4 == shape[-4:]
        if bulk:
            return len(shape) >= 5 and equal_dims
        else:
            return len(shape) == 4 and equal_dims

    def transform_components(self, dcm: ndarray) -> ndarray:
        """Return the components of the transformed numerical tensor.

        The transformation is based on the provided direction cosine matrix. It
        works in any dimension. In 3 dimensions it uses the same compiled kernels
        as :class:`~sigmaepsilon.math.linalg.tensor.Tensor4x3`.
        """
        return _tr_tensors4(self._array, dcm)


class Tensor4x3(Tensor4):
    """Dedicated class for fourth order tensors, with 3 indices per axis.

    Since the shape of the tensor is known, instances are able to
    automatically detect if the provided components resemble a single
    item or a collection.
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
        is_3x3x3x3 = arr.shape[-4:] == (3, 3, 3, 3)
        if bulk:
            return len(arr.shape) >= 5 and is_3x3x3x3
        else:
            return len(arr.shape) == 4 and is_3x3x3x3

    def transform_components(self, dcm: ndarray) -> ndarray:
        """Return the components of the transformed numerical tensor.

        The transformation is based on the provided direction cosine matrix and
        runs on compiled 3x3x3x3 kernels.
        """
        return _tr_tensors4x3(self._array, dcm)
