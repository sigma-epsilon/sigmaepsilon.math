from abc import abstractmethod
import weakref
from typing import Tuple, Union
import numbers
from functools import partial
from copy import copy, deepcopy

import numpy as np
from numpy import array_repr, array_str, ndarray
from numpy.lib.mixins import NDArrayOperatorsMixin

from sigmaepsilon.core import Wrapper
from sigmaepsilon.core.abstract import ABC_Safe

from ..utils import ascont, minmax
from ..metautils import _new_and_init_
from .exceptions import LinalgMissingInputError


__all__ = ["ArrayWrapper", "ArrayLike", "TensorLike", "FrameLike"]


class Array(ABC_Safe, ndarray):
    """
    Base backend class for array-like classes. Although you don't really need
    to directly create instances of this class, you can use it like if it was
    a ``numpy.ndarray`` instance.

    The class has a safe metaclass, which means that there is a safety mechanism
    that prevents you from unintentionally crashing the internal behaviour of the
    class upon subclassing. This practically means, that you will see an error if
    you try to shadow a definition in any of the base classes of the class. For this
    reason it is safer to subclass this class rather than to directly subclass
    NumPy's ndarray class.

    See also
    --------
    :class:`~numpy.ndarray`
    :class:`~sigmaepsilon.core.abstract.ABC_Safe`
    """

    def __new__(
        subtype,
        shape=None,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        frame=None,
    ):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments. This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super().__new__(subtype, shape, dtype, buffer, offset, strides, order)
        obj._frame = frame
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return 0to
        #    InfoArray.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self._frame = getattr(obj, "_frame", None)

    @property
    def frame(self) -> "FrameLike":
        """
        Returns the frame of the vector.
        """
        return self._frame

    @frame.setter
    def frame(self, value: "FrameLike"):
        """
        Sets the frame.
        """
        if isinstance(value, FrameLike):
            self._frame = value
        else:
            raise TypeError("Value must be a {} instance".format(FrameLike))

    def __repr__(self):
        return array_repr(self)

    def __str__(self):
        return array_str(self)


class ArrayWrapper(NDArrayOperatorsMixin, Wrapper):
    """
    Base frontend class for array-like classes. Use it like if it
    was a ``numpy.ndarray`` instance.
    """

    _array_cls_ = Array

    # One might also consider adding the built-in list type to this
    # list, to support operations like np.add(array_like, list)
    _HANDLED_TYPES_ = (np.ndarray, numbers.Number, list)

    def __init__(self, *args, cls_params=None, contiguous: bool = True, **kwargs):
        if len(args) > 0 and isinstance(args[0], np.ndarray):
            buf = ascont(args[0]) if contiguous else args[0]
        else:
            buf = np.array(*args, **kwargs)
        cls_params = dict() if cls_params is None else cls_params
        buf = np.array(buf, dtype=float)
        self._array = self._array_cls_(
            shape=buf.shape, buffer=buf, dtype=buf.dtype, **cls_params
        )
        super(ArrayWrapper, self).__init__(wrap=self._array)

    @property
    def dim(self) -> int:
        """
        Returns the dimension of the array.
        """
        return len(self._array.shape)

    @property
    def minmax(self) -> Tuple[float]:
        """
        Returns the minimum and maximum values of the array.
        """
        return minmax(self._array)

    def chop(self, tol: float = 1e-12) -> "ArrayWrapper":
        """
        Sets very small values (in an absolute sense) to zero.

        .. versionadded:: 1.0.5

        Parameters
        ----------
        tol: float, Optional
            The values whose absolute value is less than this limit are
            set to zero. Default is 1e-12.

        Returns
        -------
        ~`sigmaepsilon.math.linalg.meta.ArrayWrapper`
            The object the call was made upon.
        """
        self._array[np.where(np.abs(self._array) < tol)] = 0.0
        return self

    def __array__(self, dtype=None):
        if dtype is not None:
            return self._array.astype(dtype)
        return self._array

    def __getitem__(self, key):
        return self._array.__getitem__(key)

    def __setitem__(self, key, value):
        return self._array.__setitem__(key, value)

    def __len__(self):
        return self._array.shape[0]

    def to_numpy(self) -> np.ndarray:
        """
        Returns the data as a pure NumPy array.
        """
        return self.__array__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES_ + (Array, ArrayWrapper)):
                raise TypeError(f"Invalid type encountered at {ufunc}")

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x._array if isinstance(x, ArrayWrapper) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(
                x._array if isinstance(x, ArrayWrapper) else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            return type(self)(result)

    def __repr__(self):
        return array_repr(self)

    def __str__(self):
        return array_str(self)


class FrameLike(ArrayWrapper):
    """
    Base class for reference frames.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weakrefs = weakref.WeakValueDictionary()

    @abstractmethod
    def dcm(self) -> ndarray: ...

    @abstractmethod
    def show(self) -> ndarray: ...

    @abstractmethod
    def orient(self) -> "FrameLike": ...

    @abstractmethod
    def orient_new(self) -> "FrameLike": ...

    @abstractmethod
    def Gram(self) -> ndarray: ...

    @abstractmethod
    def dual(self) -> "FrameLike": ...

    @abstractmethod
    def transpose(self, inplace: bool = False) -> "FrameLike": ...

    @property
    def T(self) -> "FrameLike":
        """
        Returns the transpose.
        """
        return self.transpose(inplace=False)

    def _register_tensorial_(self, v: "TensorLike"):
        """
        Registers tensorial objects by appending a weak reference to the set
        of weak references. Registered objects change their components upon
        changes of their supporting frame.
        """
        self._weakrefs[id(v)] = v

    def _unregister_tensorial_(self, v: "TensorLike") -> bool:
        """
        Unregisters previously registered tensorial objects. Returns True if the
        object was found in the registry, False if it was not.
        """
        k = id(v)
        if k in self._weakrefs:
            del self._weakrefs[k]
            return True
        return False


class TensorLike(ArrayWrapper):
    """
    Abstract base class for numerical data classes that walk and talk like
    a tensor does.
    """

    _frame_cls_: FrameLike = None
    _rank_: int = None

    def __init__(
        self,
        *args,
        frame: FrameLike | None = None,
        bulk: bool | None = None,
        rank: int | None = None,
        **kwargs,
    ):
        if len(args) > 0 and isinstance(args[0], np.ndarray):
            if not self._verify_input(args[0], bulk=bulk, rank=rank):
                raise ValueError("Invalid input to Tensor class.")

        cls_params = kwargs.get("cls_params", dict())
        if frame is not None:
            if not isinstance(frame, FrameLike):
                raise TypeError(f"The frame must be of type {FrameLike}.")

            cls_params["frame"] = frame
        else:
            if not (len(args) > 0 and isinstance(args[0], np.ndarray)):
                raise LinalgMissingInputError(
                    "A frame or an array of components is required."
                )

            arr = args[0]
            if bulk:
                frame = _new_and_init_(self.__class__._frame_cls_, dim=arr.shape[1])
            else:
                frame = _new_and_init_(self.__class__._frame_cls_, dim=arr.shape[0])
            cls_params["frame"] = frame

        kwargs["cls_params"] = cls_params
        super().__init__(*args, **kwargs)

        if self._array._frame is None:
            frame_cls = self.__class__._frame_cls_
            self._array._frame = _new_and_init_(frame_cls, dim=self._array.shape)

        self.frame._register_tensorial_(self)

        self._bulk = bulk

        if rank is not None:
            if self.__class__._rank_ is not None:
                raise ValueError("Rank is already defined on the class level.")
            else:
                self._rank = rank
        else:
            self._rank = None

    def __deepcopy__(self, memo):
        return self.__copy__(memo)

    def __copy__(self, memo=None):
        cls = type(self)
        copy_function = copy if (memo is None) else partial(deepcopy, memo=memo)
        is_deep = memo is not None

        f = self.frame
        if is_deep:
            ax = deepcopy(f.axes)
            memo[id(f.axes)] = ax
            frame = _new_and_init_(cls._frame_cls_, ax)
        else:
            frame = f

        arr = copy_function(self.array)
        if is_deep:
            memo[id(self.array)] = arr

        return cls(arr, frame=frame)

    @classmethod
    def _from_any_input(cls, *args, **kwargs) -> "TensorLike":
        raise NotImplementedError

    @classmethod
    def _verify_input(cls, arr: ndarray, *_, **kwargs) -> bool:
        raise NotImplementedError

    @property
    def rank(self) -> int:
        """
        Returns the tensor rank (or order).
        """
        if self._rank:
            return self._rank
        else:
            if self.__class__._rank_:
                return self.__class__._rank_
            else:
                if self._bulk:
                    return len(self.array.shape) - 1
                else:
                    return len(self.array.shape)

    @property
    def array(self) -> Array:
        """
        Returns the coordinates of the vector.
        """
        return self._array

    @array.setter
    def array(self, value: np.ndarray):
        """
        Sets the coordinates of the vector.
        """
        array = np.array(value)
        assert array.shape == self._array.shape
        self._array[...] = array

    @property
    def frame(self) -> FrameLike:
        """
        Returns the frame of the vector.
        """
        return self.array.frame

    @frame.setter
    def frame(self, value: FrameLike | ndarray):
        """
        Sets the frame of the vector.

        Note
        ----
        Setting a new frame may change the compoenents of the instance.
        """
        if isinstance(value, FrameLike):
            f = value
        elif isinstance(value, ndarray):
            f = _new_and_init_(self._frame_cls_, value)
        else:
            raise TypeError(f"Value must be a {ndarray} or a {FrameLike} instance")

        array = self.show(f)
        self.array.frame._unregister_tensorial_(self)
        self.array = array
        self.array.frame = f
        f._register_tensorial_(self)

    @property
    def T(self) -> "TensorLike":
        """
        Returns the transpose.
        """
        return self.transpose(inplace=False)

    def transpose(self, inplace: bool = False) -> "TensorLike":
        """
        Either transposes the array of the tensor, or returns a copy
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
        r = self.rank
        shape = self.array.shape
        indices = tuple(range(len(shape)))
        data_indices = indices[:-r]
        tensor_indices = indices[len(shape) - r :]
        indices = data_indices + tensor_indices[::-1]
        if inplace:
            self._array = np.transpose(self.array, indices)
            return self
        else:
            f = self.frame
            frame = f.__class__(np.copy(f.axes))
            return self.__class__(np.transpose(self.array, indices), frame=frame)

    @abstractmethod
    def show(self) -> Array: ...

    @abstractmethod
    def orient(self) -> "TensorLike": ...

    @abstractmethod
    def orient_new(self) -> "TensorLike": ...

    def is_bulk(self):
        """
        Returns True if the object represents a collection of tensors, False
        otherwise.
        """
        if self._bulk:
            return True
        else:
            not self.rank == len(self.array.shape)


ArrayLike = Union[ArrayWrapper, ndarray, Array]
