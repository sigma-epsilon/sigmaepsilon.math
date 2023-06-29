from typing import Callable
import numpy as np


__all__ = ["squeeze"]


def squeeze_if_array(arr):
    return np.squeeze(arr) if isinstance(arr, np.ndarray) else arr


def squeeze(default=True):
    """
    A decorator that squeezes outputs of a function if
    * the result is a NumPy array
    * the result is a tuple of NumPy arrays
    * the result is a dictionary of NumPy arrays as values

    ```python
    from sigmaepsilon.math import squeeze
    import numpy as np

    @squeeze(default=True)
    def foo(arr):
        return arr

    foo(np.array([[1, 2]]))  # array([1, 2])
    foo(np.array([[1, 2]]), squeeze=False)  # array([[1, 2]])
    ```
    """

    def decorator(fnc: Callable):
        def inner(*args, squeeze: bool = default, **kwargs):
            if squeeze:
                res = fnc(*args, **kwargs)
                if isinstance(res, tuple):
                    return list(map(squeeze_if_array, res))
                elif isinstance(res, dict):
                    return {k: squeeze_if_array(v) for k, v in res.items()}
                else:
                    return squeeze_if_array(res)
            else:
                return fnc(*args, **kwargs)

        inner.__doc__ = fnc.__doc__
        return inner

    return decorator


"""def config(*args, **kwargs):
    def decorator(fnc: Callable):
        def inner(*args, **kwargs):
            return fnc(*args, **kwargs)
        return inner
    return decorator"""
