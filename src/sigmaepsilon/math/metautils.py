from typing import Any, Type


def _new_and_init_(cls: Type, *args: Any, **kwargs: Any) -> Any:
    """
    Create a new instance of a class and initialize it.
    """
    obj = cls.__new__(cls, *args, **kwargs)
    obj.__init__(*args, **kwargs)
    return obj
