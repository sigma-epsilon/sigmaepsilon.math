"""Standard benchmark test functions used to evaluate optimizers."""

from typing import Iterable

from .function import Function


class TestFunction(Function):
    """Base class for benchmark test functions with known optima."""

    __slots__ = ("optimums", "optText")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimums = []

        self.optText = "opt"
        for key, value in kwargs.items():
            if key == "optText":
                assert isinstance(value, str)
                self.optText = value
                break
            elif key == "optimums":
                assert isinstance(value, Iterable)
                for v in value:
                    self.optimums.append(v)
        return


class TestFunction2D(TestFunction):
    """Base class for two-dimensional benchmark test functions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TestMinFunction(TestFunction):
    """Base class for benchmark test functions with a known minimum."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, optText="min")


class TestMaxFunction(TestFunction):
    """Base class for benchmark test functions with a known maximum."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, optText="max")


class TestMinFunction2D(TestMinFunction, TestFunction2D):
    """Base class for two-dimensional benchmark test functions with a known minimum."""


class TestMaxFunction2D(TestMaxFunction, TestFunction2D):
    """Base class for two-dimensional benchmark test functions with a known maximum."""
