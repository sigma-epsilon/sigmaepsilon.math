import collections

from .function import Function


class TestFunction(Function):
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
                assert isinstance(value, collections.Iterable)
                for v in value:
                    self.optimums.append(v)
        return


class TestFunction2D(TestFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, d=2)


class TestMinFunction(TestFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, optText="min")


class TestMaxFunction(TestFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, optText="max")


class TestMinFunction2D(TestMinFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, d=2)


class TestMaxFunction2D(TestMaxFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, d=2)
