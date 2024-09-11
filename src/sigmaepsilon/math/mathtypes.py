from typing import TypeAlias, Sequence
from sympy import Symbol

__all__ = ["NumberLike", "BoundsLike"]

NumberLike: TypeAlias = int | float
# type NumberLike = int | float  # after Python 3.12 and higher

BoundsLike: TypeAlias = (
    Sequence[tuple[NumberLike | None, NumberLike | None]]
    | dict[Symbol, tuple[NumberLike | None, NumberLike | None]]
    | None
)
