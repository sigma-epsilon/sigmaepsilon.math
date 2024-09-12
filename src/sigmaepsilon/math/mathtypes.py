from typing import TypeAlias, Sequence
from sympy import Symbol

__all__ = ["NumberLike", "BoundsLike"]

NumberLike: TypeAlias = int | float
# type NumberLike = int | float  # after Python 3.12 and higher

BoundLike: TypeAlias = tuple[NumberLike | None, NumberLike | None]

BoundsLike: TypeAlias = BoundLike | Sequence[BoundLike] | dict[Symbol, BoundLike]
