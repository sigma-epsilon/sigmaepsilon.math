from typing import TypeAlias, Sequence

__all__ = ["Scalar", "BoundLike", "BoundsLike"]

Scalar: TypeAlias = int | float
# type NumberLike = int | float  # after Python 3.12 and higher

BoundLike: TypeAlias = tuple[Scalar | None, Scalar | None]

BoundsLike: TypeAlias = BoundLike | Sequence[BoundLike]
