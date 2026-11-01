"""Numba types and typed containers, with pure Python fallbacks.

Without Numba, :meth:`Dict.empty` returns a plain ``dict`` and the types
are ``None``, which is all the pure Python versions of the jitted
functions need.
"""

from sigmaepsilon.core.jit import NUMBA_AVAILABLE

__all__ = ["Dict", "int64", "int64_1d", "int64_2d", "int64_1d_c"]


if NUMBA_AVAILABLE:
    from numba.core.types import int64
    from numba.typed import Dict

    int64_1d = int64[:]
    int64_2d = int64[:, :]
    int64_1d_c = int64[::1]
else:
    int64 = int64_1d = int64_2d = int64_1d_c = None

    class Dict(dict):
        """Pure Python stand-in for :class:`numba.typed.Dict`."""

        @staticmethod
        def empty(key_type=None, value_type=None) -> dict:
            """Return an empty dictionary. The types are ignored."""
            return {}
