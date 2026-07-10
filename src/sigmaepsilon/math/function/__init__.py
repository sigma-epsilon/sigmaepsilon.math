# -*- coding: utf-8 -*-
"""Function and relation objects used to model optimization problems."""
from .function import Function, FunctionLike
from .relation import Equality, InEquality, Relation

__all__ = [
    "Function",
    "FunctionLike",
    "Equality",
    "InEquality",
    "Relation",
]
