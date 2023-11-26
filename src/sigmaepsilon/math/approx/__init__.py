from .mls import moving_least_squares, least_squares, weighted_least_squares
from .func import CubicWeightFunction, ConstantWeightFunction, SingularWeightFunction
from .lagrange import gen_Lagrange_1d, approx_Lagrange_1d


__all__ = [
    "moving_least_squares",
    "least_squares",
    "weighted_least_squares",
    "CubicWeightFunction",
    "ConstantWeightFunction",
    "SingularWeightFunction",
    "gen_Lagrange_1d",
    "approx_Lagrange_1d",
]
