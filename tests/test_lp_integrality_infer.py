import sympy as sy
import numpy as np
from sigmaepsilon.math.optimize.lp import LinearProgrammingProblem
from sigmaepsilon.math.function import Function, Relation
from sigmaepsilon.math.function.relation import Relations


def test_lp_integrality_inferred_from_symbolic_vars():
    # x is integer, y is not
    x = sy.symbols("x", integer=True)
    y = sy.symbols("y")
    obj = Function(x + y, variables=[x, y])
    # Simple constraint: x + y >= 1
    ineq = Relation(x + y - 1, op=Relations.ge, variables=[x, y])
    bounds = [(None, None), (None, None)]
    # Do not pass integrality, so it is inferred from variable assumptions
    problem = LinearProgrammingProblem(
        obj, [ineq], variables=[x, y], bounds=bounds, sparsify=False
    )
    _, kwargs = problem._to_scipy()
    # The integrality array should be [1, 0] (x is integer, y is not)
    assert "integrality" in kwargs
    assert np.array_equal(kwargs["integrality"], np.array([1, 0]))
