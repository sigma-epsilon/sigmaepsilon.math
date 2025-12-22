import sympy as sy
from sigmaepsilon.math.optimize.lp import LinearProgrammingProblem
from sigmaepsilon.math.function import Function, InEquality
from sigmaepsilon.math.function.relation import Relations


def make_problem_with_ineq(op):
    x, y = sy.symbols("x y")
    obj = Function(x + y, variables=[x, y])
    # The constraint is x + y op 1
    ineq = InEquality(x + y - 1, op=op, variables=[x, y])
    bounds = [(None, None), (None, None)]
    return LinearProgrammingProblem(
        obj, [ineq], variables=[x, y], bounds=bounds, sparsify=False
    )


def test_lp_gt_constraint():
    # Covers the Relations.gt branch
    problem = make_problem_with_ineq(Relations.gt)
    # Should not raise, and should produce a valid scipy linprog call
    res = problem.solve()
    assert hasattr(res, "success")


def test_lp_lt_constraint():
    # Covers the Relations.lt branch
    problem = make_problem_with_ineq(Relations.lt)
    res = problem.solve()
    assert hasattr(res, "success")
