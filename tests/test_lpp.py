import unittest
import doctest
import numpy as np

from sigmaepsilon.math.utils import atleast2d
from sigmaepsilon.math.function import Function, Equality, InEquality
from sigmaepsilon.math.optimize import LinearProgrammingProblem as LPP, NoSolutionError
import sympy as sy

from sigmaepsilon.math import optimize


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(optimize.lp))
    return tests


class TestLPP(unittest.TestCase):
    def assertFailsProperly(self, exc, fnc, *args, **kwargs):
        failed_properly = False
        try:
            fnc(*args, **kwargs)
        except exc:
            failed_properly = True
        finally:
            self.assertTrue(failed_properly)

    def test_coverage(self):
        P = LPP.example_unique()
        x1, x2 = sy.symbols(["x1", "x2"], positive=True)
        syms = [x1, x2]
        P.add_constraint(InEquality(x1 - 1, op=">=", variables=syms))
        P.add_constraint(x1 - 1, op=">=", variables=syms)
        P.simplify(inplace=True)

        x1, x2 = sy.symbols(["x1", "x2"], positive=True)
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        lpp = LPP(cost=f, constraints=[ieq1, ieq2, ieq3], variables=syms)
        lpp.feasible([0, 0])

    def test_lpp_create(self):
        x1, x2 = sy.symbols(["x1", "x2"], positive=True)
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        lpp = LPP(f, variables=syms)
        lpp.add_constraint(ieq1)
        lpp.add_constraint(ieq2)
        lpp.add_constraint(ieq3)
        lpp.simplify(inplace=True)

    def test_unique_solution(self):
        x1, x2 = sy.symbols(["x1", "x2"], positive=True)
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        lpp = LPP(cost=f, constraints=[ieq1, ieq2, ieq3], variables=syms)
        x = lpp.solve()["x"]
        _x = np.array([1.0, 1.0])
        self.assertTrue(np.all(np.isclose(_x, x)))
        self.assertTrue(lpp.feasible(x))

    def test_degenerate_solution(self):
        variables = ["x1", "x2", "x3", "x4"]
        x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
        obj2 = Function(3 * x1 + x2 - 6 * x3 + x4, variables=syms)
        eq21 = Equality(x1 + 2 * x3 + x4, variables=syms)
        eq22 = Equality(x2 + x3 - x4 - 2, variables=syms)
        P2 = LPP(cost=obj2, constraints=[eq21, eq22], variables=syms)
        P2.solve(raise_errors=False)

    def test_no_solution(self):
        """
        Example for no solution.
        """
        variables = ["x1", "x2", "x3", "x4"]
        x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
        obj3 = Function(-3 * x1 + x2 + 9 * x3 + x4, variables=syms)
        eq31 = Equality(x1 - 2 * x3 - x4 + 2, variables=syms)
        eq32 = Equality(x2 + x3 - x4 - 2, variables=syms)
        P3 = LPP(cost=obj3, constraints=[eq31, eq32], variables=syms)
        self.assertFailsProperly(NoSolutionError, P3.solve, raise_errors=True)

    def test_multiple_solution(self):
        """
        Example for multiple solutions.
        (0, 1, 1, 0)
        (0, 4, 0, 2)
        """
        variables = ["x1", "x2", "x3", "x4"]
        x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
        obj4 = Function(3 * x1 + 2 * x2 + 8 * x3 + x4, variables=syms)
        eq41 = Equality(x1 - 2 * x3 - x4 + 2, variables=syms)
        eq42 = Equality(x2 + x3 - x4 - 2, variables=syms)
        P4 = LPP(cost=obj4, constraints=[eq41, eq42], variables=syms)
        x = P4.solve(return_all=True, raise_errors=True)["x"]
        assert len(x.shape) == 2
        assert x.shape[0] == 2
        x = P4.solve(return_all=False, raise_errors=True)["x"]
        assert len(x.shape) == 1

    def test_maximize_solution(self):
        x1, x2 = sy.symbols(["x1", "x2"], positive=True)
        syms = [x1, x2]
        f = Function(-x1 - x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        lpp = LPP(cost=f, constraints=[ieq1, ieq2, ieq3], variables=syms)
        x = atleast2d(lpp.maximize()["x"])
        _x = np.array([1.0, 1.0])
        assert np.all(np.isclose(_x, x))

    def test_standardform(self):
        x1, x2 = sy.symbols(["x1", "x2"], positive=True)
        syms = [x1, x2]
        f = Function(-x1 - x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        lpp = LPP(cost=f, constraints=[ieq1, ieq2, ieq3], variables=syms)
        lpp.to_numpy()

    def test_feasible(self):
        x1, x2 = sy.symbols(["x1", "x2"], positive=True)
        syms = [x1, x2]
        f = Function(-x1 - x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        lpp = LPP(cost=f, constraints=[ieq1, ieq2, ieq3], variables=syms)
        self.assertTrue(lpp.feasible([1, 1]))
        self.assertFalse(lpp.feasible([0, 1]))
        self.assertFalse(lpp.feasible([1, 0]))

    def test_1(self):
        variables = ["x1", "x2"]
        x1, x2 = syms = sy.symbols(variables, positive=True)
        f = Function(x1 + x2, variables=syms)
        eq = Equality(x1 - 2, variables=syms)
        lpp = LPP(cost=f, constraints=[eq], variables=syms)
        x = lpp.solve(return_all=True, raise_errors=True)["x"]
        assert np.all(np.isclose(x, np.array([2.0, 0.0])))
        variables = ["x1", "x2"]
        x1, x2 = syms = sy.symbols(variables, positive=True)
        f = Function(x1 + x2, variables=syms)
        eq = Equality(x2 - 2, variables=syms)
        lpp = LPP(cost=f, constraints=[eq], variables=syms)
        x = lpp.solve(return_all=True, raise_errors=True)["x"]
        assert np.all(np.isclose(x, np.array([0.0, 2.0])))

    def test_2(self):
        x1, x2 = sy.symbols(["x1", "x2"], positive=True)
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        lpp = LPP(cost=f, constraints=[ieq1, ieq2, ieq3], variables=syms)
        x = lpp.solve(return_all=True, raise_errors=True)["x"]
        assert np.all(np.isclose(x, np.array([1.0, 1.0])))
        x1, x2 = sy.symbols(["x1", "x2"], positive=True)
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op=">=", variables=syms)
        lpp = LPP(cost=f, constraints=[ieq1, ieq2, ieq3], variables=syms)
        e = lpp.solve(return_all=True, raise_errors=True, as_dict=True)["e"]
        assert len(e) == 0


if __name__ == "__main__":
    unittest.main()
