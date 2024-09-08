import unittest
import doctest
import numpy as np

from sigmaepsilon.math.function import Function, Equality, InEquality, Relation
from sigmaepsilon.math.optimize import (
    LinearProgrammingProblem as LPP,
    NoSolutionError,
    OverDeterminedError,
)
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
        x1, x2 = sy.symbols(["x1", "x2"], nonnegative=True)
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms)
        lpp.is_feasible([0, 0])

    def test_raise_no_variables_sym(self):
        x1, x2 = syms = sy.symbols(["x1", "x2"], nonnegative=True)
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        with self.assertRaises(ValueError):
            LPP(f, [ieq1, ieq2, ieq3])
            
    def test_raise_not_symbolic_input(self):
        x1, x2 = syms = sy.symbols(["x1", "x2"], nonnegative=True)
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        with self.assertRaises(ValueError):
            LPP(f, [ieq1, Relation(lambda x:x, op=">=")], variables=syms)
        with self.assertRaises(ValueError):
            LPP(Function(lambda x:x), [ieq1], variables=syms)
        with self.assertRaises(TypeError):
            LPP(f, [ieq1, lambda x:x], variables=syms)
        with self.assertRaises(TypeError):
            LPP(lambda x:x, [ieq1], variables=syms)
            
    def test_raise_invalid_variables(self):
        x1, x2 = syms = sy.symbols(["x1", "x2"], nonnegative=True)
        x3 = sy.symbols("x3", nonnegative=True)
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        with self.assertRaises(TypeError):
            LPP(f, [ieq1, ieq2, ieq3], variables=[x1, x2, 5])
        with self.assertRaises(ValueError):
            LPP(f, [ieq1, ieq2, ieq3], variables=[x1, x2, x3])
    
    def test_unique_solution(self):
        x1, x2 = syms = sy.symbols(["x1", "x2"], nonnegative=True)
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms)
        x = lpp.solve().x
        x_ = np.array(x)
        _x = np.array([1.0, 1.0])
        self.assertTrue(np.all(np.isclose(_x, x_)))
        self.assertTrue(lpp.is_feasible(x_))

    def test_degenerate_solution(self):
        variables = ["x1", "x2", "x3", "x4"]
        x1, x2, x3, x4 = syms = sy.symbols(variables, nonnegative=True)
        obj2 = Function(3 * x1 + x2 - 6 * x3 + x4, variables=syms)
        eq21 = Equality(x1 + 2 * x3 + x4, variables=syms)
        eq22 = Equality(x2 + x3 - x4 - 2, variables=syms)
        P2 = LPP(obj2, [eq21, eq22], variables=syms)
        P2.solve(raise_errors=False)

    def test_no_solution(self):
        """
        Example for no solution.
        """
        variables = ["x1", "x2", "x3", "x4"]
        x1, x2, x3, x4 = syms = sy.symbols(variables, nonnegative=True)
        obj3 = Function(-3 * x1 + x2 + 9 * x3 + x4, variables=syms)
        eq31 = Equality(x1 - 2 * x3 - x4 + 2, variables=syms)
        eq32 = Equality(x2 + x3 - x4 - 2, variables=syms)
        P3 = LPP(obj3, [eq31, eq32], variables=syms)
        self.assertFailsProperly(NoSolutionError, P3.solve, raise_errors=True)

    def test_overdetermined_problem(self):
        syms = x1, x2 = sy.symbols(["X2", "x2"], nonnegative=True)
        f = Function(x1 + 3 * x2 - 5, variables=syms)
        eq1 = Relation(x1 + x2 - 6, op="=", variables=syms)
        eq2 = Relation(x1 - x2 - 2, op="=", variables=syms)
        eq3 = Relation(x1 - x2 + 4, op="=", variables=syms)
        lpp = LPP(f, [eq1, eq2, eq3], variables=syms)
        with self.assertRaises(OverDeterminedError):
            lpp.solve(raise_errors=True)

    def test_multiple_solution(self):
        """
        Example for multiple solutions.
        (0, 1, 1, 0)
        (0, 4, 0, 2)
        """
        variables = ["x1", "x2", "x3", "x4"]
        x1, x2, x3, x4 = syms = sy.symbols(variables, nonnegative=True)
        obj4 = Function(3 * x1 + 2 * x2 + 8 * x3 + x4, variables=syms)
        eq41 = Equality(x1 - 2 * x3 - x4 + 2, variables=syms)
        eq42 = Equality(x2 + x3 - x4 - 2, variables=syms)
        P4 = LPP(obj4, [eq41, eq42], variables=syms)
        x = P4.solve(return_all=True, raise_errors=True).x

        x_ = np.array(x)
        assert x_.shape == (2, 4)

        x = P4.solve(return_all=False, raise_errors=True).x
        x_ = np.array(x)
        assert len(x_.shape) == 1

    def test_maximize_solution(self):
        x1, x2 = sy.symbols(["x1", "x2"], nonnegative=True)
        syms = [x1, x2]
        f = Function(-x1 - x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms)
        x = lpp.solve(maximize=True).x
        x_ = np.array(x)
        _x = np.array([1.0, 1.0])
        assert np.all(np.isclose(_x, x_))

    def test_feasible(self):
        x1, x2 = sy.symbols(["x1", "x2"], nonnegative=True)
        syms = [x1, x2]
        f = Function(-x1 - x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms)
        self.assertTrue(lpp.is_feasible([1, 1]))
        self.assertFalse(lpp.is_feasible([0, 1]))
        self.assertFalse(lpp.is_feasible([1, 0]))

    def test_1(self):
        variables = ["x1", "x2"]
        x1, x2 = syms = sy.symbols(variables, nonnegative=True)
        f = Function(x1 + x2, variables=syms)
        eq = Equality(x1 - 2, variables=syms)
        lpp = LPP(f, [eq], variables=syms)
        x = lpp.solve(return_all=True).x
        x_ = np.array(x)
        assert np.all(np.isclose(x_, np.array([2.0, 0.0])))
        variables = ["x1", "x2"]
        x1, x2 = syms = sy.symbols(variables, nonnegative=True)
        f = Function(x1 + x2, variables=syms)
        eq = Equality(x2 - 2, variables=syms)
        lpp = LPP(f, [eq], variables=syms)
        x = lpp.solve(return_all=True).x
        x_ = np.array(x)
        assert np.all(np.isclose(x_, np.array([0.0, 2.0])))

    def test_2(self):
        x1, x2 = sy.symbols(["x1", "x2"], nonnegative=True)
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms)
        x = lpp.solve(return_all=True).x
        x_ = np.array(x)
        assert np.all(np.isclose(x_, np.array([1.0, 1.0])))
        x1, x2 = sy.symbols(["x1", "x2"], nonnegative=True)
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op=">=", variables=syms)
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms)
        errors = lpp.solve(return_all=True).errors
        assert len(errors) == 0


if __name__ == "__main__":
    unittest.main()
