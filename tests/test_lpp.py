import unittest
import numpy as np

from sigmaepsilon.math.function import Function, Equality, InEquality, Relation
from sigmaepsilon.math.optimize import LinearProgrammingProblem as LPP
import sympy as sy


class TestLPP(unittest.TestCase):

    def test_raise_no_variables_sym(self):
        x1, x2 = syms = sy.symbols(["x1", "x2"])
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        with self.assertRaises(ValueError):
            LPP(f, [ieq1, ieq2, ieq3])

    def test_equivalent_bounds_inputs(self):
        syms = x1, x2 = sy.symbols(["x1", "x2"])
        f = Function(x1 + x2 - 5, variables=syms)
        x_ = np.array([1.0, 1.0])

        ieq = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        bounds = [(1, None), (1, None)]
        lpp = LPP(f, [ieq], variables=syms, bounds=bounds)
        solution = lpp.solve()
        x = np.array(solution.x)
        self.assertTrue(np.all(np.isclose(x, x_)))
        self.assertTrue(np.isclose(solution.fun, -3.0))

        ieq = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        bounds = (1, None)
        lpp = LPP(f, [ieq], variables=syms, bounds=bounds)
        solution = lpp.solve()
        x = np.array(solution.x)
        self.assertTrue(np.all(np.isclose(x, x_)))
        self.assertTrue(np.isclose(solution.fun, -3.0))

        ieq1 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        ieq2 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x2 - 1, op=">=", variables=syms)
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms)
        solution = lpp.solve()
        x = np.array(solution.x)
        self.assertTrue(np.all(np.isclose(x, x_)))
        self.assertTrue(np.isclose(solution.fun, -3.0))

        ieq1 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        ieq2 = InEquality("x1 >= 1", variables=syms)
        ieq3 = InEquality("x2 >= 1", variables=syms)
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms)
        solution = lpp.solve()
        x = np.array(solution.x)
        self.assertTrue(np.all(np.isclose(x, x_)))
        self.assertTrue(np.isclose(solution.fun, -3.0))

    def test_raise_not_symbolic_input(self):
        x1, x2 = syms = sy.symbols(["x1", "x2"])
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        bounds = [(0, None), (0, None)]
        with self.assertRaises(ValueError):
            LPP(
                f, [ieq1, Relation(lambda x: x, op=">=")], variables=syms, bounds=bounds
            )
        with self.assertRaises(ValueError):
            LPP(Function(lambda x: x), [ieq1], variables=syms, bounds=bounds)
        with self.assertRaises(TypeError):
            LPP(f, [ieq1, lambda x: x], variables=syms, bounds=bounds)
        with self.assertRaises(TypeError):
            LPP(lambda x: x, [ieq1], variables=syms, bounds=bounds)

    def test_raise_invalid_variables(self):
        x1, x2 = syms = sy.symbols(["x1", "x2"])
        x3 = sy.symbols("x3", nonnegative=True)
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        bounds = [(0, None), (0, None)]
        with self.assertRaises(TypeError):
            LPP(f, [ieq1, ieq2, ieq3], variables=[x1, x2, 5], bounds=bounds)
        with self.assertRaises(ValueError):
            LPP(f, [ieq1, ieq2, ieq3], variables=[x1, x2, x3], bounds=bounds)

    def test_unique_solution(self):
        x1, x2 = syms = sy.symbols(["x1", "x2"])
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        bounds = [(0, None), (0, None)]
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms, bounds=bounds)
        x = lpp.solve().x
        x_ = np.array(x)
        _x = np.array([1.0, 1.0])
        self.assertTrue(np.all(np.isclose(_x, x_)))
        # test solution using dense matrices
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms, bounds=bounds, sparsify=False)
        x = lpp.solve().x
        x_ = np.array(x)
        _x = np.array([1.0, 1.0])
        self.assertTrue(np.all(np.isclose(_x, x_)))

    def test_no_solution(self):
        """
        Example for no solution.
        """
        variables = ["x1", "x2", "x3", "x4"]
        x1, x2, x3, x4 = syms = sy.symbols(variables)
        obj3 = Function(-3 * x1 + x2 + 9 * x3 + x4, variables=syms)
        eq31 = Equality(x1 - 2 * x3 - x4 + 2, variables=syms)
        eq32 = Equality(x2 + x3 - x4 - 2, variables=syms)
        bounds = [(0, None), (0, None), (0, None), (0, None)]
        lpp = LPP(obj3, [eq31, eq32], variables=syms, bounds=bounds)
        solution = lpp.solve()
        self.assertFalse(solution.success)
        # test solution using dense matrices
        lpp = LPP(obj3, [eq31, eq32], variables=syms, bounds=bounds, sparsify=False)
        solution = lpp.solve()
        self.assertFalse(solution.success)

    def test_overdetermined_problem(self):
        syms = x1, x2 = sy.symbols(["X2", "x2"])
        f = Function(x1 + 3 * x2 - 5, variables=syms)
        eq1 = Relation(x1 + x2 - 6, op="=", variables=syms)
        eq2 = Relation(x1 - x2 - 2, op="=", variables=syms)
        eq3 = Relation(x1 - x2 + 4, op="=", variables=syms)
        bounds = [(0, None), (0, None)]
        lpp = LPP(f, [eq1, eq2, eq3], variables=syms, bounds=bounds)
        solution = lpp.solve()
        self.assertFalse(solution.success)
        self.assertEqual(solution.status, 2)
        self.assertIn("HiGHS Status 8", solution.message)

    def test_multiple_solution(self):
        """
        Example for multiple solutions.
        (0, 1, 1, 0)
        (0, 4, 0, 2)
        """
        variables = ["x1", "x2", "x3", "x4"]
        x1, x2, x3, x4 = syms = sy.symbols(variables)
        obj4 = Function(3 * x1 + 2 * x2 + 8 * x3 + x4, variables=syms)
        eq41 = Equality(x1 - 2 * x3 - x4 + 2, variables=syms)
        eq42 = Equality(x2 + x3 - x4 - 2, variables=syms)
        bounds = [(0, None), (0, None), (0, None), (0, None)]
        lpp = LPP(obj4, [eq41, eq42], variables=syms, bounds=bounds)
        solution = lpp.solve()
        self.assertTrue(solution.success)

    def test_maximize_solution(self):
        x1, x2 = sy.symbols(["x1", "x2"])
        syms = [x1, x2]
        f = Function(-x1 - x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        bounds = [(0, None), (0, None)]
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms, bounds=bounds)
        solution = lpp.solve(maximize=True)
        x_ = np.array(solution.x)
        _x = np.array([1.0, 1.0])
        assert np.all(np.isclose(_x, x_))

    def test_mixed_integer_problem(self):
        variables = x1, x2, x3 = sy.symbols(["x1", "x2", "x3"])
        f = Function(3 * x2 + 2 * x3, variables=variables)
        eq1 = Relation(2 * x1 + 2 * x2 - 4 * x3 - 5, op="=", variables=variables)
        bounds = (0, None)
        integrality = [1, 0, 1]
        lpp = LPP(f, [eq1], variables=variables, bounds=bounds, integrality=integrality)
        solution = lpp.solve()
        self.assertTrue(solution.success)
        x = np.array(solution.x)
        self.assertTrue(np.all(np.isclose(x, np.array([2, 0.5, 0]))))
        self.assertTrue(np.isclose(solution.fun, 1.5))
        
    def test_mixed_integer_problem_dense(self):
        variables = x1, x2, x3 = sy.symbols(["x1", "x2", "x3"])
        f = Function(3 * x2 + 2 * x3, variables=variables)
        eq1 = Relation(2 * x1 + 2 * x2 - 4 * x3 - 5, op="=", variables=variables)
        bounds = (0, None)
        integrality = [1, 0, 1]
        lpp = LPP(f, [eq1], variables=variables, bounds=bounds, integrality=integrality, sparsify=False)
        solution = lpp.solve()
        self.assertTrue(solution.success)
        x = np.array(solution.x)
        self.assertTrue(np.all(np.isclose(x, np.array([2, 0.5, 0]))))
        self.assertTrue(np.isclose(solution.fun, 1.5))

    def test_cyclic_problem(self):
        syms = x1, x2 = sy.symbols(["X2", "x2"])
        f = Function(x1 + 2 * x2, variables=syms)
        eq1 = Relation(x1 + x2 - 1, op="<=", variables=syms)
        bounds = [(0, 1), (0, None)]
        lpp = LPP(f, [eq1], variables=syms, bounds=bounds)
        solution = lpp.solve()
        self.assertTrue(solution.success)

    def test_1(self):
        variables = ["x1", "x2"]
        x1, x2 = syms = sy.symbols(variables)
        f = Function(x1 + x2, variables=syms)
        eq = Equality(x1 - 2, variables=syms)
        bounds = [(0, None), (0, None)]
        lpp = LPP(f, [eq], variables=syms, bounds=bounds)
        x = lpp.solve().x
        x_ = np.array(x)
        assert np.all(np.isclose(x_, np.array([2.0, 0.0])))
        variables = ["x1", "x2"]
        x1, x2 = syms = sy.symbols(variables)
        f = Function(x1 + x2, variables=syms)
        eq = Equality(x2 - 2, variables=syms)
        lpp = LPP(f, [eq], variables=syms, bounds=bounds)
        x = lpp.solve().x
        x_ = np.array(x)
        assert np.all(np.isclose(x_, np.array([0.0, 2.0])))

    def test_2(self):
        x1, x2 = sy.symbols(["x1", "x2"])
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op="<=", variables=syms)
        bounds = [(0, None), (0, None)]
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms, bounds=bounds)
        x = lpp.solve().x
        x_ = np.array(x)
        assert np.all(np.isclose(x_, np.array([1.0, 1.0])))

        x1, x2 = sy.symbols(["x1", "x2"])
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op=">=", variables=syms)
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms, bounds=bounds)
        success = lpp.solve().success
        self.assertTrue(success)
        
        # test solution using dense matrices
        x1, x2 = sy.symbols(["x1", "x2"])
        syms = [x1, x2]
        f = Function(x1 + x2, variables=syms)
        ieq1 = InEquality(x1 - 1, op=">=", variables=syms)
        ieq2 = InEquality(x2 - 1, op=">=", variables=syms)
        ieq3 = InEquality(x1 + x2 - 4, op=">=", variables=syms)
        lpp = LPP(f, [ieq1, ieq2, ieq3], variables=syms, bounds=bounds, sparsify=False)
        success = lpp.solve().success
        self.assertTrue(success)

    def test_result_arbitrary_gradient(self):
        """
        Generate a few simple linear problems with unit gradients in the positive
        quadrant and zero bias and check if the result is always correct.
        """
        x_ = np.array([0.0, 0.0])
        obj_ = 0.0
        for _ in range(5):
            mx, my = np.random.rand(2) * 10
            variables = x1, x2 = sy.symbols(["x1", "x2"])
            f = Function(mx * x1 + my * x2, variables=variables)
            ieq1 = InEquality(x1, op=">=", variables=variables)
            ieq2 = InEquality(x2, op=">=", variables=variables)
            lpp = LPP(f, [ieq1, ieq2], variables=variables)
            solution = lpp.solve()
            x = np.array(solution.x)
            self.assertTrue(np.all(np.isclose(x, x_)))
            self.assertTrue(np.isclose(solution.fun, obj_))


if __name__ == "__main__":
    unittest.main()
