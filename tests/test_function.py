import unittest
import doctest

import sympy as sy
import numpy as np

import sigmaepsilon.math.function as fnc
from sigmaepsilon.math.function.metafunction import decode, substitute, coefficients
from sigmaepsilon.math.function import Function
from sigmaepsilon.math.function import Equality, InEquality, Relation
from sigmaepsilon.math.approx.lagrange import gen_Lagrange_1d


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(fnc.function))
    tests.addTests(doctest.DocTestSuite(fnc.relation))
    return tests


class TestFunction(unittest.TestCase):
    def assertFailsProperly(self, exc, fnc, *args, **kwargs):
        failed_properly = False
        try:
            fnc(*args, **kwargs)
        except exc:
            failed_properly = True
        finally:
            self.assertTrue(failed_properly)

    def test_bulk(self):
        def f0(x, y):
            return x**2 + y  # pragma: no cover

        def f1(x, y):
            return np.array([2 * x, 1])  # pragma: no cover

        f = Function(f0, f1, d=2)
        self.assertFalse(f.symbolic)
        # f.to_latex()
        f.coefficients()
        f.linear_coefficients()

        Function(value=f0, gradient=f1, Hessian=None, d=2)

        class MyFunction(Function):  # pragma: no cover
            def value(x, y):
                return f0(x, y)

            def gradient(x, y):
                return f1(x, y)

            def Hessian(x, y):
                ...

        f = Function()
        self.assertFailsProperly(TypeError, f, [0, 0])
        self.assertFailsProperly(TypeError, lambda x: x.to_latex(), f)
        self.assertFailsProperly(TypeError, lambda x: x.subs([0, 0]), f)
        self.assertFailsProperly(TypeError, f.g, [0, 0])
        self.assertFailsProperly(TypeError, f.G, [0, 0])

        str_expr = "x*y + y**2 + 6*b + 2"
        decode(str_expr=str_expr)
        self.assertFailsProperly(Exception, decode, expr=[])

        g = Function("3*x + 4*y - 2", variables=["x", "y", "z"])
        g.subs([0, 0, 0], ["x", "y", "z"], inplace=True)

        coefficients(g.expr)
        substitute(g.expr, [0, 0])

    def test_linearity(self):
        def f0(x=None, y=None):
            return x**2 + y  # pragma: no cover

        def f1(x=None, y=None):
            return np.array([2 * x, 1])  # pragma: no cover

        f = Function(f0, f1, d=2)
        self.assertFalse(f.symbolic)
        self.assertTrue(f.linear)

    def test_sym(self):
        f = gen_Lagrange_1d(N=2)
        f1 = Function(f[1][0], f[1][1], f[1][2])
        f2 = Function(f[2][0], f[2][1], f[2][2])
        assert f1.linear and f2.linear
        assert f1.dimension == 1
        assert f2.dimension == 1
        assert np.isclose(f1([-1]), 1.0)
        assert np.isclose(f1([1]), 0.0)
        assert np.isclose(f2([-1]), 0.0)
        assert np.isclose(f2([1]), 1.0)
        f1.coefficients()
        f1.linear_coefficients()
        f1.to_latex()
        f1.f([-1]), f1.g([-1]), f1.G([-1])
        f1.subs([1], variables=f1.variables)


class TestRelations(unittest.TestCase):
    def test_Relation(self):
        variables = ["x1", "x2", "x3", "x4"]
        x1, _, x3, x4 = syms = sy.symbols(variables, positive=True)
        r = Relation(x1 + 2 * x3 + x4 - 4, variables=syms)
        r.operator
        r = Relation(
            x1 + 2 * x3 + x4 - 4, variables=syms, op=lambda x, y: x <= y
        )  # pragma: no cover

    def test_Equality(self):
        variables = ["x1", "x2", "x3", "x4"]
        x1, _, x3, x4 = syms = sy.symbols(variables, positive=True)
        eq1 = Equality(x1 + 2 * x3 + x4 - 4, variables=syms)
        eq1.to_eq()
        eq1.operator

    def test_InEquality(self):
        gt = InEquality("x + y", op=">")
        assert not gt.relate([0.0, 0.0])

        ge = InEquality("x + y", op=">=")
        assert ge.relate([0.0, 0.0])

        le = InEquality("x + y", op=lambda x, y: x <= y)
        assert le.relate([0.0, 0.0])

        lt = InEquality("x + y", op=lambda x, y: x < y)
        assert not lt.relate([0.0, 0.0])

        failed_properly = False
        try:
            InEquality("x + y")
        except ValueError:
            failed_properly = True
        finally:
            self.assertTrue(failed_properly)


if __name__ == "__main__":
    unittest.main()
