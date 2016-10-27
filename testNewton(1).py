#!/usr/bin/env python

import newton
import unittest
import numpy as N
import functions as F

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

    def test2DLinear (self):
        A = N.matrix("1. 1.; 0. 4.")
        b = N.matrix("-1; -1")
        def f(x):
            return A * x + b
        x_soln = N.linalg.solve(N.matrix(A), N.matrix(-b))
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(N.matrix("10; 1"))
        N.testing.assert_array_almost_equal(x, x_soln)

    def testConvergence(self):
        f = F.Polynomial([1, 0, 0])
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)

        # self.assertAlmostEqual(x, 0)
        self.assertRaises(Exception, solver.solve, 2.)

    # A single step of the Newton method performs as it should.
    def testSingleStep(self):
        f = F.Polynomial([1, 0, 0])
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.step(2.0)
        self.assertLessEqual(N.linalg.norm(f(x)), N.linalg.norm(f(2.)) )

    def testIfAnalyticJacobianIsUsed(self):
        f = lambda x : 3.0 * x + 6.0
        df = lambda x : 3.0
        #it should use the analytical Jacobian since it is declared
        solver = newton.Newton(f, Df=df, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertFalse(solver._ApproxJacobianFlag)

    def testWithinRadius(self):
        f = lambda x : 3.0 * x + 6.0
        df = lambda x : 3.0
        solver = newton.Newton(f, Df=df, tol=1.e-15, maxiter=2, r=1000)
        # this test will pass if the approximated root is outside the radius 
        self.assertRaises(Exception, solver.solve, 999)

if __name__ == "__main__":
    unittest.main()
