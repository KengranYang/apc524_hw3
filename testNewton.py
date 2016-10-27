#!/usr/bin/env python

import newton
import unittest
import numpy as N
import functions as F

class TestNewton(unittest.TestCase):
    def testLinear(self):
        """Test Newton root finder with 1D linear function"""
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertEqual(x, -2.0)

    def test2DLinear (self):
        """Test Newton root finder with 2D linear function"""
        A = N.matrix("1. 1.; 0. 4.")
        b = N.matrix("-1; -1")
        def f(x):
            return A * x + b
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(N.matrix("10; 1"))
        x_soln = N.linalg.solve(N.matrix(A), N.matrix(-b))
        N.testing.assert_array_almost_equal(x, x_soln)

    def testPolynomial1D(self):
        """Test Newton root finder with 1D polynomial function"""
        f = F.Polynomial([1,1,0])
        solver = newton.Newton(f, tol=1.e-15, maxiter=100)
        x = solver.solve(-0.51)
        self.assertAlmostEqual(x, -1)

    def testPolynomial2D(self):
        """Test Newton root finder with 2D polynomial function"""
        def f(x):
            fx = N.matrix(N.zeros((2,1)))
            fx[0] = x[0]**2 - x[1]
            fx[1] = x[1] - x[0]*2
            return fx
        solver = newton.Newton(f, tol=1.e-15, maxiter=7)
        x0 = N.matrix("5.1; 5")
        x_exact = N.matrix("2; 4")
        x_soln = solver.solve(x0)
        N.testing.assert_array_almost_equal(x_exact, x_soln)

    def testConvergence(self):
        """ Test if an exception is raised when the method fails to converge after the maximum number of iterations."""
        f = F.Polynomial([1, 0, 0])
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        self.assertRaises(Exception, solver.solve, 2.)


    def testSingleStep(self):
        """ A single step of the Newton method performs as it should."""
        f = F.Polynomial([1, 0, 0])
        solver = newton.Newton(f, tol=1.e-15, maxiter=10)
        x = solver.step(2.0)
        self.assertLessEqual(N.linalg.norm(f(x)), N.linalg.norm(f(2.)) )

    def testIfAnalyticJacobianIsUsed(self):
        """Test if the Analytical Jacobian is used """
        f = lambda x : 3.0 * x + 6.0
        df = lambda x : 3.0
        #it should use the analytical Jacobian since it is declared
        solver = newton.Newton(f, Df=df, tol=1.e-15, maxiter=2)
        x = solver.solve(2.0)
        self.assertFalse(solver._ApproxJacobianFlag)

    def testWithinRadius1D(self):
        """Test if the approximated root lies within a radius r in 1D"""
        f = lambda x : 3.0 * x + 6.0
        df = lambda x : 3.0
        solver = newton.Newton(f, Df=df, tol=1.e-15, maxiter=2, r=1000)
        # this test will pass if the approximated root is outside the radius
        self.assertRaises(Exception, solver.solve, 999)

    def testWithinRadius2D(self):
        """Test if the approximated root lies within a radius r in 2D"""
        def f(x):
            fx = N.matrix(N.zeros((2,1)))
            fx[0] = x[0]**2 - x[1]
            fx[1] = x[1] - x[0]*2
            return fx
        solver = newton.Newton(f, tol=1.e-15, maxiter=7, r=1)
        x0 = N.matrix("5.1; 5")
        self.assertRaises(Exception, solver.solve, x0)

if __name__ == "__main__":
    unittest.main()
