#!/usr/bin/env python

import functions as F
import numpy as N
import unittest

class TestFunctions(unittest.TestCase):
    def testApproxJacobian1(self):
        """ Test accuracy of approx Jacobian using 1D linear function"""
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope)

    def testApproxJacobian2(self):
        """ Test accuracy of approx Jacobian using 2D linear function"""
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        # import pdb; pdb.set_trace()
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)

    def testApproxJacobianPolynomial1D(self):
        """ Test accuracy of approx Jacobian using 1D polynomial function"""
        f = F.Polynomial([1,1,0])
        x0 = 20.0
        dx = 1.e-7
        Df_x = F.ApproximateJacobian(f, x0, dx)
        slope = 2*x0+1
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope, places = 5)

    def testPolynomial(self):
        """Test Polynomial class"""
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x**2 + 2*x + 3)

    def testLinear1D(self):
        """ Test accuracy of analytical Jacobian using 1D linear function"""
        x0 = 21.0
        dx = 1.e-3
        p = F.FuncLinear1D([10, 2])
        AnalJ = p.Df(x0)
        ApproxJ = F.ApproximateJacobian(p,x0,dx)
        self.assertAlmostEqual(AnalJ, ApproxJ)

    def testLinearND(self):
        """ Test accuracy of analytical Jacobian using nD linear function
            The dimension of the linear system depends on the inputs"""

        A = N.matrix("1. 2. 3.; 3. 4. 3.; 1 1 1")
        x0 = N.matrix("5; 6 ; 7")
        dx = 1.e-3
        p = F.FuncLinearND(A)
        AnalJ = p.Df(x0)
        ApproxJ = F.ApproximateJacobian(p,x0,dx)
        N.testing.assert_array_almost_equal(AnalJ, ApproxJ)

    def testNonLinear3D(self):
        """ Test accuracy of analytical Jacobian using 3D non-linear function"""
        x0 = N.matrix("5; 6 ; 7")
        p = F.FuncNonLinear3D()
        AnalJ = p.Df(x0)
        ApproxJ = F.ApproximateJacobian(p,x0,1.e-6)
        N.testing.assert_array_almost_equal(AnalJ, ApproxJ)

if __name__ == '__main__':
    unittest.main()
