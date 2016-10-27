#!/usr/bin/env python

import functions as F
import numpy as N
import unittest

class TestFunctions(unittest.TestCase):
    def testApproxJacobian1(self):
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope)

    def testApproxJacobian2(self):
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        # import pdb; pdb.set_trace()
        Df_x = F.ApproximateJacobian(f, x0, dx)

        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)

    def testPolynomial(self):
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x**2 + 2*x + 3)

    def testLinear1D(self):
        x0 = 21.0
        dx = 1.e-3
        p = F.FuncLinear1D([10, 2])
        AnalJ = p.Df(x0)
        ApproxJ = F.ApproximateJacobian(p,x0,dx)
        self.assertAlmostEqual(AnalJ, ApproxJ)

    def testLinearND(self):
        #the dimension of the linear system depends on the inputs
        A = N.matrix("1. 2. 3.; 3. 4. 3.; 1 1 1")
        x0 = N.matrix("5; 6 ; 7")
        dx = 1.e-3
        p = F.FuncLinearND(A)
        AnalJ = p.Df(x0)
        ApproxJ = F.ApproximateJacobian(p,x0,dx)
        N.testing.assert_array_almost_equal(AnalJ, ApproxJ)

if __name__ == '__main__':
    unittest.main()
