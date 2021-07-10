import unittest
import numpy.testing as nptest
import evaLEs.LE as ly
import numpy as np
from numba import njit

class TestCore(unittest.TestCase):
    """ unittest for core module"""

    @staticmethod
    def test_expon():
        def ODE(t, val, p):
            x = val[0]
            y = val[1]
            diff = [x*p[0], y*p[1]]
            return np.array(diff)
        
        def J(t, val, p):
            D = len(val)
            J = np.eye(D)*(p)
            return J
        p = np.array([-0.01, -0.03], dtype=np.float64)
        init = np.array([0.1, 0.1], dtype=np.float64)
        ttrans = np.arange(0.0, 10, 0.1, dtype=np.float64)
        t = np.arange(0.0, 10, 0.01, dtype=np.float64)
        LE = ly.computeLE(ODE, J, init, t, p, ttrans)
        nptest.assert_allclose(LE[-1], np.array([ -0.01, -0.03 ]))

    @staticmethod
    def test_logistic():
        def logmap(t, val, p):
            mu = p[0]
            x = val[0]
            diff = [mu * x *( 1 - x )]
            return np.array(diff)
        def logmapJac(t, val, p):
            mu = p[0]
            x = val[0]
            J = np.array([mu * (1 - 2 * x)])
            return J
        mu = 4
        param = np.array([mu])
        init = np.array([0.1], dtype=np.float64)
        ttrans = np.arange(0.0, 3000, 1, dtype=np.float64)
        t = np.arange(0.0, 100, 1, dtype=np.float64)
        LE = ly.computeLE(logmap, logmapJac, init, t, param, ttrans, continuous = False)
        nptest.assert_allclose(LE[-1], np.array([ np.log(2)]), atol=0.005)
if __name__ == '__main__':
    unittest.main()
