import unittest
import numpy.testing as nptest
import evaLEs.numbaLE as ly
import numpy as np
from numba import njit

@njit
def ODE(t, val, p):
    """
    Questa funzione definisce gli incrementi secondo il formato utile a 'solve_ivp' di scykit_learn.
    """
    x = val[0]
    y = val[1]
    diff = [x*p[0], y*p[1]]
    return np.array(diff)
@njit
def J(t, val, p):
    D = len(val)
    J = np.eye(D)*(p)
    return J


class TestCore(unittest.TestCase):
    """ unittest for core module"""

    @staticmethod
    def test_lyap():
        p = np.array([-0.01, -0.03], dtype=np.float64)
        init = np.array([0.1, 0.1], dtype=np.float64)
        ttrans = np.arange(0.0, 10, 0.1, dtype=np.float64)
        t = np.arange(0.0, 10, 0.01, dtype=np.float64)
        LE = ly.computeLE(ODE, J, init, t, p, ttrans)
        nptest.assert_allclose(LE[-1], np.array([ -0.01, -0.03 ]))

if __name__ == '__main__':
    unittest.main()
