"""
Calculate the Lyapunov exponents for a set of ODEs using the method described 
in Sandri (1996), through the use of the variational matrix.
"""
import numpy as np
from evaLEs.utils import RK4, varRK4

def computeLE(func, fjac, x0, t, p=(), ttrans=None):
    """
    Computes the global Lyapunov exponents for a set of ODEs.

    :param f: ODE function. Must take arguments like f(t, x, p) where x and t are the state and
        time *now*, and p is a tuple of parameters. If there are no model paramters, p should be set to the empty tuple.
    :type f: function
    :param fjac: Jacobian of f.
    :type fjac: function.
    :param x0: Initial position for calculation. Integration of transients will begin from this point.
    :type x0: numpy array.
    :param t: Array of times over which to calculate LE.
    :type t: numpy array.
    :param p: (optional) Tuple of model parameters for f.
    :type p: float, numpy array or empty tuple.
    :param ttrans: (optional) Times over which to integrate transient behavior. 
        If not specified, assumes trajectory is on the attractor.
    :type ttrans: numpy array.
    :return: Return the Lyapunov Spectrum evaluated at each instant in a numpy array.
    :rtype: numpy array.
    """
    D = len(x0)
    N = len(t)
    if ttrans is not None:
        Ntrans = len(ttrans)
    dt = t[1] - t[0]

    # integrate transient behavior
    Phi0 = np.eye(D, dtype=np.float64).flatten() # Identità di tipo float

    
    if ttrans is not None:
        xi = x0
        for i,(t1,t2) in enumerate(zip(ttrans[:-1], ttrans[1:])):
            xip1 = xi + RK4(func, xi, t1, t2, p)
            xi = xip1
        x0 = xi
        
    # start LE calculation
    
    LE = np.zeros((N-1, D), dtype=np.float64)
    final_LE = np.zeros((N-1, D), dtype=np.float64)
    LE_aux = np.zeros((N-1, D), dtype=np.float64)
    Ssol = np.zeros((N, D*(D+1)), dtype=np.float64)
    Ssol[0] = np.append(x0, Phi0)
    for i,(t1,t2) in enumerate(zip(t[:-1], t[1:])):
        Ssol_temp = Ssol[i] + varRK4(func, fjac, Ssol[i], t1, t2, p, D)
        # perform QR decomposition on Phi
        rPhi = np.reshape(Ssol_temp[D:], (D, D))
        Q,R = np.linalg.qr(rPhi)
        Ssol[i+1] = np.append(Ssol_temp[:D], Q.flatten())
        LE[i] = np.abs(np.diag(R))
        logLE = np.log(LE[i])
        if i > 0:
            LE_aux[i, :] = LE_aux[i-1, :] + logLE
        else:
            LE_aux[i, :] = logLE
        final_LE[i] = LE_aux[i]/(t2)
    return final_LE
