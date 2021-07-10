"""
Calculate the Lyapunov exponents for a set of ODEs using the method described 
in Sandri (1996), through the use of the variational matrix.
"""
import numpy as np
from evaLEs.utils import motion, benedettin

def computeLE(func, fjac, x0, t, p=(), ttrans=None, continuous = True):
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
    :type continuos: numpy array.
    :param continuos: (optional) If set True the algorithm assumes a continue dynamical system (and integrate with RG4). Else it will assume a discrete dynamical system (map) and just uses the step defined in func.
        If not specified continuos dynamical system is assumed.
    :type continuos: boolean.
    :return: Return the Lyapunov Spectrum evaluated at each instant in a numpy array.
    :rtype: numpy array.
    """
    if ttrans is not None:
        x0 = motion(func, ttrans, x0, p, continuous)[-1]
        
    # start LE calculation
    final_LE = benedettin(func, fjac, x0, t, p, ttrans, continuous)
    return final_LE
