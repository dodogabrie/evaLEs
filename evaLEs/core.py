"""
Calculate the Lyapunov exponents for a set of ODEs using the method described 
in Sandri (1996), through the use of the variational matrix.
"""

import numpy as np
from numba import jit, njit, prange

@njit
def motion(func, t, x, p):
    for i,(t1,t2) in enumerate(zip(t[:-1], t[1:])):
        x[i+1] = x[i] + RK4(func, x[i], t1, t2, p)
    return x

@njit
def RK4(f, x, t1, t2, p):
    """
    Fourth-order, 4-step RK routine.
    Returns the step, i.e. approximation to the integral.
    If x is defined at time t_1, then stim should be an array of
    stimulus values at times t_1, (t_1+t_2)/2, and t_2 (i.e. at t1 and t2, as
    well as at the midpoint).
    Alternatively, stim may be a function pointer.
    """
    tmid = (t1 + t2)/2
    dt = t2 - t1

    K1 = f(t1, x, p)
    K2 = f(tmid, x + dt*K1/2, p)
    K3 = f(tmid, x + dt*K2/2, p)
    K4 = f(t2, x + dt*K3, p)

    return dt * (K1/2 + K2 + K3 + K4/2) / 3

@njit
def dSdt(func, fjac, t,  S, p, D):
    """
    Differential equations for combined state/variational matrix
    propagation. This combined state is called S.
    """
    x = S[:D]
    Phi = S[D:]
    rPhi = np.reshape(Phi, (D, D))
    rdPhi = np.dot(fjac(t, x, p), rPhi)
    return np.append(func(t, x, p), rdPhi.flatten())

@njit
def varRK4(func, fjac, x, t1, t2, p, D):
    """
    Fourth-order, 4-step RK routine.
    Returns the step, i.e. approximation to the integral.
    If x is defined at time t_1, then stim should be an array of
    stimulus values at times t_1, (t_1+t_2)/2, and t_2 (i.e. at t1 and t2, as
    well as at the midpoint).
    Alternatively, stim may be a function pointer.
    """
    tmid = (t1 + t2)/2
    dt = t2 - t1

    K1 = dSdt(func, fjac, t1, x, p, D)
    K2 = dSdt(func, fjac, tmid, x + dt*K1/2, p, D)
    K3 = dSdt(func, fjac, tmid, x + dt*K2/2, p, D)
    K4 = dSdt(func, fjac, t2, x + dt*K3, p, D)

    return dt * (K1/2 + K2 + K3 + K4/2) / 3

@jit
def break_cond(f, xi, lim_dead):
    c1 = np.min(xi) <= lim_dead 
    return c1

@njit
def computeLE(func, fjac, x0, t, p, ttrans=None):
    """
    Computes the global Lyapunov exponents for a set of ODEs.
    f - ODE function. Must take arguments like f(t, x, p) where x and t are 
        the state and time *now*, and p is a tuple of parameters. If there are 
        no model paramters, p should be set to the empty tuple.
    x0 - Initial position for calculation. Integration of transients will begin 
         from this point.
    t - Array of times over which to calculate LE.
    p - (optional) Tuple of model parameters for f.
    fjac - Jacobian of f.
    ttrans - (optional) Times over which to integrate transient behavior.
             If not specified, assumes trajectory is on the attractor.
    method - (optional) Integration method to be used by scipy.integrate.ode.
    """
    block = False
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
    return final_LE, block