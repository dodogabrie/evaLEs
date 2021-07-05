import numpy as np
from numba import njit

@njit
def motion(func, t, x, p):
    for i,(t1,t2) in enumerate(zip(t[:-1], t[1:])):
        x[i+1] = x[i] + RK4(func, x[i], t1, t2, p)
    return x

@njit
def RK4(f, x, t1, t2, p):
    """
    Fourth-order, 4-step RK routine.
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
    Fourth-order, 4-step RK routine for variational matrix.
    """
    tmid = (t1 + t2)/2
    dt = t2 - t1

    K1 = dSdt(func, fjac, t1, x, p, D)
    K2 = dSdt(func, fjac, tmid, x + dt*K1/2, p, D)
    K3 = dSdt(func, fjac, tmid, x + dt*K2/2, p, D)
    K4 = dSdt(func, fjac, t2, x + dt*K3, p, D)

    return dt * (K1/2 + K2 + K3 + K4/2) / 3

