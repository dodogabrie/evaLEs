import numpy as np
from numba import njit


def assemble(v, U, D):
    new_state = np.empty(D+D**2, dtype=np.float64)
    new_state[:D] = v 
    new_state[D:] = U.flatten()
    return new_state

def disassemble(state, D):
    new_v = state[:D]
    new_U = state[D:].reshape(D, D)
    return new_v, new_U

def numba_dot(a, b):
    n = a.shape[0]
    dot = 0
    for i in range(n):
        dot += a[i] * b[i]
    return dot
def numba_norm(a):
    n = a.shape[0]
    norm = 0
    for i in range(n):
        norm += a[i] * a[i]
    return np.sqrt(norm)

def gram_schmidt(U):
    D = np.shape(U)[1]
    W = U.copy().astype(np.float64)
    V = np.empty((D, D))
    norms = np.empty(D)
    for i in range(D):
        for j in range(i):
            W[:, i] = W[:, i] - numba_dot(U[:, i], V[:, j]) * V[:, j]
        norms[i] = numba_norm(W[:, i])
        V[:, i] = W[:, i] / norms[i]
    return V, norms


def benedettin(func, fjac, x, t, p, continuous):
    N = len(t)
    D = len(x)
    Phi = np.eye(D, dtype=np.float64)
    S = np.empty((N-1, D), dtype=np.float64)
    LE = np.zeros((N-1, D), dtype=np.float64)
    logW_sum = np.zeros(D, dtype=np.float64)
    dt = t[1]-t[0]
    for i, (t1, t2) in enumerate(zip(t[:-1], t[1:])):
        # Evolve the system 
        if continuous:
            S_temp = varRK4(t1, assemble(x, Phi, D), dt, func, fjac, p, D)
        else:
            S_temp = dSdt(func, fjac, t1, assemble(x, Phi, D), p, D)

        # Apply gram-schmidt algorithm
        x, Phi = disassemble(S_temp, D)
        Phi, norms = gram_schmidt(Phi)

        # Compute Lyapunov exponents
        logW = np.log(norms)
        logW_sum += logW

        # Saving results
        LE[i] = logW_sum/t2
        S[i] = x

    return LE, S

def dSdt(f, fjac, t, x, p, D):
    v, U = disassemble(x, D)
    dv = f(t, v, p)
    dU = fjac(t, v, p) @ U
    return assemble(dv, dU, D)

def varRK4(t, state, dt, ODE, J, p, D):
    tmid = t + dt*0.5
    k1 = dSdt(ODE, J, t, state, p, D)
    k2 = dSdt(ODE, J, tmid, state + dt*0.5 * k1, p, D)
    k3 = dSdt(ODE, J, tmid, state + dt*0.5 * k2, p, D)
    k4 = dSdt(ODE, J, t + dt, state + dt * k3, p, D)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

##################################################################################

def motion(func, t, x, p, continuous=True):
    """
    Basic Function for evaluate trajectory in the dynamical system with RK4
    """
    for (t1,t2) in zip(t[:-1], t[1:]):
        if continuous: x = x + RK4(func, x, t1, t2, p)
        else: x = func(t1, x, p)
    return x

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

    return dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6
