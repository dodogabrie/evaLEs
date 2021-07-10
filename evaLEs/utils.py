import numpy as np

def benedettin(func, fjac, x0, t, p, ttrans, continuos):
    N = len(t)
    D = len(x0)
    Phi0 = np.eye(D, dtype=np.float64).flatten() # Identità di tipo float
    LE = np.zeros((N-1, D), dtype=np.float64)
    final_LE = np.zeros((N-1, D), dtype=np.float64)
    LE_aux = np.zeros((N-1, D), dtype=np.float64)
    Ssol = np.zeros((N, D*(D+1)), dtype=np.float64)
    Ssol[0] = np.append(x0, Phi0)
    for i,(t1,t2) in enumerate(zip(t[:-1], t[1:])):
        if continuos:
            Ssol_temp = Ssol[i] + varRK4(func, fjac, Ssol[i], t1, t2, p, D)
        else:
            Ssol_temp = dSdt(func, fjac, t1, Ssol[i], p, D)
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

def motion(func, t, init, p, continuos = True):
    """
    Basic Function for evaluate trajectory in the dynamical system with RK4
    """
    x = np.ones((len(t), len(init)))*init
    for i,(t1,t2) in enumerate(zip(t[:-1], t[1:])):
        if continuos:
            x[i+1] = x[i] + RK4(func, x[i], t1, t2, p)
        else:
            x[i+1] = func(t1, x[i], p)
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

    return dt * (K1/2 + K2 + K3 + K4/2) / 3

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