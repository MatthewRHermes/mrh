import numpy as np
from scipy import linalg
from scipy.sparse import linalg as sparse_linalg

rng = np.random.default_rng ()

def select (vec, mask):
    ''' Select an entry from vec proportional to the magnitude of the value '''
    mask_idx = np.where (mask)[0]
    return mask_idx[_select (vec[mask])]

def _select (vec):
    if len (vec) == 1: return 0
    vec = np.abs (vec)
    offs = np.cumsum (vec)
    card = rng.random () * offs[-1]
    if card==offs[-1]: return len (vec) - 1
    idx = np.where (offs > card)[0][0]
    return idx

def find_phase (h_op, h_diag, amps, maxiter=100):
    '''
    Args:
        h_op : LinearOperator of shape (n,n)
        h_diag : ndarray of shape (n,)
        amps : ndarray of shape (n,)
            Contains nonnegative probability amplitudes

    Returns:
        w : float
            "eigenvalue"
        v : ndarray of shape (n,)
            Contains "eigenvector" with amplitudes given by amps but with signs determined by
            the iterative algorithm
    '''
    n = len (amps)
    assert (not (amps<0).any ())
    j = np.argmax (amps)
    assigned = np.zeros (n, dtype=bool)
    assigned[j] = True
    v = np.zeros_like (amps)
    v[j] = amps[j]
    E = h_diag[j]
    for it_E in range (maxiter):
        err = np.dot (h_diag - E, amps**2)
        assigned[:] = False
        assigned[j] = True
        v[:] = 0
        v[j] = amps[j]
        for it_el in range (n-1):
            hv = h_op (v)
            i = select (hv, assigned==False)
            assert (assigned[i]==False)
            delta = amps[i] * hv[i]
            if abs (err - delta) < abs (err + delta):
                v[i] = -amps[i]
                err -= delta
            else:
                v[i] = amps[i]
                err += delta
            assigned[i] = True
        assert (assigned.all ())
        E = v.conj ().T @ h_op (v)
    return E, v

def find_phase_matrix (hmat):
    w, v = linalg.eigh (hmat)
    h_diag = hmat.diagonal ()
    h_op = sparse_linalg.aslinearoperator (hmat)
    for i in range (len (w)):
        w_test, v_test = find_phase (h_op, h_diag, np.abs (v[:,i]))
        print ("result", i, w[i], w_test, np.dot (v[:,i].conj (), v_test))

if __name__=='''__main__''':
    hmat = (2*rng.random ((3,3))) - 1
    hmat += hmat.T
    find_phase_matrix (hmat)

