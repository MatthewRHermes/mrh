import numpy as np
from scipy import linalg
from scipy.sparse import linalg as sparse_linalg

rng = np.random.default_rng ()

def select (vec, mask):
    ''' Select an entry from vec proportional to the magnitude of the value in mask'''
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

def find_phase (a_op, a_diag, amps, maxiter=20, conv_tol=1e-10):
    '''
    Args:
        a_op : LinearOperator of shape (n,n)
        a_diag : ndarray of shape (n,)
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
    v = amps.copy ()
    non0 = amps > 0
    w = np.dot (v.conj ().T, a_op (v))
    av = a_op (v) - (w*v)
    err = linalg.norm (av)
    v1 = v.copy ()
    def get_ai (i):
        v1[:] = 0
        v1[i] = v[i]
        return a_op (v1)
    def get_step (i, w0=0, ai=None):
        if ai is None: ai = get_ai (i)
        #ai = get_ai (i)
        #w1 = w0 - 4*np.dot (v, ai-ai[i])
        v1[:] = v
        v1[i] *= -1
        #av1 = (av-(2*ai)) + (w0*v) - (w1*v1)
        av1 = a_op (v1)
        w1 = np.dot (v1.conj (), av1)
        av1 = av1 - w1*v1
        return av1, w1
    def get_element (w):
        i = select (av, non0)
        ai = get_ai (i) - w*v[i]
        idx_i = (np.sign (ai) == np.sign (av[i]))
        idx_j = (np.sign (ai) == np.sign (av))
        mask = non0 & (idx_i | idx_j)
        mask[i] = False
        assert (np.count_nonzero (mask) > 0), '{} {} {} {}'.format (i, np.sign (ai), np.sign (av), non0)
        j = select (av, mask)
        return i, j, ai
    for it in range (maxiter):
        i, j, ai = get_element (w)
        av_i, w_i = get_step (i, w, ai)
        av_j, w_j = get_step (j, w, ai)
        err_i = linalg.norm (av_i)
        err_j = linalg.norm (av_j)
        print (it, w, err, i, err_i, j, err_j)
        if (err_i > err) and (err_j > err):
            continue
        elif err_i < err_j:
            av[:] = av_i
            err = err_i
            w = w_i
            v[i] *= -1
        elif err_j < err_i:
            av = av_j
            err = err_j
            w = w_j
            v[j] *= -1
        if err < conv_tol:
            print (it+1, w, err, i, err_i, j, err_j)
            break
    return w, v

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

