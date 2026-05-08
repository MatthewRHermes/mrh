import numpy as np
from scipy import linalg
from scipy.sparse import linalg as sparse_linalg

rng = np.random.default_rng ()

def select (vec, mask=None):
    ''' Select an entry from vec proportional to the magnitude of the value in mask'''
    if mask is None:
        return _select (vec)
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

class MCSRState:
    def __init__(self, v, av):
        self.v = v.copy ()
        self.w = w = np.dot (v.conj ().T, av)
        self.av = av = av - (w*v)
        self.err = linalg.norm (av)

class MonteCarloSignRecoverer:
    def __init__(self, a_op, a_diag, amps, maxiter=20, conv_tol=1e-10, log=print, num_tol=1e-16):
        if (log is None) or (log==False): log = lambda * args: None
        self.a_op = a_op
        self.a_diag = a_diag
        self.amps = amps
        self.maxiter = maxiter
        self.conv_tol = conv_tol
        self.log = log
        self.state = MCSRState (amps, a_op (amps))
        self.non0 = np.abs (amps) > num_tol
        self.it = 0
        self.vbuf = np.empty_like (amps)

    def get_ai (self, i):
        v1 = self.vbuf
        v1[:] = 0
        v1[i] = self.state.v[i]
        return self.a_op (v1)

    def get_step (self, i):
        v1 = self.vbuf
        v1[:] = self.state.v
        v1[i] *= -1
        return MCSRState (v1, self.a_op (v1))

    def get_element (self):
        i = select (self.state.av, self.non0)
        v, w, av = self.state.v, self.state.w, self.state.av
        ai = self.get_ai (i) - w*v[i]
        vai = v * ai
        vav = v * av
        idx_i = (np.sign (vai) == np.sign (vav[i]))
        idx_j = (np.sign (vai) == np.sign (vav))
        mask = self.non0 & (idx_i | idx_j)
        mask[i] = False
        assert (np.count_nonzero (mask) > 0), '{} {} {} {}'.format (i, vai, vav, non0)
        j = select (vav, mask)
        return i, j

    def __call__(self):
        for self.it in range (self.maxiter):
            self.log (self.it, self.state.w, self.state.err)
            if self.state.err < self.conv_tol:
                break
            i, j = self.get_element ()
            state_i = self.get_step (i)
            state_j = self.get_step (j)
            if (state_i.err > self.state.err) and (state_j.err > self.state.err):
                pass
            elif state_i.err < state_j.err:
                self.state = state_i
            elif state_j.err < state_i.err:
                self.state = state_j
        return self.state.w, self.state.v


def find_phase (*args, **kwargs):
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
    return MonteCarloSignRecoverer (*args, **kwargs) ()

def find_phase_matrix (hmat):
    w, v = linalg.eigh (hmat)
    h_diag = hmat.diagonal ()
    h_op = sparse_linalg.aslinearoperator (hmat)
    for i in range (len (w)):
        f = MonteCarloSignRecoverer (h_op, h_diag, np.abs (v[:,i]), log=False)
        w_test, v_test = f ()
        print ("result", i, w[i], w_test, np.dot (v[:,i].conj (), v_test), f.state.err, f.it)

if __name__=='''__main__''':
    for i in range (100):
        hmat = (2*rng.random ((3,3))) - 1
        hmat += hmat.T
        find_phase_matrix (hmat)

