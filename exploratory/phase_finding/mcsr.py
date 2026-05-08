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

class MonteCarloSignRecoverer:
    def __init__(self, a_op, a_diag, amps, maxiter=20, conv_tol=1e-10, log=print, num_tol=1e-16):
        if (log is None) or (log==False): log = lambda * args: None
        self.a_op = a_op
        self.a_diag = a_diag
        self.amps = amps
        self.maxiter = maxiter
        self.conv_tol = conv_tol
        self.log = log
        self.v = v = amps.copy ()
        self.av = av = a_op (v)
        self.w = w = np.dot (v.conj ().T, av)
        self.av = av = av - (w*v)
        self.err = linalg.norm (av)
        self.non0 = np.abs (amps) > num_tol
        self.v1 = v.copy ()
        self.it = 0

    def get_ai (self, i):
        self.v1[:] = 0
        self.v1[i] = self.v[i]
        return self.a_op (self.v1)

    def get_step (self, i):
        self.v1[:] = self.v
        self.v1[i] *= -1
        av1 = self.a_op (self.v1)
        w1 = np.dot (self.v1.conj (), av1)
        av1 = av1 - w1*self.v1
        return av1, w1

    def get_element (self):
        i = select (self.av, self.non0)
        ai = self.get_ai (i) - self.w*self.v[i]
        vai = self.v * ai
        vav = self.v * self.av
        idx_i = (np.sign (vai) == np.sign (vav[i]))
        idx_j = (np.sign (vai) == np.sign (vav))
        mask = self.non0 & (idx_i | idx_j)
        mask[i] = False
        assert (np.count_nonzero (mask) > 0), '{} {} {} {}'.format (i, vai, vav, non0)
        j = select (vav, mask)
        return i, j

    def __call__(self):
        i = j = err_i = err_j = None
        for self.it in range (self.maxiter):
            self.log (self.it, self.w, self.err, i, err_i, j, err_j)
            if self.err < self.conv_tol:
                break
            i, j = self.get_element ()
            av_i, w_i = self.get_step (i)
            av_j, w_j = self.get_step (j)
            err_i = linalg.norm (av_i)
            err_j = linalg.norm (av_j)
            if (err_i > self.err) and (err_j > self.err):
                pass
            elif err_i < err_j:
                self.av[:] = av_i
                self.err = err_i
                self.w = w_i
                self.v[i] *= -1
            elif err_j < err_i:
                self.av = av_j
                self.err = err_j
                self.w = w_j
                self.v[j] *= -1
        return self.w, self.v


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
        print ("result", i, w[i], w_test, np.dot (v[:,i].conj (), v_test), f.err, f.it)

if __name__=='''__main__''':
    for i in range (100):
        hmat = (2*rng.random ((3,3))) - 1
        hmat += hmat.T
        find_phase_matrix (hmat)

