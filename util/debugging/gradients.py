import numpy as np
import scipy
from scipy import linalg
import itertools
import matplotlib.pyplot as plt

def epstable (f_op, g_vec, x, divs):
    '''Convergence table for evaluating analytical gradient functions using the error measure

    epsilon (x) = (f_op (x) - g_vec.x) / f_op (x)

    Args:
        f_op : callable
            Takes a vector of length=len(x) and dtype=x.dtype and returns a scalar of dtype=x.dtype
            The function whose gradient is to be evaluated
            Assumed prepared such that f_op(0) = 0; i.e., no constant term
        g_vec : ndarray of length=len(x) and dtype=x.dtype
            Putative gradient vector of f_op evaluated at x = [0,0,0,...]
        x : ndarray
            Trial step vector
        divs : ndarray of floats
            Divisors

    Returns:
        table : ndarray of shape=(len(exp_range),2)
            Columns: ||x||, epsilon (x)
    '''
    n = len (x)
    divs = np.asarray (divs)
    x_norm = linalg.norm (x)
    gx = np.dot (g_vec, x)
    table = np.empty ((len (divs), 2), dtype=x.dtype)
    xs = table[:,0]
    epss = table[:,1]
    xs[:] = x_norm / divs
    epss[:] = [f_op (x/div) for div in divs]
    epss[:] = (epss - gx/divs) / epss
    return table

def fit_fn (x, a, p, c):
    return a * (x**p) + c

class GradientDebugger (object):
    base=2
    exps=range(5,20)
    window=3
    tol=0.1
    def __init__(self, f_op, g_vec, **kwargs):
        g_vec = np.atleast_1d (g_vec)
        self.g_vec = g_vec
        self.x = g_vec.copy () 
        self.shape = g_vec.shape
        self.dtype = g_vec.dtype
        self.__dict__.update (**kwargs)
        self.x = np.atleast_1d (self.x).astype (self.dtype)
        if not hasattr (self, 'f0'): f0 = f_op (np.zeros (self.shape, dtype=self.dtype))
        self.f_op = lambda x: np.squeeze (f_op (x) - f0)
        if not hasattr (self, 'divs'): self.set_divrange_()

    def set_divrange_(self, base=None, exps=None):
        if base is None: base=self.base
        if exps is None: exps=self.exps
        self.divs = [base ** exp for exp in exps]

    def get_epstable (self):
        return epstable (self.f_op, self.g_vec, self.x, self.divs)

    def run (self):
        self.epstable = self.get_epstable ()
        try:
            self.fit = scipy.optimize.curve_fit (
                fit_fn,
                self.epstable[:,0],
                self.epstable[:,1],
                p0=[1,1,0],
                bounds=([-np.inf,0,-np.inf],np.inf),
                sigma=self.epstable[:,0]/100
            )
        except RuntimeError as err:
            print (self.epstable)
            raise (err) from None
        self.error = self.fit[0][2]
        self.slope = self.fit[0][1]
        return self

    def plot (self, fname=None):
        eps1 = [fit_fn (x, *self.fit[0]) for x in self.epstable[:,0]]
        plt.loglog (np.abs (self.epstable[:,0]), np.abs (self.epstable[:,1]), 'o',
                    np.abs (self.epstable[:,0]), np.abs (eps1), '-')
        if fname is not None:
            plt.savefig (fname)
            plt.close ()

if __name__=='__main__':
    dbg = GradientDebugger (np.sin, 1.0).run ()
    print (dbg.error, dbg.slope)
    dbg.plot ('correct.eps')
    dbg = GradientDebugger (np.sin, 1.0001).run ()
    print (dbg.error, dbg.slope)
    dbg.plot ('incorrect.eps')


