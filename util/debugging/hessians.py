import numpy as np
import scipy
from scipy import linalg
import itertools
import matplotlib.pyplot as plt
from mrh.util.debugging import gradients
from scipy.sparse import linalg as sparse_linalg

def epstable (f_op, j_op, x, divs):
    '''Convergence table for evaluating analytical gradient functions using the error measure

    epsilon (x) = (f_op (x) - g_vec.x) / f_op (x)

    Args:
        f_op : callable
            Takes a vector of length=len(x) and dtype=x.dtype and returns a vector
            The vector-valued function whose gradient is to be evaluated
            Assumed prepared such that f_op(0) = 0; i.e., no constant term
        j_op : 
            Takes a vector of length=len(x) and dtype=x.dtype and returns a vector
            Putative Jacobian function for f_op
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
    jx = j_op (x)
    table = np.empty ((len (divs), 2), dtype=x.dtype)
    xs = table[:,0]
    epss = table[:,1]
    xs[:] = x_norm / divs
    fs = [div * f_op (x/div) for div in divs]
    epss[:] = [linalg.norm (f-jx) / linalg.norm (f)
               for f, div in zip (fs, divs)]
    return table

class HessianDebugger (gradients.GradientDebugger):
    def __init__(self, f_op, j_op, shape=None, dtype=float, **kwargs):
        if not isinstance (j_op, sparse_linalg.LinearOperator):
            j_op = sparse_linalg.LinearOperator (
                shape=shape,
                dtype=dtype,
                matvec=j_op
            )
        self.shape = j_op.shape
        self.dtype = j_op.dtype
        if 'f0' not in kwargs:
            f0 = f_op (np.zeros (self.shape[0], dtype=self.dtype))
        else:
            f0 = kwargs['f0']
        self.x = f0.copy ()
        self.f_op = sparse_linalg.LinearOperator (
            shape=self.shape,
            dtype=self.dtype,
            matvec=lambda x: np.squeeze (f_op (x) - f0)
        )
        self.j_op = j_op
        self.__dict__.update (**kwargs)
        self.x = np.atleast_1d (self.x).astype (self.dtype)
        if not hasattr (self, 'divs'): self.set_divrange_()

    def get_epstable (self):
        t =  epstable (self.f_op, self.j_op, self.x, self.divs)
        return t

if __name__=='__main__':
    dbg = HessianDebugger (np.sin, lambda x:x, shape=(1,1), x=1, exps=range(5,20)).run ()
    print (dbg.error, dbg.slope)
    dbg.plot ('correct.eps')
    dbg = HessianDebugger (np.sin, lambda x:1.0001*x, shape=(1,1), x=1).run ()
    print (dbg.error, dbg.slope)
    dbg.plot ('incorrect.eps')


