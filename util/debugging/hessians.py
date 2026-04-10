import numpy as np
import scipy
from scipy import linalg
import itertools
import matplotlib.pyplot as plt
from mrh.util.debugging import gradients
from scipy.sparse import linalg as sparse_linalg

def epstable (f_op, j_op, x, divs, facs=None):
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

    Kwargs:
        facs : sequence of floats
            Factors of j_op (x) to include, to quickly check whether it's off by 1/2 or something.

    Returns:
        table : ndarray of shape=(len(exp_range),2)
            Columns: ||x||, epsilon (x)
    '''
    if facs is None: facs = [1,-1,2,.5,-2,-0.5]
    n = len (x)
    divs = np.asarray (divs)
    x_norm = linalg.norm (x)
    jx = j_op (x)
    table = np.empty ((len (divs), len (facs) + 1), dtype=x.dtype)
    xs = table[:,0]
    xs[:] = x_norm / divs
    fs = [div * f_op (x/div) for div in divs]
    for i, (f, div) in enumerate (zip (fs, divs)):
        for j, fac in enumerate (facs):
            table[i,j+1] = linalg.norm (f-(jx*fac)) / linalg.norm (f)
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
        t =  epstable (self.f_op, self.j_op, self.x, self.divs, self.facs)
        return t

    def split (self, slices, labels=None, slicesy=None, labelsy=None):
        slicesx = slices
        labelsx = labels
        if labelsy is None: labelsy = labels
        if slicesy is None: slicesy = slices
        slicesx = [0,] + list (slicesx) + [self.shape[0],]
        slicesy = [0,] + list (slicesy) + [self.shape[1],]
        subproblems = []
        for i in range (len (slicesx) - 1):
            p, q = slicesx[i:i+2].copy ()
            for j in range (len (slicesy) - 1):
                r, s = slicesy[j:j+2].copy ()
                myname = None
                if labelsx is not None:
                    myname = labelsx[i] + ',' + labelsy[i]
                subproblems.append (self.subproblem (p, q, r, s, name=myname))
        return subproblems

    def subproblem (self, p, q, r, s, name=None):
        def f1 (x1):
            x = np.zeros_like (self.x)
            x[r:s] = x1[:]
            return self.f_op (x)[p:q]
        f1op = sparse_linalg.LinearOperator (
            shape=(q-p,s-r),
            dtype=self.dtype,
            matvec=f1
        )
        def j1 (x1):
            x = np.zeros_like (self.x)
            x[r:s] = x1[:]
            return self.j_op (x)[p:q]
        j1op = sparse_linalg.LinearOperator (
            shape=(q-p,s-r),
            dtype=self.dtype,
            matvec=j1
        )
        if name is not None:
            if self.name is not None:
                name = self.name + ' ' + name
        kwargs = {key: val for key, val in self.__dict__.items ()}
        kwargs.pop ('f_op')
        kwargs.pop ('j_op')
        kwargs['x'] = self.x[r:s].copy ()
        kwargs['name'] = name
        kwargs['f0'] = np.zeros ((q-p), dtype=self.dtype)
        kwargs['shape'] = (q-p,s-r)
        dbg = HessianDebugger (f1op, j1op, **kwargs)
        return dbg


if __name__=='__main__':
    f = lambda x: np.sin (x+np.pi*.25)
    dbg = HessianDebugger (f, lambda x:np.sqrt(1/2), shape=(1,1), x=1, exps=range(5,20)).run ()
    dbg.name = 'Correct implementation'
    print (dbg.error, dbg.slope)
    print (dbg.sprintf_results ())
    dbg.plot ('correct_hessian.eps')
    dbg.plotall ('correct_hessian.eps')
    dbg = HessianDebugger (f, lambda x:0.7071*x, shape=(1,1), x=1).run ()
    dbg.name = 'Incorrect implementation'
    print (dbg.error, dbg.slope)
    print (dbg.sprintf_results ())
    dbg.plot ('incorrect_hessian.eps')
    dbg.plotall ('incorrect_hessian.eps')


