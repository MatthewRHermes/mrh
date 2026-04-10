import os
import numpy as np
import scipy
from scipy import linalg
import itertools
import matplotlib.pyplot as plt

def epstable (f_op, g_vec, x, divs, facs=None):
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

    Kwargs:
        facs : sequence of floats
            Factors of gx to include, to quickly check whether it's off by 1/2 or something.

    Returns:
        table : ndarray of shape=(len(exp_range),1+len(facs))
            Columns: ||x||, epsilon (x)
    '''
    if facs is None: facs = [1,-1,2,.5,-2,-0.5]
    n = len (x)
    divs = np.asarray (divs)
    x_norm = linalg.norm (x)
    gx = np.dot (g_vec, x)
    table = np.empty ((len (divs), len (facs) + 1), dtype=x.dtype)
    xs = table[:,0]
    fs = np.empty_like (xs)
    xs[:] = x_norm / divs
    fs[:] = [f_op (x/div) for div in divs]
    gx = gx * np.asarray (facs)[None,:] / divs[:,None]
    fs = fs[:,None]
    table[:,1:] = (fs - gx) / fs
    return table

def fit_fn (x, a, p, c):
    return a * (x**p) + c

class GradientDebugger (object):
    base=2
    exps=range(5,20)
    window=3
    tol=0.1
    facs = [1,-1,2,.5,-2,-.5]
    name = None
    def __init__(self, f_op, g_vec, **kwargs):
        g_vec = np.atleast_1d (g_vec)
        self.g_vec = g_vec
        self.x = g_vec.copy () 
        self.shape = g_vec.shape
        self.dtype = g_vec.dtype
        self.__dict__.update (**kwargs)
        self.x = np.atleast_1d (self.x).astype (self.dtype)
        if not hasattr (self, 'f0'):
            f0 = f_op (np.zeros (self.shape, dtype=self.dtype))
        else:
            f0 = self.f0
        self.f_op = lambda x: np.squeeze (f_op (x) - f0)
        if not hasattr (self, 'divs'): self.set_divrange_()

    def split (self, slices, labels=None):
        slices = [0,] + list (slices) + [self.shape[0],]
        subproblems = []
        for i in range (len (slices)-1):
            myname = None
            if labels is not None:
                myname = labels[i]
            subproblems.append (self.subproblem (slices[i], slices[i+1], name=myname))
        return subproblems

    def subproblem (self, p, q, name=None):
        g1 = self.g_vec[p:q]
        def f1 (x1):
            x = np.zeros_like (self.x)
            x[p:q] = x1[:]
            return self.f_op (x)
        if name is not None:
            if self.name is not None:
                name = self.name + ' ' + name
        kwargs = {key: val for key, val in self.__dict__.items ()}
        kwargs.pop ('f_op')
        kwargs.pop ('g_vec')
        kwargs['x'] = self.x[p:q].copy ()
        kwargs['name'] = name
        kwargs['f0'] = 0
        kwargs['shape'] = (q-p,)
        dbg = GradientDebugger (f1, g1, **kwargs)
        return dbg

    def set_divrange_(self, base=None, exps=None):
        if base is None: base=self.base
        if exps is None: exps=self.exps
        self.divs = [base ** exp for exp in exps]

    def get_epstable (self):
        return epstable (self.f_op, self.g_vec, self.x, self.divs, self.facs)

    def run (self):
        self.epstable = self.get_epstable ()
        self.fits = []
        self.errors = []
        self.slopes = []
        for i in range (1, self.epstable.shape[1]):
            try:
                self.fits.append (scipy.optimize.curve_fit (
                    fit_fn,
                    self.epstable[:,0],
                    self.epstable[:,i],
                    p0=[1,1,0],
                    bounds=([-np.inf,0,-np.inf],np.inf),
                    sigma=self.epstable[:,0]/100
                ))
            except RuntimeError as err:
                print (self.epstable)
                raise (err) from None
            self.errors.append (self.fits[i-1][0][2])
            self.slopes.append (self.fits[i-1][0][1])
        self.fit = self.fits[0]
        self.error = self.errors[0]
        self.slope = self.slopes[0]
        return self

    def sprintf_results (self):
        lbls = ['fac', 'intercept', 'slope']
        results = '{:>5s} {:>10s} {:>10s}\n'.format (*lbls)
        fmt_str = '{:5.1f} {:10.3e} {:10.3e}\n'
        for i in range (len (self.facs)):
            results += fmt_str.format (self.facs[i], self.errors[i], self.slopes[i])
        return results[:-1]

    def plot (self, fname=None, i=0, title=None):
        if title is None: title = self.name
        eps1 = [fit_fn (x, *self.fits[i][0]) for x in self.epstable[:,0]]
        plt.loglog (np.abs (self.epstable[:,0]), np.abs (self.epstable[:,i+1]), 'o',
                    np.abs (self.epstable[:,0]), np.abs (eps1), '-')
        plt.xlabel ('||x||')
        plt.ylabel ('epsilon(x)')
        plt.gca ().set_aspect ('equal')
        xmin, xmax = plt.xlim ()
        ymin, ymax = plt.ylim ()
        xrng = xmax / xmin
        plt.ylim (ymax/xrng, ymax)
        if title is not None: plt.title (title)
        if fname is not None:
            plt.savefig (fname)
            plt.close ()

    def plotall (self, fname, title=None):
        if title is None: title = self.name
        fhead, fext = os.path.splitext (fname) 
        for i in range (len (self.facs)):
            fname = '{}.{}{}'.format (fhead, self.facs[i], fext)
            mytitle = 'factor = {}'.format (self.facs[i])
            if title is not None:
                mytitle = title + ' ({})'.format (mytitle)
            self.plot (fname=fname, i=i, title=mytitle)

if __name__=='__main__':
    f = lambda x: np.sin (x+np.pi*.25)
    dbg = GradientDebugger (f, np.sqrt (1/2)).run ()
    dbg.name = 'Correct implementation'
    print (dbg.error, dbg.slope)
    print (dbg.sprintf_results ())
    dbg.plot ('correct_gradient.eps')
    dbg.plotall ('correct_gradient.eps')
    dbg = GradientDebugger (f, 0.7071).run ()
    dbg.name = 'Incorrect implementation'
    print (dbg.error, dbg.slope)
    print (dbg.sprintf_results ())
    dbg.plot ('incorrect_gradient.eps')
    dbg.plotall ('incorrect_gradient.eps')


