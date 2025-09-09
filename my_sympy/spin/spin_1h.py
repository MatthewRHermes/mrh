import sympy
from sympy import symbols
from sympy.core.evalf import EvalfMixin

def cg_ua (s, m):
    return sympy.sqrt ((s+m)/(2*s))

def cg_ub (s, m):
    return sympy.sqrt ((s-m)/(2*s))

def cg_da (s, m):
    return -sympy.sqrt ((s-m+1)/((2*s)+2))

def cg_db (s, m):
    return sympy.sqrt ((s+m+1)/((2*s)+2))

cg_mat = [[cg_ua, cg_ub], [cg_da, cg_db]]

def cg (s, m, t, n):
    '''The sympy expression for the Clebsch-Gordan coefficient
    <s-(1/2)+t, m-1/2+n ; 1/2, 1/2-n | s, m>

    Args:
        s : sympy expression or string
        m : sympy expression or string
        t : 0 or 1
        n : 0 or 1

    Returns:
        Sympy expression
    '''
    if not isinstance (s, EvalfMixin):
        s = symbols (s, real=True, positive=True)
    if not isinstance (m, EvalfMixin):
        m = symbols (m, real=True)
    return cg_mat[t][n] (s, m)

def cgd (ds, dm, t, n):
    s = symbols ("s", real=True, positive=True)
    m = symbols ("m", real=True)
    return cg_mat[t][n] (s+ds, m+dm)

def cgd_ua (ds, dm): return cgd (ds, dm, 0, 0)
def cgd_ub (ds, dm): return cgd (ds, dm, 0, 1)
def cgd_da (ds, dm): return cgd (ds, dm, 1, 0)
def cgd_db (ds, dm): return cgd (ds, dm, 1, 1)

if __name__=='__main__':
    from sympy.core.evalf import EvalfMixin
    print (type ("s").__mro__, type (symbols ("s")).__mro__)
    s, m = symbols ("s m", real=True)
    print (type (cg ("s", "m", 0, 0)).__mro__)


