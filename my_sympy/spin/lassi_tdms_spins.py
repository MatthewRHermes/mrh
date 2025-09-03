import numpy as np
import sympy
from mrh.my_sympy.spin import spin_1h
from sympy import S, Rational, symbols, Matrix, Poly, apart, powsimp, cancel
import itertools

s = symbols ("s", real=True, positive=True)
m = symbols ("m", real=True)

# In this case, the "CSFs" can be treated like determinants
from pyscf.fci import cistring
def get_strings (nops, d2s):
    assert ((nops % 2) == (d2s % 2))
    assert (abs (d2s) <= nops)
    return cistring.make_strings (range(nops), (nops-d2s)//2)

def str2array (st, nops):
    st = [int (bit) for bit in bin (st)[2:]]
    st = [0,] * (nops - len (st)) + st
    return st
def strs2arrays (strs, nops): return [str2array (st, nops) for st in strs]
def array2str (array):
    st = 0
    for pos, bit in enumerate (array):
        st = st | (bit << pos)
    return st
def arrays2strs (arrays):
    return numpy.array ([array2str (array) for array in arrays], dtype=np.int64)
def get_d2s_fromarray (array):
    ndown = sum (array)
    nup = len (array) - ndown
    return nup - ndown
def get_d2s_fromstr (st, nops):
    return get_d2s_fromarray (str2array (st, nops))

def cg_product (crops, csf, s_ket, m_ket):
    '''Evaluate the product of Clebsch-Gordan coefficients,
    <t_p t_q ... s_ket | cp' cq' ... | s_ket m_ket>
    '''
    crops = crops[::-1]
    csf = str2array (csf, len (crops))
    csf = csf[::-1]
    coeff = S.One
    onehalf = Rational (1, 2)
    my_s, my_m = s_ket, m_ket
    for n, t in zip (crops, csf):
        my_s += (onehalf, -onehalf)[t]
        my_m += (onehalf, -onehalf)[n]
        coeff = coeff * spin_1h.cg (my_s, my_m, t, n)
    return coeff

class TDMExpression (object):
    def __init__(self, lhs, rhs_coeffs, rhs_terms):
        self.lhs = lhs
        self.rhs_coeffs = rhs_coeffs
        self.rhs_terms = rhs_terms

    def __str__(self):
        my_solution = str (self.lhs) + ' = \n   '
        for c, t in zip (self.rhs_coeffs, self.rhs_terms): 
            my_solution += ' ' + str (c) + ' * ' + str (t) + '\n + '
        return my_solution[:-4]

    def latex (self, env='align'):
        equality = {'align': r''' =& ''',
                    'equation': r''' = ''',
                    'eqnarray': r''' &=& ''',}[env.lower ()]
        my_latex = '\\begin{' + env.lower () + '}\n'
        my_latex += self.lhs.latex () + equality
        sum_linker = '\\nonumber \\\\ & '
        first_term = True
        for c, t in zip (self.rhs_coeffs, self.rhs_terms):
            if not first_term:
                my_latex += sum_linker
            this_term = sympy.latex (c) + t.latex ()
            if (not first_term) and (not this_term.startswith ('-')):
                this_term = '+' + this_term
            my_latex += this_term
            first_term = False
        my_latex = my_latex + '\n\\end{' + env.lower () + '}'
        return my_latex

class CrVector (object):
    def __init__(self, s_bra, crops, s_ket, m_ket):
        dm = Rational (get_d2s_fromarray (crops), 2)
        self.s_bra = s_bra
        self.m_bra = m_ket + dm
        self.crops = crops
        self.s_ket = s_ket
        self.m_ket = m_ket

    def get_strings (self):
        '''These strings simultaneously represent
            1) the CSFs
            2) the operator products on the right-hand side
        '''
        d2s = int ((self.s_bra - self.s_ket) * 2)
        return get_strings (len (self.crops), d2s)

    def cg_product (self, csf):
        return cg_product (self.crops, csf, self.s_ket, self.m_ket)

    def cg_products (self, csfs):
        return Matrix ([self.cg_product (csf) for csf in csfs])

    def get_spinupvecs (self, crops_spinup):
        crvecs_spinup = [CrVector (self.s_bra, crop_spinup, self.s_ket, self.s_ket)
                      for crop_spinup in crops_spinup]
        return crvecs_spinup

    def __str__(self):
        s = '<' + str (self.s_bra) + ',' + str (self.m_bra) + '| '
        for crop, lbl in zip (self.crops, 'pqrstuvwxyz'):
            cr = ('a','b')[crop] + lbl + "' "
            s = s + cr
        s = s + '|' + str (self.s_ket) + ',' + str (self.m_ket) + '>'
        return s

    def get_A (self, A_cols):
        return self.cg_products (A_cols)

    def get_A_cols (self):
        return self.get_strings ()

    def get_B_rows (self):
        csfs = self.get_strings ()
        ops_rhs = strs2arrays (csfs, len (self.crops))
        B_rows = self.get_spinupvecs (ops_rhs)
        lvec = [symbols ("l" + str (csf), real=True) for csf in csfs] 
        lvec_lookup = {l: B_row for l, B_row in zip (lvec, B_rows)}
        lvec = Matrix (lvec)
        B_rows = [B_row.cg_products for B_row in B_rows]
        return B_rows, lvec, lvec_lookup

    def solve (self):
        A_cols = self.get_A_cols ()
        B_rows, lvec, lvec_lookup = self.get_B_rows ()
        A = self.get_A (A_cols)
        B = make_B (B_rows, A_cols)
        xvec = B.solve (lvec)
        xvec = Matrix ([x.simplify () for x in xvec])
        coeffs = 0
        for Ael, xel in zip (A, xvec):
            coeffs += (Ael * xel).simplify ()
        #coeffs = (A.T * xvec)
        #coeffs = coeffs[0]
        lvec_symbols = list (lvec_lookup.keys ())
        lvec_exprs = list (lvec_lookup.values ())
        coeffs = Poly (coeffs, lvec_symbols).coeffs ()
        coeffs = [c.simplify () for c in coeffs]
        return TDMExpression (self, coeffs, lvec_exprs)

    @property
    def H (self):
        return AnVector (self.s_ket, self.crops[::-1], self.s_bra, self.m_bra)

    def latex (self):
        my_latex = '\\braket{' + str (self.s_bra) + ',' + str (self.m_bra) + '|'
        for crop, lbl in zip (self.crops, 'pqrstuvwxyz'):
            cr = ('a','b')[crop]
            my_latex += '\\cr' + cr + 'op{' + lbl + '}'
        my_latex += '|' + str (self.s_ket) + ',' + str (self.m_ket) + '}'
        return my_latex

class AnVector (CrVector):
    def __init__(self, s_bra, anops, s_ket, m_ket):
        dm = Rational (get_d2s_fromarray (anops), 2)
        m_bra = m_ket - dm
        crops = anops[::-1]
        super().__init__(s_ket, crops, s_bra, m_bra)

    def __str__(self):
        s = '<' + str (self.s_ket) + ',' + str (self.m_ket) + '| '
        for crop, lbl in zip (self.crops[::-1], 'pqrstuvwxyz'):
            cr = ('a','b')[crop] + lbl + " "
            s = s + cr
        s = s + '|' + str (self.s_bra) + ',' + str (self.m_bra) + '>'
        return s

    def get_spinupvecs (self, anops_spinup):
        anvecs_spinup = [AnVector (self.s_ket, anop_spinup, self.s_bra, self.s_bra)
                      for anop_spinup in anops_spinup]
        return anvecs_spinup

    @property
    def H (self):
        return CrVector (self.s_bra, self.crops, self.s_ket, self.m_ket)

    def latex (self):
        my_latex = '\\braket{' + str (self.s_ket) + ',' + str (self.m_ket) + '|'
        for crop, lbl in zip (self.crops[::-1], 'pqrstuvwxyz'):
            cr = ('a','b')[crop]
            my_latex += '\\an' + cr + 'op{' + lbl + '}'
        my_latex += '|' + str (self.s_bra) + ',' + str (self.m_bra) + '}'
        return my_latex

class CrAnOperator (CrVector):
    def __init__(self, s_bra, crops, anops, s_ket, m_ket):
        dm_cr = Rational (get_d2s_fromarray (crops), 2)
        dm_an = Rational (get_d2s_fromarray (anops), 2)
        self.s_bra = s_bra
        self.m_bra = m_ket + dm_cr - dm_an
        self.crops = crops
        self.anops = anops
        self.s_ket = s_ket
        self.m_ket = m_ket

    def __str__(self):
        s = '<' + str (self.s_bra) + ',' + str (self.m_bra) + '| '
        for crop, lbl in zip (self.crops, 'pqrstuvwxyz'):
            cr = ('a','b')[crop] + lbl + "' "
            s = s + cr
        for anop, lbl in zip (self.anops, 'pqrstuvwxyz'[len(self.crops):]):
            an = ('a','b')[anop] + lbl + " "
            s = s + an
        s = s + '|' + str (self.s_ket) + ',' + str (self.m_ket) + '>'
        return s

    def latex (self):
        my_latex = '\\braket{' + str (self.s_bra) + ',' + str (self.m_bra) + '|'
        for crop, lbl in zip (self.crops, 'pqrstuvwxyz'):
            my_latex += '\\cr' + ('a','b')[crop] + 'op{' + lbl + '}'
        for anop, lbl in zip (self.anops, 'pqrstuvwxyz'[len(self.crops):]):
            my_latex += '\\an' + ('a','b')[anop] + 'op{' + lbl + '}'
        my_latex += '|' + str (self.s_ket) + ',' + str (self.m_ket) + '}'
        return my_latex

    def get_crvec (self, s_res):
        m_res = self.m_bra - Rational (get_d2s_fromarray (self.crops), 2)
        return CrVector (self.s_bra, self.crops, s_res, m_res)

    def get_anvec (self, s_res):
        return AnVector (s_res, self.anops, self.s_ket, self.m_ket)

    def get_A_cols (self):
        # I have to transpose the crvec because the CSF resolution of the identity is always on the bra
        # side in the parent, and I need to apply it on the ket/resolvent side here.
        ncrops = len (self.crops)
        nanops = len (self.anops)
        d2s_ket = int (2 * (self.s_ket - self.s_bra))
        min_d2s_r = max (-ncrops, d2s_ket - nanops)
        max_d2s_r = min (ncrops, d2s_ket + nanops) + 1
        A_cols = []
        for d2s_r in range (min_d2s_r, max_d2s_r, 2):
            s_res = self.s_bra + Rational (d2s_r, 2)
            cr_csfs = self.get_crvec (s_res).H.get_strings ()
            an_csfs = self.get_anvec (s_res).get_strings ()
            for cr_csf, an_csf in itertools.product (cr_csfs, an_csfs):
                A_cols.append ((s_res, cr_csf, an_csf))
        return A_cols

    def cg_product (self, re):
        s_res, cr_csf, an_csf = re
        # I have to transpose the crvec because the CSF resolution of the identity is always on the bra
        # side in the parent, and I need to apply it on the ket/resolvent side here.
        return self.get_crvec (s_res).H.cg_product (cr_csf) * self.get_anvec (s_res).cg_product (an_csf)

    def get_B_rows (self):
        d2s = int (2 * (self.s_bra - self.s_ket))
        ncrops = len (self.crops)
        nanops = len (self.anops)
        nops = ncrops + nanops
        B_strings = get_strings (nops, d2s)
        B_strings = np.atleast_1d (strs2arrays (B_strings, nops))
        B_crops = B_strings[:,:ncrops]
        B_anops = 1 - B_strings[:,ncrops:]
        B_rows = self.get_spinupops (B_crops, B_anops)
        lvec = [symbols ("l" + str (i), real=True) for i in range (len (B_strings))] 
        lvec_lookup = {l: B_row for l, B_row in zip (lvec, B_rows)}
        lvec = Matrix (lvec)
        B_rows = [B_row.cg_products for B_row in B_rows]
        return B_rows, lvec, lvec_lookup

    def get_spinupops (self, crops_spinup, anops_spinup):
        crvecs_spinup = [CrAnOperator (self.s_bra, crop_spinup, anop_spinup, self.s_ket, self.s_ket)
                      for crop_spinup, anop_spinup in zip (crops_spinup, anops_spinup)]
        return crvecs_spinup

def make_B (B_rows, A_cols):
    B = B_rows[0] (A_cols).T
    for i, B_row in enumerate (B_rows[1:]):
        B = B.row_insert (i+1, B_row (A_cols).T)
    return B

def solve_pure_destruction (d2s_bra, anops, d2s_ket, d2m_ket):
    return solve_pure_creation (d2s_bra, anops, d2s_ket, d2m_ket, H=True)

def solve_pure_creation (d2s_bra, crops, d2s_ket, d2m_ket, H=False):
    s_bra = s + Rational (d2s_bra, 2)
    s_ket = s + Rational (d2s_ket, 2)
    m_ket = m + Rational (d2m_ket, 2)
    if H:
        lhs = AnVector (s_bra, crops, s_ket, m_ket)
    else:
        lhs = CrVector (s_bra, crops, s_ket, m_ket)
    return lhs.solve ()

def solve_density (d2s_bra, crops, anops, d2s_ket, d2m_ket):
    s_bra = s + Rational (d2s_bra, 2)
    s_ket = s + Rational (d2s_ket, 2)
    m_ket = m + Rational (d2m_ket, 2)
    lhs = CrAnOperator (s_bra, crops, anops, s_ket, m_ket)
    return lhs.solve ()

latex_header = r'''\documentclass[prb,amsmath,amssymb,floatfix,nofootinbib,superscriptaddress,reprint,onecolumn]{revtex4-1}
\usepackage{rotating}
\usepackage{txfonts}
\usepackage{array}
\usepackage{bm}
\usepackage{dcolumn}
\usepackage{amsmath}
\usepackage{braket}
\usepackage{xfrac}
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}
\newcommand{\crop}[1]{\ensuremath{\hat{c}_{#1}^\dagger}}
\newcommand{\anop}[1]{\ensuremath{\hat{c}_{#1}}}
\newcommand{\craop}[1]{\ensuremath{\hat{a}_{#1}^\dagger}}
\newcommand{\anaop}[1]{\ensuremath{\hat{a}_{#1}}}
\newcommand{\crbop}[1]{\ensuremath{\hat{b}_{#1}^\dagger}}
\newcommand{\anbop}[1]{\ensuremath{\hat{b}_{#1}}}
\newcommand{\crsop}[1]{\ensuremath{\hat{\sigma}_{#1}^\dagger}}
\newcommand{\ansop}[1]{\ensuremath{\hat{\sigma}_{#1}}}
\newcommand{\myapprox}[1]{\mathrel{\overset{\makebox[0pt]{\mbox{\normalfont\tiny\sffamily #1}}}{\approx}}}
%\newcommand{\redsout}[1]{\textcolor{red}{\sout{#1}}}
\newcommand{\pystrlit}{\textquotesingle\textquotesingle\textquotesingle}
\newcommand{\spforall}{\ensuremath{\hspace{2mm}\forall\hspace{2mm}}}


\begin{document}

'''

if __name__=='__main__':
    print ("============= All creation/all destruction =============")
    a = []
    print ("------- Alpha only -------")
    a.append (solve_pure_destruction (-1, [0,], 0, 0))
    a.append (solve_pure_creation (-1, [0,], 0, 0))
    a.append (solve_pure_destruction (-2, [0,0], 0, 0))
    a.append (solve_pure_destruction (0, [0,0], 0, 0))
    a.append (solve_pure_creation (-2, [0,0], 0, 0))
    for expr in a: print (expr)
    b = []
    print ("\n------- Beta only -------")
    b.append (solve_pure_creation (-1, [1,], 0, 0))
    b.append (solve_pure_destruction (-1, [1,], 0, 0))
    b.append (solve_pure_creation (-2, [1,1], 0, 0))
    b.append (solve_pure_creation (0, [1,1], 0, 0))
    b.append (solve_pure_destruction (-2, [1,1], 0, 0))
    for expr in b: print (expr)
    ab = []
    print ("\n------- Mixed -------")
    ab.append (solve_pure_destruction (-2, [0,1], 0, 0))
    ab.append (solve_pure_destruction (0, [1,0], 0, 0))
    ab.append (solve_pure_creation (-2, [0,1], 0, 0))
    for expr in ab: print (expr)
    gamma1 = []
    print ("\n\n============= One-body density =============")
    gamma1.append (solve_density (0, [0,], [0,], 0, 0))
    gamma1.append (solve_density (0, [1,], [1,], 0, 0))
    gamma1.append (solve_density (-2, [0,], [0,], 0, 0))
    gamma1.append (solve_density (-2, [1,], [1,], 0, 0))
    gamma1.append (solve_density (-2, [1,], [0,], 0, 0))
    gamma1.append (solve_density (0, [1,], [0,], 0, 0))
    gamma1.append (solve_density (-2, [0,], [1,], 0, 0))
    for expr in gamma1: print (expr)
    gamma3h = []
    print ("\n\n============= Three-half-particle operators =============")
    gamma3h.append (solve_density (-3, [0,], [0,0], 0, 0))
    gamma3h.append (solve_density (-3, [1,], [1,0], 0, 0))
    gamma3h.append (solve_density (-3, [0,], [0,1], 0, 0))
    gamma3h.append (solve_density (-3, [1,], [1,1], 0, 0))
    gamma3h.append (solve_density (0, [0,], [0,0], 1, 1))
    gamma3h.append (solve_density (0, [1,], [1,0], 1, 1))
    gamma3h.append (solve_density (0, [0,], [0,1], 1, 1))
    gamma3h.append (solve_density (0, [1,], [1,1], 1, 1))
    for expr in gamma3h: print (expr)
    gamma2 = []
    print ("\n\n============= Two-body density =============")
    gamma2.append (solve_density (4, [0,0], [0,0], 0, 0))
    gamma2.append (solve_density (4, [0,1], [1,0], 0, 0))
    gamma2.append (solve_density (4, [1,1], [1,1], 0, 0))
    gamma2.append (solve_density (2, [0,0], [0,0], 0, 0))
    gamma2.append (solve_density (2, [0,1], [1,0], 0, 0))
    gamma2.append (solve_density (2, [1,1], [1,1], 0, 0))
    gamma2.append (solve_density (0, [0,0], [0,0], 0, 0))
    gamma2.append (solve_density (0, [0,1], [1,0], 0, 0))
    gamma2.append (solve_density (0, [1,1], [1,1], 0, 0))
    for expr in gamma2: print (expr)

    all_exprs = a + b + ab + gamma1 + gamma3h + gamma2
    import os
    fname = os.path.splitext (os.path.basename (__file__))[0] + '.tex'
    with open (fname, 'w') as f:
        f.write (latex_header)
        for expr in all_exprs:
            f.write (expr.latex () + '\n\n')
        f.write ('\n\n\\end{document}')


