import copy
from tqdm import tqdm
import numpy as np
import sympy
from mrh.my_sympy.spin import spin_1h
from sympy import S, Rational, symbols, Matrix, Poly, apart, powsimp, cancel
from sympy.utilities.lambdify import lambdastr
import itertools
from mrh.my_sympy.spin.lassi_tdms_spins.glob import *
from mrh.my_sympy.spin.lassi_tdms_spins.operators import CrVector, AnVector, CrAnOperator, OpSum

class TDMExpression (object):
    def __init__(self, lhs, rhs_coeffs, rhs_terms):
        fl, lhs = lhs.normal_order ()
        nl = 1
        self.lhs = lhs
        self.rhs_coeffs = []
        self.rhs_terms = []
        for c, t in zip (rhs_coeffs, rhs_terms):
            if c == S(0): continue
            fr, tno = t.normal_order ()
            self.rhs_coeffs.append (fl * fr * c)
            self.rhs_terms.append (tno)

    def normal_order_labels (self):
        fl, lhs = self.lhs.normal_order_labels ()
        rhs_coeffs, rhs_terms = [], []
        for c, t in zip (self.rhs_coeffs, self.rhs_terms):
            fr, tno = t.normal_order_labels ()
            rhs_coeffs.append (fl * fr * c)
            rhs_terms.append (tno)
        return TDMExpression (lhs, rhs_coeffs, rhs_terms)

    def transpose (self, idx):
        new_lhs = self.lhs.transpose (idx)
        new_terms = [t.transpose (idx) for t in self.rhs_terms]
        new_expr = TDMExpression (new_lhs, self.rhs_coeffs, new_terms)
        return new_expr.normal_order_labels ()

    def __str__(self):
        my_solution = str (self.lhs) + ' = \n   '
        for c, t in zip (self.rhs_coeffs, self.rhs_terms): 
            my_solution += ' ' + str (c) + ' * ' + str (t) + '\n + '
        return my_solution[:-4]

    def latex (self, env='align'):
        my_latex = '\\begin{' + env.lower () + '}\n'
        my_latex += self._latex_line (env=env)
        my_latex += '\n\\end{' + env.lower () + '}'
        return my_latex

    def _latex_line (self, env='align'):
        equality = {'align': r''' =& ''',
                    'equation': r''' = ''',
                    'eqnarray': r''' &=& ''',}[env.lower ()]
        my_latex = self.lhs.latex () + equality
        sum_linker = '\\nonumber \\\\ '
        sum_linker += {'align': r''' & ''',
                       'equation': "",
                       'eqnarray': r''' && '''}[env]
        first_term = True
        for c, t in zip (self.rhs_coeffs, self.rhs_terms):
            if not first_term:
                my_latex += sum_linker
            this_term = sympy.latex (c) + t.latex ()
            if (not first_term) and (not this_term.startswith ('-')):
                this_term = '+' + this_term
            my_latex += this_term
            first_term = False
        return my_latex

    def subs_m (self, new_m):
        new_lhs = self.lhs.subs_m (new_m)
        new_terms = [t.subs_m (new_m) for t in self.rhs_terms]
        new_coeffs = [c.subs (m, new_m) for c in self.rhs_coeffs]
        return TDMExpression (new_lhs, new_coeffs, new_terms)

    def subs_s (self, new_s):
        new_lhs = self.lhs.subs_s (new_s)
        new_terms = [t.subs_s (new_s) for t in self.rhs_terms]
        new_coeffs = [c.subs (s, new_s) for c in self.rhs_coeffs]
        return TDMExpression (new_lhs, new_coeffs, new_terms)

    def subs_sket_to_s (self):
        new_s = (2*s) - self.lhs.get_s_ket ()
        return self.subs_s (new_s)

    def subs_mket_to_m (self):
        new_m = (2*m) - self.lhs.get_m_ket ()
        return self.subs_m (new_m)

    def __add__(self, other):
        new_lhs = self.lhs + other.lhs
        new_terms = list (set (self.rhs_terms + other.rhs_terms))
        new_coeffs = []
        for t in new_terms:
            c = 0
            ins = ino = False
            if t in self.rhs_terms:
                i = self.rhs_terms.index (t)
                c += self.rhs_coeffs[i]
                ins = True
            if t in other.rhs_terms:
                i = other.rhs_terms.index (t)
                c += other.rhs_coeffs[i]
                ino = True
            if ins and ino:
                c = c.simplify ()
            new_coeffs.append (c)
        return TDMExpression (new_lhs, new_coeffs, new_terms)

    def __sub__(self, other):
        new_lhs = self.lhs - other.lhs
        new_terms = list (set (self.rhs_terms + other.rhs_terms))
        new_coeffs = []
        for t in new_terms:
            c = 0
            ins = ino = False
            if t in self.rhs_terms:
                i = self.rhs_terms.index (t)
                c += self.rhs_coeffs[i]
                ins = True
            if t in other.rhs_terms:
                i = other.rhs_terms.index (t)
                c -= other.rhs_coeffs[i]
                ino = True
            if ins and ino:
                c = c.simplify ()
            new_coeffs.append (c)
        return TDMExpression (new_lhs, new_coeffs, new_terms)

    def __mul__(self, scale):
        new_coeffs = [c * scale for c in self.rhs_coeffs]
        return TDMExpression (self.lhs, new_coeffs, self.rhs_terms)

    def __truediv__(self, scale):
        new_coeffs = [c / scale for c in self.rhs_coeffs]
        return TDMExpression (self.lhs, new_coeffs, self.rhs_terms)

    def powsimp_(self):
        self.rhs_coeffs = [powsimp (c, force=True) for c in self.rhs_coeffs]

    def simplify_(self):
        self.rhs_coeffs = [c.simplify () for c in self.rhs_coeffs]

    def python_code (self, row_scores):
        rows = ['dm_1[:,{}]'.format (i) for i in range (len (row_scores))]
        if len (rows) == 1:
            rows[0] = rows[0][:-3] + ']'
        code = rows[row_scores.index (self.lhs.get_sort_score ())] + ' = ('
        indent = len (code)
        first_term = True
        for rhs_coeff, rhs_term in zip (self.rhs_coeffs, self.rhs_terms):
            if rhs_coeff == S(0): continue
            if not first_term: code += '\n' + ' '*indent + '+ '
            rhs_code = rows[row_scores.index (rhs_term.get_sort_score ())]
            ops = rhs_term.get_indices ()
            if not ORBINDICES.startswith (''.join (ops)):
                idx = [list (ORBINDICES).index (x) for x in ops]
                rhs_code += '.transpose (0,'
                for x in ops:
                    rhs_code += str (list (ORBINDICES).index (x)+1) + ','
                rhs_code = rhs_code[:-1] + ')'
            rhs_code = rhs_code.replace ('dm_1', 'dm_0')
            code += '(' + sympy.pycode (rhs_coeff) + ') * ' + rhs_code
            first_term = False
        code += ')'
        return code

def combine_TDMSystem (systems):
    exprs = [system.exprs for system in systems]
    exprs = list (itertools.chain.from_iterable (exprs))
    return TDMSystem (exprs, _try_inverse=False)

class TDMSystem (object):
    def __init__(self, exprs, _try_inverse=True):
        self._init_from_exprs (exprs)
        if len (self.rows) == 1: self.simplify_cols_()
        if _try_inverse and (not self.same_ops ()):
            A = self.get_A ()
            op_B = self.subs_m_max ().inv ()
            new_A = A * op_B.get_A ()
            rows = self.rows
            cols = op_B.cols
            new_exprs = []
            for i, lhs in enumerate (rows):
                new_row = [el.simplify () for el in new_A.row (i)]
                new_exprs.append (TDMExpression (lhs, new_row, cols))
            self._init_from_exprs (new_exprs)

    def _init_from_exprs (self, exprs):
        self.exprs = exprs
        self.rows = [expr.lhs for expr in exprs]
        self.cols = list (set (itertools.chain.from_iterable (
            [expr.rhs_terms for expr in exprs]
        )))
        hashes = [hash (c) for c in self.cols]
        assert (len (set (hashes)) == len (hashes))

    def simplify_cols_ (self):
        nrows = len (self.rows)
        ncols = len (self.cols)
        if (ncols == 1):
            return self
        A = self.get_A ()
        new_terms = []
        skip = np.zeros (ncols, dtype=bool)
        for i in range (ncols):
            new_term = self.cols[i]
            if skip[i]: continue
            for j in range (i+1,ncols):
                if all ([A[k,i]==A[k,j] for k in range (nrows)]):
                    skip[j] = True
                    new_term = new_term + self.cols[j]
                elif all ([A[k,i]==-A[k,j] for k in range (nrows)]):
                    skip[j] = True
                    new_term = new_term - self.cols[j]
            new_terms.append (new_term)
        skip = list (np.where (~skip)[0])
        new_A = A.extract (range (nrows), skip)
        self._init_from_exprs([TDMExpression (lhs, list (new_A.row (i)), new_terms)
                               for i, lhs in enumerate (self.rows)])
        return self

    def get_sorted_exprs (self):
        ''' Get the exprs with pqrs in order on the lhs '''
        exprs = []
        for expr in self.exprs:
            nops = len (self.exprs[0].lhs.get_ops ())
            if (''.join (expr.lhs.get_indices ()) == ORBINDICES[:nops]):
                exprs.append (expr)
        assert (len (exprs))
        return exprs

    def reduce_to_sorted (self):
        exprs = self.get_sorted_exprs ()
        score = [expr.lhs.get_sort_score () for expr in exprs]
        idx = np.argsort (score)
        exprs = [exprs[i] for i in idx]
        return TDMSystem (exprs, _try_inverse=False)
        
    def __str__(self):
        return '\n'.join ([str (expr) for expr in self.get_sorted_exprs ()])

    def latex (self, env='align'):
        my_latex = '\\begin{' + env.lower () + '}\n'
        first_term = True
        row_linker = ' \\\\ \n'
        for expr in self.get_sorted_exprs ():
            if not first_term: my_latex += row_linker
            my_latex += expr._latex_line (env=env)
            first_term = False
        my_latex += '\n\\end{' + env.lower () + '}'
        return my_latex

    def same_ops (self):
        if any ([isinstance (c, OpSum) for c in self.cols]): return False
        rows_closed = all ([any ([r.cmp_ops (c) for c in self.cols]) for r in self.rows])
        cols_closed = all ([any ([r.cmp_ops (c) for r in self.rows]) for c in self.cols])
        return rows_closed and cols_closed

    def get_A (self):
        A = []
        for row, expr in zip (self.rows, self.exprs):
            Arow = []
            rhs_hashes = [hash (term) for term in expr.rhs_terms]
            for col in self.cols:
                if hash (col) in rhs_hashes:
                    Arow.append (expr.rhs_coeffs[rhs_hashes.index (hash(col))])
                else:
                    Arow.append (S(0))
            A.append (Arow)
        return Matrix (A)

    def inv (self):
        try:
            Ainv = self.get_A ().inv ()
        except Exception as err:
            # Try Moore-Penrose left pseudoinverse
            A = self.get_A ()
            AHA = A.H * A
            try:
                Ainv = AHA.inv ()
            except Exception as err:
                print ("A:", A)
                print ("AHA:", AHA)
                raise (err)
        rhs_terms = self.rows
        invexprs = []
        for i, lhs in enumerate (self.cols):
            rhs_coeffs = [el.simplify () for el in Ainv.row (i)]
            invexprs.append (TDMExpression (lhs, list (rhs_coeffs), rhs_terms))
        return TDMSystem (invexprs, _try_inverse=False)

    def subs_m (self, new_m):
        new_exprs = [e.subs_m (new_m) for e in self.exprs]
        return TDMSystem (new_exprs, _try_inverse=False)

    def subs_s (self, new_s):
        new_exprs = [e.subs_s (new_s) for e in self.exprs]
        return TDMSystem (new_exprs, _try_inverse=False)

    def subs_sket_to_s (self):
        new_exprs = [e.subs_sket_to_s () for e in self.exprs]
        return TDMSystem (new_exprs, _try_inverse=False)

    def subs_mket_to_m (self):
        new_exprs = [e.subs_mket_to_m () for e in self.exprs]
        return TDMSystem (new_exprs, _try_inverse=False)

    def subs_m_max (self):
        return self.subs_m (min ([r.max_m () for r in self.rows]))

    def powsimp_(self):
        for expr in self.exprs:
            expr.powsimp_()
        return self

    def simplify_(self):
        for expr in self.exprs:
            expr.simplify_()
        return self

    def __mul__(self, scale):
        old_scale = getattr (self, 'scale', S(1))
        scale /= old_scale
        return ScaledTDMSystem (scale**-1, self)

    def __truediv__(self, scale):
        return self.__mul__(scale**-1)

    def python_code (self):
        scores = [row.get_sort_score () for row in self.rows]
        return '\n'.join ([expr.python_code (scores) for expr in self.exprs])

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

def get_eqn_dict ():
    print ("Building equation dictionary...", flush=True)
    with tqdm(total=93) as pbar:
        #print ("============= All creation/all destruction =============")
        a = []
        #print ("------- Alpha only -------")
        a.append (TDMSystem ([solve_pure_destruction (-1, [0,], 0, 0)]))
        pbar.update (1)
        a.append (TDMSystem ([solve_pure_destruction (1, [0,], 0, 0)]))
        pbar.update (1)
        a.append (TDMSystem ([solve_pure_destruction (-2, [0,0], 0, 0)]))
        pbar.update (1)
        a.append (TDMSystem ([solve_pure_destruction (0, [0,0], 0, 0)]))
        pbar.update (1)
        a.append (TDMSystem ([solve_pure_destruction (2, [0,0], 0, 0)]))
        pbar.update (1)
        a = [e.subs_mket_to_m ().subs_sket_to_s () for e in a]
        #for expr in a: print (expr)
        b = []
        #print ("\n------- Beta only -------")
        b.append (TDMSystem ([solve_pure_destruction (1, [1,], 0, 0)]))
        pbar.update (1)
        b.append (TDMSystem ([solve_pure_destruction (-1, [1,], 0, 0)]))
        pbar.update (1)
        b.append (TDMSystem ([solve_pure_destruction (2, [1,1], 0, 0)]))
        pbar.update (1)
        b.append (TDMSystem ([solve_pure_destruction (0, [1,1], 0, 0)]))
        pbar.update (1)
        b.append (TDMSystem ([solve_pure_destruction (-2, [1,1], 0, 0)]))
        pbar.update (1)
        b = [e.subs_mket_to_m ().subs_sket_to_s () for e in b]
        #for expr in b: print (expr)
        ab = []
        #print ("\n------- Mixed -------")
        ab.append (TDMSystem ([solve_pure_destruction (-2, [1,0], 0, 0)]))
        pbar.update (1)
        ab.append (TDMSystem ([solve_pure_destruction (0, [1,0], 0, 0),
                               solve_pure_destruction (0, [0,1], 0, 0)]))
        pbar.update (4)
        ab.append (TDMSystem ([solve_pure_destruction (2, [1,0], 0, 0)]))
        pbar.update (1)
        ab = [e.subs_mket_to_m ().subs_sket_to_s () for e in ab]
        #for expr in ab: print (expr)
        gamma1 = []
        #print ("\n\n============= One-body density =============")
        gamma1.append (TDMSystem ([solve_density (0, [0,], [0,], 0, 0),
                                   solve_density (0, [1,], [1,], 0, 0)]))
        pbar.update (4)
        gamma1.append (TDMSystem ([solve_density (-2, [0,], [0,], 0, 0)]))
        pbar.update (1)
        gamma1.append (TDMSystem ([solve_density (-2, [1,], [1,], 0, 0)]))
        pbar.update (1)
        gamma1.append (TDMSystem ([solve_density (-2, [1,], [0,], 0, 0)]))
        pbar.update (1)
        gamma1.append (TDMSystem ([solve_density (0, [1,], [0,], 0, 0)]))
        pbar.update (1)
        gamma1.append (TDMSystem ([solve_density (2, [1,], [0,], 0, 0)]))
        pbar.update (1)
        gamma1 = [e.subs_mket_to_m ().subs_sket_to_s () for e in gamma1]
        #for expr in gamma1: print (expr)
        gamma3h = []
        #print ("\n\n============= Three-half-particle operators =============")
        gamma3h.append (TDMSystem ([solve_density (-2, [0,], [0,0], 1, 1)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (-2, [1,], [1,0], 1, 1)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (-2, [0,], [0,1], 1, 1)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (-2, [1,], [1,1], 1, 1)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (3, [0,], [0,0], 0, 0)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (3, [1,], [1,0], 0, 0)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (3, [0,], [0,1], 0, 0)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (3, [1,], [1,1], 0, 0)]))
        pbar.update (1)
        gamma3h.append (TDMSystem ([solve_density (0, [0,], [0,0], 1, 1),
                                    solve_density (0, [1,], [1,0], 1, 1),
                                    solve_density (0, [1,], [0,1], 1, 1)]))
        pbar.update (9)
        gamma3h.append (TDMSystem ([solve_density (0, [0,], [0,0], -1, -1),
                                    solve_density (0, [1,], [1,0], -1, -1),
                                    solve_density (0, [1,], [0,1], -1, -1)]))
        gamma3h[-1] = gamma3h[-1].subs_s(s+Rational(1,2))
        gamma3h[-1].simplify_()
        pbar.update (9)
        gamma3h.append (TDMSystem ([solve_density (0, [0,], [0,1], 1, 1),
                                    solve_density (0, [0,], [1,0], 1, 1),
                                    solve_density (0, [1,], [1,1], 1, 1)]))
        pbar.update (9)
        gamma3h.append (TDMSystem ([solve_density (0, [0,], [0,1], -1, 1),
                                    solve_density (0, [0,], [1,0], -1, 1),
                                    solve_density (0, [1,], [1,1], -1, 1)]))
        gamma3h[-1] = gamma3h[-1].subs_s(s+Rational(1,2))
        gamma3h[-1].simplify_()
        pbar.update (9)
        #gamma3h = [e.subs_mket_to_m () for e in gamma3h]
        #gamma3h = [e.subs_sket_to_s () for e in gamma3h]
        #for expr in gamma3h: print (expr)
        gamma2 = []
        #print ("\n\n============= Two-body density =============")
        gamma2.append (TDMSystem ([solve_density (0, [0,0], [0,0], 4, 0)]))
        pbar.update (1)
        gamma2.append (TDMSystem ([solve_density (0, [0,1], [1,0], 4, 0)]))
        pbar.update (1)
        gamma2.append (TDMSystem ([solve_density (0, [1,1], [1,1], 4, 0)]))
        pbar.update (1)
        gamma2.append (TDMSystem ([solve_density (0, [0,0], [0,0], 2, 0),
                                   solve_density (0, [0,1], [1,0], 2, 0),
                                   #solve_density (2, [1,0], [0,1], 0, 0),
                                   solve_density (0, [1,0], [1,0], 2, 0),
                                   solve_density (0, [1,1], [1,1], 2, 0)]))
        pbar.update (16)
        gamma2.append (TDMSystem ([solve_density (0, [0,0], [0,0], 0, 0),
                                   solve_density (0, [0,1], [1,0], 0, 0),
                                   solve_density (0, [1,0], [0,1], 0, 0),
                                   solve_density (0, [1,1], [1,1], 0, 0)],
                                  _try_inverse=False))
        #gamma2[-2].exprs.append (gamma2[-2].exprs[1].transpose ((0,1,3,2)))
        #gamma2[-2].exprs.append (gamma2[-2].exprs[2].transpose ((0,1,3,2)))
        #gamma2[-2]._init_from_exprs (gamma2[-2].exprs)
        gamma2[-1].exprs.append (gamma2[-1].exprs[1].transpose ((0,1,3,2)))
        gamma2[-1].exprs.append (gamma2[-1].exprs[2].transpose ((0,1,3,2)))
        exprs = gamma2[-1].exprs
        #exprs = [exprs[0],exprs[3],exprs[1]+exprs[2],exprs[4]+exprs[5],exprs[1]-exprs[2],exprs[4]-exprs[5]]
        gamma2[-1]._init_from_exprs (exprs)
        gamma2[-1].simplify_cols_()
        #gamma2 = [e.subs_mket_to_m () for e in gamma2]
        #gamma2 = [e.subs_sket_to_s () for e in gamma2]
        pbar.update (5)


    read_exprs = a + b + ab + gamma1 + gamma3h + gamma2

    lbls = ['ha_d', 'ha_u', 'hb_d', 'hb_u', 'hh_d', 'hh_0', 'hh_u',
                   'sm',
                   'phh_a_3d', 'phh_b_3d', 'phh_a_3u', 'phh_b_3u',
                   'dm_2', 'dm_1', 'dm_0']
    subsec = []
    subsec.append ([read_exprs[i] for i in (0,27)]) # ha_d
    subsec.append ([read_exprs[i] for i in (1,28)]) # ha_u
    subsec.append ([read_exprs[i] for i in (6,29)]) # hb_d
    subsec.append ([read_exprs[i] for i in (5,30)]) # hb_u
    subsec.append ([read_exprs[i] for i in (2,10,9)]) # hh_d
    subsec.append ([read_exprs[i] for i in (3,11,8)]) # hh_0
    subsec.append ([read_exprs[i] for i in (4,12,7)]) # hh_u
    subsec.append ([read_exprs[i] for i in (16,17,18)]) # sm
    subsec.append ([combine_TDMSystem ([read_exprs[i] for i in (19,20)])]) # phh_a_3d
    subsec.append ([combine_TDMSystem ([read_exprs[i] for i in (21,22)])]) # phh_b_3d
    subsec.append ([combine_TDMSystem ([read_exprs[i] for i in (23,24)])]) # phh_a_3u
    subsec.append ([combine_TDMSystem ([read_exprs[i] for i in (25,26)])]) # phh_b_3u
    subsec.append ([combine_TDMSystem ([read_exprs[i] for i in (31,32,33)])]) # dm_2
    subsec.append ([combine_TDMSystem ([read_exprs[i] for i in (14,15)]),
                    read_exprs[34]]) # dm_1
    subsec.append ([read_exprs[i] for i in (13,35)]) # dm_0

    return {key: val for key, val in zip (lbls, subsec)}

def standardize_m_s (eqn_dict):
    eqn_dict1 = {}
    for lbl, sector in eqn_dict.items ():
        sector1 = []
        for tdmsystem in sector:
            new_s = (2*s) - tdmsystem.exprs[0].lhs.get_s_ket ()
            new_m = (2*m) - tdmsystem.exprs[0].lhs.get_m_ket ()
            tdmsystem1 = tdmsystem.subs_m (new_m).subs_s (new_s)
            sector1.append (tdmsystem1)
        eqn_dict1[lbl] = sector1
    return eqn_dict1

mup_fn_head = '''def mup_{{dmname}} (dm_0, smult_bra, {cond_spin_op}smult_ket, spin_ket):
    dm_1 = scale_{{scalename}} (smult_bra, {cond_spin_op}smult_ket, spin_ket) * dm_0
'''
mdown_fn_head = '''def mdown_{{dmname}} (dm_0, smult_bra, {cond_spin_op}smult_ket, spin_ket):
    dm_1 = dm_0 / scale_{{scalename}} (smult_bra, {cond_spin_op}smult_ket, spin_ket)
'''
mup_or_mdown_fn_calltranspose = '''    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-{{dmndim}}:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = {cond_d2s_idx}
    key = (d2s_idx, {cond_spin_idx})
    transpose = _transpose_{mupormdown}_{{dmname}}.get (key, lambda x, s, m: x)
    s = ({cond_smult_ket} - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)
'''

def _mupormdown_fn_fmt (has_spin_op, mirror_sym, mdown):
    fmt_str = (mup_fn_head,mdown_fn_head)[int (mdown)] + mup_or_mdown_fn_calltranspose
    fields = {}
    fields['cond_d2s_idx'] = ('smult_bra-smult_ket', '-abs(smult_bra-smult_ket)')[int (mirror_sym)]
    fields['cond_smult_ket'] = ('smult_ket', 'max (smult_bra, smult_ket)')[int (mirror_sym)]
    fields['cond_spin_op'] = ('', 'spin op, ')[int (has_spin_op)]
    fields['cond_spin_idx'] = ('0', 'spin op')[int (has_spin_op)]
    fields['mupormdown'] = ('mup','mdown')[int (mdown)]
    return fmt_str.format (**fields)

def mup_fn_fmt (has_spin_op, mirror_sym):
    return _mupormdown_fn_fmt (has_spin_op, mirror_sym, False)

def mdown_fn_fmt (has_spin_op, mirror_sym):
    return _mupormdown_fn_fmt (has_spin_op, mirror_sym, True)

class TDMScaleArray (object):
    def __init__(self, name, tdmsystems_array):
        self.name = name
        nrows = len (tdmsystems_array)
        ncols = len (tdmsystems_array[0])
        self.shape = (nrows, ncols)
        self.lhs = [[[tdmsystem.exprs[0].lhs for tdmsystem in el] for el in row]
                    for row in tdmsystems_array]
        col_indices = np.zeros (self.shape + (2,), dtype=int)
        row_indices = np.zeros (self.shape, dtype=int)
        for i, row in enumerate (tdmsystems_array):
            for j, el in enumerate (row):
                col_idx = []
                row_idx = []
                for tdmsystem in el:
                    lhs = tdmsystem.exprs[0].lhs
                    col_idx.append (lhs.count_spins ())
                    row_idx.append (int(2*(lhs.get_s_bra () - lhs.get_s_ket ())))
                assert (all ([ci==col_idx[0] for ci in col_idx]))
                assert (all ([ri==row_idx[0] for ri in row_idx]))
                col_indices[i,j,:] = col_idx[0]
                row_indices[i,j] = row_idx[0]
        self.col_indices = [tuple (el) for el in col_indices[0]]
        col_indices = col_indices.reshape (nrows,-1)
        assert (all (np.all (col_indices==col_indices[0], axis=0)))
        assert (all (np.all (row_indices.T==row_indices.T[0], axis=0)))
        self.row_indices = row_indices.T[0]
        self.mat = Matrix ([[sum (el[0].get_A ().col (0)).simplify ()
                             for el in row]
                            for row in tdmsystems_array])
        self.tdmsystems_array = tdmsystems_array

    def get_dm_types (self):
        dm_types = []
        for row in self.lhs:
            for col in row:
                for el in col:
                    dm_types.append (el.count_ops ())
        return list (set (dm_types))

    def dm_type_str (self, dm_type):
        if self.name == 'sm':
            return "<bp' aq>"
        ncr, ndes = dm_type
        sym = '<'
        for i in range (ncr):
            sym += 'c' + ORBINDICES[i] + "' "
        for i in range (ndes):
            sym += 'c' + ORBINDICES[i+ncr] + " "
        sym = sym[:-1] + '>'
        return sym

    def dm_type_latex (self, dm_type):
        if self.name == 'sm':
            return '\\braket{\\crbop{p}\\anaop{q}}'
        ncr, ndes = dm_type
        sym = '\\braket{'
        for i in range (ncr):
            sym += '\\crop{' + ORBINDICES[i] + "}"
        for i in range (ndes):
            sym += '\\anop{' + ORBINDICES[i+ncr] + "}"
        sym += '}'
        return sym

    def col_index_str (self, col_index):
        nalpha, nbeta = col_index
        nalpha = int (nalpha)
        nbeta = int (nbeta)
        text = ''
        if nalpha > 0:
            text += "a'"*abs(nalpha)
        if nbeta > 0:
            text += "b'"*abs(nbeta)
        if nbeta < 0:
            text += "b"*abs(nbeta)
        if nalpha < 0:
            text += "a"*abs(nalpha)
        return text

    def __str__(self):
        text = self.name + ' ('
        for dm_type in self.get_dm_types ():
            text += self.dm_type_str (dm_type) + ', '
        text = text[:-2] + ')\n'
        for i, row_index in enumerate (self.row_indices):
            row_idx = str (Rational (row_index, 2))
            for j, col_index in enumerate (self.col_indices):
                col_idx = self.col_index_str (col_index)
                text += '[' + row_idx + ']'
                if self.shape[1] > 1:
                    text += '[' + col_idx +']'
                text += ' = ' + str (self.mat[i,j]) + '\n'
        return text

    def latex_row_lhs (self, row_index, col_index):
        superscript = sympy.latex (Rational (row_index, 2))
        subscript = self.col_index_str (col_index)
        text = '\\mathcal{N}^{(' + superscript + ')}'
        if len (subscript) > 0 and self.shape[1] > 1:
            text += '_{' + subscript + '}'
        return text

    def latex (self):
        text = self.name.replace ('_', '\\_') + ' ($'
        for dm_type in self.get_dm_types ():
            text += self.dm_type_latex (dm_type) + ', '
        text = text[:-2] + '$):\n\\begin{align}\n'
        for i, row_index in enumerate (self.row_indices):
            for j, col_index in enumerate (self.col_indices):
                text += self.latex_row_lhs (row_index, col_index)
                text += ' =& ' + sympy.latex (self.mat[i,j])
                text += ' \\\\\n'
        text = text[:-4] + '\n\\end{align}\n\n'
        return text

    def get_transpose_eqns (self, _count_only=False):
        cnt = 0
        transpose_eqns = {}
        for i, row in enumerate (self.tdmsystems_array):
            for j, col in enumerate (row):
                const = self.mat[i,j]
                for k, el in enumerate (col):
                    nops = sum (self.lhs[i][j][k].count_ops ())
                    key = (self.row_indices[i], j, nops)
                    if len (el.cols) > 1:
                        forward = (el / const).simplify_().powsimp_().simplify_()
                        if set ([x for x in forward.get_A ()]) != set ((0,1)):
                            # not effectively diagonal
                            cnt += 1
                            if not _count_only:
                                reverse = forward.inv ()
                                transpose_eqns[key] = [forward.reduce_to_sorted (),
                                                       reverse.reduce_to_sorted ()]
        if _count_only: return cnt
        return transpose_eqns

    def count_transpose_eqns (self):
        return self.get_transpose_eqns (_count_only=True)

    def get_scale_code (self):
        fn_name = 'scale_' + self.name
        code = '_' + fn_name + ' = [\n'
        first_i = True
        for i in range (self.shape[0]):
            if not first_i: code += ',\n'
            code += ' '*4
            indent = 4
            if self.shape[1] > 1:
                code += '['
                indent = 5
            first_j = True
            for j in range (self.shape[1]):
                if not first_j: code = code + ',\n' + ' '*indent
                line = str (lambdastr ((s,m), self.mat[i,j]))
                code += line.replace ('sqrt','np.sqrt')
                first_j = False
            if self.shape[1] > 1:
                code += ']'
            first_i = False
        code += '\n    ]\n\n'
        code += 'def ' + fn_name + ' (smult_bra, '
        if self.shape[1] > 1:
            code += 'spin_op, '
        code += 'smult_ket, spin_ket):\n'
        code += ' '*4 + 'd2s_idx = (smult_bra - smult_ket + '
        code += str (abs (self.row_indices[0])) + ')//2\n'
        code += ' '*4 + 'if (d2s_idx < 0) or (d2s_idx >= {}):'.format (self.shape[0])
        code += ' return 0\n'
        no_pos = np.amax (self.row_indices) == 0
        if no_pos:
            code += ' '*4 + 'if smult_bra > smult_ket:\n'
            code += ' '*8 + 'return ' + fn_name
            code += ' (smult_ket, smult_bra, spin_ket)\n'
        code += ' '*4 + 's = (smult_ket-1)/2\n'
        code += ' '*4 + 'm = spin_ket/2\n'
        code += ' '*4 + 'return _' + fn_name + '[d2s_idx]'
        if self.shape[1]>1:
            code += '[spin_op]'
        code += ' (s, m)\n\n'
        return code

    def get_mupmdown_code (self, nops, dmname, transpose_eqns=None):
        has_spin_op = (self.name == 'h')
        dmndim = nops + int (self.name != 'hh')
        is_d2 = (dmname == 'dm2')
        if transpose_eqns is None:
            transpose_eqns = self.get_transpose_eqns ()
        keys = [key[:2] for key in transpose_eqns.keys () if key[2] == nops]
        transpose_mup_lookup = '_transpose_mup_{dmname} = '.format (dmname=dmname) + '{'
        transpose_mdown_lookup = '_transpose_mdown_{dmname} = '.format (dmname=dmname) + '{'
        lookup_indent_mup = len (transpose_mup_lookup)
        lookup_indent_mdown = len (transpose_mdown_lookup)
        transpose_mup_header = '_transpose_mup_{dmname}_{{idx}}'.format (dmname=dmname)
        transpose_mdown_header = '_transpose_mdown_{dmname}_{{idx}}'.format (dmname=dmname)
        code = ''
        first_term = True
        for i, key in enumerate (keys):
            myput, myget = transpose_eqns[key + (nops,)]
            myput = '    ' + '\n    '.join (myput.python_code ().split ('\n'))
            myget = '    ' + '\n    '.join (myget.python_code ().split ('\n'))
            code += 'def ' + transpose_mup_header.format (idx=i) + '(dm_0, s, m):\n'
            code += myput + '\n'
            code += '    return dm_1\n'
            code += 'def ' + transpose_mdown_header.format (idx=i) + '(dm_0, s, m):\n'
            code += myget + '\n'
            code += '    return dm_1\n'
            if not first_term:
                transpose_mup_lookup += ',\n' + ' '*lookup_indent_mup
                transpose_mdown_lookup += ',\n' + ' '*lookup_indent_mdown
            transpose_mup_lookup += str (key) + ': '
            transpose_mup_lookup += transpose_mup_header.format (idx=i)
            transpose_mdown_lookup += str (key) + ': '
            transpose_mdown_lookup += transpose_mdown_header.format (idx=i)
            first_term = False
        transpose_mup_lookup = transpose_mup_lookup + '}\n'
        transpose_mdown_lookup = transpose_mdown_lookup + '}\n'
        code += transpose_mup_lookup
        code += transpose_mdown_lookup
        code += '\n'
        code += mup_fn_fmt (has_spin_op, mirror_sym=is_d2).format (
            dmname=dmname, dmndim=dmndim, scalename=self.name) + '\n'
        code += mdown_fn_fmt (has_spin_op, mirror_sym=is_d2).format (
            dmname=dmname, dmndim=dmndim, scalename=self.name) + '\n'
        return code


def get_scale_constants (eqn_dict):
    scale = {}
    scale['h'] = TDMScaleArray ('h',
        [[eqn_dict['phh_a_3d'], eqn_dict['phh_b_3d']],
         [eqn_dict['ha_d'], eqn_dict['hb_d']],
         [eqn_dict['ha_u'], eqn_dict['hb_u']],
         [eqn_dict['phh_a_3u'], eqn_dict['phh_b_3u']]]
    )
    scale['hh'] = TDMScaleArray ('hh', [[[el,] for el in eqn_dict[key]]
                                        for key in ('hh_d', 'hh_0', 'hh_u')])
    scale['sm'] = TDMScaleArray ('sm', [[[el,],] for el in eqn_dict['sm']])
    scale['dm'] = TDMScaleArray ('dm', [[eqn_dict['dm_2']],
                                        [eqn_dict['dm_1']],
                                        [eqn_dict['dm_0']]])
    return scale

class ScaledTDMSystem (TDMSystem):
    def __init__(self, scale, tdmsystem):
        self.scale = scale
        exprs = [expr/scale for expr in tdmsystem.exprs]
        self._init_from_exprs (exprs)

    def __str__(self):
        return '\n'.join (['c*' + str (expr) for expr in self.get_sorted_exprs ()])

    def latex (self, env='align'):
        my_latex = '\\begin{' + env.lower () + '}\n'
        first_term = True
        row_linker = ' \\\\ \n'
        for expr in self.get_sorted_exprs ():
            if not first_term: my_latex += row_linker
            my_latex += 'c*' + expr._latex_line (env=env)
            first_term = False
        my_latex += '\n\\end{' + env.lower () + '}'
        return my_latex

    def inv (self):
        myinv = super ().inv ()
        myinv = ScaledTDMSystem (1, myinv)
        myinv.scale = self.scale**-1
        return myinv

    def reduce_to_sorted (self):
        mysorted = super().reduce_to_sorted ()
        mysorted = ScaledTDMSystem (1, mysorted)
        mysorted.scale = self.scale
        return mysorted

def invert_eqn_dict (eqn_dict):
    inv_eqn_dict = {}
    barlen = sum ([len (sector) for sector in eqn_dict.values ()])
    print ("Inverting equation dictionary...")
    with tqdm(total=barlen) as pbar:
        for lbl, sector in eqn_dict.items ():
            sectorI = []
            for tdmsystem in sector:
                new_s = (2*s) - tdmsystem.exprs[0].lhs.get_s_ket ()
                new_m = (2*m) - tdmsystem.exprs[0].lhs.get_m_ket ()
                tdmsystemI = tdmsystem.inv ().subs_m (new_m).subs_s (new_s)
                pbar.update (1)
                sectorI.append (tdmsystemI)
            inv_eqn_dict[lbl] = sectorI
    return inv_eqn_dict

def invert_transpose_eqns (scale):
    cnt = sum ([scalearray.count_transpose_eqns () for scalearray in scale.values ()])
    transpose_eqns = {}
    print ("Inverting transpose equations...")
    with tqdm(total=cnt) as pbar:
        for lbl, scalearray in scale.items ():
            my_transpose_eqns = scalearray.get_transpose_eqns ()
            if len (my_transpose_eqns) > 0:
                transpose_eqns[lbl] = my_transpose_eqns
                pbar.update (len (my_transpose_eqns))
    return transpose_eqns



_docstring_scale = '''Compute the scale factor A(s',s,m) for the transition density matrices

    <s',s"{dm}|{{ops}}|s,s"> = A(s',s,m) <s',m{dm}|{{ops}}|s,m>

    where {cond_mmax} = max (s,s'){cond_dm}
    not accounting for any transposition of spin sectors among the operators if present.'''

#def get_docstring_scale (

if __name__=='__main__':
    import os, sys
    eqn_dict = get_eqn_dict ()
    #inv_eqn_dict = invert_eqn_dict (eqn_dict)
    eqn_dict = standardize_m_s (eqn_dict)
    scale = get_scale_constants (eqn_dict)
    transpose_eqns = invert_transpose_eqns (scale)
    fbase = os.path.splitext (os.path.basename (__file__))[0]
    fname_tex = fbase + '.tex'
    fname_py = fbase + '.generated.py'
    with open (fname_tex, 'w') as f:
        f.write (latex_header)
        f.write ('\\section{TDM scaling constants}\n\n')
        print ("=================== TDM scaling constants ===================")
        for lbl, scalearray in scale.items ():
            print (scalearray)
            f.write (scalearray.latex ())
        f.write ('\\section{TDM transpose equations}\n\n')
        print ("=================== TDM transpose equations ===================")
        for lbl, my_transpose_eqns in transpose_eqns.items ():
            print ("------------------ " + lbl + " ------------------")
            lbl_latex = lbl.replace ('_', '\\_')
            for key, (read_eq, write_eq) in my_transpose_eqns.items ():
                print ("Read " + str(key) + ":")
                print (read_eq)
                print ("Write " + str(key) + ":")
                print (write_eq)
                f.write ('{}, {} read:\n'.format (lbl_latex, key))
                f.write (read_eq.latex () + '\n\n')
                f.write ('{}, {} write:\n'.format (lbl_latex, key))
                f.write (write_eq.latex () + '\n\n')
        f.write ('\n\n\\end{document}')
    with open (fname_py, 'w') as f:
        f.write ('import numpy as np\n\n')
        for scalearray in scale.values ():
            f.write (scalearray.get_scale_code ())
        f.write (scale['h'].get_mupmdown_code (1, 'h', transpose_eqns=transpose_eqns['h']))
        f.write (scale['h'].get_mupmdown_code (3, 'phh', transpose_eqns=transpose_eqns['h']))
        f.write (scale['hh'].get_mupmdown_code (2, 'hh', transpose_eqns=transpose_eqns['hh']))
        f.write (scale['sm'].get_mupmdown_code (2, 'sm'))
        f.write (scale['dm'].get_mupmdown_code (2, 'dm1', transpose_eqns=transpose_eqns['dm']))
        f.write (scale['dm'].get_mupmdown_code (4, 'dm2', transpose_eqns=transpose_eqns['dm']))

