import copy
from tqdm import tqdm
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

def normal_order_factor (crops):
    nperms = 0
    nskip = 0
    for spin in crops:
        if spin==0:
            nperms += nskip
        else: nskip += 1
    factor = (1,-1)[nperms % 2]
    return factor

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

ORBINDICES = 'pqrstuvwxyz'

class CrVector (object):
    def __init__(self, s_bra, crops, s_ket, m_ket, indices=None):
        if indices is None:
            indices = list (ORBINDICES[:len(crops)])
        dm = Rational (get_d2s_fromarray (crops), 2)
        self.s_bra = s_bra
        self.m_bra = m_ket + dm
        self.crops = crops
        self.indices = indices
        self.s_ket = s_ket
        self.m_ket = m_ket

    def count_spins (self):
        nbeta = sum (self.crops)
        nalpha = len (self.crops) - nbeta
        return (nalpha, nbeta)

    def count_ops (self):
        return (len(self.crops),0)

    def transpose (self, idx):
        s_bra = self.get_s_bra ()
        m_ket = self.get_m_ket ()
        s_ket = self.get_s_ket ()
        ops = self.get_ops ()
        indices = self.get_indices ()
        indices = [indices[i] for i in idx]
        return self.__class__(s_bra, crops, s_ket, m_ket, indices=indices)

    def max_m (self):
        dm = Rational (get_d2s_fromarray (self.crops), 2)
        new_m_ket = min (self.s_ket, self.s_bra - dm)
        a, b = Poly (self.m_ket, m).all_coeffs ()
        return (new_m_ket - b) / a

    def subs_m (self, new_m):
        s_bra, ops, s_ket = self.get_s_bra (), self.get_ops (), self.get_s_ket ()
        indices = self.get_indices ()
        m_ket = self.get_m_ket ().subs (m, new_m)
        return self.__class__(s_bra, ops, s_ket, m_ket, indices=indices)

    def subs_s (self, new_s):
        ops = self.get_ops ()
        indices = self.get_indices ()
        m_ket = self.get_m_ket ().subs (s, new_s)
        s_bra = self.get_s_bra ().subs (s, new_s)
        s_ket = self.get_s_ket ().subs (s, new_s)
        return self.__class__(s_bra, ops, s_ket, m_ket, indices=indices)

    def get_ops (self): return list(self.crops)
    def get_indices (self): return list(self.indices)
    def get_s_bra (self): return self.s_bra
    def get_s_ket (self): return self.s_ket
    def get_m_bra (self): return self.m_bra
    def get_m_ket (self): return self.m_ket
    def get_sum_coeff (self): return [1,]
    def get_sum_terms (self): return [self,]

    def cmp_ops (self, other):
        sops, sidx = self.get_ops (), self.get_indices ()
        oops, oidx = other.get_ops (), self.get_indices ()
        if len (sops) != len (oops): return False
        if len (sidx) != len (oidx): return False
        return all ([x==y for x, y in zip (sops+sidx,oops+oidx)])

    def __hash__(self):
        return hash ((0,self.s_bra,self.s_ket,self.m_ket)
                     + tuple (self.crops) + tuple (self.indices))

    def __eq__(self, other):
        return (hash (self) == hash (other))

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
        s = '<' + str (self.get_s_bra()) + ',' + str (self.get_m_bra()) + '| '
        s = s + self._str_op ()
        s = s + '|' + str (self.get_s_ket()) + ',' + str (self.get_m_ket()) + '>'
        return s

    def _str_op (self):
        s = ''
        for crop, lbl in zip (self.get_ops (), self.get_indices ()):
            cr = ('a','b')[crop] + lbl + "' "
            s = s + cr
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
        my_latex = '\\braket{' + str (self.get_s_bra()) + ',' + str (self.get_m_bra()) + '|'
        my_latex += self._latex_op ()
        my_latex += '|' + str (self.get_s_ket()) + ',' + str (self.get_m_ket()) + '}'
        return my_latex

    def _latex_op (self):
        my_latex = ''
        for crop, lbl in zip (self.get_ops (), self.get_indices ()):
            cr = ('a','b')[crop]
            my_latex += '\\cr' + cr + 'op{' + lbl + '}'
        return my_latex

    def normal_order (self):
        idx = np.argsort (np.asarray (self.crops), kind='stable')
        new_crops = [self.crops[ix] for ix in idx]
        new_indices = [self.indices[ix] for ix in idx]
        factor = normal_order_factor (self.crops)
        new_vector = self.normal_order_newvector (new_crops, new_indices)
        return factor, new_vector

    def normal_order_newvector (self, crops, indices):
        return CrVector (self.s_bra, crops, self.s_ket, self.m_ket, indices=indices)

    def __add__(self, other):
        c0 = self.get_sum_coeff ()
        c1 = other.get_sum_coeff ()
        t0 = self.get_sum_terms ()
        t1 = other.get_sum_terms ()
        t2 = list (set (t0+t1))
        c2 = []
        for ti in t2:
            ci = 0
            if ti in t0:
                ci += c0[t0.index (ti)]
            if ti in t1:
                ci += c1[t1.index (ti)]
            c2.append (ci)
        #return OpSum (t0+t1,c0+c1)
        return OpSum (t2, c2)

    def __sub__(self, other):
        c0 = self.get_sum_coeff ()
        c1 = [-c for c in other.get_sum_coeff ()]
        t0 = self.get_sum_terms ()
        t1 = other.get_sum_terms ()
        t2 = list (set (t0+t1))
        c2 = []
        for ti in t2:
            ci = 0
            if ti in t0:
                ci += c0[t0.index (ti)]
            if ti in t1:
                ci += c1[t1.index (ti)]
            c2.append (ci)
        #return OpSum (t0+t1,c0+c1)
        return OpSum (t2, c2)

class AnVector (CrVector):
    '''The idea is that this is a CrVector; we just do I/O in the opposite order'''
    def __init__(self, s_bra, anops, s_ket, m_ket, indices=None):
        if indices is None:
            indices = list (ORBINDICES[:len(anops)])
        dm = Rational (get_d2s_fromarray (anops), 2)
        m_bra = m_ket - dm
        crops = anops[::-1]
        super().__init__(s_ket, crops, s_bra, m_bra, indices=indices[::-1])

    def count_spins (self):
        nbeta = sum (self.crops)
        nalpha = len (self.crops) - nbeta
        return (-nalpha, -nbeta)

    def count_ops (self):
        return (0,len(self.crops))

    def get_ops (self): return list(self.crops[::-1])
    def get_indices (self): return list(self.indices[::-1])
    def get_s_bra (self): return self.s_ket
    def get_s_ket (self): return self.s_bra
    def get_m_bra (self): return self.m_ket
    def get_m_ket (self): return self.m_bra

    def __hash__(self):
        return hash ((1,self.s_bra,self.s_ket,self.m_ket)
                     + tuple (self.crops) + tuple (self.indices))

    def normal_order_newvector (self, crops, indices):
        return AnVector (self.s_ket, crops[::-1], self.s_bra, self.m_bra, indices=indices[::-1])

    def _str_op (self):
        s = ''
        for crop, lbl in zip (self.get_ops (), self.get_indices ()):
            cr = ('a','b')[crop] + lbl + " "
            s = s + cr
        return s

    def get_spinupvecs (self, anops_spinup):
        anvecs_spinup = [AnVector (self.s_ket, anop_spinup, self.s_bra, self.s_bra)
                      for anop_spinup in anops_spinup]
        return anvecs_spinup

    @property
    def H (self):
        return CrVector (self.s_bra, self.crops, self.s_ket, self.m_ket)

    def _latex_op (self):
        my_latex = ''
        for crop, lbl in zip (self.get_ops (), self.get_indices ()):
            cr = ('a','b')[crop]
            my_latex += '\\an' + cr + 'op{' + lbl + '}'
        return my_latex

class CrAnOperator (CrVector):
    def __init__(self, s_bra, crops, anops, s_ket, m_ket, indices=None):
        if indices is None:
            indices = list(ORBINDICES[:len(crops)+len(anops)])
        dm_cr = Rational (get_d2s_fromarray (crops), 2)
        dm_an = Rational (get_d2s_fromarray (anops), 2)
        self.s_bra = s_bra
        self.m_bra = m_ket + dm_cr - dm_an
        self.crops = crops
        self.anops = anops
        self.indices = indices
        self.s_ket = s_ket
        self.m_ket = m_ket

    def count_spins (self):
        nbeta_cr = sum (self.crops)
        nalpha_cr = len (self.crops) - nbeta_cr
        nbeta_an = sum (self.anops)
        nalpha_an = len (self.anops) - nbeta_an
        return (nalpha_cr-nalpha_an, nbeta_cr-nbeta_an)

    def count_ops (self):
        return (len(self.crops),len(self.anops))

    def transpose (self, idx):
        s_bra = self.get_s_bra ()
        s_ket = self.get_s_ket ()
        m_ket = self.get_m_ket ()
        crops = self.crops
        anops = self.anops
        indices = self.get_indices ()
        indices = [indices[i] for i in idx]
        return self.__class__(s_bra, crops, anops, s_ket, m_ket, indices=indices)

    def max_m (self):
        dm = Rational (get_d2s_fromarray (self.crops), 2)
        dm -= Rational (get_d2s_fromarray (self.anops), 2)
        new_m_ket = min (self.s_ket, self.s_bra - dm)
        a, b = Poly (self.m_ket, m).all_coeffs ()
        return (new_m_ket - b) / a

    def subs_m (self, new_m):
        s_bra, s_ket = self.get_s_bra (), self.get_s_ket ()
        indices = self.get_indices ()
        crops = self.crops
        anops = self.anops
        m_ket = self.get_m_ket ().subs (m, new_m)
        return self.__class__(s_bra, crops, anops, s_ket, m_ket, indices=indices)

    def subs_s (self, new_s):
        indices = self.get_indices ()
        crops = self.crops
        anops = self.anops
        m_ket = self.get_m_ket ().subs (s, new_s)
        s_bra = self.get_s_bra ().subs (s, new_s)
        s_ket = self.get_s_ket ().subs (s, new_s)
        return self.__class__(s_bra, crops, anops, s_ket, m_ket, indices=indices)

    def get_ops (self): return list(self.crops) + list(self.anops)
    def get_indices (self): return list(self.indices)

    def __hash__(self):
        return hash ((2,len(self.crops),len(self.anops),self.s_bra,self.s_ket,self.m_ket)
                     + tuple (self.crops) + tuple (self.anops) + tuple (self.indices))

    def _str_op(self):
        s = ''
        for crop, lbl in zip (self.crops, self.indices):
            cr = ('a','b')[crop] + lbl + "' "
            s = s + cr
        for anop, lbl in zip (self.anops, self.indices[len(self.crops):]):
            an = ('a','b')[anop] + lbl + " "
            s = s + an
        return s

    def _latex_op (self):
        my_latex = ''
        for crop, lbl in zip (self.crops, self.indices):
            my_latex += '\\cr' + ('a','b')[crop] + 'op{' + lbl + '}'
        for anop, lbl in zip (self.anops, self.indices[len(self.crops):]):
            my_latex += '\\an' + ('a','b')[anop] + 'op{' + lbl + '}'
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

    def normal_order (self):
        idx_cr = list (np.argsort (self.crops, kind='stable'))
        new_crops = [self.crops[ix] for ix in idx_cr]
        idx_an = list (np.argsort (-np.asarray (self.anops), kind='stable'))
        new_anops = [self.anops[ix] for ix in idx_an]
        cr_indices = self.indices
        an_indices = self.indices[len(self.crops):]
        new_indices = [cr_indices[ix] for ix in idx_cr] + [an_indices[ix] for ix in idx_an]
        factor = normal_order_factor (self.crops) * normal_order_factor (self.anops[::-1])
        if ((len (self.crops) == 1) and (len (self.anops) == 2) and 
            (new_crops[0] != new_anops[0])):
                new_anops = new_anops[::-1]
                new_indices[1:3] = new_indices[1:3][::-1]
                factor *= -1
        if ((len (self.crops) == 2) and (len (self.anops) == 1) and 
            (new_crops[1] != new_anops[0])):
                new_crops = new_crops[::-1]
                new_indices[:2] = new_indices[:2][::-1]
                factor *= -1
        new_op = CrAnOperator (self.s_bra, new_crops, new_anops, self.s_ket, self.m_ket,
                               indices=new_indices)
        return factor, new_op

    def normal_order_labels (self):
        new_indices = [i for i in self.get_indices ()]
        factor = 1
        if ((len (self.crops) == 2) and (self.crops[0]==self.crops[1]) and
            (new_indices[0] > new_indices[1])):
                new_indices[:2] = new_indices[:2][::-1]
                factor *= -1
        if ((len (self.anops) == 2) and (self.anops[0]==self.anops[1]) and
            (new_indices[-2] > new_indices[-1])):
                new_indices[-2:] = new_indices[-2:][::-1]
                factor *= -1
        new_op = CrAnOperator (self.s_bra, self.crops, self.anops, self.s_ket, self.m_ket,
                               indices=new_indices)
        return factor, new_op

class OpSum (CrVector):
    def __init__(self, terms, coeffs):
        self.terms = terms
        self.coeffs = coeffs
        assert (all ([(t.s_bra == terms[0].s_bra) for t in terms]))
        assert (all ([(t.s_ket == terms[0].s_ket) for t in terms]))
        assert (all ([(t.m_bra == terms[0].m_bra) for t in terms]))
        assert (all ([(t.m_ket == terms[0].m_ket) for t in terms]))

    def transpose (self, idx):
        new_terms = [t.transpose (idx) for t in self.terms]
        return OpSum (new_terms, self.coeffs)

    def subs_m (self, new_m):
        return OpSum ([t.subs_m (new_m) for t in self.terms], self.coeffs)

    def get_s_bra (self): return self.terms[0].s_bra
    def get_s_ket (self): return self.terms[0].s_ket
    def get_m_bra (self): return self.terms[0].m_bra
    def get_m_ket (self): return self.terms[0].m_ket
    def get_sum_coeff (self): return self.coeffs
    def get_sum_terms (self): return self.terms

    def __hash__(self):
        return hash (tuple ((3,len(self.terms))) + tuple ((hash (t) for t in self.terms)))

    def _str_op(self):
        if self.coeffs[0] == -1:
            s = '-'
        elif self.coeffs[0] != 1:
            s = str (self.coeffs[0])
        else:
            s = ''
        s += self.terms[0]._str_op ()
        for c, t in zip (self.coeffs[1:], self.terms[1:]):
            if c < 0:
                s += ' - '
            else:
                s += ' + '
            if abs (c) == 1:
                s += t._str_op ()
            else:
                s += str (c) + t._str_op ()
        return s

    def _latex_op (self):
        if self.coeffs[0] == -1:
            s = '-'
        elif self.coeffs[0] != 1:
            s = str (self.coeffs[0])
        else:
            s = ''
        s += self.terms[0]._latex_op ()
        for c, t in zip (self.coeffs[1:], self.terms[1:]):
            if c < 0:
                s += ' - '
            else:
                s += ' + '
            if abs (c) == 1:
                s += t._latex_op ()
            else:
                s += str (c) + t._latex_op ()
        s += ')'
        return s

    def normal_order (self):
        new_coeffs = []
        new_terms = []
        for c, t in zip (self.coeffs, self.terms):
            nc, nt = t.normal_order ()
            new_terms.append (t)
            new_coeffs.append (c*nc)
        f = 1
        if new_coeffs[0] < 0:
            f = -1
            new_coeffs = [-c for c in new_coeffs]
        return f, OpSum (new_terms, new_coeffs)
        
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
        self.inv_tdmsystems_array = None

    def get_dm_types (self):
        dm_types = []
        for row in self.lhs:
            for col in row:
                for el in col:
                    dm_types.append (el.count_ops ())
        return list (set (dm_types))

    def dm_type_str (self, dm_type):
        ncr, ndes = dm_type
        sym = '<'
        for i in range (ncr):
            sym += 'c' + ORBINDICES[i] + "' "
        for i in range (ndes):
            sym += 'c' + ORBINDICES[i+ncr] + " "
        sym = sym[:-1] + '>'
        return sym

    def dm_type_latex (self, dm_type):
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
                text += '[' + row_idx + '][' + col_idx +'] = '
                text += str (self.mat[i,j]) + '\n'
        return text

    def latex_row_lhs (self, row_index, col_index):
        superscript = sympy.latex (Rational (row_index, 2))
        subscript = self.col_index_str (col_index)
        text = '\\mathcal{N}^{(' + superscript + ')}'
        if len (subscript):
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

    def get_transpose_eqns (self):
        transpose_eqns = []
        for i, row in enumerate (self.tdmsystems_array):
            rowI = self.inv_tdmsystems_array[i]
            for j, (col, colI) in enumerate (zip (row, rowI)):
                const = self.mat[i,j]
                for el, elI in zip (col, colI):
                    if len (el.cols) > 1:
                        forward = (el / const).simplify_().powsimp_().simplify_()
                        reverse = (elI * const).simplify_().powsimp_().simplify_()
                        if set ([x for x in forward.get_A ()]) != set ((0,1)):
                            # not effectively diagonal
                            transpose_eqns.append ([forward, reverse])
        return transpose_eqns

def get_scale_constants (eqn_dict, inv_eqn_dict):
    scale = {}
    scale['1_h'] = TDMScaleArray ('1_h',
        [[eqn_dict['phh_a_3d'], eqn_dict['phh_b_3d']],
         [eqn_dict['ha_d'], eqn_dict['hb_d']],
         [eqn_dict['ha_u'], eqn_dict['hb_u']],
         [eqn_dict['phh_a_3u'], eqn_dict['phh_b_3u']]]
    )
    scale['1_hh'] = TDMScaleArray ('1_hh', [[[el,] for el in eqn_dict[key]]
                                            for key in ('hh_d', 'hh_0', 'hh_u')])
    scale['sm'] = TDMScaleArray ('sm', [[[el,],] for el in eqn_dict['sm']])
    scale['dm'] = TDMScaleArray ('dm', [[eqn_dict['dm_2']],
                                        [eqn_dict['dm_1']],
                                        [eqn_dict['dm_0']]])
    eqn_dict = inv_eqn_dict
    scale['1_h'].inv_tdmsystems_array = \
        [[eqn_dict['phh_a_3d'], eqn_dict['phh_b_3d']],
         [eqn_dict['ha_d'], eqn_dict['hb_d']],
         [eqn_dict['ha_u'], eqn_dict['hb_u']],
         [eqn_dict['phh_a_3u'], eqn_dict['phh_b_3u']]]
    scale['1_hh'].inv_tdmsystems_array = [[[el,] for el in eqn_dict[key]]
                                          for key in ('hh_d', 'hh_0', 'hh_u')]
    scale['sm'].inv_tdmsystems_array = [[[el,],] for el in eqn_dict['sm']]
    scale['dm'].inv_tdmsystems_array = [[eqn_dict['dm_2']],
                                        [eqn_dict['dm_1']],
                                        [eqn_dict['dm_0']]]
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

if __name__=='__main__':
    import os, sys
    eqn_dict = get_eqn_dict ()
    inv_eqn_dict = invert_eqn_dict (eqn_dict)
    eqn_dict = standardize_m_s (eqn_dict)
    scale = get_scale_constants (eqn_dict, inv_eqn_dict)
    fname = os.path.splitext (os.path.basename (__file__))[0] + '.tex'
    with open (fname, 'w') as f:
        f.write (latex_header)
        f.write ('\\section{TDM scaling constants}\n\n')
        print ("=================== TDM scaling constants ===================")
        transpose_eqns = {}
        for lbl, scalearray in scale.items ():
            print (scalearray)
            f.write (scalearray.latex ())
            my_transpose_eqns = scalearray.get_transpose_eqns ()
            if len (my_transpose_eqns) > 0:
                transpose_eqns[lbl] = my_transpose_eqns
        f.write ('\\section{TDM transpose equations}\n\n')
        print ("=================== TDM transpose equations ===================")
        for lbl, my_transpose_eqns in transpose_eqns.items ():
            print ("------------------ " + lbl + " ------------------")
            lbl_latex = lbl.replace ('_', '\\_')
            for idx, (read_eq, write_eq) in enumerate (my_transpose_eqns):
                print ("Read " + str(idx) + ":")
                print (read_eq)
                print ("Write " + str(idx) + ":")
                print (write_eq)
                f.write ('{} {} read:\n'.format (lbl_latex, idx))
                f.write (read_eq.latex () + '\n\n')
                f.write ('{} {} write:\n'.format (lbl_latex, idx))
                f.write (write_eq.latex () + '\n\n')
        f.write ('\n\n\\end{document}')

