import copy
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

    def simplify_(self):
        for expr in self.exprs:
            expr.simplify_()

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

if __name__=='__main__':
    #print ("============= All creation/all destruction =============")
    a = []
    #print ("------- Alpha only -------")
    a.append (TDMSystem ([solve_pure_destruction (-1, [0,], 0, 0)]))
    a.append (TDMSystem ([solve_pure_destruction (1, [0,], 0, 0)]))
    a.append (TDMSystem ([solve_pure_destruction (-2, [0,0], 0, 0)]))
    a.append (TDMSystem ([solve_pure_destruction (0, [0,0], 0, 0)]))
    a.append (TDMSystem ([solve_pure_destruction (2, [0,0], 0, 0)]))
    a = [e.subs_mket_to_m ().subs_sket_to_s () for e in a]
    #for expr in a: print (expr)
    b = []
    #print ("\n------- Beta only -------")
    b.append (TDMSystem ([solve_pure_destruction (1, [1,], 0, 0)]))
    b.append (TDMSystem ([solve_pure_destruction (-1, [1,], 0, 0)]))
    b.append (TDMSystem ([solve_pure_destruction (2, [1,1], 0, 0)]))
    b.append (TDMSystem ([solve_pure_destruction (0, [1,1], 0, 0)]))
    b.append (TDMSystem ([solve_pure_destruction (-2, [1,1], 0, 0)]))
    b = [e.subs_mket_to_m ().subs_sket_to_s () for e in b]
    #for expr in b: print (expr)
    ab = []
    #print ("\n------- Mixed -------")
    ab.append (TDMSystem ([solve_pure_destruction (-2, [1,0], 0, 0)]))
    ab.append (TDMSystem ([solve_pure_destruction (0, [1,0], 0, 0),
                           solve_pure_destruction (0, [0,1], 0, 0)]))
    ab.append (TDMSystem ([solve_pure_destruction (-2, [1,0], 0, 0)]))
    ab = [e.subs_mket_to_m ().subs_sket_to_s () for e in ab]
    #for expr in ab: print (expr)
    gamma1 = []
    #print ("\n\n============= One-body density =============")
    gamma1.append (TDMSystem ([solve_density (0, [0,], [0,], 0, 0),
                               solve_density (0, [1,], [1,], 0, 0)]))
    gamma1.append (TDMSystem ([solve_density (-2, [0,], [0,], 0, 0)]))
    gamma1.append (TDMSystem ([solve_density (-2, [1,], [1,], 0, 0)]))
    gamma1.append (TDMSystem ([solve_density (-2, [1,], [0,], 0, 0)]))
    gamma1.append (TDMSystem ([solve_density (0, [1,], [0,], 0, 0)]))
    gamma1.append (TDMSystem ([solve_density (2, [1,], [0,], 0, 0)]))
    gamma1 = [e.subs_mket_to_m ().subs_sket_to_s () for e in gamma1]
    #for expr in gamma1: print (expr)
    gamma3h = []
    #print ("\n\n============= Three-half-particle operators =============")
    gamma3h.append (TDMSystem ([solve_density (-2, [0,], [0,0], 1, 1)]))
    gamma3h.append (TDMSystem ([solve_density (-2, [1,], [1,0], 1, 1)]))
    gamma3h.append (TDMSystem ([solve_density (-2, [0,], [0,1], 1, 1)]))
    gamma3h.append (TDMSystem ([solve_density (-2, [1,], [1,1], 1, 1)]))
    gamma3h.append (TDMSystem ([solve_density (3, [0,], [0,0], 0, 0)]))
    gamma3h.append (TDMSystem ([solve_density (3, [1,], [1,0], 0, 0)]))
    gamma3h.append (TDMSystem ([solve_density (3, [0,], [0,1], 0, 0)]))
    gamma3h.append (TDMSystem ([solve_density (3, [1,], [1,1], 0, 0)]))
    gamma3h.append (TDMSystem ([solve_density (0, [0,], [0,0], 1, 1),
                                solve_density (0, [1,], [1,0], 1, 1),
                                solve_density (0, [1,], [0,1], 1, 1)]))
    gamma3h.append (TDMSystem ([solve_density (0, [0,], [0,0], -1, -1),
                                solve_density (0, [1,], [1,0], -1, -1),
                                solve_density (0, [1,], [0,1], -1, -1)]))
    gamma3h[-1] = gamma3h[-1].subs_s(s+Rational(1,2))
    gamma3h[-1].simplify_()
    gamma3h.append (TDMSystem ([solve_density (0, [0,], [0,1], 1, 1),
                                solve_density (0, [0,], [1,0], 1, 1),
                                solve_density (0, [1,], [1,1], 1, 1)]))
    gamma3h.append (TDMSystem ([solve_density (0, [0,], [0,1], -1, 1),
                                solve_density (0, [0,], [1,0], -1, 1),
                                solve_density (0, [1,], [1,1], -1, 1)]))
    gamma3h[-1] = gamma3h[-1].subs_s(s+Rational(1,2))
    gamma3h[-1].simplify_()
    #gamma3h = [e.subs_mket_to_m () for e in gamma3h]
    #gamma3h = [e.subs_sket_to_s () for e in gamma3h]
    #for expr in gamma3h: print (expr)
    gamma2 = []
    #print ("\n\n============= Two-body density =============")
    gamma2.append (TDMSystem ([solve_density (0, [0,0], [0,0], 4, 0)]))
    gamma2.append (TDMSystem ([solve_density (0, [0,1], [1,0], 4, 0)]))
    gamma2.append (TDMSystem ([solve_density (0, [1,1], [1,1], 4, 0)]))
    gamma2.append (TDMSystem ([solve_density (0, [0,0], [0,0], 2, 0),
                               solve_density (0, [0,1], [1,0], 2, 0),
                               #solve_density (2, [1,0], [0,1], 0, 0),
                               solve_density (0, [1,0], [1,0], 2, 0),
                               solve_density (0, [1,1], [1,1], 2, 0)]))
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


    read_exprs = a + b + ab + gamma1 + gamma3h + gamma2

    subsec_lbls = ['ha_d', 'ha_u', 'hb_d', 'hb_u', 'hh_d', 'hh_0', 'hh_u',
                   'dm1', 'sm',
                   'phh_a_3d', 'phh_b_3d', 'phh_a_3u', 'phh_b_3u',
                   'dm2_2', 'dm2_1', 'dm2_0']
    subsec_read = []
    subsec_read.append ([read_exprs[i] for i in (0,27)]) # ha_d
    subsec_read.append ([read_exprs[i] for i in (1,28)]) # ha_u
    subsec_read.append ([read_exprs[i] for i in (6,29)]) # hb_d
    subsec_read.append ([read_exprs[i] for i in (5,30)]) # hb_u
    subsec_read.append ([read_exprs[i] for i in (2,9,10)]) # hh_d
    subsec_read.append ([read_exprs[i] for i in (3,8,11)]) # hh_0
    subsec_read.append ([read_exprs[i] for i in (4,7,12)]) # hh_u
    subsec_read.append ([read_exprs[13],
                         combine_TDMSystem ([read_exprs[i] for i in (14,15)])]) # dm1
    subsec_read.append ([read_exprs[i] for i in (16,17,18)]) # sm
    subsec_read.append ([combine_TDMSystem ([read_exprs[i] for i in (19,20)])])
    subsec_read.append ([combine_TDMSystem ([read_exprs[i] for i in (21,22)])])
    subsec_read.append ([combine_TDMSystem ([read_exprs[i] for i in (23,24)])])
    subsec_read.append ([combine_TDMSystem ([read_exprs[i] for i in (25,26)])])
    subsec_read.append ([combine_TDMSystem ([read_exprs[i] for i in (31,32,33)])])
    subsec_read.append ([read_exprs[i] for i in (34,)])
    subsec_read.append ([read_exprs[i] for i in (35,)])

    import os
    fname = os.path.splitext (os.path.basename (__file__))[0] + '.tex'
    subsec_write = []
    with open (fname, 'w') as f:
        f.write (latex_header)
        for exprs, lbl in zip (subsec_read, subsec_lbls):
            print ("================== " + lbl + " ==================")
            f.write ('\\section{' + lbl.replace ('_','\\_') + '}\n')
            exprsI = []
            for idx, expr in enumerate (exprs):
                new_s = (2*s) - expr.exprs[0].lhs.get_s_ket ()
                new_m = (2*m) - expr.exprs[0].lhs.get_m_ket ()
                my_expr = expr.subs_m (new_m).subs_s (new_s)
                print ("------------------ READ " + str (idx) + "------------------")
                print (my_expr, flush=True)
                f.write ('Read {}:\n\n'.format (idx))
                f.write (my_expr.latex () + '\n\n')
                exprI = expr.inv ().subs_m (new_m).subs_s (new_s)
                print ("------------------ WRITE " + str (idx) + "------------------")
                print (exprI, flush=True)
                f.write ('Write {}:\n\n'.format (idx))
                f.write (exprI.latex () + '\n\n')
                exprsI.append (exprI)
            subsec_write.append (exprsI)
        f.write ('\n\n\\end{document}')

