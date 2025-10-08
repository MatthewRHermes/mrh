import copy
import numpy as np
import sympy
from mrh.my_sympy.spin import spin_1h
from sympy import S, Rational, symbols, Matrix, Poly, apart, powsimp, cancel
from sympy.utilities.lambdify import lambdastr
import itertools
from mrh.my_sympy.spin.lassi_tdms_spins.glob import *

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

    def has_spin_op (self): return True
    def count_spin_sectors (self): return 1
    def get_dmndim (self): return sum (self.count_ops ())
    def has_mirror_sym (self): return False

    def subs_labels_(self, lbl_dict):
        self.indices = [lbl_dict[i] for i in self.indices]

    def get_sort_score (self):
        score = np.power (list (range (len (self.get_ops ()))), 2)[::-1]
        addr = self.crops
        return np.dot (score, addr)

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
        from mrh.my_sympy.spin.lassi_tdms_spins.main import TDMExpression
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

    def get_sort_score (self):
        score = np.power (list (range (len (self.get_ops ()))), 2)[::-1]
        addr = 1-np.asarray (self.crops[::-1])
        return np.dot (score, addr)

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

    def has_spin_op (self):
        # NOTE: I don't think this logic generalizes past 2 electrons!!
        ops = self.count_ops ()
        return (ops[0] != ops[1])

    def get_dmndim (self):
        ndim = sum (self.count_ops ())
        if self.count_spin_sectors () > 1:
            ndim += 1
        return ndim

    def count_spin_sectors (self):
        nel = min (len (self.crops), len (self.anops))
        crel = np.asarray (self.crops)[-nel:]
        anel = np.asarray (self.anops)[-nel:][::-1]
        return 2**np.count_nonzero (crel==anel)

    def has_mirror_sym (self):
        return ((not self.has_spin_op ()) and self.count_spins () == (0,0))

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

    def get_sort_score (self):
        score = np.power (list (range (len (self.get_ops ()))), 2)[::-1]
        addr = np.append (self.crops, 1-np.asarray (self.anops))
        return np.dot (score, addr)


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
            (new_crops[0] != new_anops[0]) and (new_crops[0] == new_anops[1])):
                new_anops = new_anops[::-1]
                new_indices[1:3] = new_indices[1:3][::-1]
                factor *= -1
        if ((len (self.crops) == 2) and (len (self.anops) == 1) and
            (new_crops[1] != new_anops[0]) and (new_crops[0] == new_anops[0])):
                new_crops = new_crops[::-1]
                new_indices[:2] = new_indices[:2][::-1]
                factor *= -1
        new_op = CrAnOperator (self.s_bra, new_crops, new_anops, self.s_ket, self.m_ket,
                               indices=new_indices)
        return factor, new_op

    def normal_order_labels (self, spin_priority=True, keep_particles_together=False):
        new_indices = [i for i in self.get_indices ()]
        factor = 1
        cond_cr = new_indices[0] > new_indices[1]
        cond_an = new_indices[-2] > new_indices[-1]
        if spin_priority: # absolutely preserve spin order
            cond_cr = (self.crops[0]==self.crops[1]) and cond_cr
            cond_an = (self.anops[0]==self.anops[1]) and cond_an
        if keep_particles_together:
            cond_an = cond_cr
        new_crops = [c for c in self.crops]
        new_anops = [a for a in self.anops]
        if (len (self.crops) == 2) and cond_cr:
            new_indices[:2] = new_indices[:2][::-1]
            new_crops[:2] = new_crops[:2][::-1]
            factor *= -1
        if (len (self.anops) == 2) and cond_an:
            new_indices[-2:] = new_indices[-2:][::-1]
            new_anops[:2] = new_anops[:2][::-1]
            factor *= -1
        new_op = CrAnOperator (self.s_bra, new_crops, new_anops, self.s_ket, self.m_ket,
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

