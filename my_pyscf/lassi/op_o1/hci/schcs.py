import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.lib import logger, param
from pyscf.fci import cistring 
from mrh.my_pyscf.lassi.op_o1 import stdm, frag, hams2ovlp, hsi, rdm
from mrh.my_pyscf.lassi.op_o1.utilities import *
from mrh.my_pyscf.lassi.citools import get_lroots, hci_dot_sivecs, hci_dot_sivecs_ij
from mrh.my_pyscf.lassi.op_o1.hci.chc import ContractHamCI_CHC

class ContractHamCI_SHS (rdm.LRRDM):
    __doc__ = stdm.LSTDM.__doc__ + '''

    SUBCLASS: Contract Hamiltonian on CI vectors and SI vector and integrate over all but one
    fragment, for all fragments, projecting on bra CI/SI vectors.

    Additional args:
        h0 : float
            Constant part of the Hamiltonian
        h1 : ndarray of size ncas**2 or 2*(ncas**2)
            Contains effective 1-electron Hamiltonian amplitudes in second quantization,
            optionally spin-separated
        h2 : ndarray of size ncas**4
            Contains 2-electron Hamiltonian amplitudes in second quantization
        si_bra : ndarray of shape (nprod, nroots_si_bra)
            Contains LASSI eigenvectors on the bra
        si_ket : ndarray of shape (nprod, nroots_si_ket)
            Contains LASSI eigenvectors on the ket

    Additional kwargs:
        add_transpose : logical
            If true, the term with si_bra and si_ket switching places is added to all
            interactions.
    '''
    def __init__(self, las, ints, nlas, lroots, h0, h1, h2, si_bra, si_ket,
                 mask_bra_space=None, mask_ket_space=None, pt_order=None, do_pt_order=None,
                 add_transpose=False, accum=None, log=None, max_memory=param.MAX_MEMORY,
                 dtype=np.float64):
        rdm.LRRDM.__init__(self, ints, nlas, lroots, si_bra, si_ket,
                           mask_bra_space = mask_bra_space,
                           mask_ket_space = mask_ket_space,
                           pt_order=pt_order, do_pt_order=do_pt_order,
                           log=log, max_memory=max_memory, dtype=dtype)
        self.las = las
        if h1.ndim==2: h1 = np.stack ([h1,h1], axis=0)
        self.h0 = h0
        self.h1 = np.ascontiguousarray (h1)
        self.h2 = np.ascontiguousarray (h2)
        self.mask_bra_space = mask_bra_space
        self.mask_ket_space = mask_ket_space
        self.nelec_frs = np.asarray ([[list (i.nelec_r[ket]) for i in ints]
                                      for ket in range (self.nroots)]).transpose (1,0,2)
        self._ispec = None
        self.add_transpose = add_transpose
        self.accum = accum

    get_ham_2q = hams2ovlp.HamS2Ovlp.get_ham_2q
    _hconst_ci_ = ContractHamCI_CHC._hconst_ci_
    init_profiling = ContractHamCI_CHC.init_profiling
    sprint_profile = ContractHamCI_CHC.sprint_profile
    def _add_transpose_(self): return

    # Handling for 1s1c: need to do both a'.sm.b and b'.sp.a explicitly
    ltri = False

    def get_single_rootspace_sivec (self, iroot, bra=False):
        '''A single-rootspace slice of the SI vectors.

        Args:
            iroot: integer
                Rootspace index

        Returns:
            sivec: col-major ndarray of shape (np.prod (lroots[:,iroot], nroots_si)
                SI vectors
        '''
        if self._transpose: bra = not bra
        if bra:
            si = self.si_bra
            mask = self.mask_bra_space
        else:
            si = self.si_ket
            mask = self.mask_ket_space
        iroot = np.where (mask==iroot)[0][0]
        i, j = self.offs_lroots[iroot]
        return si[i:j,:]

    def get_fdm_1space (self, rbra, rket, *inv):
        fdm = super().get_fdm_1space (rbra, rket, *inv)
        if self.add_transpose:
            with lib.temporary_env (self, _transpose=True):
                fdm += super().get_fdm_1space (rbra, rket, *inv)
        return fdm

    def _crunch_env_(self, _crunch_fn, *row):
        if self._fn_row_has_spin (_crunch_fn):
            inv = row[2:-1]
        else:
            inv = row[2:]
        _crunch_fn (*row)

    def fermion_frag_shuffle (self, iroot, frags):
        if self._ispec in frags:
            frags = list (frags)
            frags.remove (self._ispec)
        return super().fermion_frag_shuffle (iroot, frags)

    def split_exc_table_along_frag (self, tab, ifrag):
        tab_i = self.urootstr[ifrag][tab]
        idx, invs = np.unique (tab_i, return_index=True, return_inverse=True, axis=0)[1:]
        with lib.temporary_env (self, _ispec=ifrag):
            for i, ix in enumerate (idx):
                bra, ket = tab[ix]
                tab_i = tab[invs==i]
                yield bra, ket, tab_i

    def _put_hconst_(self, op, bra, ket, *inv, keyorder=None):
        spec = np.ones (self.nfrags, dtype=bool)
        spec[list(set (inv))] = False
        spec = np.where (spec)[0]
        if keyorder is None: keyorder = inv
        tab = self.nonuniq_exc[tuple((bra,ket)) + tuple (keyorder)]
        for i in spec:
            myinv = list (inv) + [i,]
            for bra, ket, tab_i in self.split_exc_table_along_frag (tab, i):
                d_rIIop = self.get_fdm (bra, ket, *myinv, _braket_table=tab_i)
                h_rII = np.tensordot (d_rIIop, op, axes=op.ndim)
                self.ints[i]._put_ham_(bra, ket, h_rII, 0, 0, hermi=1)
        assert (self._ispec is None)

    def _crunch_1d_(self, bra, ket, i):
        '''Compute a single-fragment density fluctuation, for both the 1- and 2-RDMs.'''
        d_rII = self.get_fdm (bra, ket, i) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        h1_sii = self.get_ham_2q (i,i)
        h2_iiii = self.get_ham_2q (i,i,i,i)
        h0 = self.h0 * d_rII 
        h1 = np.multiply.outer (d_rII, h1_sii)
        h2 = np.multiply.outer (d_rII, h2_iiii)
        self.ints[i]._put_ham_(bra, ket, h0, h1, h2, hermi=1)
        d1s = self.ints[i].get_dm1 (bra, ket).transpose (0,1,2,4,3)
        d2 = self.ints[i].get_dm2 (bra, ket).sum (2).transpose (0,1,3,2,5,4)
        op = np.tensordot (d1s, h1_sii, axes=3)
        op += .5*np.tensordot (d2, h2_iiii, axes=4)
        self._put_hconst_(op, bra, ket, i)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1d, self.dw_1d = self.dt_1d + dt, self.dw_1d + dw

    def _crunch_2d_(self, bra, ket, i, j):
        '''Compute a two-fragment density fluctuation.'''
        d_rJJII = self.get_fdm (bra, ket, i, j) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        def _perm (k, l, d_rKKLL):
            h2_kkll = self.get_ham_2q (k,k,l,l)
            h2_kllk = self.get_ham_2q (k,l,l,k)
            d1s_ll = self.ints[l].get_dm1 (bra, ket)
            veff_LL = -np.tensordot (d1s_ll, h2_kllk, axes=((-2,-1),(2,1)))
            vj_LL = np.tensordot (d1s_ll.sum (2), h2_kkll, axes=((-2,-1),(-2,-1)))
            veff_LL += vj_LL[:,:,None,:,:]
            h_rKKskk = np.tensordot (d_rKKLL, veff_LL, axes=2)
            h_srKKkk = h_rKKskk.transpose (3,0,1,2,4,5)
            self.ints[k]._put_ham_(bra, ket, 0, h_srKKkk[0], 0, spin=0)
            self.ints[k]._put_ham_(bra, ket, 0, h_srKKkk[1], 0, spin=3)
            op_KKLL = np.tensordot (self.ints[k].get_dm1 (bra, ket), veff_LL,
                                    axes=((-3,-2,-1),(-3,-2,-1)))
            return op_KKLL
        op_JJII = _perm (j,i,d_rJJII)
        _perm (i,j,d_rJJII.transpose (0,3,4,1,2))
        self._put_hconst_(op_JJII, bra, ket, i, j)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2d, self.dw_2d = self.dt_2d + dt, self.dw_2d + dw

    def _crunch_1c_(self, bra, ket, i, j, s1):
        '''Compute the reduced density matrix elements of a single electron hop; i.e.,
        
        <bra|j'(s1)i(s1)|ket>
        
        i.e.,
        
        j ---s1---> i
            
        and conjugate transpose
        '''
        d_rJJII = self.get_fdm (bra, ket, i, j) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        d1 = self._get_D1_(bra, ket)
        d2 = self._get_D2_(bra, ket)
        inti, intj = self.ints[i], self.ints[j]
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        fac = 1
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j), j)
        d_rJJII *= fac
        d_rIIJJ = d_rJJII.transpose (0,3,4,1,2)

        h_ij = self.get_ham_2q (i,j)[s1]
        h_ijjj = self.get_ham_2q (i,j,j,j).transpose (0,2,3,1) # Mulliken -> Dirac order
        h_iiij = self.get_ham_2q (i,i,i,j).transpose (2,0,1,3) # Mulliken -> Dirac order

        d_IIi = inti.get_p (bra, ket, s1)
        d_IIiii = inti.get_pph (bra, ket, s1).sum (2)
        h_IIj = np.tensordot (d_IIi, h_ij, axes=1)
        h_IIj += np.tensordot (d_IIiii, h_iiij, axes=3)
        h_IIjjj = np.tensordot (d_IIi, h_ijjj, axes=1)
        h_rJJj = np.tensordot (d_rJJII, h_IIj, axes=2)
        h_rJJjjj = np.tensordot (d_rJJII, h_IIjjj, axes=2)
        intj._put_ham_(bra, ket, 0, h_rJJj, h_rJJjjj, spin=s1)

        d_JJj = intj.get_h (bra, ket, s1)
        d_JJjjj = intj.get_phh (bra, ket, s1).sum (2)
        h_JJi = np.tensordot (d_JJj, h_ij.T, axes=1)
        h_JJi += np.tensordot (d_JJjjj, h_ijjj, axes=((-3,-2,-1),(1,2,3)))
        h_JJiii = np.tensordot (d_JJj, h_iiij, axes=((-1),(-1)))
        h_rIIi = np.tensordot (d_rIIJJ, h_JJi, axes=2)
        h_rIIiii = np.tensordot (d_rIIJJ, h_JJiii, axes=2)
        inti._put_ham_(bra, ket, 0, h_rIIi, h_rIIiii, spin=s1)

        h_JJII = np.tensordot (h_JJi, d_IIi, axes=((-1),(-1)))
        h_JJII += np.tensordot (h_JJiii, d_IIiii, axes=((-3,-2,-1),(-3,-2,-1)))
        self._put_hconst_(fac*h_JJII, bra, ket, i, j)

        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1c, self.dw_1c = self.dt_1c + dt, self.dw_1c + dw

    def _crunch_1c1d_(self, bra, ket, i, j, k, s1):
        '''Compute the reduced density matrix elements of a coupled electron-hop and
        density fluctuation.'''
        d_rKKJJII = self.get_fdm (bra, ket, i, j, k) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        inti, intj, intk = self.ints[i], self.ints[j], self.ints[k]
        fac = 1
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k), j)

        h_ijkk = fac * self.get_ham_2q (i,j,k,k)
        h_ikkj = fac * self.get_ham_2q (i,k,k,j)

        d_IIi = inti.get_p (bra, ket, s1)
        d_JJj = intj.get_h (bra, ket, s1)
        d_sKKkk = np.moveaxis (intk.get_dm1 (bra, ket), 2, 0)
        dj_KKkk = d_sKKkk.sum (0)
        dk_KKkk = d_sKKkk[s1]

        d_rKKJJi = np.tensordot (d_rKKJJII, d_IIi, axes=2)
        d_rKKIIj = np.tensordot (d_rKKJJII, d_JJj, axes=((3,4),(0,1)))
        d_rKKij = np.tensordot (d_rKKJJi, d_JJj, axes=((3,4),(0,1)))

        # opposite-spin: Coulomb effect only
        h_rKKkk = np.tensordot (d_rKKij, h_ijkk, axes=2)
        intk._put_ham_(bra, ket, 0, h_rKKkk, 0, spin=3*(1-s1))
        # same spin: Coulomb and exchange
        h_rKKkk -= np.tensordot (d_rKKij, h_ikkj.transpose (0,3,2,1), axes=2)
        intk._put_ham_(bra, ket, 0, h_rKKkk, 0, spin=3*s1)

        h_KKij = np.tensordot (dj_KKkk, h_ijkk, axes=((2,3),(2,3)))
        h_KKij -= np.tensordot (dk_KKkk, h_ikkj, axes=((2,3),(2,1)))
        h_rJJj = np.tensordot (d_rKKJJi, h_KKij, axes=((1,2,5),(0,1,2)))
        intj._put_ham_(bra, ket, 0, h_rJJj, 0, spin=s1)
        h_rIIi = np.tensordot (d_rKKIIj, h_KKij, axes=((1,2,5),(0,1,3)))
        inti._put_ham_(bra, ket, 0, h_rIIi, 0, spin=s1)

        h_KKiJJ = np.tensordot (h_KKij, d_JJj, axes=((-1,),(-1,)))
        h_KKJJII = np.tensordot (h_KKiJJ, d_IIi, axes=((2,),(-1)))
        self._put_hconst_(h_KKJJII, bra, ket, i, j, k)

        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1c1d, self.dw_1c1d = self.dt_1c1d + dt, self.dw_1c1d + dw

    def _crunch_1s_(self, bra, ket, i, j):
        '''Compute the reduced density matrix elements of a spin unit hop; i.e.,

        <bra|i'(a)j'(b)i(b)j(a)|ket>

        i.e.,

        j ---a---> i
        i ---b---> j
    
        and conjugate transpose
        '''
        d_rJJII = self.get_fdm (bra, ket, i, j) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        inti, intj = self.ints[i], self.ints[j]
        fac = -1
        h_jjii = fac *self.get_ham_2q (i,j,j,i).transpose (2,1,0,3)
 
        h_JJii = np.tensordot (intj.get_sm (bra, ket), h_jjii, axes=2)
        h_IIjj = np.tensordot (inti.get_sp (bra, ket), h_jjii, axes=((-2,-1),(-2,-1)))
        
        h_rJJjj = np.tensordot (d_rJJII, h_IIjj, axes=2)
        intj._put_ham_(bra, ket, 0, h_rJJjj, 0, spin=2)

        h_rIIii = np.tensordot (d_rJJII, h_JJii, axes=((1,2),(0,1)))
        inti._put_ham_(bra, ket, 0, h_rIIii, 0, spin=1)

        h_JJII = np.tensordot (h_JJii, inti.get_sp (bra, ket), axes=((-2,-1),(-2,-1)))
        self._put_hconst_(h_JJII, bra, ket, i, j)

        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1s, self.dw_1s = self.dt_1s + dt, self.dw_1s + dw

        return

    def _crunch_1s1c_(self, bra, ket, i, j, k, s1):
        '''Compute the reduced density matrix elements of a spin-charge unit hop; i.e.,

        <bra|i'(a)k'(b)j(b)k(a)|ket>

        i.e.,

        k ---a---> i
        j ---b---> k

        and conjugate transpose
        '''
        d_rKKJJII = self.get_fdm (bra, ket, i, j, k) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        inti, intj, intk = self.ints[i], self.ints[j], self.ints[k]
        s11 = s1
        s12 = 1-s1
        s2 = 2-s1
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac = -1 # i'k'jk -> i'k'kj sign
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k), j)
        h_ikkj = fac * self.get_ham_2q (i,k,k,j)

        d_IIi = inti.get_p (bra, ket, s11)
        d_JJj = intj.get_h (bra, ket, s12)
        d_KKkk = intk.get_smp (bra, ket, s1)

        d_rKKJJi = np.tensordot (d_rKKJJII, d_IIi, axes=2)
        d_rKKIIj = np.tensordot (d_rKKJJII, d_JJj, axes=((3,4),(0,1)))
        d_rKKij = np.tensordot (d_rKKJJi, d_JJj, axes=((3,4),(0,1)))

        h_rKKkk = np.tensordot (d_rKKij, h_ikkj.transpose (0,3,2,1), axes=2)
        intk._put_ham_(bra, ket, 0, h_rKKkk, 0, spin=s2)

        h_KKij = np.tensordot (d_KKkk, h_ikkj, axes=((2,3),(2,1)))
        h_rJJj = np.tensordot (d_rKKJJi, h_KKij, axes=((1,2,5),(0,1,2)))
        intj._put_ham_(bra, ket, 0, h_rJJj, 0, spin=s12)
        h_rIIi = np.tensordot (d_rKKIIj, h_KKij, axes=((1,2,5),(0,1,3)))
        inti._put_ham_(bra, ket, 0, h_rIIi, 0, spin=s11)

        h_KKiJJ = np.tensordot (h_KKij, d_JJj, axes=((-1,),(-1,)))
        h_KKJJII = np.tensordot (h_KKiJJ, d_IIi, axes=((2,),(-1)))
        self._put_hconst_(h_KKJJII, bra, ket, i, j, k)

        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1s1c, self.dw_1s1c = self.dt_1s1c + dt, self.dw_1s1c + dw

        return

    def _crunch_2c_(self, bra, ket, i, j, k, l, s2lt):
        '''Compute the reduced density matrix elements of a two-electron hop; i.e.,
    
        <bra|i'(s1)k'(s2)l(s2)j(s1)|ket>
        
        i.e.,                   
                                
        j ---s1---> i           
        l ---s2---> k           
        
        with
        
        s2lt = 0, 1, 2
        s1   = a, a, b
        s2   = a, b, b
        
        and conjugate transpose
        
        Note that this includes i=k and/or j=l cases, but no other coincident fragment indices. Any
        other coincident fragment index (that is, any coincident index between the bra and the ket)
        turns this into one of the other interactions implemented in the above _crunch_ functions:
        s1 = s2  AND SORT (ik) = SORT (jl)                 : _crunch_1d_ and _crunch_2d_
        s1 = s2  AND (i = j XOR i = l XOR j = k XOR k = l) : _crunch_1c_ and _crunch_1c1d_
        s1 != s2 AND (i = l AND j = k)                     : _crunch_1s_
        s1 != s2 AND (i = l XOR j = k)                     : _crunch_1s1c_
        '''
        d_rJLKI = self.get_fdm (bra, ket, i, k, l, j, keyorder=[i,j,k,l]) # time-profiled by itself
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        # s2lt: 0, 1, 2 -> aa, ab, bb
        # s2: 0, 1, 2, 3 -> aa, ab, ba, bb
        s2  = (0, 1, 3)[s2lt] # aa, ab, bb
        s2T = (0, 2, 3)[s2lt] # aa, ba, bb -> when you populate the e1 <-> e2 permutation
        s11 = s2 // 2
        s12 = s2 % 2
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac = 1 / (1 + int (s11==s12 and i==k and j==l))
        if i != j:
            fac *= (1,-1)[int (i>k)]
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), i)
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), k)
        if j != l:
            fac *= (1,-1)[int (j>l)]
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), j)
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), l)
        h_iklj = self.get_ham_2q (i,j,k,l).transpose (0,2,3,1) # Dirac order
        if s11==s12 and i!=k and j!=l: # exchange
            h_iklj -= self.get_ham_2q (i,l,k,j).transpose (0,2,1,3)
        h_iklj[:] *= fac

        # First pass: opposite fragment
        if i == k:
            h_KIlj = np.tensordot (self.ints[i].get_pp (bra, ket, s2lt), h_iklj, axes=2)
            h_rJLlj = np.tensordot (d_rJLKI, h_KIlj, axes=2)
        else:
            h_Iklj = np.tensordot (self.ints[i].get_p (bra, ket, s11), h_iklj, axes=1)
            h_KIlj = np.tensordot (self.ints[k].get_p (bra, ket, s12), h_Iklj,
                                   axes=((-1),(-3)))
            h_rJLlj = np.tensordot (d_rJLKI, h_KIlj, axes=4)
        if j == l:
            d_JLlj = self.ints[l].get_hh (bra, ket, s2lt)
            h_JLik = np.tensordot (d_JLlj, h_iklj, axes=((-2,-1),(-2,-1)))
            h_rKIik = np.tensordot (d_rJLKI, h_JLik, axes=((1,2),(0,1)))
            h_JLKI = np.tensordot (d_JLlj, h_KIlj, axes=((-2,-1),(-2,-1)))
        else:
            d_Ll = self.ints[l].get_h (bra, ket, s12)
            h_Likj = np.tensordot (d_Ll, h_iklj, axes=((-1),(-2)))
            h_LKIj = np.tensordot (d_Ll, h_KIlj, axes=((-1),(-2)))
            d_Jj = self.ints[j].get_h (bra, ket, s11)
            h_JLik = np.tensordot (d_Jj, h_Likj, axes=((-1),(-1)))
            h_rKIik = np.tensordot (d_rJLKI, h_JLik, axes=((1,2,3,4),(0,1,2,3)))
            h_JLKI = np.tensordot (d_Jj, h_LKIj, axes=((-1),(-1)))

        self._put_hconst_(h_JLKI, bra, ket, i, k, l, j, keyorder=[i,j,k,l])

        # Second pass: with fragment 

        if i == k:
            self.ints[i]._put_ham_(bra, ket, 0, h_rKIik, 0, spin=s2lt)
        else:
            h_rKkIi = np.moveaxis (h_rKIik, -1, 3)
            h_rKk = np.tensordot (h_rKkIi, self.ints[i].get_p (bra, ket, s11), axes=3)
            self.ints[k]._put_ham_(bra, ket, 0, h_rKk, 0, spin=s12)
            h_rIiKk = np.moveaxis (h_rKIik, (1,2), (-3,-2))
            h_rIi = np.tensordot (h_rIiKk, self.ints[k].get_p (bra, ket, s12), axes=3)
            self.ints[i]._put_ham_(bra, ket, 0, h_rIi, 0, spin=s11)
        if j == l:
            self.ints[j]._put_ham_(bra, ket, 0, h_rJLlj, 0, spin=s2lt)
        else:
            h_rJjLl = np.moveaxis (h_rJLlj, -1, 3)
            h_rJj = np.tensordot (h_rJjLl, self.ints[l].get_h (bra, ket, s12), axes=3)
            self.ints[j]._put_ham_(bra, ket, 0, h_rJj, 0, spin=s11)
            h_rLlJj = np.moveaxis (h_rJLlj, (1,2), (-3,-2))
            h_rLl = np.tensordot (h_rLlJj, self.ints[j].get_h (bra, ket, s11), axes=3)
            self.ints[l]._put_ham_(bra, ket, 0, h_rLl, 0, spin=s12)

        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2c, self.dw_2c = self.dt_2c + dt, self.dw_2c + dw

    def kernel (self):
        ''' Main driver method of class.

        Returns:
            hci_fr_pabq : list of length nfrags of list of length nroots of ndarray
        '''
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        self.init_profiling ()
        if self.accum != 1:
            for inti in self.ints: inti._init_ham_(self.nroots_si)
        self._crunch_all_()
        self.hci_fr_plab = self.get_vecs ()
        return self.hci_fr_plab, t0

    def get_vecs (self):
        t1, w1 = logger.process_clock (), logger.perf_counter ()
        hci_fr_plab = []
        for inti in self.ints:
            hci_r_plab = inti._ham_op (_init_only=(self.accum==0))
            hci_fr_plab.append ([hci_r_plab[i] for i in self.mask_bra_space])
        dt, dw = logger.process_clock () - t1, logger.perf_counter () - w1
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw
        return hci_fr_plab

