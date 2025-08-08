import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.lib import logger, param
from itertools import product
from mrh.my_pyscf.fci.csf import unpack_h1e_ab
from mrh.my_pyscf.lassi import citools
from mrh.my_pyscf.lassi.op_o1 import frag, stdm
from mrh.my_pyscf.lassi.op_o1.utilities import *

# S2 is:
# (d2aa_ppqq + d2bb_ppqq - d2ab_ppqq - d2ba_ppqq)/4 - (d2ab_pqqp + d2ba_pqqp)/2
# for two fragments this is (sign?)
# ((d1a_pp - d1b_pp) * (d1a_qq - d1b_qq))/4 - (sp_pp*sm_qq + sm_pp*sp_qq)/2

class HamS2Ovlp (stdm.LSTDM):
    __doc__ = stdm.LSTDM.__doc__ + '''

    SUBCLASS: Hamiltonian, spin-squared, and overlap matrices

    `kernel` call returns operator matrices without cacheing stdm12s array

    Additional args:
        h1 : ndarray of size ncas**2 or 2*(ncas**2)
            Contains effective 1-electron Hamiltonian amplitudes in second quantization,
            optionally spin-separated
        h2 : ndarray of size ncas**4
            Contains 2-electron Hamiltonian amplitudes in second quantization
    '''
    def __init__(self, ints, nlas, lroots, h1, h2, mask_bra_space=None,
                 mask_ket_space=None, pt_order=None, do_pt_order=None, log=None,
                 max_memory=param.MAX_MEMORY, dtype=np.float64):
        stdm.LSTDM.__init__(self, ints, nlas, lroots,
                            mask_bra_space=mask_bra_space, mask_ket_space=mask_ket_space,
                            pt_order=pt_order, do_pt_order=do_pt_order,
                            log=log, max_memory=max_memory, dtype=dtype)
        if h1.ndim==2: h1 = np.stack ([h1,h1], axis=0)
        self.h1 = np.ascontiguousarray (h1)
        self.h2 = np.ascontiguousarray (h2)

    def _init_buffers_(self): pass

    def _add_transpose_(self):
        self.ham += self.ham.conj ().T
        self.s2 += self.s2.T

    def _umat_linequiv_(self, ifrag, iroot, umat, *args):
        self.ham = umat_dot_1frag_(self.ham, umat, self.lroots, ifrag, iroot, axis=0)
        self.ham = umat_dot_1frag_(self.ham, umat, self.lroots, ifrag, iroot, axis=1)
        self.s2 = umat_dot_1frag_(self.s2, umat, self.lroots, ifrag, iroot, axis=0)
        self.s2 = umat_dot_1frag_(self.s2, umat, self.lroots, ifrag, iroot, axis=1)

    def kernel (self):
        ''' Main driver method of class.

        Returns:
            ham : ndarray of shape (nroots,nroots)
                Hamiltonian in LAS product state basis
            s2 : ndarray of shape (nroots,nroots)
                Spin-squared operator in LAS product state basis
            ovlp : ndarray of shape (nroots,nroots)
                Overlap matrix of LAS product states
            t0 : tuple of length 2
                timestamp of entry into this function, for profiling by caller
        '''
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        self.init_profiling ()
        self.ham = np.zeros ([self.nstates,]*2, dtype=self.dtype)
        self.s2 = np.zeros ([self.nstates,]*2, dtype=self.get_ci_dtype ())
        self._crunch_all_()
        t1, w1 = lib.logger.process_clock (), lib.logger.perf_counter ()
        ovlp = self.get_ovlp ()
        dt, dw = logger.process_clock () - t1, logger.perf_counter () - w1
        self.dt_o, self.dw_o = self.dt_o + dt, self.dw_o + dw
        self._umat_linequiv_loop_()
        return self.ham, self.s2, ovlp, t0

    def inv_unique (self, inv):
        invset = set ()
        inv = [i for i in inv if not (i in invset or invset.add (i))]
        return inv

    def canonical_operator_order (self, arr, inv):
        ''' Transpose an operator array for an interaction involving a single pair of rootspaces
        from fragment-major to fragment-minor (bra-ket major) order with decreasing fragment index
        from left to right. No actual data transposition is carried out, only the strides are
        permuted.

        Args:
            arr : ndarray of shape (lroots[inv[0],bra],lroots[inv[0],ket],
                                    lroots[inv[1],bra],lroots[inv[1],ket],...)
            inv : list of integers
                Fragments involved in this array

        Returns:
            arr : discontiguous ndarray of shape (...,lroots[j,bra],lroots[i,bra],
                                                  ...,lroots[j,ket],lroots[i,ket])
                Where [i,j,...] = list (set (inv))
        '''
        if arr is None or arr.ndim < 3: return arr
        assert ((arr.ndim % 2) == 0)
        inv = self.inv_unique (inv)
        ndim = arr.ndim // 2
        # sort by fragments
        axesorder = np.arange (arr.ndim, dtype=int).reshape (ndim, 2)
        inv = np.asarray (inv)
        idx = list (np.argsort (-inv))
        axesorder = axesorder[idx]
        # ...ckbjai -> ...cba...kji
        axesorder = np.ravel (axesorder.T)
        arr = arr.transpose (*axesorder)
        return arr

    def _crunch_env_(self, _crunch_fn, *row):
        if self._fn_row_has_spin (_crunch_fn):
            inv = row[2:-1]
        else:
            inv = row[2:]
        self._prepare_spec_addr_ovlp_(row[0], row[1], *inv)
        ham, s2, ninv = _crunch_fn (*row)
        ham = self.canonical_operator_order (ham, ninv)
        s2 = self.canonical_operator_order (s2, ninv)
        self._put_ham_s2_(row[0], row[1], ham, s2, *inv)

    def _put_op_(self, op, bra, ket, opval, wgt):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        op[bra,ket] += wgt * opval
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_s, self.dw_s = self.dt_s + dt, self.dw_s + dw

    def _put_ham_s2_(self, bra, ket, ham, s2, *inv):
        # TODO: vectorize this part
        bra_rng = self._get_addr_range (bra, *inv) # profiled as idx
        ket_rng = self._get_addr_range (ket, *inv) # profiled as idx
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        ham = ham.flat
        if s2 is None:
            def _put_s2_(b,k,i,w): pass
        else:
            s2 = s2.flat
            def _put_s2_(b,k,i,w): self._put_op_(self.s2, b, k, s2[i], w)
        for ix, (bra1, ket1) in enumerate (product (bra_rng, ket_rng)):
            bra2, ket2, wgt = self._get_spec_addr_ovlp (bra1, ket1, *inv)
            self._put_op_(self.ham, bra2, ket2, ham[ix], wgt)
            _put_s2_(bra2, ket2, ix, wgt)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw

    def get_ham_2q (self, *inv):
        # IDK why np.ascontiguousarray is required, but I fail unittests w/o it
        assert (len (inv) in (2,4))
        i, j = inv[:2]
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        if len (inv) == 2:
            return np.ascontiguousarray (self.h1[:,p:q,r:s]).copy ()
        k, l = inv[2:]
        t, u = self.get_range (k)
        v, w = self.get_range (l)
        return np.ascontiguousarray (self.h2[p:q,r:s,t:u,v:w]).copy ()

    def _crunch_1d_(self, bra, ket, i):
        '''Compute a single-fragment density fluctuation, for both the 1- and 2-RDMs.'''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        inti = self.ints[i]
        d1s = inti.get_dm1 (bra, ket).transpose (0,1,2,4,3)
        d2s = inti.get_dm2 (bra, ket)
        d2 = d2s.sum (2).transpose (0,1,3,2,5,4)
        ham  = np.tensordot (d1s, self.get_ham_2q (i,i), axes=3)
        ham += np.tensordot (d2, self.get_ham_2q (i,i,i,i), axes=4) * .5
        s2 = 3 * np.trace (d1s, axis1=-2, axis2=-1).sum (-1) / 4
        m2 = d2s.diagonal (axis1=3,axis2=4).diagonal (axis1=3, axis2=4).sum ((3,4)) / 4
        m2 = m2.transpose (2,0,1)
        d2s = d2s.transpose (2,0,1,3,4,5,6)
        s2 += m2[0] + m2[3] - m2[1] - m2[2]
        m2 = (d2s[1]+d2s[2]).diagonal (axis1=2, axis2=5).diagonal (axis1=2,axis2=3)
        s2 -= m2.sum ((2,3)) / 2
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1d, self.dw_1d = self.dt_1d + dt, self.dw_1d + dw
        return ham, s2, (i,)
    
    def _crunch_2d_(self, bra, ket, i, j):
        '''Compute a two-fragment density fluctuation.'''
        # 1/2 factor of h2 canceled by i<->j
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        d1s_ii = self.ints[i].get_dm1 (bra, ket)
        d1s_jj = self.ints[j].get_dm1 (bra, ket)
        h_ = self.get_ham_2q (j,j,i,i)
        h_ = np.tensordot (d1s_ii.sum (2), h_, axes=((-1,-2),(-2,-1)))
        h_ = np.tensordot (d1s_jj.sum (2), h_, axes=((-1,-2),(-2,-1)))
        ham = h_
        hs_ = self.get_ham_2q (j,i,i,j).transpose (0,3,2,1)
        hs_ = np.tensordot (d1s_ii, hs_, axes=((-1,-2),(-2,-1)))
        hs_ = hs_.transpose (2,0,1,3,4)
        d1s_ii = d1s_ii.transpose (2,0,1,3,4)
        d1s_jj = d1s_jj.transpose (2,0,1,3,4)
        for h_, d1_jj in zip (hs_, d1s_jj):
            ham -= np.tensordot (d1_jj, h_, axes=((-1,-2),(-2,-1)))
        mi = np.trace (d1s_ii, axis1=-2, axis2=-1)
        mi = (mi[0] - mi[1])
        mj = np.trace (d1s_jj, axis1=-2, axis2=-1)
        mj = (mj[0] - mj[1]) 
        s2 = np.multiply.outer (mj, mi) / 2
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2d, self.dw_2d = self.dt_2d + dt, self.dw_2d + dw
        return ham, s2, (j, i)

    def _crunch_1c_(self, bra, ket, i, j, s1):
        '''Compute the reduced density matrix elements of a single electron hop; i.e.,
        
        <bra|j'(s1)i(s1)|ket>
            
        i.e.,
            
        j ---s1---> i

        and conjugate transpose
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        fac = 1 # h1 involved, so no 1/2 here
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j), j)
        h_ = self.get_ham_2q (j,i)[s1] # BEWARE CONJ
        p_i = self.ints[i].get_p (bra, ket, s1)
        pph_i = self.ints[i].get_pph (bra, ket, s1).sum (2)
        h_j = self.ints[j].get_h (bra, ket, s1)
        phh_j = self.ints[j].get_phh (bra, ket, s1).sum (2)
        h_ = np.tensordot (p_i, h_, axes=((-1),(-1)))
        h_ = np.tensordot (h_j, h_, axes=((-1),(-1)))
        ham = h_
        h_jiii = self.get_ham_2q (j,i,i,i) # BEWARE CONJ
        h_ = np.tensordot (pph_i, h_jiii, axes=((-3,-2,-1),(-3,-1,-2)))
        ham += np.tensordot (h_j, h_, axes=((-1),(-1)))
        h_ = self.get_ham_2q (j,j,j,i) # BEWARE CONJ
        h_ = np.tensordot (p_i, h_, axes=((-1),(-1)))
        ham += np.tensordot (phh_j, h_, axes=((-3,-2,-1),(-2,-3,-1)))
        ham *= fac
        s2 = None
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1c, self.dw_1c = self.dt_1c + dt, self.dw_1c + dw
        return ham, s2, (j, i)

    def _crunch_1c1d_(self, bra, ket, i, j, k, s1):
        '''Compute the reduced density matrix elements of a coupled electron-hop and
        density fluctuation.'''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        fac = 1 # 1/2 factor of h2 canceled by ijkk, ikkj <-> kkij, kjik
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k), j)
        p_i = self.ints[i].get_p (bra, ket, s1)
        h_j = self.ints[j].get_h (bra, ket, s1)
        d1s_k = self.ints[k].get_dm1 (bra, ket)
        d1_k = d1s_k.sum (2)
        h_ = self.get_ham_2q (k,k,j,i) # BEWARE CONJ
        h_ = np.tensordot (p_i, h_, axes=((-1),(-1)))
        h_ = np.tensordot (h_j, h_, axes=((-1),(-1)))
        ham = np.tensordot (d1_k, h_, axes=((-1,-2),(-2,-1)))
        d1_k = d1s_k[:,:,s1]
        h_ = self.get_ham_2q (j,k,k,i).transpose (1,2,0,3) # BEWARE CONJ
        h_ = np.tensordot (p_i, h_, axes=((-1),(-1)))
        h_ = np.tensordot (h_j, h_, axes=((-1),(-1)))
        ham -= np.tensordot (d1_k, h_, axes=((-2,-1),(-2,-1)))
        ham *= fac
        s2 = None
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1c1d, self.dw_1c1d = self.dt_1c1d + dt, self.dw_1c1d + dw
        return ham, s2, (k, j, i)

    def _crunch_1s_(self, bra, ket, i, j):
        '''Compute the reduced density matrix elements of a spin unit hop; i.e.,

        <bra|i'(a)j'(b)i(b)j(a)|ket>

        i.e.,

        j ---a---> i
        i ---b---> j

        and conjugate transpose
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        fac = -1 # 1/2 factor of h2 canceled by i<->j
        sp_i = self.ints[i].get_sp (bra, ket)
        sm_j = self.ints[j].get_sm (bra, ket)
        ham = self.get_ham_2q (j,i,i,j).transpose (0,3,2,1)
        ham = np.tensordot (sp_i, ham, axes=((-2,-1),(-2,-1)))
        ham = np.tensordot (sm_j, ham, axes=((-2,-1),(-2,-1)))
        ham *= fac
        sp_i = np.trace (sp_i, axis1=-2, axis2=-1)
        sm_j = np.trace (sm_j, axis1=-2, axis2=-1)
        s2 = -fac * np.multiply.outer (sm_j, sp_i)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1s, self.dw_1s = self.dt_1s + dt, self.dw_1s + dw
        return ham, s2, (j, i)

    def _crunch_1s1c_(self, bra, ket, i, j, k, s1):
        '''Compute the reduced density matrix elements of a spin-charge unit hop; i.e.,

        <bra|i'(a)k'(b)j(b)k(a)|ket>

        i.e.,

        k ---a---> i
        j ---b---> k

        and conjugate transpose
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        s2 = 1-s1
        fac = -1 # a'bb'a -> a'ab'b sign; 1/2 factor of h2 canceled by ikkj<->kjik
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k), j)
        ham = self.get_ham_2q (j,k,k,i).transpose (1,2,0,3) # BEWARE CONJ
        ham = np.tensordot (self.ints[i].get_p (bra, ket, s1), ham, axes=((-1),(-1)))
        ham = np.tensordot (self.ints[j].get_h (bra, ket, s2), ham, axes=((-1),(-1)))
        if s1 == 0:
            dkk = self.ints[k].get_sm (bra, ket)
        else:
            dkk = self.ints[k].get_sp (bra, ket)
        ham = np.tensordot (dkk, ham, axes=((-2,-1),(-2,-1)))
        ham *= fac
        s2 = None
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1s1c, self.dw_1s1c = self.dt_1s1c + dt, self.dw_1s1c + dw
        return ham, s2, (k, j, i)

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
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        # s2lt: 0, 1, 2 -> aa, ab, bb
        # s2: 0, 1, 2, 3 -> aa, ab, ba, bb
        s2  = (0, 1, 3)[s2lt] # aa, ab, bb
        s2T = (0, 2, 3)[s2lt] # aa, ba, bb -> when you populate the e1 <-> e2 permutation
        s11 = s2 // 2
        s12 = s2 % 2
        nelec_f_bra = self.nelec_rf[bra]
        nelec_f_ket = self.nelec_rf[ket]
        fac = (1,.5)[int ((i,j,s11)==(k,l,s12))] # 1/2 factor of h2 canceled by ijkl <-> klij
        ham = self.get_ham_2q (l,k,j,i).transpose (0,2,3,1) # BEWARE CONJ
        if s11==s12 and i!=k and j!=l: # exchange
            ham -= self.get_ham_2q (l,i,j,k).transpose (0,2,1,3) # BEWARE CONJ
        if i == k:
            ham = np.tensordot (self.ints[i].get_pp (bra, ket, s2lt), ham, axes=((-2,-1),(-2,-1)))
        else:
            ham = np.tensordot (self.ints[k].get_p (bra, ket, s12), ham, axes=((-1),(-1)))
            ham = np.tensordot (self.ints[i].get_p (bra, ket, s11), ham, axes=((-1),(-1)))
            fac *= (1,-1)[int (i>k)]
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), i)
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), k)
        if j == l:
            ham = np.tensordot (self.ints[j].get_hh (bra, ket, s2lt), ham, axes=((-2,-1),(-2,-1)))
        else:
            ham = np.tensordot (self.ints[j].get_h (bra, ket, s11), ham, axes=((-1),(-1)))
            ham = np.tensordot (self.ints[l].get_h (bra, ket, s12), ham, axes=((-1),(-1)))
            fac *= (1,-1)[int (j>l)]
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), j)
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), l)
        ham *= fac
        s2 = None
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2c, self.dw_2c = self.dt_2c + dt, self.dw_2c + dw
        return ham, s2, (l, j, i, k)


def soc_context (h1, h2, ci, nelec_frs, soc, nlas):
    nfrags, nroots = nelec_frs.shape[:2]
    spin_shuffle_fac = None
    n = sum (nlas)
    nelec_rs = [tuple (x) for x in nelec_frs.sum (0)]
    spin_pure = len (set (nelec_rs)) == 1
    if soc and spin_pure: # In this scenario, the off-diagonal sector of h1 is pointless
        h1 = np.stack ([h1[:n,:n], h1[n:,n:]], axis=0)
    elif soc: # Engage the ``spinless mapping''
        ix = np.argsort (spin_shuffle_idx (nlas))
        h1 = h1[np.ix_(ix,ix)]
        h2_ = np.zeros ([2*n,]*4, dtype=h2.dtype)
        h2_[:n,:n,:n,:n] = h2[:]
        h2_[:n,:n,n:,n:] = h2[:]
        h2_[n:,n:,:n,:n] = h2[:]
        h2_[n:,n:,n:,n:] = h2[:]
        h2 = h2_[np.ix_(ix,ix,ix,ix)]
        ci = ci_map2spinless (ci, nlas, nelec_frs)
        nlas = [2*x for x in nlas]
        spin_shuffle_fac = [fermion_spin_shuffle (nelec_frs[:,i,0], nelec_frs[:,i,1])
                            for i in range (nroots)]
        nelec_frs = nelec_frs.copy ()
        nelec_frs[:,:,0] += nelec_frs[:,:,1]
        nelec_frs[:,:,1] = 0
    return spin_pure, h1, h2, ci, nelec_frs, nlas, spin_shuffle_fac


def ham (las, h1, h2, ci, nelec_frs, soc=0, nlas=None, _HamS2Ovlp_class=HamS2Ovlp, _do_kernel=True,
         **kwargs):
    ''' Build Hamiltonian, spin-squared, and overlap matrices in LAS product state basis

    Args:
        las : instance of :class:`LASCINoSymm`
        h1 : ndarray of size ncas**2
            Contains effective 1-electron Hamiltonian amplitudes in second quantization
        h2 : ndarray of size ncas**4
            Contains 2-electron Hamiltonian amplitudes in second quantization
        ci : list of list of ndarrays
            Contains all CI vectors
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Kwargs:
        soc : integer
            Order of spin-orbit coupling included in the Hamiltonian
        nlas : sequence of length (nfrags)
            Number of orbitals in each fragment
        _HamS2Ovlp_class : class
            The main intermediate class
        _do_kernel : logical
            If false, return the main intermediate object before running kernel, instead of the
            operator matrices
        
    Returns: 
        ham : ndarray of shape (nroots,nroots)
            Hamiltonian in LAS product state basis
        s2 : ndarray of shape (nroots,nroots)
            Spin-squared operator in LAS product state basis
        ovlp : ndarray of shape (nroots,nroots)
            Overlap matrix of LAS product states 
        _get_ovlp : callable with kwarg rootidx
            Produce the overlap matrix between model states in a set of rootspaces,
            identified by ndarray or list "rootidx"
    '''     
    verbose = kwargs.get ('verbose', las.verbose)
    log = lib.logger.new_logger (las, verbose) 
    if nlas is None: nlas = las.ncas_sub
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)
    dtype = h1.dtype
    nfrags, nroots = nelec_frs.shape[:2]
    if soc>1: raise NotImplementedError ("Spin-orbit coupling of second order")

    # Handle possible SOC
    spin_pure, h1, h2, ci, nelec_frs, nlas, spin_shuffle_fac = soc_context (
        h1, h2, ci, nelec_frs, soc, nlas)

    # First pass: single-fragment intermediates
    ints, lroots = frag.make_ints (las, ci, nelec_frs, nlas=nlas)
    nstates = np.sum (np.prod (lroots, axis=0))
        
    # Memory check
    current_memory = lib.current_memory ()[0]
    required_memory = dtype.itemsize*nstates*nstates*3/1e6
    if current_memory + required_memory > max_memory:
        raise MemoryError ("current: {}; required: {}; max: {}".format (
            current_memory, required_memory, max_memory))

    # Second pass: upper-triangle
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    outerprod = _HamS2Ovlp_class (ints, nlas, lroots, h1, h2, dtype=dtype,
                                     max_memory=max_memory, log=log)
    if soc and not spin_pure:
        outerprod.spin_shuffle = spin_shuffle_fac
    lib.logger.timer (las, 'LASSI ham setup', *t0)
    if not _do_kernel: return outerprod
    ham, s2, ovlp, t0 = outerprod.kernel ()
    lib.logger.timer (las, 'LASSI ham crunching', *t0)
    if las.verbose >= lib.logger.TIMER_LEVEL:
        lib.logger.info (las, 'LASSI ham crunching profile:\n%s', outerprod.sprint_profile ())

    #raw2orth = citools.get_orth_basis (ci, las.ncas_sub, nelec_frs,
    #                                   _get_ovlp=outerprod.get_ovlp)
    return ham, s2, ovlp, outerprod.get_ovlp


