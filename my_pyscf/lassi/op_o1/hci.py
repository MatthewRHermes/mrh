import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.fci import cistring 
from mrh.my_pyscf.lassi.op_o1 import stdm, frag, hams2ovlp, hsi
from mrh.my_pyscf.lassi.op_o1.utilities import *
from mrh.my_pyscf.lassi.citools import get_lroots, hci_dot_sivecs

class ContractHamCI (stdm.LSTDM):
    __doc__ = stdm.LSTDM.__doc__ + '''

    SUBCLASS: Contract Hamiltonian on CI vectors and integrate over all but one fragment,
    for all fragments.

    Additional args:
        h1 : ndarray of size ncas**2 or 2*(ncas**2)
            Contains effective 1-electron Hamiltonian amplitudes in second quantization,
            optionally spin-separated
        h2 : ndarray of size ncas**4
            Contains 2-electron Hamiltonian amplitudes in second quantization
    '''
    def __init__(self, ints, nlas, hopping_index, lroots, h1, h2, mask_bra_space=None,
                 mask_ket_space=None, log=None, max_memory=2000, dtype=np.float64):
        hams2ovlp.HamS2Ovlp.__init__(self, ints, nlas, hopping_index, lroots, h1, h2,
                                     mask_bra_space = mask_bra_space,
                                     mask_ket_space = mask_ket_space,
                                     log=log, max_memory=max_memory, dtype=dtype)
        self.nbra = len (mask_bra_space)
        self.hci_fr_pabq = self._init_vecs ()

    get_ham_2q = hams2ovlp.HamS2Ovlp.get_ham_2q

    # Handling for 1s1c: need to do both a'.sm.b and b'.sp.a explicitly
    all_interactions_full_square = True
    interaction_has_spin = ('_1c_', '_1c1d_', '_1s1c_', '_2c_')
    ltri_ambiguous = False

    def _init_vecs (self):
        hci_fr_pabq = []
        nfrags, nroots, nbra = self.nfrags, self.nroots, self.nbra
        nprods_ket = np.sum (np.prod (self.lroots[:,:-nbra], axis=0))
        for i in range (nfrags):
            lroots_bra = self.lroots.copy ()[:,-nbra:]
            lroots_bra[i,:] = 1
            nprods_bra = np.prod (lroots_bra, axis=0)
            hci_r_pabq = []
            norb = self.ints[i].norb
            for r in range (self.nbra):
                nelec = self.ints[i].nelec_r[r+self.nroots-self.nbra]
                ndeta = cistring.num_strings (norb, nelec[0])
                ndetb = cistring.num_strings (norb, nelec[1])
                hci_r_pabq.append (np.zeros ((nprods_ket, nprods_bra[r], ndeta, ndetb),
                                             dtype=self.dtype).transpose (1,2,3,0))
            hci_fr_pabq.append (hci_r_pabq)
        return hci_fr_pabq

    def _crunch_1d_(self, bra, ket, i):
        '''Compute a single-fragment density fluctuation, for both the 1- and 2-RDMs.'''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        hci_f_ab, iad, skip = self._get_vecs_(bra, ket, i)
        if skip: return
        h_00 = 0
        h_11 = self.get_ham_2q (i,i)
        h_22 = self.get_ham_2q (i,i,i,i)
        hci_f_ab[i] += self.ints[i].contract_h00 (h_00, h_11, h_22, ket)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1d, self.dw_1d = self.dt_1d + dt, self.dw_1d + dw
        self._put_vecs_(bra, ket, hci_f_ab, i)
        return

    def _crunch_2d_(self, bra, ket, i, j):
        '''Compute a two-fragment density fluctuation.'''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        hci_f_ab, iad, skip = self._get_vecs_(bra, ket, i, j)
        if skip: return
        iad, jad = iad
        def _perm (k, l):
            h2_kkll = self.get_ham_2q (k,k,l,l)
            h2_kllk = self.get_ham_2q (k,l,l,k)
            d1s_ll = self.ints[l].get_1_dm1 (bra, ket)
            d1_ll = d1s_ll.sum (0)
            vj = np.tensordot (h2_kkll, d1_ll, axes=2)
            vk = np.tensordot (d1s_ll, h2_kllk, axes=((1,2),(2,1)))
            # NOTE: you cannot use contract_h00 here b/c d1s_ll is nonsymmetric
            hket  = self.ints[k].contract_h11 (0, vj-vk[0], ket)
            hket += self.ints[k].contract_h11 (3, vj-vk[1], ket)
            return hket
        if jad: hci_f_ab[j] += _perm (j,i)
        if iad: hci_f_ab[i] += _perm (i,j)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2d, self.dw_2d = self.dt_2d + dt, self.dw_2d + dw
        self._put_vecs_(bra, ket, hci_f_ab, i, j)
        return
 
    def _crunch_1c_(self, bra, ket, i, j, s1):
        '''Perform a single electron hop; i.e.,
        
        <bra|j'(s1)i(s1)|ket>
        
        i.e.,
        
        j ---s1---> i
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        hci_f_ab, iad, skip = self._get_vecs_(bra, ket, i, j)
        if skip: return
        iad, jad = iad
        fac = 1
        nelec_f_bra = self.nelec_rf[self.rootaddr[bra]]
        nelec_f_ket = self.nelec_rf[self.rootaddr[ket]]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j), j)
        h1_ij = self.get_ham_2q (i,j)[s1]
        h2_ijjj = self.get_ham_2q (i,j,j,j)
        h2_iiij = self.get_ham_2q (i,i,i,j)
        if iad:
            D_j = self.ints[j].get_1_h (bra, ket, s1)
            D_jjj = self.ints[j].get_1_phh (bra, ket, s1).sum (0)
            h_10 = np.dot (h1_ij, D_j) + np.tensordot (h2_ijjj, D_jjj,
                axes=((1,2,3),(2,0,1)))
            h_21 = np.dot (h2_iiij, D_j).transpose (2,0,1)
            hci_f_ab[i] += fac * self.ints[i].contract_h10 (s1, h_10, h_21, ket)
        if jad:
            D_i = self.ints[i].get_1_p (bra, ket, s1)
            D_iii = self.ints[i].get_1_pph (bra, ket, s1).sum (0)
            h_01 = np.dot (D_i, h1_ij) + np.tensordot (D_iii, h2_iiij,
                axes=((0,1,2),(2,0,1)))
            h_12 = np.tensordot (D_i, h2_ijjj, axes=1).transpose (1,2,0)
            hci_f_ab[j] += fac * self.ints[j].contract_h01 (s1, h_01, h_12, ket)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1c, self.dw_1c = self.dt_1c + dt, self.dw_1c + dw
        self._put_vecs_(bra, ket, hci_f_ab, i, j)
        return

    def _crunch_1c1d_(self, bra, ket, i, j, k, s1):
        '''Compute the reduced density matrix elements of a coupled electron-hop and
        density fluctuation.'''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        hci_f_ab, iad, skip = self._get_vecs_(bra, ket, i, j, k)
        if skip: return
        iad, jad, kad = iad
        fac = 1
        nelec_f_bra = self.nelec_rf[self.rootaddr[bra]]
        nelec_f_ket = self.nelec_rf[self.rootaddr[ket]]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k), j)
        h2j = self.get_ham_2q (i,j,k,k)
        h2k = self.get_ham_2q (i,k,k,j)
        p_i = self.ints[i].get_1_p (bra, ket, s1)
        h_j = self.ints[j].get_1_h (bra, ket, s1)
        if iad or jad:
            d1s_kk = self.ints[k].get_1_dm1 (bra, ket)
            d1_kk = d1s_kk.sum (0)
            h1_ij  = np.tensordot (h2j, d1_kk, axes=2)
            h1_ij -= np.tensordot (d1s_kk[s1], h2k, axes=((0,1),(2,1)))
            if iad:
                h_ = np.dot (h1_ij, h_j)
                hci_f_ab[i] += fac * self.ints[i].contract_h10 (s1, h_, None, ket)
            if jad:
                h_ = np.dot (p_i, h1_ij)
                hci_f_ab[j] += fac * self.ints[j].contract_h01 (s1, h_, None, ket)
        if kad:
            h2j = np.tensordot (p_i, h2j, axes=1)
            h2j = np.tensordot (h_j, h2j, axes=1)
            h_ = h2j # opposite-spin; Coulomb effect only
            hci_f_ab[k] += fac * self.ints[k].contract_h11 (3*(1-s1), h_, ket)
            h2k = np.tensordot (p_i, h2k, axes=1)
            h2k = np.tensordot (h2k, h_j, axes=1)
            h_ -= h2k.T # same-spin; Coulomb and exchange
            hci_f_ab[k] += fac * self.ints[k].contract_h11 (3*s1, h_, ket)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1c1d, self.dw_1c1d = self.dt_1c1d + dt, self.dw_1c1d + dw
        self._put_vecs_(bra, ket, hci_f_ab, i,j,k)
        return

    def _crunch_1s_(self, bra, ket, i, j):
        '''Compute the reduced density matrix elements of a spin unit hop; i.e.,

        <bra|i'(a)j'(b)i(b)j(a)|ket>

        i.e.,

        j ---a---> i
        i ---b---> j

        and conjugate transpose
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        hci_f_ab, iad, skip = self._get_vecs_(bra, ket, i, j)
        if skip: return
        iad, jad = iad
        h2_ijji = self.get_ham_2q (i,j,j,i)
        if iad:
            h1 = lib.einsum ('psrq,rs->pq', h2_ijji, self.ints[j].get_1_sm (bra, ket))
            hci_f_ab[i] -= self.ints[i].contract_h11 (1, h1, ket)
        if jad:
            h1 = lib.einsum ('psrq,pq->rs', h2_ijji, self.ints[i].get_1_sp (bra, ket))
            hci_f_ab[j] -= self.ints[j].contract_h11 (2, h1, ket)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1s, self.dw_1s = self.dt_1s + dt, self.dw_1s + dw
        self._put_vecs_(bra, ket, hci_f_ab, i,j)
        return

    def _crunch_1s1c_(self, bra, ket, i, j, k, s1):
        '''Compute the reduced density matrix elements of a spin-charge unit hop; i.e.,

        <bra|i'(a)k'(b)j(b)k(a)|ket>

        i.e.,

        k ---a---> i
        j ---b---> k

        and conjugate transpose
        '''
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        hci_f_ab, iad, skip = self._get_vecs_(bra, ket, i, j, k)
        if skip: return
        iad, jad, kad = iad
        s11 = s1
        s12 = 1-s1
        s2 = 2-s1
        nelec_f_bra = self.nelec_rf[self.rootaddr[bra]]
        nelec_f_ket = self.nelec_rf[self.rootaddr[ket]]
        fac = -1 # a'bb'a -> a'ab'b signi
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k), j)
        h2 = self.get_ham_2q (i,k,k,j)
        p_i = self.ints[i].get_1_p (bra, ket, s11)
        h_j = self.ints[j].get_1_h (bra, ket, s12)
        if iad or jad:
            s_k = self.ints[k].get_1_smp (bra, ket, s1)
            h1_ij = np.tensordot (h2, s_k, axes=((2,1),(0,1)))
            if iad:
                h_ = np.dot (h1_ij, h_j)
                hci_f_ab[i] += fac * self.ints[i].contract_h10 (s11, h_, None, ket)
            if jad:
                h_ = np.dot (p_i, h1_ij)
                hci_f_ab[j] += fac * self.ints[j].contract_h01 (s12, h_, None, ket)
        if kad:
            h_ = np.tensordot (h2, h_j, axes=1)
            h_ = np.tensordot (p_i, h_, axes=1).T
            hci_f_ab[k] += fac * self.ints[k].contract_h11 (s2, h_, ket)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_1s1c, self.dw_1s1c = self.dt_1s1c + dt, self.dw_1s1c + dw
        self._put_vecs_(bra, ket, hci_f_ab, i,j,k)
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
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        hci_f_ab, iad, skip = self._get_vecs_(bra, ket, i, j, k, l)
        if skip: return
        iad, jad, kad, lad = iad
        # s2lt: 0, 1, 2 -> aa, ab, bb
        # s2: 0, 1, 2, 3 -> aa, ab, ba, bb
        s2  = (0, 1, 3)[s2lt] # aa, ab, bb
        s2T = (0, 2, 3)[s2lt] # aa, ba, bb -> when you populate the e1 <-> e2 permutation
        s11 = s2 // 2
        s12 = s2 % 2
        nelec_f_bra = self.nelec_rf[self.rootaddr[bra]]
        nelec_f_ket = self.nelec_rf[self.rootaddr[ket]]
        # TODO: debug this for 4-fragment interactions
        fac = 1 / (1 + int (s11==s12 and i==k and j==l))
        h_iklj = self.get_ham_2q (i,j,k,l).transpose (0,2,3,1) # Dirac order
        if s11==s12 and i!=k and j!=l: # exchange
            h_iklj -= self.get_ham_2q (i,l,k,j).transpose (0,2,1,3)
        # First pass: Hamiltonians
        if i == k:
            h02_lj = np.tensordot (self.ints[i].get_1_pp (bra, ket, s2lt), h_iklj, axes=2)
        else:
            h02_lj = np.tensordot (self.ints[i].get_1_p (bra, ket, s11), h_iklj, axes=1)
            h02_lj = np.tensordot (self.ints[k].get_1_p (bra, ket, s12), h02_lj, axes=1)
            fac *= (1,-1)[int (i>k)]
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), i)
            fac *= fermion_des_shuffle (nelec_f_bra, (i, j, k, l), k)
        if j == l:
            h20_ik = np.tensordot (h_iklj, self.ints[j].get_1_hh (bra, ket, s2lt), axes=2)
        else:
            h20_ik = np.tensordot (h_iklj, self.ints[j].get_1_h (bra, ket, s11), axes=1)
            h20_ik = np.tensordot (h20_ik, self.ints[l].get_1_h (bra, ket, s12), axes=1)
            fac *= (1,-1)[int (j>l)]
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), j)
            fac *= fermion_des_shuffle (nelec_f_ket, (i, j, k, l), l)
        # Second pass: apply to vectors
        if i == k:
            if iad: hci_f_ab[i] += fac * self.ints[i].contract_h20 (s2lt, h20_ik, ket)
        else:
            if iad:
                h10_i = np.dot (h20_ik, self.ints[k].get_1_p (bra, ket, s12))
                hci_f_ab[i] += fac * self.ints[i].contract_h10 (s11, h10_i, None, ket)
            if kad:
                h10_k = np.dot (self.ints[i].get_1_p (bra, ket, s11), h20_ik)
                hci_f_ab[k] += fac * self.ints[k].contract_h10 (s12, h10_k, None, ket)
        if j == l:
            if jad: hci_f_ab[j] += fac * self.ints[j].contract_h02 (s2lt, h02_lj, ket)
        else:
            if jad:
                h01_j = np.dot (self.ints[l].get_1_h (bra, ket, s12), h02_lj)
                hci_f_ab[j] += fac * self.ints[j].contract_h01 (s11, h01_j, None, ket)
            if lad:
                h01_l = np.dot (h02_lj, self.ints[j].get_1_h (bra, ket, s11))
                hci_f_ab[l] += fac * self.ints[l].contract_h01 (s12, h01_l, None, ket)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_2c, self.dw_2c = self.dt_2c + dt, self.dw_2c + dw
        self._put_vecs_(bra, ket, hci_f_ab, i,j,k,l)
        return

    def env_addr_fragpop (self, bra, i, r):
        rem = 0
        if i>1:
            div = np.prod (self.lroots[:i,r], axis=0)
            bra, rem = divmod (bra, div)
        bra, err = divmod (bra, self.lroots[i,r])
        assert (err==0)
        return bra + rem

    def _bra_address (self, bra, *inv):
        bra_r = self.rootaddr[bra]
        bra_env = self.envaddr[bra]
        lroots_bra_r = self.lroots[:,bra_r]
        bra_r = bra_r + self.nbra - self.nroots
        addressible = set (np.where (bra_env==0)[0])
        addressible = addressible.intersection (set (inv))
        bra_envaddr = []
        for i in addressible:
            lroots_i = lroots_bra_r.copy ()
            lroots_i[i] = 1
            strides = np.append ([1], np.cumprod (lroots_i[:-1]))
            bra_envaddr.append (np.dot (strides, bra_env))
        return bra_r, bra_envaddr, addressible

    def _get_vecs_(self, bra, ket, *inv):
        bra_r, bra_envaddr, addressible = self._bra_address (bra, *inv)
        hci_f_ab = [0 for i in range (self.nfrags)]
        for i, addr in zip (addressible, bra_envaddr):
            hci_r_pabq = self.hci_fr_pabq[i]
            # TODO: buffer
            hci_f_ab[i] = np.zeros_like (hci_r_pabq[bra_r][addr,:,:,ket])
        iad = [i in addressible for i in inv]
        skip = not any (iad)
        return hci_f_ab, iad, skip

    def _put_vecs_(self, bra, ket, vecs, *inv):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        bras, kets, facs = self._get_spec_addr_ovlp (bra, ket, *inv)
        for bra, ket, fac in zip (bras, kets, facs):
            self._put_Svecs_(bra, ket, [fac*vec for vec in vecs], *inv)
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw

    def _put_Svecs_(self, bra, ket, vecs, *inv):
        bra_r, bra_envaddr, addressible = self._bra_address (bra, *inv)
        for i, addr in zip (addressible, bra_envaddr):
            self.hci_fr_pabq[i][bra_r][addr,:,:,ket] += vecs[i]

    #def _crunch_all_(self):
    #    for row in self.exc_1c: self._crunch_env_(self._crunch_1c_, *row)

    def _orbrange_env_kwargs (self, inv): return {}
    def _add_transpose_(self): return

    def _umat_linequiv_(self, ifrag, iroot, umat, *args):
        # TODO: is this even possible?
        pass

    def kernel (self):
        ''' Main driver method of class.

        Returns:
            hci_fr_pabq : list of length nfrags of list of length nroots of ndarray
        '''
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        self.init_profiling ()
        for hci_r_pabq in self.hci_fr_pabq:
            for hci_pabq in hci_r_pabq:
                hci_pabq[:,:,:,:] = 0.0
        self._crunch_all_()
        self._umat_linequiv_loop_()
        return self.hci_fr_pabq, t0

def contract_ham_ci (las, h1, h2, ci_fr_ket, nelec_frs_ket, ci_fr_bra, nelec_frs_bra,
                     si_bra=None, si_ket=None, h0=0, soc=0, orbsym=None, wfnsym=None):
    '''Evaluate the action of the state interaction Hamiltonian on a set of ket CI vectors,
    projected onto a basis of bra CI vectors, leaving one fragment of the bra uncontracted.

    Args:
        las : instance of class LASSCF
        h1 : ndarray of shape (ncas, ncas)
            Spin-orbit-free one-body CAS Hamiltonian
        h2 : ndarray of shape (ncas, ncas, ncas, ncas)
            Spin-orbit-free two-body CAS Hamiltonian
        ci_fr_ket : nested list of shape (nfrags, nroots_ket)
            Contains CI vectors for the ket; element [i,j] is ndarray of shape
            (ndeta_ket[i,j],ndetb_ket[i,j])
        nelec_frs_ket : ndarray of shape (nfrags, nroots_ket, 2)
            Number of electrons of each spin in each rootspace in each
            fragment for the ket vectors
        ci_fr_bra : nested list of shape (nfrags, nroots_bra)
            Contains CI vectors for the bra; element [i,j] is ndarray of shape
            (ndeta_bra[i,j],ndetb_bra[i,j])
        nelec_frs_bra : ndarray of shape (nfrags, nroots_bra, 2)
            Number of electrons of each spin in each
            fragment for the bra vectors

    Kwargs:
        si_bra : ndarray of shape (ndim_bra, *)
            SI vectors for the bra. If provided, the p dimension on the return object is contracted
        si_ket : ndarray of shape (ndim_ket, *)
            SI vectors for the bra. If provided, the q dimension on the return object is contracted
        soc : integer
            Order of spin-orbit coupling included in the Hamiltonian
        h0 : float
            Constant term in the Hamiltonian
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        hket_fr_pabq : nested list of shape (nfrags, nroots_bra)
            Element i,j is an ndarray of shape (ndim_bra//ci_fr_bra[i][j].shape[0],
            ndeta_bra[i,j],ndetb_bra[i,j],ndim_ket).
    '''
    log = lib.logger.new_logger (las, las.verbose)
    nlas = las.ncas_sub
    nfrags, nbra = nelec_frs_bra.shape[:2]
    nket = nelec_frs_ket.shape[1]
    ci = [ci_r_ket + ci_r_bra for ci_r_bra, ci_r_ket in zip (ci_fr_bra, ci_fr_ket)]
    nelec_frs = np.append (nelec_frs_ket, nelec_frs_bra, axis=1)
    nroots = nbra + nket
    mask_bra_space = list (range (nket,nroots))
    mask_ket_space = list (range (nket))
    mask_ints = np.zeros ((nroots,nroots), dtype=bool)
    mask_ints[np.ix_(mask_bra_space,mask_ket_space)] = True

    # First pass: single-fragment intermediates
    hopping_index, ints, lroots = frag.make_ints (las, ci, nelec_frs, nlas=nlas,
                                                  screen_linequiv=False,
                                                  mask_ints=mask_ints)

    # Second pass: upper-triangle
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)
    contracter = ContractHamCI (ints, nlas, hopping_index, lroots, h1, h2,
                                mask_bra_space=mask_bra_space, mask_ket_space=mask_ket_space,
                                dtype=ci[0][0].dtype, max_memory=max_memory, log=log)
    lib.logger.timer (las, 'LASSI Hamiltonian contraction second intermediate indexing setup', *t0)        
    hket_fr_pabq, t0 = contracter.kernel ()
    lib.logger.timer (las, 'LASSI Hamiltonian contraction second intermediate crunching', *t0)

    # Third pass: multiplicative part
    if nfrags>1:
        for ifrag in range (nfrags):
            gen_hket = gen_contract_ham_ci_const (ifrag, nbra, las, h1, h2, ci, nelec_frs, soc=soc,
                                                  h0=h0, orbsym=orbsym, wfnsym=wfnsym)
            for ibra, hket_pabq in enumerate (gen_hket):
                hket_fr_pabq[ifrag][ibra][:] += hket_pabq[:]
    elif h0:
        for ibra in range (nbra):
            nelec_bra = tuple (nelec_frs_bra[0,ibra])
            for iket in range (nket):
                i, j = contracter.offs_lroots[iket]
                nelec_ket = tuple (nelec_frs_ket[0,iket])
                if nelec_bra==nelec_ket:
                    h0ket = h0 * ci_fr_ket[0][iket].transpose (1,2,0)
                    hket_fr_pabq[0][ibra][0,:,:,i:j] += h0ket
    return hci_dot_sivecs (hket_fr_pabq, si_bra, si_ket, get_lroots (ci_fr_bra))

def gen_contract_ham_ci_const (ifrag, nbra, las, h1, h2, ci, nelec_frs, soc=0, h0=0, orbsym=None,
                               wfnsym=None):
    '''Constant-term parts of contract_ham_ci for fragment ifrag'''
    log = lib.logger.new_logger (las, las.verbose)
    nlas = np.asarray (las.ncas_sub)
    nfrags, nroots = nelec_frs.shape[:2]
    nket = nroots - nbra
    dtype = ci[0][0].dtype
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)

    lroots = get_lroots (ci)
    nprods_ket = np.sum (np.prod (lroots[:,:-nbra], axis=0))
    norb_i = nlas[ifrag]
    ci_i = ci[ifrag]
    nelec_i_rs = nelec_frs[ifrag]

    # index down to omit fragment
    idx = np.ones (nfrags, dtype=bool)
    idx[ifrag] = False
    nelec_frs_j = nelec_frs[idx]
    ci_jfrag = [c for i,c in enumerate (ci) if i != ifrag]
    j = sum (nlas[:ifrag+1])
    i = j - nlas[ifrag]
    ix = np.ones (h1.shape[-1], dtype=bool)
    ix[i:j] = False
    if h1.ndim==3:
        h1 = h1[np.ix_([True,True],ix,ix)]
    else:
        h1 = h1[np.ix_(ix,ix)]
    h2 = h2[np.ix_(ix,ix,ix,ix)]
    nlas_j = nlas[idx]

    # Fix sign convention for omitted fragment
    nelec_rf = nelec_frs.sum (-1).T
    class HamS2Ovlp (hsi.HamS2OvlpOperators):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.spin_shuffle = [fermion_spin_shuffle (nelec_frs[:,i,0], nelec_frs[:,i,1])
                                 for i in range (nroots)]
            self.log.verbose = 0
        def fermion_frag_shuffle (self, iroot, frags):
            frags = [f if f<ifrag else f+1 for f in frags]
            return fermion_frag_shuffle (nelec_rf[iroot], frags)

    # Get the intermediate object, rather than just the ham matrix, so that I can use the members
    # of the intermediate to keep track of the difference between the full-system indices and the
    # nfrag-1--system indices
    outerprod = hsi.gen_contract_op_si_hdiag (las, h1, h2, ci_jfrag, nelec_frs_j, nlas=nlas_j,
                                              _HamS2Ovlp_class=HamS2Ovlp, _return_int=True)
    ham_op = outerprod.get_ham_op ()
    ovlp_op = outerprod.get_ovlp_op ()

    for ibra in range (nket, nroots):
        i, j = outerprod.offs_lroots[ibra]
        eye = np.zeros ((ham_op.shape[0], j-i), dtype=ham_op.dtype)
        eye[i:j,:] = np.eye (j-i)
        ham_ij = ham_op (eye) + (h0 * ovlp_op (eye))
        nelec_i = nelec_i_rs[ibra]
        ndeta = cistring.num_strings (norb_i, nelec_i[0])
        ndetb = cistring.num_strings (norb_i, nelec_i[1])
        hket_pabq = np.zeros ((nprods_ket, j-i, ndeta, ndetb),
                              dtype=outerprod.dtype).transpose (1,2,3,0)
        n = 0
        for iket in range (nket):
            m = n
            ci_i_iket = ci_i[iket]
            if ci_i_iket.ndim == 2: ci_i_iket = ci_i_iket[None,...]
            nq1, na, nb = ci_i_iket.shape
            k, l = outerprod.offs_lroots[iket]
            nq2, np1 = ham_ij[k:l].shape
            n = m + nq1*nq2
            if tuple (nelec_i_rs[iket]) != tuple (nelec_i): continue
            hket = np.multiply.outer (ci_i_iket, ham_ij[k:l,:].conj ().T) # qabpq
            new_shape = [nq1,na,nb,np1] + list (outerprod.lroots[::-1,iket])
            hket = np.moveaxis (hket.reshape (new_shape), 0, -(1+ifrag))
            new_shape = [na,nb,np1,nq1*nq2]
            hket = hket.reshape (new_shape).transpose (2,0,1,3)
            hket_pabq[:,:,:,m:n] = hket[:]
        yield hket_pabq

