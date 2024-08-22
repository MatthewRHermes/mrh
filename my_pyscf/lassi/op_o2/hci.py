import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.fci import cistring 
from mrh.my_pyscf.lassi.op_o2 import stdm, frag, hams2ovlp
from mrh.my_pyscf.lassi.op_o2.utilities import *

class ContractHamCI (stdm.LSTDMint2):
    __doc__ = stdm.LSTDMint2.__doc__ + '''

    SUBCLASS: Contract Hamiltonian on CI vectors and integrate over all but one fragment,
    for all fragments.

    Additional args:
        h1 : ndarray of size ncas**2 or 2*(ncas**2)
            Contains effective 1-electron Hamiltonian amplitudes in second quantization,
            optionally spin-separated
        h2 : ndarray of size ncas**4
            Contains 2-electron Hamiltonian amplitudes in second quantization
    '''
    def __init__(self, ints, nlas, hopping_index, lroots, h1, h2, nbra=1,
                 log=None, max_memory=2000, dtype=np.float64):
        nfrags, _, nroots, _ = hopping_index.shape
        if nfrags > 2: raise NotImplementedError ("Spectator fragments in _crunch_1c_")
        nket = nroots - nbra
        hams2ovlp.HamS2ovlpint.__init__(self, ints, nlas, hopping_index, lroots, h1, h2,
                                        mask_bra_space = list (range (nket, nroots)),
                                        mask_ket_space = list (range (nket)),
                                        log=log, max_memory=max_memory, dtype=dtype)
        self.nbra = nbra
        self.hci_fr_pabq = self._init_vecs ()

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

    def _crunch_1c_(self, bra, ket, i, j, s1):
        '''Perform a single electron hop; i.e.,
        
        <bra|j'(s1)i(s1)|ket>
        
        i.e.,
        
        j ---s1---> i
        '''
        hci_f_ab, excfrags = self._get_vecs_(bra, ket)
        excfrags = excfrags.intersection ({i, j})
        if self.nfrags > 2: raise NotImplementedError ("Spectator fragments in _crunch_1c_")
        if not len (excfrags): return
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        fac = 1
        nelec_f_bra = self.nelec_rf[self.rootaddr[bra]]
        nelec_f_ket = self.nelec_rf[self.rootaddr[ket]]
        fac *= fermion_des_shuffle (nelec_f_bra, (i, j), i)
        fac *= fermion_des_shuffle (nelec_f_ket, (i, j), j)
        h1_ij = self.h1[s1,p:q,r:s]
        h2_ijjj = self.h2[p:q,r:s,r:s,r:s]
        h2_iiij = self.h2[p:q,p:q,p:q,r:s]
        if i in excfrags:
            D_j = self.ints[j].get_1_h (bra, ket, s1)
            D_jjj = self.ints[j].get_1_phh (bra, ket, s1).sum (0)
            h_10 = np.dot (h1_ij, D_j) + np.tensordot (h2_ijjj, D_jjj,
                axes=((1,2,3),(2,0,1)))
            h_21 = np.dot (h2_iiij, D_j).transpose (2,0,1)
            hci_f_ab[i] += fac * self.ints[i].contract_h10 (s1, h_10, h_21, ket)
        if j in excfrags:
            D_i = self.ints[i].get_1_p (bra, ket, s1)
            D_iii = self.ints[i].get_1_pph (bra, ket, s1).sum (0)
            h_01 = np.dot (D_i, h1_ij) + np.tensordot (D_iii, h2_iiij,
                axes=((0,1,2),(2,0,1)))
            h_12 = np.tensordot (D_i, h2_ijjj, axes=1).transpose (1,2,0)
            hci_f_ab[j] += fac * self.ints[j].contract_h01 (s1, h_01, h_12, ket)
        self._put_vecs_(bra, ket, hci_f_ab, i, j)
        return

    def _crunch_1s_(self, bra, ket, i, j):
        raise NotImplementedError

    def _crunch_1s1c_(self, bra, ket, i, j, k):
        raise NotImplementedError

    def _crunch_2c_(self, bra, ket, i, j, k, l, s2lt):
        raise NotImplementedError

    def env_addr_fragpop (self, bra, i, r):
        rem = 0
        if i>1:
            div = np.prod (self.lroots[:i,r], axis=0)
            bra, rem = divmod (bra, div)
        bra, err = divmod (bra, self.lroots[i,r])
        assert (err==0)
        return bra + rem

    def _bra_address (self, bra):
        bra_r = self.rootaddr[bra]
        bra_env = self.envaddr[bra]
        lroots_bra_r = self.lroots[:,bra_r]
        bra_r = bra_r + self.nbra - self.nroots
        excfrags = set (np.where (bra_env==0)[0])
        bra_envaddr = []
        for i in excfrags:
            lroots_i = lroots_bra_r.copy ()
            lroots_i[i] = 1
            strides = np.append ([1], np.cumprod (lroots_i[:-1]))
            bra_envaddr.append (np.dot (strides, bra_env))
        return bra_r, bra_envaddr, excfrags

    def _get_vecs_(self, bra, ket):
        bra_r, bra_envaddr, excfrags = self._bra_address (bra)
        hci_f_ab = [0 for i in range (self.nfrags)]
        for i, addr in zip (excfrags, bra_envaddr):
            hci_r_pabq = self.hci_fr_pabq[i]
            # TODO: buffer
            hci_f_ab[i] = np.zeros_like (hci_r_pabq[bra_r][addr,:,:,ket])
        return hci_f_ab, excfrags

    def _put_vecs_(self, bra, ket, vecs, *inv):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        bras, kets, facs = self._get_spec_addr_ovlp (bra, ket, *inv)
        for bra, ket, fac in zip (bras, kets, facs):
            hci_r_pabq = self.hci_fr_pabq[i]
            # TODO: buffer
            hci_f_ab[i] = np.zeros_like (hci_r_pabq[bra_r][addr,:,:,ket])
        return hci_f_ab, excfrags

    def _put_vecs_(self, bra, ket, vecs, *inv):
        t0, w0 = logger.process_clock (), logger.perf_counter ()
        bras, kets, facs = self._get_spec_addr_ovlp (bra, ket, *inv)
        for bra, ket, fac in zip (bras, kets, facs):
            self._put_Svecs_(bra, ket, [fac*vec for vec in vecs])
        dt, dw = logger.process_clock () - t0, logger.perf_counter () - w0
        self.dt_p, self.dw_p = self.dt_p + dt, self.dw_p + dw

    def _put_Svecs_(self, bra, ket, vecs):
        bra_r, bra_envaddr, excfrags = self._bra_address (bra)
        for i, addr in zip (excfrags, bra_envaddr):
            self.hci_fr_pabq[i][bra_r][addr,:,:,ket] += vecs[i]

    def _crunch_all_(self):
        for row in self.exc_1c: self._crunch_env_(self._crunch_1c_, *row)

    def _orbrange_env_kwargs (self, inv): return {}

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
                     soc=0, orbsym=None, wfnsym=None):
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
        soc : integer
            Order of spin-orbit coupling included in the Hamiltonian
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
    log = lib.logger.new_logger (las, las.verbose)
    nlas = las.ncas_sub
    nfrags, nbra = nelec_frs_bra.shape[:2]
    nket = nelec_frs_ket.shape[1]
    ci = [ci_r_ket + ci_r_bra for ci_r_bra, ci_r_ket in zip (ci_fr_bra, ci_fr_ket)]
    nelec_frs = np.append (nelec_frs_ket, nelec_frs_bra, axis=1)

    # First pass: single-fragment intermediates
    hopping_index, ints, lroots = frag.make_ints (las, ci, nelec_frs, screen_linequiv=False)

    # Second pass: upper-triangle
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)
    contracter = ContractHamCI (ints, nlas, hopping_index, lroots, h1, h2, nbra=nbra,
                                dtype=ci[0][0].dtype, max_memory=max_memory, log=log)
    lib.logger.timer (las, 'LASSI Hamiltonian contraction second intermediate indexing setup', *t0)        
    hket_fr_pabq, t0 = contracter.kernel ()
    lib.logger.timer (las, 'LASSI Hamiltonian contraction second intermediate crunching', *t0)

    return hket_fr_pabq


