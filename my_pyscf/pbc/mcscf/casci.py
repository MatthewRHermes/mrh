
import os
import sys
import tempfile
import numpy as np
from functools import reduce

from pyscf import lib, mcscf, __config__
from pyscf.lib import logger
from pyscf.pbc import scf
from pyscf.fci.addons import _unpack_nelec

from mrh.my_pyscf.pbc import fci as pbc_fci
from mrh.my_pyscf.pbc.mcscf.k2R import  get_mo_coeff_k2R

# Global variable to store the 2e integrals
_CDERI_CACHE_PATH = None


WITH_META_LOWDIN = getattr(__config__, 'mcscf_analyze_with_meta_lowdin', True)


if sys.version_info < (3,):
    RANGE_TYPE = list
else:
    RANGE_TYPE = range

# 
# Generalization of the CASCI module with complex integrals for PBC systems.
# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>
#

'''
Structure is as follow:
k-space mo-coeff: -> h1, h2 in r-space: do ci: -> back transform the mo_coeff to k-space:
in case of rdm also, since the ci are in r-space, construct the 1-RDM and 2-RDM and backtransform it.
'''

def _basis_transformation(operator, mo):
    return reduce(np.dot, (mo.conj().T, operator, mo))

def h1e_for_cas(mc, mo_coeff=None, ncas=None, ncore=None):
    '''
    Compute the 1e Hamiltonian for CAS space and core energy.
    Args:
        mc : pbc.mcscf.CASCI
            The CASCI object.
        mo_coeff : np.ndarray [nk, nao, nmo_k]
            orbitals at each k-point.
        ncas : int
            number of active space orbitals in unit cell (i.e. at each k-point).
        ncore : int
            number of core orbitals in unit cell (i.e. at each k-point).
    Returns:
        h1e_cas : np.ndarray [nk, ncas, ncas]
            The one-electron Hamiltonian in CAS space, still in k-point basis.
        ecore : np.complex128
            The core energy.
    '''
    
    if mo_coeff is None: 
        mo_coeff = mc.mo_coeff
    if ncas is None: 
        ncas = mc.ncas
    if ncore is None: 
        ncore = mc.ncore
    
    cell = mc.cell
    nao = cell.nao_nr()

    dtype = mc.mo_coeff[0].dtype
    nkpts = mc.nkpts
    mo_core_kpts = [mo[:, :ncore] for mo in mo_coeff]

    h1ao_k = mc.get_hcore()

    # Remember, I am multiplying by nkpts here because total energy would be divided by nkpts later.
    ecore = mc.energy_nuc() * nkpts
    if len(mo_core_kpts) == 0:
        corevhf_kpts = 0
    else:
        coredm_kpts = np.asarray([2.0 * (mo_core_kpts[k] @ mo_core_kpts[k].conj().T) 
                                  for k in range(nkpts)], dtype=dtype)
        corevhf_kpts = mc._scf.get_veff(cell, coredm_kpts, hermi=1)
        fock = h1ao_k + 0.5 * corevhf_kpts
        ecore += sum(np.einsum('ij,ji', coredm_kpts[k], fock[k]) for k in range(nkpts))
        fock = None  # Free memory

    h1ao_k += corevhf_kpts

    phase, mo_coeff_R = get_mo_coeff_k2R(mc._scf, mo_coeff, ncore, ncas)[1:3]
    h1ao_R = np.einsum('Rk,kij,Sk->RiSj', phase, h1ao_k, phase.conj())
    h1ao_R = h1ao_R.reshape(nkpts*nao, nkpts*nao)
    h1eff_R = _basis_transformation(h1ao_R, mo_coeff_R)
    return h1eff_R, ecore

@lib.with_doc(mcscf.casci.get_fock.__doc__)
def get_fock(mc, mo_coeff=None, ci=None, eris=None, casdm1=None, verbose=None):
    
    if ci is None: 
        ci = mc.ci
    if mo_coeff is None: 
        mo_coeff = mc.mo_coeff
    
    cell = mc.cell
    nkpts = mc.nkpts
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas

    if casdm1 is None:
        casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
    
    dtype = casdm1.dtype

    mo_core_kpts = [mo[:, :ncore] for mo in mo_coeff]
    dm_core = np.asarray([2.0 * (mo_core_kpts[k] @ mo_core_kpts[k].conj().T) 
                                for k in range(nkpts)], dtype=dtype)
    
    hcore_k = mc.get_hcore()
    fock = np.empty_like(hcore_k, dtype=dtype)

    for k in range(nkpts):
        mocas = mo_coeff[k][:,ncore:nocc]
        dm = dm_core[k]
        dm += reduce(np.dot, (mocas, casdm1[k], mocas.conj().T))
        vj, vk = mc._scf.get_jk(cell, dm)
        fock[k] = hcore_k[k] + vj - 0.5*vk
    
    hcore_k = dm_core = mo_core_kpts = None

    return fock

def cas_natorb(**kwargs):
    # TODO: Currently, the natural orbitals are not the prime goal.
    raise NotImplementedError

@lib.with_doc(mcscf.casci.canonicalize.__doc__)
def canonicalize(mc, mo_coeff=None, ci=None, eris=None, sort=False, 
                 cas_natorb=False, casdm1=None, verbose=logger.NOTE,
                 with_meta_lowdin=WITH_META_LOWDIN, stav_dm1=False):
    from pyscf.mcscf import addons
    log = logger.new_logger(mc, verbose)
    log.debug('Canonicalizing CAS orbitals')

    if mo_coeff is None: 
        mo_coeff = mc.mo_coeff
    if ci is None: 
        ci = mc.ci
        
    nkpts = mc.nkpts
    ncas = mc.ncas
    nelecas = mc.nelecas
    ncore = mc.ncore
    nocc = ncore + ncas
    nmo = mo_coeff[0].shape[1]

    if casdm1 is None:
        if (isinstance(ci, (list, tuple, RANGE_TYPE)) and
                not isinstance(mc.fcisolver, addons.StateAverageFCISolver)):
            if stav_dm1:
                log.warn('Mulitple states found in CASCI solver. '
                         'Use state-average 1RDM  to compute the Fock matrix'
                         ' and natural orbitals in the active space.')
                
                casdm1 = mc.fcisolver.make_rdm1(ci[0], nkpts*ncas, (nkpts*nelecas[0], nkpts*nelecas[1]))
                for root in range(1, len(ci)):
                    casdm1 += mc.fcisolver.make_rdm1(ci[root], nkpts*ncas,
                                                     (nkpts*nelecas[0], nkpts*nelecas[1]))
                casdm1 /= len(ci)
            else:
                log.warn('Mulitple states found in CASCI solver. '
                         'First state is used to compute the Fock matrix'
                         ' and natural orbitals in active space.')
                casdm1 = mc.fcisolver.make_rdm1(ci[0], nkpts*ncas, (nkpts*nelecas[0], nkpts*nelecas[1]))
        else:
            casdm1 = mc.fcisolver.make_rdm1(ci, nkpts*ncas, (nkpts*nelecas[0], nkpts*nelecas[1]))
    
    fock_ao = get_fock(mc, mo_coeff=mo_coeff, ci=ci, casdm1=casdm1, verbose=verbose)

    if cas_natorb:
        # Currently not implemented.
        mo_coeff = cas_natorb(mc, mo_coeff, casdm1, verbose=verbose)
    else:
        mo_coeff1 = mo_coeff.copy()
        log.info('Density matrix diagonal elements')
        for k in range(nkpts):
            mo_cas = mo_coeff[k][:, ncore:nocc]
            # This function needed to be updated
            dm_k = mo_cas @ casdm1[k] @ mo_cas.conj().T 
            log.info('k-point %d, diagonal elements of dm: %s', k, np.diag(dm_k))

    mo_energy = [np.einsum('pi, pi-> i', mo_coeff1.conj(), fock_ao[k] @ mo_coeff1) 
                 for k in range(nkpts)]
    
    if getattr(mo_coeff, 'orbsym', None) is not None:
        raise NotImplementedError('Orbital symmetry is not implemented for PBC CASCI yet.')
    else:
        orbsym = np.zeros(nmo, dtype=int)

    extrasym = getattr(mc, 'extrasym', None)
    if extrasym is not None:
        raise NotImplementedError('Extra symmetry is not implemented for PBC CASCI yet.')
    else:
        orbsym_extra = orbsym

    def _diag_subfock_(idx):
        if idx.size > 1:
            for k in range(nkpts):
                c = mo_coeff1[k][:,idx]
                fock = reduce(np.dot, (c.conj().T, fock_ao[k], c))
                w, c = mc._eig(fock, None, None, orbsym_extra[idx])

                if sort:
                    sub_order = np.argsort(w.round(9), kind='mergesort')
                    w = w[sub_order]
                    c = c[:,sub_order]
                    orbsym[idx] = orbsym[idx][sub_order]

                mo_coeff1[k][:,idx] = mo_coeff1[k][:,idx].dot(c)
                mo_energy[k][idx] = w

    mask = np.ones(nmo, dtype=bool)
    frozen = getattr(mc, 'frozen', None)
    if frozen is not None:
        if isinstance(frozen, (int, np.integer)):
            mask[:frozen] = False
        else:
            mask[frozen] = False

    # The loop over k-pts is done under the _diag_subfock.
    core_idx = np.where(mask[:ncore])[0]
    vir_idx = np.where(mask[nocc:])[0] + nocc
    _diag_subfock_(core_idx)
    _diag_subfock_(vir_idx)

    # Not needed, but when I will implement the symmetry...
    if getattr(mo_coeff, 'orbsym', None) is not None:
        mo_coeff1 = lib.tag_array(mo_coeff1, orbsym=orbsym)

    if log.verbose >= logger.DEBUG:
        for k in range(nkpts):
            log.debug('k-point %d', k)
            for i in range(nmo):
                    log.debug('i = %d  <i|F|i> = %12.8f', i+1, mo_energy[k][i])

    return mo_coeff1, ci, mo_energy

def kernel(mc, mo_coeff=None, ci0=None, verbose=logger.NOTE):
    '''
    '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci0 is None: ci0 = mc.ci

    log = logger.new_logger(mc, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start CASCI')

    nkpts = mc.nkpts
    ncas = mc.ncas
    nelecas = mc.nelecas
    nelecas = _unpack_nelec(nelecas, mc._scf.cell.spin)
    eri_cas = mc.get_h2eff(mo_coeff)
    t1 = log.timer('integral transformation to CAS space', *t0)
    h1eff, energy_core = mc.get_h1eff(mo_coeff)
    log.debug('core energy = %.15g', energy_core)
    max_memory = max(4000, mc.max_memory-lib.current_memory()[0])
    
    assert eri_cas.shape == (nkpts*ncas, nkpts*ncas, nkpts*ncas, nkpts*ncas)
    assert h1eff.shape == (nkpts*ncas, nkpts*ncas)

    e_tot, fcivec = mc.fcisolver.kernel(h1eff, eri_cas, 
                                           nkpts*ncas, (nkpts*nelecas[0],nkpts*nelecas[1]), 
                                           ci0=ci0, verbose=log, max_memory=max_memory, 
                                           ecore=energy_core)
    t1 = log.timer('FCI solver', *t1)
    e_cas = e_tot - energy_core

    # The energy is per-unit cell
    e_cas /= nkpts
    e_tot /= nkpts
    return e_tot, e_cas, fcivec

class PBCCASBASE(mcscf.casci.CASBase):
    """
    Sub class of the CASBase. I should blindly copy all the objects of the parent
    CASBase, instead should rewrite it more carefully.

    args:
        kmf: instance of pbc.scf.hf 
            PBC scf object
        ncas: int
            number of active space orbitals (per k-point)
        nelecas: int or tuple
            number of active electrons (per k-point)
    
    attributes:

        verbose: int (kmf.verbose)
            Verbosity level for logging and output control.
        max_memory: int (kmf.max_memory)
            Maximum memory allowed for the calculation in MB.
        ncas: int
            Number of active space orbitals (per k-point).
        nelecas: int or tuple
            Number of active electrons (per k-point).
        ncore: int (I haven't extended the c-FCI to UHF-cFCI, in that case it
                should be a tuple.) See original code for more information.
        natorb: bool (False)
            Active space in the natural orbital basis. In case of the DMRG
            we should be careful with this, because it can affect the wave function itself.
        sorting_mo_energy: bool (False)
            Sort the orbitals based on the diagonal elements of the general Fock matrix. 
        fcisolver: an instance of class cFCI
            There are different varieties of the FCIsolver in PySCF, but currently I 
            have only translated the fci.direct_spin1.
            TODO: See if you can translate the other FCI solvers?
            Note: fcisolver itself will have a lot of attributes, which can be modified as >>> mc.fcisolver....
        
    Saved results:
        e_tot: np.complex64
            Total CASCI energy + nulcear repulsion energy, in case of more than one-k-points, it would be average energy per k-point.
        e_cas: np.complex64
           CAS space energy without core energy:  in case of more than one-k-points, it would be average energy per k-point.
        ci: list or ndarray
            CI vector(s) representing the wavefunction in the CAS space.
        mo_coeff: list
            [nk, nao, nmo] 
            nk: number of k-points
            nao: number of atomic orbitals
            nmo: number of molecular orbitals
        mo_energy: list
            [nk, nmo]
            Diagonal elements of Gen. fock matrix
        mo_occ: list
            [nk, nmo]
            Occupation numbers of molecular orbitals.
            It should be equals to the natural orbital if the natural orbitals are true, otherwise canonical orbital occupation number.
    """

    # Some global variables: initialized as consistent with molecular CASCI global variable: still I have
    # kept them separate to avoid any conflict in future.
    natorb = getattr(__config__, 'pbc_mcscf_casci_CASCI_natorb', False)
    canonicalization = getattr(__config__, 'pbc_mcscf_casci_CASCI_canonicalization', True)
    sorting_mo_energy = getattr(__config__, 'pbc_mcscf_casci_CASCI_sorting_mo_energy', False)

    _keys = {
        'natorb', 'canonicalization', 'sorting_mo_energy', 'cell', 'max_memory',
        'ncas', 'nelecas', 'ncore', 'fcisolver', 'frozen', 'extrasym',
        'e_tot', 'e_cas', 'ci', 'mo_coeff', 'mo_energy', 'mo_occ', 'converged',
        'nkpts', 'scell'
    }

    def __init__(self, kmf, ncas=0, nelecas=0, ncore=None):
        cell = kmf.cell
        self.cell = cell
        self.nkpts = len(kmf.kpts)
        self._scf = kmf
        self.verbose = cell.verbose
        self.stdout = cell.stdout
        self.max_memory = kmf.max_memory
        self.ncas = ncas
        if isinstance(nelecas, (int, np.integer)):
            nelecb = (nelecas-cell.spin)//2
            neleca = nelecas - nelecb
            self.nelecas = (neleca, nelecb)
        else:
            self.nelecas = (nelecas[0],nelecas[1])
        self.ncore = ncore
        singlet = (getattr(__config__, 'mcscf_casci_CASCI_fcisolver_direct_spin0', False)
                   and self.nelecas[0] == self.nelecas[1])  # leads to direct_spin1
        self.fcisolver = pbc_fci.solver(cell, singlet, symm=False)
        self.fcisolver.lindep = getattr(__config__, 'mcscf_casci_CASCI_fcisolver_lindep', 1e-12)
        self.fcisolver.max_cycle = getattr(__config__, 'mcscf_casci_CASCI_fcisolver_max_cycle', 200)
        self.fcisolver.conv_tol = getattr(__config__, 'mcscf_casci_CASCI_fcisolver_conv_tol', 1e-8)
        self.frozen = None # [nk, nmo]
        self.extrasym = None

        self.e_tot = 0
        self.e_cas = None
        self.ci = None
        self.mo_coeff = kmf.mo_coeff
        self.mo_energy = kmf.mo_energy
        self.mo_occ = None
        self.converged = False
    
    @property
    def ncore(self):
        if self._ncore is None:
            ncorelec = self.cell.nelectron - sum(self.nelecas)
            assert ncorelec % 2 == 0
            assert ncorelec >= 0
            return ncorelec // 2
        else: return self._ncore
    
    @ncore.setter
    def ncore(self, value):
        assert value is None or isinstance(value, (int, np.integer)) or value >=0
        self._ncore = value

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        ncore = self.ncore
        ncas = self.ncas
        nvir = self.mo_coeff[0].shape[1] - ncore - ncas
        nkpts = self.nkpts
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d',
                 self.nelecas[0], self.nelecas[1], ncas, ncore, nvir)
        if self.frozen is not None: raise NotImplementedError
        if self.extrasym is not None: raise NotImplementedError
        log.info('nkpts = %d', nkpts)
        log.info('natorb = %s', self.natorb)
        log.info('canonicalization = %s', self.canonicalization)
        log.info('sorting_mo_energy = %s', self.sorting_mo_energy)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        self.fcisolver.dump_flags(log.verbose)

        if self.mo_coeff is None:
            log.error('MO coefficients are not set')
        
        if (getattr(self._scf, 'with_solvent', None) and
            not getattr(self, 'with_solvent', None)): raise NotImplementedError
        
        return self
    
    def check_sanity(self):
        super().check_sanity()
        assert self.ncas > 0
        ncore = self.ncore
        nvir = self.mo_coeff[0].shape[1] - ncore - self.ncas
        assert ncore >= 0
        assert nvir >= 0
        assert ncore * 2 + sum(self.nelecas) == self.cell.nelectron # I think it should be warning. As this would be required for the band gap calculations.
        assert 0 <= self.nelecas[0] <= self.ncas
        assert 0 <= self.nelecas[1] <= self.ncas
        return self
    
    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
            self.fcisolver.cell = cell
        self._scf.reset(cell)
        return self

    def energy_nuc(self):
        return self._scf.energy_nuc()
    
    def get_veff(self, cell=None, dm=None, hermi=1, kpt=None):
        # Note this would be in k-space: would need transformation
        # before its direct use.
        return self._scf.get_veff(cell=cell, dm=dm, hermi=hermi, kpt=kpt)
    
    # Defining all of these functions here to initialize it. Do we really need it?
    def _eig(**kwargs):
        pass

    def get_h2cas(**kwargs):
        pass

    def get_h2eff(**kwargs):
        pass
    
    def ao2mo(**kwargs):
        pass
    
    def get_hcore(self, **kwargs):
        '''
        Wrapper to avoid printing a huge output for computing the hcore.
        '''
        with lib.temporary_env(self._scf, verbose=0):
            self._scf.cell.verbose = 0
            return self._scf.get_hcore(**kwargs)

    def get_h1cas(**kwargs):
        pass
    
    def get_h1cas(self, mo_coeff=None, ncas=None, ncore=None):
        '''An alias of get_h1eff method'''
        return self.get_h1eff(mo_coeff, ncas, ncore)

    get_h1eff = h1e_for_cas = h1e_for_cas

    @lib.with_doc(scf.hf.get_jk.__doc__)
    def get_jk(self, cell, dm, hermi=1, vhfopt=None, kpt=np.zeros(3),
           kpts_band=None, with_j=True, with_k=True, omega=None, **kwargs):
        return self._scf.get_jk(cell=cell, dm=dm, hermi=hermi, vhfopt=vhfopt, 
                                kpt=kpt, kpts_band=kpts_band, with_j=with_j, 
                                with_k=with_k, omega=omega, **kwargs)
    
    def _finalize(self):
        log = logger.Logger(self.stdout, self.verbose)
        nkpts = self.nkpts
        if log.verbose >= logger.NOTE and getattr(self.fcisolver, 'spin_square', None):
            if isinstance(self.e_cas, (np.complex128, np.float64)):
                try:
                    ss = self.fcisolver.spin_square(self.ci, nkpts*self.ncas, (nkpts*self.nelecas[0], nkpts*self.nelecas[1]))
                    log.note('CASCI E (per k-point)= %#.15g  E(CI) = %#.15g  S^2 = %.7f',
                             self.e_tot.real, self.e_cas.real, ss[0])
                except NotImplementedError:
                    log.note('CASCI E (per k-point) = %#.15g  E(CI) = %#.15g',
                             self.e_tot.real, self.e_cas.real)
            else:
                for i, e in enumerate(self.e_cas):
                    try:
                        ss = self.fcisolver.spin_square(self.ci[i], nkpts*self.ncas, nkpts*self.nelecas)
                        log.note('CASCI E (per k-point) state %3d  E = %#.15g  E(CI) = %#.15g  S^2 = %.7f',
                                 i, self.e_tot[i].real, e.real, ss[0])
                    except NotImplementedError:
                        log.note('CASCI E (per k-point) state %3d  E = %#.15g  E(CI) = %#.15g',
                                 i, self.e_tot[i].real, e.real)

        else:
            if isinstance(self.e_cas, (np.complex128, np.float64)):
                log.note('CASCI E (per k-point)= %#.15g  E(CI) = %#.15g', self.e_tot, self.e_cas)
            else:
                for i, e in enumerate(self.e_cas):
                    log.note('CASCI E (per k-point) state %3d  E = %#.15g  E(CI) = %#.15g',
                             i, self.e_tot[i], e)
        return self

    def kernel(**kwargs):
        pass


class PBCCASCI(PBCCASBASE):
    '''
    Child class for the PBC CASCI
    '''

    def get_h2eff(self, mo_coeff=None):
        '''
        The mo_coeff are in k-space, while the eris returned by this function is in r-space.
        '''
        global _CDERI_CACHE_PATH

        from pyscf.pbc import scf
        kmf = self._scf
        ncore = self.ncore
        ncas = self.ncas
        nkpts = self.nkpts
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        
        scell, phase, mo_coeff_R, mo_phase = get_mo_coeff_k2R(kmf, mo_coeff, ncore, ncas)

        # Currently, I am relaying on generating one-more eris, because I didn't go through the
        # math to transform the eris from k-space to r-space.
        mf = scf.RHF(scell).density_fit(kmf.with_df.auxbasis)
        mf.exxdiv = kmf.exxdiv

        # During a cycle of CASCI, the get_h2eff is called multiple times, and building the cderi object is the most expensive step. So I am caching the cderi object to avoid rebuilding it multiple times. The integrals are stored on the disk.
        if _CDERI_CACHE_PATH is not None and \
            os.path.isfile(_CDERI_CACHE_PATH) and \
                  os.path.getsize(_CDERI_CACHE_PATH) > 0:
            mf.with_df._cderi = _CDERI_CACHE_PATH
        else:
            fd, cderi_path = tempfile.mkstemp(dir=lib.param.TMPDIR, prefix="pyscf_cderi_", suffix=".h5")
            os.close(fd)
            try:
                os.remove(cderi_path)
            except FileNotFoundError:
                pass
            mf.with_df._cderi_to_save = cderi_path
            mf.with_df.build()
            _CDERI_CACHE_PATH = cderi_path

        eri = mf.with_df.ao2mo(mo_coeff_R,)
        eris = eri.reshape(nkpts*ncas, nkpts*ncas, nkpts*ncas, nkpts*ncas)
        return eris
    

    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        '''
        args:
            mo_coeff:
            ci0:
            verbose:
        returns:
            e_tot:
            e_cas:
            ci:
            mo_coeff:
            mo_energy:
        '''
        if mo_coeff is None: 
            mo_coeff = self.mo_coeff
        
        # Set the mo_coeff
        self.mo_coeff = mo_coeff

        if ci0 is None:
            ci0 = self.ci
        
        log = logger.new_logger(self, verbose)
        
        self.check_sanity()
        self.dump_flags(log)

        self.e_tot, self.e_cas, self.ci = kernel(self, mo_coeff=mo_coeff, ci0=ci0, verbose=verbose)

        if self.canonicalization:
            pass
            #raise NotImplementedError
        
        if self.natorb:
            raise NotImplementedError

        # Check for convergence:
        if getattr(self.fcisolver, 'converged', None) is not None:
            self.converged = np.all(self.fcisolver.converged)
            if self.converged: log.info('CASCI converged')
            else: log.info('CASCI not converged')
        else: 
            self.converged = True

        self._finalize()

        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy
    
    def nuc_grad_method(self):
        raise NotImplementedError

    def analyze(self):
        #TODO:
        raise NotImplementedError

CASCI = PBCCASCI