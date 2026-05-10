# !/usr/bin/env python

import sys
import numpy as np
from functools import reduce

from pyscf import lib, mcscf, __config__
from pyscf.lib import logger
from pyscf.pbc import scf
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.x2c import sfx2c1e

from mrh.my_pyscf.pbc.fci.addons import _unpack_nelec
from mrh.my_pyscf.pbc import fci as pbc_fci
from mrh.my_pyscf.pbc.mcscf.k2R import  get_mo_coeff_k2R

WITH_META_LOWDIN = getattr(__config__, 'mcscf_analyze_with_meta_lowdin', True)
PENALTY = getattr(__config__, 'mcscf_casci_CASCI_fix_spin_shift', 0.2)

if sys.version_info < (3,):
    RANGE_TYPE = list
else:
    RANGE_TYPE = range

# 
# Generalization of the CASCI module with complex integrals for PBC systems.
# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>
#

#TODO:
# 1. Implement the CASNatorb function.
# 2.

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
        h1e_cas : np.ndarray [nk*ncas, nk*ncas]
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

    h1ao_k = mc.get_hcore().astype(dtype)

    # Remember, I am multiplying by nkpts here because total energy would be divided by nkpts later.
    ecore = mc.energy_nuc() * nkpts
    if len(mo_core_kpts) == 0:
        corevhf_kpts = 0
    else:
        coredm_kpts = np.asarray([2.0 * (mo_core_kpts[k] @ mo_core_kpts[k].conj().T) 
                                  for k in range(nkpts)], dtype=dtype)
        # corevhf_kpts = mc._scf.get_veff(cell, coredm_kpts, hermi=1)
        corevhf_kpts = mc.get_veff(cell, coredm_kpts, hermi=1, kpts=mc._scf.kpts)
        fock = h1ao_k + 0.5 * corevhf_kpts
        ecore += sum(np.einsum('ij,ji', coredm_kpts[k], fock[k]) for k in range(nkpts))
        fock = None  # Free memory

    h1ao_k += corevhf_kpts

    phase, mo_coeff_R = get_mo_coeff_k2R(mc._scf, mo_coeff, ncore, ncas, kmesh=mc.kmesh)[1:3]
    h1ao_R = np.einsum('Rk,kij,Sk->RiSj', phase, h1ao_k, phase.conj())
    h1ao_R = h1ao_R.reshape(nkpts*nao, nkpts*nao)

    # h1eff_R = _basis_transformation(h1ao_R, mo_coeff_R)
    h1eff_R = reduce(np.dot, (mo_coeff_R.conj().T, h1ao_R, mo_coeff_R))
    return h1eff_R, ecore

@lib.with_doc(mcscf.casci.get_fock.__doc__)
def get_fock(mc, mo_coeff=None, ci=None, eris=None, casdm1=None, verbose=None):
    '''
    Constructing the Generalized Fock matrix for a given casdm1.
    args:
        mc : pbc.mcscf.CASCI
            The CASCI object.
        mo_coeff : np.ndarray [nk, nao, nmo_k]
            orbitals at each k-point.
        ci : list or np.ndarray
            CI vector(s) representing the wavefunction in the CAS space.
        eris : np.ndarray
            Two-electron integrals in the CAS space. This is not used currently, 
            but I am keeping it for consistency with molecular CASCI.
        casdm1 : np.ndarray [nk*ncas, nk*ncas]
            The 1-particle density matrix in the CAS space, still in k-point basis.
        verbose : int
            Verbosity level for logging and output control.
    returns:
        fock : np.ndarray [nk, nao, nao]
            The generalized Fock matrix at each k-point in AO basis.
    '''
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
    kmf = mc._scf

    if casdm1 is None:
        casdm1 = mc.fcisolver.make_rdm1(ci, nkpts*ncas, (nkpts*nelecas[0], nkpts*nelecas[1]))
    
    dtype = casdm1.dtype

    mo_core_kpts = [mo[:, :ncore] for mo in mo_coeff]
    dm_core = np.asarray([2.0 * (mo_core_kpts[k] @ mo_core_kpts[k].conj().T) 
                                for k in range(nkpts)], dtype=dtype)
    
    hcore_k = mc.get_hcore()
    fock = np.empty_like(hcore_k, dtype=dtype)

    mo_phase = get_mo_coeff_k2R(kmf, mo_coeff, ncore, ncas, kmesh=mc.kmesh)[-1]
    
    dm_k = np.empty_like(dm_core)

    for k in range(nkpts):
        mocas = mo_coeff[k][:,ncore:nocc]
        dm = dm_core[k]
        dm += reduce(np.dot, (mocas, mo_phase[k], casdm1, mo_phase[k].conj().T @ mocas.conj().T))
        dm_k[k] = dm
    
    # veff = mc._scf.get_veff(cell, dm_k, hermi=1)
    veff = mc.get_veff(cell, dm_k, hermi=1, kpts=kmf.kpts)

    fock = np.array([hcore_k[k] + veff[k] for k in range(nkpts)], dtype=dtype)
    
    hcore_k = dm_core = mo_core_kpts = veff = None

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
    kmf = mc._scf

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

    mo_phase = get_mo_coeff_k2R(kmf, mo_coeff, ncore, ncas, kmesh=mc.kmesh)[-1]

    if cas_natorb:
        # Currently not implemented.
        mo_coeff = cas_natorb(mc, mo_coeff, casdm1, verbose=verbose)
    else:
        mo_coeff1 = mo_coeff.copy()
        log.info('Density matrix diagonal elements')
        for k in range(nkpts):
            dm_k = mo_phase[k] @ casdm1 @ mo_phase[k].conj().T
            log.info("k-point %d, only real diagonal = %s",
                     k,
                     np.array2string(np.diag(dm_k).real, precision=5, floatmode='fixed', separator=', '))
    mo_energy = [np.einsum('pi, pi-> i', mo_coeff1[k].conj(), fock_ao[k] @ mo_coeff1[k]) 
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
                    log.debug('i = %d  <i|F|i> = %12.8f', i+1, mo_energy[k][i].real)

    return mo_coeff1, ci, mo_energy

def kernel(mc, mo_coeff=None, ci0=None, verbose=logger.NOTE, envs=None):
    '''
    # Passing env to be consistent with molecular CASCI, but currently this is
    # not used.
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
    log.debug('core energy = %.15g', energy_core.real)
    max_memory = max(4000, mc.max_memory-lib.current_memory()[0])

    assert eri_cas.shape == (nkpts*ncas, nkpts*ncas, nkpts*ncas, nkpts*ncas)
    assert h1eff.shape == (nkpts*ncas, nkpts*ncas)

    if log.verbose >= logger.DEBUG1:
        assert np.linalg.norm(h1eff - h1eff.conj().T) < 1e-10,\
            "1e Hamiltonian hermiticity error"
        assert np.linalg.norm(eri_cas - eri_cas.transpose(2, 3, 0, 1)) < 1e-10,\
            "ERI permutation symmetry error"
        assert np.linalg.norm(eri_cas - eri_cas.conj().transpose(1, 0, 3, 2)) < 1e-10,\
            "ERI hermiticity error"

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
        'nkpts', 'scell', 'kmesh'
    }

    def __init__(self, kmf, ncas=0, nelecas=0, ncore=None):
        cell = kmf.cell
        self.cell = cell
        self.nkpts = len(kmf.kpts)
        self.kmesh = None
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
    
    # Removing this is causing problem.
    @ncore.setter
    def ncore(self, value):
        assert value is None or isinstance(value, (int, np.integer)) or value >=0
        self._ncore = value

    def dump_flags(self, verbose=None):
        if self.mo_coeff is None:
            log.error('MO coefficients are not set')
        mo_coeff_backup = self.mo_coeff.copy()
        self.mo_coeff = self.mo_coeff[0] # Because the dump_flags in molecular code only works for one set of mo_coeff
        mcscf.casci.CASBase.dump_flags(self, verbose)
        self.mo_coeff = mo_coeff_backup
        del mo_coeff_backup
        if (getattr(self._scf, 'with_solvent', None) and
            not getattr(self, 'with_solvent', None)): raise NotImplementedError
        log = logger.new_logger(self, verbose)
        log.info('nkpts = %d', self.nkpts)
        return self
    

    def _dump_flags(self, verbose=None):
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
        # This goes to parent CASCI and that will break the sanity check.
        # super().check_sanity()
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
    
    def get_veff(self, cell=None, dm_kpts=None, hermi=1, kpts=None, **kwargs):
        # Note this would be in k-space: would need transformation
        # before its direct use.
        vj,vk = self.get_jk(cell, dm_kpts, hermi, kpts, **kwargs)
        veff = vj - 0.5 * vk
        return veff
        #return self._scf.get_veff(cell=cell, dm_kpts=dm_kpts, hermi=hermi, kpts=kpts, **kwargs)
    
    def get_hcore(self, **kwargs):
        '''
        Wrapper to avoid printing a huge output for computing the hcore.
        '''
        with lib.temporary_env(self._scf, verbose=0):
            with lib.temporary_env(self._scf.cell, verbose=0):
                return self._scf.get_hcore(**kwargs)

    def get_h1cas(self, mo_coeff=None, ncas=None, ncore=None):
        '''An alias of get_h1eff method'''
        return self.get_h1eff(mo_coeff, ncas, ncore)

    get_h1eff = h1e_for_cas = h1e_for_cas

    @lib.with_doc(scf.hf.get_jk.__doc__)
    def get_jk(self, cell, dm_kpts, hermi=1, vhfopt=None, kpts=np.zeros(3),
           kpts_band=None, with_j=True, with_k=True, omega=None, **kwargs):
        '''
        Compute the J and K matrices for the given density matrix.
        Basically, this is wrapper around RHF function. See that function 
        for more details.
        '''
        vj, vk = self._scf.get_jk(cell=cell, dm_kpts=dm_kpts, hermi=hermi, vhfopt=vhfopt, 
                                kpts=kpts, kpts_band=kpts_band, with_j=with_j, 
                                with_k=with_k, omega=omega, **kwargs)
        assert vj.shape[0] == dm_kpts.shape[0]
        assert vk.shape[0] == dm_kpts.shape[0]
        # In case of ROHF mean-field, two J matrices are returned.
        if vj.ndim == 4 and vj.shape[1] == 2:
            vj = vj[:, 0] + vj[:, 1]
        return vj, vk
    
    canonicalize = canonicalize

    @lib.with_doc(canonicalize.__doc__)
    def canonicalize_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                      cas_natorb=False, casdm1=None, verbose=None,
                      with_meta_lowdin=WITH_META_LOWDIN):
        self.mo_coeff, ci, self.mo_energy = \
                canonicalize(self, mo_coeff, ci, eris,
                             sort, cas_natorb, casdm1, verbose, with_meta_lowdin)
        if cas_natorb:  # When active space is changed, the ci solution needs to be updated
            self.ci = ci
        return self.mo_coeff, ci, self.mo_energy
    
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
                        nelecastot = (nkpts*self.nelecas[0], nkpts*self.nelecas[1])
                        ss = self.fcisolver.spin_square(self.ci[i], nkpts*self.ncas, nelecastot)
                        log.note('CASCI E (per k-point) state %3d  E = %#.15g  E(CI) = %#.15g  S^2 = %.7f',
                                 i, self.e_tot[i].real, e.real, ss[0])
                    except NotImplementedError:
                        log.note('CASCI E (per k-point) state %3d  E = %#.15g  E(CI) = %#.15g',
                                 i, self.e_tot[i].real, e.real)

        else:
            if isinstance(self.e_cas, (np.complex128, np.float64)):
                log.note('CASCI E (per k-point)= %#.15g  E(CI) = %#.15g', self.e_tot.real, self.e_cas.real)
            else:
                for i, e in enumerate(self.e_cas):
                    log.note('CASCI E (per k-point) state %3d  E = %#.15g  E(CI) = %#.15g',
                             i, self.e_tot[i].real, e.real)
        return self

    # def kernel(**kwargs):
    #     pass
    
    get_fock = get_fock

    def make_rdm1s(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                   ncore=None, **kwargs):
        '''
        Spin-separated one-particle density matrices for alpha and beta spin on AO basis
        args:
            mo_coeff : np.ndarray [nk, nao, nmo_k]
                orbitals at each k-point.
            ci : list or np.ndarray
                CI vector(s) representing the wavefunction in the CAS space.
            ncas : int
                number of active space orbitals in unit cell (i.e. at each k-point).
            nelecas : int or tuple
                number of active electrons in unit cell (i.e. at each k-point).
            ncore : int
                number of core orbitals in unit cell (i.e. at each k-point).
        returns:
            dm1a : np.ndarray [nk, nao, nao]
                Alpha spin one-particle density matrix in AO basis for each k-point.
            dm1b : np.ndarray [nk, nao, nao]
                Beta spin one-particle density matrix in AO basis for each k-point.
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas is None: ncas = self.ncas
        if nelecas is None: nelecas = self.nelecas
        if ncore is None: ncore = self.ncore

        nkpts = self.nkpts
        kmesh = self.kmesh
        mo_phase = get_mo_coeff_k2R(self._scf, mo_coeff, ncore, ncas, kmesh=kmesh)[-1]
        nao = mo_coeff[0].shape[0]
        casdm1a, casdm1b = self.fcisolver.make_rdm1s(ci, nkpts*ncas, (nkpts*nelecas[0], nkpts*nelecas[1]))
        
        dm1a = np.empty((nkpts, nao, nao), dtype=casdm1a.dtype)
        dm1b = np.empty((nkpts, nao, nao), dtype=casdm1b.dtype)

        for k in range(nkpts):
            mocore = mo_coeff[k][:,:ncore]
            mocas = mo_coeff[k][:,ncore:ncore+ncas]
            dm1b[k] = np.dot(mocore, mocore.conj().T)
            # dm1a[k] = dm1b[k] + reduce(np.dot, (mocas, mo_phase[k], casdm1a, mo_phase[k].conj().T,mocas.conj().T))
            # dm1b[k] += reduce(np.dot, (mocas, mo_phase[k], casdm1b, mo_phase[k].conj().T,mocas.conj().T))
            umat = mocas @ mo_phase[k]
            dm1a[k] = dm1b[k] + reduce(np.dot, (umat, casdm1a, umat.conj().T))
            dm1b[k] += reduce(np.dot, (umat, casdm1b, umat.conj().T))
        
        casdm1a = casdm1b = None  # Free memory
        umat = None  # Free memory
        return dm1a, dm1b
    
    def make_rdm1(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                  ncore=None, **kwargs):
        '''
        Spin-summed one-particle density matrix in AO representation
        args:
            See above make_rdm1s function.
        returns:
            dm1 : np.ndarray [nk, nao, nao]
                Spin-summed one-particle density matrix in AO basis for each k-point.
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas is None: ncas = self.ncas
        if nelecas is None: nelecas = self.nelecas
        if ncore is None: ncore = self.ncore

        nkpts = self.nkpts
        nao = mo_coeff[0].shape[0]

        casdm1 = self.fcisolver.make_rdm1(ci, nkpts*ncas, (nkpts*nelecas[0], nkpts*nelecas[1]))
        mo_phase = get_mo_coeff_k2R(self._scf, mo_coeff, ncore, ncas, kmesh=self.kmesh)[-1]

        dm1 = np.empty((nkpts, nao, nao), dtype=casdm1.dtype)
        for k in range(nkpts):
            mocore = mo_coeff[k][:,:ncore]
            mocas = mo_coeff[k][:,ncore:ncore+ncas]
            dm1[k] = np.dot(mocore, mocore.conj().T) * 2
            # dm1[k] += reduce(np.dot, (mocas, mo_phase[k], casdm1, mo_phase[k].conj().T,mocas.conj().T))
            umat = mocas @ mo_phase[k]
            dm1[k] += reduce(np.dot, (umat, casdm1, umat.conj().T))
        
        casdm1 = mo_phase = umat = None  # Free memory

        return dm1

    @lib.with_doc(mcscf.casci.CASBase.fix_spin.__doc__)
    def fix_spin_(self, shift=PENALTY, ss=None):
        from mrh.my_pyscf.pbc import fci
        fci.addons.fix_spin_(self.fcisolver, shift, ss)
        return self
    
    fix_spin = fix_spin_
    
    # Pointing to the sfx2c1e object for the PBC system.
    sfx2c1e = sfx2c1e.sfx2c1e

class PBCCASCI(PBCCASBASE):
    '''
    Child class for the PBC CASCI
    '''

    def get_h2eff(self, mo_coeff=None):
        '''
        The mo_coeff are in k-space, while the eris returned by this function is in r-space.
        '''

        kmf = self._scf
        ncore = self.ncore
        ncas = self.ncas
        nkpts = self.nkpts
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        
        mo_phase = get_mo_coeff_k2R(kmf, mo_coeff, ncore, ncas, kmesh=self.kmesh)[-1]
        
        kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
        kpts = kmf.kpts

        # Do the ao2mo transformation in k-space and then transform the eris to r-space.
        # Basically, I am using the same eris integrals computed for the cell object.
        # Initial version:
        # eri_k = np.zeros((nkpts, nkpts, nkpts, ncas,ncas, ncas,ncas), dtype=np.complex128)
        # for kp in range(nkpts):
        #     for kq in range(nkpts):
        #         for kr in range(nkpts):
        #             ks = kconserv[kp, kq, kr]
        #             mo_tuple = [mo_coeff[i][:, ncore:ncore+ncas] for i in (kp, kq, kr, ks)]
        #             kp_tuple = [kpts[i] for i in (kp, kq, kr, ks)]
        #             eri_pqrs = kmf.with_df.ao2mo(mo_tuple, kp_tuple, compact=False)
        #             eri_k[kp, kq, kr] = eri_pqrs.reshape(ncas, ncas, ncas,ncas)
        
        # There is optimized version as well:
        mo_cas_kpts = np.array([mo_coeff[i][:, ncore:ncore+ncas] for i in range(nkpts)])
        eri_k = kmf.with_df.ao2mo_7d(mo_cas_kpts, kpts=kpts)
        
        kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
        
        mo_ks = mo_phase[kconserv]
        # This einsum looks very scary but it is just the transformation of the eris from k-space mo to r-space mo.
        eris = np.einsum('auR,bvS,abcuvwt,cwT,abctU->RSTU',
                         mo_phase.conj(), mo_phase, eri_k, mo_phase.conj(), mo_ks, optimize=True)
        eris *= 1.0/nkpts
        
        assert eris.shape == (nkpts*ncas, nkpts*ncas, nkpts*ncas, nkpts*ncas)
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
            self.canonicalize_(mo_coeff, self.ci,
                               sort=self.sorting_mo_energy,
                               cas_natorb=self.natorb, verbose=log)
            
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