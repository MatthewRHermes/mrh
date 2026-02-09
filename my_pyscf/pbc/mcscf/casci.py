from ast import Sub
from fileinput import filename

import os
import tempfile
import sys
import warnings
import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto, scf, ao2mo, fci
from pyscf.mcscf import addons
from pyscf import __config__
from pyscf import mcscf
from mrh.my_pyscf.pbc.fci import direct_com_real
from mrh.my_pyscf.pbc.mcscf.k2R import  get_mo_coeff_k2R

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

def h1e_for_cas(casci, mo_coeff=None, ncas=None, ncore=None):
    '''
    Compute the 1e Hamiltonian for CAS space and core energy.
    Args:
        casci : pbc.mcscf.CASCI
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
    
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    if ncas is None: ncas = casci.ncas
    if ncore is None: ncore = casci.ncore
    
    cell = casci.cell
    scell = casci.scell
    kmf = casci.kmf
    nao = cell.nao_nr()

    dtype = casci.mo_coeff[0].dtype
    nkpts = len(casci.kpts)

    mo_core_kpts = [mo[:, :ncore] for mo in mo_coeff]

    h1ao_k = casci.get_hcore()

    # Remember, I am multiplying by nkpts here because total energy would be divided by nkpts later.
    ecore = casci.energy_nuc() * nkpts 

    if len(mo_core_kpts) == 0:
        corevhf_kpts = 0
    else:
        coredm_kpts = np.asarray([2.0 * (mo_core_kpts[k] @ mo_core_kpts[k].conj().T) 
                                  for k in range(nkpts)], dtype=dtype)
        corevhf_kpts = casci.get_veff(cell, coredm_kpts, hermi=1)
        ecore += np.einsum('ij,ji', h1ao_k, coredm_kpts)
        ecore += 0.5 * np.einsum('ij,ji', corevhf_kpts, coredm_kpts)

    h1ao_k += corevhf_kpts

    phase, mo_coeff_R = get_mo_coeff_k2R(casci, mo_coeff, ncore, ncas)[1:3]

    h1ao_R = np.einsum('Rk,kij,Sk->RiSj', phase, h1ao_k, phase.conj())
    h1ao_R = h1ao_R.reshape(nkpts*nao, nkpts*nao)
    h1eff_R = _basis_transformation(h1ao_R, mo_coeff_R)

    return h1eff_R, ecore


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
            nelecb = (nelecas-kmf.spin)//2
            neleca = nelecas - nelecb
            self.nelecas = (neleca, nelecb)
        else:
            self.nelecas = (nelecas[0],nelecas[1])
        self.ncore = ncore
        singlet = (getattr(__config__, 'mcscf_casci_CASCI_fcisolver_direct_spin0', False)
                   and self.nelecas[0] == self.nelecas[1])  # leads to direct_spin1
        self.fcisolver = direct_com_real(cell, singlet, symm=False)
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
    
    def get_veff(self, cell=None, dm=None, hermi=1):
        # Note this would be in k-space: would need transformation
        # before its direct use.
        if cell is None: cell = self.cell
        if dm is None:
            ncore = self.ncore
            nkpts = self.nkpts
            mo_core = [self.mo_coeff[k][:, :ncore] for k in range(nkpts)]
            dm = [mo_core[k] @ mo_core[k].conj().T for k in range(nkpts)]
        
        vj, vk = self.get_jk(cell, dm, hermi=hermi)
        return vj, vk
    
    # Defining all of these functions here to initialize it. Do we really need it?
    def _eig(**kwargs):
        pass

    def get_h2cas(**kwargs):
        pass

    def get_h2eff(**kwargs):
        pass
    
    def ao2mo(**kwargs):
        pass
    
    def get_h1cas(**kwargs):
        pass

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
        tmpfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        mf.with_df._cderi = tmpfile
        if not os.path.getsize(tmpfile.name):
            mf.with_df._cderi_to_save = tmpfile
            mf.with_df.build()

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
            raise NotImplementedError
        
        if self.natorb:
            raise NotImplementedError

        if getattr(self.fcisolver, 'converged', None) is not None:
            self.converged = np.all(self.fcisolver.converged)
            if self.converged: log.info('CASCI converged')
            else: log.info('CASCI not converged')
        else: self.converged = True

        self._finalize()

        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy
    
    def nuc_grad_method(self):
        raise NotImplementedError
