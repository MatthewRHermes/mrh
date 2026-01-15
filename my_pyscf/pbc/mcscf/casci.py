import sys
import warnings
import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto, scf, ao2mo, fci
from pyscf.mcscf import addons
from pyscf import __config__

#
# Authou: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>
#

def _basis_transformation(operator, mo):
    return reduce(np.dot, (mo.conj().T, operator, mo))

def h1e_for_cas(casci, mo_coeff=None, ncas=None, ncore=None):
    '''
    Compute the 1e Hamiltonian for CAS space and core energy.
    Args:
        casci : pbc.mcscf.CASCI
            The CASCI object.
        mo_coeff : np.ndarray [nk, nao, nmo]
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
        mo_coeff = casci.mo_coeff
    if ncas is None:
        ncas = casci.ncas
    if ncore is None:
        ncore = casci.ncore
    
    mo_core_kpts = [mo[:, :ncore] for mo in mo_coeff]
    mo_cas_kpts = [mo[:, ncore:ncore+ncas] for mo in mo_coeff]

    h1e_kpts = casci.get_hcore()
    ecore = casci.energy_nuc()

    if len(mo_core_kpts) == 0:
        corevhf_kpts = 0
    else:
        coredm_kpts = 2 * np.dot(mo_core_kpts, mo_core_kpts.conj().T)
        corevhf_kpts = casci.get_veff(casci.cell, coredm_kpts, hermi=1)
        ecore += np.einsum('ij,ji', h1e_kpts, coredm_kpts)
        ecore += 0.5 * np.einsum('ij,ji', corevhf_kpts, coredm_kpts)

    h1e_kpts += corevhf_kpts
    h1eff = [_basis_transformation(h1e_kpts[k], mo_cas_kpts[k])[ncore:ncore+ncas, ncore:ncore+ncas]
             for k in range(len(mo_cas_kpts))]
    return h1eff, ecore



