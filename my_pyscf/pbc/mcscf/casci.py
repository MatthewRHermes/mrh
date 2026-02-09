import sys
import warnings
import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto, scf, ao2mo, fci
from pyscf.mcscf import addons
from pyscf import __config__
from mrh.my_pyscf.pbc.mcscf.k2R import  get_mo_coeff_k2R

# 
# Generalization of the CASCI module with complex integrals for PBC systems.
# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>
#

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



