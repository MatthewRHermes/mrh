#!/bin/bash

import unittest
import numpy as np
from functools import reduce
from pyscf import mcscf, ao2mo
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf
from pyscf.fci import direct_spin1
from pyscf.tools import molden
from pyscf.pbc.tools import k2gamma
from mrh.my_pyscf.pbc.fci import direct_com_real
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.k2R import get_mo_coeff_k2R


# While debugging the RDM construction I came across a lot of test
# 1. Number of electron = Tr(1-RDM)
# 2. Order of RDM matter   

def veff_core(kmf, cell, coredm):
    '''Speciall care has to be taken for the zero core density matrix.
    This is not a permanent fix.'''
    ovlp = kmf.get_ovlp()
    if np.isclose(np.trace(coredm @ ovlp).sum(), 0, atol=1e-5):
        return np.zeros_like(coredm, dtype=coredm.dtype)
    return kmf.get_veff(cell, coredm)

def get_xyz(nU=1, d= 2.47):
    coords = [
    ("C", -0.5892731038,  0.3262391909,  0.0000000000),
    ("H", -0.5866101958,  1.4126530287,  0.0000000000),
    ("C",  0.5916281105, -0.3261693897,  0.0000000000),
    ("H",  0.5889652025, -1.4125832275,  0.0000000000)]
    translated_coords = [(elem, x + t * d, y, z) 
                         for t in range(nU) 
                         for elem, x, y, z in coords]
    return translated_coords

def get_cell(nU=1, d=2.47, basis='6-31G', pseudo=None, maxMem=4000):
    cell = pgto.Cell(atom = get_xyz(nU, d),
                    a = np.diag([d*nU, 17.5, 17.5]),
                    basis = basis,
                    pseudo = pseudo,
                    precision = 1e-10,
                    verbose = 3,
                    # output = '/dev/null',
                    max_memory = maxMem,
                    ke_cutoff = 40)
    cell.build()
    return cell

def setUpModule():
    aux_basis='def2-svp-jkfit'
    kmesh=[1, 1, 1]

    global cell, kmf
    cell = get_cell(nU=2, d=2.47, basis='6-31G', pseudo=None, maxMem=4000)
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    kmf = scf.KRHF(cell, kpts).density_fit(auxbasis=aux_basis)
    kmf.exxdiv = None
    kmf.conv_tol = 1e-10
    kmf.kernel()
    
    e_kmf = kmf.e_tot

    # Define the active space
    ncas = cell.natm
    nelecas = (ncas//2, ncas//2)
    nao  = cell.nao_nr()
    ncore = int((sum(cell.nelec) - sum(nelecas)) // 2)

    mo_coeff = kmf.mo_coeff
    mo_core = [mo_coeff[k][:, :ncore] for k in range(len(kpts))]

    # Get the core Hamiltonian and the core density matrix
    hcore = kmf.get_hcore()
    coredm = np.asarray([2.0 * mo_core[k] @ mo_core[k].conj().T 
                         for k in range(len(kpts))], 
                         dtype=mo_coeff[0].dtype)
    veff = veff_core(kmf, cell, coredm)

    # Core energy
    fock = hcore + 0.5 * veff
    ecore = sum([np.einsum('ij,ji->', coredm[k], fock[k]) for k in range(len(kpts))])
    ecore += nkpts * kmf.energy_nuc() # Because at the end we will divide the FCI energy by nkpts

    fock = coredm = None
     
    # Now construct the h1, and h2 integrals in the active space and run complexFCI
    # This kernel is transforming the k-point MO coefficient to the R-space MO coefficient
    scell, phase, mo_coeff_R = get_mo_coeff_k2R(kmf, mo_coeff, ncore, ncas)[:3]
  
    h1eao = hcore + veff
    h1ao_R = np.einsum('Rk,kij,Sk->RiSj', phase, h1eao, phase.conj())
    h1ao_R = h1ao_R.reshape(nkpts * nao, nkpts * nao)

    # Now get the h1e in CAS MO space
    h1e = reduce(np.dot, (mo_coeff_R.conj().T,  h1ao_R, mo_coeff_R))
    h1e = np.asarray(h1e).reshape(nkpts*ncas, nkpts*ncas)

    # Now get the 2e integrals in CAS MO space, I am using the pyscf DF module here.
    # More appropriate would be directly use the previously generated k-space integrals.
    smf = scf.RHF(scell).density_fit(auxbasis=aux_basis)
    smf.exxdiv = None
    smf.with_df.auxbasis = aux_basis
    smf.with_df.build()
    eri = smf.with_df.ao2mo(mo_coeff_R, compact=False)
    h2e = eri.reshape(nkpts*ncas, nkpts*ncas, nkpts*ncas, nkpts*ncas)
   
    # The 2e integrals:
    h1e = 0.5*(h1e + h1e.T.conj())
    h2e = 0.5*(h2e + h2e.transpose(2, 3, 0, 1).conj())
    h2e = 0.5*(h2e + h2e.transpose(1, 0, 3, 2).conj())
    h2e = 0.5*(h2e + h2e.transpose(3, 2, 1, 0).conj())
    
    fcisolver = direct_com_real.FCI()
    e_fci, ci = fcisolver.kernel(h1e, h2e, nkpts*ncas, (nkpts*nelecas[0], nkpts*nelecas[1]))
    e_casci_ref = e_fci + ecore
    
    rdm1, rdm2 = fcisolver.make_rdm12_py(ci, nkpts*ncas, (nkpts*nelecas[0], nkpts*nelecas[1]))
    e0 = ecore
    e1 = np.einsum('ij,ji', h1e, rdm1)
    e2 = 0.5 * np.einsum('ijkl,ijkl', h2e, rdm2)

    # I will benchmark both the energy computations here.
    assert np.isclose(e_casci_ref, e0 + e1 + e2, atol=1e-10), \
        "RDM computations have some inconsistency."
    return (e_fci, e_casci_ref, e_kmf), (e0, e1, e2), mo_coeff_R

    

class KnownValues(unittest.TestCase):

    def test_singleref_rdm(self):
        pass

    def test_rdm_py_vs_c(self):
        pass

    def test_rdm_traces(self):
        pass

    def test_rdm_properties(self):
        pass


if __name__ == "__main__":
    print("Running Test for RDMs in complex FCI code...")
    unittest.main()
