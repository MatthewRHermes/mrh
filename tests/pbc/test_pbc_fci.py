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


# Okay, let us start the benchmarking the complex FCI code:
# TODO: I think I should replace the test molecule with H-Chain or something like that. But
# for now let us keep it simple.

'''
# Test-0: Construct the random h1, h2 and use that to compute the FCI energy using real FCI code 
#         and complex FCI code. I am expecting for the smaller imag part both the energies should match.
# Test-1: For gamma point calculation (with complex integrals) the results should match with real FCI code.
# Test-2: For more than one k-point, I will compare the results to the HF limit and CCSD limit.
'''

def veff_core(kmf, cell, coredm):
    '''Speciall care has to be taken for the zero core density matrix.
    This is not a permanent fix.'''
    ovlp = kmf.get_ovlp()
    if np.isclose(np.trace(coredm @ ovlp).sum(), 0, atol=1e-5):
        return np.zeros_like(coredm, dtype=coredm.dtype)
    return kmf.get_veff(cell, coredm)

def get_xyz(nU=1, dintra= 1.3, dinter=1.4):
    unit = [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, dintra)]
    repeat = 2.0 * dintra + dinter
    translated = [(elem, x, y, z + t * repeat) 
                  for t in range(nU) 
                  for elem, x, y, z in unit]
    return translated

def get_cell(nU=1, dintra=1.3, dinter=1.4, basis='STO-6G', pseudo=None, maxMem=4000, verbose=4):
    cell = pgto.Cell(atom = get_xyz(nU, dintra, dinter),
                    a = np.diag([17.5, 17.5, (dinter+dintra)*nU]),
                    basis = basis,
                    pseudo = pseudo,
                    precision = 1e-10,
                    verbose = verbose,
                    # output = '/dev/null',
                    max_memory = maxMem,
                    ke_cutoff = 40)
    cell.build()
    return cell

def run_complex_fci(cell, kmesh=[1, 1, 1], aux_basis='def2-svp-jkfit'):    
    # Here, I am running the KRHF with GDF, and then transforming the integrals to 
    # real-space (complex integrals, not necessarily at the gamma point) for the active space.
    # Then I will run the complex FCI code on top of that.

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

def run_real_fci(cell, kmesh=[1, 1, 1], aux_basis='def2-svp-jkfit'):
    # Get the supercell at gamma point
    kpts = cell.make_kpts(kmesh)
    scell = k2gamma.get_phase(cell, kpts)[0]

    mf = scf.RHF(scell).density_fit(auxbasis=aux_basis)
    mf.conv_tol = 1e-10
    mf.exxdiv = None
    mf.kernel()

    e_mf = mf.e_tot
    # For H-Chain in minimal basis, I don't need to worry about choosing the active space.
    mo_coeff = mf.mo_coeff

    # Construct the h1, and h2 integrals in the active space and run FCI
    ncas = scell.natm
    nelecas = (ncas//2, ncas//2)
    ncore = int((sum(scell.nelec) - sum(nelecas)) // 2)
    mo_cas = mo_coeff[:, ncore:ncore+ncas]
    mo_core = mo_coeff[:, :ncore]

    hcore = mf.get_hcore()
    coredm = 2.0 * (mo_core @ mo_core.conj().T)
    veff = mf.get_veff(scell, coredm)

    # Core energy
    fock = hcore + 0.5 * veff
    ecore = np.einsum('ij,ji->', coredm, fock)
    ecore += mf.energy_nuc()

    fock = coredm = None

    # Now get the h1e in CAS MO space
    h1eao = hcore + veff
    h1e = reduce(np.dot, (mo_cas.conj().T,  h1eao, mo_cas))

    # The 2e integrals:
    eri = mf.with_df.ao2mo(mo_cas, compact=False)
    h2e = ao2mo.restore(1, eri, ncas)
    
    fcisolver = direct_spin1.FCI()
    e_fci = fcisolver.kernel(h1e, h2e, ncas, nelecas)[0]
    e_casci_ref = e_fci + ecore

    # Also run the CASCI calculation for comparison
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver = direct_spin1.FCI()
    mc.kernel(mo_coeff)

    e_casci = mc.e_tot

    rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelecas)
    e0 = ecore
    e1 = np.einsum('ij,ji', h1e, rdm1)
    e2 = 0.5 * np.einsum('ijkl,ijkl', h2e, rdm2)

    # This is redundant, Still I will benchmark both the energy computations here.
    assert np.isclose(e_casci, e0 + e1 + e2, atol=1e-10), \
        "There is some inconsistency in RDM computations."

    assert np.isclose(e_casci, e_casci_ref, atol=1e-10), \
        "CASCI energies from FCI solver and MC solver do not match."
    
    return (e_fci, e_casci, e_mf), (e0, e1, e2), mo_coeff[:, mc.ncore:mc.ncore+ncas], scell

def compare_act_space(mo_coeff_R, ovlp, mo_coeff):
    s = reduce(np.dot, (mo_coeff_R.conj().T, ovlp, mo_coeff_R))
    max_dev = np.max(np.abs(s - np.eye(s.shape[0])))
    assert max_dev < 1e-7, "Active space is not consistent between complex and real FCI codes."

class KnownValues(unittest.TestCase):

    def test_cfci_vanilla(self):
        np.random.seed(12)
        norb = 6
        h1 = np.random.random((norb,norb))
        h2 = np.random.random((norb,norb,norb,norb))

        h1 = np.asarray(h1).astype(np.complex128)
        h2 = np.asarray(h2).astype(np.complex128)

        h1.imag += 1e-6
        h2.imag += 1e-6

        # Restore symmetries
        h1 = h1 + h1.conj().T
        h2 = h2 + h2.conj().transpose(1,0,2,3)
        h2 = h2 + h2.conj().transpose(0,1,3,2)
        h2 = h2 + h2.conj().transpose(2,3,0,1)

        # FCI from PySCF  
        cisolver = direct_spin1.FCI()
        e = cisolver.kernel(h1.real, h2.real, norb, 8) [0]
        
        # FCI from complex_com_real
        cisolver = direct_com_real.FCI()
        e_com = cisolver.kernel(h1, h2, norb, 8)[0]
        
        self.assertAlmostEqual(e, e_com, places=8, msg="FCI energies do not match between vanilla and complex FCI codes.")
    
    def test_cfci_fci(self):
        nU = 1
        dinter = 1.3
        dintra = 1.5
        basis = 'STO-6G'
        verbose = 0
        pseudo = None
        maxMem = 4000
        kmesh = [1, 1, 1]

        cell = get_cell(nU, dintra, dinter, basis, pseudo, maxMem, verbose)
        (e_fci_c, e_casci_ref_c, e_kmf_c), (e0_c, e1_c, e2_c), mo_coeff_R = run_complex_fci(cell, kmesh)
        (e_fci, e_casci, e_mf), (e0, e1, e2), mo_coeff, scell = run_real_fci(cell, kmesh)

        compare_act_space(mo_coeff_R, scell.pbc_intor('int1e_ovlp_sph'), mo_coeff)
        
        self.assertAlmostEqual(e_mf, e_kmf_c, places=10, msg="HF energies do not match.")
        self.assertAlmostEqual(e_fci, e_fci_c, places=10, msg="FCI energies do not match.")
        self.assertAlmostEqual(e_casci, e_casci_ref_c, places=10, msg="CASCI energies do not match.")
        self.assertAlmostEqual(e0, e0_c, places=10, msg="E0 energies do not match.")
        self.assertAlmostEqual(e1, e1_c, places=10, msg="E1 energies do not match.")
        self.assertAlmostEqual(e2, e2_c, places=10, msg="E2 energies do not match.")

if __name__ == "__main__":
    unittest.main()