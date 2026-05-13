#!/usr/bin/env python
import unittest
import numpy as np
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc.mcscf import avas

# Unit test for the p-AVAS. This is generalization of AVAS at k-points with appropriate care
# of complex numbers and cell object. Instead of testing the AVAS code in-depth (which should 
# be done for molecular code) I will be testing the my generalization.

# Description of the tests: I will select the active space using the p-AVAS code, then using those
# mo_coeff, I will run the kmf again, the computed energies in the rotated basis should match.

# Test-0: Gamma point periodi mean-field using scf.RHF (It stored mo_coeff like molecular code)
# Test-1: Gamma-point periodic mean-field calculation using scf.KRHF (it stores mo_coeff like kmf)
# Test-2: k-point periodic mean-field calculation, additionally I have also compared the energy 
# at each k-point, which should also match.
# TODO: Check with Pseudo potentials, which will be more relevant for real applications.

def get_xyz(nU=1, dintra= 1.3, dinter=1.4):
    unit = [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, dintra)]
    repeat = 2.0 * dintra + dinter
    translated = [(elem, x, y, z + t * repeat) 
                  for t in range(nU) 
                  for elem, x, y, z in unit]
    return translated

def get_cell(nU=1, dintra=1.3, dinter=1.4, basis='6-31G', pseudo=None, 
             maxMem=4000, verbose=0):
    cell = pgto.Cell(atom = get_xyz(nU, dintra, dinter),
                    a = np.diag([17.5, 17.5, (dinter+dintra)*nU]),
                    basis = basis,
                    pseudo = pseudo,
                    precision = 1e-10,
                    verbose = verbose,
                    max_memory = maxMem,
                    ke_cutoff = 40)
    cell.build()
    return cell

def run_kmf(cell, kmesh=[1, 1, 1]):
    kpts = cell.make_kpts(kmesh)
    kmf = scf.KRHF (cell, kpts=kpts).density_fit()
    kmf.exxdiv = None
    kmf.conv_tol = 1e-10
    kmf.kernel()
    return kmf

def run_mf(cell):
    mf = scf.RHF(cell).density_fit()
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf

class KnownValues(unittest.TestCase):

    def test_kmf_gamma0(self):
        cell = get_cell(nU=1)
        mf = run_mf(cell)
        e_mf = mf.e_tot
        ncore = cell.nelectron // 2
        mo_coeff = avas.kernel(mf, ['H 1s'], minao=cell.basis)[2]
        dm = 2.0*mo_coeff[:, :ncore] @ mo_coeff[:, :ncore].conj().T
        e_mf_avas = mf.energy_tot(dm=dm)
        self.assertAlmostEqual(e_mf, e_mf_avas, places=7)

    def test_kmf_gamma1(self):
        cell = get_cell(nU=1)
        kmf = run_kmf(cell, kmesh=[1, 1, 1])
        e_kmf = kmf.e_tot
        ncore = cell.nelectron // 2
        mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]
        dm = [2.0*mo_coeff[:, :ncore] @ mo_coeff[:, :ncore].conj().T]
        e_mf_avas = kmf.energy_tot(dm=dm)
        self.assertAlmostEqual(e_kmf, e_mf_avas, places=7)

    def test_kmf_kpoints(self):
        cell = get_cell(nU=1)
        kmesh = [1, 1, 3]
        kmf = run_kmf(cell, kmesh=kmesh)
        nkpts = np.prod(kmesh)
        e_kmf = kmf.e_tot

        # Also computing the energy at each k-point:
        dm_kpts = kmf.make_rdm1()
        hcore = kmf.get_hcore()
        veff = kmf.get_veff(dm_kpts)
        e_kmf_kpts = [np.einsum('ij,ji->', hcore[k] + 0.5*veff[k], dm_kpts[k]) 
                      for k in range(nkpts)]

        mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]
        ncore = cell.nelectron // 2
        dm_kpts_avas = [2.0 * mo_coeff[k][:, :ncore] @ mo_coeff[k][:, :ncore].conj().T 
                        for k in range(nkpts)]
        veff_rotated = kmf.get_veff(dm_kpts_avas)
        e_kmf_kpts_avas = [np.einsum('ij,ji->', hcore[k] + 0.5*veff_rotated[k], dm_kpts_avas[k]) 
                              for k in range(nkpts)]
        e_mf_avas = kmf.energy_tot(dm=dm_kpts_avas)

        e_kmf_kpts_avas = np.array(e_kmf_kpts_avas)
        e_kmf_kpts = np.array(e_kmf_kpts)
        
        self.assertAlmostEqual(e_kmf, e_mf_avas, places=7)
        self.assertTrue(np.allclose(e_kmf_kpts, e_kmf_kpts_avas, atol=1e-7))

if __name__ == "__main__":
    unittest.main()