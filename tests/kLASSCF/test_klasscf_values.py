#!/usr/bin/env python

"""Check H2 k-LASSCF energies for 1, 2, and 3 k-points with 20 Angstrom vacuum.
Compare Gamma with PySCF CASCI/CASSCF and other meshes with fixed references.
The three-point reference includes the complete active rotation map, including
k-dependent imaginary diagonal generators; its value was also checked against
synchronous LASSCF using the same periodic Hamiltonian in supercell coordinates.
"""

import unittest

import numpy as np
from pyscf import lib, mcscf as gamma_mcscf
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas


class KnownValuesH2KLASSCF(unittest.TestCase):

    def _run_h2(self, nk):
        # Finite-vacuum chain: periodic sampling only along x; energies per cell.
        cell = gto.Cell()
        cell.a = np.diag([4.0, 20.0, 20.0])
        cell.atom = "H 0 0 0; H 1.5 0 0"
        cell.basis = "6-31g"
        cell.unit = "Angstrom"
        cell.precision = 1e-10
        cell.ke_cutoff = 20
        cell.verbose = lib.logger.QUIET
        cell.build()

        kmesh = (nk, 1, 1)
        kpts = cell.make_kpts(kmesh, wrap_around=True)
        if nk == 1:
            mf = scf.RHF(cell).density_fit()
        else:
            mf = scf.KRHF(cell, kpts=kpts).density_fit()
        mf.exxdiv = None
        mf.conv_tol = 1e-11
        mf.kernel()
        self.assertTrue(mf.converged)

        if nk == 1:
            kmf = scf.addons.convert_to_kscf(mf)
            kmf.mo_coeff = mf.mo_coeff[None, :, :]
        else:
            kmf = mf
        mo_coeff = avas.kernel(kmf, ["H 1s"], minao=cell.basis)[2]
        if nk == 1:
            # AVAS squeezes the Gamma axis; k-LASSCF requires (nk, nao, nmo).
            mo_coeff = mo_coeff[None, :, :]
        # The periodic integral transformation uses complex orbital buffers.
        mo_coeff = np.asarray(mo_coeff, dtype=np.complex128)
        las = mcscf.KLASSCF(kmf, 2, (1, 1), kmesh=kmesh)
        mo_guess = las.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff, stabilize_virtuals=True,
        )
        las.conv_tol_grad = 1e-6
        las.max_cycle_macro = 100
        # Start each cell in the same doubly occupied singlet determinant.
        ci0 = [[np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)]
               for _ in range(las.nfrags)]
        las.kernel(mo_coeff=mo_guess, ci0=ci0)
        self.assertTrue(las.converged)

        ugg = las.get_ugg(mo_coeff=las.mo_coeff, ci=las.ci)
        gradient = las.get_grad(mo_coeff=las.mo_coeff, ci=las.ci, ugg=ugg)
        self.assertLess(np.linalg.norm(gradient[:ugg.nvar_orb]), las.conv_tol_grad)
        self.assertLess(np.linalg.norm(gradient[ugg.nvar_orb:]), las.conv_tol_grad)
        return las, mf, mo_guess

    def test_one_kpoint_matches_cas_energies(self):
        las, mf, mo_guess = self._run_h2(1)

        casci = gamma_mcscf.CASCI(mf, 2, (1, 1)).density_fit(with_df=mf.with_df)
        casci.fcisolver.conv_tol = 1e-12
        casci.kernel(mo_coeff=las.mo_coeff[0].real)
        self.assertAlmostEqual(las.e_tot.real, casci.e_tot.real, delta=1e-8)

        casscf = gamma_mcscf.CASSCF(mf, 2, (1, 1)).density_fit(with_df=mf.with_df)
        casscf.conv_tol = 1e-10
        casscf.conv_tol_grad = 1e-7
        casscf.fcisolver.conv_tol = 1e-12
        casscf.kernel(mo_coeff=mo_guess[0].real)
        self.assertTrue(casscf.converged)
        self.assertAlmostEqual(las.e_tot.real, casscf.e_tot.real, delta=1e-7)

    def test_two_kpoint_reference_energy(self):
        las, _, _ = self._run_h2(2)
        self.assertAlmostEqual(las.e_tot.real, -0.985036667832758, delta=1e-7)

    def test_three_kpoint_reference_energy(self):
        las, _, _ = self._run_h2(3)
        self.assertAlmostEqual(las.e_tot.real, -0.972509694148662, delta=1e-7)


if __name__ == "__main__":
    unittest.main()
