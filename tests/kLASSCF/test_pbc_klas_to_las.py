"""
Check kLAS-to-LAS energy preservation for an H2 chain.
"""

import unittest

import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.util.klas_to_las import unpack_klas


class KnownValuesKLASToLASEnergies(unittest.TestCase):

    def check_mesh(self, nk):
        cell = gto.Cell()
        cell.a = np.diag([4.0, 10.0, 10.0])
        cell.atom = "H 0.0 0.0 0.0; H 1.5 0.0 0.0"
        cell.basis = "6-31G"
        cell.unit = "Angstrom"
        cell.precision = 1e-10
        cell.verbose = lib.logger.QUIET
        cell.build()

        kmesh = (nk, 1, 1)
        kmf = scf.KRHF(cell, kpts=cell.make_kpts(kmesh, wrap_around=True)).density_fit()
        kmf.exxdiv = None
        kmf.conv_tol = 1e-10
        kmf.kernel()
        self.assertTrue(kmf.converged)

        mo_avas = np.asarray(
            avas.kernel(kmf, ["H 1s"], minao=cell.basis)[2], dtype=complex,
        ).reshape(nk, cell.nao_nr(), -1)

        klas = mcscf.KLASSCF(kmf, 2, (1, 1), kmesh=kmesh)
        mo_guess = klas.localize_init_guess(["H 1s"], mo_coeff=mo_avas,
                                            stabilize_virtuals=True)
        klas.conv_tol_grad = 1e-6
        klas.max_cycle_macro = 100
        klas.kernel(mo_coeff=mo_guess,)
        self.assertTrue(klas.converged)

        mo, ham, ci, las = unpack_klas(klas)
        self.assertAlmostEqual(klas.e_tot.real, las.e_tot / nk, delta=1e-8)
        las.kernel(mo_coeff=mo, ci0=ci)
        
        self.assertTrue(las.converged)
        self.assertAlmostEqual(klas.e_tot.real, las.e_tot / nk, delta=1e-8)

    def test_one_kpoint_matches_synchronous_lasscf(self):
        self.check_mesh(1)

    def test_two_kpoints_match_synchronous_lasscf(self):
        self.check_mesh(2)

    def test_three_kpoints_match_synchronous_lasscf(self):
        self.check_mesh(3)

    def test_five_kpoints_match_synchronous_lasscf_slow(self):
        self.check_mesh(5)


if __name__ == '__main__':
    unittest.main()
