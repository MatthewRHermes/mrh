"""Localized charged LAS sectors, densities, and the one-cell CASCI limit."""

import unittest

import numpy as np
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf


class ChargedKLASCITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.a = np.diag([3.0, 10.0, 10.0])
        cell.atom = "H 0 0 0; H 0.74 0 0"
        cell.basis = "6-31g"
        cell.precision = 1e-8
        cell.ke_cutoff = 20
        cell.verbose = 0
        cell.build()
        cls.systems = {}
        for nk in (1, 3):
            mesh = (nk, 1, 1)
            kmf = scf.KRHF(cell, cell.make_kpts(mesh, wrap_around=True)).density_fit()
            kmf.exxdiv = None
            kmf.max_cycle = 0
            kmf.kernel()
            neutral = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=mesh)
            mo = neutral.localize_init_guess(["H 1s"], mo_coeff=np.asarray(kmf.mo_coeff))
            cls.systems[nk] = (kmf, mo)

    def make_las(self, nk=3, **kwargs):
        kmf, mo = self.systems[nk]
        las = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=(nk, 1, 1), **kwargs)
        las.mo_coeff = mo.copy()
        las.conv_tol_grad = 1e-7
        las.conv_tol_self = 1e-12
        for box in las.fciboxes:
            box.fcisolvers[0].conv_tol = 1e-12
        return las

    def test_invalid_options(self):
        cases = [
            ({"charge": 2}, "charge must"),
            ({"charge": 0.5}, "charge must"),
            ({"charge": 1, "ref_cell": -1}, "ref_cell must be in"),
            ({"charge": 1, "ref_cell": 3}, "ref_cell must be in"),
            ({"charge": 1, "trans_sym": True}, "trans_sym=False"),
            ({"charge": 1, "charged_spin": 0}, "parity"),
            ({"charge": 1, "charged_spin": 3}, "invalid active electrons"),
            ({"charge": 1, "charged_spin": 1.0}, "charged_spin must"),
            ({"charge": 1, "charged_spin_mult": 1}, "charged_spin_mult"),
            ({"charge": 1, "charged_spin_mult": 4}, "charged_spin_mult"),
            ({"charged_spin": 1}, "require charge"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                self.make_las(**kwargs)
        kmf, _ = self.systems[3]
        for ncas, nelecas, charge in ((1, 2, -1), (2, 0, 1)):
            with self.assertRaisesRegex(ValueError, "invalid active electrons"):
                mcscf.KLASCI(kmf, ncas, nelecas, kmesh=(3, 1, 1), charge=charge)


if __name__ == "__main__":
    unittest.main()
