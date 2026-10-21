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

    def check_density(self, las):
        expected = las.get_nelec_frs()[:, 0]
        dm_frag = np.asarray(las.make_casdm1s_sub())
        np.testing.assert_allclose(np.trace(dm_frag, axis1=-2, axis2=-1), expected,
                                   atol=1e-8)
        ovlp = las._scf.get_ovlp()
        for include_core in (False, True):
            dm = las.make_rdm1s_sub(include_core=include_core)
            counts = np.einsum("kij,skji->sk", ovlp, dm).real
            target = expected.sum(axis=0) / las.nkpts
            target = target + (las.ncore if include_core else 0)
            np.testing.assert_allclose(counts, np.repeat(target[:, None], las.nkpts, axis=1),
                                       atol=1e-8)

    def test_reference_cell_sectors_and_translation(self):
        for charge in (1, -1):
            energies = []
            for ref in (0, 2):
                with self.subTest(charge=charge, ref=ref):
                    las = self.make_las(charge=charge, ref_cell=ref)
                    expected = np.ones((3, 2), dtype=int)
                    expected[ref] = (1, 0) if charge == 1 else (2, 1)
                    np.testing.assert_array_equal(las.nelecas_sub, expected)
                    self.assertEqual(las.nelecas, (1, 1))
                    self.assertEqual(las.ncore, 0)
                    self.assertEqual(las.nelecas_sub.sum(), 6 - charge)
                    self.assertEqual(las.ref_cell, ref)
                    self.assertEqual(las.charged_spin_mult, 2)
                    before = las.mo_coeff.copy()
                    las.kernel()
                    self.assertTrue(las.converged)
                    np.testing.assert_array_equal(las.mo_coeff, before)
                    self.check_density(las)
                    energies.append(las.e_tot)
            np.testing.assert_allclose(energies[0], energies[1], atol=1e-8)

    def test_gamma_matches_charged_kcasci(self):
        kmf, mo = self.systems[1]
        for charge in (1, -1):
            with self.subTest(charge=charge):
                las = self.make_las(nk=1, charge=charge)
                las.kernel()
                cas = mcscf.KCASCI(kmf, 2, (1, 1), charge=charge, target_k=0)
                cas.kmesh = (1, 1, 1)
                cas.fcisolver.conv_tol = 1e-11
                cas.kernel(mo)
                self.assertTrue(las.converged and cas.converged)
                np.testing.assert_allclose(las.e_tot, cas.e_tot, atol=1e-8)
                self.check_density(las)

    def test_core_count_is_kept(self):
        kmf, mo = self.systems[1]
        # Neutral H2 has a doubly occupied core and an empty active space.
        las = mcscf.KLASCI(kmf, 2, (0, 0), kmesh=(1, 1, 1), charge=-1)
        self.assertEqual(las.ncore, 1)
        self.assertEqual(las.nelecas, (0, 0))
        las.kernel(mo)
        self.check_density(las)
        dm = las.make_rdm1s().sum(axis=0)
        self.assertAlmostEqual(np.einsum("kij,kji->", kmf.get_ovlp(), dm).real, 3, places=8)

    def test_explicit_spin_and_multiplicity(self):
        las = self.make_las(charge=1, charged_spin=-1)
        np.testing.assert_array_equal(las.nelecas_sub[0], (0, 1))
        las.kernel()
        self.check_density(las)
        positive = self.make_las(charge=1, charged_spin=1)
        positive.kernel()
        np.testing.assert_allclose(las.e_tot, positive.e_tot, atol=1e-8)

        kmf, _ = self.systems[3]
        high_spin = mcscf.KLASCI(kmf, 3, (1, 1), kmesh=(3, 1, 1), charge=-1,
                                 charged_spin=1, charged_spin_mult=4)
        self.assertEqual(high_spin.fciboxes[0].fcisolvers[0].smult, 4)
        self.assertEqual(high_spin.fciboxes[1].fcisolvers[0].smult, 1)

    def test_setters_reinitialize_sector(self):
        las = self.make_las(charge=1, ref_cell=0)
        las.ref_cell = 2
        expected = np.ones((3, 2), dtype=int)
        expected[2] = (1, 0)
        np.testing.assert_array_equal(las.nelecas_sub, expected)
        self.assertEqual(las.ref_cell, 2)
        self.assertEqual((las.charged_spin, las.charged_spin_mult), (1, 2))
        # Flip the spin projection of the charged cell
        las.charged_spin = -1
        np.testing.assert_array_equal(las.nelecas_sub[2], (0, 1))
        self.assertEqual(las.charged_spin_mult, 2)
        las.charged_spin_mult = 2
        self.assertEqual(las.charged_spin_mult, 2)
        # A rejected edit leaves the object untouched.
        with self.assertRaises(ValueError):
            las.ref_cell = 7
        with self.assertRaises(ValueError):
            las.charged_spin_mult = 4
        self.assertEqual(las.ref_cell, 2)
        np.testing.assert_array_equal(las.nelecas_sub[2], (0, 1))
        self.assertEqual(las.nelecas_sub.sum(), 5)
        # Neutral objects cannot be edited this way.
        neutral = self.make_las()
        self.assertIsNone(neutral.charged_spin)
        for name in ("ref_cell", "charged_spin", "charged_spin_mult"):
            with self.assertRaises(ValueError):
                setattr(neutral, name, 1)

    def test_setter_matches_constructor(self):
        las = self.make_las(charge=1, ref_cell=0)
        las.ref_cell = 2
        las.kernel()
        self.assertTrue(las.converged)
        ref = self.make_las(charge=1, ref_cell=2)
        ref.kernel()
        np.testing.assert_allclose(las.e_tot, ref.e_tot, atol=1e-8)
        las.ref_cell = 1
        self.assertIsNone(las.ci)
        self.assertFalse(las.converged)

    def test_neutral_defaults(self):
        neutral = self.make_las()
        zero = self.make_las(charge=0)
        neutral.kernel()
        zero.kernel()
        np.testing.assert_allclose(neutral.e_tot, zero.e_tot, atol=1e-10)
        self.check_density(neutral)

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
