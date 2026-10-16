#!/usr/bin/env python

"""Compare the optimized Wannier-basis active ERIs against the original (slow) one."""

import unittest
import numpy as np

from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.klasci import kLASCI, h2e_for_cas, h2e_for_cas_slow

# Author: Bhavnesh Jangid

# Test-0: h2e_for_cas must match h2e_for_cas_slow for the SCF orbitals of AVAS.
# Test-1: Same, when the RAM pair cache is forced to spill to HDF5 (max_memory=0).
# Test-2: Same, for orbitals with an arbitrary k-dependent unitary in the active space.


def _build(kmesh):
    cell = gto.Cell()
    cell.a = np.diag([3.0, 10.0, 10.0])
    cell.atom = "H 0 0 0; H 0.74 0 0"
    cell.basis = "6-31g"
    cell.unit = "Angstrom"
    cell.precision = 1e-8
    cell.ke_cutoff = 20
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts(kmesh, wrap_around=True)
    kmf = scf.KRHF(cell, kpts=kpts).density_fit()
    kmf.exxdiv = None
    kmf.max_cycle = 0
    kmf.kernel()
    mo_coeff = avas.kernel(kmf, ["H 1s"], minao=cell.basis)[2]
    mo_coeff = np.asarray(mo_coeff).reshape(len(kpts), cell.nao_nr(), -1)
    return kmf, mo_coeff, kLASCI(kmf, 2, (1, 1), kmesh=kmesh)


def _rotate_active(las, mo_coeff, seed=7):
    rng = np.random.default_rng(seed)
    mo = np.array(mo_coeff, dtype=complex)
    ncore, ncas = las.ncore, las.ncas
    for k in range(mo.shape[0]):
        a = rng.standard_normal((ncas, ncas)) + 1j * rng.standard_normal((ncas, ncas))
        u = np.linalg.qr(a)[0]
        mo[k][:, ncore:ncore+ncas] = mo[k][:, ncore:ncore+ncas] @ u
    return mo


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.systems = {tuple(kmesh): _build(kmesh) for kmesh in ([1, 1, 1], [2, 1, 1])}

    def _check(self, las, mo_coeff):
        ref = h2e_for_cas_slow(las, mo_coeff)
        new = h2e_for_cas(las, mo_coeff)
        self.assertEqual(new.shape, ref.shape)
        np.testing.assert_allclose(new, ref, atol=1e-9, rtol=1e-9)

    def test_matches_slow(self):
        for kmesh, (kmf, mo_coeff, las) in self.systems.items():
            with self.subTest(kmesh=kmesh):
                self._check(las, mo_coeff)

    def test_matches_slow_spill(self):
        for kmesh, (kmf, mo_coeff, las) in self.systems.items():
            with self.subTest(kmesh=kmesh):
                old, las.max_memory = las.max_memory, 0
                try:
                    self._check(las, mo_coeff)
                finally:
                    las.max_memory = old

    def test_matches_slow_rotated_active(self):
        for kmesh, (kmf, mo_coeff, las) in self.systems.items():
            with self.subTest(kmesh=kmesh):
                self._check(las, _rotate_active(las, mo_coeff))



if __name__ == '__main__':
    unittest.main()
