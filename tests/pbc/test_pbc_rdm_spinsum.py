"""Regression tests for the ordinary spin-summed complex native RDM kernel."""
import math
import unittest
import numpy as np
from pyscf import gto, scf, ao2mo
from pyscf.fci import direct_spin1
from mrh.my_pyscf.pbc.fci import direct_spin1_cplx as fci


def from_spin_blocks(ci, norb, nelec, link_index=None, reorder=True):
    (a, b), (aa, ab, bb) = fci.make_rdm12s(
        ci, norb, nelec, link_index=link_index, reorder=reorder)
    return (a+b).conj().T, aa+bb+ab+ab.transpose(2, 3, 0, 1)


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(401)

    def ci(self, norb, nelec, complex_value=True):
        shape = tuple(math.comb(norb, n) for n in nelec)
        c = self.rng.normal(size=shape)
        if complex_value:
            c = c + 1j*self.rng.normal(size=shape)
        return c / np.linalg.norm(c)

    def assert_rdms_close(self, a, b):
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-10)

    def test_spin_blocks_and_real_pyscf(self):
        cases = [(6, (0, 0)), (6, (4, 0)), (6, (0, 4)), (6, (6, 6)),
                 (6, (3, 3)), (6, (4, 2)), (8, (4, 4)), (8, (5, 3))]
        for norb, nelec in cases:
            for complex_value in (False, True):
                ci = self.ci(norb, nelec, complex_value)
                for reorder in (False, True):
                    with self.subTest(norb=norb, nelec=nelec,
                                      complex=complex_value, reorder=reorder):
                        dm = fci.make_rdm12(ci, norb, nelec, reorder=reorder)
                        ref = from_spin_blocks(ci, norb, nelec, reorder=reorder)
                        self.assert_rdms_close(dm, ref)
                        if not complex_value:
                            self.assert_rdms_close(dm, direct_spin1.make_rdm12(
                                ci, norb, nelec, reorder=reorder))
                        if reorder:
                            n = sum(nelec)
                            self.assertAlmostEqual(np.trace(dm[0]).real, n, places=10)
                            self.assertAlmostEqual(np.einsum('ppqq->', dm[1]).real,
                                                   n*(n-1), places=9)
                            np.testing.assert_allclose(
                                np.einsum('pqrr->pq', dm[1]), (n-1)*dm[0].T,
                                atol=1e-10, rtol=0)

    def _check_solved_fci_spins(self, norb, sectors):
        # H_n/STO-3G has exactly n spatial orbitals and n electrons: full
        # CAS(n,n). Reuse one common spatial-orbital basis for all spin states.
        mol = gto.M(atom=[('H', (0, 0, 1.1*i)) for i in range(norb)],
                    basis='sto-3g', spin=norb % 2, verbose=0)
        mf = scf.ROHF(mol) if mol.spin else scf.RHF(mol)
        mf.conv_tol = 1e-11
        mf.kernel()
        self.assertTrue(mf.converged)
        mo = mf.mo_coeff
        self.assertEqual(mo.shape[1], norb)
        h1 = mo.T @ mf.get_hcore() @ mo
        h2 = ao2mo.restore(1, ao2mo.kernel(mol, mo), norb)
        ecore = mol.energy_nuc()
        for nelec, spin in sectors:
            with self.subTest(norb=norb, nelec=nelec, spin=spin):
                solver = direct_spin1.FCISolver(mol)
                solver.spin = nelec[0]-nelec[1]
                solver.conv_tol = 1e-12
                energy, ci = solver.kernel(h1, h2, norb, nelec, ecore=ecore)
                self.assertTrue(solver.converged)
                h2eff = solver.absorb_h1e(h1, h2, norb, nelec, .5)
                residual = np.linalg.norm(solver.contract_2e(h2eff, ci, norb, nelec)
                                          -(energy-ecore)*ci)
                self.assertLess(residual, 1e-9)
                ss, mult = solver.spin_square(ci, norb, nelec)
                self.assertAlmostEqual(ss, spin*(spin+1), places=9)
                self.assertAlmostEqual(mult, 2*spin+1, places=9)
                complex_ci = np.asarray(ci, dtype=complex)
                for reorder in (False, True):
                    new = fci.make_rdm12(complex_ci, norb, nelec, reorder=reorder)
                    old = from_spin_blocks(complex_ci, norb, nelec, reorder=reorder)
                    py = solver.make_rdm12(ci, norb, nelec, reorder=reorder)
                    self.assert_rdms_close(new, old)
                    self.assert_rdms_close(new, py)
                    if reorder:
                        reconstructed = (np.einsum('pq,qp->', h1, new[0])
                            + .5*np.einsum('pqrs,pqrs->', h2, new[1])+ecore)
                        self.assertAlmostEqual(reconstructed.real, energy, places=9)
                        self.assertLess(abs(reconstructed.imag), 1e-10)

    def test_solved_fci_4_4_singlet_triplet_quintet(self):
        self._check_solved_fci_spins(4, [((2, 2), 0), ((3, 1), 1), ((4, 0), 2)])

    def test_solved_fci_5_5_doublet_quartet_sextet(self):
        self._check_solved_fci_spins(5, [((3, 2), .5), ((4, 1), 1.5), ((5, 0), 2.5)])


if __name__ == '__main__':
    unittest.main()
