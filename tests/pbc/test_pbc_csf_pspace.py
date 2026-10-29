#!/bin/env python

import unittest
import numpy as np

from pyscf import gto

from mrh.my_pyscf.pbc.fci import cplx_csf_helper, csf_cplx

# Author: Bhavnesh Jangid

# Reference tests for csf_cplx.pspace. The p-space Hamiltonian must equal
# the exact CSF-basis Hamiltonian <I|H|J> restricted to the returned CSF
# addresses, independent of how pspace builds it internally. The reference
# is built from the full determinant-basis contraction, so these tests
# guard any rewrite that avoids the determinant-basis p-space matrix.
#
# Test-0: Complex Hermitian H, doublet, p-space smaller than the full CSF space.
# Test-1: Complex Hermitian H, doublet, p-space spanning the full CSF space.
# Test-2: Complex Hermitian H, quartet (more unpaired electrons).
# Test-3: Purely real H must give a real p-space matrix.
# Test-4: Purely imaginary off-diagonal part must give an
#         anti-symmetric imaginary block with zero imaginary diagonal.
# Test-5: Returned CSF addresses are the lowest-energy ones by hdiag_csf.

def gen_complex_hermi_ham(norb, seed=34):
    rng = np.random.default_rng(seed)
    h1_ao = rng.standard_normal((norb, norb))
    h1_ao = h1_ao + h1_ao.T
    chol = rng.standard_normal((2 * norb, norb, norb))
    chol = chol + chol.transpose(0, 2, 1)
    eri_ao = np.einsum('Lpq,Lrs->pqrs', chol, chol)
    mo = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    mo = np.linalg.qr(mo)[0]
    h1 = np.einsum('pi,pq,qj->ij', mo.conj(), h1_ao, mo)
    eri = np.einsum('pi,qj,rk,sl,pqrs->ijkl', mo.conj(), mo, mo.conj(), mo,
                    eri_ao, optimize=True)
    return h1, eri

def gen_real_ham(norb, seed=35):
    h1, eri = gen_complex_hermi_ham(norb, seed=seed)
    rng = np.random.default_rng(seed)
    h1_ao = rng.standard_normal((norb, norb))
    h1 = (h1_ao + h1_ao.T).astype(np.complex128)
    chol = rng.standard_normal((2 * norb, norb, norb))
    chol = chol + chol.transpose(0, 2, 1)
    eri = np.einsum('Lpq,Lrs->pqrs', chol, chol).astype(np.complex128)
    return h1, eri

def make_solver(norb, nelec, smult):
    solver = csf_cplx.FCISolver(gto.M(verbose=0), smult=smult)
    solver.norb = norb
    solver.nelec = nelec
    solver.check_transformer_cache()
    return solver

def full_csf_hamiltonian(solver, h1e, eri, norb, nelec):
    '''Exact CSF-basis H from the full determinant-basis contraction.'''
    transformer = solver.transformer
    h2eff = solver.absorb_h1e(h1e, eri, norb, nelec, fac=0.5)
    umat = cplx_csf_helper.vec_csf2det_cplx(
        transformer, np.eye(transformer.ncsf, dtype=np.complex128),
        normalize=False)
    hu = np.stack([solver.contract_2e(h2eff, u, norb, nelec).ravel()
                   for u in umat])
    return umat.conj() @ hu.T

class KnownValues(unittest.TestCase):
    def run_pspace(self, h1e, eri, norb, nelec, smult, npsp):
        solver = make_solver(norb, nelec, smult)
        hdiag_det = solver.make_hdiag(h1e, eri, norb, nelec)
        hdiag_csf = solver.make_hdiag_csf(h1e, eri, norb, nelec,
                                          hdiag_det=hdiag_det)
        csf_addr, h0 = solver.pspace(h1e, eri, norb, nelec,
                                     hdiag_det=hdiag_det,
                                     hdiag_csf=hdiag_csf, npsp=npsp)
        hfull = full_csf_hamiltonian(solver, h1e, eri, norb, nelec)
        return solver, hdiag_csf, csf_addr, h0, hfull

    def check_against_full(self, h1e, eri, norb, nelec, smult, npsp):
        solver, hdiag_csf, csf_addr, h0, hfull = self.run_pspace(
            h1e, eri, norb, nelec, smult, npsp)
        self.assertEqual(h0.shape, (csf_addr.size, csf_addr.size))
        self.assertGreaterEqual(csf_addr.size, min(npsp, solver.transformer.ncsf))
        ref = hfull[np.ix_(csf_addr, csf_addr)]
        self.assertLess(np.max(np.abs(h0 - ref)), 1e-9)
        self.assertLess(np.max(np.abs(h0 - h0.conj().T)), 1e-10)
        return solver, hdiag_csf, csf_addr, h0

    def test_complex_doublet_subspace(self):
        norb, nelec = 5, (3, 2)
        h1e, eri = gen_complex_hermi_ham(norb)
        _, _, _, h0 = self.check_against_full(h1e, eri, norb, nelec, 2, 10)
        self.assertGreater(np.max(np.abs(h0.imag)), 1e-3)

    def test_complex_doublet_full_space(self):
        norb, nelec = 5, (3, 2)
        h1e, eri = gen_complex_hermi_ham(norb)
        solver, _, csf_addr, _ = self.check_against_full(
            h1e, eri, norb, nelec, 2, 1000)
        self.assertEqual(csf_addr.size, solver.transformer.ncsf)

    def test_complex_quartet(self):
        norb, nelec = 6, (4, 1)
        h1e, eri = gen_complex_hermi_ham(norb, seed=36)
        self.check_against_full(h1e, eri, norb, nelec, 4, 8)

    def test_real_hamiltonian_gives_real_h0(self):
        norb, nelec = 5, (3, 2)
        h1e, eri = gen_real_ham(norb)
        _, _, _, h0 = self.check_against_full(h1e, eri, norb, nelec, 2, 10)
        self.assertEqual(np.max(np.abs(h0.imag)), 0)

    def test_imag_block_antisymmetric(self):
        norb, nelec = 5, (3, 2)
        h1e, eri = gen_complex_hermi_ham(norb)
        _, _, _, h0 = self.check_against_full(h1e, eri, norb, nelec, 2, 12)
        self.assertLess(np.max(np.abs(np.diag(h0).imag)), 1e-12)
        self.assertLess(np.max(np.abs(h0.imag + h0.imag.T)), 1e-10)
        self.assertLess(np.max(np.abs(h0.real - h0.real.T)), 1e-10)

    def test_lowest_energy_csfs_selected(self):
        norb, nelec, npsp = 5, (3, 2), 10
        h1e, eri = gen_complex_hermi_ham(norb)
        _, hdiag_csf, csf_addr, _ = self.check_against_full(
            h1e, eri, norb, nelec, 2, npsp)
        lowest = np.sort(np.argsort(hdiag_csf.real)[:npsp])
        self.assertTrue(set(lowest).issubset(set(csf_addr.tolist())))

if __name__ == "__main__":
    print("Full Tests for complex CSF pspace")
    unittest.main()
