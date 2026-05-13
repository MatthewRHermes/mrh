#!/bin/bash
import unittest
import numpy as np

from pyscf.fci import cistring

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx, direct_spin1_cplx_opt, direct_spin0_cplx

# Author: Bhavnesh Jangid

'''
In this file, there are test for the optimization of the cplx FCI solver.

Unit-Test-1: This will check the correctness of the optimized code `direct_spin1_cplx_opt` 
    with the reference code `direct_spin1_cplx`. We will compare the FCI energies and the 
    output of the `contract_2e` function for a random CI vector.

Unit-Test-2: This will check the correctness of the optimized spin0 code `direct_spin0_cplx` 
    with the reference spin1 code `direct_spin1_cplx`. 
'''


def gen_hermi_ham(norb):
    np.random.seed(12)
    h1 = np.random.random((norb,norb))
    h2 = np.random.random((norb,norb,norb,norb))

    h1 = np.asarray(h1).astype(np.complex128)
    h2 = np.asarray(h2).astype(np.complex128)

    h1.imag += 1e-2 * np.random.random(h1.shape)
    h2.imag += 1e-2 * np.random.random(h2.shape)

    # Restore symmetries
    h1 = h1 + h1.conj().T
    h2 = h2 + h2.conj().transpose(1,0,2,3)
    h2 = h2 + h2.conj().transpose(0,1,3,2)
    h2 = h2 + h2.conj().transpose(2,3,0,1)
    return h1, h2

class KnownValues(unittest.TestCase):
    def test_direct_spin1_cplx_opt(self):
        # Doing the calculation of (4o, 4e) and various spin states
        norb = 4
        nelecas = [(2, 2), (3, 1), (4, 0)]
        h1, h2 = gen_hermi_ham(norb)
        
        # Reference values
        for nele in nelecas:
            cisolver1 = direct_spin1_cplx.FCISolver()
            eref, ciref = cisolver1.kernel(h1, h2, norb, nele)

            # My computed values
            cisolver2 = direct_spin1_cplx_opt.FCISolver()
            e_com = cisolver2.kernel(h1.copy(), h2.copy(), norb, nele)[0]
            msg = "There is the mismatch in the FCI energies between the optimized and the reference code."
            self.assertAlmostEqual(eref, e_com, places=8, msg=msg)

            h2eff = h2
            ci0 = np.random.random(ciref.shape) + 1j * np.random.random(ciref.shape)
            ci0 = ci0.astype(np.complex128)

            ci0 /= np.linalg.norm(ci0)
            sigma_ref = cisolver1.contract_2e(h2eff, ci0, norb, nele)
            sigma_com = cisolver2.contract_2e(h2eff, ci0, norb, nele)
            msg = "There is the mismatch in the contract_2e outputs between the optimized and the reference code."
            self.assertTrue(np.allclose(sigma_ref, sigma_com, atol=1e-8), msg=msg)

    def test_direct_spin0_cplx(self):
        # Doing the calculation of (4o, 4e)
        norb = 4
        nelecas = (2, 2)
        h1, h2 = gen_hermi_ham(norb)

        # Reference spin1 complex solver
        cisolver_ref = direct_spin1_cplx.FCISolver()
        eref, ciref = cisolver_ref.kernel(h1, h2, norb, nelecas)

        # Optimized spin0 complex solver
        cisolver_spin0 = direct_spin0_cplx.FCISolver()
        e_spin0, ci_spin0 = cisolver_spin0.kernel(h1.copy(), h2.copy(), norb, nelecas)

        msg = (
            "There is a mismatch in the FCI energies between "
            "the optimized spin0 code and the reference spin1 code."
        )
        self.assertAlmostEqual(eref, e_spin0, places=8, msg=msg)

        # Build a random spin0-compatible CI vector.
        ci0 = np.random.random(ciref.shape) + 1j * np.random.random(ciref.shape)
        ci0 = ci0.astype(np.complex128)

        na = cistring.num_strings(norb, nelecas[0])
        nb = cistring.num_strings(norb, nelecas[1])
        ci0 = ci0.reshape(na, nb)
        ci0 = 0.5 * (ci0 + ci0.T)
        ci0 = ci0.ravel().astype(np.complex128)
        ci0 /= np.linalg.norm(ci0)

        # Reference sigma from full spin1 contraction.
        sigma_ref = cisolver_ref.contract_2e(h2, ci0, norb, nelecas)

        # Since input is spin0-symmetric, reference output should also be symmetric.
        # Symmetrize only to remove tiny numerical noise from the reference path.
        sigma_ref = 0.5 * (sigma_ref + sigma_ref.T)

        # Spin0 optimized sigma.
        sigma_spin0 = cisolver_spin0.contract_2e(h2.copy(), ci0.copy(), norb, nelecas)

        msg = (
            "There is a mismatch in contract_2e outputs between "
            "the optimized spin0 code and the symmetrized spin1 reference."
        )
        self.assertTrue(np.allclose(sigma_ref, sigma_spin0, atol=1e-8), msg=msg)

if __name__ == "__main__":
    unittest.main()