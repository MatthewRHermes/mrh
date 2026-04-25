#!/bin/bash
import unittest
import numpy as np
from ase import Atoms
from functools import reduce

from pyscf import ao2mo, fci
from pyscf.pbc import gto as pgto, scf
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.tools import k2gamma

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx, cplx_fci_opt
from mrh.my_pyscf.pbc.mcscf.k2R import get_mo_coeff_k2R


# Author: Bhavnesh Jangid

'''
In this file, there are test for the optimization of the cplx FCI solver.

Unit-Test-1: Optimization of the contract_2e function.
'''


def gen_hermi_ham(norb):
    np.random.seed(12)
    h1 = np.random.random((norb,norb))
    h2 = np.random.random((norb,norb,norb,norb))

    h1 = np.asarray(h1).astype(np.complex128)
    h2 = np.asarray(h2).astype(np.complex128)

    h1.imag += 1e-2
    h2.imag += 1e-2

    # Restore symmetries
    h1 = h1 + h1.conj().T
    h2 = h2 + h2.conj().transpose(1,0,2,3)
    h2 = h2 + h2.conj().transpose(0,1,3,2)
    h2 = h2 + h2.conj().transpose(2,3,0,1)
    return h1, h2

class KnownValues(unittest.TestCase):
    def test_cplx_fci_opt(self):
        # Doing the calculation of (4o, 4e)
        norb = 4
        nelecas = (2, 2)
        h1, h2 = gen_hermi_ham(norb)
        
        # Reference values
        cisolver1 = direct_spin1_cplx.FCISolver()
        eref, ciref = cisolver1.kernel(h1, h2, norb, nelecas)
        
        # My computed values
        cisolver2 = cplx_fci_opt.FCISolver()
        e_com = cisolver2.kernel(h1, h2, norb, nelecas)[0]
        msg = "There is the mismatch in the FCI energies between the optimized and the reference code."
        self.assertAlmostEqual(eref, e_com, places=8, msg=msg)

        # Also comparing the contract_2e function outputs
        h2eff = h2
        ci0 = np.random.random(ciref.shape) + 1j * np.random.random(ciref.shape)
        ci0 = ci0.astype(np.complex128)
        ci0 /= np.linalg.norm(ci0)
        sigma_ref = cisolver1.contract_2e(h2eff, ci0, norb, nelecas)
        sigma_com = cisolver2.contract_2e(h2eff, ci0, norb, nelecas)
        msg = "There is the mismatch in the contract_2e outputs between the optimized and the reference code."
        self.assertTrue(np.allclose(sigma_ref, sigma_com, atol=1e-8), msg=msg)


if __name__ == "__main__":
    unittest.main()