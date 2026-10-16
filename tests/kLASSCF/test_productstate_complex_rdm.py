import unittest
import numpy as np

from mrh.my_pyscf.pbc.mcscf.productstate import PBCProductStateFCISolver

# Author: Bhavnesh Jangid

'''
Test-0: Test the calculation of complex, disconnected two-particle reduced density matrices.
'''

class FakeFCISolver:
    spin = 0
    charge = 0

    def __init__(self, dm1a, dm1b, dm2):
        self.dm1a = dm1a
        self.dm1b = dm1b
        self.dm2 = dm2

    def make_rdm1s(self, ci, norb, nelec):
        return self.dm1a, self.dm1b

    def make_rdm2(self, ci, norb, nelec):
        return self.dm2


class KnownValues(unittest.TestCase):

    def test_complex_disconnected_rdm2(self):
        """Check complex inter-fragment 2-RDM assembly.
        Convert native density matrices to <p^+ q> before LAS contraction."""
        dm1a = np.array([[0.7, 0.1j], [-0.1j, 0.3]])
        dm1b = np.array([[0.2, 0.04j], [-0.04j, 0.8]])
        dm1c = np.array([[0.6, 0.2j], [-0.2j, 0.4]])
        dm1d = np.array([[0.1, 0.03j], [-0.03j, 0.9]])
        local_dm2 = np.zeros((2, 2, 2, 2), dtype=complex)
        solvers = [FakeFCISolver(dm1a, dm1b, local_dm2),
                   FakeFCISolver(dm1c, dm1d, local_dm2)]
        solver = PBCProductStateFCISolver(solvers)

        dm2 = solver.make_rdm2([np.ones(1, dtype=complex)] * 2,
                               [2, 2], [(1, 1), (1, 1)])

        dm1_0 = (dm1a + dm1b).T
        dm1_1 = (dm1c + dm1d).T
        direct = np.multiply.outer(dm1_0, dm1_1)
        exchange = -np.einsum('il,kj->ijkl', dm1a.T, dm1c.T)
        exchange -= np.einsum('il,kj->ijkl', dm1b.T, dm1d.T)
        self.assertTrue(np.iscomplexobj(dm2))
        np.testing.assert_allclose(dm2[:2, :2, 2:, 2:], direct)
        np.testing.assert_allclose(dm2[2:, 2:, :2, :2],
                                   direct.transpose(2, 3, 0, 1))
        np.testing.assert_allclose(dm2[:2, 2:, 2:, :2],
                                   exchange)
        np.testing.assert_allclose(dm2[2:, :2, :2, 2:],
                                   exchange.conj().transpose(1, 0, 3, 2))

if __name__ == '__main__':
    unittest.main()
