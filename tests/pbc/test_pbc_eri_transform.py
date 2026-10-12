"""Bloch-to-Wannier transformation of the active-space ERIs."""

import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.mc_ao2mo_opt import transform_eri_kpts_to_wannier
from test_pbc_casdm2_transform import phase_and_momenta


class ERIBackTransformTests(unittest.TestCase):
    def test_back_transform_matches_original_block_sum(self):
        rng = np.random.default_rng(923)
        for mesh in ((1, 1, 1), (2, 1, 1), (2, 2, 1)):
            for phase_kind in ('positive_fourier', 'negative_fourier', 'general'):
                with self.subTest(mesh=mesh, phase=phase_kind):
                    phase, momenta = phase_and_momenta(mesh, 2, sign=-1 if phase_kind == 'negative_fourier' else 1)
                    if phase_kind == 'general':
                        phase = rng.normal(size=phase.shape) + 1j*rng.normal(size=phase.shape)
                    nkpts, ncas, nactive = phase.shape
                    shape = (nkpts,)*3 + (ncas,)*4
                    blocks = rng.normal(size=shape) + 1j*rng.normal(size=shape)
                    reference = np.zeros((nactive,)*4, dtype=complex)
                    for k1, k2, k3 in np.ndindex((nkpts,)*3):
                        k4 = momenta[k1, k2, k3]
                        reference += np.einsum(
                            'aP,bQ,abcd,cR,dS->PQRS',
                            phase[k1].conj(), phase[k2], blocks[k1,k2,k3],
                            phase[k3].conj(), phase[k4], optimize=True,
                        )
                    actual = transform_eri_kpts_to_wannier(blocks, phase, momenta)
                    np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=2e-12)


if __name__ == '__main__':
    unittest.main()
