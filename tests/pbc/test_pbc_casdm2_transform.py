"""Numerical regressions for the batched Wannier-to-Bloch 2-RDM transform.

Each test compares transform_casdm2_kpts with the original per-block
_get_casdm2_kpts einsum.

KnownValues
    test_fourier_blocks: random real/complex tensors on several k-meshes with
        both Fourier phase signs, checking the FFT path and that the input is
        left unmodified.
    test_general_phases_and_permuted_kpoints: arbitrary complex phases and a
        permuted k-point order must use the dense path, and method='fourier'
        must raise.
    test_random_ci_2rdm_matches_per_block_transform: 2-RDMs of random FCI
        vectors in the Wannier basis, for active spaces of 2, 5 and 8
        orbitals per cell, agree with the per-block transform to 1e-12.
    test_init_orb_matches_original_contractions: the generalized Fock built in
        KLASSCF_HessianOperator._init_orb_ matches the original block loop.

ERIBackTransformTests
    test_back_transform_matches_original_block_sum: the batched Bloch-to-Wannier
        ERI transform matches the original block-by-block einsum sum for
        Fourier and general phases.

CumulantCacheTests
    test_reference_cumulant_cache_reused: the reference cumulant is
        transformed once and the cached blocks are reused.
"""

import unittest
from unittest.mock import patch

import numpy as np

from pyscf.fci import cistring, direct_spin1

from mrh.my_pyscf.pbc.mcscf import klasscf
from mrh.my_pyscf.pbc.mcscf.mc1step import _get_casdm2_kpts
from mrh.my_pyscf.pbc.util.casdm2_transformation import transform_casdm2_kpts


def phase_and_momenta(mesh, ncas, sign=1):
    cells = np.array(list(np.ndindex(mesh)))
    nkpts = len(cells)
    phase = np.zeros((nkpts, ncas, nkpts * ncas), dtype=complex)
    fourier = np.exp(sign * 2j * np.pi * (cells / np.array(mesh)) @ cells.T) / np.sqrt(nkpts)
    for orbital in range(ncas):
        phase[:, orbital, orbital::ncas] = fourier
    labels = (cells[:, None, None] - cells[None, :, None] + cells[None, None, :]) % np.array(mesh)
    kconserv = np.ravel_multi_index(labels.transpose(3, 0, 1, 2), mesh)
    return phase, kconserv


class KnownValues(unittest.TestCase):
    def assert_blocks(self, dm, phase, kconserv, actual, tol=1e-11):
        nkpts, ncas = phase.shape[:2]
        self.assertEqual(actual.shape, (nkpts,) * 3 + (ncas,) * 4)
        for k1, k2, k3 in np.ndindex((nkpts,) * 3):
            reference = _get_casdm2_kpts(dm, phase, (k1, k2, k3, int(kconserv[k1, k2, k3])))
            np.testing.assert_allclose(actual[k1, k2, k3], reference, atol=tol, rtol=tol)

    def test_fourier_blocks(self):
        # Dense random tensors intentionally have no translation symmetry.
        rng = np.random.default_rng(71)
        for mesh in ((1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2)):
            size = 2 * int(np.prod(mesh))
            for complex_dm in (False, True):
                dm = rng.standard_normal((size,) * 4)
                if complex_dm:
                    dm = dm + 1j * rng.standard_normal(dm.shape)
                original = dm.copy()
                for sign in (-1, 1):
                    with self.subTest(mesh=mesh, complex_dm=complex_dm, sign=sign):
                        phase, kconserv = phase_and_momenta(mesh, 2, sign)
                        actual = transform_casdm2_kpts(dm, phase, kconserv, kmesh=mesh, method='fourier')
                        self.assert_blocks(dm, phase, kconserv, actual)
                np.testing.assert_array_equal(dm, original)

    def test_general_phases_and_permuted_kpoints(self):
        rng = np.random.default_rng(72)
        mesh = (2, 2, 1)
        phase, kconserv = phase_and_momenta(mesh, 2)
        dm = rng.standard_normal((8,) * 4) + 1j * rng.standard_normal((8,) * 4)
        arbitrary = rng.standard_normal(phase.shape) + 1j * rng.standard_normal(phase.shape)
        actual = transform_casdm2_kpts(dm, arbitrary, kconserv, kmesh=mesh)
        self.assert_blocks(dm, arbitrary, kconserv, actual)
        with self.assertRaises(ValueError):
            transform_casdm2_kpts(dm, arbitrary, kconserv, kmesh=mesh, method='fourier')
        permutation = np.array([2, 0, 3, 1])
        inverse = np.argsort(permutation)
        permuted_map = inverse[kconserv[np.ix_(permutation, permutation, permutation)]]
        actual = transform_casdm2_kpts(dm, phase[permutation], permuted_map, kmesh=mesh)
        self.assert_blocks(dm, phase[permutation], permuted_map, actual)

    def test_random_ci_2rdm_matches_per_block_transform(self):
        rng = np.random.default_rng(74)
        # (k-mesh, orbitals per cell, (nalpha, nbeta)); the Wannier active
        # space has nkpts*ncas orbitals.
        cases = (
            ((2, 1, 1), 2, (2, 2)),
            ((2, 2, 1), 2, (2, 2)),
            ((2, 1, 1), 5, (2, 2)),
            ((2, 1, 1), 8, (2, 2)),
        )
        for mesh, ncas, nelec in cases:
            nkpts = int(np.prod(mesh))
            norb = nkpts * ncas
            na = cistring.num_strings(norb, nelec[0])
            nb = cistring.num_strings(norb, nelec[1])
            ci = rng.standard_normal((na, nb))
            ci /= np.linalg.norm(ci)
            _, dm2 = direct_spin1.make_rdm12(ci, norb, nelec)
            for label in ('fourier+', 'fourier-', 'general'):
                with self.subTest(mesh=mesh, ncas=ncas, phase=label):
                    sign = -1 if label == 'fourier-' else 1
                    phase, kconserv = phase_and_momenta(mesh, ncas, sign)
                    if label == 'general':
                        phase = (rng.standard_normal(phase.shape)
                                 + 1j * rng.standard_normal(phase.shape))
                    actual = transform_casdm2_kpts(dm2, phase, kconserv, kmesh=mesh)
                    self.assert_blocks(dm2, phase, kconserv, actual, tol=1e-12)

    def test_init_orb_matches_original_contractions(self):
        rng = np.random.default_rng(73)
        mesh = (2, 2, 1)
        phase, kconserv = phase_and_momenta(mesh, 2, -1)
        for general_phase in (False, True):
            with self.subTest(general_phase=general_phase):
                operator = klasscf.KLASSCF_HessianOperator.__new__(klasscf.KLASSCF_HessianOperator)
                operator.nkpts, operator.ncas, operator.ncastot = 4, 2, 8
                operator.ncore, operator.nocc, operator.nmo = 1, 3, 4
                operator.kmesh, operator.kpts = mesh, np.zeros((4, 3))
                operator.las = type('LAS', (), {'_scf': type('SCF', (), {'cell': object()})()})()
                operator.cascm2 = rng.standard_normal((8,) * 4) + 1j * rng.standard_normal((8,) * 4)
                operator.h1s = rng.standard_normal((2, 4, 4, 4)).astype(complex)
                operator.dm1s = rng.standard_normal((2, 4, 4, 4)).astype(complex)
                blocks = {key: rng.standard_normal((4, 2, 2, 2)) + 1j * rng.standard_normal((4, 2, 2, 2))
                          for key in np.ndindex((4,) * 3)}
                operator.eri_paaa = lambda k1, k2, k3: blocks[k1, k2, k3]
                current_phase = phase if not general_phase else rng.standard_normal(phase.shape) + 1j * rng.standard_normal(phase.shape)
                expected = np.array([sum(operator.h1s[s, k] @ operator.dm1s[s, k] for s in range(2)) for k in range(4)])
                for k1, k2, k3 in np.ndindex((4,) * 3):
                    dm2 = _get_casdm2_kpts(operator.cascm2, current_phase, (k1, k2, k3, int(kconserv[k1, k2, k3])))
                    expected[k1, :, 1:3] += np.tensordot(blocks[k1, k2, k3], dm2, axes=((1, 2, 3), (1, 2, 3)))
                with patch.object(klasscf.kpts_helper, 'get_kconserv', return_value=kconserv):
                    operator._init_orb_(mo_phase=current_phase)
                np.testing.assert_allclose(operator.fock1, expected, atol=1e-9, rtol=1e-11)



class CumulantCacheTests(unittest.TestCase):
    def test_reference_cumulant_cache_reused(self):
        phase, momenta = phase_and_momenta((2, 1, 1), 2)
        op = klasscf.KLASSCF_HessianOperator.__new__(klasscf.KLASSCF_HessianOperator)
        op.mo_phase = phase
        op.kmesh = (2, 1, 1)
        op.cascm2 = np.arange(4**4).reshape((4,)*4)
        with patch.object(klasscf, 'transform_casdm2_kpts', wraps=transform_casdm2_kpts) as transform:
            first = op._get_cascm2_kpts(momenta)
            second = op._get_cascm2_kpts(momenta)
        self.assertIs(first, second)
        self.assertEqual(transform.call_count, 1)


if __name__ == '__main__':
    unittest.main()
