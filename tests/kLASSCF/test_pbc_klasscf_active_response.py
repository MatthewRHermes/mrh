import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from mrh.my_pyscf.pbc.mcscf.klasscf import KLASSCF_HessianOperator
from mrh.my_pyscf.pbc.mcscf.active_active_rotation_map import ActiveActiveRotationMap


class ActiveResponseTests(unittest.TestCase):
    def make_operator(self, n, complex_data=True):
        rng = np.random.default_rng(781)
        def random(shape):
            return rng.normal(size=shape) + (1j*rng.normal(size=shape) if complex_data else 0)
        op = object.__new__(KLASSCF_HessianOperator)
        op.ncastot = n
        op.eri_cas = random((n,)*4)
        op.casdm1s = random((2,n,n))
        op.cascm2 = random((n,)*4)
        h1, fock1 = random((n,n)), random((n,n))
        op._active_wannier_intermediates = lambda: (h1, fock1)
        op.mo_coeff = np.empty((1, n, n), dtype=complex)
        return op, rng

    def test_response_matches_slow_real_and_complex(self):
        for n in (2, 4, 16):
            for complex_data in (False, True):
                with self.subTest(n=n, complex_data=complex_data):
                    op, rng = self.make_operator(n, complex_data)
                    for _ in range(3):
                        kappa = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
                        kappa -= kappa.conj().T
                        expected = op._orbital_hessian_response_active_active_wannier_slow(kappa)
                        actual = op._orbital_hessian_response_active_active_wannier(kappa)
                        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-12)

    def test_response_cache_reused(self):
        op, rng = self.make_operator(4)
        kappa = rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4))
        kappa -= kappa.conj().T
        first = op._orbital_hessian_response_active_active_wannier(kappa)
        cache = op._active_wannier_response_cache
        with patch.object(np, 'einsum', side_effect=AssertionError('repeated ERI contraction')):
            second = op._orbital_hessian_response_active_active_wannier(kappa)
        self.assertIs(cache, op._active_wannier_response_cache)
        np.testing.assert_allclose(first, second)

    def test_full_projected_hessian_matches_slow(self):
        op, _ = self.make_operator(16)
        phase = np.kron(np.fft.fft(np.eye(8))/np.sqrt(8), np.eye(2)).reshape(8,2,16)
        rotation_map = ActiveActiveRotationMap(phase, [2]*8)
        op.ugg = SimpleNamespace(active_active_map=rotation_map,
                                 nvar_orb_active_active=rotation_map.nvar,
                                 active_coordinates_real=True)
        actual = op._get_Horb_active_active()
        del op._Horb_active_active_cache
        op._orbital_hessian_response_active_active_wannier = op._orbital_hessian_response_active_active_wannier_slow
        expected = op._get_Horb_active_active()
        for result, reference in zip(actual, expected):
            np.testing.assert_allclose(result, reference, atol=2e-10, rtol=2e-12)

    def test_low_memory_uses_disk_cache(self):
        for n in (2, 4, 6):
            for complex_data in (False, True):
                with self.subTest(n=n, complex_data=complex_data):
                    op, rng = self.make_operator(n, complex_data)
                    op.las = SimpleNamespace(max_memory=0)
                    kappa = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
                    kappa -= kappa.conj().T
                    expected = op._orbital_hessian_response_active_active_wannier_slow(kappa)
                    with patch.object(op, '_orbital_hessian_response_active_active_wannier_slow') as slow:
                        actual = op._orbital_hessian_response_active_active_wannier(kappa)
                    slow.assert_not_called()
                    _, direct, conjugate = op._active_wannier_response_cache
                    self.assertIsInstance(direct, h5py.Dataset)
                    self.assertIsInstance(conjugate, h5py.Dataset)
                    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-12)
                    # The disk cache is reused by later directions.
                    again = op._orbital_hessian_response_active_active_wannier(kappa)
                    self.assertIs(op._active_wannier_response_cache[1], direct)
                    np.testing.assert_allclose(again, expected, atol=2e-10, rtol=2e-12)

    def test_disk_and_ram_matrices_agree(self):
        op, _ = self.make_operator(5)
        ram = op._active_wannier_response_intermediates()
        op._active_wannier_response_cache = None
        op.las = SimpleNamespace(max_memory=0)
        disk = op._active_wannier_response_intermediates()
        np.testing.assert_allclose(disk[0], ram[0], atol=1e-12)
        for on_disk, in_ram in zip(disk[1:], ram[1:]):
            np.testing.assert_allclose(on_disk[()], in_ram, atol=1e-12)
