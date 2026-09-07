import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.klasscf import (
    KLASSCF_UnitaryGroupGenerators,
    get_ugg,
)


# Author: Bhavnesh Jangid

"""Unit tests for k-LASSCF unitary-group generators.

Test-0: Forward arguments through the public unitary-group-generator factory.
Test-1: Pack and unpack complex orbital and CI coordinates.
Test-2: Include projected active-active orbital coordinates.
Test-3: Omit frozen CI coordinates and restore them as zero on unpacking.
"""


def _fourier_mo_phase(nkpts, ncas):
    phase = np.zeros(
        (nkpts, ncas, nkpts * ncas), dtype=np.complex128,
    )
    fourier = np.exp(
        2j * np.pi * np.arange(nkpts)[:, None]
        * np.arange(nkpts)[None, :] / nkpts
    ) / np.sqrt(nkpts)
    for k in range(nkpts):
        for cell in range(nkpts):
            i = cell * ncas
            phase[k, :, i:i + ncas] = fourier[k, cell] * np.eye(ncas)
    return phase


class _IdentityTransformer:
    ncsf = 2
    ndet = 2

    @staticmethod
    def vec_det2csf(vector, order="C", normalize=False):
        return np.array(vector, copy=True)

    @staticmethod
    def vec_csf2det(vector, order="C", normalize=False):
        return np.array(vector, copy=True)


class _FakeSolver:
    def __init__(self):
        self.transformer = _IdentityTransformer()

    def check_transformer_cache(self):
        pass


class _FakeFCIBox:
    def __init__(self):
        self.fcisolvers = [_FakeSolver()]

    @staticmethod
    def _get_nelec(solver, nelec):
        return tuple(nelec)


class _FakeKLAS:
    def __init__(self):
        self.kpts = np.zeros((2, 3))
        self.mo_coeff = np.zeros((2, 3, 4), dtype=np.complex128)
        self.ncore = 1
        self.ncas = 1
        self.frozen = None
        self.frozen_ci = None
        self.ncas_sub = np.array([1, 1])
        self.nelecas_sub = np.array([(1, 0), (1, 0)])
        self.fciboxes = [_FakeFCIBox(), _FakeFCIBox()]
        self.ci = [
            [np.array([1.0, 0.0], dtype=np.complex128)],
            [np.array([1.0, 0.0], dtype=np.complex128)],
        ]
        self.mo_phase = _fourier_mo_phase(2, 1)


class _FakeKLASActive(_FakeKLAS):
    def __init__(self):
        super().__init__()
        self.ncas = 2
        self.ncas_sub = np.array([2, 2])
        self.mo_phase = _fourier_mo_phase(2, 2)


class UnitaryGroupGeneratorTests(unittest.TestCase):

    def test_get_ugg_forwards_constructor_arguments(self):
        sentinel = object()
        calls = []

        class _FactoryOwner:
            @staticmethod
            def _ugg(klas, **kwargs):
                calls.append((klas, kwargs))
                return sentinel

        klas = _FactoryOwner()
        mo_coeff = object()
        ci = object()
        mo_phase = object()

        result = get_ugg(
            klas, mo_coeff=mo_coeff, ci=ci, mo_phase=mo_phase,
        )

        self.assertIs(result, sentinel)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], klas)
        self.assertIs(calls[0][1]["mo_coeff"], mo_coeff)
        self.assertIs(calls[0][1]["ci"], ci)
        self.assertIs(calls[0][1]["mo_phase"], mo_phase)

    def test_complex_orbital_and_ci_pack_unpack(self):
        ugg = KLASSCF_UnitaryGroupGenerators(_FakeKLAS())

        x_orb = (
            np.arange(ugg.nvar_orb, dtype=float)
            + 1j * np.arange(ugg.nvar_orb, dtype=float)[::-1]
        ) / 7.0
        kappa = ugg.unpack_orb(x_orb)
        np.testing.assert_allclose(ugg.pack_orb(kappa), x_orb)
        np.testing.assert_allclose(
            kappa + kappa.conj().transpose(0, 2, 1), 0.0,
        )

        ci = [
            [np.array([0.2 + 0.3j, -0.4j])],
            [np.array([-0.1 + 0.5j, 0.7])],
        ]
        packed = ugg.pack(kappa, ci)
        kappa_out, ci_out = ugg.unpack(packed)

        self.assertEqual(packed.shape, (ugg.nvar_tot,))
        self.assertTrue(np.issubdtype(packed.dtype, np.complexfloating))
        np.testing.assert_allclose(kappa_out, kappa)
        for actual_r, expected_r in zip(ci_out, ci):
            np.testing.assert_allclose(actual_r[0], expected_r[0])

    def test_includes_projected_active_active_coordinates(self):
        klas = _FakeKLASActive()
        ugg = KLASSCF_UnitaryGroupGenerators(klas)

        self.assertEqual(ugg.nvar_orb_external, 10)
        self.assertEqual(ugg.nvar_orb_active_active, 1)
        self.assertEqual(ugg.nvar_orb, 11)
        self.assertFalse(np.any(ugg.get_gx_idx()))

        x_orb = np.linspace(0.1, 1.1, ugg.nvar_orb).astype(complex)
        x_orb += 1j * np.linspace(-0.7, 0.4, ugg.nvar_orb)
        kappa = ugg.unpack_orb(x_orb)

        np.testing.assert_allclose(ugg.pack_orb(kappa), x_orb)
        np.testing.assert_allclose(
            kappa + kappa.conj().transpose(0, 2, 1), 0.0,
            atol=1e-13,
        )
        active = slice(klas.ncore, klas.ncore + klas.ncas)
        active_rotation = kappa[:, active, active]
        np.testing.assert_allclose(
            active_rotation[0, 1, 0] + active_rotation[1, 1, 0],
            0.0,
            atol=1e-13,
        )
        wannier_rotation = ugg.active_active_map.bloch_to_wannier(
            active_rotation,
        )
        np.testing.assert_allclose(
            wannier_rotation[:2, :2], 0.0, atol=1e-13,
        )
        np.testing.assert_allclose(
            wannier_rotation[2:, 2:], 0.0, atol=1e-13,
        )

    def test_frozen_ci_is_omitted_and_unpacks_to_zero(self):
        klas = _FakeKLAS()
        klas.frozen_ci = [1]
        ugg = KLASSCF_UnitaryGroupGenerators(klas)
        ci = [
            [np.array([0.2 + 0.3j, -0.4j])],
            [np.array([-0.1 + 0.5j, 0.7])],
        ]

        packed = ugg.pack_ci(ci)
        unpacked = ugg.unpack_ci(packed)

        self.assertEqual(ugg.nvar_ci, 2)
        np.testing.assert_allclose(unpacked[0][0], ci[0][0])
        np.testing.assert_array_equal(unpacked[1][0], np.zeros(2))


if __name__ == "__main__":
    unittest.main()
