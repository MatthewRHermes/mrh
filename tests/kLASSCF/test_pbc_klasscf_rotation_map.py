import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.klasscf import ActiveActiveRotationMap

# Author: Bhavnesh Jangid


"""Unit tests for active-active k-LASSCF orbital-rotation maps.

Test-0: Transform anti-Hermitian rotations between Bloch and Wannier bases.
Test-1: Project redundant k-point pair rotations onto independent coordinates.
Test-2: Handle an active space with no independent rotation pairs.
"""


def _fourier_mo_phase(nkpts, ncas):
    """Return a band-preserving complex k-to-Wannier unitary."""
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


class ActiveActiveRotationMapTests(unittest.TestCase):

    def test_wannier_bloch_matrix_map(self):
        nkpts = 3
        ncas = 2
        rotation_map = ActiveActiveRotationMap(
            _fourier_mo_phase(nkpts, ncas),
            np.full(nkpts, ncas),
        )
        rng = np.random.default_rng(12)
        lower = (
            rng.standard_normal((nkpts, ncas, ncas))
            + 1j * rng.standard_normal((nkpts, ncas, ncas))
        )
        lower = np.tril(lower, -1)
        diagonal = 1j * rng.standard_normal((nkpts, ncas))
        bloch_rotation = lower - lower.conj().transpose(0, 2, 1)
        bloch_rotation += np.asarray([
            np.diag(values) for values in diagonal
        ])

        wannier_rotation = rotation_map.bloch_to_wannier(bloch_rotation)
        bloch_round_trip = rotation_map.wannier_to_bloch(wannier_rotation)

        np.testing.assert_allclose(bloch_round_trip, bloch_rotation)
        np.testing.assert_allclose(
            wannier_rotation + wannier_rotation.conj().T, 0.0,
            atol=1e-13,
        )

        arbitrary_wannier = (
            rng.standard_normal((nkpts * ncas,) * 2)
            + 1j * rng.standard_normal((nkpts * ncas,) * 2)
        )
        projected = rotation_map.bloch_to_wannier(
            rotation_map.wannier_to_bloch(arbitrary_wannier)
        )
        projected_twice = rotation_map.bloch_to_wannier(
            rotation_map.wannier_to_bloch(projected)
        )
        np.testing.assert_allclose(projected_twice, projected)

    def test_pair_projection_removes_redundancy(self):
        rotation_map = ActiveActiveRotationMap(
            _fourier_mo_phase(2, 2), np.array([2, 2]),
        )

        self.assertEqual(rotation_map.pair_map.shape, (2, 4))
        self.assertEqual(rotation_map.nvar, 1)
        expected_projector = np.array([
            [0.5, -0.5],
            [-0.5, 0.5],
        ])
        np.testing.assert_allclose(
            rotation_map.basis @ rotation_map.basis.conj().T,
            expected_projector,
            atol=1e-13,
        )

        coordinates = np.array([0.4 - 0.3j])
        bloch_rotation = rotation_map.unpack(coordinates)
        np.testing.assert_allclose(
            rotation_map.pack(bloch_rotation), coordinates,
        )
        np.testing.assert_allclose(
            bloch_rotation[0, 1, 0] + bloch_rotation[1, 1, 0], 0.0,
            atol=1e-13,
        )

        arbitrary = np.zeros((2, 2, 2), dtype=np.complex128)
        arbitrary[:, 1, 0] = [0.2 + 0.1j, -0.4 + 0.3j]
        arbitrary -= arbitrary.conj().transpose(0, 2, 1)
        projected = rotation_map.unpack(rotation_map.pack(arbitrary))
        lower_pairs = arbitrary[rotation_map.bloch_pair_idx]
        expected_pairs = (
            rotation_map.basis @ rotation_map.basis.conj().T @ lower_pairs
        )
        np.testing.assert_allclose(
            projected[rotation_map.bloch_pair_idx], expected_pairs,
        )

    def test_empty_pair_space(self):
        rotation_map = ActiveActiveRotationMap(
            np.ones((1, 1, 1)), np.array([1]),
        )

        self.assertEqual(rotation_map.nvar, 0)
        self.assertEqual(rotation_map.pair_map.shape, (0, 0))
        np.testing.assert_array_equal(
            rotation_map.unpack([]), np.zeros((1, 1, 1)),
        )

if __name__ == "__main__":
    unittest.main()
