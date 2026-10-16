"""Complete real-linear active rotation map: diagonals, metric and redundancy."""

import unittest
import numpy as np
from mrh.my_pyscf.pbc.mcscf.active_active_rotation_map import ActiveActiveRotationMap


def _fourier_mo_phase(nkpts, ncas):
    fourier = np.exp(2j * np.pi * np.arange(nkpts)[:, None]
                     * np.arange(nkpts)[None, :] / nkpts) / np.sqrt(nkpts)
    return np.kron(fourier, np.eye(ncas)).reshape(nkpts, ncas, nkpts*ncas)


class ActiveActiveRotationMapTests(unittest.TestCase):
    def test_fourier_rank_counts_real_directions_including_diagonals(self):
        for nk, ncas in [(1, 2), (2, 2), (3, 1), (3, 2), (3, 5)]:
            with self.subTest(nk=nk, ncas=ncas):
                mapping = ActiveActiveRotationMap(_fourier_mo_phase(nk, ncas), [ncas]*nk)
                # k-diagonal u(ncas) has nk*ncas**2 real generators; its
                # k-independent intra-fragment u(ncas) is redundant.
                self.assertEqual(len(mapping.generators), nk*ncas**2)
                self.assertEqual(mapping.nvar, (nk-1)*ncas**2)

    def test_three_point_diagonal_generator_is_retained_and_real_in_wannier_space(self):
        mapping = ActiveActiveRotationMap(_fourier_mo_phase(3, 2), [2]*3)
        rotation = np.zeros((3, 2, 2), dtype=complex)
        rotation[:, 0, 0] = 1j * np.array([0., 1., -1.])
        coordinates = mapping.pack(rotation)
        self.assertTrue(np.issubdtype(coordinates.dtype, np.complexfloating))
        np.testing.assert_array_equal(coordinates.imag, 0.)
        np.testing.assert_allclose(mapping.unpack(coordinates), rotation, atol=1e-12)
        wannier = mapping.bloch_to_wannier(rotation)
        np.testing.assert_allclose(wannier.imag, 0., atol=1e-12)
        np.testing.assert_allclose(wannier.T, -wannier, atol=1e-12)
        for fragment in range(3):
            np.testing.assert_allclose(wannier[2*fragment:2*fragment+2,
                                               2*fragment:2*fragment+2], 0., atol=1e-12)

    def test_roundtrip_frobenius_metric_and_projection_adjoint(self):
        mapping = ActiveActiveRotationMap(_fourier_mo_phase(3, 2), [2]*3)
        rng = np.random.default_rng(27)
        x = rng.normal(size=mapping.nvar).astype(complex)
        rotation = mapping.unpack(x)
        self.assertTrue(np.issubdtype(rotation.dtype, np.complexfloating))
        np.testing.assert_allclose(rotation + rotation.conj().transpose(0, 2, 1),
                                   0., atol=1e-13)
        np.testing.assert_allclose(mapping.pack(rotation), x, atol=1e-12)
        self.assertAlmostEqual(np.vdot(rotation, rotation).real,
                               np.vdot(x, x).real, places=12)
        # Use a general matrix, not only anti-Hermitian inputs, to check the
        # full Frobenius adjoint required for future gradient projection.
        matrix = rng.normal(size=rotation.shape) + 1j*rng.normal(size=rotation.shape)
        self.assertAlmostEqual(np.vdot(rotation, matrix).real,
                               np.vdot(x, mapping.pack(matrix)).real, places=12)
        projected = mapping.unpack(mapping.pack(matrix))
        np.testing.assert_allclose(mapping.unpack(mapping.pack(projected)),
                                   projected, atol=1e-12)

    def test_constant_intra_fragment_rotations_are_removed(self):
        mapping = ActiveActiveRotationMap(_fourier_mo_phase(3, 2), [2]*3)
        constant = np.array([[.7j, .4+.8j], [-.4+.8j, -.2j]])
        redundant = np.broadcast_to(constant, (3, 2, 2))
        np.testing.assert_allclose(mapping.pack(redundant), 0., atol=1e-12)
        for column in range(mapping.nvar):
            direction = np.eye(mapping.nvar)[column]
            wannier = mapping.bloch_to_wannier(mapping.unpack(direction))
            for fragment in range(3):
                np.testing.assert_allclose(wannier[2*fragment:2*fragment+2,
                                                   2*fragment:2*fragment+2],
                                           0., atol=1e-12)

    def test_frozen_band_mask_includes_diagonals_and_excludes_frozen_motion(self):
        mask = np.broadcast_to(np.tril(np.ones((3, 3), dtype=bool)), (3, 3, 3)).copy()
        mask[:, 1, :] = False
        mask[:, :, 1] = False
        mapping = ActiveActiveRotationMap(_fourier_mo_phase(3, 3), [3]*3,
                                         bloch_pair_mask=mask)
        self.assertEqual(mapping.nvar, 8)
        rotation = mapping.unpack(np.arange(mapping.nvar))
        np.testing.assert_array_equal(rotation[:, 1, :], 0.)
        np.testing.assert_array_equal(rotation[:, :, 1], 0.)
        # A forbidden diagonal must project to zero too.
        frozen = np.zeros((3, 3, 3), dtype=complex)
        frozen[:, 1, 1] = 1j*np.array([0., 1., -1.])
        np.testing.assert_allclose(mapping.pack(frozen), 0.)

    def test_complex_gauge_and_unequal_fragments(self):
        rng = np.random.default_rng(91)
        # Arbitrary square unitary, with unequal Wannier fragments. Check the
        # actual quotient space; intra-fragment projection need not preserve k.
        z = rng.normal(size=(6, 6)) + 1j*rng.normal(size=(6, 6))
        unitary, _ = np.linalg.qr(z)
        mapping = ActiveActiveRotationMap(unitary.reshape(3, 2, 6), [1, 2, 3])
        matrices = [mapping.unpack(np.eye(mapping.nvar)[i]) for i in range(mapping.nvar)]
        flat = np.array(matrices).reshape(mapping.nvar, -1)
        np.testing.assert_allclose((flat.conj() @ flat.T).real,
                                   np.eye(mapping.nvar), atol=1e-12)
        for rotation in matrices:
            np.testing.assert_allclose(
                mapping.wannier_to_bloch(mapping.bloch_to_wannier(rotation)),
                rotation, atol=1e-12)
        # Global phase is redundant for any gauge and fragment partition.
        np.testing.assert_allclose(mapping.pack(1j*np.broadcast_to(np.eye(2), (3, 2, 2))),
                                   0., atol=1e-12)
        x = rng.normal(size=mapping.nvar)
        np.testing.assert_allclose(mapping.pack(mapping.unpack(x)), x, atol=1e-12)

    def test_redundancy_for_unequal_fragments_with_identity_gauge(self):
        mapping = ActiveActiveRotationMap(np.eye(4).reshape(2, 2, 4), [1, 3])
        # Only the two real components connecting orbitals 0 and 1 survive.
        # The complete second k-block lies within the larger fragment.
        self.assertEqual(mapping.nvar, 2)
        rotation = mapping.unpack([.3, -.7])
        np.testing.assert_allclose(rotation[1], 0., atol=1e-12)
        np.testing.assert_allclose(np.diagonal(rotation, axis1=1, axis2=2), 0., atol=1e-12)
        wannier = mapping.bloch_to_wannier(rotation)
        np.testing.assert_allclose(wannier[1:, 1:], 0., atol=1e-12)

    def test_bloch_wannier_projection(self):
        mapping = ActiveActiveRotationMap(_fourier_mo_phase(3, 2), [2]*3)
        rng = np.random.default_rng(12)
        arbitrary = rng.normal(size=(6, 6)) + 1j*rng.normal(size=(6, 6))
        projected = mapping.bloch_to_wannier(mapping.wannier_to_bloch(arbitrary))
        np.testing.assert_allclose(
            mapping.bloch_to_wannier(mapping.wannier_to_bloch(projected)),
            projected, atol=1e-12)

    def test_rank_and_coordinates_survive_roundoff(self):
        phase = _fourier_mo_phase(3, 5)
        rng = np.random.default_rng(23)
        noisy = phase + 2e-13*(rng.normal(size=phase.shape) + 1j*rng.normal(size=phase.shape))
        reference = ActiveActiveRotationMap(phase, [5]*3)
        perturbed = ActiveActiveRotationMap(noisy, [5]*3)
        self.assertEqual(reference.nvar, 50)
        self.assertEqual(perturbed.nvar, 50)
        np.testing.assert_allclose(perturbed.basis, reference.basis, atol=1e-10)
        np.testing.assert_allclose(perturbed.basis.T @ perturbed.basis, np.eye(50), atol=1e-12)
        # Every singular value is at most one in the Frobenius metric.
        self.assertEqual(ActiveActiveRotationMap(phase, [5]*3, svd_tol=1.1).nvar, 0)

    def test_empty_spaces(self):
        for nk, sizes in [(1, [1]), (3, [3])]:
            mapping = ActiveActiveRotationMap(_fourier_mo_phase(nk, 1), sizes)
            self.assertEqual(mapping.nvar, 0)
            np.testing.assert_array_equal(mapping.unpack([]), np.zeros((nk, 1, 1)))
        mapping = ActiveActiveRotationMap(_fourier_mo_phase(3, 2), [2]*3,
                                         bloch_pair_mask=np.zeros((3, 2, 2), dtype=bool))
        self.assertEqual(mapping.nvar, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
