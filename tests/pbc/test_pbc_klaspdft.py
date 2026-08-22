#!/usr/bin/env python

"""Unit tests for periodic kLAS-PDFT adapters."""

import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from mrh.my_pyscf.pbc.mcpdft import klaspdft_helper


class _RecordingFragmentSolver:
    """Small fragment solver returning prescribed complex density matrices."""

    def __init__(self, dm1a, dm1b, dm2, spin):
        self.dm1a = np.asarray(dm1a, dtype=np.complex128)
        self.dm1b = np.asarray(dm1b, dtype=np.complex128)
        self.dm2 = np.asarray(dm2, dtype=np.complex128)
        self.spin = spin
        self.seen_ci = []

    def make_rdm1s(self, ci, norb, nelec):
        self.seen_ci.append(("dm1", ci, norb, tuple(nelec)))
        return self.dm1a, self.dm1b

    def make_rdm2(self, ci, norb, nelec):
        self.seen_ci.append(("dm2", ci, norb, tuple(nelec)))
        return self.dm2


def _make_fake_klas():
    roots = []
    for iroot in range(2):
        shift = 0.1 * iroot
        frag0 = _RecordingFragmentSolver(
            [[0.75 + shift]], [[0.25 - shift]], [[[[0.2 + shift]]]],
            spin=1,
        )
        frag1 = _RecordingFragmentSolver(
            [[0.4 - shift]], [[0.6 + shift]], [[[[0.3 - shift]]]],
            spin=-1,
        )
        roots.append((frag0, frag1))
    return SimpleNamespace(
        nroots=2,
        ncas_sub=np.asarray([1, 1]),
        nelecas_sub=np.asarray([[1, 0], [0, 1]]),
        fciboxes=[
            SimpleNamespace(fcisolvers=[roots[0][0], roots[1][0]]),
            SimpleNamespace(fcisolvers=[roots[0][1], roots[1][1]]),
        ],
        ci=[[np.asarray([[0.0]]), np.asarray([[1.0]])],
            [np.asarray([[10.0]]), np.asarray([[11.0]])]],
        stdout=None,
        verbose=0,
    )


class KLASPDFTRDMTests(unittest.TestCase):

    def test_context_selects_one_root_from_every_fragment(self):
        klas = _make_fake_klas()
        solvers, ci, ncas_sub, nelecas_sub = \
            klaspdft_helper._get_klas_rdm_context(klas, state=1)

        self.assertIs(solvers[0], klas.fciboxes[0].fcisolvers[1])
        self.assertIs(solvers[1], klas.fciboxes[1].fcisolvers[1])
        np.testing.assert_array_equal(ci[0], [[1.0]])
        np.testing.assert_array_equal(ci[1], [[11.0]])
        np.testing.assert_array_equal(ncas_sub, [1, 1])
        np.testing.assert_array_equal(nelecas_sub, [[1, 0], [0, 1]])

    def test_product_state_rdms_are_complex_and_have_full_active_shape(self):
        klas = _make_fake_klas()
        casdm1s, casdm2 = klaspdft_helper.make_one_casdm12_klas(
            klas, state=0,
        )

        self.assertEqual(casdm1s.shape, (2, 2, 2))
        self.assertEqual(casdm2.shape, (2, 2, 2, 2))
        self.assertTrue(np.issubdtype(casdm1s.dtype, np.complexfloating))
        self.assertTrue(np.issubdtype(casdm2.dtype, np.complexfloating))
        np.testing.assert_allclose(casdm1s[0], np.diag([0.75, 0.4]))
        np.testing.assert_allclose(casdm1s[1], np.diag([0.25, 0.6]))
        self.assertAlmostEqual(casdm2[0, 0, 0, 0], 0.2)
        self.assertAlmostEqual(casdm2[1, 1, 1, 1], 0.3)
        self.assertAlmostEqual(casdm2[0, 0, 1, 1], 1.0)
        self.assertAlmostEqual(casdm2[1, 1, 0, 0], 1.0)
        self.assertAlmostEqual(casdm2[0, 1, 1, 0], -0.45)
        self.assertAlmostEqual(casdm2[1, 0, 0, 1], -0.45)

    def test_rdm_builder_passes_the_selected_ci_to_fragment_solvers(self):
        klas = _make_fake_klas()
        klaspdft_helper.make_one_casdm12_klas(klas, state=1)

        for ifrag in range(2):
            solver = klas.fciboxes[ifrag].fcisolvers[1]
            self.assertEqual(len(solver.seen_ci), 3)
            self.assertTrue(all(
                np.array_equal(item[1], [[10.0 * ifrag + 1.0]])
                for item in solver.seen_ci
            ))

    def test_invalid_state_is_rejected(self):
        klas = _make_fake_klas()
        with self.assertRaisesRegex(TypeError, "state must be an integer"):
            klaspdft_helper.make_one_casdm12_klas(klas, state=0.5)
        with self.assertRaisesRegex(ValueError, "state must lie"):
            klaspdft_helper.make_one_casdm12_klas(klas, state=2)

    def test_missing_fragment_ci_is_rejected(self):
        klas = _make_fake_klas()
        klas.ci[1][0] = None
        with self.assertRaisesRegex(ValueError, "Fragment 1 CI vector"):
            klaspdft_helper.make_one_casdm12_klas(klas, state=0)


class KLASPDFTPhaseTests(unittest.TestCase):

    @staticmethod
    def _make_phase_context():
        return SimpleNamespace(
            _scf=object(),
            kmesh=(2, 1, 1),
            kpts=np.zeros((2, 3)),
            ncore=1,
            ncas=1,
            ncas_sub=np.asarray([1, 1]),
            mo_coeff=np.asarray([
                [[10.0, 1.0, 20.0], [30.0, 2.0, 40.0]],
                [[50.0, 3.0, 60.0], [70.0, 4.0, 80.0]],
            ], dtype=np.complex128),
        )

    def test_phase_uses_the_kLAS_wannier_active_orbitals(self):
        klas = self._make_phase_context()
        mo_phase = np.asarray([
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ], dtype=np.complex128)

        with mock.patch.object(
                klaspdft_helper, "get_wannier_orbs",
                return_value=("wannier", "indices", mo_phase)) as get_phase:
            result = klaspdft_helper.get_klas_mo_phase(klas)

        np.testing.assert_array_equal(result, mo_phase)
        self.assertIs(get_phase.call_args.args[0], klas._scf)
        self.assertEqual(get_phase.call_args.args[1], klas.kmesh)
        np.testing.assert_array_equal(
            get_phase.call_args.args[2],
            klas.mo_coeff[:, :, 1:2],
        )

    def test_nonunitary_phase_is_rejected(self):
        klas = self._make_phase_context()
        bad_phase = np.ones((2, 1, 2), dtype=np.complex128)
        with mock.patch.object(
                klaspdft_helper, "get_wannier_orbs",
                return_value=(None, None, bad_phase)):
            with self.assertRaisesRegex(ValueError, "must be unitary"):
                klaspdft_helper.get_klas_mo_phase(klas)

    def test_phase_dimensions_are_validated_before_wannierization(self):
        klas = self._make_phase_context()
        klas.ncas_sub = np.asarray([1])
        with mock.patch.object(
                klaspdft_helper, "get_wannier_orbs") as get_phase:
            with self.assertRaisesRegex(ValueError, r"sum\(ncas_sub\)"):
                klaspdft_helper.get_klas_mo_phase(klas)
        get_phase.assert_not_called()


def _make_kconserv(nkpts):
    """Return a cyclic momentum-conservation table for test meshes."""
    return np.fromfunction(
        lambda k1, k2, k3: (k1 - k2 + k3) % nkpts,
        (nkpts, nkpts, nkpts),
        dtype=int,
    ).astype(int)


class KLASPDFTKBlockTests(unittest.TestCase):

    def test_wannier_rdms_are_transformed_to_expected_k_blocks(self):
        rng = np.random.default_rng(24)
        nkpts, ncas = 2, 2
        ncastot = nkpts * ncas
        phase_matrix = np.linalg.qr(
            rng.normal(size=(ncastot, ncastot))
            + 1j * rng.normal(size=(ncastot, ncastot)),
        )[0]
        mo_phase = phase_matrix.reshape(nkpts, ncas, ncastot)
        casdm1s = (
            rng.normal(size=(2, ncastot, ncastot))
            + 1j * rng.normal(size=(2, ncastot, ncastot))
        )
        casdm1s += casdm1s.swapaxes(-1, -2).conj()
        casdm2 = (
            rng.normal(size=(ncastot,) * 4)
            + 1j * rng.normal(size=(ncastot,) * 4)
        )
        kconserv = _make_kconserv(nkpts)

        casdm1s_kpts, cascm2_kpts = \
            klaspdft_helper.make_klas_rdms_kpts(
                casdm1s, casdm2, mo_phase, kconserv,
            )

        self.assertEqual(casdm1s_kpts.shape, (2, 2, 2, 2))
        self.assertEqual(cascm2_kpts.shape, (2, 2, 2, 2, 2, 2, 2))
        expected_dm1s = np.stack([
            np.stack([
                mo_phase[k] @ dm1 @ mo_phase[k].conj().T
                for k in range(nkpts)
            ])
            for dm1 in casdm1s
        ])
        np.testing.assert_allclose(casdm1s_kpts, expected_dm1s)

        cascm2 = klaspdft_helper.dm2_cumulant_complex(casdm2, casdm1s)
        for k1 in range(nkpts):
            for k2 in range(nkpts):
                for k3 in range(nkpts):
                    k4 = kconserv[k1, k2, k3]
                    expected = np.einsum(
                        "ap,bq,pqrs,cr,ds->abcd",
                        mo_phase[k1].conj(),
                        mo_phase[k2],
                        cascm2,
                        mo_phase[k3].conj(),
                        mo_phase[k4],
                        optimize=True,
                    )
                    np.testing.assert_allclose(
                        cascm2_kpts[k1, k2, k3], expected,
                    )

    def test_k_blocks_are_invariant_to_wannier_gauge_rotation(self):
        rng = np.random.default_rng(91)
        nkpts, ncas = 2, 1
        ncastot = nkpts * ncas
        phase = np.linalg.qr(
            rng.normal(size=(ncastot, ncastot))
            + 1j * rng.normal(size=(ncastot, ncastot)),
        )[0]
        gauge = np.linalg.qr(
            rng.normal(size=(ncastot, ncastot))
            + 1j * rng.normal(size=(ncastot, ncastot)),
        )[0]
        dm1s = (
            rng.normal(size=(2, ncastot, ncastot))
            + 1j * rng.normal(size=(2, ncastot, ncastot))
        )
        dm1s += dm1s.swapaxes(-1, -2).conj()
        dm2 = (
            rng.normal(size=(ncastot,) * 4)
            + 1j * rng.normal(size=(ncastot,) * 4)
        )
        kconserv = _make_kconserv(nkpts)

        reference = klaspdft_helper.make_klas_rdms_kpts(
            dm1s, dm2, phase.reshape(nkpts, ncas, ncastot), kconserv,
        )
        dm1s_rot = np.einsum(
            "pi,spq,qj->sij",
            gauge, dm1s, gauge.conj(),
            optimize=True,
        )
        dm2_rot = np.einsum(
            "pi,qj,pqrs,rk,sl->ijkl",
            gauge.conj(), gauge, dm2, gauge.conj(), gauge,
            optimize=True,
        )
        phase_rot = (phase @ gauge.conj()).reshape(
            nkpts, ncas, ncastot,
        )
        rotated = klaspdft_helper.make_klas_rdms_kpts(
            dm1s_rot, dm2_rot, phase_rot, kconserv,
        )

        np.testing.assert_allclose(rotated[0], reference[0], atol=1e-11)
        np.testing.assert_allclose(rotated[1], reference[1], atol=1e-11)

    def test_k_block_layout_is_validated(self):
        casdm1s = np.zeros((2, 2, 2))
        casdm2 = np.zeros((2, 2, 2, 2))
        mo_phase = np.eye(2).reshape(2, 1, 2)
        with self.assertRaisesRegex(ValueError, "kconserv shape"):
            klaspdft_helper.make_klas_rdms_kpts(
                casdm1s, casdm2, mo_phase, np.zeros((2, 2), dtype=int),
            )


if __name__ == "__main__":
    unittest.main()
