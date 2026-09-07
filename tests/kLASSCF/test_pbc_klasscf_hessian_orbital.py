import unittest
from unittest.mock import patch

import numpy as np
from scipy import linalg

from mrh.my_pyscf.pbc.mcscf import klasscf
from mrh.my_pyscf.pbc.mcscf.klasscf import KLASSCF_HessianOperator


class _OrbitalUGG:

    pairs = ((1, 0), (2, 0), (2, 1))
    nvar_orb = len(pairs)

    def unpack_orb(self, vector):
        vector = np.asarray(vector)
        kappa = np.zeros((1, 3, 3), dtype=np.result_type(vector, complex))
        for value, (p, q) in zip(vector, self.pairs):
            kappa[0, p, q] = value
            kappa[0, q, p] = -value.conjugate()
        return kappa

    def pack_orb(self, kappa):
        return np.asarray([kappa[0, p, q] for p, q in self.pairs])


class _ExternalUGG:

    def __init__(self, nkpts):
        self.uniq_orb_idx = (
            np.repeat(np.arange(nkpts), 3),
            np.tile(np.array([1, 2, 2]), nkpts),
            np.tile(np.array([0, 0, 1]), nkpts),
        )
        self.nvar_orb_external = 3 * nkpts


class _ExternalERIs:

    def __init__(self, rng, nkpts, nmo, ncore):
        self.vhf_c = _random_hermitian(rng, (nkpts, nmo, nmo))
        self.j_pc = _random_complex(rng, (nkpts, nmo, ncore))
        self.k_pc = _random_complex(rng, (nkpts, nmo, ncore))
        self.ppaa_blocks = {
            key: _random_complex(rng, (nmo, nmo, 1, 1))
            for key in np.ndindex((nkpts,) * 3)
        }
        self.papa_blocks = {
            key: _random_complex(rng, (nmo, 1, nmo, 1))
            for key in np.ndindex((nkpts,) * 3)
        }
        self.paap_blocks = {
            key: _random_complex(rng, (nmo, 1, 1, nmo))
            for key in np.ndindex((nkpts,) * 3)
        }

    def ppaa(self, k1, k2, k3):
        return self.ppaa_blocks[k1, k2, k3]

    def papa(self, k1, k2, k3):
        return self.papa_blocks[k1, k2, k3]

    def paap(self, k1, k2, k3):
        return self.paap_blocks[k1, k2, k3]


def _random_complex(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _random_hermitian(rng, shape):
    matrix = _random_complex(rng, shape)
    return matrix + matrix.conj().swapaxes(-1, -2)


def _make_external_operator():
    rng = np.random.default_rng(149)
    operator = KLASSCF_HessianOperator.__new__(KLASSCF_HessianOperator)
    operator.nkpts = 2
    operator.nmo = 3
    operator.ncore = 1
    operator.ncas = 1
    operator.nocc = 2
    operator.kpts = np.zeros((2, 3))
    operator.las = type("LAS", (), {
        "_scf": type("SCF", (), {"cell": object()})(),
    })()
    operator.hcore = _random_hermitian(rng, (2, 3, 3))
    operator.h1s = _random_hermitian(rng, (2, 2, 3, 3))
    operator.dm1s = _random_hermitian(rng, (2, 2, 3, 3))
    operator.fock1 = _random_hermitian(rng, (2, 3, 3))
    operator.casdm2 = _random_complex(rng, (2,) * 4)
    operator.mo_phase = _random_complex(rng, (2, 1, 2))
    operator.eris = _ExternalERIs(rng, 2, 3, 1)
    operator.ugg = _ExternalUGG(2)
    operator._Horb_diag_external_cache = None
    dm2_blocks = {
        key: _random_complex(rng, (1,) * 4)
        for key in np.ndindex((2,) * 3)
    }
    return operator, dm2_blocks


def _external_diagonal_reference(operator, dm2_blocks):
    nkpts = operator.nkpts
    nmo = operator.nmo
    ncore = operator.ncore
    nocc = operator.nocc
    active_orbital = ncore
    dm1 = operator.dm1s.sum(axis=0)
    casdm1 = dm1[:, active_orbital, active_orbital]
    vhf_ca = (operator.h1s[0] + operator.h1s[1]) / 2.0 - operator.hcore

    jkcaa = np.zeros((nkpts, nocc), dtype=np.complex128)
    for k in range(nkpts):
        ppaa = operator.eris.ppaa_blocks[k, k, k]
        paap = operator.eris.paap_blocks[k, k, k]
        papa = operator.eris.papa_blocks[k, k, k]
        for p in range(nocc):
            bra_ket_pair = (
                -2.0 * ppaa[p, p, 0, 0] * casdm1[k]
                + 4.0 * paap[p, 0, 0, p] * casdm1[k]
            )
            jkcaa[k, p] = (
                bra_ket_pair.real
                + 2.0 * papa[p, 0, p, 0] * casdm1[k]
            )

    hdm2 = np.zeros((nkpts, nmo), dtype=np.complex128)
    for k in range(nkpts):
        for kw in range(nkpts):
            papa = operator.eris.papa_blocks[k, kw, k]
            ppaa = operator.eris.ppaa_blocks[k, k, kw]
            paap = operator.eris.paap_blocks[k, kw, kw]
            for p in range(nmo):
                hdm2[k, p] += (
                    papa[p, 0, p, 0] * dm2_blocks[k, kw, k].item()
                    + (
                        ppaa[p, p, 0, 0]
                        * dm2_blocks[kw, kw, k].item()
                    ).conjugate()
                    + (
                        paap[p, 0, 0, p]
                        * dm2_blocks[k, kw, kw].item()
                    ).conjugate()
                )

    hdiag = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    for k in range(nkpts):
        fock_diag = operator.fock1[k].diagonal().real
        potential_diag = vhf_ca[k].diagonal().real
        core_eri = 6.0 * operator.eris.k_pc[k] - 2.0 * operator.eris.j_pc[k]
        active_active = (
            -operator.eris.vhf_c[k, active_orbital, active_orbital]
            * casdm1[k]
        )
        for p in range(nmo):
            for q in range(nmo):
                one_body_pq = (
                    operator.hcore[k, p, p] * dm1[k, q, q]
                    - operator.hcore[k, p, q] * dm1[k, p, q]
                )
                one_body_qp = (
                    operator.hcore[k, q, q] * dm1[k, p, p]
                    - operator.hcore[k, q, p] * dm1[k, q, p]
                )
                value = one_body_pq + one_body_qp.conjugate()
                value -= fock_diag[p] + fock_diag[q]
                if p == q:
                    value += 2.0 * fock_diag[p]
                if q < ncore:
                    value += 2.0 * potential_diag[p]
                if p < ncore:
                    value += 2.0 * potential_diag[q]
                if p == q and p < ncore:
                    value -= 4.0 * potential_diag[p]
                if q == active_orbital:
                    value += operator.eris.vhf_c[k, p, p] * casdm1[k]
                if p == active_orbital:
                    value += (
                        operator.eris.vhf_c[k, q, q] * casdm1[k]
                    ).conjugate()
                if p == q == active_orbital:
                    value += active_active + active_active.conjugate()
                if p >= ncore and q < ncore:
                    value += core_eri[p, q]
                if p < ncore and q >= ncore:
                    value += core_eri[q, p].conjugate()
                if p < nocc and q == active_orbital:
                    value -= jkcaa[k, p]
                if p == active_orbital and q < nocc:
                    value -= jkcaa[k, q].conjugate()
                if q == active_orbital:
                    value += hdm2[k, p]
                if p == active_orbital:
                    value += hdm2[k, q].conjugate()
                hdiag[k, p, q] = value

    return hdiag[operator.ugg.uniq_orb_idx] / 2.0


class KnownValues(unittest.TestCase):

    def test_periodic_orbital_update_uses_half_generator_per_kpoint(self):
        rng = np.random.default_rng(311)
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.nkpts = 3
        operator.nao = 4
        operator.nmo = 3
        operator.mo_coeff = np.asarray([
            np.linalg.qr(
                rng.standard_normal((4, 3))
                + 1j * rng.standard_normal((4, 3))
            )[0]
            for _ in range(operator.nkpts)
        ])
        mo0 = np.array(operator.mo_coeff, copy=True)
        kappa = (
            rng.standard_normal((3, 3, 3))
            + 1j * rng.standard_normal((3, 3, 3))
        )
        kappa = kappa - kappa.conj().transpose(0, 2, 1)
        kappa *= 0.2 / np.linalg.norm(kappa)

        mo1 = operator._update_mo(kappa)

        expected = np.asarray([
            mo_k @ linalg.expm(kappa_k / 2.0)
            for mo_k, kappa_k in zip(mo0, kappa)
        ])
        np.testing.assert_allclose(mo1, expected, atol=2e-14, rtol=2e-14)
        np.testing.assert_array_equal(operator.mo_coeff, mo0)
        for k in range(operator.nkpts):
            np.testing.assert_allclose(
                mo1[k].conj().T @ mo1[k],
                mo0[k].conj().T @ mo0[k],
                atol=2e-13,
                rtol=2e-13,
            )

    def test_periodic_orbital_update_rejects_invalid_generators(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.nkpts = 1
        operator.nao = 2
        operator.nmo = 2
        operator.mo_coeff = np.eye(2, dtype=np.complex128)[None]

        with self.assertRaisesRegex(ValueError, "kappa has shape"):
            operator._update_mo(np.zeros((2, 2)))

        nonfinite = np.zeros((1, 2, 2), dtype=np.complex128)
        nonfinite[0, 1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "only finite values"):
            operator._update_mo(nonfinite)

        nonantihermitian = np.zeros((1, 2, 2), dtype=np.complex128)
        nonantihermitian[0, 1, 0] = 0.1 + 0.2j
        with self.assertRaisesRegex(ValueError, "must be anti-Hermitian"):
            operator._update_mo(nonantihermitian)

    def test_orbital_hdiag_matches_unit_matvecs_and_is_cached(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = _OrbitalUGG()
        operator._Horb_diag_matvec_cache = None
        hessian = np.array([
            [3.0, 0.2 - 0.1j, -0.4j],
            [0.7 + 0.3j, -1.5, 0.6],
            [0.1, -0.2j, 2.25],
        ], dtype=np.complex128)
        calls = []

        def orbital_response(kappa):
            direction = operator.ugg.pack_orb(kappa)
            calls.append(np.array(direction, copy=True))
            return 2.0 * operator.ugg.unpack_orb(hessian @ direction)

        operator._orbital_hessian_response = orbital_response

        diagonal = operator._get_Horb_diag_matvec()

        np.testing.assert_allclose(diagonal, np.diag(hessian))
        np.testing.assert_allclose(calls, np.eye(3))
        self.assertEqual(len(calls), operator.ugg.nvar_orb)

        diagonal[0] = 999.0
        cached = operator._get_Horb_diag_matvec()
        np.testing.assert_allclose(cached, np.diag(hessian))
        self.assertEqual(len(calls), operator.ugg.nvar_orb)

    def test_external_hdiag_matches_explicit_momentum_resolved_formula(self):
        operator, dm2_blocks = _make_external_operator()
        kconserv = np.fromfunction(
            lambda k1, k2, k3: (k1 - k2 + k3) % operator.nkpts,
            (operator.nkpts,) * 3, dtype=int,
        ).astype(int)
        transform_calls = []

        def transform(casdm2, mo_phase, klabel):
            self.assertIs(casdm2, operator.casdm2)
            self.assertIs(mo_phase, operator.mo_phase)
            klabel = tuple(klabel)
            transform_calls.append(klabel)
            return dm2_blocks[klabel[:3]]

        with patch.object(
                klasscf.kpts_helper, "get_kconserv",
                return_value=kconserv), patch.object(
                klasscf, "_get_casdm2_kpts", side_effect=transform):
            actual = operator._get_Horb_diag_external()

        expected = _external_diagonal_reference(operator, dm2_blocks)
        np.testing.assert_allclose(actual, expected)
        self.assertCountEqual(
            [call[:3] for call in transform_calls],
            list(np.ndindex((operator.nkpts,) * 3)),
        )
        for k1, k2, k3, k4 in transform_calls:
            self.assertEqual(k4, kconserv[k1, k2, k3])

        actual[0] = 999.0
        cached = operator._get_Horb_diag_external()
        np.testing.assert_allclose(cached, expected)
        self.assertEqual(len(transform_calls), operator.nkpts ** 3)

    def test_external_hdiag_rejects_core_eri_shape_mismatch(self):
        operator, _ = _make_external_operator()
        operator.eris.j_pc = np.zeros((2, 3, 2))

        with self.assertRaisesRegex(ValueError, "j_pc has shape"):
            operator._get_Horb_diag_external()

    def test_active_active_hessian_keeps_direct_and_conjugate_blocks(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("UGG", (), {
            "nvar_orb_active_active": 2,
        })()
        operator.mo_coeff = np.zeros((1, 1, 1), dtype=np.complex128)
        operator._Horb_active_active_cache = None
        direct = np.array([
            [1.2 + 0.1j, -0.3j],
            [0.4 - 0.2j, 0.7],
        ])
        conjugate = np.array([
            [0.25, 0.1 + 0.15j],
            [-0.2j, -0.35 + 0.05j],
        ])
        calls = []

        def apply(coordinates):
            calls.append(np.array(coordinates, copy=True))
            return (
                direct @ coordinates
                + conjugate @ coordinates.conj()
            )

        operator._apply_Horb_active_active = apply

        actual_direct, actual_conjugate = (
            operator._get_Horb_active_active()
        )

        np.testing.assert_allclose(actual_direct, direct)
        np.testing.assert_allclose(actual_conjugate, conjugate)
        np.testing.assert_allclose(calls, [
            [1.0, 0.0], [1.0j, 0.0],
            [0.0, 1.0], [0.0, 1.0j],
        ])
        probe = np.array([0.2 - 0.4j, -0.1 + 0.3j])
        np.testing.assert_allclose(
            actual_direct @ probe + actual_conjugate @ probe.conj(),
            direct @ probe + conjugate @ probe.conj(),
        )

        actual_direct[0, 0] = 999.0
        cached_direct, cached_conjugate = (
            operator._get_Horb_active_active()
        )
        np.testing.assert_allclose(cached_direct, direct)
        np.testing.assert_allclose(cached_conjugate, conjugate)
        self.assertEqual(len(calls), 4)

    def test_active_active_hessian_handles_empty_projected_space(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("UGG", (), {
            "nvar_orb_active_active": 0,
        })()
        operator.mo_coeff = np.zeros((1, 1, 1))
        operator._Horb_active_active_cache = None
        operator._apply_Horb_active_active = lambda vector: self.fail(
            "active-active response should not be evaluated"
        )

        direct, conjugate = operator._get_Horb_active_active()

        self.assertEqual(direct.shape, (0, 0))
        self.assertEqual(conjugate.shape, (0, 0))
        self.assertEqual(direct.dtype, np.dtype(np.complex128))
        self.assertEqual(conjugate.dtype, np.dtype(np.complex128))

    def test_horb_diag_combines_external_and_active_slices(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("UGG", (), {
            "nvar_orb": 4,
            "nvar_orb_external": 2,
            "nvar_orb_active_active": 2,
        })()
        operator._get_Horb_diag_external = lambda: np.array(
            [1.0, 2.0], dtype=np.complex128,
        )
        direct = np.array([
            [3.0, 0.2j],
            [-0.1j, 4.0],
        ])
        conjugate = np.array([
            [0.5, 0.1],
            [0.2, -0.25],
        ])
        operator._get_Horb_active_active = lambda: (direct, conjugate)

        diagonal = operator._get_Horb_diag()

        np.testing.assert_allclose(diagonal, [1.0, 2.0, 3.5, 3.75])

    def test_horb_diag_skips_an_empty_active_active_slice(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("UGG", (), {
            "nvar_orb": 2,
            "nvar_orb_external": 2,
            "nvar_orb_active_active": 0,
        })()
        external = np.array([1.5, -0.25], dtype=np.complex128)
        operator._get_Horb_diag_external = lambda: external
        operator._get_Horb_active_active = lambda: self.fail(
            "active-active diagonal should not be evaluated"
        )

        diagonal = operator._get_Horb_diag()

        np.testing.assert_allclose(diagonal, external)

    def test_horb_diag_rejects_combined_layout_mismatch(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("UGG", (), {
            "nvar_orb": 3,
            "nvar_orb_active_active": 0,
        })()
        operator._get_Horb_diag_external = lambda: np.array([1.0, 2.0])

        with self.assertRaisesRegex(
                ValueError, "orbital_hessian_diagonal has shape"):
            operator._get_Horb_diag()

    def test_orbital_hdiag_handles_an_empty_orbital_space(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("EmptyUGG", (), {"nvar_orb": 0})()
        operator._Horb_diag_matvec_cache = None
        operator._orbital_hessian_response = lambda kappa: self.fail(
            "orbital response should not be evaluated"
        )

        diagonal = operator._get_Horb_diag_matvec()

        self.assertEqual(diagonal.shape, (0,))
        self.assertEqual(diagonal.dtype, np.dtype(np.complex128))
        np.testing.assert_array_equal(
            operator._Horb_diag_matvec_cache, diagonal,
        )

    def test_orbital_hdiag_rejects_response_shape_mismatch(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = _OrbitalUGG()
        operator._Horb_diag_matvec_cache = None
        operator._orbital_hessian_response = lambda kappa: np.zeros((2, 2))

        with self.assertRaisesRegex(
                ValueError, r"orbital_hessian_response\[0\] has shape"):
            operator._get_Horb_diag_matvec()


    def test_complex_orbital_step_builds_one_sided_density_responses(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.nkpts = 2
        operator.nmo = 3
        operator.ncas = 1
        operator.ncore = 1
        operator.nocc = 2
        operator.kpts = np.zeros((2, 3))
        operator.las = type("LAS", (), {
            "_scf": type("SCF", (), {"cell": object()})(),
        })()
        operator.mo_phase = np.ones((2, 1, 2), dtype=np.complex128)
        operator.cascm2 = np.ones((2,) * 4, dtype=np.complex128)
        operator.dm1s = np.arange(36, dtype=float).reshape(2, 2, 3, 3)
        operator.dm1s = operator.dm1s.astype(np.complex128)
        operator.dm1s += 0.2j * operator.dm1s.transpose(0, 1, 3, 2)

        kappa = np.zeros((2, 3, 3), dtype=np.complex128)
        kappa[:, 1, 0] = [0.3 + 0.2j, -0.1 + 0.4j]
        kappa[:, 0, 1] = -kappa[:, 1, 0].conj()
        kappa[:, 2, 1] = [-0.2j, 0.25 - 0.15j]
        kappa[:, 1, 2] = -kappa[:, 2, 1].conj()
        kconserv = np.fromfunction(
            lambda k1, k2, k3: (k1 - k2 + k3) % 2,
            (2, 2, 2), dtype=int,
        ).astype(int)

        def transform(cumulant, phase, klabel):
            value = 1.0 + klabel[0] + 2 * klabel[1] + 4 * klabel[2]
            return np.full((1,) * 4, value + 0.5j)

        with patch.object(
                klasscf.kpts_helper, "get_kconserv",
                return_value=kconserv), patch.object(
                klasscf, "_get_casdm2_kpts", side_effect=transform):
            odm1s, ocm2 = operator._make_orbital_response_dm(kappa)

        np.testing.assert_allclose(
            odm1s,
            -np.einsum("skpr,krq->skpq", operator.dm1s, kappa),
        )
        for k1, k2, k3 in klasscf.kpts_helper.loop_kkk(2):
            k4 = kconserv[k1, k2, k3]
            value = 1.0 + k1 + 2 * k2 + 4 * k3 + 0.5j
            np.testing.assert_allclose(
                ocm2[k1, k2, k3, 0, 0, 0],
                -value * kappa[k4, 1],
            )

    def test_jk_response_uses_hermitian_complex_ao_density(self):
        captured = {}

        class FakeLAS:
            _scf = type("SCF", (), {"cell": object()})()

            @staticmethod
            def get_veff(cell, dm_kpts=None, hermi=None, kpts=None):
                captured["hermi"] = hermi
                captured["kpts"] = np.array(kpts, copy=True)
                captured["dm_kpts"] = np.array(dm_kpts, copy=True)
                return (1.7 - 0.2j) * dm_kpts

        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.las = FakeLAS()
        operator.nkpts = 1
        operator.nao = 2
        operator.nmo = 2
        operator.kpts = np.zeros((1, 3))
        operator.mo_coeff = np.array([[
            [1.0, 0.2j],
            [0.1 - 0.3j, 0.9],
        ]], dtype=np.complex128)
        odm1s = np.array([[[
            [0.2 + 0.1j, -0.3j],
            [0.4 + 0.2j, -0.1],
        ]], [[
            [-0.2j, 0.15 + 0.05j],
            [-0.25j, 0.3 - 0.1j],
        ]]], dtype=np.complex128)

        actual = operator._get_veff_response(odm1s)

        edm1s = odm1s + odm1s.conj().transpose(0, 1, 3, 2)
        mo = operator.mo_coeff[0]
        expected_dm_ao = mo @ edm1s[:, 0] @ mo.conj().T
        expected_veff = (
            mo.conj().T @ ((1.7 - 0.2j) * expected_dm_ao) @ mo
        )
        np.testing.assert_allclose(captured["dm_kpts"][:, 0], expected_dm_ao)
        self.assertEqual(captured["hermi"], 1)
        np.testing.assert_array_equal(captured["kpts"], operator.kpts)
        np.testing.assert_allclose(
            captured["dm_kpts"],
            captured["dm_kpts"].conj().transpose(0, 1, 3, 2),
        )
        np.testing.assert_allclose(actual[:, 0], expected_veff)

        captured.clear()
        actual_ci = operator._get_ci_veff_response(edm1s)
        np.testing.assert_allclose(captured["dm_kpts"][:, 0], expected_dm_ao)
        self.assertEqual(captured["hermi"], 1)
        np.testing.assert_array_equal(captured["kpts"], operator.kpts)
        np.testing.assert_allclose(actual_ci[:, 0], expected_veff)

    def test_orbital_ci_response_assembles_all_terms_and_normalization(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.nkpts = 2
        operator.nmo = 3
        rng = np.random.default_rng(101)
        operator.h1s = (
            rng.standard_normal((2, 2, 3, 3))
            + 1j * rng.standard_normal((2, 2, 3, 3))
        )
        operator.dm1s = (
            rng.standard_normal((2, 2, 3, 3))
            + 1j * rng.standard_normal((2, 2, 3, 3))
        )
        tdm1s_block = (
            rng.standard_normal((2, 2, 3, 3))
            + 1j * rng.standard_normal((2, 2, 3, 3))
        )
        veff_ci = (
            rng.standard_normal((2, 2, 3, 3))
            + 1j * rng.standard_normal((2, 2, 3, 3))
        )
        cumulant_fock = (
            rng.standard_normal((2, 3, 3))
            + 1j * rng.standard_normal((2, 3, 3))
        )
        tdm1rs = np.array([17.0])
        tcm2 = np.array([23.0])
        calls = []

        def transform_dm1(actual):
            np.testing.assert_array_equal(actual, tdm1rs)
            calls.append("dm1")
            return tdm1s_block

        def respond_jk(actual):
            np.testing.assert_array_equal(actual, tdm1s_block)
            calls.append("jk")
            return veff_ci

        def contract_cumulant(actual):
            np.testing.assert_array_equal(actual, tcm2)
            calls.append("cumulant")
            return cumulant_fock

        operator._transition_dm1s_to_block = transform_dm1
        operator._get_ci_veff_response = respond_jk
        operator._transition_cumulant_to_block_fock = contract_cumulant

        fock_expected = np.array(cumulant_fock, copy=True)
        for k in range(operator.nkpts):
            for spin in range(2):
                fock_expected[k] += (
                    operator.h1s[spin, k] @ tdm1s_block[spin, k]
                )
                fock_expected[k] += (
                    veff_ci[spin, k] @ operator.dm1s[spin, k]
                )
        expected = 2.0 * (
            fock_expected - fock_expected.conj().transpose(0, 2, 1)
        )

        actual = operator._orbital_ci_hessian_response(tdm1rs, tcm2)

        self.assertEqual(calls, ["dm1", "jk", "cumulant"])
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(
            actual + actual.conj().transpose(0, 2, 1), 0.0,
            atol=1e-13,
        )

    def test_complex_external_cumulant_response_matches_integral_derivative(self):
        rng = np.random.default_rng(19)
        nkpts = 3
        nmo = 3
        active = slice(1, 2)
        kconserv = np.fromfunction(
            lambda k1, k2, k3: (k1 - k2 + k3) % nkpts,
            (nkpts,) * 3, dtype=int,
        ).astype(int)
        # Unlike a two-point mesh, this mesh distinguishes the ERI rule
        # k1-k2+k3-k4 from the regrouped Hessian rule k1+k2-k3-k4.
        self.assertNotEqual(kconserv[0, 1, 0], (0 + 1 - 0) % nkpts)
        eri = {
            klabel: (
                rng.standard_normal((nmo,) * 4)
                + 1j * rng.standard_normal((nmo,) * 4)
            )
            for klabel in np.ndindex((nkpts,) * 3)
        }
        cumulant = {
            klabel: np.full(
                (1,) * 4,
                rng.standard_normal() + 1j * rng.standard_normal(),
            )
            for klabel in np.ndindex((nkpts,) * 3)
        }

        class FakeERIs:
            @staticmethod
            def ppaa(k1, k2, k3):
                return eri[k1, k2, k3][:, :, active, active]

            @staticmethod
            def papa(k1, k2, k3):
                return eri[k1, k2, k3][:, active, :, active]

            @staticmethod
            def paap(k1, k2, k3):
                return eri[k1, k2, k3][:, active, active, :]

        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.nkpts = nkpts
        operator.nmo = nmo
        operator.ncas = 1
        operator.ncore = 1
        operator.nocc = 2
        operator.kpts = np.zeros((nkpts, 3))
        operator.las = type("LAS", (), {
            "_scf": type("SCF", (), {"cell": object()})(),
        })()
        operator.eris = FakeERIs()
        operator.cascm2 = np.zeros((nkpts,) * 4, dtype=np.complex128)
        operator.mo_phase = np.ones(
            (nkpts, 1, nkpts), dtype=np.complex128,
        )

        kappa = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
        kappa[:, 1, 0] = [0.3 + 0.2j, -0.2 + 0.1j, 0.15 - 0.25j]
        kappa[:, 2, 0] = [-0.1 + 0.15j, 0.25j, -0.3 - 0.1j]
        kappa[:, 2, 1] = [0.25 - 0.35j, -0.2j, 0.1 + 0.3j]
        kappa -= kappa.conj().transpose(0, 2, 1)
        fock1_cumulant = np.zeros(
            (nkpts, nmo, nmo), dtype=np.complex128,
        )
        for k1, k2, k3 in np.ndindex((nkpts,) * 3):
            fock1_cumulant[k1, :, 1] += (
                eri[k1, k2, k3][:, 1, 1, 1]
                * cumulant[k1, k2, k3].item()
            )

        def transform(cascm2, mo_phase, klabel):
            return cumulant[tuple(klabel[:3])]

        with patch.object(
                klasscf.kpts_helper, "get_kconserv",
                return_value=kconserv), patch.object(
                klasscf, "_get_casdm2_kpts", side_effect=transform):
            response = operator._orbital_response_external_cumulant(
                kappa, fock1_cumulant,
            )

        commutator = (
            fock1_cumulant @ kappa - kappa @ fock1_cumulant
        ) / 2.0
        actual = response + commutator
        actual = actual - actual.conj().transpose(0, 2, 1)

        def cumulant_gradient(step):
            unitary = np.asarray([
                linalg.expm(step * kappa_k) for kappa_k in kappa
            ])
            fock = np.zeros(
                (nkpts, nmo, nmo), dtype=np.complex128,
            )
            for k1, k2, k3 in np.ndindex((nkpts,) * 3):
                k4 = kconserv[k1, k2, k3]
                eri_rotated = np.einsum(
                    "xp,yr,zs,wt,xyzw->prst",
                    unitary[k1].conj(), unitary[k2],
                    unitary[k3].conj(), unitary[k4],
                    eri[k1, k2, k3], optimize=True,
                )
                fock[k1, :, 1] += (
                    eri_rotated[:, 1, 1, 1]
                    * cumulant[k1, k2, k3].item()
                )
            return fock - fock.conj().transpose(0, 2, 1)

        step = 1e-5
        gradient_derivative = (
            cumulant_gradient(step) - cumulant_gradient(-step)
        ) / (2.0 * step)
        connection = commutator - commutator.conj().transpose(0, 2, 1)
        expected = gradient_derivative - connection
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(actual + actual.conj().transpose(0, 2, 1), 0)

    def test_orbital_hessian_response_is_skew_hermitian_for_complex_step(self):
        rng = np.random.default_rng(29)
        nmo = 3
        active = slice(1, 2)
        eri = (
            rng.standard_normal((nmo,) * 4)
            + 1j * rng.standard_normal((nmo,) * 4)
        )
        cumulant = np.full((1,) * 4, 0.4, dtype=np.complex128)

        class FakeLAS:
            _scf = type("SCF", (), {"cell": object()})()

            @staticmethod
            def get_veff(cell, dm_kpts=None, hermi=None, kpts=None):
                return np.zeros_like(dm_kpts)

        class FakeERIs:
            @staticmethod
            def ppaa(k1, k2, k3):
                return eri[:, :, active, active]

            @staticmethod
            def papa(k1, k2, k3):
                return eri[:, active, :, active]

            @staticmethod
            def paap(k1, k2, k3):
                return eri[:, active, active, :]

            @staticmethod
            def paaa(k1, k2, k3):
                return eri[:, active, active, active]

        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.las = FakeLAS()
        operator.eris = FakeERIs()
        operator.eri_paaa = operator.eris.paaa
        operator.nkpts = 1
        operator.nao = nmo
        operator.nmo = nmo
        operator.ncas = 1
        operator.ncore = 1
        operator.nocc = 2
        operator.kpts = np.zeros((1, 3))
        operator.mo_coeff = np.eye(nmo, dtype=np.complex128)[None]
        operator.mo_phase = np.ones((1, 1, 1), dtype=np.complex128)
        operator.cascm2 = cumulant
        operator.dm1s = np.zeros((2, 1, nmo, nmo), dtype=np.complex128)
        operator.dm1s[:, 0] = np.asarray([
            np.diag([1.0, 0.7, 0.0]),
            np.diag([1.0, 0.3, 0.0]),
        ])
        operator.h1s = (
            rng.standard_normal((2, 1, nmo, nmo))
            + 1j * rng.standard_normal((2, 1, nmo, nmo))
        )
        operator.fock1 = np.zeros((1, nmo, nmo), dtype=np.complex128)
        for spin in range(2):
            operator.fock1[0] += (
                operator.h1s[spin, 0] @ operator.dm1s[spin, 0]
            )
        operator.fock1[0, :, 1] += eri[:, 1, 1, 1] * cumulant.item()

        kappa = np.zeros((1, nmo, nmo), dtype=np.complex128)
        kappa[0, 1, 0] = 0.2 + 0.3j
        kappa[0, 2, 1] = -0.15 + 0.25j
        kappa -= kappa.conj().transpose(0, 2, 1)

        with patch.object(
                klasscf.kpts_helper, "get_kconserv",
                return_value=np.zeros((1, 1, 1), dtype=int)), patch.object(
                klasscf, "_get_casdm2_kpts", return_value=cumulant):
            response = operator._orbital_hessian_response(kappa)

        self.assertGreater(np.linalg.norm(response), 1e-10)
        np.testing.assert_allclose(
            response + response.conj().transpose(0, 2, 1), 0,
            atol=1e-13,
        )



if __name__ == "__main__":
    unittest.main()
