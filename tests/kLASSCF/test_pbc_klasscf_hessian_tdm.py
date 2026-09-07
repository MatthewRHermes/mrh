import unittest
from unittest.mock import patch

import numpy as np

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.mcscf import klasscf
from mrh.my_pyscf.pbc.mcscf.klasscf import KLASSCF_HessianOperator


class _TransitionFCIBox:
    _state_args = staticmethod(lambda value: value)
    _solver_args = staticmethod(lambda value: value)

    def __init__(self):
        self.fcisolvers = [object()]
        self.collect_calls = 0
        self.transition_operator = np.array(
            [[0.7, 0.2 - 0.3j], [-0.1 + 0.4j, -0.2]],
            dtype=np.complex128,
        )
        self.dm1a_operator = np.array(
            [[1.0, 0.2j], [-0.3j, 0.4]], dtype=np.complex128,
        )
        self.dm1b_operator = np.array(
            [[-0.2, 0.1 + 0.2j], [0.3j, 0.8]], dtype=np.complex128,
        )
        self.dm2_operator = (
            np.arange(16, dtype=float).reshape((2,) * 4)
            + 0.1j * np.arange(16, 0, -1).reshape((2,) * 4)
        ) / 17.0
        self.dm2_scales = np.array([0.5, -0.2j, 0.3j, 0.7])

    @staticmethod
    def _get_nelec(solver, nelec):
        return tuple(nelec)

    def _collect(
            self, name, ci1, ci0, norb, nelec, link_index=None, **kwargs):
        if name not in (
                "trans_rdm1s", "trans_rdm1s_py",
                "trans_rdm12s", "trans_rdm12s_py"):
            raise AssertionError(f"unexpected contraction {name}")
        self.collect_calls += 1
        bra = np.asarray(ci1[0]).reshape(-1)
        ket = np.asarray(ci0[0]).reshape(-1)
        amplitude = np.vdot(bra, self.transition_operator @ ket)
        dm1s = (
            amplitude * self.dm1a_operator,
            amplitude * self.dm1b_operator,
        )
        if name.startswith("trans_rdm12s"):
            dm2s = tuple(
                amplitude * scale * self.dm2_operator
                for scale in self.dm2_scales
            )
            return [(dm1s, dm2s)]
        return [dm1s]


class _SingleSolverFCIBox:
    """Minimal state-average wrapper around one complex FCI solver."""

    _state_args = staticmethod(lambda value: value)
    _solver_args = staticmethod(lambda value: value)

    def __init__(self):
        self.solver = direct_spin1_cplx.FCISolver()
        self.fcisolvers = [self.solver]

    @staticmethod
    def _get_nelec(solver, nelec):
        return tuple(nelec)

    def _collect(
            self, name, ci1, ci0, norb, nelec, link_index=None, **kwargs):
        method = getattr(self.solver, name)
        link = None if link_index is None else link_index[0]
        return [method(
            ci1[0], ci0[0], norb, nelec[0], link_index=link,
        )]


def make_operator():
    operator = KLASSCF_HessianOperator.__new__(KLASSCF_HessianOperator)
    operator.nroots = 1
    operator.ncas_sub = np.array([2, 2])
    operator.nelecas_sub = np.array([(1, 0), (1, 0)])
    operator.ncastot = 4
    operator.weights = np.array([1.0])
    operator.fciboxes = [_TransitionFCIBox(), _TransitionFCIBox()]
    operator.linkstr = [None, None]
    operator.eri_cas = np.zeros((4,) * 4, dtype=np.complex128)
    operator.casdm1frs = [
        np.zeros((1, 2, 2, 2), dtype=np.complex128)
        for _ in range(2)
    ]
    operator.casdm1s = np.zeros((2, 4, 4), dtype=np.complex128)
    operator.casdm2fr = [
        np.zeros((1, 2, 2, 2, 2), dtype=np.complex128)
        for _ in range(2)
    ]
    ci_ref = np.array([[0.8], [0.6j]], dtype=np.complex128)
    operator.ci = [[ci_ref], [ci_ref.copy()]]
    return operator


class KnownValues(unittest.TestCase):

    def test_transition_rdm_contracts_every_fragment(self):
        operator = make_operator()
        trial = np.array([[0.3j], [0.4]], dtype=np.complex128)
        ci1 = [[trial], [trial.copy()]]

        tdm1rs = operator.make_tdm1s_sub(ci1)

        self.assertEqual(tdm1rs.shape, (1, 2, 4, 4))
        self.assertEqual(
            [box.collect_calls for box in operator.fciboxes], [1, 1]
        )
        np.testing.assert_allclose(
            tdm1rs[:, :, :2, :2], tdm1rs[:, :, 2:, 2:],
        )
        np.testing.assert_allclose(
            tdm1rs, tdm1rs.swapaxes(-1, -2).conj(),
        )

    def test_transition_cumulant_uses_stored_state_average_casdm1s(self):
        operator = make_operator()
        trial = np.array([[0.3j], [0.4]], dtype=np.complex128)
        ci1 = [[trial], [trial.copy()]]
        operator.casdm1s[0] = np.diag([0.7, 0.2, 0.4, 0.1])
        operator.casdm1s[1] = np.diag([0.1, 0.5, 0.2, 0.6])
        operator.casdm1frs = [
            np.array([[np.diag([0.6, 0.1]), np.diag([0.2, 0.4])]])
            for _ in range(2)
        ]
        operator.casdm2fr = [
            np.full(
                (1, 2, 2, 2, 2), 0.03 * (cell + 1),
                dtype=np.complex128,
            )
            for cell in range(2)
        ]

        tdm1rs, tcm2 = operator.make_tdm1s2c_sub(ci1)

        tdm1rs_one_sided = np.zeros_like(tdm1rs)
        tdm2_one_sided = np.zeros_like(tcm2)
        for cell, box in enumerate(operator.fciboxes):
            i, j = 2 * cell, 2 * (cell + 1)
            c0 = operator.ci[cell][0]
            overlap = np.vdot(trial, c0)
            amplitude = np.vdot(
                trial.reshape(-1),
                box.transition_operator @ c0.reshape(-1),
            )
            tdm1rs_one_sided[0, :, i:j, i:j] = (
                amplitude * np.stack(
                    (box.dm1a_operator, box.dm1b_operator), axis=0,
                )
                - overlap * operator.casdm1frs[cell][0]
            )
            transition_dm2 = (
                amplitude * box.dm2_scales.sum() * box.dm2_operator
            )
            tdm2_one_sided[i:j, i:j, i:j, i:j] = (
                transition_dm2
                - overlap * operator.casdm2fr[cell][0]
            ) / 2.0

        expected_tdm1rs = (
            tdm1rs_one_sided
            + tdm1rs_one_sided.swapaxes(-1, -2).conj()
        )
        expected_tdm2 = np.array(tdm2_one_sided, copy=True)
        expected_tdm2 += expected_tdm2.conj().transpose(1, 0, 3, 2)
        expected_tdm2 += expected_tdm2.transpose(2, 3, 0, 1)

        tdm1s_0 = expected_tdm1rs[0, :, :2, :2]
        tdm1s_1 = expected_tdm1rs[0, :, 2:, 2:]
        dm1s_0 = operator.casdm1frs[0][0]
        dm1s_1 = operator.casdm1frs[1][0]
        coulomb = np.einsum(
            "ij,kl->ijkl", tdm1s_0.sum(axis=0), dm1s_1.sum(axis=0),
        )
        coulomb += np.einsum(
            "ij,kl->ijkl", dm1s_0.sum(axis=0), tdm1s_1.sum(axis=0),
        )
        expected_tdm2[:2, :2, 2:, 2:] = coulomb
        expected_tdm2[2:, 2:, :2, :2] = coulomb.transpose(2, 3, 0, 1)
        exchange = sum(
            np.einsum("ij,kl->ilkj", tdm1s_0[spin], dm1s_1[spin])
            + np.einsum("ij,kl->ilkj", dm1s_0[spin], tdm1s_1[spin])
            for spin in range(2)
        )
        expected_tdm2[:2, 2:, 2:, :2] = -exchange
        expected_tdm2[2:, :2, :2, 2:] = (
            -exchange.conj().transpose(1, 0, 3, 2)
        )

        tdm1s = expected_tdm1rs[0]
        expected_tcm2 = np.array(expected_tdm2, copy=True)
        expected_tcm2 -= np.multiply.outer(
            tdm1s.sum(axis=0), operator.casdm1s.sum(axis=0),
        )
        expected_tcm2 -= np.multiply.outer(
            operator.casdm1s.sum(axis=0), tdm1s.sum(axis=0),
        )
        for spin in range(2):
            expected_tcm2 += np.multiply.outer(
                tdm1s[spin], operator.casdm1s[spin],
            ).transpose(0, 3, 2, 1)
            expected_tcm2 += np.multiply.outer(
                operator.casdm1s[spin], tdm1s[spin],
            ).transpose(0, 3, 2, 1)

        np.testing.assert_allclose(tdm1rs, expected_tdm1rs)
        np.testing.assert_allclose(tcm2, expected_tcm2)
        np.testing.assert_allclose(
            tdm1rs, tdm1rs.swapaxes(-1, -2).conj(),
        )
        np.testing.assert_allclose(
            tcm2, tcm2.conj().transpose(1, 0, 3, 2),
        )
        np.testing.assert_allclose(
            tcm2, tcm2.transpose(2, 3, 0, 1),
        )

    def test_complex_transition_cumulant_matches_finite_difference(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        box = _SingleSolverFCIBox()
        norb = 2
        nelec = (1, 1)
        rng = np.random.default_rng(91)
        c0 = (
            rng.standard_normal((2, 2))
            + 1j * rng.standard_normal((2, 2))
        )
        c0 /= np.linalg.norm(c0)
        c1 = (
            rng.standard_normal((2, 2))
            + 1j * rng.standard_normal((2, 2))
        )
        c1 -= np.vdot(c0, c1) * c0
        c1 /= np.linalg.norm(c1)

        dm1s_ref = np.asarray(box.solver.make_rdm1s(c0, norb, nelec))
        dm2_ref = np.asarray(box.solver.make_rdm12(c0, norb, nelec)[1])
        operator.nroots = 1
        operator.ncastot = norb
        operator.ncas_sub = np.array([norb])
        operator.nelecas_sub = np.array([nelec])
        operator.weights = np.array([1.0])
        operator.fciboxes = [box]
        operator.linkstr = [None]
        operator.ci = [[c0]]
        operator.eri_cas = np.zeros((norb,) * 4, dtype=np.complex128)
        operator.casdm1frs = [dm1s_ref[None]]
        operator.casdm1s = dm1s_ref
        operator.casdm2fr = [dm2_ref[None]]

        tdm1rs, tcm2 = operator.make_tdm1s2c_sub([[c1]])

        def make_cumulant(c):
            dm1s = np.asarray(box.solver.make_rdm1s(c, norb, nelec))
            dm2 = np.asarray(box.solver.make_rdm12(c, norb, nelec)[1])
            dm1 = dm1s.sum(axis=0)
            cumulant = dm2 - np.multiply.outer(dm1, dm1)
            for spin in range(2):
                cumulant += np.multiply.outer(
                    dm1s[spin], dm1s[spin],
                ).transpose(0, 3, 2, 1)
            return dm1s, cumulant

        step = 1e-5
        c_plus = c0 + step * c1
        c_plus /= np.linalg.norm(c_plus)
        c_minus = c0 - step * c1
        c_minus /= np.linalg.norm(c_minus)
        dm1s_plus, cumulant_plus = make_cumulant(c_plus)
        dm1s_minus, cumulant_minus = make_cumulant(c_minus)
        dm1s_derivative = (dm1s_plus - dm1s_minus) / (2.0 * step)
        cumulant_derivative = (
            cumulant_plus - cumulant_minus
        ) / (2.0 * step)

        np.testing.assert_allclose(
            tdm1rs[0], dm1s_derivative, atol=2e-9, rtol=2e-9,
        )
        np.testing.assert_allclose(
            tcm2, cumulant_derivative, atol=2e-9, rtol=2e-9,
        )

    def test_h1eff_response_follows_periodic_las_projection(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.nroots = 2
        operator.ncas_sub = np.array([1, 2])
        operator.ncastot = 3
        operator.ncore = 0
        operator.nocc = 3
        operator.nkpts = 1
        operator.kmesh = (1, 1, 1)
        operator.weights = np.array([0.25, 0.75])
        operator.las = type("FakeLAS", (), {"_scf": object()})()
        rng = np.random.default_rng(103)
        operator.eri_cas = (
            rng.standard_normal((3,) * 4)
            + 1j * rng.standard_normal((3,) * 4)
        )
        tdm1rs = np.zeros((2, 2, 3, 3), dtype=np.complex128)
        offsets = np.array([0, 1, 3])
        for i, j in zip(offsets[:-1], offsets[1:]):
            block = (
                rng.standard_normal((2, 2, j - i, j - i))
                + 1j * rng.standard_normal((2, 2, j - i, j - i))
            )
            tdm1rs[:, :, i:j, i:j] = (
                block + block.conj().transpose(0, 1, 3, 2)
            )

        veff_wannier = (
            rng.standard_normal((2, 3, 3))
            + 1j * rng.standard_normal((2, 3, 3))
        )
        operator._transition_dm1s_to_block = lambda value: np.zeros(
            (2, 1, 3, 3), dtype=value.dtype,
        )
        operator._get_ci_veff_response = lambda value: (
            veff_wannier[:, None]
        )
        with patch.object(
                klasscf, "_convert_h1e_mo_k_to_wann",
                side_effect=lambda scf, kmesh, value: value[0]):
            actual = operator.get_h1eff_response(tdm1rs)

        average = np.einsum(
            "r,rspq->spq", operator.weights, tdm1rs,
        )
        delta = tdm1rs - average[None]
        response_all = np.tensordot(
            delta, operator.eri_cas, axes=((2, 3), (2, 3)),
        )
        response_all += response_all[:, ::-1]
        response_all -= np.tensordot(
            delta, operator.eri_cas, axes=((2, 3), (2, 1)),
        )
        response_all += veff_wannier[None]
        expected = []
        for i, j in zip(offsets[:-1], offsets[1:]):
            dm1rs = tdm1rs[:, :, i:j, i:j]
            h2 = operator.eri_cas[i:j, i:j, i:j, i:j]
            response_self = np.tensordot(
                dm1rs, h2, axes=((2, 3), (2, 3)),
            )
            response_self += response_self[:, ::-1]
            response_self -= np.tensordot(
                dm1rs, h2, axes=((2, 3), (2, 1)),
            )
            expected.append(
                response_all[:, :, i:j, i:j] - response_self
            )

        self.assertEqual(
            [response.shape for response in actual],
            [(2, 2, 1, 1), (2, 2, 2, 2)],
        )
        for actual_fragment, expected_fragment in zip(actual, expected):
            np.testing.assert_allclose(actual_fragment, expected_fragment)

    def test_transition_dm1s_transforms_to_active_block_mos(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.nroots = 2
        operator.weights = np.array([0.3, 0.7])
        operator.nkpts = 2
        operator.ncas = 2
        operator.ncastot = 4
        operator.ncore = 1
        operator.nocc = 3
        operator.nmo = 5
        operator.mo_phase = np.zeros((2, 2, 4), dtype=np.complex128)
        fourier = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
        for k in range(2):
            for cell in range(2):
                for band in range(2):
                    operator.mo_phase[k, band, 2 * cell + band] = (
                        fourier[k, cell]
                    )

        rng = np.random.default_rng(113)
        tdm1rs = (
            rng.standard_normal((2, 2, 4, 4))
            + 1j * rng.standard_normal((2, 2, 4, 4))
        )
        tdm1rs += tdm1rs.swapaxes(-1, -2).conj()

        actual = operator._transition_dm1s_to_block(tdm1rs)

        averaged = np.einsum(
            "r,rspq->spq", operator.weights, tdm1rs, optimize=True,
        )
        expected_active = np.asarray([
            [
                operator.mo_phase[k] @ averaged[spin]
                @ operator.mo_phase[k].conj().T
                for k in range(operator.nkpts)
            ]
            for spin in range(2)
        ])
        expected = np.zeros((2, 2, 5, 5), dtype=np.complex128)
        expected[:, :, 1:3, 1:3] = expected_active

        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(
            actual, actual.conj().transpose(0, 1, 3, 2),
        )
        np.testing.assert_allclose(actual[:, :, :1], 0.0)
        np.testing.assert_allclose(actual[:, :, 3:], 0.0)
        np.testing.assert_allclose(actual[:, :, :, :1], 0.0)
        np.testing.assert_allclose(actual[:, :, :, 3:], 0.0)

    def test_transition_cumulant_uses_bra_ket_momentum_blocks(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        nkpts = 3
        operator.nkpts = nkpts
        operator.ncas = 1
        operator.ncastot = nkpts
        operator.ncore = 1
        operator.nocc = 2
        operator.nmo = 3
        operator.kpts = np.zeros((nkpts, 3))
        operator.las = type("LAS", (), {
            "_scf": type("SCF", (), {"cell": object()})(),
        })()
        phase = np.exp(
            2j * np.pi * np.arange(nkpts)[:, None]
            * np.arange(nkpts)[None, :] / nkpts
        ) / np.sqrt(nkpts)
        operator.mo_phase = phase[:, None, :]
        operator.eri_cas = np.zeros((nkpts,) * 4, dtype=np.complex128)
        rng = np.random.default_rng(127)
        tcm2 = (
            rng.standard_normal((nkpts,) * 4)
            + 1j * rng.standard_normal((nkpts,) * 4)
        )
        paaa = {
            key: (
                rng.standard_normal((operator.nmo, 1, 1, 1))
                + 1j * rng.standard_normal((operator.nmo, 1, 1, 1))
            )
            for key in np.ndindex((nkpts,) * 3)
        }
        calls = []

        def get_paaa(k1, k2, k3):
            calls.append((k1, k2, k3))
            return paaa[k1, k2, k3]

        operator.eri_paaa = get_paaa
        kconserv = np.fromfunction(
            lambda k1, k2, k3: (k1 - k2 + k3) % nkpts,
            (nkpts,) * 3, dtype=int,
        ).astype(int)

        expected = np.zeros(
            (nkpts, operator.nmo, operator.nmo), dtype=np.complex128,
        )
        for k1, k2, k3 in np.ndindex((nkpts,) * 3):
            k4 = kconserv[k1, k2, k3]
            transformed = np.einsum(
                "iP,jQ,PQRS,kR,lS->ijkl",
                operator.mo_phase[k1].conj(),
                operator.mo_phase[k2],
                tcm2,
                operator.mo_phase[k3].conj(),
                operator.mo_phase[k4],
                optimize=True,
            )
            expected[k1, :, 1:2] += np.tensordot(
                paaa[k1, k2, k3], transformed,
                axes=((1, 2, 3), (1, 2, 3)),
            )

        with patch.object(
                klasscf.kpts_helper, "get_kconserv",
                return_value=kconserv):
            actual = operator._transition_cumulant_to_block_fock(tcm2)

        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(actual[:, :, :1], 0.0)
        np.testing.assert_allclose(actual[:, :, 2:], 0.0)
        self.assertCountEqual(calls, list(np.ndindex((nkpts,) * 3)))


if __name__ == "__main__":
    unittest.main()
