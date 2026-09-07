import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.klasci import (
    PBCLASCINoSymm,
    PBCLASCITransSymm,
)
from mrh.my_pyscf.pbc.mcscf.klasscf import (
    KLASSCF_HessianOperator,
    KLASSCF_TransSymmHessianOperator,
)


from mrh.my_pyscf.pbc.mcscf.klasscf import (
    KLASSCF_HessianOperator,
    KLASSCF_TransSymmHessianOperator,
)


class _FakeFCIBox:
    """Small transition-RDM backend for CI-Hessian unit tests."""

    _state_args = staticmethod(lambda value: value)
    _solver_args = staticmethod(lambda value: value)

    def __init__(self):
        self.fcisolvers = [object()]
        self.collect_calls = 0
        self.hdiag_calls = 0
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

    def _get_nelec(self, solver, nelec):
        return tuple(nelec)

    def states_make_hdiag_csf(self, h1, h2, norb, nelec):
        self.hdiag_calls += 1
        return [np.array([1.25, 2.5], dtype=np.complex128)]

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


class _IdentityCSFTransformer:
    """Two-determinant/two-CSF transform used to test complex packing."""

    ndet = 2
    ncsf = 2

    @staticmethod
    def vec_det2csf(civec, order="C", normalize=False):
        return np.array(civec, copy=True)

    @staticmethod
    def vec_csf2det(civec, order="C", normalize=False):
        return np.array(civec, copy=True)

    @staticmethod
    def pack_csf(civec):
        return np.array(civec, copy=True)


def _set_csf_layout(operator):
    operator.ci_transformers = [
        [_IdentityCSFTransformer()] for _ in operator.ci
    ]
    operator.frozen_ci = set()
    operator.nvar_ci = sum(
        transformer.ncsf
        for transformers in operator.ci_transformers
        for transformer in transformers
    )


def _new_tdm_operator(cls, phases=None):
    operator = cls.__new__(cls)
    operator.nroots = 1
    operator.ncas_sub = np.array([2, 2])
    operator.nelecas_sub = np.array([(1, 0), (1, 0)])
    operator.ncas = 2
    operator.ncastot = 4
    operator.ncell = 2
    operator.weights = np.array([1.0])
    operator.fciboxes = [_FakeFCIBox(), _FakeFCIBox()]
    operator.linkstr = [None, None]
    operator.eri_cas = np.zeros((4, 4, 4, 4), dtype=np.complex128)
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
    if phases is None:
        phases = np.ones(2, dtype=np.complex128)
    phases = np.asarray(phases)
    operator.ref_cell = 0
    operator.phase_per_frag = phases
    operator.ci = [[phase * ci_ref] for phase in phases]
    _set_csf_layout(operator)
    return operator, ci_ref


class _CIOnlyUGG:

    def __init__(self, operator):
        self.operator = operator
        self.nvar_orb = 0
        self.nvar_ci = operator.nvar_ci
        self.nvar_tot = self.nvar_ci

    def unpack(self, vector):
        kappa = np.zeros((1, 0, 0), dtype=np.complex128)
        return kappa, self.operator._unpack_ci_vector(vector)

    def pack(self, kappa, ci):
        return self.operator._flatten_ci_vector(ci)


def _set_toy_matvec_pipeline(operator):
    def make_tdm1s2c_sub(ci1):
        operator._last_ci1 = ci1
        return "tdm", "tcm2"

    operator.make_tdm1s2c_sub = make_tdm1s2c_sub
    operator._transition_dm1s_to_block = lambda tdm: "tdm-block"
    operator._get_ci_veff_response = lambda tdm: "veff"
    operator._orbital_ci_hessian_response = (
        lambda tdm1rs, tcm2, tdm1s_block=None, veff_ci=None:
        np.zeros((1, 0, 0), dtype=np.complex128)
    )
    operator.get_h1eff_response = (
        lambda tdm, tdm1s_block=None, veff_block=None: "h1-response"
    )
    operator.ci_response_diag = lambda ci1: [
        [2.0 * trial for trial in trial_r] for trial_r in ci1
    ]
    operator.ci_response_offdiag = lambda h1: [
        [3.0 * trial for trial in trial_r] for trial_r in operator._last_ci1
    ]


class KnownValuesKLASSCFTransSymmHessianOperator(unittest.TestCase):

    def test_is_child_of_general_k_lasscf_hessian(self):
        self.assertTrue(issubclass(
            KLASSCF_TransSymmHessianOperator,
            KLASSCF_HessianOperator,
        ))

    def test_ci_pack_and_unpack_apply_translation_phases(self):
        phases = np.exp(1j * np.array([0.0, 0.43]))
        operator, ci_ref = _new_tdm_operator(
            KLASSCF_TransSymmHessianOperator, phases=phases,
        )

        packed = operator._pack_ci(operator.ci)
        unpacked = operator._unpack_cif(packed)

        np.testing.assert_allclose(packed[0], ci_ref)
        for phase, ci_cell in zip(phases, unpacked):
            np.testing.assert_allclose(ci_cell[0], phase * ci_ref)

    def test_transition_rdm_uses_only_packed_reference_ci(self):
        phases = np.exp(1j * np.array([0.0, -0.37]))
        plain, _ = _new_tdm_operator(
            KLASSCF_HessianOperator, phases=phases,
        )
        adapted, _ = _new_tdm_operator(
            KLASSCF_TransSymmHessianOperator, phases=phases,
        )
        trial_ref = np.array([[0.3j], [0.4]], dtype=np.complex128)
        ci1 = [[phase * trial_ref] for phase in phases]

        expected = plain.make_tdm1s_sub(ci1)
        actual = adapted.make_tdm1s_sub(ci1)

        np.testing.assert_allclose(actual, expected)
        self.assertEqual(
            [box.collect_calls for box in adapted.fciboxes], [1, 0]
        )

    def test_phase_validation_and_reference_normalization(self):
        phases = KLASSCF_TransSymmHessianOperator._normalize_phase_per_frag(
            np.exp(1j * np.array([0.2, -0.3])), 2, 1,
        )
        np.testing.assert_allclose(phases[1], 1.0)
        np.testing.assert_allclose(np.abs(phases), 1.0)
        with self.assertRaisesRegex(ValueError, "unit magnitude"):
            KLASSCF_TransSymmHessianOperator._normalize_phase_per_frag(
                np.array([1.0, 2.0]), 2, 0,
            )

    def test_registered_hop_matches_periodic_ci_layout(self):
        self.assertIs(PBCLASCINoSymm._hop, KLASSCF_HessianOperator)
        self.assertIs(
            PBCLASCITransSymm._hop,
            KLASSCF_TransSymmHessianOperator,
        )

    def test_matvec_keeps_full_phase_adapted_ci_layout(self):
        phases = np.exp(1j * np.array([0.0, 0.61]))
        operator, _ = _new_tdm_operator(
            KLASSCF_TransSymmHessianOperator, phases=phases,
        )
        operator.ugg = _CIOnlyUGG(operator)
        operator.level_shift = 0.25
        _set_toy_matvec_pipeline(operator)
        trial_ref = np.array([[0.2 + 0.1j], [-0.3j]])
        trial = np.concatenate([
            (phase * trial_ref).reshape(-1) for phase in phases
        ])

        result = operator._matvec(trial)

        np.testing.assert_allclose(result, 5.25 * trial)
        result_cells = result.reshape(2, -1)
        np.testing.assert_allclose(
            result_cells[1], phases[1] * result_cells[0]
        )


if __name__ == "__main__":
    unittest.main()
