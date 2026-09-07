import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.klasscf import KLASSCF_HessianOperator


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


class _DispatchUGG:
    """One-orbital-variable UGG for combined Hessian dispatch tests."""

    def __init__(self, operator):
        self.operator = operator
        self.nvar_orb = 1
        self.nvar_ci = operator.nvar_ci
        self.nvar_tot = self.nvar_orb + self.nvar_ci

    def unpack(self, x):
        x = np.asarray(x).reshape(-1)
        if x.size != self.nvar_tot:
            raise ValueError(
                f"combined vector has size {x.size}; expected {self.nvar_tot}"
            )
        kappa = self.unpack_orb(x[:self.nvar_orb])
        ci = self.operator._unpack_ci_vector(x[self.nvar_orb:])
        return kappa, ci

    def pack(self, kappa, ci):
        x_orb = self.pack_orb(kappa)
        x_ci = self.operator._flatten_ci_vector(ci)
        return np.concatenate((x_orb, x_ci))

    def unpack_orb(self, x_orb):
        x_orb = np.asarray(x_orb).reshape(-1)
        if x_orb.size != self.nvar_orb:
            raise ValueError(
                f"orbital vector has size {x_orb.size}; "
                f"expected {self.nvar_orb}"
            )
        kappa = np.zeros(
            (1, 2, 2), dtype=np.result_type(x_orb, complex),
        )
        kappa[0, 1, 0] = x_orb[0]
        kappa[0, 0, 1] = -x_orb[0].conjugate()
        return kappa

    @staticmethod
    def pack_orb(kappa):
        return np.asarray([kappa[0, 1, 0]])


def _set_toy_matvec_pipeline(operator):
    operator.make_tdm1s_sub = lambda ci1: "tdm"

    def make_tdm1s2c_sub(ci1):
        operator._last_ci1 = ci1
        return "tdm", "tcm2"

    operator.make_tdm1s2c_sub = make_tdm1s2c_sub

    def orbital_ci_response(
            tdm1rs, tcm2, tdm1s_block=None, veff_ci=None):
        assert tdm1rs == "tdm"
        assert tcm2 == "tcm2"
        assert tdm1s_block == "tdm-block"
        assert veff_ci == "veff"
        value = operator._last_ci1[0][0][0, 0]
        response = np.zeros((1, 2, 2), dtype=np.complex128)
        response[0, 1, 0] = 2.0 * value
        response[0, 0, 1] = -2.0 * value.conjugate()
        return response

    operator._orbital_ci_hessian_response = orbital_ci_response
    operator._transition_dm1s_to_block = lambda tdm: "tdm-block"
    operator._get_ci_veff_response = lambda tdm: "veff"
    operator.get_h1eff_response = (
        lambda tdm, tdm1s_block=None, veff_block=None: "h1-response"
    )
    operator.ci_response_diag = lambda ci1: [
        [2.0 * trial for trial in trial_r] for trial_r in ci1
    ]
    operator.ci_response_offdiag = lambda h1: [
        [3.0 * trial for trial in trial_r] for trial_r in operator._last_ci1
    ]

    original_diag = operator.ci_response_diag

    def cache_and_apply(ci1):
        operator._last_ci1 = ci1
        return original_diag(ci1)

    operator.ci_response_diag = cache_and_apply


class KnownValues(unittest.TestCase):

    def test_matvec_dispatches_combined_vector_to_ci_block(self):
        operator = KLASSCF_HessianOperator.__new__(KLASSCF_HessianOperator)
        operator.ci = [
            [np.zeros((2, 1), dtype=np.complex128)],
            [np.zeros((2, 1), dtype=np.complex128)],
        ]
        _set_csf_layout(operator)
        operator.ugg = _DispatchUGG(operator)
        operator.level_shift = 0.25
        _set_toy_matvec_pipeline(operator)

        ci_trial = np.array(
            [0.2 + 0.1j, -0.3j, 0.4 - 0.2j, -0.1],
            dtype=np.complex128,
        )
        trial = np.concatenate(([0.0j], ci_trial))
        result = operator._matvec(trial)

        self.assertEqual(result.shape, trial.shape)
        self.assertEqual(operator.shape, (trial.size, trial.size))
        self.assertTrue(np.issubdtype(result.dtype, np.complexfloating))
        np.testing.assert_allclose(result[0], ci_trial[0])
        np.testing.assert_allclose(result[1:], 5.25 * ci_trial)

    def test_matvec_dispatches_orbital_only_step(self):
        operator = KLASSCF_HessianOperator.__new__(KLASSCF_HessianOperator)
        operator.ci = [
            [np.zeros((2, 1), dtype=np.complex128)],
            [np.zeros((2, 1), dtype=np.complex128)],
        ]
        _set_csf_layout(operator)
        operator.ugg = _DispatchUGG(operator)
        operator.level_shift = 0.0
        calls = []

        def orbital_response(kappa1):
            calls.append(np.array(kappa1, copy=True))
            return 2.0 * kappa1

        operator._orbital_hessian_response = orbital_response

        def ci_orbital_response(kappa1):
            value = kappa1[0, 1, 0]
            return [
                [value * np.ones_like(c0) for c0 in ci0_r]
                for ci0_r in operator.ci
            ]

        operator._ci_orbital_hessian_response = ci_orbital_response
        trial = np.zeros(operator.ugg.nvar_tot, dtype=np.complex128)
        trial[0] = 0.3 - 0.2j

        result = operator._matvec(trial)

        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(calls[0][0, 1, 0], trial[0])
        np.testing.assert_allclose(result[0], trial[0])
        np.testing.assert_allclose(result[1:], trial[0])


if __name__ == "__main__":
    unittest.main()
