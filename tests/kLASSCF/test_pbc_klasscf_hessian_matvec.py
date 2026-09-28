import unittest
from types import SimpleNamespace

import numpy as np

from pyscf import lib

from mrh.my_pyscf.pbc.mcscf.klasscf import (
    KLASSCF_HessianOperator,
    KLASSCF_UnitaryGroupGenerators,
)

# Author: Bhavnesh Jangid:


"""Tests for combined orbital/CI k-LASSCF Hessian-vector dispatch.

These tests check that a packed trial vector is split into its orbital and CI
components, passed to the appropriate Hessian-response routines, and combined
again in the expected packed order. They also verify the orbital-CI coupling
terms and their normalization factors.
"""

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


class _DispatchUGG(KLASSCF_UnitaryGroupGenerators):
    """Use real CI packing with one orbital variable for dispatch tests."""

    nvar_orb = 1

    def __init__(self, operator):
        self.ci = operator.ci
        self.ci_transformers = [
            [_IdentityCSFTransformer() for _ in ci_r] for ci_r in self.ci
        ]
        self.frozen_ci = []

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
        """Combine CI-input responses and level shift in packed-vector order."""
        operator = KLASSCF_HessianOperator.__new__(KLASSCF_HessianOperator)
        operator.ci = [
            [np.zeros((2, 1), dtype=np.complex128)],
            [np.zeros((2, 1), dtype=np.complex128)],
        ]
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
        """Route an orbital-only trial through orbital and CI output responses."""
        operator = KLASSCF_HessianOperator.__new__(KLASSCF_HessianOperator)
        operator.ci = [
            [np.zeros((2, 1), dtype=np.complex128)],
            [np.zeros((2, 1), dtype=np.complex128)],
        ]
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

    def test_matvec_returns_zero_for_a_zero_trial_vector(self):
        """Return zero for a zero trial with and without the fast-path guards."""
        operator = KLASSCF_HessianOperator.__new__(KLASSCF_HessianOperator)
        operator.ci = [
            [np.zeros((2, 1), dtype=np.complex128)],
            [np.zeros((2, 1), dtype=np.complex128)],
        ]
        operator.ugg = _DispatchUGG(operator)
        operator.level_shift = 0.25
        _set_toy_matvec_pipeline(operator)
        calls = []

        def orbital_response(kappa1):
            calls.append("orbital-orbital")
            return 2.0 * kappa1

        operator._orbital_hessian_response = orbital_response

        def ci_orbital_response(kappa1):
            calls.append("ci-orbital")
            return [
                [kappa1[0, 1, 0] * np.ones_like(c0) for c0 in ci0_r]
                for ci0_r in operator.ci
            ]

        operator._ci_orbital_hessian_response = ci_orbital_response
        trial = np.zeros(operator.ugg.nvar_tot, dtype=np.complex128)

        guarded = operator._matvec(trial)
        self.assertEqual(guarded.shape, trial.shape)
        self.assertTrue(np.issubdtype(guarded.dtype, np.complexfloating))
        np.testing.assert_array_equal(guarded, 0.0)
        self.assertEqual(
            calls, [],
            msg="a zero trial must skip both response blocks by default",
        )

        # The guards make A @ 0 == 0 hold by construction, so the assertion
        # above says nothing about the response routines themselves.  Raising
        # the verbosity drives the same zero trial through both blocks, which
        # tests that the Hessian action is genuinely homogeneous.
        operator.las = SimpleNamespace(verbose=lib.logger.DEBUG1)
        unguarded = operator._matvec(trial)
        self.assertEqual(calls, ["orbital-orbital", "ci-orbital"])
        self.assertEqual(unguarded.shape, trial.shape)
        np.testing.assert_array_equal(unguarded, 0.0)


if __name__ == "__main__":
    unittest.main()
