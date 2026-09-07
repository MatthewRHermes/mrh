import unittest

import numpy as np

from mrh.my_pyscf.mcscf.lasscf_sync_o0 import (
    LASSCF_HessianOperator as MolecularHessianOperator,
)
from mrh.my_pyscf.pbc.mcscf.klasscf import KLASSCF_HessianOperator


class _IdentityTransformer:
    ncsf = 2
    ndet = 2

    @staticmethod
    def vec_det2csf(vector, order="C", normalize=False):
        return np.array(vector, copy=True)

    @staticmethod
    def vec_csf2det(vector, order="C", normalize=False):
        return np.array(vector, copy=True)


class _FakeUGG:
    nvar_tot = 7


def make_operator(frozen_ci=()):
    operator = KLASSCF_HessianOperator.__new__(KLASSCF_HessianOperator)
    operator.ugg = _FakeUGG()
    operator.ci = [
        [np.array([1.0, 0.0], dtype=np.complex128)],
        [np.array([0.0, 1.0], dtype=np.complex128)],
    ]
    operator.ci_transformers = [
        [_IdentityTransformer()],
        [_IdentityTransformer()],
    ]
    operator.frozen_ci = set(frozen_ci)
    operator.nvar_ci = 2 * (2 - len(operator.frozen_ci))
    return operator


class KnownValues(unittest.TestCase):

    def test_inherits_molecular_hessian_interface(self):
        self.assertTrue(
            issubclass(KLASSCF_HessianOperator, MolecularHessianOperator)
        )
        self.assertEqual(make_operator().shape, (7, 7))

    def test_complex_csf_layout_round_trip(self):
        operator = make_operator()
        packed = np.array([
            0.2 + 0.1j, -0.3j, 0.4 - 0.2j, 0.5j,
        ])

        ci = operator._unpack_ci_vector(packed)

        np.testing.assert_allclose(operator._flatten_ci_vector(ci), packed)
        self.assertEqual(len(ci), 2)
        self.assertEqual(len(ci[0]), 1)

    def test_frozen_ci_is_zero_and_omitted(self):
        operator = make_operator(frozen_ci=(1,))
        packed = np.array([0.2 + 0.1j, -0.3j])

        ci = operator._unpack_ci_vector(packed)

        np.testing.assert_allclose(ci[0][0], packed)
        np.testing.assert_array_equal(ci[1][0], np.zeros(2))
        np.testing.assert_allclose(operator._flatten_ci_vector(ci), packed)

    def test_rejects_inconsistent_ci_vector_sizes(self):
        operator = make_operator()
        with self.assertRaisesRegex(ValueError, "trial vector has size"):
            operator._unpack_ci_vector(np.zeros(3))
        with self.assertRaisesRegex(ValueError, "one entry per cell"):
            operator._flatten_ci_vector([[np.zeros(2)]])

    def test_zero_ci_helpers_preserve_layout_and_dtype(self):
        operator = make_operator()
        zero = operator._zero_ci_step(np.complex128)

        self.assertTrue(operator._ci_step_is_zero(zero))
        self.assertTrue(np.issubdtype(zero[0][0].dtype, np.complexfloating))
        zero[1][0][0] = 1.0j
        self.assertFalse(operator._ci_step_is_zero(zero))


if __name__ == "__main__":
    unittest.main()
