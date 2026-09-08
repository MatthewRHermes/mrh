import unittest
import numpy as np

from mrh.my_pyscf.pbc.mcscf.klasscf import KLASSCF_HessianOperator

# Author: Bhavnesh Jangid

"""Tests for k-LASSCF CI updates, Hessian responses, and vector layouts.

Eleven focused tests cover packed complex CI layouts, combined orbital/CI
updates, complex CI tangent rotations, diagonal and off-diagonal CI Hessian
actions, Hessian-diagonal assembly, and complex gradient construction.
"""

class _RecordingFCIBox:

    def __init__(self, diagonals):
        self.diagonals = diagonals
        self.call_count = 0

    def states_make_hdiag_csf(self, h1, h2, norb, nelec):
        self.call_count += 1
        return [np.array(diagonal, copy=True) for diagonal in self.diagonals]


class _SelectingTransformer:

    def __init__(self, indices):
        self.indices = np.asarray(indices)
        self.ncsf = len(self.indices)

    def pack_csf(self, diagonal):
        return np.asarray(diagonal)[self.indices]


class _RecordingGradientUGG:

    nvar_tot = 5

    def __init__(self):
        self.calls = []

    def pack(self, gorb, gci):
        self.calls.append((gorb, gci))
        ci_flat = [
            np.asarray(vector).reshape(-1)
            for fragment in gci
            for vector in fragment
        ]
        return np.concatenate(([gorb[0, 1, 0]], *ci_flat))


class _RecordingStepUGG:

    nvar_tot = 3

    def __init__(self, kappa, dci):
        self.kappa = kappa
        self.dci = dci
        self.calls = []

    def unpack(self, step):
        self.calls.append(np.array(step, copy=True))
        return self.kappa, self.dci


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


def make_layout_operator(frozen_ci=()):
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


def make_diagonal_operator():
    operator = KLASSCF_HessianOperator.__new__(KLASSCF_HessianOperator)
    boxes = [
        _RecordingFCIBox([
            np.array([0.2, 0.4, 0.6]),
            np.array([-0.1, 0.3]),
        ]),
        _RecordingFCIBox([np.array([9.0])]),
        _RecordingFCIBox([
            np.array([1.0 + 1.0j, 2.0 - 0.5j]),
        ]),
    ]
    operator.fciboxes = boxes
    operator.ncas_sub = np.array([1, 2, 1])
    operator.nelecas_sub = np.array([(1, 0), (1, 1), (0, 1)])
    operator.h1frs = [object(), object(), object()]
    operator.ci_transformers = [
        [_SelectingTransformer([2, 0]), _SelectingTransformer([1])],
        [_SelectingTransformer([0])],
        [_SelectingTransformer([1, 0])],
    ]
    operator.frozen_ci = {1}
    operator.eri_cas = (
        np.arange(4 ** 4, dtype=float).reshape((4,) * 4)
        + 0.1j
    )
    return operator, boxes


class KnownValues(unittest.TestCase):

    def test_complex_csf_layout_round_trip(self):
        """Unpack complex CSF coordinates into determinant CI vectors.

        Repacking must reproduce the original coordinates, and the unpacked
        vectors must retain the expected fragment/root layout.
        """
        operator = make_layout_operator()
        packed = np.array([
            0.2 + 0.1j, -0.3j, 0.4 - 0.2j, 0.5j,
        ])

        ci = operator._unpack_ci_vector(packed)

        np.testing.assert_allclose(operator._flatten_ci_vector(ci), packed)
        self.assertEqual(len(ci), 2)
        self.assertEqual(len(ci[0]), 1)

    def test_frozen_ci_is_zero_and_omitted(self):
        """Omit frozen CI variables when packing and restore zero responses."""
        operator = make_layout_operator(frozen_ci=(1,))
        packed = np.array([0.2 + 0.1j, -0.3j])

        ci = operator._unpack_ci_vector(packed)

        np.testing.assert_allclose(ci[0][0], packed)
        np.testing.assert_array_equal(ci[1][0], np.zeros(2))
        np.testing.assert_allclose(operator._flatten_ci_vector(ci), packed)

    def test_zero_ci_helpers_preserve_layout_and_dtype(self):
        """Create and detect zero CI steps without changing layout or dtype."""
        operator = make_layout_operator()
        zero = operator._zero_ci_step(np.complex128)

        self.assertTrue(operator._ci_step_is_zero(zero))
        self.assertTrue(np.issubdtype(zero[0][0].dtype, np.complexfloating))
        zero[1][0][0] = 1.0j
        self.assertFalse(operator._ci_step_is_zero(zero))

    def test_combined_update_dispatches_packed_orbital_and_ci_rotations(self):
        """Dispatch unpacked orbital and CI steps to their update methods."""
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        kappa = np.array([[[0.0, -0.2j], [-0.2j, 0.0]]])
        dci = [[np.array([0.1, -0.1j])]]
        operator.ugg = _RecordingStepUGG(kappa, dci)
        mo_result = np.array([[[1.0, 0.2j], [0.1, 0.9]]])
        ci_result = [[np.array([0.8, 0.6j])]]
        updates = {}

        def update_mo(actual):
            updates["mo"] = np.array(actual, copy=True)
            return mo_result

        def update_ci(actual):
            updates["ci"] = actual
            return ci_result

        operator._update_mo = update_mo
        operator._update_ci = update_ci
        step = np.array([[0.2 + 0.1j], [-0.3j], [0.4]])

        mo1, ci1 = operator.update_mo_ci(step)

        np.testing.assert_allclose(mo1, mo_result)
        np.testing.assert_allclose(ci1[0][0], ci_result[0][0])
        np.testing.assert_allclose(updates["mo"], kappa)
        np.testing.assert_allclose(updates["ci"][0][0], dci[0][0])
        self.assertEqual(len(operator.ugg.calls), 1)
        np.testing.assert_array_equal(operator.ugg.calls[0], step.reshape(-1))

    def test_combined_update_rebuilds_periodic_active_integrals(self):
        """Rebuild active two-electron integrals after updating orbitals."""
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ncastot = 2
        step = np.array([0.2 + 0.1j, -0.3j])
        mo1 = np.array([[
            [1.0, 0.2j],
            [0.1 - 0.1j, 0.9],
        ]])
        ci1 = [[np.array([0.8, 0.6j])]]
        update_calls = []

        def update(step_vector):
            update_calls.append(step_vector)
            return mo1, ci1

        operator.update_mo_ci = update
        new_h2 = np.full((2,) * 4, 0.4 - 0.2j)

        class RecordingLAS:
            def __init__(self):
                self.calls = []

            def get_h2cas(self, mo_coeff):
                self.calls.append(mo_coeff)
                return new_h2

        operator.las = RecordingLAS()
        old_h2 = np.full((2,) * 4, 9.0)

        actual_mo, actual_ci, actual_h2 = operator.update_mo_ci_eri(
            step, h2eff_sub=old_h2,
        )

        np.testing.assert_allclose(actual_mo, mo1)
        np.testing.assert_allclose(actual_ci[0][0], ci1[0][0])
        np.testing.assert_allclose(actual_h2, new_h2)
        self.assertEqual(len(update_calls), 1)
        np.testing.assert_allclose(update_calls[0], step)
        self.assertEqual(len(operator.las.calls), 1)
        np.testing.assert_allclose(operator.las.calls[0], mo1)

    def test_periodic_ci_update_rotates_complex_projected_tangents(self):
        """Rotate projected complex CI tangents while preserving normalization."""
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        ci00 = np.array([[1.0], [1.0j]]) / np.sqrt(2.0)
        ci10 = np.array([[0.5, -0.5j, np.sqrt(0.5)]])
        operator.ci = [[ci00], [ci10]]
        dci = [
            [np.array([[0.3 + 0.1j], [-0.2 + 0.4j]])],
            [np.array([[0.1j, 0.25, -0.35j]])],
        ]
        dci0 = [
            [np.array(dc, copy=True) for dc in dc_r]
            for dc_r in dci
        ]

        ci1 = operator._update_ci(dci)

        for ifrag, (ci0_r, dc_r, ci1_r) in enumerate(zip(
                operator.ci, dci, ci1)):
            for iroot, (ci0, dc, c1) in enumerate(zip(
                    ci0_r, dc_r, ci1_r)):
                reference = ci0.reshape(-1)
                tangent = dc.reshape(-1)
                tangent = tangent - reference * np.vdot(
                    reference, tangent,
                )
                tangent_norm = np.linalg.norm(tangent)
                expected = (
                    np.cos(tangent_norm) * reference
                    + np.sinc(tangent_norm / np.pi) * tangent
                ).reshape(ci0.shape)
                with self.subTest(cell=ifrag, root=iroot):
                    np.testing.assert_allclose(c1, expected, atol=1e-13)
                    np.testing.assert_allclose(np.linalg.norm(c1), 1.0)
                    np.testing.assert_allclose(
                        np.vdot(reference, tangent), 0.0, atol=1e-13,
                    )
                    np.testing.assert_array_equal(dc, dci0[ifrag][iroot])

        parallel_step = [
            [(0.2 - 0.3j) * ci00],
            [np.zeros_like(ci10)],
        ]
        parallel_result = operator._update_ci(parallel_step)
        np.testing.assert_allclose(parallel_result[0][0], ci00)
        np.testing.assert_allclose(parallel_result[1][0], ci10)

    def test_ci_response_diag_uses_hermitian_projection(self):
        """Apply the diagonal CI Hessian as the Hermitian-projected action."""
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        c0 = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2)
        trial = np.array([0.3 + 0.2j, -0.4j], dtype=np.complex128)
        hamiltonian = np.array(
            [[0.7, 0.2 - 0.5j], [0.2 + 0.5j, -0.1]],
            dtype=np.complex128,
        )
        energy = np.vdot(c0, hamiltonian @ c0)
        shifted_hamiltonian = hamiltonian - energy * np.eye(2)
        residual = shifted_hamiltonian @ c0
        operator.ci = [[c0]]
        operator.e0 = [[energy]]
        operator.hci0 = [[residual]]
        operator.h1frs = object()
        operator.eri_cas = object()
        shifts = []

        def apply_hamiltonian(h0, h1, h2, ci):
            shifts.append(h0)
            return [[shifted_hamiltonian @ ci[0][0]]]

        operator.Hci_all = apply_hamiltonian

        actual = operator.ci_response_diag([[trial]])[0][0]

        projector = np.eye(2) - np.outer(c0, c0.conj())
        expected = (
            2.0 * projector @ shifted_hamiltonian @ projector @ trial
        )
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(len(shifts), 1)
        np.testing.assert_allclose(shifts[0], [[-energy]])

    def test_ci_response_offdiag_contracts_and_projects_each_fragment(self):
        """Contract and project the off-diagonal CI response per fragment."""
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        boxes = [object(), object()]
        operator.fciboxes = boxes
        operator.ncas_sub = np.array([1, 2])
        operator.nelecas_sub = np.array([(1, 0), (1, 1)])
        operator.nroots = 2
        operator.eri_cas = np.zeros((3,) * 4, dtype=np.complex128)
        operator.linkstrl = ["links-0", "links-1"]
        operator.ci = [
            [
                np.array([1.0, 0.0], dtype=np.complex128),
                np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2),
            ],
            [
                np.array([0.6, 0.8j], dtype=np.complex128),
                np.array([0.0, 1.0], dtype=np.complex128),
            ],
        ]
        h1_response = [
            np.full((2, 2, 1, 1), 0.2 + 0.1j),
            np.full((2, 2, 2, 2), -0.3j),
        ]
        raw_response = [
            [
                np.array([0.4 + 0.2j, -0.1j]),
                np.array([0.3, -0.2 + 0.5j]),
            ],
            [
                np.array([-0.1 + 0.4j, 0.7]),
                np.array([0.2j, -0.3]),
            ],
        ]
        calls = []

        def apply_hamiltonian(
                fcibox, norb, nelec, h0, h1, h2, ci, linkstrl=None):
            ifrag = boxes.index(fcibox)
            calls.append((ifrag, h1, h2))
            return [np.array(hc, copy=True) for hc in raw_response[ifrag]]

        operator.Hci = apply_hamiltonian

        actual = operator.ci_response_offdiag(h1_response)

        self.assertEqual(len(calls), 2)
        for ifrag, call in enumerate(calls):
            _, h1, h2 = call
            np.testing.assert_allclose(h1, h1_response[ifrag])
            np.testing.assert_array_equal(h2, 0.0)

            for iroot, (hc, c0) in enumerate(zip(
                    raw_response[ifrag], operator.ci[ifrag])):
                expected = 2.0 * (hc - np.vdot(c0, hc) * c0)
                np.testing.assert_allclose(actual[ifrag][iroot], expected)
                np.testing.assert_allclose(
                    np.vdot(c0, actual[ifrag][iroot]), 0.0, atol=1e-13,
                )

    def test_hci_diag_packs_nonfrozen_fragments_in_layout_order(self):
        """Pack nonfrozen CI Hessian diagonals in external layout order."""
        operator, boxes = make_diagonal_operator()

        actual = operator._get_Hci_diag()

        self.assertEqual(len(actual), 3)
        np.testing.assert_allclose(actual[0], [0.6, 0.2])
        np.testing.assert_allclose(actual[1], [0.3])
        np.testing.assert_allclose(actual[2], [2.0 - 0.5j, 1.0 + 1.0j])
        self.assertEqual(boxes[0].call_count, 1)
        self.assertEqual(boxes[1].call_count, 0)
        self.assertEqual(boxes[2].call_count, 1)

    def test_hdiag_combines_orbital_and_ci_operator_order(self):
        """Assemble the Hessian diagonal with orbital entries before CI."""
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("UGG", (), {"nvar_tot": 6})()
        operator._get_Horb_diag = lambda: np.array(
            [1.0, 2.0], dtype=np.complex128,
        )
        operator._get_Hci_diag = lambda: [
            np.array([3.0]),
            np.array([4.0, 5.0, 6.0]),
        ]

        diagonal = operator._get_Hdiag()

        np.testing.assert_allclose(diagonal, [1, 2, 3, 4, 5, 6])

    def test_operator_gradient_uses_complex_adjoint_and_combined_order(self):
        """Pack the complex orbital and CI gradients in combined order."""
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = _RecordingGradientUGG()
        operator.fock1 = np.array([[
            [0.7, 0.2 + 0.4j],
            [-0.3 + 0.1j, -0.2],
        ]], dtype=np.complex128)
        operator.hci0 = [
            [
                np.array([0.15 + 0.2j]),
                np.array([-0.25j, 0.3 - 0.1j]),
            ],
            [np.array([0.4j])],
        ]

        gradient = operator.get_grad()

        gorb = operator.fock1 - operator.fock1.conj().transpose(0, 2, 1)
        expected = np.concatenate((
            [gorb[0, 1, 0]],
            *[
                2.0 * residual.reshape(-1)
                for residual_r in operator.hci0
                for residual in residual_r
            ],
        ))
        np.testing.assert_allclose(gradient, expected)
        self.assertEqual(len(operator.ugg.calls), 1)
        packed_gorb, packed_gci = operator.ugg.calls[0]
        np.testing.assert_allclose(packed_gorb, gorb)
        for response_r, residual_r in zip(packed_gci, operator.hci0):
            for response, residual in zip(response_r, residual_r):
                np.testing.assert_allclose(response, 2.0 * residual)


if __name__ == "__main__":
    unittest.main()
