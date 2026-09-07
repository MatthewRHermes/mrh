import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.klasscf import KLASSCF_HessianOperator


class _RecordingFCIBox:

    def __init__(self, diagonals):
        self.diagonals = diagonals
        self.calls = []

    def states_make_hdiag_csf(self, h1, h2, norb, nelec):
        self.calls.append((h1, h2, norb, nelec))
        return [np.array(diagonal, copy=True) for diagonal in self.diagonals]


class _SelectingTransformer:

    def __init__(self, indices, ncsf=None):
        self.indices = np.asarray(indices)
        self.ncsf = len(self.indices) if ncsf is None else ncsf

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

    def test_combined_update_dispatches_packed_orbital_and_ci_rotations(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        kappa = np.array([[[0.0, -0.2j], [-0.2j, 0.0]]])
        dci = [[np.array([0.1, -0.1j])]]
        operator.ugg = _RecordingStepUGG(kappa, dci)
        mo_result = np.array([[[1.0, 0.2j], [0.1, 0.9]]])
        ci_result = [[np.array([0.8, 0.6j])]]
        calls = []

        def update_mo(actual):
            calls.append(("mo", actual))
            return mo_result

        def update_ci(actual):
            calls.append(("ci", actual))
            return ci_result

        operator._update_mo = update_mo
        operator._update_ci = update_ci
        step = np.array([[0.2 + 0.1j], [-0.3j], [0.4]])

        mo1, ci1 = operator.update_mo_ci(step)

        self.assertIs(mo1, mo_result)
        self.assertIs(ci1, ci_result)
        self.assertEqual([name for name, _ in calls], ["mo", "ci"])
        self.assertIs(calls[0][1], kappa)
        self.assertIs(calls[1][1], dci)
        self.assertEqual(len(operator.ugg.calls), 1)
        np.testing.assert_array_equal(operator.ugg.calls[0], step.reshape(-1))

    def test_combined_update_rejects_packed_step_size_mismatch(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("UGG", (), {
            "nvar_tot": 2,
            "unpack": staticmethod(lambda step: self.fail(
                "invalid step should not be unpacked"
            )),
        })()

        with self.assertRaisesRegex(ValueError, "step_vector has shape"):
            operator.update_mo_ci(np.zeros(3))

    def test_combined_update_rebuilds_periodic_active_integrals(self):
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

        self.assertIs(actual_mo, mo1)
        self.assertIs(actual_ci, ci1)
        self.assertIs(actual_h2, new_h2)
        self.assertEqual(len(update_calls), 1)
        self.assertIs(update_calls[0], step)
        self.assertEqual(len(operator.las.calls), 1)
        self.assertIs(operator.las.calls[0], mo1)
        np.testing.assert_array_equal(old_h2, 9.0)

    def test_combined_update_rejects_rebuilt_integral_shape(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ncastot = 2
        operator.update_mo_ci = lambda step: (np.eye(2)[None], [[np.ones(1)]])
        operator.las = type("LAS", (), {
            "get_h2cas": staticmethod(lambda mo_coeff: np.zeros((2,) * 3)),
        })()

        with self.assertRaisesRegex(ValueError, "updated_h2eff_sub has shape"):
            operator.update_mo_ci_eri(np.zeros(1))

    def test_periodic_ci_update_rotates_complex_projected_tangents(self):
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
                    np.testing.assert_allclose(c1, expected, atol=2e-14)
                    np.testing.assert_allclose(np.linalg.norm(c1), 1.0)
                    np.testing.assert_allclose(
                        np.vdot(reference, tangent), 0.0, atol=2e-16,
                    )
                    np.testing.assert_array_equal(dc, dci0[ifrag][iroot])

        parallel_step = [
            [(0.2 - 0.3j) * ci00],
            [np.zeros_like(ci10)],
        ]
        parallel_result = operator._update_ci(parallel_step)
        np.testing.assert_allclose(parallel_result[0][0], ci00)
        np.testing.assert_allclose(parallel_result[1][0], ci10)

    def test_periodic_ci_update_validates_layout_and_vectors(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        ci0 = np.array([1.0, 0.0], dtype=np.complex128)
        operator.ci = [[ci0]]

        with self.assertRaisesRegex(ValueError, "contains 0 cells; expected 1"):
            operator._update_ci([])
        with self.assertRaisesRegex(ValueError, "contains 0 roots; expected 1"):
            operator._update_ci([[]])
        with self.assertRaisesRegex(ValueError, "dci_0_0 has shape"):
            operator._update_ci([[np.zeros((2, 1))]])

        nonfinite = np.array([np.nan, 0.0])
        with self.assertRaisesRegex(ValueError, "must contain only finite"):
            operator._update_ci([[nonfinite]])

        operator.ci = [[2.0 * ci0]]
        with self.assertRaisesRegex(ValueError, "has norm 2.0; expected 1"):
            operator._update_ci([[np.zeros_like(ci0)]])

    def test_ci_response_diag_uses_hermitian_projection(self):
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
        calls = []

        def apply_hamiltonian(h0, h1, h2, ci):
            calls.append((h0, h1, h2, ci))
            return [[shifted_hamiltonian @ ci[0][0]]]

        operator.Hci_all = apply_hamiltonian

        actual = operator.ci_response_diag([[trial]])[0][0]

        projector = np.eye(2) - np.outer(c0, c0.conj())
        expected = (
            2.0 * projector @ shifted_hamiltonian @ projector @ trial
        )
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(len(calls), 1)
        h0, h1, h2, ci = calls[0]
        np.testing.assert_allclose(h0, [[-energy]])
        self.assertIs(h1, operator.h1frs)
        self.assertIs(h2, operator.eri_cas)
        self.assertIs(ci[0][0], trial)

    def test_ci_response_offdiag_contracts_and_projects_each_fragment(self):
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
            calls.append((
                ifrag, norb, nelec, h0, h1, h2, ci, linkstrl,
            ))
            return [np.array(hc, copy=True) for hc in raw_response[ifrag]]

        operator.Hci = apply_hamiltonian

        actual = operator.ci_response_offdiag(h1_response)

        self.assertEqual(len(calls), 2)
        for ifrag, call in enumerate(calls):
            _, norb, nelec, h0, h1, h2, ci, linkstrl = call
            self.assertEqual(norb, operator.ncas_sub[ifrag])
            np.testing.assert_array_equal(
                nelec, operator.nelecas_sub[ifrag],
            )
            np.testing.assert_array_equal(h0, [0.0, 0.0])
            self.assertIs(h1, h1_response[ifrag])
            self.assertEqual(h2.shape, (norb,) * 4)
            self.assertEqual(h2.dtype, operator.eri_cas.dtype)
            np.testing.assert_array_equal(h2, 0.0)
            self.assertIs(ci, operator.ci[ifrag])
            self.assertEqual(linkstrl, operator.linkstrl[ifrag])

            for iroot, (hc, c0) in enumerate(zip(
                    raw_response[ifrag], operator.ci[ifrag])):
                expected = 2.0 * (hc - np.vdot(c0, hc) * c0)
                np.testing.assert_allclose(actual[ifrag][iroot], expected)
                np.testing.assert_allclose(
                    np.vdot(c0, actual[ifrag][iroot]), 0.0, atol=2e-16,
                )

    def test_ci_response_offdiag_validates_fragment_count(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.fciboxes = [object(), object()]

        with self.assertRaisesRegex(
                ValueError, "h1frs_response contains 1 cells; expected 2"):
            operator.ci_response_offdiag([np.zeros(1)])

    def test_ci_hessian_response_builds_densities_and_combines_blocks(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        ci1 = [
            [np.array([0.2 + 0.1j]), np.array([-0.4j])],
            [np.array([0.3, -0.2j])],
        ]
        tdm1rs = [object(), object()]
        h1frs_response = [object(), object()]
        diagonal = [
            [np.array([0.7 - 0.1j]), np.array([0.2 + 0.5j])],
            [np.array([-0.3j, 0.4])],
        ]
        offdiagonal = [
            [np.array([-0.2 + 0.4j]), np.array([0.6 - 0.1j])],
            [np.array([0.1 + 0.2j, -0.5j])],
        ]
        calls = []

        def make_tdm(actual_ci1):
            calls.append(("tdm", actual_ci1))
            return tdm1rs

        def get_h1eff(actual_tdm1rs):
            calls.append(("h1eff", actual_tdm1rs))
            return h1frs_response

        def diagonal_response(actual_ci1):
            calls.append(("diag", actual_ci1))
            return diagonal

        def offdiagonal_response(actual_h1frs):
            calls.append(("offdiag", actual_h1frs))
            return offdiagonal

        operator.make_tdm1s_sub = make_tdm
        operator.get_h1eff_response = get_h1eff
        operator.ci_response_diag = diagonal_response
        operator.ci_response_offdiag = offdiagonal_response

        actual = operator._ci_hessian_response(ci1)

        self.assertEqual(
            [name for name, _ in calls],
            ["tdm", "h1eff", "diag", "offdiag"],
        )
        self.assertIs(calls[0][1], ci1)
        self.assertIs(calls[1][1], tdm1rs)
        self.assertIs(calls[2][1], ci1)
        self.assertIs(calls[3][1], h1frs_response)
        for actual_r, diagonal_r, offdiagonal_r in zip(
                actual, diagonal, offdiagonal):
            for result, diag, offdiag in zip(
                    actual_r, diagonal_r, offdiagonal_r):
                np.testing.assert_allclose(result, diag + offdiag)

    def test_ci_hessian_response_reuses_supplied_densities(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        ci1 = [[np.array([0.2j])]]
        tdm1rs = [object()]
        h1frs_response = [object()]
        density_calls = []

        operator.make_tdm1s_sub = lambda actual_ci1: self.fail(
            "supplied transition densities should be reused"
        )

        def get_h1eff(actual_tdm1rs):
            density_calls.append(actual_tdm1rs)
            return h1frs_response

        operator.get_h1eff_response = get_h1eff
        operator.ci_response_diag = lambda actual_ci1: [[np.array([0.3])]]
        operator.ci_response_offdiag = (
            lambda actual_h1frs: [[np.array([-0.1j])]]
        )

        actual = operator._ci_hessian_response(ci1, tdm1rs=tdm1rs)

        self.assertEqual(len(density_calls), 1)
        self.assertIs(density_calls[0], tdm1rs)
        np.testing.assert_allclose(actual[0][0], [0.3 - 0.1j])

    def test_hci_diag_packs_nonfrozen_fragments_in_layout_order(self):
        operator, boxes = make_diagonal_operator()

        actual = operator._get_Hci_diag()

        self.assertEqual(len(actual), 3)
        np.testing.assert_allclose(actual[0], [0.6, 0.2])
        np.testing.assert_allclose(actual[1], [0.3])
        np.testing.assert_allclose(actual[2], [2.0 - 0.5j, 1.0 + 1.0j])
        self.assertEqual(len(boxes[0].calls), 1)
        self.assertEqual(len(boxes[1].calls), 0)
        self.assertEqual(len(boxes[2].calls), 1)

        h1_0, h2_0, norb_0, nelec_0 = boxes[0].calls[0]
        self.assertIs(h1_0, operator.h1frs[0])
        np.testing.assert_array_equal(
            h2_0, operator.eri_cas[0:1, 0:1, 0:1, 0:1],
        )
        self.assertEqual(norb_0, 1)
        np.testing.assert_array_equal(nelec_0, [1, 0])

        h1_2, h2_2, norb_2, nelec_2 = boxes[2].calls[0]
        self.assertIs(h1_2, operator.h1frs[2])
        np.testing.assert_array_equal(
            h2_2, operator.eri_cas[3:4, 3:4, 3:4, 3:4],
        )
        self.assertEqual(norb_2, 1)
        np.testing.assert_array_equal(nelec_2, [0, 1])

    def test_hci_diag_rejects_inconsistent_root_count(self):
        operator, boxes = make_diagonal_operator()
        boxes[0].diagonals = [np.array([0.2, 0.4, 0.6])]

        with self.assertRaisesRegex(
                ValueError, "cell 0 produced 1 Hamiltonian diagonals for 2"):
            operator._get_Hci_diag()

    def test_hci_diag_rejects_inconsistent_packed_size(self):
        operator, _ = make_diagonal_operator()
        operator.ci_transformers[0][0] = _SelectingTransformer(
            [0], ncsf=2,
        )

        with self.assertRaisesRegex(
                ValueError, "cell 0, root 0 packed Hamiltonian diagonal"):
            operator._get_Hci_diag()

    def test_hdiag_combines_orbital_and_ci_operator_order(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("UGG", (), {"nvar_tot": 6})()
        calls = []

        def orbital_diagonal():
            calls.append("orbital")
            return np.array([1.0, 2.0], dtype=np.complex128)

        def ci_diagonal():
            calls.append("ci")
            return [
                np.array([3.0]),
                np.array([4.0, 5.0, 6.0]),
            ]

        operator._get_Horb_diag = orbital_diagonal
        operator._get_Hci_diag = ci_diagonal

        diagonal = operator._get_Hdiag()

        np.testing.assert_allclose(diagonal, [1, 2, 3, 4, 5, 6])
        self.assertEqual(calls, ["orbital", "ci"])

    def test_hdiag_handles_an_empty_parameter_space(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("UGG", (), {"nvar_tot": 0})()
        operator._get_Horb_diag = lambda: np.empty(
            0, dtype=np.complex128,
        )
        operator._get_Hci_diag = lambda: []

        diagonal = operator._get_Hdiag()

        self.assertEqual(diagonal.shape, (0,))
        self.assertEqual(diagonal.dtype, np.dtype(np.complex128))

    def test_hdiag_rejects_total_layout_mismatch(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("UGG", (), {"nvar_tot": 4})()
        operator._get_Horb_diag = lambda: np.array([1.0, 2.0])
        operator._get_Hci_diag = lambda: [np.array([3.0])]

        with self.assertRaisesRegex(ValueError, "Hdiag has shape"):
            operator._get_Hdiag()

    def test_operator_gradient_uses_complex_adjoint_and_combined_order(self):
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

    def test_operator_gradient_rejects_packed_layout_mismatch(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("UGG", (), {
            "nvar_tot": 2,
            "pack": staticmethod(lambda gorb, gci: np.array([1.0])),
        })()
        operator.fock1 = np.zeros((1, 1, 1), dtype=np.complex128)
        operator.hci0 = []

        with self.assertRaisesRegex(ValueError, "gradient has shape"):
            operator.get_grad()


if __name__ == "__main__":
    unittest.main()
