"""Tests for the density, Hamiltonian, orbital, and CI Hessian intermediates."""

import unittest
from unittest.mock import patch

import numpy as np

from mrh.my_pyscf.pbc.mcscf import klasscf
from mrh.my_pyscf.pbc.mcscf.klasscf import KLASSCF_HessianOperator


class _IdentitySCF:

    cell = object()

    @staticmethod
    def get_ovlp(kpts=None):
        return np.broadcast_to(np.eye(3), (len(kpts), 3, 3)).copy()


class _DensityLAS:
    def __init__(self, casdm2, hcore=None):
        self._scf = _IdentitySCF()
        self.casdm2 = np.asarray(casdm2)
        if hcore is None:
            hcore = np.zeros((2, 3, 3))
        self.hcore = np.asarray(hcore)

    @staticmethod
    def make_casdm1s_sub(casdm1frs=None):
        return [np.asarray(dm[0]) for dm in casdm1frs]

    @staticmethod
    def states_make_casdm1s(casdm1frs=None):
        result = np.zeros((1, 2, 2, 2), dtype=np.complex128)
        for cell, dm1 in enumerate(casdm1frs):
            result[:, :, cell, cell] = dm1[:, :, 0, 0]
        return result

    def make_casdm2(self, **kwargs):
        return np.array(self.casdm2, copy=True)

    def get_hcore(self, kpts=None):
        return np.array(self.hcore, copy=True)


def make_operator(casdm2):
    operator = KLASSCF_HessianOperator.__new__(KLASSCF_HessianOperator)
    operator.las = _DensityLAS(casdm2)
    operator.ci = [[np.ones(1)], [np.ones(1)]]
    operator.ncas_sub = np.array([1, 1])
    operator.nelecas_sub = np.array([(1, 0), (1, 0)])
    operator.weights = np.array([1.0])
    operator.nroots = 1
    operator.ncastot = 2
    operator.ncore = 1
    operator.ncas = 1
    operator.nocc = 2
    operator.nkpts = 2
    operator.nao = 3
    operator.nmo = 3
    operator.mo_coeff = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    operator.kpts = np.zeros((2, 3))
    return operator


class _LazyERIs:

    @staticmethod
    def ppaa(k1, k2, k3):
        return None

    @staticmethod
    def papa(k1, k2, k3):
        return None

    @staticmethod
    def paap(k1, k2, k3):
        return None

    @staticmethod
    def paaa(k1, k2, k3):
        return None


class _RecordingERIs(_LazyERIs):
    def __init__(self):
        self.paaa_calls = []

    def paaa(self, k1, k2, k3):
        self.paaa_calls.append((k1, k2, k3))
        value = 1.0 + k1 + 2.0 * k2 + 3.0 * k3
        return value * np.arange(1, 4, dtype=float).reshape(3, 1, 1, 1)


class _LinkFCIBox:
    @staticmethod
    def states_gen_linkstr(norb, nelec, tril):
        return f"links-{norb}-{tuple(nelec)}"


class KnownValues(unittest.TestCase):

    def test_density_and_cumulant_intermediates(self):
        """Construct state-averaged densities and the active-space cumulant."""
        casdm1frs = [
            np.array([[[[0.8]], [[0.2]]]], dtype=np.complex128),
            np.array([[[[0.3]], [[0.7]]]], dtype=np.complex128),
        ]
        casdm2fr = [np.zeros((1, 1, 1, 1, 1)) for _ in range(2)]
        casdm2 = np.arange(16, dtype=float).reshape((2,) * 4) / 13.0
        dm1s_kpts = np.zeros((2, 2, 3, 3), dtype=np.complex128)
        dm1s_kpts[0, 0] = np.diag([1.0, 0.8, 0.0])
        dm1s_kpts[1, 0] = np.diag([1.0, 0.2, 0.0])
        dm1s_kpts[0, 1] = np.diag([1.0, 0.3, 0.0])
        dm1s_kpts[1, 1] = np.diag([1.0, 0.7, 0.0])
        operator = make_operator(casdm2)

        operator._init_dms_(
            casdm1frs, casdm2fr=casdm2fr, dm1s_kpts=dm1s_kpts,
        )

        expected_casdm1s = np.zeros((2, 2, 2), dtype=np.complex128)
        expected_casdm1s[:, 0, 0] = [0.8, 0.2]
        expected_casdm1s[:, 1, 1] = [0.3, 0.7]
        casdm1 = expected_casdm1s.sum(axis=0)
        expected_cascm2 = casdm2 - np.multiply.outer(casdm1, casdm1)
        for spin_dm in expected_casdm1s:
            expected_cascm2 += np.multiply.outer(
                spin_dm, spin_dm,
            ).transpose(0, 3, 2, 1)

        np.testing.assert_allclose(operator.casdm1s, expected_casdm1s)
        np.testing.assert_allclose(operator.casdm2, casdm2)
        np.testing.assert_allclose(operator.cascm2, expected_cascm2)
        np.testing.assert_allclose(operator.dm1s_kpts, dm1s_kpts)
        np.testing.assert_allclose(operator.dm1s, dm1s_kpts)

    def test_hamiltonian_intermediates(self):
        """Store one- and two-electron Hamiltonian intermediates consistently."""
        hcore = np.array([
            np.diag([1.0, 2.0, 3.0]),
            np.diag([1.5, 2.5, 3.5]),
        ])
        veff_kpts = np.zeros((2, 2, 3, 3), dtype=np.complex128)
        veff_kpts[0, 0] = np.diag([0.1, 0.2, 0.3])
        veff_kpts[1, 0] = np.diag([0.4, 0.5, 0.6])
        veff_kpts[0, 1] = np.diag([0.7, 0.8, 0.9])
        veff_kpts[1, 1] = np.diag([1.0, 1.1, 1.2])
        h2eff = np.arange(16, dtype=float).reshape((2,) * 4) / 17.0
        h1eff = [
            np.zeros((1, 2, 1, 1), dtype=np.complex128)
            for _ in range(2)
        ]
        operator = make_operator(np.zeros((2,) * 4))
        operator.las.hcore = hcore

        operator._init_ham_(h1eff, h2eff, veff_kpts=veff_kpts)

        np.testing.assert_allclose(operator.hcore, hcore)
        np.testing.assert_allclose(
            operator.h1s, hcore[None, :, :, :] + veff_kpts,
        )
        np.testing.assert_allclose(operator.eri_cas, h2eff)
        for actual, expected in zip(operator.h1frs, h1eff):
            np.testing.assert_allclose(actual, expected)



    def test_orbital_fock_intermediate(self):
        """Build the periodic orbital Fock intermediate from one- and two-body terms."""
        operator = make_operator(np.zeros((2,) * 4))
        operator.h1s = np.zeros((2, 2, 3, 3), dtype=np.complex128)
        operator.dm1s = np.zeros((2, 2, 3, 3), dtype=np.complex128)
        operator.h1s[:, 0] = [np.eye(3), 2.0 * np.eye(3)]
        operator.h1s[:, 1] = [3.0 * np.eye(3), 4.0 * np.eye(3)]
        operator.dm1s[:, 0] = [0.5 * np.eye(3), 0.25 * np.eye(3)]
        operator.dm1s[:, 1] = [0.2 * np.eye(3), 0.1 * np.eye(3)]
        operator.cascm2 = np.zeros((2,) * 4)
        operator.kmesh = (2, 1, 1)
        eris = _RecordingERIs()
        operator._init_eri_(eris)
        mo_phase = np.array([
            [[1.0, 1.0]],
            [[1.0, -1.0]],
        ], dtype=np.complex128) / np.sqrt(2.0)
        transformed = {
            (k1, k2, k3): np.full(
                (1, 1, 1, 1), 0.25 + k1 + 2 * k2 + 4 * k3,
            )
            for k1 in range(2)
            for k2 in range(2)
            for k3 in range(2)
        }

        def transform(cumulant, phase, klabel):
            return transformed[tuple(klabel[:3])]

        with patch.object(
                klasscf.kpts_helper, "get_kconserv",
                return_value=np.zeros((2, 2, 2), dtype=int)), patch.object(
                klasscf, "_get_casdm2_kpts", side_effect=transform):
            operator._init_orb_(mo_phase=mo_phase)

        expected = np.array([
            np.eye(3), np.eye(3),
        ], dtype=np.complex128)
        expected[1] *= 1.0
        for k1, k2, k3 in klasscf.kpts_helper.loop_kkk(2):
            paaa = eris.paaa(k1, k2, k3).reshape(3)
            expected[k1, :, 1] += (
                paaa * transformed[(k1, k2, k3)].item()
            )
        np.testing.assert_allclose(operator.fock1, expected)

    def test_ci_reference_actions_and_residuals(self):
        """Build CI link tables, reference energies, and projected residuals."""
        operator = make_operator(np.zeros((2,) * 4))
        operator.fciboxes = [_LinkFCIBox(), _LinkFCIBox()]
        operator.h1frs = [object(), object()]
        operator.eri_cas = object()
        operator.ci = [
            [np.array([1.0, 0.0], dtype=np.complex128)],
            [np.array([0.0, 1.0], dtype=np.complex128)],
        ]
        hc0 = [
            [np.array([2.0, 0.3j])],
            [np.array([-0.2j, 3.0])],
        ]
        operator.Hci_all = lambda *args: hc0

        operator._init_ci_()

        np.testing.assert_allclose(operator.e0, [[2.0], [3.0]])
        np.testing.assert_allclose(operator.hci0[0][0], [0.0, 0.3j])
        np.testing.assert_allclose(operator.hci0[1][0], [-0.2j, 0.0])



if __name__ == "__main__":
    unittest.main()
