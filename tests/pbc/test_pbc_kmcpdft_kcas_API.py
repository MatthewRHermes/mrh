"""API tests for momentum-resolved periodic CASCI-PDFT."""

import unittest
from functools import lru_cache
from types import SimpleNamespace
from unittest import mock

import numpy as np
from pyscf.lib import logger, param
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf, mcpdft as pbc_mcpdft
from mrh.my_pyscf.pbc.fci import direct_spin1_kfci
from mrh.my_pyscf.pbc.mcpdft import _dms as pbc_dms
from mrh.my_pyscf.pbc.mcpdft import kmcpdft
from mrh.my_pyscf.pbc.mcpdft._dms import dm2_cumulant_complex
from mrh.my_pyscf.pbc.mcpdft import otfnalperiodic


class RecordingSolver:

    def __init__(self, nroots=1):
        self.nroots = nroots
        self.calls = []

    def make_rdm1s(self, ci, norb, nelec, **kwargs):
        self.calls.append(("make_rdm1s", ci, norb, nelec, kwargs))
        return np.stack((np.eye(norb), 2.0 * np.eye(norb)))

    def make_rdm2(self, ci, norb, nelec, **kwargs):
        self.calls.append(("make_rdm2", ci, norb, nelec, kwargs))
        return np.zeros((norb,) * 4)


def make_mc(solver, target_k=0, nkpts=3, ncas=2, nelecas=(1, 1)):
    return SimpleNamespace(
        nkpts=nkpts,
        ncas=ncas,
        nelecas=nelecas,
        target_k=target_k,
        fcisolver=solver,
    )


def make_kconserv(nkpts):
    kconserv = np.empty((nkpts, nkpts, nkpts), dtype=int)
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            for k3 in range(nkpts):
                kconserv[k1, k2, k3] = (k1 - k2 + k3) % nkpts
    return kconserv


@lru_cache(maxsize=1)
def build_periodic_h2():
    cell = gto.Cell()
    cell.a = np.diag([2.24, 2.24, 12.0])
    cell.atom = [
        ["H", (0.0, 0.0, 6.0)],
        ["H", (0.74, 0.0, 6.0)],
    ]
    cell.basis = "sto-6g"
    cell.unit = "Angstrom"
    cell.precision = 1e-9
    cell.verbose = 0
    cell.build()

    kmesh = (2, 1, 1)
    kpts = cell.make_kpts(kmesh, wrap_around=True)
    kmf = scf.KRHF(cell, kpts=kpts).density_fit()
    kmf.exxdiv = None
    kmf.conv_tol = 1e-9
    kmf.verbose = 0
    kmf.kernel()
    if not kmf.converged:
        raise RuntimeError("Periodic H2 KRHF did not converge")
    return cell, kmf, kmesh, np.asarray(kmf.mo_coeff)


class KCASPDFTRDMTests(unittest.TestCase):

    def test_select_charged_result_uses_requested_sector(self):
        results = [
            {"target_k": 0, "ci": "sector-0"},
            {"target_k": 2, "ci": "sector-2"},
        ]
        mc = SimpleNamespace(
            nkpts=3, target_k=None, charged_results=results,
        )

        result = kmcpdft._select_charged_kcas_result(
            mc, target_k=-1,
        )

        self.assertIs(result, results[1])

    def test_select_charged_result_defaults_to_object_sector(self):
        results = [{"target_k": 1, "ci": "sector-1"}]
        mc = SimpleNamespace(
            nkpts=3, target_k=4, charged_results=results,
        )

        result = kmcpdft._select_charged_kcas_result(mc)

        self.assertIs(result, results[0])

    def test_select_charged_result_infers_only_stored_sector(self):
        results = [{"target_k": 2, "ci": "sector-2"}]
        mc = SimpleNamespace(
            nkpts=3, target_k=None, charged_results=results,
        )

        result = kmcpdft._select_charged_kcas_result(mc)

        self.assertIs(result, results[0])

    def test_select_charged_result_rejects_ambiguous_sector(self):
        mc = SimpleNamespace(
            nkpts=3,
            target_k=None,
            charged_results=[{"target_k": 0}, {"target_k": 1}],
        )

        with self.assertRaisesRegex(ValueError, "target_k is required"):
            kmcpdft._select_charged_kcas_result(mc)

    def test_select_charged_result_rejects_missing_sector(self):
        mc = SimpleNamespace(
            nkpts=3,
            target_k=None,
            charged_results=[{"target_k": 0}],
        )

        with self.assertRaisesRegex(ValueError, "target_k=1"):
            kmcpdft._select_charged_kcas_result(mc, target_k=1)

    def test_get_charged_context_uses_sector_ci_and_electron_count(self):
        solver = RecordingSolver()
        result = {
            "target_k": 2,
            "ci": "charged-ci",
            "nelecastot": (3, 2),
        }
        mc = make_mc(solver, target_k=-1, nkpts=3, ncas=2)
        mc.charged_results = [result]
        mc.charged_nelecastot = (9, 9)

        context = kmcpdft._get_charged_kcas_rdm_context(mc)

        self.assertEqual(
            context,
            (solver, "charged-ci", 6, (3, 2),
             {"nkpts": 3, "target_k": 2}),
        )

    def test_get_charged_context_selects_root_from_explicit_ci(self):
        solver = RecordingSolver(nroots=2)
        mc = make_mc(solver, target_k=1, nkpts=2, ncas=2)
        mc.charged_results = [{
            "target_k": 1,
            "ci": ["stored-0", "stored-1"],
            "nelecastot": (2, 1),
        }]

        context = kmcpdft._get_charged_kcas_rdm_context(
            mc, ci=["override-0", "override-1"], state=1,
        )

        self.assertIs(context[0], solver)
        self.assertEqual(context[1], "override-1")
        self.assertEqual(context[2:], (
            4, (2, 1), {"nkpts": 2, "target_k": 1},
        ))

    def test_get_charged_context_falls_back_to_object_electron_count(self):
        solver = RecordingSolver()
        mc = make_mc(solver, target_k=0, nkpts=2, ncas=1)
        mc.charged_results = [{"target_k": 0, "ci": "charged-ci"}]
        mc.charged_nelecastot = (1, 0)

        context = kmcpdft._get_charged_kcas_rdm_context(mc)

        self.assertEqual(context[3], (1, 0))

    def test_get_charged_context_rejects_missing_ci(self):
        mc = make_mc(RecordingSolver(), target_k=0, nkpts=2, ncas=1)
        mc.charged_results = [{
            "target_k": 0,
            "nelecastot": (1, 0),
        }]

        with self.assertRaisesRegex(ValueError, "has no CI vector"):
            kmcpdft._get_charged_kcas_rdm_context(mc)

    def test_make_one_charged_casdm1s_uses_sector_context(self):
        solver = RecordingSolver()
        mc = make_mc(solver, target_k=2, nkpts=3, ncas=2)
        mc.charged_results = [{
            "target_k": 2,
            "ci": "charged-ci",
            "nelecastot": (3, 2),
        }]

        casdm1s = kmcpdft.make_one_casdm1s_charged_kcas(mc)

        self.assertEqual(casdm1s.shape, (2, 6, 6))
        self.assertEqual(
            solver.calls,
            [("make_rdm1s", "charged-ci", 6, (3, 2),
              {"nkpts": 3, "target_k": 2})],
        )

    def test_make_one_charged_casdm1s_accepts_ci_and_sector_override(self):
        solver = RecordingSolver(nroots=2)
        mc = make_mc(solver, target_k=None, nkpts=2, ncas=1)
        mc.charged_results = [
            {"target_k": 0, "ci": "sector-0", "nelecastot": (1, 0)},
            {"target_k": 1, "ci": "sector-1", "nelecastot": (1, 0)},
        ]

        kmcpdft.make_one_casdm1s_charged_kcas(
            mc,
            ci=["override-root-0", "override-root-1"],
            state=1,
            target_k=-1,
        )

        self.assertEqual(
            solver.calls,
            [("make_rdm1s", "override-root-1", 2, (1, 0),
              {"nkpts": 2, "target_k": 1})],
        )

    def test_make_one_charged_casdm1s_rejects_invalid_shape(self):
        class BadShapeSolver(RecordingSolver):
            def make_rdm1s(self, ci, norb, nelec, **kwargs):
                return np.zeros((norb, norb))

        mc = make_mc(BadShapeSolver(), target_k=0, nkpts=2, ncas=1)
        mc.charged_results = [{
            "target_k": 0,
            "ci": "charged-ci",
            "nelecastot": (1, 0),
        }]

        with self.assertRaisesRegex(ValueError, "charged KCASCI 1-RDM"):
            kmcpdft.make_one_casdm1s_charged_kcas(mc)

    def test_make_one_charged_casdm2_uses_rdm12_and_sector_context(self):
        calls = []

        class RDM12Solver:
            nroots = 1

            def make_rdm12(self, ci, norb, nelec, **kwargs):
                calls.append((ci, norb, nelec, kwargs))
                return np.eye(norb), np.zeros((norb,) * 4)

        mc = make_mc(RDM12Solver(), target_k=-1, nkpts=3, ncas=2)
        mc.charged_results = [{
            "target_k": 2,
            "ci": "charged-ci",
            "nelecastot": (3, 2),
        }]

        casdm2 = kmcpdft.make_one_casdm2_charged_kcas(mc)

        self.assertEqual(casdm2.shape, (6, 6, 6, 6))
        self.assertEqual(
            calls,
            [("charged-ci", 6, (3, 2),
              {"nkpts": 3, "target_k": 2})],
        )

    def test_make_one_charged_casdm2_falls_back_to_make_rdm2(self):
        solver = RecordingSolver()
        mc = make_mc(solver, target_k=1, nkpts=2, ncas=1)
        mc.charged_results = [{
            "target_k": 1,
            "ci": "charged-ci",
            "nelecastot": (1, 0),
        }]

        casdm2 = kmcpdft.make_one_casdm2_charged_kcas(mc)

        self.assertEqual(casdm2.shape, (2, 2, 2, 2))
        self.assertEqual(
            solver.calls,
            [("make_rdm2", "charged-ci", 2, (1, 0),
              {"nkpts": 2, "target_k": 1})],
        )

    def test_make_one_charged_casdm2_accepts_root_and_sector_override(self):
        solver = RecordingSolver(nroots=2)
        mc = make_mc(solver, target_k=None, nkpts=2, ncas=1)
        mc.charged_results = [
            {"target_k": 0, "ci": "sector-0", "nelecastot": (1, 0)},
            {"target_k": 1, "ci": "sector-1", "nelecastot": (1, 0)},
        ]

        kmcpdft.make_one_casdm2_charged_kcas(
            mc,
            ci=["override-root-0", "override-root-1"],
            state=1,
            target_k=3,
        )

        self.assertEqual(
            solver.calls,
            [("make_rdm2", "override-root-1", 2, (1, 0),
              {"nkpts": 2, "target_k": 1})],
        )

    def test_make_one_charged_casdm2_rejects_invalid_shape(self):
        class BadShapeSolver(RecordingSolver):
            def make_rdm2(self, ci, norb, nelec, **kwargs):
                return np.zeros((norb, norb))

        mc = make_mc(BadShapeSolver(), target_k=0, nkpts=2, ncas=1)
        mc.charged_results = [{
            "target_k": 0,
            "ci": "charged-ci",
            "nelecastot": (1, 0),
        }]

        with self.assertRaisesRegex(ValueError, "charged KCASCI 2-RDM"):
            kmcpdft.make_one_casdm2_charged_kcas(mc)

    def test_make_one_casdm1s_passes_target_sector(self):
        solver = RecordingSolver()
        mc = make_mc(solver, target_k=5)

        casdm1s = pbc_dms.make_one_casdm1s_kcas(mc, "ci")

        self.assertEqual(casdm1s.shape, (2, 6, 6))
        self.assertEqual(
            solver.calls,
            [("make_rdm1s", "ci", 6, (3, 3),
              {"nkpts": 3, "target_k": 2})],
        )

    def test_make_one_casdm1s_selects_requested_root(self):
        solver = RecordingSolver(nroots=2)
        mc = make_mc(solver, target_k=1, nkpts=2, ncas=1)

        pbc_dms.make_one_casdm1s_kcas(
            mc, ["root-0", "root-1"], state=1,
        )

        _, ci, norb, nelec, kwargs = solver.calls[0]
        self.assertEqual(ci, "root-1")
        self.assertEqual(norb, 2)
        self.assertEqual(nelec, (2, 2))
        self.assertEqual(kwargs, {"nkpts": 2, "target_k": 1})

    def test_neutral_context_unpacks_scalar_electron_count(self):
        solver = RecordingSolver()
        mc = make_mc(
            solver, target_k=0, nkpts=2, ncas=2, nelecas=2,
        )
        mc.cell = SimpleNamespace(spin=0)

        context = pbc_dms._get_kcas_rdm_context(mc, "ci")

        self.assertEqual(context[2], 4)
        self.assertEqual(context[3], (2, 2))

    def test_make_one_casdm2_passes_target_sector(self):
        solver = RecordingSolver()
        mc = make_mc(solver, target_k=-1)

        casdm2 = pbc_dms.make_one_casdm2_kcas(mc, "ci")

        self.assertEqual(casdm2.shape, (6, 6, 6, 6))
        self.assertEqual(
            solver.calls,
            [("make_rdm2", "ci", 6, (3, 3),
              {"nkpts": 3, "target_k": 2})],
        )

    def test_make_one_casdm2_falls_back_to_make_rdm12(self):
        calls = []

        class RDM12Solver:
            nroots = 1

            def make_rdm12(self, ci, norb, nelec, **kwargs):
                calls.append((ci, norb, nelec, kwargs))
                return np.eye(norb), np.zeros((norb,) * 4)

        mc = make_mc(RDM12Solver(), target_k=0, nkpts=2, ncas=1)
        casdm2 = pbc_dms.make_one_casdm2_kcas(mc, "ci")

        self.assertEqual(casdm2.shape, (2, 2, 2, 2))
        self.assertEqual(
            calls,
            [("ci", 2, (2, 2), {"nkpts": 2, "target_k": 0})],
        )

    def test_invalid_rdm_shapes_are_rejected(self):
        class BadShapeSolver(RecordingSolver):
            def make_rdm1s(self, ci, norb, nelec, **kwargs):
                return np.zeros((norb, norb))

            def make_rdm2(self, ci, norb, nelec, **kwargs):
                return np.zeros((norb, norb))

        mc = make_mc(BadShapeSolver(), nkpts=2, ncas=1)
        with self.assertRaisesRegex(ValueError, "1-RDM shape"):
            pbc_dms.make_one_casdm1s_kcas(mc, "ci")
        with self.assertRaisesRegex(ValueError, "2-RDM shape"):
            pbc_dms.make_one_casdm2_kcas(mc, "ci")


class KCASPDFTKBlockTests(unittest.TestCase):

    def test_casdm1s_to_kpts_extracts_diagonal_k_blocks(self):
        nkpts, ncas = 3, 2
        ncastot = nkpts * ncas
        casdm1s = np.zeros((2, ncastot, ncastot), dtype=complex)
        expected = np.empty((2, nkpts, ncas, ncas), dtype=complex)
        for spin in range(2):
            for k in range(nkpts):
                block = np.arange(4).reshape(2, 2)
                block = block + 10 * k + 100 * spin + 1j * (spin + k)
                p0 = k * ncas
                casdm1s[spin, p0:p0 + ncas, p0:p0 + ncas] = block
                expected[spin, k] = block

        result = pbc_dms.casdm1s_to_kpts(
            casdm1s, nkpts, ncas,
        )
        np.testing.assert_allclose(result, expected)

    def test_casdm1s_to_kpts_rejects_forbidden_blocks(self):
        casdm1s = np.zeros((2, 2, 2))
        casdm1s[0, 0, 1] = 1e-3

        with self.assertRaisesRegex(ValueError, "momentum-forbidden"):
            pbc_dms.casdm1s_to_kpts(
                casdm1s, nkpts=2, ncas=1, momentum_tol=1e-8,
            )

        result = pbc_dms.casdm1s_to_kpts(
            casdm1s, nkpts=2, ncas=1, momentum_tol=None,
        )
        np.testing.assert_allclose(result, 0.0)

    def test_cascm2_to_kpts_uses_kconserv(self):
        nkpts, ncas = 3, 2
        ncastot = nkpts * ncas
        kconserv = make_kconserv(nkpts)
        cascm2 = np.zeros((ncastot,) * 4, dtype=complex)
        full = cascm2.reshape(
            nkpts, ncas, nkpts, ncas, nkpts, ncas, nkpts, ncas,
        )
        expected = np.empty(
            (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
            dtype=complex,
        )
        for k1 in range(nkpts):
            for k2 in range(nkpts):
                for k3 in range(nkpts):
                    k4 = kconserv[k1, k2, k3]
                    values = np.arange(ncas ** 4).reshape((ncas,) * 4)
                    values = values + k1 + 10 * k2 + 100 * k3 + 0.5j
                    full[k1, :, k2, :, k3, :, k4, :] = values
                    expected[k1, k2, k3] = values

        result = pbc_dms.cascm2_to_kpts(
            cascm2, nkpts, ncas, kconserv,
        )
        np.testing.assert_allclose(result, expected)

    def test_cascm2_to_kpts_rejects_forbidden_blocks(self):
        nkpts = 2
        kconserv = make_kconserv(nkpts)
        cascm2 = np.zeros((nkpts,) * 4)
        forbidden_k4 = 1 - kconserv[0, 0, 0]
        cascm2[0, 0, 0, forbidden_k4] = 1e-4

        with self.assertRaisesRegex(ValueError, "momentum-forbidden"):
            pbc_dms.cascm2_to_kpts(
                cascm2, nkpts, 1, kconserv,
            )

    def test_make_kcas_rdms_kpts_builds_complex_cumulant(self):
        nkpts, ncas = 2, 1
        ncastot = nkpts * ncas
        kconserv = make_kconserv(nkpts)
        casdm1s = np.zeros((2, ncastot, ncastot), dtype=complex)
        casdm1s[0] = np.diag([0.8 + 0.0j, 0.2 + 0.0j])
        casdm1s[1] = np.diag([0.3 + 0.0j, 0.7 + 0.0j])
        casdm2 = np.zeros((ncastot,) * 4, dtype=complex)

        dm1s_kpts, cascm2_kpts = pbc_dms.make_kcas_rdms_kpts(
            casdm1s, casdm2, nkpts, ncas, kconserv,
        )

        cumulant = dm2_cumulant_complex(casdm2, casdm1s)
        expected_cumulant = pbc_dms.cascm2_to_kpts(
            cumulant, nkpts, ncas, kconserv,
        )
        np.testing.assert_allclose(
            dm1s_kpts[:, :, 0, 0], [[0.8, 0.2], [0.3, 0.7]],
        )
        np.testing.assert_allclose(cascm2_kpts, expected_cumulant)

    def test_casdm1s_kpts_to_dm1s_matches_explicit_transformation(self):
        nkpts, nao, nmo, ncore, ncas = 3, 4, 4, 1, 2
        rng = np.random.default_rng(23)
        mo_coeff = (
            rng.normal(size=(nkpts, nao, nmo))
            + 1j * rng.normal(size=(nkpts, nao, nmo))
        )
        casdm1s = (
            rng.normal(size=(2, nkpts, ncas, ncas))
            + 1j * rng.normal(size=(2, nkpts, ncas, ncas))
        )

        result = pbc_dms.casdm1s_kpts_to_dm1s(
            object(), casdm1s, mo_coeff, ncore,
        )

        expected = np.empty((2, nkpts, nao, nao), dtype=complex)
        for k in range(nkpts):
            core = mo_coeff[k, :, :ncore]
            active = mo_coeff[k, :, ncore:ncore + ncas]
            core_dm = core @ core.conj().T
            for spin in range(2):
                expected[spin, k] = (
                    active @ casdm1s[spin, k] @ active.conj().T + core_dm
                )
        np.testing.assert_allclose(result, expected)

    def test_kfci_rdms_obey_momentum_block_layout(self):
        nkpts, ncas = 2, 2
        norb = nkpts * ncas
        nelec = (1, 1)
        kconserv = make_kconserv(nkpts)
        rng = np.random.default_rng(19)

        for target_k in range(nkpts):
            with self.subTest(target_k=target_k):
                solver = direct_spin1_kfci.FCISolver(
                    nkpts=nkpts, target_k=target_k,
                )
                sector_size = direct_spin1_kfci.sector_size(
                    norb, nelec, nkpts, target_k,
                )
                ci = (
                    rng.normal(size=sector_size)
                    + 1j * rng.normal(size=sector_size)
                )
                ci /= np.linalg.norm(ci)

                casdm1s = solver.make_rdm1s(ci, norb, nelec)
                _, casdm2 = solver.make_rdm12(ci, norb, nelec)
                casdm1s_kpts, cascm2_kpts = \
                    pbc_dms.make_kcas_rdms_kpts(
                        casdm1s, casdm2, nkpts, ncas, kconserv,
                    )

                self.assertEqual(
                    casdm1s_kpts.shape, (2, nkpts, ncas, ncas),
                )
                self.assertEqual(
                    cascm2_kpts.shape,
                    (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
                )
                self.assertAlmostEqual(np.trace(casdm1s[0]).real, 1.0)
                self.assertAlmostEqual(np.trace(casdm1s[1]).real, 1.0)


class KCASPDFTOnTopEnergyTests(unittest.TestCase):

    def test_energy_ot_uses_bloch_kspace_preparation(self):
        nkpts, ncas = 2, 1
        ncastot = nkpts * ncas
        kconserv = make_kconserv(nkpts)
        casdm1s = np.zeros((2, ncastot, ncastot))
        casdm2 = np.zeros((ncastot,) * 4)
        mo_coeff = np.ones((nkpts, 1, 1), dtype=complex)
        prepared_dm1s = np.ones((2, nkpts, ncas, ncas))
        prepared_cm2 = np.ones(
            (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
        )
        ot = SimpleNamespace(
            xctype="LDA",
            kconserv=kconserv,
            cell=object(),
            kpts=np.zeros((nkpts, 3)),
        )

        with mock.patch.object(
                pbc_dms, "make_kcas_rdms_kpts",
                return_value=(prepared_dm1s, prepared_cm2)) as prepare, \
             mock.patch.object(
                otfnalperiodic, "_energy_ot_from_kpts",
                return_value=1.25) as evaluate, \
             mock.patch.object(
                otfnalperiodic, "get_mo_coeff_k2R_wokmf",
                side_effect=AssertionError("unexpected Wannier transform")):
            energy = otfnalperiodic.otfnalperiodic_kpts.energy_ot(
                ot, casdm1s, casdm2, mo_coeff, ncore=0,
                rdm_representation="bloch",
            )

        self.assertEqual(energy, 1.25)
        prepare.assert_called_once_with(
            casdm1s, casdm2, nkpts, ncas, kconserv,
            momentum_tol=1e-8,
        )
        evaluate.assert_called_once()
        self.assertEqual(
            evaluate.call_args.args,
            (ot, prepared_dm1s, prepared_cm2, mo_coeff, 0, kconserv),
        )

    def test_existing_wannier_path_uses_shared_kspace_backend(self):
        nkpts, ncas = 2, 1
        ncastot = nkpts * ncas
        casdm1s = np.asarray([
            np.diag([0.8, 0.2]),
            np.diag([0.3, 0.7]),
        ])
        casdm2 = np.zeros((ncastot,) * 4)
        mo_coeff = np.ones((nkpts, 1, 1), dtype=complex)
        mo_phase = np.zeros((nkpts, ncas, ncastot), dtype=complex)
        mo_phase[0, 0, 0] = 1.0
        mo_phase[1, 0, 1] = 1.0
        kconserv = make_kconserv(nkpts)
        ot = SimpleNamespace(
            xctype="LDA",
            cell=object(),
            kpts=np.zeros((nkpts, 3)),
            kmesh=(nkpts, 1, 1),
        )

        def transform_cm2(cascm2, phase, ks):
            self.assertIs(phase, mo_phase)
            return np.full((ncas,) * 4, sum(ks), dtype=complex)

        with mock.patch.object(
                otfnalperiodic, "get_mo_coeff_k2R_wokmf",
                return_value=(None, None, mo_phase)), \
             mock.patch.object(
                otfnalperiodic.kpts_helper, "get_kconserv",
                return_value=kconserv), \
             mock.patch.object(
                otfnalperiodic, "_basis_transform_casdm2_kpts",
                side_effect=transform_cm2), \
             mock.patch.object(
                otfnalperiodic, "_energy_ot_from_kpts",
                return_value=2.5) as evaluate:
            energy = otfnalperiodic.otfnalperiodic_kpts.energy_ot(
                ot, casdm1s, casdm2, mo_coeff, ncore=0,
            )

        self.assertEqual(energy, 2.5)
        prepared_dm1s = evaluate.call_args.args[1]
        np.testing.assert_allclose(
            prepared_dm1s[:, :, 0, 0], [[0.8, 0.2], [0.3, 0.7]],
        )
        np.testing.assert_array_equal(evaluate.call_args.args[5], kconserv)

    def test_shared_kspace_backend_runs_grid_evaluation(self):
        nkpts, nao, ncas = 2, 2, 1
        kconserv = make_kconserv(nkpts)
        casdm1s_kpts = np.zeros((2, nkpts, ncas, ncas))
        casdm1s_kpts[0, :, 0, 0] = [0.8, 0.6]
        casdm1s_kpts[1, :, 0, 0] = [0.2, 0.4]
        cascm2_kpts = np.zeros(
            (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
        )
        mo_coeff = np.tile(np.eye(nao, dtype=complex), (nkpts, 1, 1))
        weight = np.asarray([0.25, 0.75])
        seen_dms = []
        seen_eval = {}

        class FakeNumInt:
            def _gen_rho_evaluator(self, cell, dm, hermi, with_lapl):
                seen_dms.append(np.asarray(dm))

                def make_rho(idm, ao, mask, xctype):
                    return np.asarray([0.5, 0.75])

                return make_rho, 1, nao

            def block_loop(self, cell, grids, nao_arg, **kwargs):
                self_nao = nao_arg
                if self_nao != nao:
                    raise AssertionError((self_nao, nao))
                ao = np.ones((nkpts, 1, weight.size, nao), dtype=complex)
                yield ao, ao, None, weight, None

        def eval_ot(rho, Pi, **kwargs):
            seen_eval["rho"] = rho
            seen_eval["Pi"] = Pi
            return (np.asarray([2.0, 4.0]),)

        ot = SimpleNamespace(
            xctype="LDA",
            dens_deriv=0,
            Pi_deriv=0,
            _numint=FakeNumInt(),
            cell=object(),
            grids=object(),
            kpts=np.zeros((nkpts, 3)),
            eval_ot=eval_ot,
            verbose=logger.QUIET,
        )

        with mock.patch.object(
                otfnalperiodic, "get_ontop_pair_density_kpts",
                return_value=np.asarray([0.1, 0.2])) as get_pi:
            energy = otfnalperiodic._energy_ot_from_kpts(
                ot, casdm1s_kpts, cascm2_kpts, mo_coeff,
                ncore=1, kconserv=kconserv,
            )

        self.assertAlmostEqual(energy, 3.5)
        self.assertEqual(len(seen_dms), 2)
        self.assertEqual(seen_dms[0].shape, (nkpts, nao, nao))
        self.assertEqual(seen_dms[1].shape, (nkpts, nao, nao))
        self.assertEqual(seen_eval["rho"].shape, (2, 1, weight.size))
        self.assertEqual(seen_eval["Pi"].shape, (1, weight.size))
        self.assertIs(get_pi.call_args.args[3], cascm2_kpts)


class KCASPDFTWavefunctionEnergyTests(unittest.TestCase):

    def test_energy_mcwfn_from_kpts_hybrid_components(self):
        nkpts, ncas, nao = 2, 1, 1
        casdm1s_kpts = np.zeros((2, nkpts, ncas, ncas))
        cascm2_kpts = np.full(
            (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), 0.2,
        )
        dm1s_kpts = np.asarray([
            [[[1.0]], [[0.5]]],
            [[[0.25]], [[0.75]]],
        ])
        hcore = np.asarray([[[2.0]], [[3.0]]])
        vj_spin = np.asarray([
            [[[0.4]], [[0.6]]],
            [[[0.1]], [[0.2]]],
        ])
        vk_spin = np.asarray([
            [[[0.3]], [[0.4]]],
            [[[0.2]], [[0.1]]],
        ])
        h2eff = np.full(cascm2_kpts.shape, 0.1)
        jk_calls = []

        class FakeSCF:
            def get_jk(self, cell, dm_kpts, kpts, **kwargs):
                jk_calls.append((np.asarray(dm_kpts), kwargs))
                return vj_spin, vk_spin

        class FakeNumInt:
            def rsh_and_hybrid_coeff(self, otxc, spin):
                self.seen = (otxc, spin)
                return 0.0, 0.0, (0.25, 0.25)

        numint = FakeNumInt()
        ot = SimpleNamespace(_numint=numint, otxc="tPBE0")
        mc = SimpleNamespace(
            otfnal=ot,
            mo_coeff=np.ones((nkpts, nao, ncas), dtype=complex),
            ci="ci",
            verbose=logger.QUIET,
            nkpts=nkpts,
            ncas=ncas,
            ncore=0,
            kpts=np.zeros((nkpts, 3)),
            cell=SimpleNamespace(spin=0),
            _scf=FakeSCF(),
            energy_nuc=lambda: 1.5,
            get_hcore=lambda **kwargs: hcore,
        )

        with mock.patch.object(
                pbc_dms, "casdm1s_kpts_to_dm1s",
                return_value=dm1s_kpts), \
             mock.patch.object(
                kmcpdft, "get_h2eff_kpts",
                return_value=h2eff) as get_h2eff:
            result = kmcpdft._energy_mcwfn_from_kpts(
                mc, casdm1s_kpts, cascm2_kpts,
            )

        dm1_kpts = dm1s_kpts.sum(axis=0)
        energy_one = np.einsum("kij,kji->", hcore, dm1_kpts) / nkpts
        vj_kpts = vj_spin.sum(axis=0)
        energy_j = 0.5 * np.einsum(
            "kij,kji->", vj_kpts, dm1_kpts,
        ) / nkpts
        energy_x = -0.5 * (
            np.einsum("kij,kji->", vk_spin[0], dm1s_kpts[0])
            + np.einsum("kij,kji->", vk_spin[1], dm1s_kpts[1])
        ) / nkpts
        energy_c = np.einsum(
            "abcuvxy,abcuvxy->", h2eff, cascm2_kpts,
        ) / (2 * nkpts)
        reference = (
            1.5 + energy_one + energy_j
            + 0.25 * energy_x + 0.25 * energy_c
        )

        np.testing.assert_allclose(result, reference)
        self.assertEqual(numint.seen, ("tPBE0", 0))
        self.assertEqual(len(jk_calls), 1)
        np.testing.assert_allclose(jk_calls[0][0], dm1s_kpts)
        get_h2eff.assert_called_once_with(mc, mc.mo_coeff)

    def test_energy_mcwfn_from_kpts_pure_functional_skips_exchange_and_eris(self):
        nkpts, ncas = 2, 1
        casdm1s_kpts = np.zeros((2, nkpts, ncas, ncas))
        cascm2_kpts = np.zeros(
            (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
        )
        dm1s_kpts = np.ones((2, nkpts, 1, 1)) * 0.5
        get_h2eff = mock.Mock(side_effect=AssertionError("unexpected ERIs"))

        class FakeSCF:
            def get_jk(self, cell, dm_kpts, kpts, **kwargs):
                self.dm_kpts = np.asarray(dm_kpts)
                self.kwargs = kwargs
                return np.ones((nkpts, 1, 1)) * 0.4, None

        fake_scf = FakeSCF()
        numint = SimpleNamespace(
            rsh_and_hybrid_coeff=lambda otxc, spin:
                (0.0, 0.0, (0.0, 0.0)),
        )
        mc = SimpleNamespace(
            otfnal=SimpleNamespace(_numint=numint, otxc="tPBE"),
            mo_coeff=np.ones((nkpts, 1, 1)),
            ci="ci",
            verbose=logger.QUIET,
            nkpts=nkpts,
            ncas=ncas,
            ncore=0,
            kpts=np.zeros((nkpts, 3)),
            cell=SimpleNamespace(spin=0),
            _scf=fake_scf,
            energy_nuc=lambda: 1.0,
            get_hcore=lambda **kwargs: np.ones((nkpts, 1, 1)),
        )

        with mock.patch.object(
                pbc_dms, "casdm1s_kpts_to_dm1s",
                return_value=dm1s_kpts), \
             mock.patch.object(
                kmcpdft, "get_h2eff_kpts", get_h2eff):
            result = kmcpdft._energy_mcwfn_from_kpts(
                mc, casdm1s_kpts, cascm2_kpts,
            )

        # Per cell: Vnn=1, E1=1, EJ=0.2.
        self.assertAlmostEqual(result, 2.2)
        np.testing.assert_allclose(fake_scf.dm_kpts, dm1s_kpts.sum(axis=0))
        self.assertEqual(fake_scf.kwargs, {"hermi": 1, "with_k": False})
        get_h2eff.assert_not_called()



class KCASPDFTRoutingTests(unittest.TestCase):

    def test_kmcpdft_energy_tot_is_real(self):
        mc = object.__new__(kmcpdft._MCPDFTCPLX)
        with mock.patch.object(
                kmcpdft._PeriodicMCPDFT, "energy_tot",
                return_value=(1.25 + 2.0j, 0.5 + 0.25j)):
            e_tot, e_ot = mc.energy_tot()

        self.assertEqual(e_tot, 1.25)
        self.assertEqual(e_ot, 0.5 + 0.25j)

    def test_second_order_krhf_is_routed_as_mean_field(self):
        _, kmf, _, _ = build_periodic_h2()
        second_order_kmf = kmf.newton()
        conventional_mc = SimpleNamespace()
        momentum_mc = SimpleNamespace(charge=0)
        conventional_pdft = object()
        momentum_pdft = object()

        make_conventional_mc = mock.Mock(return_value=conventional_mc)
        make_conventional_pdft = mock.Mock(return_value=conventional_pdft)
        result = pbc_mcpdft._MCPDFT(
            make_conventional_mc, second_order_kmf, "tPBE", 2, 2,
            ncore=0, get_mcpdft_child_class=make_conventional_pdft,
        )

        self.assertIs(result, conventional_pdft)
        make_conventional_mc.assert_called_once_with(
            second_order_kmf, 2, 2, ncore=0,
        )
        make_conventional_pdft.assert_called_once_with(
            conventional_mc, "tPBE",
        )

        with mock.patch.object(
                pbc_mcpdft.pbc_mcscf, "KCASCI",
                return_value=momentum_mc) as make_momentum_mc, \
             mock.patch.object(
                pbc_mcpdft, "get_kcas_mcpdft_child_class",
                return_value=momentum_pdft) as make_momentum_pdft:
            result = pbc_mcpdft.KCASCI(
                second_order_kmf, "tPBE", 2, 2, ncore=0,
                target_k=1,
            )

        self.assertIs(result, momentum_pdft)
        make_momentum_mc.assert_called_once_with(
            second_order_kmf, 2, 2, ncore=0, target_k=1,
        )
        make_momentum_pdft.assert_called_once_with(
            momentum_mc, "tPBE",
        )

    def test_kcasci_default_preserves_conventional_route(self):
        sentinel = object()
        with mock.patch.object(
                pbc_mcpdft, "_MCPDFT", return_value=sentinel) as factory:
            result = pbc_mcpdft.KCASCI(
                "kmf", "tPBE", 2, (1, 1), ncore=0,
            )

        self.assertIs(result, sentinel)
        factory.assert_called_once_with(
            pbc_mcpdft.pbc_mcscf.CASCI,
            "kmf", "tPBE", 2, (1, 1),
            ncore=0, frozen=None,
        )

    def test_momentum_route_constructs_kcasci_with_target(self):
        kmf = object()
        kmc = SimpleNamespace()
        pdft = object()
        with mock.patch.object(
                pbc_mcpdft, "_sanity_check_for_kmf",
                return_value=kmf), \
             mock.patch.object(
                pbc_mcpdft.pbc_mcscf, "KCASCI",
                return_value=kmc) as make_kcasci, \
             mock.patch.object(
                pbc_mcpdft, "get_kcas_mcpdft_child_class",
                return_value=pdft) as make_child:
            result = pbc_mcpdft.KCASCI(
                kmf, "tPBE", 2, (1, 1), ncore=0,
                target_k=3,
                grids_level=4,
            )

        self.assertIs(result, pdft)
        make_kcasci.assert_called_once_with(
            kmf, 2, (1, 1), ncore=0, target_k=3,
        )
        make_child.assert_called_once_with(
            kmc, "tPBE", grids_level=4,
        )

    def test_charged_momentum_route_preserves_all_sector_sweep(self):
        kmf = object()
        kmc = SimpleNamespace(charge=1)
        pdft = object()
        with mock.patch.object(
                pbc_mcpdft, "_sanity_check_for_kmf",
                return_value=kmf), \
             mock.patch.object(
                pbc_mcpdft.pbc_mcscf, "KCASCI",
                return_value=kmc) as make_kcasci, \
             mock.patch.object(
                pbc_mcpdft, "get_charged_kcas_mcpdft_child_class",
                return_value=pdft) as make_child:
            result = pbc_mcpdft.KCASCI(
                kmf, "tPBE", 2, (1, 1), ncore=0,
                charge=1, charged_spin=1, grids_level=4,
            )

        self.assertIs(result, pdft)
        make_kcasci.assert_called_once_with(
            kmf, 2, (1, 1), ncore=0, charge=1,
            target_k=None, charged_spin=1,
        )
        make_child.assert_called_once_with(
            kmc, "tPBE", grids_level=4,
        )

    def test_existing_kcasci_sector_is_preserved_or_reset(self):
        from mrh.my_pyscf.pbc.mcscf.kcasci import PBCKCASCI

        kmc = object.__new__(PBCKCASCI)
        kmc._scf = object()
        kmc.nkpts = 3
        kmc.target_k = 2
        pdft = object()
        with mock.patch.object(
                pbc_mcpdft, "_sanity_check_for_kmf"), \
             mock.patch.object(
                pbc_mcpdft, "get_kcas_mcpdft_child_class",
                return_value=pdft) as make_child:
            result = pbc_mcpdft.KCASCI(
                kmc, "tPBE", 2, (1, 1),
            )
            self.assertIs(result, pdft)
            make_child.assert_called_once_with(kmc, "tPBE")

            reset_kmc = SimpleNamespace(charge=0)
            with mock.patch.object(
                    pbc_mcpdft.pbc_mcscf, "KCASCI",
                    return_value=reset_kmc) as reset_kcasci, \
                 mock.patch.object(pbc_mcpdft.logger, "warn") as warn:
                result = pbc_mcpdft.KCASCI(
                    kmc, "tPBE", 2, (1, 1),
                    target_k=1,
                )
            self.assertIs(result, pdft)
            reset_kcasci.assert_called_once_with(
                kmc._scf, 2, (1, 1), ncore=None, target_k=1,
            )
            make_child.assert_called_with(reset_kmc, "tPBE")
            warn.assert_called_once()

    def test_existing_charged_kcasci_uses_charged_wrapper(self):
        from mrh.my_pyscf.pbc.mcscf.kcasci import ChargedPBCKCASCI

        kmc = object.__new__(ChargedPBCKCASCI)
        kmc._scf = object()
        kmc.nkpts = 3
        kmc.target_k = None
        kmc.charge = -1
        kmc.charged_spin = 1
        pdft = object()
        with mock.patch.object(
                pbc_mcpdft, "_sanity_check_for_kmf"), \
             mock.patch.object(
                pbc_mcpdft, "get_charged_kcas_mcpdft_child_class",
                return_value=pdft) as make_child:
            result = pbc_mcpdft.KCASCI(
                kmc, "tPBE", 2, (1, 1),
            )

        self.assertIs(result, pdft)
        make_child.assert_called_once_with(kmc, "tPBE")

        with mock.patch.object(
                pbc_mcpdft, "_sanity_check_for_kmf"):
            reset_kmc = SimpleNamespace(charge=1)
            with mock.patch.object(
                    pbc_mcpdft.pbc_mcscf, "KCASCI",
                    return_value=reset_kmc) as reset_kcasci, \
                 mock.patch.object(
                    pbc_mcpdft, "get_charged_kcas_mcpdft_child_class",
                    return_value=pdft) as make_reset_child, \
                 mock.patch.object(pbc_mcpdft.logger, "warn") as warn:
                result = pbc_mcpdft.KCASCI(
                    kmc, "tPBE", 2, (1, 1),
                    charge=1,
                )
            self.assertIs(result, pdft)
            reset_kcasci.assert_called_once_with(
                kmc._scf, 2, (1, 1), ncore=None, target_k=None,
                charge=1, charged_spin=1,
            )
            make_reset_child.assert_called_once_with(reset_kmc, "tPBE")
            warn.assert_called_once()

    def test_momentum_route_rejects_ignored_options(self):
        with self.assertRaisesRegex(ValueError, "charged_spin requires"):
            pbc_mcpdft.KCASCI(
                "kmf", "tPBE", 2, (1, 1), charged_spin=1,
            )
        with self.assertRaisesRegex(ValueError, "Frozen orbitals"):
            pbc_mcpdft.KCASCI(
                "kmf", "tPBE", 2, (1, 1), target_k=0, frozen=1,
            )

    def test_kcas_pdft_mixin_uses_momentum_methods(self):
        self.assertEqual(
            kmcpdft._MCPDFTCPLX._mcwfn_rdm_representation,
            "wannier",
        )
        self.assertEqual(
            kmcpdft._kCASPDFT._mcwfn_rdm_representation,
            "bloch",
        )
        self.assertIs(
            kmcpdft._kCASPDFT.make_one_casdm1s,
            pbc_dms.make_one_casdm1s_kcas,
        )
        self.assertIs(
            kmcpdft._kCASPDFT.make_one_casdm2,
            pbc_dms.make_one_casdm2_kcas,
        )
        self.assertIs(
            kmcpdft._kCASPDFT.energy_mcwfn,
            kmcpdft.energy_mcwfn,
        )
        self.assertIs(
            kmcpdft._kCASPDFT.energy_dft,
            kmcpdft.energy_dft_kcas,
        )

    def test_charged_kcas_pdft_mixin_uses_sector_rdm_methods(self):
        self.assertIs(
            kmcpdft._kChargedCASPDFT.make_one_casdm1s,
            kmcpdft.make_one_casdm1s_charged_kcas,
        )
        self.assertIs(
            kmcpdft._kChargedCASPDFT.make_one_casdm2,
            kmcpdft.make_one_casdm2_charged_kcas,
        )
        self.assertIs(
            kmcpdft._kChargedCASPDFT.energy_tot,
            kmcpdft.energy_tot_charged_kcas,
        )

    def test_charged_energy_selects_one_sector(self):
        ot = SimpleNamespace(
            otxc="tPBE", reset=mock.Mock(),
        )
        casdm1s = np.zeros((2, 2, 2))
        casdm2 = np.zeros((2, 2, 2, 2))
        mc = SimpleNamespace(
            nkpts=2,
            target_k=None,
            charged_results=[
                {"target_k": 0, "ci": "sector-0"},
                {"target_k": 1, "ci": "sector-1"},
            ],
            otfnal=ot,
            mol="mol",
            mo_coeff="mo",
            verbose=logger.QUIET,
            make_one_casdm1s=mock.Mock(return_value=casdm1s),
            make_one_casdm2=mock.Mock(return_value=casdm2),
            energy_mcwfn=mock.Mock(return_value=1.25 + 0.2j),
            energy_dft=mock.Mock(return_value=0.5 - 0.1j),
        )

        result = kmcpdft.energy_tot_charged_kcas(
            mc, target_k=-1, state=1,
        )

        self.assertEqual(result, (1.75, 0.5 - 0.1j))
        ot.reset.assert_called_once_with(mol="mol")
        mc.make_one_casdm1s.assert_called_once_with(
            ci="sector-1", state=1, target_k=1,
        )
        mc.make_one_casdm2.assert_called_once_with(
            ci="sector-1", state=1, target_k=1,
        )
        mc.energy_mcwfn.assert_called_once_with(
            ot=ot, mo_coeff="mo", casdm1s=casdm1s,
            casdm2=casdm2, verbose=logger.QUIET,
        )

    def test_charged_compute_pdft_energy_loops_over_sectors_and_roots(self):
        energy_tot = mock.Mock(
            side_effect=lambda **kwargs: (
                10 * kwargs["target_k"] + kwargs["state"] + 0.5,
                10 * kwargs["target_k"] + kwargs["state"] + 0.25,
            ),
        )
        mc = SimpleNamespace(
            mo_coeff="mo",
            target_k=None,
            nkpts=2,
            charge=1,
            charged_results=[
                {
                    "target_k": 0, "charge": 1, "nkpts": 2,
                    "ci": ["0-root-0", "0-root-1"],
                    "e_tot": -1.0,
                },
                {
                    "target_k": 1, "charge": 1, "nkpts": 2,
                    "ci": ["1-root-0", "1-root-1"],
                    "e_tot": -0.9,
                },
            ],
            fcisolver=SimpleNamespace(nroots=2),
            otfnal=SimpleNamespace(verbose=logger.QUIET),
            verbose=logger.QUIET,
            energy_tot=energy_tot,
        )

        output = kmcpdft._kChargedCASPDFT.compute_pdft_energy_(mc)

        np.testing.assert_allclose(mc.e_tot, [[0.5, 1.5], [10.5, 11.5]])
        np.testing.assert_allclose(mc.e_ot, [[0.25, 1.25], [10.25, 11.25]])
        self.assertEqual(len(mc.charged_pdft_results), 2)
        self.assertEqual(
            [item["e_mcscf"] for item in mc.charged_pdft_results],
            [-1.0, -0.9],
        )
        self.assertIs(output[2], mc.charged_pdft_results)
        self.assertEqual(
            [(call.kwargs["target_k"], call.kwargs["state"],
              call.kwargs["ci"])
             for call in energy_tot.call_args_list],
            [
                (0, 0, ["0-root-0", "0-root-1"]),
                (0, 1, ["0-root-0", "0-root-1"]),
                (1, 0, ["1-root-0", "1-root-1"]),
                (1, 1, ["1-root-0", "1-root-1"]),
            ],
        )

    def test_charged_pdft_band_energies_use_pdft_results(self):
        from mrh.my_pyscf.pbc.mcscf import kcasci

        pdft_results = [
            {
                "target_k": 0, "charge": 1, "nkpts": 2,
                "e_tot": -1.25,
            },
        ]
        mc = SimpleNamespace(
            charged_pdft_results=pdft_results,
            charged_results=[{"target_k": 0, "e_tot": -9.0}],
            charge=1,
            nkpts=2,
            _scf=SimpleNamespace(kpts="kpts"),
            cell="cell",
            kconserv="kconserv",
        )
        sentinel = object()
        with mock.patch.object(
                kcasci, "_get_kmom_for_kcasci",
                return_value="kmom"), \
             mock.patch.object(
                kcasci, "compute_band_energies",
                return_value=sentinel) as compute:
            result = kmcpdft._kChargedCASPDFT.band_energies(
                mc, -1.5, root=1, per_cell=True,
                reference_target_k=1,
            )

        self.assertIs(result, sentinel)
        compute.assert_called_once_with(
            pdft_results,
            -1.5,
            charge=1,
            root=1,
            kpts="kpts",
            nkpts=2,
            per_cell=True,
            reference_target_k=1,
            kmom="kmom",
            cell="cell",
            kconserv="kconserv",
        )

    def test_energy_dft_kcas_delegates_to_direct_ot_method(self):
        ot = SimpleNamespace(energy_ot=mock.Mock(return_value=0.75))
        mc = SimpleNamespace(
            otfnal=ot,
            mo_coeff="mo",
            ci="ci",
            ncore=2,
            max_memory=param.MAX_MEMORY,
        )
        casdm1s = np.zeros((2, 1, 1))
        casdm2 = np.zeros((1, 1, 1, 1))

        result = kmcpdft.energy_dft_kcas(
            mc, casdm1s=casdm1s, casdm2=casdm2,
        )

        self.assertEqual(result, 0.75)
        ot.energy_ot.assert_called_once()
        self.assertEqual(
            ot.energy_ot.call_args.args,
            (casdm1s, casdm2, "mo", 2),
        )
        self.assertEqual(
            ot.energy_ot.call_args.kwargs["rdm_representation"],
            "bloch",
        )


if __name__ == "__main__":
    unittest.main()
