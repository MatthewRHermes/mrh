#!/usr/bin/env python

"""Unit tests for periodic kLAS-PDFT adapters."""

import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc.mcpdft import klaspdft_helper
from mrh.my_pyscf.pbc.mcpdft import klaspdft
from mrh.my_pyscf.pbc import mcpdft as pbc_mcpdft
from mrh.my_pyscf.pbc import mcscf as pbc_mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.klasci import PBCLASCINoSymm
from mrh.my_pyscf.pbc.mcscf.klasscf import PBCLASSCFNoSymm


class _RecordingFragmentSolver:
    """Small fragment solver returning prescribed complex density matrices."""

    def __init__(self, dm1a, dm1b, dm2, spin):
        self.dm1a = np.asarray(dm1a, dtype=np.complex128)
        self.dm1b = np.asarray(dm1b, dtype=np.complex128)
        self.dm2 = np.asarray(dm2, dtype=np.complex128)
        self.spin = spin
        self.seen_ci = []

    def make_rdm1s(self, ci, norb, nelec):
        self.seen_ci.append(("dm1", ci, norb, tuple(nelec)))
        return self.dm1a, self.dm1b

    def make_rdm2(self, ci, norb, nelec):
        self.seen_ci.append(("dm2", ci, norb, tuple(nelec)))
        return self.dm2


def _make_fake_klas():
    roots = []
    for iroot in range(2):
        shift = 0.1 * iroot
        frag0 = _RecordingFragmentSolver(
            [[0.75 + shift]], [[0.25 - shift]], [[[[0.2 + shift]]]],
            spin=1,
        )
        frag1 = _RecordingFragmentSolver(
            [[0.4 - shift]], [[0.6 + shift]], [[[[0.3 - shift]]]],
            spin=-1,
        )
        roots.append((frag0, frag1))
    return SimpleNamespace(
        nroots=2,
        ncas_sub=np.asarray([1, 1]),
        nelecas_sub=np.asarray([[1, 0], [0, 1]]),
        fciboxes=[
            SimpleNamespace(fcisolvers=[roots[0][0], roots[1][0]]),
            SimpleNamespace(fcisolvers=[roots[0][1], roots[1][1]]),
        ],
        ci=[[np.asarray([[0.0]]), np.asarray([[1.0]])],
            [np.asarray([[10.0]]), np.asarray([[11.0]])]],
        stdout=None,
        verbose=0,
    )


class KLASPDFTRDMTests(unittest.TestCase):

    def test_context_selects_one_root_from_every_fragment(self):
        klas = _make_fake_klas()
        solvers, ci, ncas_sub, nelecas_sub = \
            klaspdft_helper._get_klas_rdm_context(klas, state=1)

        self.assertIs(solvers[0], klas.fciboxes[0].fcisolvers[1])
        self.assertIs(solvers[1], klas.fciboxes[1].fcisolvers[1])
        np.testing.assert_array_equal(ci[0], [[1.0]])
        np.testing.assert_array_equal(ci[1], [[11.0]])
        np.testing.assert_array_equal(ncas_sub, [1, 1])
        np.testing.assert_array_equal(nelecas_sub, [[1, 0], [0, 1]])

    def test_product_state_rdms_are_complex_and_have_full_active_shape(self):
        klas = _make_fake_klas()
        casdm1s, casdm2 = klaspdft_helper.make_one_casdm12_klas(
            klas, state=0,
        )

        self.assertEqual(casdm1s.shape, (2, 2, 2))
        self.assertEqual(casdm2.shape, (2, 2, 2, 2))
        self.assertTrue(np.issubdtype(casdm1s.dtype, np.complexfloating))
        self.assertTrue(np.issubdtype(casdm2.dtype, np.complexfloating))
        np.testing.assert_allclose(casdm1s[0], np.diag([0.75, 0.4]))
        np.testing.assert_allclose(casdm1s[1], np.diag([0.25, 0.6]))
        self.assertAlmostEqual(casdm2[0, 0, 0, 0], 0.2)
        self.assertAlmostEqual(casdm2[1, 1, 1, 1], 0.3)
        self.assertAlmostEqual(casdm2[0, 0, 1, 1], 1.0)
        self.assertAlmostEqual(casdm2[1, 1, 0, 0], 1.0)
        self.assertAlmostEqual(casdm2[0, 1, 1, 0], -0.45)
        self.assertAlmostEqual(casdm2[1, 0, 0, 1], -0.45)

    def test_rdm_builder_passes_the_selected_ci_to_fragment_solvers(self):
        klas = _make_fake_klas()
        klaspdft_helper.make_one_casdm12_klas(klas, state=1)

        for ifrag in range(2):
            solver = klas.fciboxes[ifrag].fcisolvers[1]
            self.assertEqual(len(solver.seen_ci), 3)
            self.assertTrue(all(
                np.array_equal(item[1], [[10.0 * ifrag + 1.0]])
                for item in solver.seen_ci
            ))

    def test_invalid_state_is_rejected(self):
        klas = _make_fake_klas()
        with self.assertRaisesRegex(TypeError, "state must be an integer"):
            klaspdft_helper.make_one_casdm12_klas(klas, state=0.5)
        with self.assertRaisesRegex(ValueError, "state must lie"):
            klaspdft_helper.make_one_casdm12_klas(klas, state=2)

    def test_missing_fragment_ci_is_rejected(self):
        klas = _make_fake_klas()
        klas.ci[1][0] = None
        with self.assertRaisesRegex(ValueError, "Fragment 1 CI vector"):
            klaspdft_helper.make_one_casdm12_klas(klas, state=0)


class KLASPDFTPhaseTests(unittest.TestCase):

    @staticmethod
    def _make_phase_context():
        return SimpleNamespace(
            _scf=object(),
            kmesh=(2, 1, 1),
            kpts=np.zeros((2, 3)),
            ncore=1,
            ncas=1,
            ncas_sub=np.asarray([1, 1]),
            mo_coeff=np.asarray([
                [[10.0, 1.0, 20.0], [30.0, 2.0, 40.0]],
                [[50.0, 3.0, 60.0], [70.0, 4.0, 80.0]],
            ], dtype=np.complex128),
        )

    def test_phase_uses_the_kLAS_wannier_active_orbitals(self):
        klas = self._make_phase_context()
        mo_phase = np.asarray([
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ], dtype=np.complex128)

        with mock.patch.object(
                klaspdft_helper, "get_wannier_orbs",
                return_value=("wannier", "indices", mo_phase)) as get_phase:
            result = klaspdft_helper.get_klas_mo_phase(klas)

        np.testing.assert_array_equal(result, mo_phase)
        self.assertIs(get_phase.call_args.args[0], klas._scf)
        self.assertEqual(get_phase.call_args.args[1], klas.kmesh)
        np.testing.assert_array_equal(
            get_phase.call_args.args[2],
            klas.mo_coeff[:, :, 1:2],
        )

    def test_nonunitary_phase_is_rejected(self):
        klas = self._make_phase_context()
        bad_phase = np.ones((2, 1, 2), dtype=np.complex128)
        with mock.patch.object(
                klaspdft_helper, "get_wannier_orbs",
                return_value=(None, None, bad_phase)):
            with self.assertRaisesRegex(ValueError, "must be unitary"):
                klaspdft_helper.get_klas_mo_phase(klas)

    def test_phase_dimensions_are_validated_before_wannierization(self):
        klas = self._make_phase_context()
        klas.ncas_sub = np.asarray([1])
        with mock.patch.object(
                klaspdft_helper, "get_wannier_orbs") as get_phase:
            with self.assertRaisesRegex(ValueError, r"sum\(ncas_sub\)"):
                klaspdft_helper.get_klas_mo_phase(klas)
        get_phase.assert_not_called()


def _make_kconserv(nkpts):
    """Return a cyclic momentum-conservation table for test meshes."""
    return np.fromfunction(
        lambda k1, k2, k3: (k1 - k2 + k3) % nkpts,
        (nkpts, nkpts, nkpts),
        dtype=int,
    ).astype(int)


class KLASPDFTKBlockTests(unittest.TestCase):

    def test_wannier_rdms_are_transformed_to_expected_k_blocks(self):
        rng = np.random.default_rng(24)
        nkpts, ncas = 2, 2
        ncastot = nkpts * ncas
        phase_matrix = np.linalg.qr(
            rng.normal(size=(ncastot, ncastot))
            + 1j * rng.normal(size=(ncastot, ncastot)),
        )[0]
        mo_phase = phase_matrix.reshape(nkpts, ncas, ncastot)
        casdm1s = (
            rng.normal(size=(2, ncastot, ncastot))
            + 1j * rng.normal(size=(2, ncastot, ncastot))
        )
        casdm1s += casdm1s.swapaxes(-1, -2).conj()
        casdm2 = (
            rng.normal(size=(ncastot,) * 4)
            + 1j * rng.normal(size=(ncastot,) * 4)
        )
        kconserv = _make_kconserv(nkpts)

        casdm1s_kpts, cascm2_kpts = \
            klaspdft_helper.make_klas_rdms_kpts(
                casdm1s, casdm2, mo_phase, kconserv,
            )

        self.assertEqual(casdm1s_kpts.shape, (2, 2, 2, 2))
        self.assertEqual(cascm2_kpts.shape, (2, 2, 2, 2, 2, 2, 2))
        expected_dm1s = np.stack([
            np.stack([
                mo_phase[k] @ dm1 @ mo_phase[k].conj().T
                for k in range(nkpts)
            ])
            for dm1 in casdm1s
        ])
        np.testing.assert_allclose(casdm1s_kpts, expected_dm1s)

        cascm2 = klaspdft_helper.dm2_cumulant_complex(casdm2, casdm1s)
        for k1 in range(nkpts):
            for k2 in range(nkpts):
                for k3 in range(nkpts):
                    k4 = kconserv[k1, k2, k3]
                    expected = np.einsum(
                        "ap,bq,pqrs,cr,ds->abcd",
                        mo_phase[k1].conj(),
                        mo_phase[k2],
                        cascm2,
                        mo_phase[k3].conj(),
                        mo_phase[k4],
                        optimize=True,
                    )
                    np.testing.assert_allclose(
                        cascm2_kpts[k1, k2, k3], expected,
                    )

    def test_k_blocks_are_invariant_to_wannier_gauge_rotation(self):
        rng = np.random.default_rng(91)
        nkpts, ncas = 2, 1
        ncastot = nkpts * ncas
        phase = np.linalg.qr(
            rng.normal(size=(ncastot, ncastot))
            + 1j * rng.normal(size=(ncastot, ncastot)),
        )[0]
        gauge = np.linalg.qr(
            rng.normal(size=(ncastot, ncastot))
            + 1j * rng.normal(size=(ncastot, ncastot)),
        )[0]
        dm1s = (
            rng.normal(size=(2, ncastot, ncastot))
            + 1j * rng.normal(size=(2, ncastot, ncastot))
        )
        dm1s += dm1s.swapaxes(-1, -2).conj()
        dm2 = (
            rng.normal(size=(ncastot,) * 4)
            + 1j * rng.normal(size=(ncastot,) * 4)
        )
        kconserv = _make_kconserv(nkpts)

        reference = klaspdft_helper.make_klas_rdms_kpts(
            dm1s, dm2, phase.reshape(nkpts, ncas, ncastot), kconserv,
        )
        dm1s_rot = np.einsum(
            "pi,spq,qj->sij",
            gauge, dm1s, gauge.conj(),
            optimize=True,
        )
        dm2_rot = np.einsum(
            "pi,qj,pqrs,rk,sl->ijkl",
            gauge.conj(), gauge, dm2, gauge.conj(), gauge,
            optimize=True,
        )
        phase_rot = (phase @ gauge.conj()).reshape(
            nkpts, ncas, ncastot,
        )
        rotated = klaspdft_helper.make_klas_rdms_kpts(
            dm1s_rot, dm2_rot, phase_rot, kconserv,
        )

        np.testing.assert_allclose(rotated[0], reference[0], atol=1e-11)
        np.testing.assert_allclose(rotated[1], reference[1], atol=1e-11)

    def test_k_block_layout_is_validated(self):
        casdm1s = np.zeros((2, 2, 2))
        casdm2 = np.zeros((2, 2, 2, 2))
        mo_phase = np.eye(2).reshape(2, 1, 2)
        with self.assertRaisesRegex(ValueError, "kconserv shape"):
            klaspdft_helper.make_klas_rdms_kpts(
                casdm1s, casdm2, mo_phase, np.zeros((2, 2), dtype=int),
            )


class KLASPDFTEnergyRoutingTests(unittest.TestCase):

    def test_mixin_uses_kLAS_specific_energy_and_rdm_methods(self):
        self.assertIs(
            klaspdft._kLASPDFT.make_one_casdm1s,
            klaspdft_helper.make_one_casdm1s_klas,
        )
        self.assertIs(
            klaspdft._kLASPDFT.make_one_casdm2,
            klaspdft_helper.make_one_casdm2_klas,
        )
        self.assertIs(
            klaspdft._kLASPDFT.energy_mcwfn,
            klaspdft.energy_mcwfn_klas,
        )
        self.assertIs(
            klaspdft._kLASPDFT.energy_dft,
            klaspdft.energy_dft_klas,
        )
        self.assertIs(
            klaspdft._kLASPDFT.energy_tot,
            klaspdft.energy_tot_klas,
        )

    def test_total_energy_builds_and_shares_one_rdm_and_phase_set(self):
        casdm1s = np.zeros((2, 2, 2), dtype=complex)
        casdm2 = np.zeros((2, 2, 2, 2), dtype=complex)
        mo_phase = np.eye(2, dtype=complex).reshape(2, 1, 2)
        ot = SimpleNamespace(otxc="tPBE", reset=mock.Mock())
        mc = SimpleNamespace(
            otfnal=ot,
            mol="mol",
            mo_coeff="mo",
            ci="ci",
            verbose=0,
            energy_mcwfn=mock.Mock(return_value=1.25),
            energy_dft=mock.Mock(return_value=0.5),
        )
        with mock.patch.object(
                klaspdft_helper, "make_one_casdm12_klas",
                return_value=(casdm1s, casdm2)) as make_rdms, \
             mock.patch.object(
                klaspdft_helper, "get_klas_mo_phase",
                return_value=mo_phase) as get_phase:
            result = klaspdft.energy_tot_klas(mc, state=1)

        self.assertEqual(result, (1.75, 0.5))
        ot.reset.assert_called_once_with(mol="mol")
        make_rdms.assert_called_once_with(mc, ci="ci", state=1)
        get_phase.assert_called_once_with(mc, mo_coeff="mo")
        for evaluator in (mc.energy_mcwfn, mc.energy_dft):
            self.assertIs(evaluator.call_args.kwargs["casdm1s"], casdm1s)
            self.assertIs(evaluator.call_args.kwargs["casdm2"], casdm2)
            self.assertIs(evaluator.call_args.kwargs["mo_phase"], mo_phase)

    def test_on_top_energy_reuses_shared_kspace_backend(self):
        casdm1s = np.zeros((2, 2, 2), dtype=complex)
        casdm2 = np.zeros((2, 2, 2, 2), dtype=complex)
        mo_phase = np.eye(2, dtype=complex).reshape(2, 1, 2)
        prepared_dm1s = np.zeros((2, 2, 1, 1), dtype=complex)
        prepared_cm2 = np.zeros((2, 2, 2, 1, 1, 1, 1), dtype=complex)
        kconserv = _make_kconserv(2)
        mc = SimpleNamespace(
            otfnal="ot",
            mo_coeff="mo",
            ci="ci",
            max_memory=1234,
            kconserv=kconserv,
            ncore=1,
        )
        with mock.patch.object(
                klaspdft_helper, "make_klas_rdms_kpts",
                return_value=(prepared_dm1s, prepared_cm2)) as prepare, \
             mock.patch.object(
                klaspdft, "_energy_ot_from_kpts",
                return_value=0.75) as evaluate:
            result = klaspdft.energy_dft_klas(
                mc,
                mo_coeff="mo",
                ci="ci",
                ot="ot",
                casdm1s=casdm1s,
                casdm2=casdm2,
                mo_phase=mo_phase,
            )

        self.assertEqual(result, 0.75)
        prepare.assert_called_once_with(
            casdm1s, casdm2, mo_phase, kconserv,
        )
        evaluate.assert_called_once_with(
            "ot", prepared_dm1s, prepared_cm2, "mo", 1, kconserv,
            max_memory=1234, hermi=1,
        )

    def test_wavefunction_energy_uses_kLAS_phase_and_integrals(self):
        casdm1s = np.zeros((2, 2, 2), dtype=complex)
        casdm2 = np.zeros((2, 2, 2, 2), dtype=complex)
        mo_phase = np.eye(2, dtype=complex).reshape(2, 1, 2)
        h2eff = np.zeros((2, 2, 2, 2), dtype=complex)
        mc = SimpleNamespace(
            mo_coeff="mo",
            ci="ci",
            get_h2cas=mock.Mock(return_value=h2eff),
        )
        with mock.patch.object(
                klaspdft.kmcpdft, "energy_mcwfn",
                return_value=1.5) as evaluate:
            result = klaspdft.energy_mcwfn_klas(
                mc,
                mo_coeff="mo",
                ci="ci",
                ot="ot",
                state=1,
                casdm1s=casdm1s,
                casdm2=casdm2,
                mo_phase=mo_phase,
                verbose=4,
            )

        self.assertEqual(result, 1.5)
        mc.get_h2cas.assert_called_once_with("mo")
        evaluate.assert_called_once_with(
            mc,
            mo_coeff="mo",
            ci="ci",
            ot="ot",
            state=1,
            casdm1s=casdm1s,
            casdm2=casdm2,
            verbose=4,
            mo_phase=mo_phase,
            h2eff=h2eff,
        )


def _make_bare_klas(klas_class):
    """Create an uninitialized typed kLAS object for routing tests."""
    klas = object.__new__(klas_class)
    klas._scf = SimpleNamespace(kpts=np.zeros((2, 3)))
    klas.nroots = 1
    klas.ncas = 1
    klas.ncas_sub = np.asarray([1, 1])
    klas.mo_coeff = np.zeros((2, 1, 1), dtype=complex)
    klas.ci = [[np.asarray([[1.0]])], [np.asarray([[1.0]])]]
    return klas


class KLASPDFTPublicRoutingTests(unittest.TestCase):

    def test_klasci_accepts_only_existing_klasci(self):
        klas = _make_bare_klas(PBCLASCINoSymm)
        sentinel = object()
        with mock.patch.object(
                klaspdft, "get_klas_mcpdft_child_class",
                return_value=sentinel) as wrap:
            result = pbc_mcpdft.KLASCI(klas, "tPBE")

        self.assertIs(result, sentinel)
        wrap.assert_called_once_with(klas, "tPBE")
        with self.assertRaisesRegex(TypeError, "existing KLASCI"):
            pbc_mcpdft.KLASCI(SimpleNamespace(), "tPBE")

    def test_klasscf_accepts_only_existing_klasscf(self):
        klasscf = _make_bare_klas(PBCLASSCFNoSymm)
        sentinel = object()
        with mock.patch.object(
                klaspdft, "get_klas_mcpdft_child_class",
                return_value=sentinel) as wrap:
            result = pbc_mcpdft.KLASSCF(klasscf, "tPBE")

        self.assertIs(result, sentinel)
        wrap.assert_called_once_with(klasscf, "tPBE")
        with self.assertRaisesRegex(TypeError, "existing KLASCI"):
            pbc_mcpdft.KLASCI(klasscf, "tPBE")

    def test_initial_public_scope_rejects_multiple_roots(self):
        klas = _make_bare_klas(PBCLASCINoSymm)
        klas.nroots = 2
        with self.assertRaisesRegex(NotImplementedError, "supports one root"):
            pbc_mcpdft.KLASCI(klas, "tPBE")

    def test_child_factory_copies_state_without_running_parent_init(self):
        class FakeKLAS:
            """Minimal completed kLAS-like object for factory testing."""

            def kernel(self):
                raise AssertionError("The parent kernel must not run")

        original = FakeKLAS()
        original._keys = {"original"}
        original._scf = SimpleNamespace()
        original.e_tot = -1.25
        original.mo_coeff = "mo"
        original.ci = "ci"
        original.e_cas = -0.5
        original.mo_energy = "mo-energy"
        original.max_memory = 500
        original.verbose = 0

        def initialize_grids(pdft, ot, grids_attr=None):
            pdft.otfnal = SimpleNamespace(
                name=ot,
                grids=SimpleNamespace(**(grids_attr or {})),
            )

        with mock.patch.object(
                klaspdft._kLASPDFT, "_init_ot_grids",
                initialize_grids):
            pdft = klaspdft.get_klas_mcpdft_child_class(
                original, "ot", grids_level=4,
            )

        self.assertIsNot(pdft, original)
        self.assertEqual(pdft.mo_coeff, "mo")
        self.assertEqual(pdft.ci, "ci")
        self.assertEqual(pdft.e_mcscf, -1.25)
        self.assertEqual(pdft.grids.level, 4)
        pdft.optimize_mcscf_()
        self.assertEqual(pdft.e_mcscf, -1.25)


class KLASPDFTEndToEndTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.a = np.diag([4.0, 10.0, 10.0])
        cell.atom = "H 0 0 0; H 1.5 0 0"
        cell.basis = "sto-3g"
        cell.unit = "Angstrom"
        cell.precision = 1e-10
        cell.ke_cutoff = 20
        cell.verbose = lib.logger.QUIET
        cell.build()

        cls.kmesh = (2, 1, 1)
        kpts = cell.make_kpts(cls.kmesh, wrap_around=True)
        kmf = scf.KRHF(cell, kpts=kpts).density_fit()
        kmf.exxdiv = None
        kmf.max_cycle = 0
        kmf.kernel()
        mo_avas = avas.kernel(kmf, ["H 1s"], minao=cell.basis)[2]

        klas = pbc_mcscf.KLASCI(
            kmf, 2, (1, 1), kmesh=cls.kmesh, trans_sym=False,
        )
        mo_guess = klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_avas,
        )
        klas.kernel(mo_guess)
        cls.klas = klas

        klasscf = pbc_mcscf.KLASSCF(
            kmf, 2, (1, 1), kmesh=cls.kmesh, trans_sym=False,
        )
        klasscf.max_cycle_macro = 0
        klasscf.kernel(np.array(mo_guess, copy=True))
        cls.klasscf = klasscf

    def test_klasci_pdft_functional_coverage(self):
        references = {
            "tLDA": -0.9015232976311289,
            "tPBE": -1.0085601356590703,
            "tPBE0": -0.9663974749640505,
        }
        mo_before = np.array(self.klas.mo_coeff, copy=True)
        e_klas_before = self.klas.e_tot

        for otxc, reference in references.items():
            pdft = pbc_mcpdft.KLASCI(
                self.klas, otxc, grids_level=1,
            )
            result = pdft.kernel()
            self.assertAlmostEqual(pdft.e_tot.real, reference, 7)
            self.assertAlmostEqual(result[0].real, reference, 7)
            self.assertLess(abs(pdft.e_tot.imag), 1e-12)

        np.testing.assert_allclose(self.klas.mo_coeff, mo_before)
        np.testing.assert_allclose(self.klas.e_tot, e_klas_before)

    def test_klasscf_intake_runs_fixed_wavefunction_pdft(self):
        pdft = pbc_mcpdft.KLASSCF(
            self.klasscf, "tLDA", grids_level=1,
        )
        result = pdft.kernel()

        self.assertTrue(np.isfinite(result[0]))
        self.assertTrue(np.isfinite(result[1]))
        np.testing.assert_allclose(pdft.e_mcscf, self.klasscf.e_tot)
        self.assertLess(abs(pdft.e_tot.imag), 1e-12)

    def test_product_state_rdm_electron_traces(self):
        casdm1s, casdm2 = klaspdft_helper.make_one_casdm12_klas(
            self.klas,
        )
        ncastot = np.prod(self.kmesh) * self.klas.ncas

        self.assertEqual(casdm1s.shape, (2, ncastot, ncastot))
        self.assertEqual(casdm2.shape, (ncastot,) * 4)
        self.assertAlmostEqual(np.trace(casdm1s[0]).real, 2.0, 10)
        self.assertAlmostEqual(np.trace(casdm1s[1]).real, 2.0, 10)
        self.assertLess(abs(np.trace(casdm1s[0]).imag), 1e-12)
        self.assertLess(abs(np.trace(casdm1s[1]).imag), 1e-12)


if __name__ == "__main__":
    unittest.main()
