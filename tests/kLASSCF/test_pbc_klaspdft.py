#!/usr/bin/env python

"""Tests for periodic kLAS-PDFT.

KLASPDFTRDMTests: Fragment root selection and product-state RDM assembly.
KLASPDFTPhaseTests: Active-orbital Wannier phases and transformation checks.
KLASPDFTKBlockTests: RDM k-block transformations and Wannier gauge invariance.
KLASPDFTEnergyRoutingTests: Wavefunction and on-top energy evaluation paths.
KLASPDFTPublicRoutingTests: KLASCI/KLASSCF inputs and PDFT wrapper construction.
KLASPDFTEndToEndTests: Fixed-orbital LASCI and fully optimized LASSCF H2 energies,
electron counts, and fixed-wavefunction PDFT reuse.
KLASPDFTMolecularComparisonTests: nk=3 kLASPDFT versus molecular LASPDFT
using transferred GDF integrals, orbitals and CI. The molecular calculation
uses periodic supercell AOs and translated copies of the primitive grid to
match the density and quadrature; energies agree within 1e-8 Ha per cell.
"""

import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from pyscf import dft, lib
from pyscf.pbc import gto, scf
from pyscf.pbc.tools import k2gamma

from mrh.my_pyscf import mcpdft as molecular_mcpdft
from mrh.my_pyscf.pbc.util.klas_to_las import unpack_klas

from mrh.my_pyscf.pbc.mcpdft import klaspdft_helper
from mrh.my_pyscf.pbc.mcpdft import klaspdft, kmcpdft, otfnalperiodic
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

    def test_mixin_inherits_shared_energy_methods(self):
        self.assertIs(klaspdft._kLASPDFT.energy_mcwfn, kmcpdft.energy_mcwfn)
        self.assertIs(klaspdft._kLASPDFT.energy_dft, kmcpdft.energy_dft)
        self.assertIs(klaspdft._kLASPDFT.energy_tot, kmcpdft._kMCPDFT.energy_tot)
        self.assertEqual(klaspdft._kLASPDFT._mcwfn_rdm_representation, "klas")
        self.assertIs(klaspdft._kLASPDFT.make_one_casdm1s,
                      klaspdft_helper.make_one_casdm1s_klas)
        self.assertIs(klaspdft._kLASPDFT.make_one_casdm2,
                      klaspdft_helper.make_one_casdm2_klas)

    def test_shared_preparation_matches_kLAS_helper(self):
        cell = gto.Cell()
        cell.a = np.diag([4., 10., 10.])
        cell.atom = "H 0 0 0; H 1.5 0 0"
        cell.basis = "sto-3g"
        cell.verbose = 0
        cell.build()
        kmesh = (2, 1, 1)
        kpts = cell.make_kpts(kmesh, wrap_around=True)
        kmf = scf.KRHF(cell, kpts=kpts)
        overlap = kmf.get_ovlp()
        mo = np.asarray([np.linalg.inv(np.linalg.cholesky(sk)).conj().T
                         for sk in overlap], dtype=complex)
        ncas = cell.nao_nr()
        context = SimpleNamespace(_scf=kmf, cell=cell, kpts=kpts, kmesh=kmesh,
                                  ncore=0, ncas=ncas, ncas_sub=[ncas, ncas])
        rng = np.random.default_rng(12)
        size = 2 * ncas
        dm1s = rng.normal(size=(2, size, size)).astype(complex)
        dm1s += dm1s.swapaxes(-1, -2).conj()
        dm2 = rng.normal(size=(size,) * 4) + 1j * rng.normal(size=(size,) * 4)
        phase = klaspdft_helper.get_klas_mo_phase(context, mo_coeff=mo)
        kconserv = _make_kconserv(2)
        expected = klaspdft_helper.make_klas_rdms_kpts(dm1s, dm2, phase, kconserv)
        # The functional context has no SCF object; both routes must agree.
        ot = SimpleNamespace(cell=cell, kpts=kpts, kmesh=kmesh)
        for obj in (context, ot):
            actual = otfnalperiodic._prepare_kpts_rdms(
                obj, dm1s, dm2, mo, 0, "klas", 1e-8)
            np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
            np.testing.assert_allclose(actual[1], expected[1], atol=1e-12)
            np.testing.assert_array_equal(actual[2], kconserv)

    def test_shared_dft_selects_kLAS_representation(self):
        ot = SimpleNamespace(energy_ot=mock.Mock(return_value=0.75))
        mc = SimpleNamespace(otfnal=ot, mo_coeff="mo", ci="ci", ncore=1,
                             max_memory=1234, _mcwfn_rdm_representation="klas")
        dm1s = np.zeros((2, 2, 2), dtype=complex)
        dm2 = np.zeros((2,) * 4, dtype=complex)
        self.assertEqual(kmcpdft.energy_dft(mc, casdm1s=dm1s, casdm2=dm2), 0.75)
        self.assertEqual(ot.energy_ot.call_args.kwargs["rdm_representation"], "klas")

    def test_shared_wavefunction_energy_preserves_Wannier_integral_contraction(self):
        rng = np.random.default_rng(23)
        dm1s = np.zeros((2, 2, 2), dtype=complex)
        dm2 = rng.normal(size=(2,) * 4).astype(complex)
        h2 = rng.normal(size=(2,) * 4).astype(complex)
        prepared = (np.zeros((2, 2, 1, 1)), np.zeros((2, 2, 2, 1, 1, 1, 1)),
                    _make_kconserv(2))
        mc = SimpleNamespace(mo_coeff="mo", ci="ci", ncore=0, ncas=1, nkpts=2,
                             _mcwfn_rdm_representation="klas",
                             get_h2cas=mock.Mock(return_value=h2))
        with mock.patch.object(kmcpdft, "_prepare_kpts_rdms", return_value=prepared), \
             mock.patch.object(kmcpdft, "_energy_mcwfn_from_kpts", return_value=1.5) as evaluate:
            self.assertEqual(kmcpdft.energy_mcwfn(mc, casdm1s=dm1s, casdm2=dm2), 1.5)
        mc.get_h2cas.assert_called_once_with("mo")
        self.assertAlmostEqual(evaluate.call_args.kwargs["cumulant_energy"],
                               np.tensordot(h2, dm2, axes=4) / 4)


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
        klasscf.conv_tol_grad = 1e-6
        klasscf.max_cycle_macro = 100
        ci0 = [[np.array([[1., 0.], [0., 0.]], dtype=complex)]
               for _ in range(np.prod(cls.kmesh))]
        klasscf.kernel(mo_coeff=np.array(mo_guess, copy=True), ci0=ci0)
        cls.klasscf = klasscf

    def test_klasci_pdft_functional_coverage(self):
        references = {
            "tLDA": -0.9015147705690623,
            "tPBE": -1.0085486386570832,
            "tPBE0": -0.9663888510398926,
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
        # References use fully optimized orbitals from the complete rotation map.
        references = {
            "tLDA": -0.9015233081062843,
            "tPBE": -1.0085601497846977,
            "tPBE0": -0.9663974855582702,
        }
        self.assertTrue(self.klasscf.converged)
        for otxc, reference in references.items():
            with self.subTest(otxc=otxc):
                pdft = pbc_mcpdft.KLASSCF(self.klasscf, otxc, grids_level=1)
                result = pdft.kernel()
                self.assertAlmostEqual(pdft.e_tot.real, reference, delta=1e-7)
                self.assertAlmostEqual(result[0].real, reference, delta=1e-7)
                self.assertAlmostEqual(pdft.e_mcscf.real, self.klasscf.e_tot.real,
                                       delta=1e-8)

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


class _ReplicatedPeriodicGrids(dft.gen_grid.Grids):
    """Fixed supercell quadrature retained when PDFT resets its grids."""
    def __init__(self, mol, primitive_grid, translations):
        self._coords = np.concatenate([primitive_grid.coords + r for r in translations])
        self._weights = np.tile(primitive_grid.weights, len(translations))
        super().__init__(mol)
        self.reset(mol)

    def reset(self, mol=None):
        super().reset(mol)
        self.coords = self._coords
        self.weights = self._weights
        return self


class _PeriodicSupercellAO:
    """Use molecular density/on-top contractions with periodic supercell AOs."""
    def __init__(self, supercell):
        self.supercell = supercell

    def eval_ao(self, mol, coords, deriv=0, **kwargs):
        name = 'GTOval_sph' if deriv == 0 else f'GTOval_sph_deriv{deriv}'
        return self.supercell.pbc_eval_gto(name, coords, kpt=np.zeros(3))


class KLASPDFTMolecularComparisonTests(unittest.TestCase):
    def test_three_kpoints_match_molecular_laspdft(self):
        nk = 3
        cell = gto.Cell()
        cell.a = np.diag([4.0, 10.0, 10.0])
        cell.atom = 'H 0 0 0; H 1.5 0 0'
        cell.basis = '6-31G'
        cell.unit = 'Angstrom'
        cell.precision = 1e-10
        cell.verbose = 0
        cell.build()
        kmesh = (nk, 1, 1)
        kpts = cell.make_kpts(kmesh, wrap_around=True)
        kmf = scf.KRHF(cell, kpts=kpts).density_fit()
        kmf.exxdiv = None
        kmf.conv_tol = 1e-10
        kmf.kernel()
        self.assertTrue(kmf.converged)

        mo_avas = np.asarray(avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2],
                             dtype=complex).reshape(nk, cell.nao_nr(), -1)
        klas = pbc_mcscf.KLASSCF(kmf, 2, (1, 1), kmesh=kmesh)
        mo_guess = klas.localize_init_guess(['H 1s'], mo_coeff=mo_avas,
                                            stabilize_virtuals=True)
        klas.conv_tol_grad = 1e-6
        klas.max_cycle_macro = 100
        ci0 = [[np.array([[1., 0.], [0., 0.]], dtype=complex)] for _ in range(nk)]
        klas.kernel(mo_coeff=mo_guess, ci0=ci0)
        self.assertTrue(klas.converged)

        mo, _, ci, las = unpack_klas(klas)
        self.assertAlmostEqual(las.e_tot / nk, klas.e_tot.real, delta=1e-8)
        supercell = k2gamma.get_phase(cell, kpts, kmesh)[0]
        translations = k2gamma.translation_vectors_for_kmesh(cell, kmesh)

        for functional in ('tLDA', 'tPBE', 'tPBE0'):
            with self.subTest(functional=functional):
                periodic = pbc_mcpdft.KLASSCF(klas, functional, grids_level=1)
                periodic.kernel()
                molecular = molecular_mcpdft.LASSCF(las, functional, grids_level=1)
                molecular.grids = _ReplicatedPeriodicGrids(
                    las.mol, periodic.grids, translations)
                molecular.otfnal._numint.eval_ao = _PeriodicSupercellAO(supercell).eval_ao
                molecular.compute_pdft_energy_(mo_coeff=mo, ci=ci)

                self.assertAlmostEqual(molecular.e_tot / nk, periodic.e_tot.real,
                                       delta=1e-8)
                self.assertAlmostEqual(molecular.e_ot / nk, periodic.e_ot.real,
                                       delta=1e-8)
                self.assertAlmostEqual((molecular.e_tot-molecular.e_ot) / nk,
                                       (periodic.e_tot-periodic.e_ot).real, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
