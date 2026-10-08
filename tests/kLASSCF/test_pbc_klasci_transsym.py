#!/usr/bin/env python

import unittest
from unittest import mock

import numpy as np

from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf import klasci as klasci_module
from mrh.my_pyscf.pbc.mcscf.klasci import (
    PBCLASCINoSymm,
    PBCLASCITransSymm,
    kLASCI,
)
from mrh.my_pyscf.pbc.mcscf.productstate import (
    PBCProductStateFCISolver,
    PBCTransSymmImpureProductStateFCISolver,
)
from mrh.my_pyscf.pbc.util.transym import TranslationSymm

# Author: Bhavnesh Jangid

# Test-0: trans_sym=True should select PBCLASCITransSymm, validate the
#          Wannier orbitals and Hamiltonians, and use the translation-symmetric
#         product-state solver with the requested reference cell.
# Test-1: PBCLASCITransSymm and PBCLASCINoSymm should produce the same energy
#         when they use the same localized active orbitals.
# Test-2: The product-state CI helpers should select the reference-cell CI
#         vector and reconstruct independent fragment vectors with optional
#         translation phases.
# Test-3: The reference-cell initial guess should be generated from the real
#         H2 active-space Hamiltonian and preserve a supplied CI vector.
# Test-4: The reference-fragment Hamiltonian projection should match the
#         reference block selected from the parent full-fragment projection.
# Test-5: The reference-fragment CI gradient should match the corresponding
#         section of the parent full-fragment gradient.
# Test-6: The translation-symmetric product-state kernel should optimize only
#         the reference fragment and assemble all returned CI vectors.
# Test-7: Full one- and two-body density matrices should be assembled from the
#         reference-fragment density matrices.
# Test-8: Packed one- and two-electron Hamiltonians should reproduce every
#         translated block of the full Hamiltonians.
# Test-9: The packed reference-cell energy should reproduce the dense total
#         product-state energy after multiplication by the number of cells.



cell = kmf = mo_coeff = None
kmesh = [2, 1, 1]


def setUpModule():
    global cell, kmf, mo_coeff
    cell = gto.Cell()
    cell.a = np.diag([3.0, 10.0, 10.0])
    cell.atom = "H 0 0 0; H 0.74 0 0"
    cell.basis = "6-31g"
    cell.unit = "Angstrom"
    cell.precision = 1e-8
    cell.ke_cutoff = 20
    cell.verbose = 0
    cell.build()

    kpts = cell.make_kpts(kmesh, wrap_around=True)
    kmf = scf.KRHF(cell, kpts=kpts).density_fit()
    kmf.exxdiv = None
    kmf.max_cycle = 0
    kmf.kernel()
    mo_coeff = avas.kernel(kmf, ["H 1s"], minao=cell.basis)[2]


class KnownValues(unittest.TestCase):

    def test_trans_sym_checks_wannier_hamiltonians(self):
        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh, trans_sym=True, ref_cell=1,
        )
        mo_loc = trans_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )

        with mock.patch.object(
                klasci_module, "check_wannier_orbital_translation",
                wraps=klasci_module.check_wannier_orbital_translation
             ) as check_orbitals, \
             mock.patch.object(
                klasci_module, "check_h1e_translation",
                wraps=klasci_module.check_h1e_translation) as check_h1e, \
             mock.patch.object(
                klasci_module, "check_h2e_translation",
                wraps=klasci_module.check_h2e_translation) as check_h2e, \
             mock.patch.object(
                klasci_module, "PBCTransSymmImpureProductStateFCISolver",
                wraps=PBCTransSymmImpureProductStateFCISolver
             ) as trans_solver:
            trans_klas.kernel(mo_loc)

        self.assertIs(type(trans_klas), PBCLASCITransSymm)
        self.assertTrue(trans_klas.trans_sym)
        self.assertEqual(trans_klas.ref_cell, 1)
        self.assertEqual(len(trans_klas.ci), np.prod(kmesh))
        phase_per_frag = trans_klas.get_phase_per_frag(mo_loc)
        self.assertEqual(phase_per_frag.shape, (np.prod(kmesh),))
        self.assertLess(
            np.max(np.abs(phase_per_frag - np.ones(np.prod(kmesh)))),
            1e-12,
        )
        check_orbitals.assert_called_once()
        check_h1e.assert_called_once()
        check_h2e.assert_called_once()
        trans_solver.assert_called_once()
        self.assertEqual(trans_solver.call_args.kwargs["ref_cell"], 1)
        self.assertLess(
            np.max(np.abs(
                trans_solver.call_args.kwargs["phase_per_frag"]
                - phase_per_frag
            )), 1e-12,
        )
        self.assertTrue(callable(trans_solver.call_args.kwargs["pack_h1"]))
        self.assertTrue(callable(trans_solver.call_args.kwargs["pack_h2"]))

        ncas_sub = trans_klas.ncas_sub.copy()
        try:
            trans_klas.ncas_sub[1] += 1
            with self.assertRaisesRegex(
                    ValueError, "active-space consistency check failed"):
                trans_klas._sanity_check_active_space_consistency(mo_loc)
        finally:
            trans_klas.ncas_sub = ncas_sub

        with self.assertRaisesRegex(TypeError, "trans_sym must be a boolean"):
            kLASCI(kmf, 2, (1, 1), kmesh=kmesh, trans_sym="yes")
        with self.assertRaisesRegex(ValueError, "ref_cell must be in"):
            kLASCI(
                kmf, 2, (1, 1), kmesh=kmesh,
                trans_sym=True, ref_cell=np.prod(kmesh),
            )

    def test_kpts_must_match_mean_field(self):
        supplied_kpts = np.array(kmf.kpts, copy=True)
        klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh, kpts=supplied_kpts,
        )
        np.testing.assert_allclose(klas.kpts, kmf.kpts, atol=0.0, rtol=0.0)

        returned_kpts = klas.kpts
        returned_kpts[0, 0] += 1.0
        np.testing.assert_allclose(klas.kpts, kmf.kpts, atol=0.0, rtol=0.0)

        mismatched_kpts = np.array(kmf.kpts, copy=True)
        mismatched_kpts[0, 0] += 1e-4
        with self.assertRaisesRegex(ValueError, "kpts must match kmf.kpts"):
            kLASCI(
                kmf, 2, (1, 1), kmesh=kmesh, kpts=mismatched_kpts,
            )

    def test_trans_sym_class_api_and_energy(self):
        plain_klas = kLASCI(kmf, 2, (1, 1), kmesh=kmesh)
        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh,
            trans_sym=True, ref_cell=1,
        )

        self.assertIs(type(plain_klas), PBCLASCINoSymm)
        self.assertIs(type(trans_klas), PBCLASCITransSymm)
        self.assertTrue(trans_klas.trans_sym)
        self.assertEqual(trans_klas.ref_cell, 1)
        self.assertTrue(callable(trans_klas.pack_h1))
        self.assertTrue(callable(trans_klas.pack_h2))

        mo_loc = plain_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )
        energy_plain = plain_klas.kernel(np.array(mo_loc, copy=True))[1]
        energy_trans = trans_klas.kernel(np.array(mo_loc, copy=True))[1]

        self.assertAlmostEqual(energy_plain.real, energy_trans.real, places=10)
        self.assertAlmostEqual(energy_plain.imag, energy_trans.imag, places=10)
        self.assertAlmostEqual(
            plain_klas.e_tot,
            np.dot(plain_klas.weights, plain_klas.e_states), places=12,
        )
        self.assertAlmostEqual(
            trans_klas.e_tot,
            np.dot(trans_klas.weights, trans_klas.e_states), places=12,
        )
        dm1s = plain_klas.make_rdm1s_sub(
            mo_coeff=mo_loc, include_core=True,
        )
        expected_shape = (2, len(kmf.kpts), cell.nao_nr(), cell.nao_nr())
        self.assertEqual(dm1s.shape, expected_shape)

    def test_productstate_pack_and_unpack_ci(self):

        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh,
            trans_sym=True, ref_cell=1,)
        
        mo_loc = trans_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )
        trans_klas.kernel(mo_loc)

        fcisolvers = [box.fcisolvers[0] for box in trans_klas.fciboxes]
        input_phases = np.exp(1j * np.array([0.3, -0.2]))
        solver = PBCTransSymmImpureProductStateFCISolver(
            fcisolvers,
            lweights=[[1.0], [1.0]],
            ref_cell=1,
            phase_per_frag=input_phases,
        )
        ci = [np.asarray(ci_frag) 
              for ci_frag in trans_klas.ci]

        ci_ref = solver._pack_ci(ci)
        ci_fragments = solver._unpack_cif(ci_ref)

        self.assertEqual(ci_ref.shape, (1, 2, 2))
        self.assertAlmostEqual(np.linalg.norm(ci_ref), 1.0, places=10)
        self.assertLess(np.max(np.abs(ci_ref - ci[1])), 1e-12)
        self.assertIsNot(ci_ref, ci[1])
        self.assertAlmostEqual(solver.phase_per_frag[1].real, 1.0)
        self.assertAlmostEqual(solver.phase_per_frag[1].imag, 0.0)

        for ifrag, ci_unpacked in enumerate(ci_fragments):
            self.assertLess(
                np.max(np.abs(
                    ci_unpacked
                    - solver.phase_per_frag[ifrag] * ci_ref
                )), 1e-12,
            )
            self.assertFalse(np.shares_memory(ci_unpacked, ci_ref))
            
        self.assertIsNone(solver._pack_ci(None))
        self.assertEqual(solver._unpack_cif(None), [None, None])
        with self.assertRaisesRegex(ValueError, "contains 2 roots"):
            solver._unpack_cif(np.concatenate([ci_ref, ci_ref], axis=0))

    def test_productstate_ref_init_guess(self):
        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh,
            trans_sym=True, ref_cell=1,
        )
        mo_loc = trans_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )
        h1e = trans_klas.h1e_for_cas(
            mo_coeff=mo_loc, ncas=trans_klas.ncas,
            ncore=trans_klas.ncore,
        )[0]
        h2e = trans_klas.get_h2cas(mo_loc)

        fcisolvers = [box.fcisolvers[0] for box in trans_klas.fciboxes]
        solver = PBCTransSymmImpureProductStateFCISolver(
            fcisolvers,
            lweights=[[1.0], [1.0]],
            ref_cell=1,
            phase_per_frag=np.exp(1j * np.array([0.3, -0.2])),
        )
        ci_ref = solver._get_ref_init_guess(
            None, trans_klas.ncas_sub, trans_klas.nelecas_sub, h1e, h2e,
        )

        self.assertEqual(ci_ref.shape, (1, 2, 2))
        self.assertAlmostEqual(np.linalg.norm(ci_ref), 1.0, places=10)

        ci_supplied = np.exp(0.3j) * ci_ref
        ci_preserved = solver._get_ref_init_guess(
            ci_supplied, trans_klas.ncas_sub, trans_klas.nelecas_sub,
            h1e, h2e,
        )
        self.assertLess(np.max(np.abs(ci_preserved - ci_supplied)), 1e-12)
        self.assertIsNot(ci_preserved, ci_supplied)

        ci = solver._unpack_cif(ci_supplied)
        ci_preserved = solver.get_init_guess(
            ci,
            trans_klas.ncas_sub, trans_klas.nelecas_sub, h1e, h2e,
        )
        for ifrag, ci_frag in enumerate(ci_preserved):
            self.assertLess(
                np.max(np.abs(
                    ci_frag - solver.phase_per_frag[ifrag] * ci_supplied
                )), 1e-12,
            )

    def test_productstate_project_ref_hfrag(self):
        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh,
            trans_sym=True, ref_cell=1,
        )
        mo_loc = trans_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )
        h1e = trans_klas.h1e_for_cas(
            mo_coeff=mo_loc, ncas=trans_klas.ncas,
            ncore=trans_klas.ncore,
        )[0]
        h2e = trans_klas.get_h2cas(mo_loc)

        fcisolvers = [box.fcisolvers[0] for box in trans_klas.fciboxes]
        solver = PBCTransSymmImpureProductStateFCISolver(
            fcisolvers,
            lweights=[[1.0], [1.0]],
            ref_cell=1,
            phase_per_frag=np.exp(1j * np.array([0.3, -0.2])),
        )
        ci_ref = solver._get_ref_init_guess(
            None, trans_klas.ncas_sub, trans_klas.nelecas_sub, h1e, h2e,
        )
        ci = solver._unpack_cif(ci_ref)
        plain_solver = PBCProductStateFCISolver(fcisolvers)

        h1eff, h0eff, _ = plain_solver.project_hfrag(
            h1e, h2e, ci, trans_klas.ncas_sub, trans_klas.nelecas_sub,
        )
        h1eff_ref, h0eff_ref = solver._project_ref_hfrag(
            h1e, h2e, ci,
            trans_klas.ncas_sub, trans_klas.nelecas_sub,
        )

        self.assertEqual(h1eff_ref.shape, h1eff[1].shape)
        self.assertLess(np.max(np.abs(h1eff_ref - h1eff[1])), 1e-12)
        self.assertAlmostEqual(h0eff_ref.real, h0eff[1].real, places=10)
        self.assertAlmostEqual(h0eff_ref.imag, h0eff[1].imag, places=10)

        h1eff_out, h0eff_out, ci_out = solver.project_hfrag(
            h1e, h2e, ci, trans_klas.ncas_sub, trans_klas.nelecas_sub,
        )
        for h1eff_frag, h0eff_frag, ci_out_frag, ci_frag in zip(
                h1eff_out, h0eff_out, ci_out, ci):
            self.assertLess(np.max(np.abs(h1eff_frag - h1eff_ref)), 1e-12)
            self.assertAlmostEqual(h0eff_frag, h0eff_ref, places=10)
            self.assertLess(np.max(np.abs(ci_out_frag - ci_frag)), 1e-12)

    def test_productstate_get_ref_grad(self):
        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh,
            trans_sym=True, ref_cell=1,
        )
        mo_loc = trans_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )
        h1e = trans_klas.h1e_for_cas(
            mo_coeff=mo_loc, ncas=trans_klas.ncas,
            ncore=trans_klas.ncore,
        )[0]
        h2e = trans_klas.get_h2cas(mo_loc)

        fcisolvers = [box.fcisolvers[0] for box in trans_klas.fciboxes]
        solver = PBCTransSymmImpureProductStateFCISolver(
            fcisolvers,
            lweights=[[1.0], [1.0]],
            ref_cell=1,
        )
        ci_ref = solver._get_ref_init_guess(
            None, trans_klas.ncas_sub, trans_klas.nelecas_sub, h1e, h2e,
        )
        for ifrag, fcisolver in enumerate(solver.fcisolvers):
            fcisolver.norb = trans_klas.ncas_sub[ifrag]
            fcisolver.nelec = solver._get_nelec(
                fcisolver, trans_klas.nelecas_sub[ifrag],
            )
            fcisolver.check_transformer_cache()
        ci = solver._unpack_cif(ci_ref[0])
        plain_solver = PBCProductStateFCISolver(fcisolvers)
        h1eff, _, _ = plain_solver.project_hfrag(
            h1e, h2e, ci, trans_klas.ncas_sub, trans_klas.nelecas_sub,
        )

        grad = plain_solver._get_grad(
            h1eff, h2e, ci,
            trans_klas.ncas_sub, trans_klas.nelecas_sub,
        )
        i = sum(trans_klas.ncas_sub[:solver.ref_cell])
        j = i + trans_klas.ncas_sub[solver.ref_cell]
        h2e_ref = h2e[i:j, i:j, i:j, i:j]
        grad_ref = solver._get_ref_grad(
            h1eff[1], h2e_ref, ci_ref,
            trans_klas.ncas_sub, trans_klas.nelecas_sub,
        )
        ref_offset = (
            solver.fcisolvers[0].nroots
            * solver.fcisolvers[0].transformer.ncsf
        )

        self.assertEqual(grad_ref.shape, (grad_ref.size,))
        self.assertLess(
            np.max(np.abs(
                grad_ref - grad[ref_offset:ref_offset + grad_ref.size]
            )), 1e-12,
        )
        grad_out = solver._get_grad(
            h1eff, h2e, ci,
            trans_klas.ncas_sub, trans_klas.nelecas_sub,
        )
        self.assertLess(
            np.max(np.abs(grad_out - solver._unpack_grad(grad_ref))),
            1e-12,
        )

    def test_productstate_reference_optimized_kernel(self):
        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh,
            trans_sym=True, ref_cell=1,
        )
        mo_loc = trans_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )
        h1e = trans_klas.h1e_for_cas(
            mo_coeff=mo_loc, ncas=trans_klas.ncas,
            ncore=trans_klas.ncore,
        )[0]
        h2e = trans_klas.get_h2cas(mo_loc)

        fcisolvers = [box.fcisolvers[0] for box in trans_klas.fciboxes]
        phases = np.exp(1j * np.array([0.3, -0.2]))
        solver = PBCTransSymmImpureProductStateFCISolver(
            fcisolvers,
            lweights=[[1.0], [1.0]],
            ref_cell=1,
            phase_per_frag=phases,
        )
        for ifrag, fcisolver in enumerate(solver.fcisolvers):
            fcisolver.norb = trans_klas.ncas_sub[ifrag]
            fcisolver.nelec = solver._get_nelec(
                fcisolver, trans_klas.nelecas_sub[ifrag],
            )
            fcisolver.check_transformer_cache()

        with mock.patch.object(
                solver.fcisolvers[0], 'kernel',
                wraps=solver.fcisolvers[0].kernel) as other_kernel, \
             mock.patch.object(
                solver.fcisolvers[1], 'kernel',
                wraps=solver.fcisolvers[1].kernel) as ref_kernel:
            _, _, ci = solver.kernel(
                h1e, h2e, trans_klas.ncas_sub,
                trans_klas.nelecas_sub,
            )

        self.assertEqual(other_kernel.call_count, 0)
        self.assertGreater(ref_kernel.call_count, 0)
        self.assertEqual(len(ci), len(fcisolvers))
        ci_ref = solver._pack_ci(ci)
        self.assertAlmostEqual(np.linalg.norm(ci_ref), 1.0, places=10)
        for ifrag, ci_frag in enumerate(ci):
            self.assertLess(
                np.max(np.abs(
                    ci_frag - solver.phase_per_frag[ifrag] * ci_ref
                )), 1e-12,
            )

    def test_productstate_reference_rdms(self):
        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh,
            trans_sym=True, ref_cell=1,
        )
        mo_loc = trans_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )
        trans_klas.kernel(mo_loc)

        fcisolvers = [box.fcisolvers[0] for box in trans_klas.fciboxes]
        solver = PBCTransSymmImpureProductStateFCISolver(
            fcisolvers,
            lweights=[[1.0], [1.0]],
            ref_cell=1,
            phase_per_frag=np.exp(1j * np.array([0.3, -0.2])),
        )
        ci_ref = np.asarray(trans_klas.ci[1])[0]
        ci = solver._unpack_cif(ci_ref)
        plain_solver = PBCProductStateFCISolver(fcisolvers)

        dm1s_ref = plain_solver.make_rdm1s(
            ci, trans_klas.ncas_sub, trans_klas.nelecas_sub,
        )
        dm1_ref = plain_solver.make_rdm1(
            ci, trans_klas.ncas_sub, trans_klas.nelecas_sub,
        )
        dm2_ref = plain_solver.make_rdm2(
            ci, trans_klas.ncas_sub, trans_klas.nelecas_sub,
        )

        dm1s = solver.make_rdm1s(
            ci, trans_klas.ncas_sub, trans_klas.nelecas_sub,
        )
        dm1 = solver.make_rdm1(
            ci, trans_klas.ncas_sub, trans_klas.nelecas_sub,
        )
        dm2 = solver.make_rdm2(
            ci, trans_klas.ncas_sub, trans_klas.nelecas_sub,
        )

        self.assertEqual(dm1.shape, dm1_ref.shape)
        self.assertEqual(dm2.shape, dm2_ref.shape)
        self.assertLess(np.max(np.abs(dm1s[0] - dm1s_ref[0])), 1e-12)
        self.assertLess(np.max(np.abs(dm1s[1] - dm1s_ref[1])), 1e-12)
        self.assertLess(np.max(np.abs(dm1 - dm1_ref)), 1e-12)
        self.assertLess(np.max(np.abs(dm2 - dm2_ref)), 1e-12)

    def test_pack_translation_symmetric_hamiltonians(self):
        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh,
            trans_sym=True, ref_cell=1,
        )
        mo_loc = trans_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )
        h1e = trans_klas.h1e_for_cas(
            mo_coeff=mo_loc, ncas=trans_klas.ncas,
            ncore=trans_klas.ncore,
        )[0]
        h2e = trans_klas.get_h2cas(mo_loc)
        h1_packed = trans_klas.pack_h1(h1e)
        h2_packed = trans_klas.pack_h2(h2e)

        ncell = np.prod(kmesh)
        ncas = trans_klas.ncas
        self.assertEqual(h1_packed.shape, (ncell, ncas, ncas))
        self.assertEqual(
            h2_packed.shape,
            (ncell, ncell, ncell, ncas, ncas, ncas, ncas),
        )

        ts = TranslationSymm(cell, kmesh, kpts=kmf.kpts)
        h1_blocks = h1e.reshape(ncell, ncas, ncell, ncas)
        h2_blocks = h2e.reshape((ncell, ncas) * 4)
        for iR, R in enumerate(ts.R_indices):
            for idS, dS in enumerate(ts.R_indices):
                iS = ts.R_to_i[ts.mod_index(R + dS)]
                self.assertLess(
                    np.max(np.abs(
                        h1_blocks[iR, :, iS, :] - h1_packed[idS]
                    )), 1e-10,
                )
                for idU, dU in enumerate(ts.R_indices):
                    iU = ts.R_to_i[ts.mod_index(R + dU)]
                    for idV, dV in enumerate(ts.R_indices):
                        iV = ts.R_to_i[ts.mod_index(R + dV)]
                        self.assertLess(
                            np.max(np.abs(
                                h2_blocks[
                                    iR, :, iS, :, iU, :, iV, :
                                ] - h2_packed[idS, idU, idV]
                            )), 1e-10,
                        )

    def test_packed_reference_energy(self):
        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh,
            trans_sym=True, ref_cell=1,
        )
        mo_loc = trans_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )
        trans_klas.kernel(mo_loc)
        h1e = trans_klas.h1e_for_cas(
            mo_coeff=mo_loc, ncas=trans_klas.ncas,
            ncore=trans_klas.ncore,
        )[0]
        h2e = trans_klas.get_h2cas(mo_loc)

        fcisolvers = [box.fcisolvers[0] for box in trans_klas.fciboxes]
        solver = PBCTransSymmImpureProductStateFCISolver(
            fcisolvers,
            lweights=[[1.0], [1.0]],
            ref_cell=1,
            phase_per_frag=trans_klas.get_phase_per_frag(mo_loc),
            pack_h1=trans_klas.pack_h1,
            pack_h2=trans_klas.pack_h2,
        )
        ci_ref = np.asarray(trans_klas.ci[1])[0]
        ci = solver._unpack_cif(ci_ref)
        h1_packed = trans_klas.pack_h1(h1e)
        h2_packed = trans_klas.pack_h2(h2e)
        ecore = 0.37

        energy_ref = solver.energy_ref(
            h1_packed, h2_packed, ci_ref,
            trans_klas.ncas_sub, trans_klas.nelecas_sub,
            ecore=ecore,
        )
        energy = solver.energy_elec(
            h1e, h2e, ci,
            trans_klas.ncas_sub, trans_klas.nelecas_sub,
            ecore=ecore,
        )
        plain_solver = PBCProductStateFCISolver(fcisolvers)
        energy_dense = plain_solver.energy_elec(
            h1e, h2e, ci,
            trans_klas.ncas_sub, trans_klas.nelecas_sub,
            ecore=ecore,
        )

        self.assertAlmostEqual(
            energy, len(fcisolvers) * energy_ref, places=10,
        )
        self.assertAlmostEqual(energy.real, energy_dense.real, places=10)
        self.assertAlmostEqual(energy.imag, energy_dense.imag, places=10)


if __name__ == "__main__":
    unittest.main()
