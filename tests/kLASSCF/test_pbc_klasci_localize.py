#!/usr/bin/env python

"""Check active localization, phase alignment and virtual gauge stabilization.

Virtual stabilization must preserve the fixed-CI LAS energy and produce the
same orbitals after arbitrary input virtual rotations. Active phase alignment
must reproduce the same initialization after arbitrary input band phases.
At Gamma, alignment preserves energy when CI is transformed with the orbitals.
A k-dependent active phase can change the periodic LAS wavefunction and energy,
even when the complete active-band projector is unchanged. The absolute energy
tolerance is 1e-8 Ha per cell.
"""

import unittest
import numpy as np

from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.klasci import kLASCI
from mrh.my_pyscf.pbc.mcscf.productstate import ImpureProductStateFCISolver
from mrh.my_pyscf.pbc.util.orth import meta_lowdin_orbitals

# Author: Bhavnesh Jangid

# Test-0: The molecular-style periodic localizer should preserve the complete
#         active-band space, MO orthonormality, and the core/external orbitals.
# Test-1: The localized active bands should not depend on an arbitrary unitary
#         gauge applied to the input active orbitals at each k-point.
# Test-2: The localizer should reject an orthonormal local-orbital space smaller
#         than the active-band space with a clear error.
# Test-3: For a simple periodic Be atom, the localized active orbital should
#         align with the orthonormal Be 2s orbital at every k-point.

cell = kmf = mo_coeff = klas = None
kmesh = [2, 1, 1]

def setUpModule():
    global cell, kmf, mo_coeff, klas
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
    klas = kLASCI(kmf, 2, (1, 1), kmesh=kmesh)

def _fixed_ci_energy(las, orbitals, ci):
    """Evaluate the same product-state CI without optimizing orbitals or CI."""
    h1, ecore = las.h1e_for_cas(mo_coeff=orbitals)
    solver = ImpureProductStateFCISolver(
        [box.fcisolvers[0] for box in las.fciboxes],
        lweights=[[1.] for _ in las.fciboxes], stdout=las.stdout, verbose=0,
    )
    return solver.energy_elec(
        h1, las.get_h2cas(orbitals), ci, las.ncas_sub, las.nelecas_sub,
        ecore=ecore,
    ) / las.nkpts


def _correlated_ci(nfrags):
    ci = np.array([[.8, .2+.1j], [-.1+.15j, .5]], dtype=complex)
    ci /= np.linalg.norm(ci)
    return [ci.copy() for _ in range(nfrags)]


class KnownValues(unittest.TestCase):

    def test_active_space_conserved(self):
        mo_loc, umat, svals = klas.localize_init_guess(["H 1s"], mo_coeff=mo_coeff,
                                                       return_umat=True, return_svals=True,)
        ovlp = kmf.get_ovlp()
        ncore = klas.ncore
        nocc = ncore + klas.ncas

        self.assertEqual(mo_loc.shape, mo_coeff.shape)
        self.assertEqual(svals.shape, (len(kmf.kpts), klas.ncas))
        for k in range(len(kmf.kpts)):
            c0 = mo_coeff[k, :, ncore:nocc]
            c1 = mo_loc[k, :, ncore:nocc]
            p0 = c0 @ c0.conj().T @ ovlp[k]
            p1 = c1 @ c1.conj().T @ ovlp[k]

            # assertLess is used instead of assertAllclose to avoid 
            # false positives due to numerical noise. Basically assertLess checks 
            # if the maximum absolute difference is below a threshold.
            self.assertLess(np.max(np.abs(p0 - p1)), 1e-10)
            # Check that the transformation matrix is unitary.
            self.assertLess(np.max(
                np.abs(umat[k].conj().T @ umat[k] - np.eye(umat.shape[-1]))),
                1e-10)
            self.assertLess(
                np.max(np.abs(mo_loc[k].conj().T @ ovlp[k] @ mo_loc[k]
                              - np.eye(mo_loc.shape[-1]))),
                1e-10,
            )

            # Core orbitals: edge case: in this case its zero.
            np.testing.assert_allclose(
                mo_loc[k, :, :ncore], mo_coeff[k, :, :ncore], atol=1e-12,)
            
            # External orbitals
            np.testing.assert_allclose(
                mo_loc[k, :, nocc:], mo_coeff[k, :, nocc:], atol=1e-12,)

    def test_active_gauge_invariance(self):
        mo_ref = klas.localize_init_guess(["H 1s"], mo_coeff=mo_coeff)
        mo_rot = np.array(mo_coeff, copy=True)
        rng = np.random.default_rng(19)
        ncore = klas.ncore
        nocc = ncore + klas.ncas

        # Randomly rotate the active orbitals at each k-point and check 
        # that the localized orbitals are invariant under this transformation.
        for k in range(len(kmf.kpts)):
            x = (
                rng.standard_normal((klas.ncas, klas.ncas))
                + 1j * rng.standard_normal((klas.ncas, klas.ncas))
            )

            u = np.linalg.qr(x)[0]
            mo_rot[k, :, ncore:nocc] = (
                mo_rot[k, :, ncore:nocc] @ u
            )

        mo_test = klas.localize_init_guess(["H 1s"], mo_coeff=mo_rot)
        ovlp = kmf.get_ovlp()
        for k in range(len(kmf.kpts)):
            cref = mo_ref[k, :, ncore:nocc]
            ctest = mo_test[k, :, ncore:nocc]
            overlap = cref.conj().T @ ovlp[k] @ ctest
            self.assertLess(
                np.max(np.abs(overlap - np.diag(np.diag(overlap)))),
                1e-9,
            )
            self.assertLess(
                np.max(np.abs(np.abs(np.diag(overlap)) - 1)),
                1e-9,
            )

    def test_fock_canonicalization_order(self):
        ncore = klas.ncore
        nocc = ncore + klas.ncas
        active = slice(ncore, nocc)
        ovlp = kmf.get_ovlp()

        # Reverse the local active-orbital basis so that localization alone
        # puts the higher-energy active orbital first.  With mo_occ omitted,
        # both active orbitals belong to the same occupation sector and the
        # Fock canonicalization must restore increasing-energy order.
        lo_coeff = np.array(mo_coeff[:, :, active][:, :, ::-1], copy=True)
        orbital_energy = np.arange(mo_coeff.shape[-1], dtype=float)
        fock = np.empty(
            (len(kmf.kpts), mo_coeff.shape[1], mo_coeff.shape[1]),
            dtype=mo_coeff.dtype,
        )
        for k in range(len(kmf.kpts)):
            fock[k] = (
                ovlp[k]
                @ mo_coeff[k]
                @ np.diag(orbital_energy)
                @ mo_coeff[k].conj().T
                @ ovlp[k]
            )

        mo_loc = klas.localize_init_guess(
            list(range(klas.ncas)),
            mo_coeff=mo_coeff,
            lo_coeff=lo_coeff,
            fock=fock,
            frags_by_AOs=True,
        )

        for k in range(len(kmf.kpts)):
            overlap = (
                mo_coeff[k, :, active].conj().T
                @ ovlp[k]
                @ mo_loc[k, :, active]
            )
            self.assertLess(
                np.max(np.abs(np.abs(overlap) - np.eye(klas.ncas))),
                1e-10,
            )
            localized_fock = (
                mo_loc[k, :, active].conj().T
                @ fock[k]
                @ mo_loc[k, :, active]
            )
            np.testing.assert_allclose(
                localized_fock,
                np.diag(orbital_energy[active]),
                atol=1e-10,
            )

    def test_be_2s_orbital_alignment(self):
        be_cell = gto.Cell()
        be_cell.a = np.diag([4.0, 8.0, 8.0])
        be_cell.atom = "Be 0 0 0"
        be_cell.basis = "sto-3g"
        be_cell.unit = "Angstrom"
        be_cell.precision = 1e-8
        be_cell.ke_cutoff = 20
        be_cell.verbose = 0
        be_cell.build()

        be_kmesh = [2, 1, 1]
        be_kpts = be_cell.make_kpts(be_kmesh, wrap_around=True)
        be_kmf = scf.KRHF(be_cell, kpts=be_kpts).density_fit()
        be_kmf.exxdiv = None
        be_kmf.max_cycle = 0
        be_kmf.kernel()

        be_mo = avas.kernel(be_kmf, ["Be 2s"], minao=be_cell.basis)[2]
        be_klas = kLASCI(be_kmf, 1, (1, 1), kmesh=be_kmesh)
        mo_loc, svals = be_klas.localize_init_guess(
            ["Be 2s"], mo_coeff=be_mo, return_svals=True,
        )

        ovlp = be_kmf.get_ovlp()
        ortho_lo = meta_lowdin_orbitals(be_cell, ovlp)
        be_2s = be_cell.search_ao_label("Be 2s")
        ncore = be_klas.ncore
        nocc = ncore + be_klas.ncas
        active = slice(ncore, nocc)

        for k in range(len(be_kpts)):
            c_2s = ortho_lo[k][:, be_2s]
            c_act = mo_loc[k, :, active]
            overlap_2s = (c_2s.conj().T @ ovlp[k] @ c_act)[0, 0]

            self.assertGreater(abs(overlap_2s), 0.95)
            np.testing.assert_allclose(
                abs(overlap_2s), svals[k, 0], atol=1e-10,
            )
            np.testing.assert_allclose(
                mo_loc[k].conj().T @ ovlp[k] @ mo_loc[k],
                np.eye(mo_loc.shape[-1]),
                atol=1e-10,
            )

            # The localizer must leave the Be 1s core orbital unchanged.
            np.testing.assert_allclose(
                mo_loc[k, :, :ncore], be_mo[k, :, :ncore], atol=1e-12,
            )

            # The unoccupied Be 2p orbitals must also remain unchanged.
            np.testing.assert_allclose(
                mo_loc[k, :, nocc:], be_mo[k, :, nocc:], atol=1e-12,
            )

    def test_virtual_stabilization_preserves_fixed_ci_energy(self):
        original = klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff, stabilize_virtuals=False)
        stable = klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff, stabilize_virtuals=True)
        nocc = klas.ncore + klas.ncas
        np.testing.assert_allclose(stable[:, :, :nocc], original[:, :, :nocc],
                                   atol=1e-10, rtol=0)
        overlap = kmf.get_ovlp()
        for k in range(len(kmf.kpts)):
            before = original[k, :, nocc:]
            after = stable[k, :, nocc:]
            np.testing.assert_allclose(before @ before.conj().T @ overlap[k],
                                       after @ after.conj().T @ overlap[k],
                                       atol=1e-10, rtol=0)
        ci = _correlated_ci(klas.nfrags)
        self.assertAlmostEqual(_fixed_ci_energy(klas, original, ci).real,
                               _fixed_ci_energy(klas, stable, ci).real, delta=1e-8)

    def test_virtual_stabilization_is_reproducible_under_virtual_rotations(self):
        reference = klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff, stabilize_virtuals=True)
        rotated = np.array(mo_coeff, dtype=complex, copy=True)
        nocc = klas.ncore + klas.ncas
        nvir = rotated.shape[-1] - nocc
        rng = np.random.default_rng(73)
        for k in range(len(kmf.kpts)):
            unitary = np.linalg.qr(rng.normal(size=(nvir, nvir))
                                  + 1j*rng.normal(size=(nvir, nvir)))[0]
            rotated[k, :, nocc:] = rotated[k, :, nocc:] @ unitary
        stable = klas.localize_init_guess(
            ["H 1s"], mo_coeff=rotated, stabilize_virtuals=True)
        np.testing.assert_allclose(stable, reference, atol=1e-9, rtol=0)

    def test_phase_alignment_is_reproducible_under_input_band_phases(self):
        reference = klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff, align_phases=True)
        phased = np.array(mo_coeff, dtype=complex, copy=True)
        active = slice(klas.ncore, klas.ncore + klas.ncas)
        phases = np.exp(1j*np.array([[.7, -.2], [-.4, 1.1]]))
        phased[:, :, active] *= phases[:, None, :]
        aligned = klas.localize_init_guess(
            ["H 1s"], mo_coeff=phased, align_phases=True)
        np.testing.assert_allclose(aligned, reference, atol=1e-9, rtol=0)
        ci = _correlated_ci(klas.nfrags)
        self.assertAlmostEqual(_fixed_ci_energy(klas, aligned, ci).real,
                               _fixed_ci_energy(klas, reference, ci).real, delta=1e-8)

    def test_gamma_phase_alignment_preserves_energy_with_transformed_ci(self):
        gamma_mf = scf.KRHF(cell, kpts=cell.make_kpts((1, 1, 1))).density_fit()
        gamma_mf.exxdiv = None
        gamma_mf.max_cycle = 0
        gamma_mf.kernel()
        gamma_mo = np.asarray(avas.kernel(gamma_mf, ["H 1s"], minao=cell.basis)[2],
                              dtype=complex).reshape(1, cell.nao_nr(), -1)
        gamma_las = kLASCI(gamma_mf, 2, (1, 1), kmesh=(1, 1, 1))
        overlap = gamma_mf.get_ovlp()
        local = np.asarray(meta_lowdin_orbitals(cell, overlap), dtype=complex)
        local *= np.exp(.7j)
        original = gamma_las.localize_init_guess(
            ["H 1s"], mo_coeff=gamma_mo, lo_coeff=local, align_phases=False)
        active = slice(gamma_las.ncore, gamma_las.ncore + gamma_las.ncas)
        # Distinct band phases make this a nontrivial CI transformation,
        # rather than an overall phase of the many-electron wavefunction.
        original[:, :, active] *= np.exp(1j*np.array([.4, -.9]))[None, None, :]
        aligned = gamma_las.localize_init_guess(
            ["H 1s"], mo_coeff=original, lo_coeff=local, align_phases=True)
        rotation = original[0, :, active].conj().T @ overlap[0] @ aligned[0, :, active]
        phases = np.diag(rotation)
        np.testing.assert_allclose(rotation, np.diag(phases), atol=1e-10, rtol=0)
        np.testing.assert_allclose(np.abs(phases), 1., atol=1e-10, rtol=0)
        self.assertGreater(np.linalg.norm(phases-1), .1)
        ci = _correlated_ci(1)
        transformed = [ci[0] * phases.conj()[:, None] * phases.conj()[None, :]]
        original_energy = _fixed_ci_energy(gamma_las, original, ci).real
        unchanged_ci_energy = _fixed_ci_energy(gamma_las, aligned, ci).real
        self.assertGreater(abs(unchanged_ci_energy-original_energy), 1e-6)
        self.assertAlmostEqual(original_energy,
                               _fixed_ci_energy(gamma_las, aligned, transformed).real,
                               delta=1e-8)

    def test_k_dependent_band_phase_can_change_fixed_ci_las_energy(self):
        original = klas.localize_init_guess(["H 1s"], mo_coeff=mo_coeff)
        phased = np.array(original, dtype=complex, copy=True)
        phased[1, :, klas.ncore] *= np.exp(.4j)
        ci = _correlated_ci(klas.nfrags)
        difference = (_fixed_ci_energy(klas, phased, ci)
                      - _fixed_ci_energy(klas, original, ci)).real
        self.assertGreater(abs(difference), 1e-6)

    def test_api_for_localization_init_guess(self):
        with self.assertRaisesRegex(
                ValueError, "Cannot localize 2 active bands using only 1"):
            klas.localize_init_guess(
                [0], mo_coeff=mo_coeff, frags_by_AOs=True,
            )

if __name__ == "__main__":
    unittest.main()
